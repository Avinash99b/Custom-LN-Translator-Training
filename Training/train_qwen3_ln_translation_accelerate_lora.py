import os
import gc
import math
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers.trainer_utils import get_last_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

BASE_MODEL    = "Qwen/Qwen3-4B"
TOKENIZED_ROOT = Path("/kaggle/working/tokenized_dataset_1")
OUTPUT_ROOT    = Path("/kaggle/working/qwen3_ln_translation")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 1536
SEED        = 42

PHASES = [
    {"name": "phase1_chunk_heavy", "full_w": 0.2, "chunk_w": 0.8, "epochs": 1},
    {"name": "phase2_balanced",    "full_w": 0.5, "chunk_w": 0.5, "epochs": 1},
    {"name": "phase3_full_heavy",  "full_w": 0.8, "chunk_w": 0.2, "epochs": 1},
]

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_split(path): return load_from_disk(str(path))

full_train  = load_split(TOKENIZED_ROOT / "full_chapter"  / "train")
full_val    = load_split(TOKENIZED_ROOT / "full_chapter"  / "validation")
chunk_train = load_split(TOKENIZED_ROOT / "chunked"       / "train")
chunk_val   = load_split(TOKENIZED_ROOT / "chunked"       / "validation")


def sample_dataset(ds, n, seed):
    if n <= 0 or not len(ds): return ds.select([])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds), size=n).tolist() if n > len(ds) else rng.permutation(len(ds))[:n].tolist()
    return ds.select(idx)

def build_mixture(full_ds, chunk_ds, full_w, chunk_w, seed):
    total = max(
        math.ceil(len(full_ds)  / max(full_w,  1e-9)) if len(full_ds)  else 0,
        math.ceil(len(chunk_ds) / max(chunk_w, 1e-9)) if len(chunk_ds) else 0,
    )
    n_full  = max(1 if len(full_ds)  else 0, int(round(total * full_w)))
    n_chunk = max(1 if len(chunk_ds) else 0, int(round(total * chunk_w)))
    parts = [s for s in [sample_dataset(full_ds, n_full, seed+1), sample_dataset(chunk_ds, n_chunk, seed+2)] if len(s)]
    return (parts[0] if len(parts) == 1 else concatenate_datasets(parts)).shuffle(seed=seed)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def load_model(local_rank):
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, quantization_config=quant,
        device_map={"": local_rank}, attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # FIX: use_reentrant=False is critical — reentrant checkpointing re-runs forward
    # passes and spikes VRAM unpredictably on T4.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, bias="none", inference_mode=False,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    ))
    model.print_trainable_parameters()
    return model


class PackingCausalCollator:
    """
    Packs tokenized examples into fixed-length blocks.

    OOM fix: allocate lazily via list-append rather than pre-allocating
    `max_blocks` tensors. The old ceiling-division pre-alloc was correct in
    theory but caused large single-shot allocations that fragmented the CUDA
    allocator on T4, leaving insufficient contiguous memory for activations.
    Growing a list and stacking once at the end is slightly slower but keeps
    peak allocation proportional to actual output, not worst-case input.
    """
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        blocks_ids, blocks_mask, blocks_labels = [], [], []

        buf_ids    = torch.full((self.max_length,), self.pad_id, dtype=torch.long)
        buf_mask   = torch.zeros(self.max_length, dtype=torch.long)
        buf_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        pos = 0

        def flush():
            blocks_ids.append(buf_ids.clone())
            blocks_mask.append(buf_mask.clone())
            blocks_labels.append(buf_labels.clone())

        for feat in features:
            ids    = feat["input_ids"]
            labels = feat["labels"]
            if isinstance(ids,    torch.Tensor): ids    = ids.tolist()
            if isinstance(labels, torch.Tensor): labels = labels.tolist()

            start = 0
            while start < len(ids):
                take = min(self.max_length - pos, len(ids) - start)
                buf_ids   [pos:pos+take] = torch.tensor(ids   [start:start+take], dtype=torch.long)
                buf_mask  [pos:pos+take] = 1
                buf_labels[pos:pos+take] = torch.tensor(labels[start:start+take], dtype=torch.long)
                pos += take; start += take
                if pos == self.max_length:
                    flush()
                    # Reset buffers in-place — avoids re-alloc on every block.
                    buf_ids.fill_(self.pad_id); buf_mask.zero_(); buf_labels.fill_(-100)
                    pos = 0

        if pos > 0: flush()

        return {
            "input_ids":      torch.stack(blocks_ids),
            "attention_mask": torch.stack(blocks_mask),
            "labels":         torch.stack(blocks_labels),
        }


class TokenThroughputTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t0 = None; self._tokens = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        if (labels := inputs.get("labels")) is not None:
            # FIX: detach before .sum() so the label tensor doesn't stay pinned
            # in the computation graph between steps.
            with torch.no_grad():
                self._tokens += int((labels != -100).sum().item())
        if self._t0 is None: self._t0 = time.time()
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def log(self, logs, start_time=None):
        if self._t0:
            elapsed = max(1e-6, time.time() - self._t0)
            logs = {**logs, "tokens_seen": self._tokens, "tok_per_sec": round(self._tokens / elapsed, 1)}
        # FIX: removed gpu_stats() from log() — pynvml calls inside the training
        # loop hold device handles that interfere with the CUDA allocator on T4.
        super().log(logs, start_time=start_time)


def train_phase(phase_cfg, seed, init_adapter=None):
    set_seed(seed)
    phase_dir = OUTPUT_ROOT / phase_cfg["name"]
    phase_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    train_ds  = build_mixture(full_train, chunk_train, phase_cfg["full_w"], phase_cfg["chunk_w"], seed)
    eval_ds   = build_mixture(full_val,   chunk_val,   phase_cfg["full_w"], phase_cfg["chunk_w"], seed+1000)
    print(f"[{phase_cfg['name']}] train={len(train_ds)} eval={len(eval_ds)}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = load_model(local_rank)
    if init_adapter:
        # FIX: load adapter weights in-place rather than wrapping with a second
        # PeftModel — the double-wrap kept both the old and new adapter tensors
        # alive simultaneously, effectively doubling adapter VRAM.
        model.load_adapter(init_adapter, adapter_name="default")

    args = TrainingArguments(
        output_dir=str(phase_dir),
        num_train_epochs=phase_cfg["epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1.5e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="no",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        logging_dir=str(phase_dir / "logs"),
        seed=seed, data_seed=seed,
        remove_unused_columns=False,
        # FIX: num_workers=0 eliminates the forked worker processes that each
        # hold a pinned-memory buffer pool — on T4 these push you over the edge.
        # pin_memory=False for the same reason.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=False,
        label_names=["labels"],
        save_safetensors=True,
    )

    trainer = TokenThroughputTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=PackingCausalCollator(tokenizer, MAX_SEQ_LEN),
        tokenizer=tokenizer,
    )

    resume = get_last_checkpoint(str(phase_dir))
    trainer.train(resume_from_checkpoint=resume)

    final = phase_dir / "final_adapter"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    return str(final)


def merge_final_model():
    tokenizer  = load_tokenizer()
    base       = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    merged_dir = OUTPUT_ROOT / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    peft_model = PeftModel.from_pretrained(base, str(OUTPUT_ROOT / PHASES[-1]["name"] / "final_adapter"))
    peft_model.merge_and_unload().save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(merged_dir))
    print("Merged →", merged_dir)


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    set_seed(SEED + local_rank)
    last_adapter = None
    for idx, phase in enumerate(PHASES):
        last_adapter = train_phase(phase, seed=SEED + idx*100 + local_rank, init_adapter=last_adapter)

    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized(): dist.barrier()
    if int(os.environ.get("RANK", "0")) == 0:
        merge_final_model()