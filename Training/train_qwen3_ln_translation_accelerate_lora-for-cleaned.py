import os
import gc
import math
import time
import json
import glob
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers.trainer_utils import get_last_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

# Kaggle + DDP on notebook kernels can inherit a forked multiprocessing context.
# Force spawn before any CUDA-backed worker is created.

import torch.distributed as dist

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "Qwen/Qwen3-4B"
# Point this to the directory produced by the preprocessing script.
TOKENIZED_ROOT = Path("/kaggle/working/tokenized_dataset_1")  # change for your dataset mount
OUTPUT_ROOT = Path("/kaggle/working/qwen3_ln_translation")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Training lengths
MAX_SEQ_LEN = 512  # good compromise for dual T4 + long narrative context
FULL_WEIGHT = 0.2   # curriculum phase 1
CHUNK_WEIGHT = 0.8
PHASES = [
    {"name": "phase1_chunk_heavy", "full_w": 0.2, "chunk_w": 0.8, "epochs": 1},
    {"name": "phase2_balanced",    "full_w": 0.5, "chunk_w": 0.5, "epochs": 1},
    {"name": "phase3_full_heavy",  "full_w": 0.8, "chunk_w": 0.2, "epochs": 1},
]

# Optimization
# Adapter-only fine-tuning: the base model stays frozen and only LoRA/QLoRA adapters are trained.
TRAINING_MODE = "qlora_lora_only"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 1.5e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 20
SAVE_STEPS = 100
EVAL_STEPS = 100
SEED = 42

# LoRA
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# 4-bit quantization for QLoRA
# Keep this ON for Kaggle T4 efficiency; this is adapter training, not full-model fine-tuning.
USE_4BIT = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_stats() -> List[Dict]:
    stats = []
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            stats.append({
                "gpu": i,
                "utilization_percent": int(util.gpu),
                "memory_used_mb": round(mem.used / (1024**2), 1),
                "memory_total_mb": round(mem.total / (1024**2), 1),
            })
    except Exception:
        # Fallback to nvidia-smi.
        try:
            out = subprocess.check_output(
                ["bash", "-lc", "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()
            for line in out:
                if not line.strip():
                    continue
                idx, util, used, total = [x.strip() for x in line.split(",")]
                stats.append({
                    "gpu": int(idx),
                    "utilization_percent": int(util),
                    "memory_used_mb": float(used),
                    "memory_total_mb": float(total),
                })
        except Exception:
            pass
    return stats


def load_split(path: Path):
    return load_from_disk(str(path))

full_train = load_split(TOKENIZED_ROOT / "full_chapter" / "train")
full_val   = load_split(TOKENIZED_ROOT / "full_chapter" / "validation")
chunk_train = load_split(TOKENIZED_ROOT / "chunked" / "train")
chunk_val   = load_split(TOKENIZED_ROOT / "chunked" / "validation")

print("Loaded:")
print("full_train:", len(full_train), "full_val:", len(full_val))
print("chunk_train:", len(chunk_train), "chunk_val:", len(chunk_val))


def sample_dataset(ds, n: int, seed: int):
    """Sample n rows, with replacement when needed."""
    if n <= 0 or len(ds) == 0:
        return ds.select([])
    rng = np.random.default_rng(seed)
    if n <= len(ds):
        idx = rng.permutation(len(ds))[:n].tolist()
    else:
        idx = rng.integers(0, len(ds), size=n).tolist()
    return ds.select(idx)

def build_mixture(full_ds, chunk_ds, full_w: float, chunk_w: float, seed: int):
    """
    Build a weighted mixture for a curriculum phase.
    We oversample the smaller side when needed so the ratio is respected.
    """
    if len(full_ds) == 0 and len(chunk_ds) == 0:
        raise ValueError("Both datasets are empty.")

    total_target = max(
        math.ceil(len(full_ds) / max(full_w, 1e-9)) if len(full_ds) else 0,
        math.ceil(len(chunk_ds) / max(chunk_w, 1e-9)) if len(chunk_ds) else 0,
    )
    n_full = max(0, int(round(total_target * full_w)))
    n_chunk = max(0, int(round(total_target * chunk_w)))

    # Ensure we never end up with an empty curriculum when a side exists.
    if len(full_ds) and n_full == 0:
        n_full = 1
    if len(chunk_ds) and n_chunk == 0:
        n_chunk = 1

    full_sel = sample_dataset(full_ds, n_full, seed=seed + 1)
    chunk_sel = sample_dataset(chunk_ds, n_chunk, seed=seed + 2)

    parts = []
    if len(full_sel):
        parts.append(full_sel)
    if len(chunk_sel):
        parts.append(chunk_sel)

    if len(parts) == 1:
        mixed = parts[0]
    else:
        mixed = concatenate_datasets(parts)

    mixed = mixed.shuffle(seed=seed)
    return mixed


def choose_attention_impl() -> str:
    """
    FlashAttention2 is great when available, but T4 often benefits from falling back to SDPA.
    The notebook auto-selects the best supported attention implementation.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("flash_attn") is not None:
            major, minor = torch.cuda.get_device_capability()
            # Conservative gate: only enable FA2 on Ampere+.
            if major >= 8:
                return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"



def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def load_model_for_training(model_name: str, local_rank: int):
    """
    Load Qwen3-4B as the frozen base model, then attach trainable LoRA adapters.
    Each DDP worker loads onto its own GPU only.
    """
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    # Important: this function must run only inside the Accelerate worker processes.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        quantization_config=quant_config,
        device_map=device_map,
        attn_implementation=ATTN_IMPL,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Qwen-family linear projections are generally named like this.
    # These are the only trainable weights in the whole run.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        inference_mode=False,
    )
    assert TRAINING_MODE.startswith("qlora"), "This script is intended for adapter-only fine-tuning."
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


class PackingCausalCollator:
    """
    Packs already-tokenized examples into fixed-length sequences without any re-tokenization.
    Each example must already contain input_ids, attention_mask, and labels.
    """
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_id is None:
            self.pad_id = 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        blocks_input_ids = []
        blocks_attention = []
        blocks_labels = []

        cur_ids: List[int] = []
        cur_labels: List[int] = []

        def flush():
            if not cur_ids:
                return
            pad_len = self.max_length - len(cur_ids)
            blocks_input_ids.append(cur_ids + [self.pad_id] * pad_len)
            blocks_attention.append([1] * len(cur_ids) + [0] * pad_len)
            blocks_labels.append(cur_labels + [-100] * pad_len)

        for feat in features:
            ids = feat["input_ids"]
            labels = feat["labels"]

            # Ensure Python lists.
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            # Extremely long examples are split at arbitrary boundaries only as a safety fallback.
            start = 0
            while start < len(ids):
                remaining = self.max_length - len(cur_ids)
                take = min(remaining, len(ids) - start)
                cur_ids.extend(ids[start:start + take])
                cur_labels.extend(labels[start:start + take])
                start += take

                if len(cur_ids) == self.max_length:
                    flush()
                    cur_ids = []
                    cur_labels = []

        flush()

        if not blocks_input_ids:
            # Defensive fallback.
            blocks_input_ids = [[self.pad_id] * self.max_length]
            blocks_attention = [[0] * self.max_length]
            blocks_labels = [[-100] * self.max_length]

        return {
            "input_ids": torch.tensor(blocks_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(blocks_attention, dtype=torch.long),
            "labels": torch.tensor(blocks_labels, dtype=torch.long),
        }


class TokenThroughputTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_start_time = None
        self._tokens_seen = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is not None:
            with torch.no_grad():
                self._tokens_seen += int((labels != -100).sum().item())
        if self._train_start_time is None:
            self._train_start_time = time.time()
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        logs = dict(logs)
        if self._train_start_time is not None:
            elapsed = max(1e-6, time.time() - self._train_start_time)
            logs["train_tokens_seen"] = self._tokens_seen
            logs["tokens_per_sec"] = self._tokens_seen / elapsed
        logs["gpu_stats"] = json.dumps(gpu_stats(), ensure_ascii=False)
        return super().log(logs, start_time=start_time)


def train_phase(
    phase_cfg: Dict,
    seed: int,
    init_adapter: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    set_seed(seed)

    phase_name = phase_cfg["name"]
    phase_dir = OUTPUT_ROOT / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(BASE_MODEL)

    train_ds = build_mixture(full_train, chunk_train, phase_cfg["full_w"], phase_cfg["chunk_w"], seed=seed)
    eval_ds  = build_mixture(full_val, chunk_val, phase_cfg["full_w"], phase_cfg["chunk_w"], seed=seed + 1000)

    print(f"[{phase_name}] train samples={len(train_ds)} eval samples={len(eval_ds)}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    base_model = load_model_for_training(BASE_MODEL, local_rank=local_rank)

    # Continue training from the previous phase's adapter when available.
    if init_adapter is not None:
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        model = base_model

    collator = PackingCausalCollator(tokenizer, max_length=MAX_SEQ_LEN)

    args = TrainingArguments(
        output_dir=str(phase_dir),
        num_train_epochs=phase_cfg["epochs"],
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=MAX_GRAD_NORM,
        report_to=["tensorboard"],
        logging_dir=str(phase_dir / "logs"),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
    )

    trainer = TokenThroughputTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Resume from the last checkpoint if provided or if one already exists.
    if resume_from_checkpoint is None:
        last = get_last_checkpoint(str(phase_dir))
        resume_from_checkpoint = last

    last = get_last_checkpoint(str(phase_dir))
    print("DEBUG PHASE DIR:", phase_dir)
    print("DEBUG LAST CHECKPOINT:", last)
    resume_from_checkpoint = last
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    metrics["eval_samples"] = len(eval_ds)

    final_adapter_dir = phase_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    with open(phase_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    eval_metrics = trainer.evaluate()
    with open(phase_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    del trainer, model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return str(final_adapter_dir)

def train_worker(process_index: int = 0):
    global ATTN_IMPL

    rank = int(os.environ.get("RANK", process_index))
    local_rank = int(os.environ.get("LOCAL_RANK", process_index))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ATTN_IMPL = choose_attention_impl()

    print(
        f"[rank {rank} / local_rank {local_rank}] "
        f"attention_impl={ATTN_IMPL}"
    )
    set_seed(SEED + rank)
    last_adapter = None
    for idx, phase in enumerate(PHASES):
        phase_seed = SEED + idx * 100 + rank
        last_adapter = train_phase(
            phase,
            seed=phase_seed,
            init_adapter=last_adapter,
            resume_from_checkpoint=None,
        )
    return last_adapter



def merge_final_model():
    FINAL_PHASE_DIR = OUTPUT_ROOT / PHASES[-1]["name"] / "final_adapter"
    MERGED_DIR = OUTPUT_ROOT / "merged_model"
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPL,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(FINAL_PHASE_DIR))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(MERGED_DIR))

    print(f"Merged model saved to: {MERGED_DIR}")



if __name__ == "__main__":
    train_worker()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if int(os.environ.get("RANK", "0")) == 0:
        merge_final_model()
