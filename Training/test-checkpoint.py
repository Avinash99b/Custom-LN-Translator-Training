import gc
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# =====================================================
# CONFIG
# =====================================================

BASE_MODEL = "Qwen/Qwen3-4B"

CHECKPOINT_ROOTS = [
    "/home/avinash/Projects/Custom-LN-Translator-Training/Model-Checkpoints/Arifureta/GPU/ArrangedAndCleanedDataset",
    "/home/avinash/Projects/Custom-LN-Translator-Training/Model-Checkpoints/Arifureta/GPU/ArrangedButNonCleanedDataset",  # change if needed
]

# =====================================================
# TEST PASSAGE
# =====================================================

jp_text = """
ハジメは深く息を吐いた。

「本当に行くのか？」

ユエは小さく微笑みながら頷いた。

「うん。だって、ハジメと一緒だから」

その言葉に、ハジメは肩をすくめる。

外では激しい雨が降っていた。
"""

prompt = f"""Translate the following Japanese light novel passage into natural English.

Japanese:
{jp_text}

English Translation:
"""

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

# =====================================================
# QUANTIZATION
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# =====================================================
# CHECKPOINT DISCOVERY
# =====================================================

def discover_checkpoints():
    checkpoints = []

    for root in CHECKPOINT_ROOTS:
        root = Path(root)

        if not root.exists():
            print(f"Missing folder: {root}")
            continue

        found = sorted(
            [
                p
                for p in root.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint-")
            ],
            key=lambda x: int(x.name.split("-")[-1]),
        )

        checkpoints.extend(found)

    return checkpoints


# =====================================================
# MODEL LOADING
# =====================================================

def load_base_model():
    print("\nTrying full GPU load...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        model.cuda()

        print("✓ Fully loaded on GPU")

        return model

    except RuntimeError as e:
        print(f"GPU load failed: {e}")
        print("Falling back to device_map='auto'")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        return model


# =====================================================
# GENERATION
# =====================================================

def translate(model, title):
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(result)
    print("=" * 120 + "\n")


# =====================================================
# BASE MODEL TEST
# =====================================================

base_model = load_base_model()

translate(
    base_model,
    "BASE MODEL",
)

# =====================================================
# CHECKPOINT TESTING
# =====================================================

checkpoints = discover_checkpoints()

print(f"\nFound {len(checkpoints)} checkpoints\n")

for ckpt in checkpoints:
    print(f"\nLoading {ckpt}")

    try:
        model = PeftModel.from_pretrained(
            base_model,
            str(ckpt),
        )

        translate(
            model,
            f"CHECKPOINT: {ckpt.name}",
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed: {ckpt}")
        print(e)