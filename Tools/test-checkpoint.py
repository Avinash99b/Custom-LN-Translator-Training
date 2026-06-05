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

LORA_PATH = (
    "Model-Checkpoints/Arifureta/GPU/"
    "ArrangedButNonCleanedDataset/checkpoint-300"
)

USE_LORA = False

# =====================================================
# LOAD TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

# =====================================================
# LOAD 4-BIT MODEL
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# =====================================================
# LOAD LORA
# =====================================================

if USE_LORA:
    print(f"Loading LoRA: {LORA_PATH}")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
    )

model.eval()

# =====================================================
# TEST STORY
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
# GENERATE
# =====================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

print("\n" + "=" * 80)
print(result)
print("=" * 80)