import os
import subprocess
from pathlib import Path

# ==================================================

# CONFIG

# ==================================================

BASE_MODEL = "Qwen/Qwen3-4B"

LORA_PATH = Path(
"Model-Checkpoints/Arifureta/GPU/ArrangedAndCleanedDataset/checkpoint-100"
)

MERGED_DIR = Path("Model-Outputs/Merged/Qwen3-4B-Arifureta")
GGUF_DIR = Path("Model-Outputs/GGUF")

LLAMA_CPP_DIR = Path("~/llama.cpp").expanduser()

GGUF_DIR.mkdir(exist_ok=True)

# ==================================================

# STEP 1: MERGE LORA

# ==================================================

print("Loading model...")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
BASE_MODEL,
torch_dtype=torch.float16,
device_map="cpu",
)

model = PeftModel.from_pretrained(
base_model,
str(LORA_PATH),
)

print("Merging LoRA...")

merged_model = model.merge_and_unload()

MERGED_DIR.mkdir(exist_ok=True)

merged_model.save_pretrained(
MERGED_DIR,
safe_serialization=True,
)

tokenizer.save_pretrained(MERGED_DIR)

print("Merged model saved.")

# ==================================================

# STEP 2: HF -> GGUF FP16

# ==================================================

fp16_gguf = GGUF_DIR / "qwen3-arifureta-f16.gguf"

convert_cmd = [
"python",
str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
str(MERGED_DIR),
"--outfile",
str(fp16_gguf),
"--outtype",
"f16",
]

subprocess.run(convert_cmd, check=True)

print("FP16 GGUF created.")

# ==================================================

# STEP 3: QUANTIZE

# ==================================================

q4_file = GGUF_DIR / "qwen3-arifureta-q4km.gguf"

quant_cmd = [
str(LLAMA_CPP_DIR / "build/bin/llama-quantize"),
str(fp16_gguf),
str(q4_file),
"Q4_K_M",
]

subprocess.run(quant_cmd, check=True)

print("Done!")
print(q4_file)
