import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-4B"

LORA_PATH = (
    "/home/avinash/Projects/Custom-LN-Translator-Training/"
    "Model-Checkpoints/Arifureta/GPU/ArrangedAndCleanedDataset/checkpoint-100"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    offload_folder="auto"
)


prompt = """
Translate Japanese light novel text to natural English.

Japanese:
南雲ハジメは迷宮へ向かった。

English:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))