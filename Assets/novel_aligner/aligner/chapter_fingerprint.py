import re
import json
import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ChapterFingerprint:
    def __init__(self, embedding_engine, max_chars: int = 2500, mode: str = "summary"):
        self.engine = embedding_engine
        self.max_chars = max_chars
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Only initialize the LLM if we are in summary mode and have a GPU available
        if self.mode == "summary" and self.device == "cuda":
            print("Loading local LLM into VRAM (4-bit)...")
            self._init_llm()
        else:
            self.llm_model = None
            self.tokenizer = None

    def _init_llm(self):
        # Qwen is highly optimized for bilingual JP/EN tasks
        model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
        
        # 4-bit quantization ensures it fits on laptop GPUs (uses ~1.2GB VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def extract(self, file_path: str, absolute_path: str) -> Dict[str, Any]:
        with open(absolute_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()

        lines = content.split('\n')
        title = lines[0] if lines else ""
        text_body = content[:self.max_chars]

        characters, events = [], []
        summary = ""

        if self.mode == "summary" and self.llm_model:
            extracted_data = self._extract_with_llm(text_body)
            summary = extracted_data.get("summary", "")
            characters = extracted_data.get("characters", [])
            events = extracted_data.get("events", [])
            
            # Fallback if LLM returns empty summary
            if not summary:
                summary = text_body[:500]
                
            target_text = f"{title}\n{summary}"
        else:
            # CPU Fallback
            summary = text_body[:500]
            target_text = text_body

        # Encode using the SentenceTransformer engine
        embedding = self.engine.encode([target_text])[0]

        return {
            "file": file_path,
            "title": title,
            "summary": summary,
            "characters": characters,
            "events": events,
            "embedding": embedding
        }

    def _extract_with_llm(self, text: str) -> dict:
        prompt = f"""Analyze the following excerpt from a Japanese Light Novel (it may be in English or Japanese).
Extract the following information:
1. A concise 2-sentence summary of what happens.
2. A list of character names mentioned.
3. A list of key locations or events mentioned.

Excerpt:
{text}

Provide the output STRICTLY in the following JSON format, and do not include any other text:
{{
    "summary": "...",
    "characters": ["...", "..."],
    "events": ["...", "..."]
}}"""
        
        messages = [
            {"role": "system", "content": "You are a highly accurate, bilingual data extraction assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        
        try:
            # Generate output
            outputs = self.llm_model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1, # Low temp for deterministic JSON output
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated tokens
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            
            # Robust JSON extraction using Regex in case the LLM includes conversational filler
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"summary": "", "characters": [], "events": []}
                
        except Exception as e:
            print(f"LLM Extraction failed for chunk: {e}")
            return {"summary": "", "characters": [], "events": []}