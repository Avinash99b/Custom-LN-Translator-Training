import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "thefrigidliquidation/opt-1.3b-lightnovels"

print("=" * 80)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("=" * 80)
print("MODEL INFO")
print("=" * 80)

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocab size:", tokenizer.vocab_size)
print("Model type:", model.config.model_type)

print()

# --------------------------------------------------
# TOKENIZATION TEST
# --------------------------------------------------

jp_text = "彼女は静かに微笑んだ。"

tokens = tokenizer.tokenize(jp_text)

print("=" * 80)
print("TOKENIZATION TEST")
print("=" * 80)

print("Input:")
print(jp_text)

print()
print("Token count:", len(tokens))
print("Tokens:")
print(tokens)

print()

# --------------------------------------------------
# GENERATION HELPER
# --------------------------------------------------

def generate(prompt, max_new_tokens=200):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return decoded


# --------------------------------------------------
# TEST 1
# TRANSLATION
# --------------------------------------------------

translation_prompt = """
Translate the following Japanese light novel passage into natural English.

Japanese:
彼女は静かに微笑んだ。

English:
"""

print("=" * 80)
print("TEST 1: JP -> EN TRANSLATION")
print("=" * 80)

print(generate(translation_prompt))

print()

# --------------------------------------------------
# TEST 2
# LONGER TRANSLATION
# --------------------------------------------------

translation_prompt2 = """
Translate the following Japanese light novel text into English.

Japanese:
「お兄ちゃん、本当に行っちゃうの？」
彼女は不安そうな表情で俺を見上げた。

English:
"""

print("=" * 80)
print("TEST 2: DIALOGUE TRANSLATION")
print("=" * 80)

print(generate(translation_prompt2))

print()

# --------------------------------------------------
# TEST 3
# ENGLISH GENERATION
# --------------------------------------------------

english_prompt = """
Chapter 1

The girl looked up at me and smiled.
"""

print("=" * 80)
print("TEST 3: ENGLISH GENERATION")
print("=" * 80)

print(generate(english_prompt))

print()

# --------------------------------------------------
# TEST 4
# JAPANESE GENERATION
# --------------------------------------------------

japanese_prompt = """
第一章

少女は俺を見上げた。
"""

print("=" * 80)
print("TEST 4: JAPANESE GENERATION")
print("=" * 80)

print(generate(japanese_prompt))

print()

# --------------------------------------------------
# TEST 5
# FEW SHOT TRANSLATION
# --------------------------------------------------

few_shot = """
Japanese: こんにちは。
English: Hello.

Japanese: 私は学生です。
English: I am a student.

Japanese: 彼女は静かに微笑んだ。
English:
"""

print("=" * 80)
print("TEST 5: FEW-SHOT TRANSLATION")
print("=" * 80)

print(generate(few_shot))

print()

# --------------------------------------------------
# SCORECARD
# --------------------------------------------------

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("""
Good signs:
- Outputs English in translation tests
- Preserves meaning
- Doesn't continue story
- Japanese token count is reasonable

Bad signs:
- Outputs only Japanese
- Repeats phrases endlessly
- Hallucinates story continuation
- Tokenization looks corrupted

If translation tests fail but English generation succeeds,
the model is a story-generation model, not a translation model.
""")