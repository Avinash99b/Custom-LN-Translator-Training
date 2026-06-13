# Observations

## Chapter & Paragraph Segmentation Report
- This report summarizes the observations i have observed while solving the problem of segmenting the chapters and finding their correct associated JP<->EN pairs for training.
  
## Approaches I tried.
 
### Embeding Based Approach 
1. I first tried implementing my idea of diagonal based segment analysis to also find the sentences based embeding to find the paragraphs.
2. This approach worked but its far too noisy for training a model. And as for why i have classified it as noisy is because the sentences contain artifcats like ---------- or ********** which are used to separate the paragraphs in the novels. And these artifacts are not present in the training data and thus it would be a problem for the model to learn to ignore them.

### LLM Based Approach
1. I then tried using the LLM to segment the chapters and find the correct JP<->EN pairs. This approach worked much better than the embedding based approach and the results were much cleaner and more suitable for training a model. 
2. Now the main problem i ran into while doing the LLM based approach is that the reliability is not as good as i would like it to be. The LLM sometimes fails to segment the chapters correctly and thus we end up with some noisy data. This is especially true when the chapters are very long and contain a lot of dialogue.


## Models i tried with LLM Based Approach

### GPT-5
1. By far this is the most powerful model i have tried and it gives the best results. The segmentation is very good and the JP<->EN pairs are also very accurate. However, the main problem with this model is that it is very expensive to use and thus it is not feasible to use it for segmenting a large number of chapters.
2. The expenses nearly got 80$+ for just 800 chapters and thus it is not a viable option for segmenting the entire dataset.

### Deepseek V4-pro
1. My observation is that this model though cheaper and has more intellegence is performing less reliably or not even comming close to the quality i want.

Example:-

```
サトゥーです。リアルダンジョンアタックはもうしたくないサトゥーです。
迷宮から蒸発した腕悪魔も気になりますが……。
とりあえず平穏を享受したいと思います。
◇
迷宮を出たそこは学校の校庭くらいの広さの更地だった。

Deepseek returned this area as correct for this input->

I’d like to be together with a person of my type for once.
◇
A stark naked little girl is straddling on top of my waist.
...What the heck is this, a dream?
I’ve experienced similar situation with my younger cousins during a long vacation on my grandfather’s home in the countryside, long ago.
```
Example 2:
```
この世界に基本的人権を広めるのは茨の道ですね。
広める気もありませんが……。

Deepseek returned this area as correct for this input->

Satou’s here. The typical Japanese who becomes so happy when they meet another Japanese in a foreign country, he let his guard down, Satou.
Since we’re speaking the same language, and because our sense of value and foundation are close, I become relieved.

```
### My Idea
1. After thinking for a while i had an idea.
2. That is to train a custom mini model locally which is best at segementation itself.
3. Normally we asked GPT-5 to return something like this:
```
SYSTEM_PROMPT = """You are a bilingual segmentation assistant for Japanese light novel translation data.

You will receive a JP chapter and its EN translation. Noise/header lines have already been
removed — only story lines are shown. Each language uses 1-indexed line numbers that reflect
the ORIGINAL file (some numbers may be absent because they were pre-filtered).

─── TASK  ·  Segment the story lines ──────────────────────────────────────────
Partition ALL visible JP lines and ALL visible EN lines into semantically coherent
training segments. Each segment pairs a JP span with its corresponding EN span. Rules:
• Each segment must be a complete meaning unit: a scene, a dialogue exchange, a paragraph
  cluster, or a continuous inner monologue.
• Estimated combined token count per segment ≤ {max_tokens}
  (estimate: 1 token ≈ 1.5 JP chars, 4 EN chars).
• jp_begin/jp_end and en_begin/en_end are line numbers from the ORIGINAL file
  (use the numbers shown in the input, not re-indexed positions).
• Asymmetric boundaries are expected — JP and EN rarely split at the same lines.
• Cover every visible line. No gaps, no overlaps within each side.
• Prefer grouping short lines into a meaningful unit over splitting mid-thought.
• Dialogue exchanges should be one segment.

─── OUTPUT FORMAT ──────────────────────────────────────────────────────────────
Return ONLY a JSON object (no markdown, no prose):
{{
  "segments": [
    {{
      "jp_begin": 5,
      "jp_end": 10,
      "en_begin": 3,
      "en_end": 9,
      "semantic_unit": "narration"
    }}
  ]
}}

• semantic_unit: one of "narration", "dialogue", "inner_monologue", "action", "mixed".
"""

USER_PROMPT_TEMPLATE = """Novel: {novel}
Chapter: {chapter_id}
Max tokens per segment: {max_tokens}

=== JP TEXT (story lines only) ===
{jp_numbered}

=== EN TEXT (story lines only) ===
{en_numbered}

Return only the JSON object."""
``` 
3. My idea is that we can use the already existing 10k lines of data which i so painfully costly fetched through GPT 5 and train Qwen 3 on the segmentation dataset to precisely segment the input.

