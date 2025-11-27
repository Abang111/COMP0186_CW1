import os
import json
import pandas as pd
from tqdm import tqdm
from urllib.parse import unquote

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================
# CONFIGS —— 请按需要改这几个变量
# =====================================================
MODEL_DIR  = "/workspace/fy/tbx/Meta-Llama-3.1-8B-Instruct"   # 你的模型文件夹
INPUT_CSV  = "topic_categories.csv"        # ← 改成你的 CSV 名
OUTPUT_CSV = "lectures_with_fos.csv"
ENCODING   = "latin-1"    # 如果你的文件是 UTF-8，就改成 "utf-8"

# 输出 tokens 数量（分类任务 64 足够）
MAX_NEW_TOKENS = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# LOAD MODEL
# =====================================================
print(f"[INFO] Loading Llama model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
print(f"[INFO] Model loaded on {DEVICE}")


# =====================================================
# PROMPTS
# =====================================================
SYSTEM_PROMPT = (
    "You are an expert in academic discipline classification. "
    "You must follow the format strictly and output valid JSON only."
)

FOS_TEXT = """
1. Natural sciences
2. Engineering & Technology
3. Medical & Health sciences
4. Agricultural sciences
5. Social sciences
6. Humanities & Arts
""".strip()


def clean_title(title: str) -> str:
    """Convert URL-like title to readable English."""
    title = unquote(str(title))
    title = title.replace("_", " ")
    return title.strip()


def build_prompt(title: str, categories: str) -> str:
    """Construct user prompt."""
    title = clean_title(title)
    categories = str(categories)

    return f"""We have an online lecture topic.

Title: {title}
Wikipedia categories: {categories}

Classify this topic into ONE of the following UNESCO FOS 2015 broad fields:

{FOS_TEXT}

Return ONLY a JSON object in this exact format:
{{"fos_id": <1-6>, "fos_name": "<name>"}}
"""


# =====================================================
# CALL MODEL FOR ONE ROW
# =====================================================
def classify_one(title: str, categories: str):
    """
    调用 LLAMA，返回 (fos_id, fos_name)
    出错时返回 (None, "Unknown")
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(title, categories)},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.1,
        )

    gen = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # 清理可能的 ```json 包裹
    if text.startswith("```"):
        text = text.strip("`\n ")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        obj = json.loads(text)
        fos_id_raw = obj.get("fos_id", None)
        fos_name   = obj.get("fos_name", "Unknown")

        fos_id = int(fos_id_raw) if fos_id_raw is not None else None
        if fos_id not in range(1, 7):
            return None, "Unknown"

        return fos_id, fos_name

    except Exception:
        print(f"[WARN] Failed to parse JSON from: {text}")
        return None, "Unknown"


# =====================================================
# LOAD CSV
# =====================================================
print(f"[INFO] Loading CSV: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, encoding=ENCODING)

# 直接使用你提供的列名
title_col = "topic_title_raw"
cat_col   = "topic_categories"

if title_col not in df.columns or cat_col not in df.columns:
    raise ValueError("CSV must contain 'topic_title_raw' and 'topic_categories' columns.")

# 若已有结果，则保留（支持断点续跑）
if "fos_id" not in df.columns:
    df["fos_id"] = None
if "fos_name" not in df.columns:
    df["fos_name"] = None


# =====================================================
# MAIN LOOP —— 带 tqdm 进度条
# =====================================================
print("[INFO] Start classification...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
    # 已有结果则跳过
    if pd.notna(row["fos_id"]) and str(row["fos_name"]).strip() not in ["", "Unknown", "None"]:
        continue

    title = row[title_col]
    cats  = row[cat_col]

    fos_id, fos_name = classify_one(title, cats)

    df.at[idx, "fos_id"] = fos_id
    df.at[idx, "fos_name"] = fos_name

    # 每 100 行自动保存一次 checkpoint
    if idx % 100 == 0 and idx > 0:
        df.to_csv(OUTPUT_CSV, index=False, encoding=ENCODING)
        print(f"[INFO] Checkpoint saved at {idx} rows.")

# 最终保存
df.to_csv(OUTPUT_CSV, index=False, encoding=ENCODING)
print(f"\n[INFO] All done! Saved to {OUTPUT_CSV}")
