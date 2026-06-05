import os
import requests
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

BASE_URL = (
    "https://kkb-production.jupyter-proxy.kaggle.net/"
    "k/324607562/"
    "eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..41kNjujIVAhH2AedlZJa4w."
    "sn8sGZNQRvggDl3QSUkUMyCEc1DHbY3tw74WO8vHMQzLyzeqtqW_lhpHRO4vX7L_y37gEZfknYEzihdXg9WWd7vUz4efSanZ_a-8OGtPmA8y5ZiFijOacdi1wmRiUMHn8ihsVsOHZJsZD7dbKvYR3gIY3i_S_WNf4myF_vKnGm4jllsDm0t1btiPjHnOlqiD-dbHzg3TQe3yMQAV0vDSJtVoRiyxupqf2c0PTw_V0ITGfOTg6PyED6d6-TK7wbIr."
    "90kVSIr2mcj3huwcGSheEQ/proxy"
)

CHECKPOINT_NO = 300
CHECKPOINT_PATH = (
    f"qwen3_ln_translation/phase1_chunk_heavy/checkpoint-{CHECKPOINT_NO}"
)

OUTPUT_DIR = Path(f"/home/avinash/Projects/Custom-LN-Translator-Training/Model-Checkpoints/Arifureta/GPU/ArrangedButNonCleanedDataset/checkpoint-{CHECKPOINT_NO}")

HEADERS = {
    "accept": "*/*",
    "origin": "https://www.kaggle.com",
    "referer": "https://www.kaggle.com/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
}

# =====================================================
# DOWNLOAD LOGIC
# =====================================================

session = requests.Session()
session.headers.update(HEADERS)


def download_file(remote_path: str):
    url = f"{BASE_URL}/files/{remote_path}"

    local_path = OUTPUT_DIR / Path(remote_path).relative_to(CHECKPOINT_PATH)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {remote_path}")

    with session.get(url, stream=True) as r:
        r.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Saved: {local_path}")


def walk_directory(remote_path: str):
    url = f"{BASE_URL}/api/contents/{remote_path}"

    r = session.get(url)
    r.raise_for_status()

    data = r.json()

    for item in data["content"]:
        item_type = item["type"]
        item_path = item["path"]

        if item_type == "directory":
            walk_directory(item_path)

        elif item_type == "file":
            download_file(item_path)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Enumerating checkpoint...")
    walk_directory(CHECKPOINT_PATH)

    print("Done.")