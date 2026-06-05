import os
import requests
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
# curl 'https://kkb-production.jupyter-proxy.kaggle.net/k/324668314/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..qe_g9uvm9lBVsu7r3pBJ0w.-Y8YGY0sJdR-crrDbezeEpU2EJpIZU_VjR-zzOMhRCKiymX7TBgc-4UvwK8-EeqFLDZ5U-0ejW3J4zLt-eBPWFlm8YlvOnDdBCQJJYGmPw3kNGZHTRKU5KrL3W5_zBfCwXPGE9K3L29oJjAo1Hg2FCGmP8LUp7F_O7_ghbrrDK0jWuoGK3M5ltiZ2sPihEjWzomGUImwAF-DSYZ7MzP2Ti7Xx0sKCVraYioxphzlw5Gfe2ULzrBmoypHy7AGU9n0.BGn2peHpBoq2cyE9dOI1Xg/proxy/api/contents/qwen3_ln_translation/phase1_chunk_heavy' \
#   -H 'accept: */*' \
#   -H 'accept-language: en-US,en;q=0.9' \
#   -H 'cache-control: max-age=0' \
#   -H 'if-modified-since: Fri, 05 Jun 2026 06:21:46 GMT' \
#   -H 'if-none-match: "955ab99718c11c4c54e8e8422d587e1b0f1ddf9c"' \
#   -H 'origin: https://www.kaggle.com' \
#   -H 'priority: u=1, i' \
#   -H 'referer: https://www.kaggle.com/' \
#   -H 'sec-ch-ua: "Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"' \
#   -H 'sec-ch-ua-mobile: ?0' \
#   -H 'sec-ch-ua-platform: "Linux"' \
#   -H 'sec-fetch-dest: empty' \
#   -H 'sec-fetch-mode: cors' \
#   -H 'sec-fetch-site: cross-site' \
#   -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'

BASE_URL = (
    "https://kkb-production.jupyter-proxy.kaggle.net/"
    "k/324668314/"
    "eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..qe_g9uvm9lBVsu7r3pBJ0w.-Y8YGY0sJdR-crrDbezeEpU2EJpIZU_VjR-zzOMhRCKiymX7TBgc-4UvwK8-EeqFLDZ5U-0ejW3J4zLt-eBPWFlm8YlvOnDdBCQJJYGmPw3kNGZHTRKU5KrL3W5_zBfCwXPGE9K3L29oJjAo1Hg2FCGmP8LUp7F_O7_ghbrrDK0jWuoGK3M5ltiZ2sPihEjWzomGUImwAF-DSYZ7MzP2Ti7Xx0sKCVraYioxphzlw5Gfe2ULzrBmoypHy7AGU9n0.BGn2peHpBoq2cyE9dOI1Xg/proxy"
)

CHECKPOINT_NO = 100
CHECKPOINT_PATH = (
    f"qwen3_ln_translation/phase1_chunk_heavy/checkpoint-{CHECKPOINT_NO}"
)

OUTPUT_DIR = Path(f"/home/avinash/Projects/Custom-LN-Translator-Training/Model-Checkpoints/Arifureta/GPU/ArrangedAndCleanedDataset/checkpoint-{CHECKPOINT_NO}")

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