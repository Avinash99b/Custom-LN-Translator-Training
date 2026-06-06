import os
import requests
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

# curl 'https://kkb-production.jupyter-proxy.kaggle.net/k/324718180/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..-pXTQkX2S2_feovc7-xJsw.Es88bT4ScxRPI0_t5up61AwLIw0tp0Q-VVkqsNwLdFaab8yHFrUPe6CMBsAYXY65YXNM6MFNhU7srs6OpUQJx5izkhjRWMFkOfRqu-0EX0H1EWZGHxkYDHsU8ct4bBQO8xDNWBAuNt4D2afoQnoa-EC-Bw0ZTWiReWOET1yc2ylTNFrSelFnyd2710HQs9HRucTcZ1r3smGj6tXQvnbJjQr8uoy0yZyl6Q_h-wk5j0Vnlw3L5O9v9r0EHUBdQ7kn.HMSCquM7W966mKCx1IuuXw/proxy/api/contents/qwen3_ln_translation/phase1_chunk_heavy/checkpoint-400' \
#   -H 'accept: */*' \
#   -H 'accept-language: en-US,en;q=0.6' \
#   -H 'cache-control: max-age=0' \
#   -H 'origin: https://www.kaggle.com' \
#   -H 'priority: u=1, i' \
#   -H 'referer: https://www.kaggle.com/' \
#   -H 'sec-ch-ua: "Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"' \
#   -H 'sec-ch-ua-mobile: ?0' \
#   -H 'sec-ch-ua-platform: "Linux"' \
#   -H 'sec-fetch-dest: empty' \
#   -H 'sec-fetch-mode: cors' \
#   -H 'sec-fetch-site: cross-site' \
#   -H 'sec-gpc: 1' \
#   -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'

BASE_URL = (
    "https://kkb-production.jupyter-proxy.kaggle.net/"
    "k/324803337/"
    "eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..MqDumrxozPLPrUOnqSanew."
    "pw41nInel22t7n0bSS0lFLmlv__6agEPF3zI3JEDZyMHxLvaTe3NJDvMfH39puwIQ7M1emKAXdoWiID6T7N2ADfoD_K-XeNTySWcc8_Ah7hi11UyfC7W4H40ww2BEMObJAQsH1z8dwhN_2hFY0xVds_GmCHDcPM1xa3MB70SfkQh8RB7BLhljbfBM3mE02cdTdoukCeF7eY9myhVDoeTbH4wWe82sWrix6ZP2GBFv6DOjvHPLjefj9o6T_dfUzO6."
    "S-6DKw-UG-cfqDtyoPDfyA/proxy"
)

CHECKPOINT_NO = 1100
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