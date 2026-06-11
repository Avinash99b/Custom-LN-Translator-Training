import os
import requests
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

# curl 'https://kkb-production.jupyter-proxy.kaggle.net/k/325783236/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..F0DnME67a8eCey372AP1iQ.8sZ8NTTg9IuvkWezJrCk_pirS1DNuSiluPfb8VRDAnBObf_OfW-nrLoX2m47g6euH6mFT2z2mYJyMTu92H2lXM5mQaZgSv1oSI4uyt4oJfAq-rS1G_jbErWNG4pWnlJvk3sdrC-aCOd2YRCeJDGjizLtfVhQk59oDzFs7TSunQ0swE_3cFna7HAzbEw4Z_-NkIt03fvjNgSGg7OwWezE0Zgv5enFSvcGcff6PeN-dTUm8qAFP6Sa1q2pwe4GiMBy.xXhx_JWusJ09ubtew7hwaw/proxy/api/contents/merged_qwen3_ln_translator' \
#   -H 'accept: */*' \
#   -H 'accept-language: en-US,en;q=0.6' \
#   -H 'cache-control: max-age=0' \
#   -H 'if-modified-since: Wed, 10 Jun 2026 02:31:25 GMT' \
#   -H 'if-none-match: "8adff1fca64bb67176d18f0270352c09b6484156"' \
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
    "k/325783236/"
    "eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..F0DnME67a8eCey372AP1iQ."
    "8sZ8NTTg9IuvkWezJrCk_pirS1DNuSiluPfb8VRDAnBObf_OfW-nrLoX2m47g6euH6mFT2z2mYJyMTu92H2lXM5mQaZgSv1oSI4uyt4oJfAq-rS1G_jbErWNG4pWnlJvk3sdrC-aCOd2YRCeJDGjizLtfVhQk59oDzFs7TSunQ0swE_3cFna7HAzbEw4Z_-NkIt03fvjNgSGg7OwWezE0Zgv5enFSvcGcff6PeN-dTUm8qAFP6Sa1q2pwe4GiMBy."
    "xXhx_JWusJ09ubtew7hwaw/proxy"
)

CHECKPOINT_NO = 1100
CHECKPOINT_PATH = (
    f"merged_qwen3_ln_translator"
)

OUTPUT_DIR = Path(f"/home/avinash/Projects/Custom-LN-Translator-Training/Model-Checkpoints/merged_qwen3_ln_translator")

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