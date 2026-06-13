import os
import tarfile
import requests

VPS_URL = "http://vps.avinash9.tech:3000"
TOKEN = "i-am-god"
RUN_ID = "qwen3-4b-jp2en"

CHECKPOINT_ROOT = "/teamspace/studios/this_studio/qwen3-4b-jp2en-lora"


def upload_checkpoint(checkpoint_num):
    checkpoint_dir = os.path.join(
        CHECKPOINT_ROOT,
        f"checkpoint-{checkpoint_num}"
    )

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint folder not found: {checkpoint_dir}"
        )

    tar_name = f"checkpoint-{checkpoint_num}.tar"

    print(f"Creating {tar_name}...")

    with tarfile.open(tar_name, "w") as tar:
        tar.add(
            checkpoint_dir,
            arcname=f"checkpoint-{checkpoint_num}"
        )

    print("Tar created.")

    url = f"{VPS_URL}/upload/{RUN_ID}"

    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }

    with open(tar_name, "rb") as f:
        files = {
            "file": (
                tar_name,
                f,
                "application/x-tar"
            )
        }

        print("Uploading...")

        response = requests.post(
            url,
            headers=headers,
            files=files,
            timeout=None
        )

    print(response.status_code)
    print(response.text)

    response.raise_for_status()

    print("Upload successful.")

    os.remove(tar_name)
    print(f"Deleted local tar: {tar_name}")


if __name__ == "__main__":
    checkpoint_num = input(
        "Checkpoint number: "
    ).strip()

    upload_checkpoint(checkpoint_num)