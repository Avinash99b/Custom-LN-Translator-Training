from pathlib import Path
import re
import sys

def rename_chapters(folder_path):
    folder = Path(folder_path)

    for file in folder.glob("*.txt"):
        # Extract chapter number from filenames like:
        # n9669bk-jap-ch-190.txt
        match = re.search(r"ch-(\d+)\.txt$", file.name)

        if not match:
            print(f"Skipping: {file.name}")
            continue

        chapter_no = match.group(1)
        new_name = folder / f"{chapter_no}.txt"

        if new_name.exists():
            print(f"Already exists, skipping: {new_name.name}")
            continue

        file.rename(new_name)
        print(f"{file.name} -> {new_name.name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <folder_path>")
        sys.exit(1)

    rename_chapters(sys.argv[1])