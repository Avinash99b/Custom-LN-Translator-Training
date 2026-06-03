import os
import re

folder = input("Folder path: ").strip()

files = []

for name in os.listdir(folder):
    m = re.fullmatch(r"(.*?)(\d+)(\.txt)", name)

    if m:
        prefix = m.group(1)
        number = int(m.group(2))
        suffix = m.group(3)

        files.append(
            (number, name, prefix, suffix)
        )

if not files:
    print("No matching files found.")
    exit()

files.sort(key=lambda x: x[0])

print(f"Found {len(files)} files")

# Pass 1: temporary names
for _, name, _, _ in files:
    old_path = os.path.join(folder, name)
    temp_path = os.path.join(folder, f"__tmp__{name}")

    os.rename(old_path, temp_path)

# Pass 2: sequential numbering
for new_num, (_, original_name, prefix, suffix) in enumerate(files, start=1):

    old_temp = os.path.join(
        folder,
        f"__tmp__{original_name}"
    )

    new_name = f"{prefix}{new_num}{suffix}"

    new_path = os.path.join(
        folder,
        new_name
    )

    os.rename(old_temp, new_path)

    if original_name != new_name:
        print(
            f"{original_name} -> {new_name}"
        )

print("Done.")