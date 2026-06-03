import os

# ---------------- CONFIG ----------------

KEYWORDS = [
    "Side Story",
    "Extra Chapter",
    "Bonus Chapter",
    "Special Chapter"
]

CONTEXT_LINES = 8

# ----------------------------------------


folder = input("Folder path: ").strip()

if not os.path.isdir(folder):
    print("Invalid folder.")
    raise SystemExit(1)

txt_files = sorted(
    f for f in os.listdir(folder)
    if f.endswith(".txt")
)

deleted = 0
skipped = 0

for filename in txt_files:

    path = os.path.join(folder, filename)

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            lines = f.readlines()

    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        continue

    match_line = None
    matched_keyword = None

    for idx, line in enumerate(lines):

        lower_line = line.lower()

        for keyword in KEYWORDS:

            if keyword.lower() in lower_line:
                match_line = idx
                matched_keyword = keyword
                break

        if match_line is not None:
            break

    if match_line is None:
        continue

    start = max(
        0,
        match_line - CONTEXT_LINES
    )

    end = min(
        len(lines),
        match_line + CONTEXT_LINES + 1
    )

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print(f"MATCHED: {matched_keyword}")
    print("-" * 80)

    for i in range(start, end):

        marker = ">>> " if i == match_line else "    "

        print(
            f"{marker}{i+1:5d}: "
            f"{lines[i].rstrip()}"
        )

    print("-" * 80)
    print(
        "[ENTER] Delete | "
        "[n] Skip | "
        "[q] Quit"
    )

    choice = input("> ").strip().lower()

    if choice == "q":
        break

    if choice == "n":
        skipped += 1
        continue

    try:
        os.remove(path)

        deleted += 1

        print(
            f"Deleted: {filename}"
        )

    except Exception as e:

        print(
            f"Failed to delete "
            f"{filename}: {e}"
        )

print("\nFinished.")
print(f"Deleted: {deleted}")
print(f"Skipped: {skipped}")