#!/usr/bin/env python3

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_txt_name(value):
    if value is None:
        raise ValueError("chapter name is None")

    s = str(value).strip()

    if not s:
        raise ValueError("empty chapter name")

    if s.endswith(".txt"):
        return s

    return f"{Path(s).stem}.txt"


def resolve_data_path(root: Path, path_str: str | None):
    """
    Handles:
        JP
        EN
        JP-Working
        EN-Working
        Novels/Arifureta/JP
        /absolute/path
    """

    if not path_str:
        return None

    p = Path(path_str)

    # Absolute path
    if p.is_absolute():
        return p

    # Exists relative to cwd
    if p.exists():
        return p.resolve()

    # Exists relative to root
    candidate = root / p
    if candidate.exists():
        return candidate

    # Strip root folder name if embedded
    root_name = root.name

    parts = list(p.parts)

    if root_name in parts:
        idx = parts.index(root_name)

        stripped = Path(*parts[idx + 1 :])

        candidate = root / stripped

        if candidate.exists():
            return candidate

    return candidate


def find_existing_dir(root: Path, candidates: list[str]):
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def resolve_input_dirs(data, root: Path):
    jp_dir = resolve_data_path(root, data.get("jp_dir"))
    en_dir = resolve_data_path(root, data.get("en_dir"))

    if jp_dir is None or not jp_dir.exists():
        jp_dir = find_existing_dir(
            root,
            [
                "JP-Working",
                "JP",
            ],
        )

    if en_dir is None or not en_dir.exists():
        en_dir = find_existing_dir(
            root,
            [
                "EN-Working",
                "EN",
            ],
        )

    if jp_dir is None:
        raise RuntimeError("Could not locate JP source directory")

    if en_dir is None:
        raise RuntimeError("Could not locate EN source directory")

    return jp_dir, en_dir


def resolve_output_dirs(data, root: Path):
    output_dirs = data.get("output_dirs", {})

    jp_out = resolve_data_path(root, output_dirs.get("jp"))
    en_out = resolve_data_path(root, output_dirs.get("en"))

    if jp_out is None:
        jp_out = root / "JP-Output"

    if en_out is None:
        en_out = root / "EN-Output"

    return jp_out, en_out


def build_plan(data, root: Path):
    jp_dir, en_dir = resolve_input_dirs(data, root)
    jp_out, en_out = resolve_output_dirs(data, root)

    chapters = data.get("chapters", {})

    plan = []
    warnings = []

    used_output_names = set()

    for chapter_key, info in chapters.items():

        status = str(info.get("status", "")).upper()

        if status != "PASS":
            continue

        jp_chapter = info.get("jp_chapter")
        en_chapter = info.get("en_chapter")

        if not jp_chapter or not en_chapter:
            warnings.append(
                f"[SKIP] {chapter_key}: missing jp_chapter/en_chapter"
            )
            continue

        try:
            jp_src_name = normalize_txt_name(jp_chapter)
            en_src_name = normalize_txt_name(en_chapter)
        except Exception as e:
            warnings.append(f"[SKIP] {chapter_key}: {e}")
            continue

        output_chapter = info.get("output_chapter")

        if output_chapter is None:
            output_name = normalize_txt_name(Path(en_src_name).stem)
        else:
            output_name = normalize_txt_name(output_chapter)

        if output_name in used_output_names:
            warnings.append(
                f"[SKIP] {chapter_key}: duplicate output target {output_name}"
            )
            continue

        used_output_names.add(output_name)

        jp_src = jp_dir / jp_src_name
        en_src = en_dir / en_src_name

        jp_dst = jp_out / output_name
        en_dst = en_out / output_name

        if not jp_src.exists():
            warnings.append(
                f"[SKIP] {chapter_key}: missing JP source {jp_src}"
            )
            continue

        if not en_src.exists():
            warnings.append(
                f"[SKIP] {chapter_key}: missing EN source {en_src}"
            )
            continue

        plan.append(
            {
                "chapter_key": chapter_key,
                "jp_src": jp_src,
                "en_src": en_src,
                "jp_dst": jp_dst,
                "en_dst": en_dst,
                "output_name": output_name,
                "reason": info.get("reason"),
                "confidence": info.get("final_confidence"),
            }
        )

    return plan, warnings, jp_dir, en_dir, jp_out, en_out


def verify_copy(src: Path, dst: Path):
    if not filecmp.cmp(src, dst, shallow=False):
        raise RuntimeError(
            f"Verification failed:\n{src}\n->\n{dst}"
        )


def execute_plan(plan):
    for item in plan:

        item["jp_dst"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        item["en_dst"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            item["jp_src"],
            item["jp_dst"],
        )

        shutil.copy2(
            item["en_src"],
            item["en_dst"],
        )

        verify_copy(
            item["jp_src"],
            item["jp_dst"],
        )

        verify_copy(
            item["en_src"],
            item["en_dst"],
        )


def print_preview(
    plan,
    warnings,
    jp_dir,
    en_dir,
    jp_out,
    en_out,
):
    print("=" * 80)
    print("ARRANGEMENT PREVIEW")
    print("=" * 80)

    print(f"\nJP Source : {jp_dir}")
    print(f"EN Source : {en_dir}")

    print(f"JP Output : {jp_out}")
    print(f"EN Output : {en_out}")

    print(f"\nPASS chapters: {len(plan)}")
    print(f"Skipped      : {len(warnings)}")

    if warnings:
        print("\nSkipped entries:")
        for w in warnings:
            print(" ", w)

    print("\nPlanned changes:")
    print("-" * 80)

    for item in plan:
        print(
            f"[{item['chapter_key']}] "
            f"=> {item['output_name']}"
        )

        print(
            f"  JP: {item['jp_src'].name}"
            f" -> {item['jp_dst'].name}"
        )

        print(
            f"  EN: {item['en_src'].name}"
            f" -> {item['en_dst'].name}"
        )

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "arrangement_json",
        help="Path to arrangement_state.json",
    )

    parser.add_argument(
        "root_folder",
        help="Novel root folder",
    )

    args = parser.parse_args()

    arrangement_json = Path(args.arrangement_json).resolve()
    root = Path(args.root_folder).resolve()

    if not arrangement_json.exists():
        print(
            f"ERROR: arrangement file not found:\n"
            f"{arrangement_json}"
        )
        return 1

    if not root.exists():
        print(
            f"ERROR: root folder not found:\n"
            f"{root}"
        )
        return 1

    data = load_json(arrangement_json)

    try:
        (
            plan,
            warnings,
            jp_dir,
            en_dir,
            jp_out,
            en_out,
        ) = build_plan(
            data,
            root,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    if not plan:
        print(
            "Nothing to do. "
            "No PASS entries with valid source files were found."
        )

        if warnings:
            print("\nSkipped entries:")
            for w in warnings:
                print(w)

        return 0

    print_preview(
        plan,
        warnings,
        jp_dir,
        en_dir,
        jp_out,
        en_out,
    )

    response = (
        input(
            "\nApply these changes? [y/N]: "
        )
        .strip()
        .lower()
    )

    if response not in {"y", "yes"}:
        print("Cancelled.")
        return 0

    try:
        execute_plan(plan)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(
        f"\nDone. Successfully arranged "
        f"{len(plan)} chapter pairs."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())