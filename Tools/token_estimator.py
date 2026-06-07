#!/usr/bin/env python3
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("pip install tiktoken")
    raise SystemExit(1)

def token_count(text: str, encoder):
    return len(encoder.encode(text))

def analyze_novel(novel_dir: Path, encoder):
    en_dir = novel_dir / "EN-Output"
    jp_dir = novel_dir / "JP-Output"
    
    if not en_dir.exists() or not jp_dir.exists():
        return None
        
    en_files = {
        f.name: f
        for f in en_dir.glob("*.txt")
    }
    jp_files = {
        f.name: f
        for f in jp_dir.glob("*.txt")
    }
    
    paired_names = sorted(
        set(en_files.keys()) &
        set(jp_files.keys())
    )
    # These are already sorted arrays of the filenames
    orphan_en_names = sorted(
        set(en_files.keys()) -
        set(jp_files.keys())
    )
    orphan_jp_names = sorted(
        set(jp_files.keys()) -
        set(en_files.keys())
    )
    
    # Store the actual Path objects for deletion later
    orphan_en_paths = [en_files[name] for name in orphan_en_names]
    orphan_jp_paths = [jp_files[name] for name in orphan_jp_names]
    
    en_tokens = 0
    jp_tokens = 0
    en_chars = 0
    jp_chars = 0
    pair_count = 0
    
    for filename in paired_names:
        try:
            en_text = en_files[filename].read_text(
                encoding="utf-8",
                errors="ignore"
            )
            jp_text = jp_files[filename].read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue
            
        pair_count += 1
        en_chars += len(en_text)
        jp_chars += len(jp_text)
        en_tokens += token_count(
            en_text,
            encoder
        )
        jp_tokens += token_count(
            jp_text,
            encoder
        )
        
    return {
        "novel": novel_dir.name,
        "pairs": pair_count,
        "en_tokens": en_tokens,
        "jp_tokens": jp_tokens,
        "en_chars": en_chars,
        "jp_chars": jp_chars,
        "orphan_en_paths": orphan_en_paths,
        "orphan_jp_paths": orphan_jp_paths,
        "orphan_en_names": orphan_en_names,
        "orphan_jp_names": orphan_jp_names,
    }

def main():
    root_input = "/home/avinash/Projects/Custom-LN-Translator-Training/Assets/Novels".strip()
    root = Path(root_input)
    
    if not root.exists():
        print("Folder not found.")
        return
        
    print("\nLoading tokenizer...")
    encoder = tiktoken.get_encoding("o200k_base")
    
    total_pairs = 0
    total_en_tokens = 0
    total_jp_tokens = 0
    total_en_chars = 0
    total_jp_chars = 0
    
    # Tracking orphans for the table and deletion
    all_orphan_en_paths = []
    all_orphan_jp_paths = []
    orphan_summary = []
    
    novels_processed = 0
    
    print("\nPER NOVEL")
    print("=" * 100)
    
    for novel_dir in sorted(root.iterdir()):
        if not novel_dir.is_dir():
            continue
            
        result = analyze_novel(novel_dir, encoder)
        if result is None:
            continue
            
        novels_processed += 1
        total_pairs += result["pairs"]
        total_en_tokens += result["en_tokens"]
        total_jp_tokens += result["jp_tokens"]
        total_en_chars += result["en_chars"]
        total_jp_chars += result["jp_chars"]
        
        en_orphans = result["orphan_en_paths"]
        jp_orphans = result["orphan_jp_paths"]
        
        all_orphan_en_paths.extend(en_orphans)
        all_orphan_jp_paths.extend(jp_orphans)
        
        if en_orphans or jp_orphans:
            orphan_summary.append({
                "novel": result["novel"],
                "en_count": len(en_orphans),
                "jp_count": len(jp_orphans),
                "en_names": result["orphan_en_names"],
                "jp_names": result["orphan_jp_names"]
            })
        
        combined_tokens = result["en_tokens"] + result["jp_tokens"]
        
        print(
            f"{result['novel'][:45]:45} | "
            f"Pairs={result['pairs']:6,} | "
            f"Tokens={combined_tokens:10,} | "
            f"Orphan EN={len(en_orphans):4,} | "
            f"Orphan JP={len(jp_orphans):4,}"
        )
        
    combined_tokens = total_en_tokens + total_jp_tokens
    combined_chars = total_en_chars + total_jp_chars
    total_orphan_en = len(all_orphan_en_paths)
    total_orphan_jp = len(all_orphan_jp_paths)
    
    print("\n")
    print("=" * 100)
    print("FINAL TRAINING CORPUS STATISTICS")
    print("=" * 100)
    print(f"Novels Processed       : {novels_processed:,}")
    print(f"Aligned Chapter Pairs  : {total_pairs:,}")
    print()
    print(f"EN Tokens              : {total_en_tokens:,}")
    print(f"JP Tokens              : {total_jp_tokens:,}")
    print(f"Combined Tokens        : {combined_tokens:,}")
    print()
    print(f"EN Dataset Size        : {total_en_tokens/1_000_000:.3f}M")
    print(f"JP Dataset Size        : {total_jp_tokens/1_000_000:.3f}M")
    print(f"Combined Dataset Size  : {combined_tokens/1_000_000:.3f}M")
    print()
    print(f"Characters             : {combined_chars:,}")
    print(f"Orphan EN Files        : {total_orphan_en:,}")
    print(f"Orphan JP Files        : {total_orphan_jp:,}")
    
    if total_pairs:
        print()
        print(f"Avg EN Tokens/Pair     : {total_en_tokens/total_pairs:,.1f}")
        print(f"Avg JP Tokens/Pair     : {total_jp_tokens/total_pairs:,.1f}")
        print(f"Avg Total Tokens/Pair  : {combined_tokens/total_pairs:,.1f}")

    # --- Orphaned Files Tabular Report & Deletion ---
    if orphan_summary:
        print("\n")
        print("=" * 100)
        print("ORPHANED FILES SUMMARY (AGGREGATED BY FOLDER)")
        print("=" * 100)
        print(f"{'Novel Folder':<60} | {'EN Orphans':<12} | {'JP Orphans':<12}")
        print("-" * 100)
        
        for summary in orphan_summary:
            # Print the base folder stats
            print(f"{summary['novel'][:60]:<60} | {summary['en_count']:<12,} | {summary['jp_count']:<12,}")
            
            # Print the sorted arrays of orphaned chapters underneath
            if summary['en_names']:
                print(f"    ↳ EN Chapters: {summary['en_names']}")
            if summary['jp_names']:
                print(f"    ↳ JP Chapters: {summary['jp_names']}")
            
            # Add a small divider for readability if there are chapters listed
            print("-" * 100)
        
        print("\n")
        choice = input(f"Do you want to permanently delete all {total_orphan_en + total_orphan_jp} orphaned files? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            deleted_count = 0
            for file_path in all_orphan_en_paths + all_orphan_jp_paths:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
            print(f"\nSuccessfully deleted {deleted_count} orphaned files.")
        else:
            print("\nDeletion skipped.")
    else:
        print("\nNo orphaned files found! Dataset is perfectly aligned.")

    print("\nDone.")

if __name__ == "__main__":
    main()