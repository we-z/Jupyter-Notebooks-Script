#!/usr/bin/env python3
"""
Count approximate total tokens for *all files* in a dataset folder.
This treats every byte of every file (text or binary) as data.
"""

import os
from pathlib import Path
from tqdm import tqdm
import tiktoken
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "notebooks_dataset"
OUTPUT_FILE = BASE_DIR / "total_folder_token_count.json"

# GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def file_token_count(path: Path) -> int:
    """Tokenize the contents of any file (text or binary)."""
    try:
        # read bytes safely
        data = path.read_bytes()
        # decode bytes to latin-1 (1 byte = 1 char, no errors)
        # ensures we preserve every byte in a 1:1 mapping
        text = data.decode("latin-1")
        tokens = len(enc.encode(text))
        return tokens
    except Exception:
        return 0

def main():
    total_tokens = 0
    total_bytes = 0
    file_count = 0

    all_files = [p for p in DATASET_DIR.rglob("*") if p.is_file()]
    print(f"📦 Found {len(all_files):,} files to process in {DATASET_DIR}")

    for f in tqdm(all_files, desc="Tokenizing all files"):
        try:
            size = f.stat().st_size
            total_bytes += size
            file_count += 1
            total_tokens += file_token_count(f)
        except Exception:
            continue

    result = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(DATASET_DIR),
        "total_files": file_count,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "tokens_in_billions": round(total_tokens / 1e9, 2),
        "approx_bytes_per_token": round(total_bytes / total_tokens, 3) if total_tokens else None,
    }

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"📦 Total folder size: {total_bytes / (1024**3):.2f} GiB")
    print(f"🔤 Total tokens (all bytes tokenized): {total_tokens:,}")
    print(f"📊 In billions: {total_tokens / 1e9:.2f} B tokens")
    print(f"💰 Avg bytes/token: {total_bytes / total_tokens:.2f}" if total_tokens else "")
    print("="*80)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()