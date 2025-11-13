#!/usr/bin/env python3
"""
Dataset Audit Utility
Checks how much of your dataset is actual notebook text
vs other heavy binary content.
"""

import os
from pathlib import Path
import json
import nbformat
import tiktoken
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "notebooks_dataset"
OUTPUT_FILE = BASE_DIR / "dataset_audit_report.json"
SAMPLE_COUNT = 100  # how many notebooks to tokenize for sampling

enc = tiktoken.get_encoding("cl100k_base")

def size_gib(n_bytes): return n_bytes / (1024 ** 3)

def tokenize_nb(nb_path: Path):
    """Count tokens (only code/markdown/raw) in one notebook quickly"""
    try:
        nb = nbformat.read(nb_path.open(encoding="utf-8", errors="ignore"), as_version=4)
        tokens = 0
        for cell in nb.cells:
            if cell.cell_type in ("code", "markdown", "raw"):
                tokens += len(enc.encode(cell.source))
        return tokens
    except Exception:
        return 0

def main():
    print("="*80)
    print("NOTEBOOK DATASET AUDIT")
    print("="*80)
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Generating audit report...")

    total_bytes = 0
    total_ipynb_bytes = 0
    notebooks = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            fp = Path(root) / f
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            total_bytes += size
            if f.endswith(".ipynb") and ".ipynb_checkpoints" not in str(fp):
                total_ipynb_bytes += size
                notebooks.append(fp)

    print(f"📦 Total folder size: {size_gib(total_bytes):.2f} GiB")
    print(f"📓 Total .ipynb size: {size_gib(total_ipynb_bytes):.2f} GiB "
          f"({100*total_ipynb_bytes/total_bytes:.1f}% of folder)")

    # sample a few notebooks for token density
    import random
    random.shuffle(notebooks)
    sample = notebooks[:SAMPLE_COUNT]
    sample_tokens = 0
    sample_bytes = 0
    for nb in sample:
        t = tokenize_nb(nb)
        b = nb.stat().st_size
        sample_tokens += t
        sample_bytes += b

    if sample:
        avg_tokens_per_byte = sample_tokens / sample_bytes if sample_bytes else 0
        est_total_tokens = int(avg_tokens_per_byte * total_ipynb_bytes)
    else:
        avg_tokens_per_byte = 0
        est_total_tokens = 0

    print(f"🧮 Sampled {len(sample)} notebooks")
    print(f"   → avg {avg_tokens_per_byte*4:.2f} chars/token equivalent "
          f"({1/avg_tokens_per_byte if avg_tokens_per_byte else 0:.1f} bytes/token)")
    print(f"   → estimated total tokens in .ipynb files: {est_total_tokens:,}")
    print(f"     (≈ {est_total_tokens/1e9:.2f} B tokens)")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(DATASET_DIR),
        "total_folder_bytes": total_bytes,
        "total_ipynb_bytes": total_ipynb_bytes,
        "sample_notebooks": len(sample),
        "sample_tokens": sample_tokens,
        "sample_bytes": sample_bytes,
        "avg_tokens_per_byte": avg_tokens_per_byte,
        "estimated_total_tokens": est_total_tokens,
        "estimated_tokens_in_billions": round(est_total_tokens / 1e9, 2)
    }
    json.dump(report, open(OUTPUT_FILE, "w"), indent=2)
    print(f"\n💾 Saved JSON report to: {OUTPUT_FILE}")

    # Optional: list largest files (comment out if not needed)
    # large_files = sorted(Path(DATASET_DIR).rglob("*"), key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)[:50]
    # import csv
    # with open(BASE_DIR / "largest_files.csv", "w", newline="") as csvf:
    #     import csv
    #     w = csv.writer(csvf)
    #     w.writerow(["file", "size_bytes", "size_GiB"])
    #     for f in large_files:
    #         w.writerow([f, f.stat().st_size, size_gib(f.stat().st_size)])
    # print("📄 largest_files.csv created for inspection.")

if __name__ == "__main__":
    main()
