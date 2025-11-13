#!/usr/bin/env python3
"""
Count actual tokens in the notebooks dataset using a proper tokenizer.
"""

import os
import sys
from pathlib import Path
import nbformat
from tqdm import tqdm
import tiktoken
import json
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks_dataset"
OUTPUT_FILE = BASE_DIR / "actual_token_count.json"

# Use GPT-4 tokenizer (cl100k_base encoding)
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_notebook_tokens(notebook_path: Path) -> dict:
    """Count actual tokens in a Jupyter notebook using tiktoken"""
    try:
        with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
            nb = nbformat.read(f, as_version=4)
        
        total_tokens = 0
        cell_tokens = 0
        output_tokens = 0
        
        for cell in nb.cells:
            # Count cell source
            if cell.cell_type in ['code', 'markdown', 'raw']:
                cell_text = cell.source
                cell_tokens += len(tokenizer.encode(cell_text))
            
            # Count outputs for code cells
            if cell.cell_type == 'code' and 'outputs' in cell:
                for output in cell.outputs:
                    if 'text' in output:
                        text = output['text']
                        if isinstance(text, list):
                            text = ''.join(text)
                        output_tokens += len(tokenizer.encode(str(text)))
                    elif 'data' in output:
                        for key, data in output['data'].items():
                            if isinstance(data, str) and key.startswith('text/'):
                                output_tokens += len(tokenizer.encode(data))
        
        total_tokens = cell_tokens + output_tokens
        
        return {
            'success': True,
            'total_tokens': total_tokens,
            'cell_tokens': cell_tokens,
            'output_tokens': output_tokens,
            'size_bytes': os.path.getsize(notebook_path)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'size_bytes': 0
        }


def main():
    print("=" * 80)
    print("ACTUAL TOKEN COUNTER FOR NOTEBOOKS DATASET")
    print("=" * 80)
    print(f"Dataset directory: {NOTEBOOKS_DIR}")
    print(f"Tokenizer: tiktoken (cl100k_base / GPT-4)")
    print("=" * 80)
    
    # Find all notebooks
    print("\n🔍 Scanning for notebooks...")
    notebook_files = []
    for nb_file in NOTEBOOKS_DIR.rglob("*.ipynb"):
        # Skip checkpoint files
        if '.ipynb_checkpoints' not in str(nb_file):
            notebook_files.append(nb_file)
    
    print(f"📊 Found {len(notebook_files):,} notebooks to process\n")
    
    if len(notebook_files) == 0:
        print("❌ No notebooks found!")
        return
    
    # Process notebooks with progress bar
    total_tokens = 0
    total_cell_tokens = 0
    total_output_tokens = 0
    total_size_bytes = 0
    processed = 0
    failed = 0
    
    print("🔢 Counting tokens...")
    for nb_file in tqdm(notebook_files, desc="Processing", unit="notebook"):
        result = count_notebook_tokens(nb_file)
        
        if result['success']:
            total_tokens += result['total_tokens']
            total_cell_tokens += result['cell_tokens']
            total_output_tokens += result['output_tokens']
            total_size_bytes += result['size_bytes']
            processed += 1
        else:
            failed += 1
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"📓 Total notebooks processed: {processed:,}")
    print(f"❌ Failed to process: {failed:,}")
    print(f"📦 Total size: {total_size_bytes / (1024**3):.2f} GB")
    print()
    print(f"🔤 TOTAL TOKENS: {total_tokens:,}")
    print(f"   Cell tokens: {total_cell_tokens:,}")
    print(f"   Output tokens: {total_output_tokens:,}")
    print()
    print(f"📊 In billions: {total_tokens / 1e9:.2f}B tokens")
    print(f"💰 Avg tokens per notebook: {total_tokens / processed:,.0f}")
    print("=" * 80)
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "notebooks_processed": processed,
        "notebooks_failed": failed,
        "total_tokens": total_tokens,
        "cell_tokens": total_cell_tokens,
        "output_tokens": total_output_tokens,
        "total_size_bytes": total_size_bytes,
        "tokenizer": "tiktoken cl100k_base (GPT-4)",
        "tokens_in_billions": round(total_tokens / 1e9, 2)
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

