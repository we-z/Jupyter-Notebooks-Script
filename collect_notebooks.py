#!/usr/bin/env python3
"""
Jupyter Notebook Dataset Collector
Searches GitHub for repositories containing Jupyter Notebooks and clones them locally.
"""

import os
import sys
import json
import time
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import nbformat
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_DIR = Path(__file__).parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks_dataset"
LOGS_DIR = BASE_DIR / "logs"
CLONE_LOG_DIR = LOGS_DIR / "clone_logs"
METADATA_FILE = BASE_DIR / "collection_metadata.json"
STATS_FILE = BASE_DIR / "collection_stats.json"
SEARCH_CACHE_FILE = BASE_DIR / "search_cache.json"

# GitHub API settings
GITHUB_API_BASE = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Collection settings
TARGET_TOKENS = 100_000_000_000  # 100B tokens
MAX_REPOS_PER_SEARCH = 1000
REPOS_PER_PAGE = 100
MAX_WORKERS = 5  # Parallel cloning workers
RATE_LIMIT_BUFFER = 10  # Keep this many API calls in reserve

# Logging setup
LOGS_DIR.mkdir(exist_ok=True)
CLONE_LOG_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Main logger
log_file = LOGS_DIR / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CollectionStats:
    """Track collection statistics in real-time"""
    
    def __init__(self):
        self.repos_discovered = 0
        self.repos_cloned = 0
        self.repos_failed = 0
        self.notebooks_collected = 0
        self.total_tokens = 0
        self.total_size_bytes = 0
        self.start_time = time.time()
        self.last_save = time.time()
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + value)
        
        # Auto-save stats every 30 seconds
        if time.time() - self.last_save > 30:
            self.save()
            
    def save(self):
        """Save stats to file"""
        stats = {
            "repos_discovered": self.repos_discovered,
            "repos_cloned": self.repos_cloned,
            "repos_failed": self.repos_failed,
            "notebooks_collected": self.notebooks_collected,
            "total_tokens": self.total_tokens,
            "total_size_bytes": self.total_size_bytes,
            "elapsed_time_seconds": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat()
        }
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        self.last_save = time.time()
        
    def display(self):
        """Display current stats"""
        elapsed = time.time() - self.start_time
        tokens_gb = self.total_tokens / 1e9
        target_gb = TARGET_TOKENS / 1e9
        progress_pct = (self.total_tokens / TARGET_TOKENS) * 100
        
        logger.info("=" * 80)
        logger.info("COLLECTION STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Repos Discovered: {self.repos_discovered:,}")
        logger.info(f"Repos Cloned: {self.repos_cloned:,}")
        logger.info(f"Repos Failed: {self.repos_failed:,}")
        logger.info(f"Notebooks Collected: {self.notebooks_collected:,}")
        logger.info(f"Total Tokens: {self.total_tokens:,} ({tokens_gb:.2f}B / {target_gb:.0f}B)")
        logger.info(f"Progress: {progress_pct:.2f}%")
        logger.info(f"Total Size: {self.total_size_bytes / (1024**3):.2f} GB")
        logger.info(f"Elapsed Time: {elapsed / 3600:.2f} hours")
        if self.repos_cloned > 0:
            logger.info(f"Avg Notebooks/Repo: {self.notebooks_collected / self.repos_cloned:.1f}")
            logger.info(f"Avg Tokens/Notebook: {self.total_tokens / self.notebooks_collected:,.0f}")
        logger.info("=" * 80)


stats = CollectionStats()


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Using rough estimate: 1 token ≈ 4 characters for code/technical content
    """
    return len(text) // 4


def count_notebook_tokens(notebook_path: Path) -> int:
    """Count tokens in a Jupyter notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        total_tokens = 0
        for cell in nb.cells:
            if cell.cell_type in ['code', 'markdown', 'raw']:
                total_tokens += estimate_tokens(cell.source)
            # Count outputs too
            if cell.cell_type == 'code' and 'outputs' in cell:
                for output in cell.outputs:
                    if 'text' in output:
                        total_tokens += estimate_tokens(str(output['text']))
                    elif 'data' in output:
                        for data in output['data'].values():
                            if isinstance(data, str):
                                total_tokens += estimate_tokens(data)
        
        return total_tokens
    except Exception as e:
        logger.debug(f"Error counting tokens in {notebook_path}: {e}")
        # Fallback to file size estimation
        return os.path.getsize(notebook_path) // 4


def check_rate_limit() -> Dict:
    """Check GitHub API rate limit"""
    try:
        response = requests.get(f"{GITHUB_API_BASE}/rate_limit", headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        return None


def wait_for_rate_limit():
    """Wait if rate limit is close to being exceeded"""
    rate_info = check_rate_limit()
    if rate_info:
        remaining = rate_info['resources']['search']['remaining']
        reset_time = rate_info['resources']['search']['reset']
        
        if remaining < RATE_LIMIT_BUFFER:
            wait_seconds = reset_time - time.time() + 5
            if wait_seconds > 0:
                logger.warning(f"Rate limit low ({remaining} remaining). Waiting {wait_seconds:.0f} seconds...")
                time.sleep(wait_seconds)


def search_github_repos(query: str, page: int = 1, per_page: int = 100) -> Optional[Dict]:
    """Search GitHub for repositories"""
    wait_for_rate_limit()
    
    url = f"{GITHUB_API_BASE}/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page
    }
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            logger.error("Rate limit exceeded. Waiting...")
            wait_for_rate_limit()
            return search_github_repos(query, page, per_page)
        else:
            logger.error(f"GitHub API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error searching GitHub: {e}")
        return None


def clone_repository(repo_url: str, repo_name: str, clone_dir: Path) -> bool:
    """Clone a GitHub repository with live logging"""
    repo_path = clone_dir / repo_name
    
    # Skip if already cloned
    if repo_path.exists():
        logger.info(f"⏭️  Repository {repo_name} already exists, skipping...")
        return True
    
    log_file = CLONE_LOG_DIR / f"{repo_name.replace('/', '_')}.log"
    
    logger.info(f"🚀 Starting clone: {repo_name}")
    logger.info(f"   URL: {repo_url}")
    logger.info(f"   Destination: {repo_path}")
    
    try:
        # Clone with progress
        cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_path)]
        
        with open(log_file, 'w') as f:
            f.write(f"Cloning {repo_name}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("-" * 80 + "\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    # Write to log file
                    f.write(line + "\n")
                    f.flush()
                    # Display live progress
                    logger.info(f"   📥 {repo_name}: {line}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"✅ Successfully cloned: {repo_name}")
                f.write(f"\nCompleted: {datetime.now().isoformat()}\n")
                f.write("Status: SUCCESS\n")
                return True
            else:
                logger.error(f"❌ Failed to clone {repo_name} (exit code: {process.returncode})")
                f.write(f"\nCompleted: {datetime.now().isoformat()}\n")
                f.write(f"Status: FAILED (exit code: {process.returncode})\n")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error cloning {repo_name}: {e}")
        with open(log_file, 'a') as f:
            f.write(f"\nError: {str(e)}\n")
            f.write("Status: ERROR\n")
        return False


def process_repository(repo_info: Dict) -> Dict:
    """Process a single repository: clone and count notebooks"""
    repo_name = repo_info['full_name']
    repo_url = repo_info['clone_url']
    
    result = {
        'repo_name': repo_name,
        'repo_url': repo_url,
        'cloned': False,
        'notebooks_found': 0,
        'tokens': 0,
        'size_bytes': 0
    }
    
    # Clone repository
    if clone_repository(repo_url, repo_name, NOTEBOOKS_DIR):
        result['cloned'] = True
        stats.update(repos_cloned=1)
        
        # Find and process notebooks
        repo_path = NOTEBOOKS_DIR / repo_name
        notebook_files = list(repo_path.rglob("*.ipynb"))
        
        logger.info(f"📊 Found {len(notebook_files)} notebooks in {repo_name}")
        
        for nb_file in notebook_files:
            # Skip checkpoint files
            if '.ipynb_checkpoints' in str(nb_file):
                continue
                
            try:
                tokens = count_notebook_tokens(nb_file)
                size = os.path.getsize(nb_file)
                
                result['notebooks_found'] += 1
                result['tokens'] += tokens
                result['size_bytes'] += size
                
                logger.debug(f"   📓 {nb_file.name}: {tokens:,} tokens, {size / 1024:.1f} KB")
                
            except Exception as e:
                logger.warning(f"   ⚠️  Error processing {nb_file.name}: {e}")
        
        stats.update(
            notebooks_collected=result['notebooks_found'],
            total_tokens=result['tokens'],
            total_size_bytes=result['size_bytes']
        )
        
        logger.info(f"✨ {repo_name}: {result['notebooks_found']} notebooks, {result['tokens']:,} tokens")
        
    else:
        stats.update(repos_failed=1)
    
    stats.display()
    return result


def save_search_cache(repos: List[Dict]):
    """Save discovered repositories to cache file"""
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "total_repos": len(repos),
        "repositories": repos
    }
    
    with open(SEARCH_CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.info(f"💾 Search results cached to {SEARCH_CACHE_FILE}")
    logger.info(f"   Cached {len(repos)} repositories")


def load_search_cache() -> Optional[List[Dict]]:
    """Load repositories from cache file if it exists"""
    if not SEARCH_CACHE_FILE.exists():
        return None
    
    try:
        with open(SEARCH_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        repos = cache_data.get('repositories', [])
        timestamp = cache_data.get('timestamp', 'unknown')
        
        logger.info(f"📂 Loaded {len(repos)} repositories from cache")
        logger.info(f"   Cache timestamp: {timestamp}")
        
        return repos
    except Exception as e:
        logger.warning(f"⚠️  Failed to load search cache: {e}")
        return None


def discover_repositories(use_cache: bool = True) -> List[Dict]:
    """Discover repositories containing Jupyter notebooks"""
    
    # Try to load from cache first
    if use_cache:
        cached_repos = load_search_cache()
        if cached_repos:
            logger.info("✅ Using cached search results")
            return cached_repos
    
    logger.info("🔍 Discovering repositories with Jupyter Notebooks...")
    
    all_repos = []
    seen_repos = set()
    
    # Multiple search queries to get diverse results
    # Each query can return up to 1000 results (GitHub API limit)
    search_queries = [
        "language:jupyter-notebook stars:>10",
        "language:jupyter-notebook stars:>50",
        "language:jupyter-notebook stars:>100",
        "language:jupyter-notebook stars:>500",
        "language:jupyter-notebook stars:>1000",
        "extension:ipynb stars:>10",
        "extension:ipynb stars:>50",
        "extension:ipynb stars:>100",
        "extension:ipynb stars:>500",
        "machine learning extension:ipynb",
        "deep learning extension:ipynb",
        "data science extension:ipynb",
        "python jupyter notebook",
        "pytorch jupyter",
        "tensorflow jupyter",
        "keras jupyter",
        "scikit-learn jupyter",
        "pandas jupyter",
        "numpy jupyter",
        "data analysis ipynb",
        "data visualization ipynb",
        "neural network jupyter",
        "computer vision jupyter",
        "nlp jupyter notebook",
        "image processing ipynb",
        "time series jupyter",
        "statistics jupyter",
        "research jupyter notebook",
        "tutorial jupyter notebook",
        "course jupyter notebook",
    ]
    
    for query in search_queries:
        logger.info(f"🔎 Searching: {query}")
        query_repos = 0
        
        page = 1
        # Remove the all_repos limit - let each query run to completion
        while True:
            results = search_github_repos(query, page=page, per_page=REPOS_PER_PAGE)
            
            if not results or 'items' not in results:
                break
            
            items = results['items']
            if not items:
                break
            
            for repo in items:
                repo_id = repo['id']
                if repo_id not in seen_repos:
                    seen_repos.add(repo_id)
                    all_repos.append(repo)
                    query_repos += 1
                    stats.update(repos_discovered=1)
            
            logger.info(f"   Page {page}: Found {len(items)} repos ({query_repos} new from this query, {len(all_repos)} total unique)")
            
            # Check if we've reached the last page
            if len(items) < REPOS_PER_PAGE:
                break
            
            page += 1
            
            # GitHub only allows up to 1000 results per search (10 pages of 100)
            if page > 10:
                logger.info(f"   Reached GitHub's 1000 result limit for this query")
                break
        
        logger.info(f"✅ Query complete: +{query_repos} new repos from this query. Total unique: {len(all_repos)}")
        
        # Stop if we have enough repos to potentially reach our target
        # Increased from 5000 to 20000 for better coverage
        if len(all_repos) > 20000:
            logger.info("📊 Sufficient repositories discovered for target dataset size")
            break
    
    logger.info(f"🎯 Total repositories discovered: {len(all_repos)}")
    
    # Save to cache for future runs
    save_search_cache(all_repos)
    
    return all_repos


def save_metadata(repos_processed: List[Dict]):
    """Save collection metadata"""
    metadata = {
        "collection_date": datetime.now().isoformat(),
        "target_tokens": TARGET_TOKENS,
        "repositories": repos_processed,
        "statistics": {
            "repos_discovered": stats.repos_discovered,
            "repos_cloned": stats.repos_cloned,
            "repos_failed": stats.repos_failed,
            "notebooks_collected": stats.notebooks_collected,
            "total_tokens": stats.total_tokens,
            "total_size_bytes": stats.total_size_bytes
        }
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Metadata saved to {METADATA_FILE}")


def main(use_cache: bool = True):
    """Main collection process"""
    logger.info("=" * 80)
    logger.info("JUPYTER NOTEBOOK DATASET COLLECTOR")
    logger.info("=" * 80)
    logger.info(f"Target: {TARGET_TOKENS:,} tokens ({TARGET_TOKENS / 1e9:.0f}B)")
    logger.info(f"Output directory: {NOTEBOOKS_DIR}")
    logger.info(f"Log directory: {LOGS_DIR}")
    logger.info(f"Cache enabled: {use_cache}")
    logger.info("=" * 80)
    
    # Check GitHub API access
    rate_info = check_rate_limit()
    if rate_info:
        logger.info(f"✅ GitHub API authenticated")
        logger.info(f"   Rate limit: {rate_info['resources']['search']['remaining']}/{rate_info['resources']['search']['limit']}")
    else:
        logger.error("❌ Failed to authenticate with GitHub API")
        return
    
    # Discover repositories
    repos = discover_repositories(use_cache=use_cache)
    
    if not repos:
        logger.error("❌ No repositories found!")
        return
    
    logger.info(f"📦 Starting collection from {len(repos)} repositories...")
    
    # Process repositories
    repos_processed = []
    
    # Use single-threaded processing for better log visibility
    for i, repo in enumerate(repos, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PROCESSING REPOSITORY {i}/{len(repos)}")
        logger.info(f"{'=' * 80}")
        
        result = process_repository(repo)
        repos_processed.append(result)
        
        # Save metadata periodically
        if i % 10 == 0:
            save_metadata(repos_processed)
        
        # Check if we've reached target
        if stats.total_tokens >= TARGET_TOKENS:
            logger.info(f"🎉 TARGET REACHED! {stats.total_tokens:,} tokens collected")
            break
    
    # Final save
    save_metadata(repos_processed)
    stats.save()
    
    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 80)
    stats.display()
    logger.info(f"📁 Notebooks saved to: {NOTEBOOKS_DIR}")
    logger.info(f"📊 Metadata saved to: {METADATA_FILE}")
    logger.info(f"📝 Main log: {log_file}")
    logger.info(f"📝 Clone logs: {CLONE_LOG_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Jupyter Notebooks from GitHub repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_notebooks.py              # Use cached search results if available
  python collect_notebooks.py --no-cache   # Force fresh GitHub search
  python collect_notebooks.py --refresh    # Same as --no-cache
        """
    )
    parser.add_argument(
        '--no-cache', 
        dest='use_cache',
        action='store_false',
        help='Bypass search cache and perform fresh GitHub search'
    )
    parser.add_argument(
        '--refresh',
        dest='use_cache', 
        action='store_false',
        help='Alias for --no-cache'
    )
    
    args = parser.parse_args()
    
    try:
        main(use_cache=args.use_cache)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Collection interrupted by user")
        stats.save()
        logger.info("📊 Statistics saved")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        stats.save()


