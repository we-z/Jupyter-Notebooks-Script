# Jupyter Notebook Dataset Collector

A Python script to collect a large-scale dataset of Jupyter Notebooks from GitHub for machine learning and data science research.

## Features

- 🔍 **Automated Discovery**: Searches GitHub using multiple queries to find diverse Jupyter Notebook repositories
- 📥 **Live Clone Logs**: Shows real-time progress as each repository is being cloned
- 📊 **Token Estimation**: Estimates token counts for collected notebooks (targeting 100B+ tokens)
- 📈 **Real-time Statistics**: Displays live progress including repos cloned, notebooks collected, and token counts
- 💾 **Comprehensive Logging**: Maintains detailed logs for main process and individual repository clones
- 🔄 **Resume Support**: Skips already-cloned repositories to support interrupted collections
- ⚡ **Rate Limit Handling**: Automatically manages GitHub API rate limits

## Target

**100 Billion+ Tokens** of Jupyter Notebook content from diverse repositories

## Setup

### Prerequisites

- Python 3.7+
- Git
- GitHub Personal Access Token with `repo` and `public_repo` scopes

### Installation

1. Clone this repository:
```bash
git clone <this-repo-url>
cd Jupyter-Notebooks-Script
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The GitHub API token is already configured in the script

## Usage

### Run the collector:

```bash
python collect_notebooks.py
```

The script will:
1. Authenticate with GitHub API
2. Search for repositories containing Jupyter Notebooks
3. Clone repositories one by one with live progress
4. Count notebooks and estimate tokens
5. Display real-time statistics
6. Save metadata and statistics

### Output Structure

```
Jupyter-Notebooks-Script/
├── collect_notebooks.py          # Main script
├── requirements.txt               # Dependencies
├── collection_metadata.json      # Detailed collection metadata
├── collection_stats.json         # Real-time statistics
├── notebooks_dataset/            # Cloned repositories
│   ├── user1/repo1/
│   ├── user2/repo2/
│   └── ...
└── logs/
    ├── collection_YYYYMMDD_HHMMSS.log  # Main log
    └── clone_logs/                      # Individual repo clone logs
        ├── user1_repo1.log
        ├── user2_repo2.log
        └── ...
```

## Logging

### Main Log
- Located in `logs/collection_YYYYMMDD_HHMMSS.log`
- Contains overall progress, statistics, and errors
- Also outputs to console in real-time

### Clone Logs
- Located in `logs/clone_logs/`
- One log file per repository
- Contains detailed git clone output

### Statistics File
- `collection_stats.json` - Updated every 30 seconds
- Contains current progress metrics

### Metadata File
- `collection_metadata.json` - Complete collection metadata
- Saved periodically and at completion
- Includes repository details and final statistics

## Real-time Statistics

The script displays live statistics including:
- Repositories discovered
- Repositories successfully cloned
- Repositories failed
- Total notebooks collected
- Total tokens (with progress toward 100B target)
- Total dataset size in GB
- Elapsed time
- Average notebooks per repository
- Average tokens per notebook

## Features in Detail

### Token Estimation
Uses a conservative estimate of ~4 characters per token for code content. Includes:
- Code cells
- Markdown cells
- Raw cells
- Cell outputs

### Search Strategy
Uses multiple targeted queries to find diverse notebooks:
- General Jupyter Notebook repositories by star count
- Machine learning focused
- Deep learning focused
- Data science focused
- Framework-specific (PyTorch, TensorFlow)

### Error Handling
- Automatic retry on rate limit
- Skips corrupted notebooks
- Continues on clone failures
- Saves progress on interruption (Ctrl+C)

## Monitoring Progress

Watch the console output for:
- 🔍 Discovery phase
- 🚀 Clone start notifications
- 📥 Real-time clone progress
- ✅ Successful clones
- ❌ Failed clones
- 📊 Notebook counts per repo
- ✨ Token counts per repo
- 📈 Overall statistics updates

## Interrupting and Resuming

- Press `Ctrl+C` to safely interrupt
- Statistics and metadata are saved automatically
- Re-run the script to resume (already-cloned repos are skipped)

## Rate Limits

The script respects GitHub API rate limits:
- Monitors remaining API calls
- Automatically waits when limits are close
- Keeps 10 calls in reserve as buffer

## Estimated Collection Time

For 100B tokens:
- Depends on repository sizes and network speed
- Estimated: Several hours to days
- Can be interrupted and resumed safely

## License

MIT

## Notes

- The script uses `--depth 1` for faster cloning (no git history)
- Skips `.ipynb_checkpoints` folders
- Authenticates with provided GitHub token
- Suitable for research and dataset creation purposes
