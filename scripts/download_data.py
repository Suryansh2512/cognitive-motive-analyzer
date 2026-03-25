"""
download_data.py
----------------
Downloads Reddit datasets from Kaggle into data/raw/.

Usage:
    python scripts/download_data.py

Requirements:
    - kaggle.json placed at C:\\Users\\surya\\.kaggle\\kaggle.json
      (download from https://www.kaggle.com/settings -> API -> Create New Token)
    - pip install kaggle
"""

import os
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# These are real Kaggle datasets with Reddit posts from the subreddits we need.
# Format: "kaggle-username/dataset-name"
DATASETS = [
    "parisa1365/reddit-relationship-advice",       # r/relationship_advice posts
    "abhi1nandy2/reddit-aita-posts",               # r/AmITheAsshole posts
    "josephmisiti/reddit-relationships-dataset",   # general relationship posts
]

def download(dataset: str):
    name = dataset.split("/")[-1]
    dest = RAW_DIR / name

    if dest.exists():
        print(f"[skip] {name} already downloaded")
        return

    print(f"[download] {dataset}")
    os.system(f"kaggle datasets download -d {dataset} -p {RAW_DIR} --unzip")
    print(f"[done] {name}")


def main():
    print("Checking kaggle credentials...")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "\nERROR: kaggle.json not found.\n"
            "Go to https://www.kaggle.com/settings\n"
            "Click 'API' -> 'Create New Token'\n"
            f"Place the downloaded file at: {kaggle_json}\n"
        )
        return

    for dataset in DATASETS:
        try:
            download(dataset)
        except Exception as e:
            print(f"[error] {dataset}: {e}")

    print("\nAll done. Files are in data/raw/")


if __name__ == "__main__":
    main()
