"""
download_data.py
----------------
Downloads Reddit datasets into data/raw/ from two sources:
  1. Kaggle — pre-scraped CSV datasets
  2. HuggingFace — larger, more reliable datasets via the datasets library

Usage:
    python scripts/download_data.py

Requirements:
    - kaggle.json at C:\\Users\\surya\\.kaggle\\kaggle.json
    - pip install kaggle datasets
"""

import os
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Verified working Kaggle datasets (checked March 2026)
KAGGLE_DATASETS = [
    "nird96/aita-reddit-posts",
    "thedevastator/unveiling-relationship-dynamics-with-reddit-rela",
    "janldeboer/reddit-relationships",
    "rishitagagrani/relationship-advice-reddit-dataset",
]

# HuggingFace datasets — more reliable, no Kaggle account needed for these
HF_DATASETS = [
    {
        "name": "aita-270k",
        "repo": "OsamaBsher/AITA-Reddit-Dataset",
        "description": "270,000 AITA posts with top 2 comments (2013-2023)",
    },
]


def download_kaggle(dataset: str):
    name = dataset.split("/")[-1]
    dest = RAW_DIR / name

    if dest.exists():
        print(f"  [skip] {name} - already downloaded")
        return

    print(f"  [downloading] {dataset}")
    exit_code = os.system(
        f"kaggle datasets download -d {dataset} -p {RAW_DIR} --unzip"
    )

    if exit_code != 0:
        print(f"  [failed] {dataset} - skipping")
    else:
        print(f"  [done] {name}")


def download_huggingface(entry: dict):
    name = entry["name"]
    dest = RAW_DIR / f"{name}.jsonl"

    if dest.exists():
        print(f"  [skip] {name} - already downloaded")
        return

    print(f"  [downloading] {entry['repo']}")
    print(f"  {entry['description']}")

    try:
        from datasets import load_dataset

        ds = load_dataset(entry["repo"], split="train")

        with open(dest, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  [done] {len(ds):,} rows saved to {dest}")

    except Exception as e:
        print(f"  [failed] {entry['repo']}: {e}")


def main():
    print("Checking kaggle credentials...")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "\nERROR: kaggle.json not found.\n"
            "Go to https://www.kaggle.com/settings\n"
            "Click API -> Create New Token\n"
            f"Place the file at: {kaggle_json}\n"
        )
        return

    print("\n── Kaggle datasets ───────────────────────────────────")
    for dataset in KAGGLE_DATASETS:
        try:
            download_kaggle(dataset)
        except Exception as e:
            print(f"  [error] {dataset}: {e}")

    print("\n── HuggingFace datasets ──────────────────────────────")
    for entry in HF_DATASETS:
        try:
            download_huggingface(entry)
        except Exception as e:
            print(f"  [error] {entry['repo']}: {e}")

    print("\nAll done. Files are in data/raw/")
    print("Next step: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
