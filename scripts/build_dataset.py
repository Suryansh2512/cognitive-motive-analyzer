"""
build_dataset.py
----------------
Turns raw Reddit posts into (input -> output) training pairs
and saves them to data/cleaned/train.jsonl and data/cleaned/val.jsonl

Usage:
    python scripts/build_dataset.py

Output format (each line in the jsonl):
{
  "input":  "A person did the following: <post title + body>",
  "output": "<top comment explaining the behaviour>"
}
"""

import json
import re
import random
from pathlib import Path
from tqdm import tqdm

RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = CLEANED_DIR / "train.jsonl"
VAL_FILE   = CLEANED_DIR / "val.jsonl"

VAL_SPLIT  = 0.1   # 10% goes to validation
MIN_CHARS  = 80    # minimum output length to keep


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"\*\*|\*|~~|__", "", text)     # remove markdown
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces
    return text.strip()


def format_input(record: dict) -> str:
    title = clean_text(record.get("title", ""))
    body  = clean_text(record.get("body", ""))

    if body:
        return f"A person did the following:\n{title}\n\n{body}"
    return f"A person did the following:\n{title}"


def format_output(comment: str) -> str:
    return clean_text(comment)


def is_useful_comment(comment: str) -> bool:
    c = comment.lower()

    # Filter out low-quality comments
    junk_phrases = [
        "nta", "yta", "esh", "nah",          # AITA verdicts without explanation
        "deleted", "removed",
        "lol", "lmao", "haha",
        "this.", "same.",
        "edit:",
    ]
    if any(phrase in c for phrase in junk_phrases):
        return False

    if len(comment) < MIN_CHARS:
        return False

    return True


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_pairs() -> list[dict]:
    pairs = []

    # Load scraped Reddit data
    scrape_file = RAW_DIR / "reddit_scrape.jsonl"
    if scrape_file.exists():
        records = load_jsonl(scrape_file)
        print(f"Loaded {len(records)} scraped Reddit posts")

        for record in tqdm(records, desc="Processing Reddit scrape"):
            input_text = format_input(record)
            for comment in record.get("comments", []):
                if is_useful_comment(comment):
                    pairs.append({
                        "input":  input_text,
                        "output": format_output(comment),
                        "source": record.get("source", "reddit"),
                    })
                    break  # one output per post is enough for now

    # Load any Kaggle csv files in data/raw/
    import pandas as pd
    for csv_file in RAW_DIR.glob("**/*.csv"):
        print(f"Found Kaggle file: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file, nrows=5000)
            # Try common column name patterns
            text_cols  = [c for c in df.columns if any(k in c.lower() for k in ["body", "text", "post", "content", "selftext"])]
            label_cols = [c for c in df.columns if any(k in c.lower() for k in ["comment", "response", "reply", "label"])]

            if text_cols and label_cols:
                for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_file.name):
                    body    = str(row[text_cols[0]])
                    comment = str(row[label_cols[0]])
                    if is_useful_comment(comment) and len(body) > 50:
                        pairs.append({
                            "input":  f"A person did the following:\n{clean_text(body)}",
                            "output": format_output(comment),
                            "source": csv_file.stem,
                        })
            else:
                print(f"  [skip] Could not find usable columns in {csv_file.name}")
                print(f"  Available columns: {list(df.columns)}")

        except Exception as e:
            print(f"  [error] {csv_file.name}: {e}")

    return pairs


def main():
    print("Building training dataset...\n")
    pairs = build_pairs()

    if not pairs:
        print("\nNo data found. Run download_data.py and scrape_reddit.py first.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)

    split_idx   = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    # Save
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_pairs, TRAIN_FILE)
    save_jsonl(val_pairs,   VAL_FILE)

    print(f"\nDataset built:")
    print(f"  Train: {len(train_pairs)} examples -> {TRAIN_FILE}")
    print(f"  Val:   {len(val_pairs)} examples  -> {VAL_FILE}")
    print(f"\nNext step: python scripts/train.py")


if __name__ == "__main__":
    main()
