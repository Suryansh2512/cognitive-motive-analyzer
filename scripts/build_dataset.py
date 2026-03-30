"""
build_dataset.py
----------------
Turns raw Reddit data into (input -> output) training pairs.
Saves to data/cleaned/train.jsonl and data/cleaned/val.jsonl

Usage:
    python scripts/build_dataset.py
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

VAL_SPLIT       = 0.1
MIN_COMMENT_LEN = 60


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\*\*|\*|~~|__", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_verdict(text: str) -> str:
    """
    Remove leading verdict labels like 'nta', 'yta', 'esh', 'nah'
    and keep the explanation that follows.
    e.g. 'nta. you did nothing wrong...' -> 'you did nothing wrong...'
    """
    pattern = r"^(nta|yta|esh|nah|info|nt a|y t a)[\.\,\s]+"
    cleaned = re.sub(pattern, "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def is_useful(text: str) -> bool:
    if not text or len(text) < MIN_COMMENT_LEN:
        return False
    t = text.lower()
    # Only skip truly useless comments — NOT nta/yta since we strip those
    junk = ["[removed]", "[deleted]", "lol", "lmao", "this.", "same."]
    return not any(j in t for j in junk)


def load_aita_jsonl(path: Path) -> list[dict]:
    """
    Columns: id, title, text, verdict, comment1, comment2, score
    input  = title + text (what the person did)
    output = comment1 or comment2 with verdict prefix stripped
    """
    pairs = []
    print(f"Loading {path.name}...")

    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=path.name):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            title = clean_text(str(row.get("title", "")))
            text  = clean_text(str(row.get("text",  "")))

            if text.lower() in ("", "nan", "[removed]", "[deleted]"):
                body = title
            else:
                body = f"{title}\n\n{text}"

            if len(body) < 50:
                continue

            # Try comment1 first, fall back to comment2
            for key in ["comment1", "comment2"]:
                raw_comment = clean_text(str(row.get(key, "")))
                comment = strip_verdict(raw_comment)
                if is_useful(comment):
                    pairs.append({
                        "input":  f"A person did the following:\n{body}",
                        "output": comment,
                        "source": "aita-270k",
                    })
                    break

    print(f"  -> {len(pairs):,} usable pairs")
    return pairs


def main():
    print("Building training dataset...\n")
    pairs = []

    aita_jsonl = RAW_DIR / "aita-270k.jsonl"
    if aita_jsonl.exists():
        pairs.extend(load_aita_jsonl(aita_jsonl))
    else:
        print("aita-270k.jsonl not found — run download_data.py first")
        return

    if not pairs:
        print("No usable pairs found.")
        return

    # Shuffle and split
    pairs = pairs[:5000]
    random.seed(42)
    random.shuffle(pairs)

    split_idx   = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_pairs, TRAIN_FILE)
    save_jsonl(val_pairs,   VAL_FILE)

    print(f"\nDataset built successfully:")
    print(f"  Train : {len(train_pairs):,} examples  ->  {TRAIN_FILE}")
    print(f"  Val   : {len(val_pairs):,} examples   ->  {VAL_FILE}")
    print(f"\nNext step: python scripts/train.py")


if __name__ == "__main__":
    main()
