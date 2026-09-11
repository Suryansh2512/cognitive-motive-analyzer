"""Download permitted religious texts and turn them into training examples."""

import json
import random
from pathlib import Path

import requests


RAW_DIR = Path("data/religious/raw")
OUTPUT_DIR = Path("data/religious")
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"

SOURCES = {
    "bible-kjv": {
        "title": "King James Bible",
        "url": "https://www.gutenberg.org/files/10/10-0.txt",
        "license": "Public domain text; verify local edition rights before redistribution.",
    },
    "bhagavad-gita-arnold": {
        "title": "The Song Celestial (Bhagavad Gita), Edwin Arnold translation",
        "url": "https://www.gutenberg.org/files/2388/2388.txt",
        "license": "Public domain text; verify local edition rights before redistribution.",
    },
    "dhammapada-muller": {
        "title": "The Dhammapada, F. Max Muller translation",
        "url": "https://www.gutenberg.org/files/2017/2017-0.txt",
        "license": "Public domain text; verify local edition rights before redistribution.",
    },
    "tao-te-ching-legge": {
        "title": "The Tao Teh King, James Legge translation",
        "url": "https://www.gutenberg.org/files/216/216.txt",
        "license": "Public domain text; verify local edition rights before redistribution.",
    },
    "quran-sale": {
        "title": "The Koran, George Sale translation",
        "url": "https://www.gutenberg.org/cache/epub/2800/pg2800.txt",
        "license": "Public domain translation; verify local edition rights before redistribution.",
    },
    "plato-republic-jowett": {
        "title": "The Republic, Benjamin Jowett translation",
        "url": "https://www.gutenberg.org/files/1497/1497-0.txt",
        "license": "Public domain translation; verify local edition rights before redistribution.",
    },
    "nietzsche-zarathustra": {
        "title": "Thus Spake Zarathustra, Thomas Common translation",
        "url": "https://www.gutenberg.org/files/1998/1998-0.txt",
        "license": "Public domain translation; verify local edition rights before redistribution.",
    },
}


def download_source(source_id: str, source: dict) -> str:
    path = RAW_DIR / f"{source_id}.txt"
    if not path.exists():
        path.touch()
    for _ in range(8):
        downloaded = path.stat().st_size
        headers = {"User-Agent": "cognitive-motive-analyzer/1.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        try:
            response = requests.get(source["url"], headers=headers, timeout=60, stream=True)
            if response.status_code == 416 and downloaded:
                return path.read_text(encoding="utf-8", errors="replace")
            response.raise_for_status()
            if downloaded and response.status_code == 200:
                path.write_bytes(b"")
                downloaded = 0
            with path.open("ab") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
            return path.read_text(encoding="utf-8", errors="replace")
        except requests.RequestException:
            continue
    raise RuntimeError(f"Could not download {source['title']} after several retries")


def passages(text: str, size: int = 900) -> list[str]:
    paragraphs = [" ".join(part.split()) for part in text.split("\n\n")]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) < 80:
            continue
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = ""
        current = f"{current} {paragraph}".strip()
    if len(current) >= 80:
        chunks.append(current)
    return chunks


def make_examples(source_id: str, source: dict, text: str) -> list[dict]:
    examples = []
    for passage in passages(text):
        examples.append(
            {
                "input": (
                    f"Study this passage from {source['title']}. Preserve its meaning and "
                    "summarize its central philosophical or spiritual teaching in a "
                    "careful, non-dogmatic way:\n\n"
                    f"{passage}"
                ),
                "output": (
                    f"This passage from {source['title']} presents a spiritual or "
                    "philosophical teaching. Explain its meaning using only the ideas "
                    "supported by the passage, and avoid treating a religious belief "
                    "as a psychological diagnosis.\n\n"
                    f"Passage under study:\n{passage}"
                ),
                "source": source_id,
                "license": source["license"],
            }
        )
    return examples


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for source_id, source in SOURCES.items():
        print(f"Downloading or reading {source['title']}...")
        examples.extend(make_examples(source_id, source, download_source(source_id, source)))

    random.Random(42).shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    save_jsonl(examples[:split], TRAIN_FILE)
    save_jsonl(examples[split:], VAL_FILE)
    print(f"Saved {len(examples[:split])} religious training examples to {TRAIN_FILE}")
    print(f"Saved {len(examples[split:])} religious validation examples to {VAL_FILE}")


if __name__ == "__main__":
    main()