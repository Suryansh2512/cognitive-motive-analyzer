"""
main.py
-------
Entry point for the Cognitive Motive Analyzer.

Usage:
    python main.py

Before running, make sure you have trained the model:
    python scripts/download_data.py
    python scripts/scrape_reddit.py
    python scripts/build_dataset.py
    python scripts/train.py
"""

from src.memory.history import save_case
from src.model.inference import analyze

WIDTH = 72


def print_header() -> None:
    print()
    print("=" * WIDTH)
    print("COGNITIVE MOTIVE ANALYZER".center(WIDTH))
    print("Behavioral reasoning workspace | Llama 3.1 8B Instruct".center(WIDTH))
    print("=" * WIDTH)


def print_section(title: str) -> None:
    print()
    print(title.upper())
    print("-" * len(title))


def print_report(action: str, history: dict, result: str) -> None:
    print_header()
    print_section("Case summary")
    print(f"Observed behavior: {action}")

    print_section("Context")
    if history:
        for key, value in history.items():
            print(f"{key.title():<18} {value}")
    else:
        print("No additional context provided.")

    print_section("Analysis")
    print(result.strip())
    print()
    print("=" * WIDTH)
    print("Assessment complete | Three competing hypotheses generated")
    print("=" * WIDTH)


def get_history() -> dict:
    print_section("Optional context")
    print("Press Enter to skip any field.")
    fields = {
        "religion": "Religion / cultural background",
        "trauma": "Past traumas or significant events",
        "relationships": "Relationship or family context",
        "career": "Career or financial situation",
    }
    history = {}
    for key, label in fields.items():
        value = input(f"  {label}: ").strip()
        if value:
            history[key] = value
    return history


def main():
    print_header()

    print_section("New case")
    action = input("Describe the observed behavior:\n> ").strip()
    if not action:
        print("No action entered.")
        return

    history = get_history()
    print("\nAnalyzing case. Please wait...\n")
    result = analyze(action, history or None)

    print_report(action, history, result)
    save_case(action, {"history": history, "analysis": result, "model": "meta-llama/Llama-3.1-8B-Instruct"})
    print("Case saved to data/history.json")


if __name__ == "__main__":
    main()
