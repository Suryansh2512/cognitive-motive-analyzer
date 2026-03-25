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

from src.model.inference import analyze
from src.memory.history import save_case


def get_history() -> dict:
    """Optionally collect background info about the person."""
    print("\nDo you have background info about this person? (press Enter to skip each)")
    history = {}

    fields = {
        "religion":      "Religion / cultural background",
        "trauma":        "Past traumas or significant events",
        "relationships": "Relationship or family context",
        "career":        "Career or financial situation",
    }

    for key, label in fields.items():
        value = input(f"  {label}: ").strip()
        if value:
            history[key] = value

    return history


def main():
    print("=" * 55)
    print("  Cognitive Motive Analyzer")
    print("=" * 55)

    action = input("\nDescribe what the person did:\n> ").strip()
    if not action:
        print("No action entered.")
        return

    history = get_history()

    print("\nAnalyzing...\n")
    result = analyze(action=action, history=history if history else None)

    print("─" * 55)
    print(result)
    print("─" * 55)

    save_case(action, {"history": history, "analysis": result})
    print("\n(Saved to data/history.json)")


if __name__ == "__main__":
    main()
