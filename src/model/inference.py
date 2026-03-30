"""
inference.py
------------
Loads your fine-tuned model and runs motive analysis.
Falls back to a clear error message if the model hasn't been trained yet.
"""

from pathlib import Path
import torch

MODEL_DIR = Path("models/motive-model")

_model     = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}.\n"
            "You need to train it first:\n"
            "  1. python scripts/download_data.py\n"
            "  2. python scripts/scrape_reddit.py\n"
            "  3. python scripts/build_dataset.py\n"
            "  4. python scripts/train.py\n"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading model from {MODEL_DIR}...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        dtype=torch.float16,
        device_map="auto",
    )
    _model = PeftModel.from_pretrained(base, MODEL_DIR)
    _model.eval()
    print("Model loaded.")


def analyze(action: str, history: dict | None = None) -> str:
    """
    Analyze why a person did something.

    Args:
        action:  Description of what the person did.
        history: Optional dict with keys: religion, trauma,
                 relationships, career — any can be omitted.

    Returns:
        A string containing the motive analysis.
    """
    global _model, _tokenizer

    if _model is None:
        _load_model()

    # Build the input prompt
    if history:
        filled = {k: v for k, v in history.items() if v}
        history_text = "\n".join(f"- {k}: {v}" for k, v in filled.items())
        prompt = (
            f"<s>[INST] A person did the following:\n{action}\n\n"
            f"Background information:\n{history_text}\n\n"
            f"Why did this person do this? [/INST]"
        )
    else:
        prompt = (
            f"<s>[INST] A person did the following:\n{action}\n\n"
            f"Why did this person do this? [/INST]"
        )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Strip the input prompt from the output
    generated = output[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()
