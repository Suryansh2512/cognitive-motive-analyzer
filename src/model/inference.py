"""
inference.py
------------
Loads your fine-tuned model and runs motive analysis.
Falls back to a clear error message if the model hasn't been trained yet.
"""

from pathlib import Path
import torch
from transformers import BitsAndBytesConfig


MODEL_DIR = Path("models/motive-model")

_model     = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}.\n"
            "Train it first: python scripts/train.py"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import gc

    # Free any existing memory first
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading model from {MODEL_DIR}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,      # ← add this
        trust_remote_code=True,
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
            "<s>[INST] You are a forensic behavioral analyst. You do not converse. "
            "You investigate. Construct exactly 3 hypotheses, label each with a confidence "
            "level (High/Medium/Low), name the psychological or philosophical framework "
            "you are drawing from, state what the evidence does not tell you, and end with "
            "a MOST PROBABLE READING that takes a clear position.\n\n"
            f"A person did the following:\n{action}\n\n"
            f"Background information:\n{history_text}\n\n"
            "Analyze why this person did this. [/INST]"
        )
    else:
        prompt = (
            "<s>[INST] You are a forensic behavioral analyst. You do not converse. "
            "You investigate. Construct exactly 3 hypotheses, label each with a confidence "
            "level (High/Medium/Low), name the psychological or philosophical framework "
            "you are drawing from, state what the evidence does not tell you, and end with "
            "a MOST PROBABLE READING that takes a clear position.\n\n"
            f"A person did the following:\n{action}\n\n"
            "Analyze why this person did this. [/INST]"
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
