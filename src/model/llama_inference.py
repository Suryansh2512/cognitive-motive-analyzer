"""Inference for the Llama 3.1 8B Instruct motive analyzer."""

from pathlib import Path
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR = Path("models/llama-3.1-religious-model")

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer

    load_dotenv()
    print(f"Loading {BASE_MODEL}...")
    token = os.getenv("HF_TOKEN")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    _tokenizer.pad_token = _tokenizer.eos_token

    load_options = {"device_map": "auto", "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_options["torch_dtype"] = torch.float16
    else:
        load_options["torch_dtype"] = torch.float32

    _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=token, **load_options)

    if ADAPTER_DIR.exists():
        from peft import PeftModel

        _model = PeftModel.from_pretrained(_model, ADAPTER_DIR)

    _model.eval()
    print("Model loaded.")


def _build_messages(action: str, history: dict | None) -> list[dict[str, str]]:
    background = ""
    if history:
        filled = {key: value for key, value in history.items() if value}
        background = "\n".join(f"- {key}: {value}" for key, value in filled.items())

    user_message = (
        "A person did the following:\n"
        f"{action}\n\n"
        "Background information:\n"
        f"{background or 'None provided.'}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a forensic behavioral analyst. Construct exactly 3 hypotheses. "
                "For each, give a High, Medium, or Low confidence level and name the "
                "psychological or philosophical framework. State what the evidence does "
                "not tell us. End with a MOST PROBABLE READING. Avoid diagnosing anyone."
            ),
        },
        {"role": "user", "content": user_message},
    ]


def analyze(action: str, history: dict | None = None) -> str:
    global _model, _tokenizer

    if _model is None:
        _load_model()

    prompt = _tokenizer.apply_chat_template(
        _build_messages(action, history),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.inference_mode():
        output = _model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()