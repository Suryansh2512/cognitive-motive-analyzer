"""
train.py
--------
Fine-tunes Mistral-7B on your cleaned dataset using QLoRA.
Runs on an RTX 4060 (8GB VRAM) with 4-bit quantization.

Usage:
    python scripts/train.py

Output:
    models/motive-model/  <-- your fine-tuned model saved here
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"   # base model from HuggingFace
TRAIN_FILE  = Path("data/cleaned/train.jsonl")
VAL_FILE    = Path("data/cleaned/val.jsonl")
OUTPUT_DIR  = Path("models/motive-model")

MAX_LENGTH  = 128    # was 256
BATCH_SIZE  = 4      # keep
GRAD_ACCUM  = 2      # was 4
EPOCHS      = 1      # keep
LR          = 2e-4

# LoRA settings — these control how much of the model we adapt
LORA_R      = 16     # rank — higher = more expressive but more memory
LORA_ALPHA  = 32
LORA_TARGET = ["q_proj", "v_proj"]   # which layers to adapt


# ── Load data ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(example: dict) -> str:
    """
    Wraps each example in Mistral's instruction format.
    The model learns to respond to this structure.
    """
    return (
        f"<s>[INST] {example['input']} [/INST] "
        f"{example['output']} </s>"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {TRAIN_FILE}...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data   = load_jsonl(VAL_FILE)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    # Format prompts
    train_texts = [format_prompt(e) for e in train_data]
    val_texts   = [format_prompt(e) for e in val_data]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset   = Dataset.from_dict({"text": val_texts})

    # ── Load model in 4-bit (fits in 8GB VRAM) ─────────────────────────────────
    print(f"\nLoading {BASE_MODEL} in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Apply LoRA ──────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Tokenize ────────────────────────────────────────────────────────────────
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    print("\nTokenizing dataset...")
    train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized   = val_dataset.map(tokenize,   batched=True, remove_columns=["text"])

    # Labels = input_ids (the model predicts its own tokens)
    train_tokenized = train_tokenized.map(lambda x: {"labels": x["input_ids"]})
    val_tokenized   = val_tokenized.map(lambda x:   {"labels": x["input_ids"]})

    # ── Training args ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,              # save checkpoint every 500 steps
        save_total_limit=3,          # keep only 3 most recent checkpoints
        load_best_model_at_end=True,
        report_to="none",
        run_name="motive-model-v1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. Run main.py to use your model.")


if __name__ == "__main__":
    main()
