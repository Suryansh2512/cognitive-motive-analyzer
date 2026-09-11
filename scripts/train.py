"""Fine-tune Llama 3.1 8B Instruct with the project's cleaned JSONL data."""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TRAIN_FILE = Path("data/cleaned/train.jsonl")
VAL_FILE = Path("data/cleaned/val.jsonl")
RELIGIOUS_TRAIN_FILE = Path("data/religious/train.jsonl")
RELIGIOUS_VAL_FILE = Path("data/religious/val.jsonl")
OUTPUT_DIR = Path("models/llama-3.1-motive-model")
MAX_LENGTH = 256


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_prompt(example: dict, tokenizer) -> str:
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-val-samples", type=int, default=100)
    parser.add_argument(
        "--include-religious",
        action="store_true",
        help="Include all available religious and philosophical records in the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the trained adapter and checkpoints.",
    )
    parser.add_argument(
        "--start-adapter",
        type=Path,
        help="Load an existing adapter before starting a new training phase.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint in the output directory.",
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    load_dotenv()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required for Llama 3.1 8B QLoRA training. "
            "Install a CUDA-enabled PyTorch build and run on an NVIDIA GPU."
        )

    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)
    if args.include_religious:
        train_records.extend(load_jsonl(RELIGIOUS_TRAIN_FILE))
        val_records.extend(load_jsonl(RELIGIOUS_VAL_FILE))
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples:
        val_records = val_records[: args.max_val_samples]
    train_texts = [format_prompt(item, tokenizer) for item in train_records]
    val_texts = [format_prompt(item, tokenizer) for item in val_records]
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

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
        token=token,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    if args.resume:
        checkpoints = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if not checkpoints:
            raise SystemExit("--resume requested, but no checkpoint was found.")
        resume_checkpoint = str(checkpoints[-1])
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
        print(f"Loaded adapter weights from {resume_checkpoint}; using a fresh optimizer.")
    elif args.start_adapter:
        model = PeftModel.from_pretrained(model, args.start_adapter, is_trainable=True)
        print(f"Loaded starting adapter from {args.start_adapter}.")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.print_trainable_parameters()

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    train_tokenized = train_tokenized.map(lambda batch: {"labels": batch["input_ids"]})
    val_tokenized = val_tokenized.map(lambda batch: {"labels": batch["input_ids"]})

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=1,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        ),
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Llama 3.1 adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
