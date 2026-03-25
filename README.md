# Cognitive Motive Analyzer

A fine-tuned language model that analyzes why a person did something —
drawing on psychology, philosophy, religion, and real human behaviour data.

Given a description of what someone did, the model explains the most likely
psychological and philosophical drivers. When background is provided
(upbringing, trauma, beliefs, relationships), it pins down the specific
framework most likely at play for that individual.

---

## What this is

Most NLP systems label emotions or classify sentiment. This model does
something different — it *reasons* about human motivation. It is trained
on real accounts of human behaviour (Reddit, biographical data) and shaped
by philosophical and psychological frameworks (Nietzsche, Aristotle,
Maslow, Freud, the Gita, the Quran, the Bible, wartime psychology).

The model has two modes:

- **No history** — gives a multi-lens overview of plausible drivers
- **With history** — picks the single most likely framework given what
  is known about the person, flags contradictions as red flags, and
  states its confidence level

---

## Project structure

```
cognitive-motive-analyzer/
│
├── data/
│   ├── raw/                  ← Kaggle + Reddit dumps (not committed to git)
│   ├── cleaned/              ← formatted training pairs (not committed)
│   └── history.json          ← saved session analyses
│
├── scripts/
│   ├── download_data.py      ← pulls Reddit datasets from Kaggle
│   ├── scrape_reddit.py      ← scrapes live Reddit posts via PRAW
│   ├── build_dataset.py      ← formats raw data into training pairs
│   └── train.py              ← QLoRA fine-tuning on Mistral-7B
│
├── src/
│   ├── model/
│   │   └── inference.py      ← loads model + runs analysis
│   └── memory/
│       └── history.py        ← saves and loads session history
│
├── main.py                   ← run this to use the analyzer
├── requirements.txt
├── .env.example              ← copy to .env and fill in your keys
└── .gitignore
```

---

## First time setup

### 1. Add this project to git and push to GitHub

If you are starting fresh from the zip:

```bash
cd cognitive-motive-analyzer
git init
git remote add origin https://github.com/Suryansh2512/cognitive-motive-analyzer.git
```

Make sure `.gitignore` is in place (it is — it excludes data/, models/, venv/, .env),
then stage everything and push:

```bash
git add .
git commit -m "feat: v2 — new training pipeline, QLoRA fine-tuning setup"
git branch -M main
git push --force origin main
```

The `--force` is needed because this is a fresh folder replacing the old repo.
After this first push, never use `--force` again — just `git push origin main`.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install PyTorch with CUDA

This must be installed separately before anything else.
Your GPU is an RTX 4060 which uses CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify it worked:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This should print `True`. If it prints `False` your CUDA drivers need updating.

### 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up your API keys

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` and fill in your keys. See the API keys section below.

---

## API keys

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| Reddit client ID + secret | reddit.com/prefs/apps → Create App → script | Yes, for scraping |
| Kaggle token | kaggle.com/settings → API → Create New Token | Yes, for datasets |
| Weights & Biases | wandb.ai/settings | Optional — tracks training |
| HuggingFace token | huggingface.co/settings/tokens | Optional — upload model |

**Reddit setup:**
1. Go to reddit.com/prefs/apps
2. Click "Create App" at the bottom
3. Choose type: **script**
4. Name it anything, set redirect URI to `http://localhost`
5. Copy the ID shown under your app name → `REDDIT_CLIENT_ID`
6. Copy the secret → `REDDIT_CLIENT_SECRET`

**Kaggle setup:**
1. Go to kaggle.com/settings
2. Scroll to API → click "Create New Token"
3. A file called `kaggle.json` downloads
4. Place it at `C:\Users\surya\.kaggle\kaggle.json`
5. No env var needed — the kaggle CLI reads it automatically

---

## Training the model

Run these four steps in order. Each one must finish before the next:

```bash
# Step 1 — Download Reddit datasets from Kaggle
python scripts/download_data.py

# Step 2 — Scrape live Reddit posts (requires Reddit keys in .env)
python scripts/scrape_reddit.py

# Step 3 — Clean and format data into training pairs
python scripts/build_dataset.py

# Step 4 — Fine-tune Mistral-7B using QLoRA (takes 2–4 hours)
python scripts/train.py
```

The trained model is saved to `models/motive-model/`.
This folder is in `.gitignore` — model weights are too large for GitHub.

---

## Running the analyzer

Once training is complete:

```bash
python main.py
```

You will be prompted to describe what the person did, then optionally
provide background information. The model will explain the likely drivers.

---

## Everyday git workflow

After the first push, use this whenever you make changes:

```bash
git add .
git commit -m "your message here"
git push origin main
```

Common commit messages for this project:
- `data: add cleaned training pairs`
- `train: adjust LoRA rank and learning rate`
- `fix: handle empty Reddit comments in build_dataset`
- `feat: add history-aware prompt in inference`

---

## Tech stack

| Component | Library |
|-----------|---------|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning method | QLoRA via PEFT |
| Quantization | bitsandbytes (4-bit NF4) |
| Training framework | HuggingFace Transformers + Trainer |
| Data collection | PRAW (Reddit), Kaggle CLI |
| Experiment tracking | Weights & Biases |
| Interface | Gradio (coming) / CLI (now) |
