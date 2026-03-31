# 🧠 Cognitive Motive Analyzer

> *"The detective does not guess. He reasons. Every broken object, every open door, every missing word is a sentence in a confession."*

A forensic behavioral AI that reasons about **why** people do things — not just what they did.  
Fine-tuned on Mistral-7B-Instruct using QLoRA, drawing from psychology, philosophy, criminology, and religious texts.

---

## What This Is

Most NLP systems label emotions or classify sentiment. This is different.

Given a description of a behavior — or a crime scene — the model **investigates**. It constructs competing hypotheses, ranks them by likelihood, draws from multiple frameworks (Freudian, existentialist, Jungian, Dharmic, criminological), and arrives at a structured reading of probable mental state and motive.

It does not converse. It does not guess. It reasons.

### Two Modes

| Mode | Input | Output |
|------|-------|--------|
| **Behavioral Analysis** | What someone did + optional background | Ranked hypotheses with framework citations |
| **Crime Scene Analysis** *(in development)* | Scene description, victim findings, physical evidence | Reconstructed perpetrator mental state at time of act |

### Example Output (target behavior post-training)

```
Input: A man left flowers at his estranged father's grave every year
       but never attended the funeral.

HYPOTHESIS 1 [High confidence] — Unresolved attachment with avoidance
  The subject maintains ritual connection (flowers) while refusing
  communal grief (funeral). Classic ambivalent attachment. The gesture
  is private — not performed for others — suggesting the relationship
  was real to him, but unbearable to display.

HYPOTHESIS 2 [Medium confidence] — Guilt without resolution
  Psychoanalytically: the flowers are reparative, a symbolic attempt
  to repair an object relation that was never closed. The absence at
  the funeral may represent punishment of the self — he does not
  deserve to mourn publicly.

HYPOTHESIS 3 [Low confidence] — Cultural/religious obligation
  In some frameworks (Hindu, Confucian), honoring the dead is duty
  regardless of relationship quality. The act may be ritual rather
  than emotional. Requires more context.

MOST PROBABLE: The subject loved his father and never said it.
The flowers are what he could not say in life.
```

---

## The Crime Scene Vision

The system's long-term goal is forensic psychological reconstruction.

When presented with a crime scene — physical evidence, victim state, spatial layout — the model should not just pattern-match. It should ask: *what was the perpetrator thinking?*

A broken flower pot is not necessarily a weapon. It may have tipped over during a panicked exit. A door left open is not necessarily evidence of haste — it may be staging. The model is designed to hold **multiple causal chains simultaneously** and rank them, rather than collapsing to the first plausible story.

This is the difference between correlation-based inference and **causal mental-state modeling**.

---

## Project Structure

```
cognitive-motive-analyzer/
│
├── data/
│   ├── raw/                  ← Kaggle + Reddit dumps (gitignored)
│   ├── cleaned/              ← formatted training pairs (gitignored)
│   └── history.json          ← saved session analyses
│
├── scripts/
│   ├── download_data.py      ← pulls behavioral datasets from Kaggle/HF
│   ├── scrape_reddit.py      ← scrapes live Reddit posts via PRAW
│   ├── build_dataset.py      ← formats raw data into training pairs
│   └── train.py              ← QLoRA fine-tuning on Mistral-7B
│
├── src/
│   ├── model/
│   │   └── inference.py      ← loads model, runs forensic analysis
│   └── memory/
│       └── history.py        ← saves and loads session history
│
├── models/
│   └── motive-model/         ← trained LoRA adapter (gitignored)
│
├── main.py                   ← CLI entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### Prerequisites

- Python **3.11** (not 3.12 or 3.13 — PyTorch wheels don't exist for those yet)
- NVIDIA GPU with CUDA 12.1+ (tested on RTX 4060)
- ~15GB free disk space for model weights

### 1. Clone and enter the project

```bash
git clone https://github.com/Suryansh2512/cognitive-motive-analyzer.git
cd cognitive-motive-analyzer
```

### 2. Create a virtual environment with Python 3.11

```bash
py -3.11 -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install PyTorch with CUDA (must be first)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True  /  NVIDIA GeForce RTX 4060
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Fix bitsandbytes for Windows CUDA

```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes --prefer-binary --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

### 6. Configure environment

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

Fill in `.env`:

```env
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=CognitiveMotive/1.0
HF_TOKEN=hf_yourtoken
KAGGLE_USERNAME=your_username
WANDB_API_KEY=your_key        # optional, for training tracking
```

---

## API Keys

| Key | Source | Required for |
|-----|--------|-------------|
| Reddit Client ID + Secret | reddit.com/prefs/apps → Create App → script | Scraping behavioral data |
| Kaggle token | kaggle.com/settings → API → Create Token → place at `~/.kaggle/kaggle.json` | Dataset download |
| HuggingFace token | huggingface.co/settings/tokens | Model download / upload |
| Weights & Biases | wandb.ai/settings | Training visualization (optional) |

---

## Training the Model

Run the pipeline in order:

```bash
# 1. Download behavioral datasets (Reddit AITA, etc.)
python scripts/download_data.py

# 2. Scrape live Reddit posts via PRAW
python scripts/scrape_reddit.py

# 3. Clean and format into training pairs
python scripts/build_dataset.py

# 4. Fine-tune Mistral-7B with QLoRA (~2–4 hours on RTX 4060)
python scripts/train.py
```

Trained adapter is saved to `models/motive-model/`.

### Training Data Philosophy

The model is not trained on raw text. It is trained on **structured reasoning examples** — each one models the *process* of hypothesis construction, not just the answer. Training sources include:

- Reddit AITA + behavioral threads (real human moral reasoning)
- Project Gutenberg texts: Bhagavad Gita, Bible, Quran, Meditations (Marcus Aurelius), Crime and Punishment
- Criminal Minds scripts (forensic reasoning patterns)
- Synthesized crime scene analysis examples

The goal is a model that reasons like a detective and cites like a scholar.

---

## Running the Analyzer

```bash
python main.py
```

You will be prompted to describe a behavior or scene. Background context is optional but improves specificity.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| v1 — Keyword matching | ✅ Complete | Baseline framework classification |
| v2 — QLoRA fine-tuning | 🔄 In progress | Full training pipeline on Mistral-7B |
| v3 — Crime scene mode | 📋 Planned | Forensic scene → perpetrator mental state |
| v4 — Multi-source training | 📋 Planned | Religious texts, criminology literature |
| v5 — Web interface | 📋 Planned | Gradio or FastAPI frontend |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning | QLoRA via PEFT |
| Quantization | bitsandbytes 4-bit NF4 |
| Training framework | HuggingFace Transformers |
| Experiment tracking | Weights & Biases |
| Data collection | PRAW, Kaggle CLI, HuggingFace Datasets |
| Interface | CLI (now) → Gradio (planned) |

---

## Academic Context

This project sits at the intersection of:

- **Computational psychology** — modeling human motivation computationally
- **Forensic AI** — applying language models to behavioral reconstruction
- **Explainable AI** — producing structured, citable reasoning rather than black-box outputs
- **Cross-cultural NLP** — training on philosophical and religious frameworks across traditions

Independent research project. BTech student, 4th semester.

---

## Git Workflow

```bash
git add .
git commit -m "your message"
git push origin main
```

Suggested commit prefixes:
- `data:` — dataset changes
- `train:` — training script or config changes
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — README or documentation

---

*Built independently. Ongoing.*
