# Phase 1 — Foundations (Notebooks 00–02) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project scaffolding and the first three teaching notebooks: environment setup (00), a runnable word-level Bag-of-Words next-word baseline (01), and dense embeddings that measurably beat it (02).

**Architecture:** Self-contained Jupyter notebooks authored as jupytext "percent" `.py` files (the diff-friendly source of truth) and rendered to `.ipynb`. Each notebook runs top-to-bottom as a plain Python script and contains embedded `assert` sanity-checks that act as its tests. The pedagogical thread: BoW (sparse, order-blind) → embeddings (dense, fewer params, lower perplexity, still order-blind) — motivating attention in Phase 2.

**Tech Stack:** Python 3.11+, PyTorch (MPS/CPU), NumPy, Matplotlib, tqdm, JupyterLab, jupytext. No HuggingFace / high-level modeling libraries.

## Global Constraints

- Python 3.11+; PyTorch with MPS backend, CPU fallback. (verbatim from spec: "PyTorch with MPS backend, CPU fallback")
- No HuggingFace `transformers`, `accelerate`, or similar high-level libraries. `tiktoken` only allowed later (Phase 3, BPE notebook) — not in Phase 1.
- Notebooks are **largely self-contained** — code lives in cells; do not extract a shared package. The one allowed cross-notebook artifact is the committed metrics file `assets/phase1_metrics.json`.
- Notebook source of truth = jupytext percent `.py` in `notebooks/`. Each notebook MUST run clean as a script and MUST render to `.ipynb`.
- Notebooks must run fast on a Mac (target < 60s per notebook as a script) — use small subsets / modest training so the teaching point lands without long waits.
- Matplotlib: every plot is saved with `plt.savefig(...)` to `assets/` AND followed by `plt.show()`. Script-mode tests run with `MPLBACKEND=Agg` so `show()` is a harmless no-op.
- `data/` and `checkpoints/` are gitignored; `assets/` is tracked.
- **Explanation standard (every notebook, 00–10):** markdown must teach, not just name concepts. Plain beginner-friendly language; define jargon inline on first use; split dense code cells so each function/class is explained immediately before it; walk through every function/class step by step (parameters → what each step does → output) with a tiny concrete worked example. Depth target = "explain functions and classes step by step" (not exhaustive line-by-line). See memory `notebook-explanation-depth`.
- **Training-loop convention:** extract scalar losses with `loss.item()` (not `float(loss)`, which warns on grad-bearing tensors). Comment any deviation from a plan-specified hyperparameter inline in the code.

## Plan Conventions (read once)

- **Authoring a notebook:** write `notebooks/NN_name.py` using jupytext percent format. Markdown cells start with `# %% [markdown]` and use `#`-prefixed lines; code cells start with `# %%`.
- **Test cycle for every notebook task:**
  1. Run as script: `MPLBACKEND=Agg python notebooks/NN_name.py` → must exit 0 (all asserts hold).
  2. Render: `jupytext --to notebook --execute notebooks/NN_name.py -o notebooks/NN_name.ipynb` → must succeed.
- **TDD adaptation:** the "failing test" is the embedded `assert`. Write the assert/expectation first in the cell, run the script to see it fail (e.g. NameError / AssertionError), implement the cell above it, re-run to green.
- All commands assume the repo root `/Users/carlosbarbosa/Documents/GitHub/train-llm` and an activated venv (created in Task 1).

---

## File Structure

- Create: `requirements.txt` — pinned-ish dependency list.
- Create: `.gitignore` — ignore venv, data, checkpoints, caches.
- Create: `README.md` — overview, setup, how to run, phase status.
- Create: `data/.gitkeep`, `checkpoints/.gitkeep`, `assets/.gitkeep` — keep dirs.
- Create: `notebooks/00_setup_and_tour.py` (+ generated `.ipynb`) — env + device + project map.
- Create: `notebooks/01_data_and_bag_of_words.py` (+ `.ipynb`) — data, word tokenizer, BoW baseline.
- Create: `notebooks/02_embeddings.py` (+ `.ipynb`) — embedding model beats BoW; switch to char-level + batching.
- Create (by notebook 01): `assets/phase1_metrics.json` — committed metrics interface consumed by notebook 02.

---

### Task 1: Project scaffolding & environment

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `data/.gitkeep`, `checkpoints/.gitkeep`, `assets/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: an activated venv with `torch`, `numpy`, `matplotlib`, `tqdm`, `jupyterlab`, `jupytext` importable; a `device` selection convention reused by all notebooks: prefer `mps`, else `cpu`.

- [ ] **Step 1: Create `requirements.txt`**

```
torch>=2.2
numpy>=1.26
matplotlib>=3.8
tqdm>=4.66
jupyterlab>=4.0
jupytext>=1.16
```

- [ ] **Step 2: Create `.gitignore`**

```
# Python
__pycache__/
*.pyc
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Project data & artifacts
data/*
!data/.gitkeep
checkpoints/*
!checkpoints/.gitkeep
```

- [ ] **Step 3: Create directory keepers**

Create empty files `data/.gitkeep`, `checkpoints/.gitkeep`, `assets/.gitkeep`.

- [ ] **Step 4: Create `README.md`**

```markdown
# Build an LLM from Scratch

A progressive, notebook-driven course that builds a modern decoder-only
language model from raw text up to a Mixture-of-Experts variant — entirely
from scratch in PyTorch (no HuggingFace), sized to train on a Mac in minutes.

Each new method earns its place by beating a **measured** number from the
step before: Bag-of-Words → embeddings → attention → modern components → MoE.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optionally use [`uv`](https://github.com/astral-sh/uv): `uv venv && uv pip install -r requirements.txt` — faster.)

## How to run

Notebooks live in `notebooks/` as paired files: a jupytext `.py` source and a
generated `.ipynb`. Open the `.ipynb` in JupyterLab:

```bash
jupyter lab
```

Or run a notebook headlessly:

```bash
MPLBACKEND=Agg python notebooks/01_data_and_bag_of_words.py
```

## Status

- [x] Phase 1 — Foundations (notebooks 00–02)
- [ ] Phase 2 — The working transformer (notebooks 03–07)
- [ ] Phase 3 — Advanced: tuning, BPE, MoE (notebooks 08–10)

See `docs/superpowers/specs/2026-06-20-llm-from-scratch-design.md` for the full design.
```

- [ ] **Step 5: Create venv and install deps**

Run:
```bash
cd /Users/carlosbarbosa/Documents/GitHub/train-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Expected: installs complete without error.

- [ ] **Step 6: Verify environment (the test)**

Run:
```bash
source .venv/bin/activate
python -c "import torch, numpy, matplotlib, tqdm, jupytext; print('torch', torch.__version__); print('mps', torch.backends.mps.is_available())"
```
Expected: prints a torch version and `mps True` (or `mps False` on non-Apple-Silicon — both acceptable, CPU fallback is fine).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore README.md data/.gitkeep checkpoints/.gitkeep assets/.gitkeep
git commit -m "chore: project scaffolding for LLM-from-scratch (Phase 1)"
```

---

### Task 2: Notebook 00 — Setup & tour

**Files:**
- Create: `notebooks/00_setup_and_tour.py`
- Create (generated): `notebooks/00_setup_and_tour.ipynb`

**Interfaces:**
- Consumes: venv from Task 1.
- Produces: demonstrates the `pick_device()` convention (returns `torch.device`) that later notebooks reuse by copy (self-contained).

- [ ] **Step 1: Write the notebook source**

Create `notebooks/00_setup_and_tour.py`:

```python
# %% [markdown]
# # 00 — Setup & Tour
#
# Welcome. This course builds a modern language model from scratch in PyTorch,
# small enough to train on a Mac. This first notebook just checks your
# environment and maps out where we're going.

# %%
import sys
import platform
import torch

print("Python :", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch :", torch.__version__)

# %% [markdown]
# ## Picking a device
#
# We prefer Apple's **MPS** GPU backend; otherwise we fall back to CPU. Every
# later notebook reuses this exact helper.

# %%
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

# %% [markdown]
# Quick sanity op on the chosen device:

# %%
x = torch.randn(3, 3, device=device)
y = x @ x
assert y.shape == (3, 3)
print("Matmul on", device, "ok. Sum =", float(y.sum()))

# %% [markdown]
# ## The journey
#
# | # | Notebook | Idea |
# |---|----------|------|
# | 00 | Setup & tour | you are here |
# | 01 | Data & Bag-of-Words | a runnable baseline that ignores word order |
# | 02 | Embeddings | dense vectors beat BoW with fewer parameters |
# | 03 | Attention | finally, order-aware |
# | 04 | Modern components | RoPE, RMSNorm, SwiGLU |
# | 05 | Assembling the model | the full decoder-only transformer |
# | 06 | Training loop | actually train it |
# | 07 | Evaluation & generation | sample Shakespeare; KV-cache |
# | 08 | Tuning | learning-rate schedules, sweeps |
# | 09 | BPE tokenizer | subword tokenization from scratch |
# | 10 | Mixture-of-Experts | the capstone |
#
# Each step beats a *measured* number from the one before. On to notebook 01.

# %%
print("Environment looks good. Continue to 01_data_and_bag_of_words.")
```

- [ ] **Step 2: Run as script to verify it passes (the test)**

Run:
```bash
source .venv/bin/activate
MPLBACKEND=Agg python notebooks/00_setup_and_tour.py
```
Expected: exits 0; prints device and "Matmul ... ok"; the `assert y.shape` holds.

- [ ] **Step 3: Render to notebook**

Run:
```bash
jupytext --to notebook --execute notebooks/00_setup_and_tour.py -o notebooks/00_setup_and_tour.ipynb
```
Expected: produces `notebooks/00_setup_and_tour.ipynb` without error.

- [ ] **Step 4: Commit**

```bash
git add notebooks/00_setup_and_tour.py notebooks/00_setup_and_tour.ipynb
git commit -m "feat: notebook 00 — setup & tour"
```

---

### Task 3: Notebook 01 (part A) — data download & word-level tokenizer

**Files:**
- Create: `notebooks/01_data_and_bag_of_words.py`

**Interfaces:**
- Consumes: venv from Task 1.
- Produces (within the notebook, for Task 4 to build on): `text` (str, full corpus); `WordTokenizer` with `.encode(str) -> list[int]`, `.decode(list[int]) -> str`, `.vocab_size` (int); `train_text`, `val_text` (str split 90/10). Vocabulary is capped to the top `VOCAB_SIZE` most frequent words plus `<unk>`.

- [ ] **Step 1: Write the data + tokenizer cells**

Create `notebooks/01_data_and_bag_of_words.py`:

```python
# %% [markdown]
# # 01 — Data & a Bag-of-Words Baseline
#
# Before any neural net, we build the simplest model that could possibly work:
# a **Bag-of-Words** next-word predictor. It will be mediocre — and crucially,
# it throws away word order. That measured failure motivates everything later.

# %%
import os
import re
import json
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)

# %% [markdown]
# ## Get the data
#
# TinyShakespeare (~1MB). Downloaded once and cached in `data/`.

# %%
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"

if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    print("Downloading TinyShakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("Characters:", len(text))
print("----- sample -----")
print(text[:250])

# %% [markdown]
# ## Word-level tokenization
#
# "Bag of **words**" is about words, so here we tokenize on words. We lowercase,
# split words and punctuation, cap the vocabulary to the most common words, and
# map everything else to `<unk>`.

# %%
def tokenize_words(s: str) -> list[str]:
    # words and standalone punctuation become tokens
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

VOCAB_SIZE = 2000  # cap for a tractable BoW model

class WordTokenizer:
    def __init__(self, corpus: str, max_vocab: int = VOCAB_SIZE):
        counts = Counter(tokenize_words(corpus))
        most_common = [w for w, _ in counts.most_common(max_vocab - 1)]
        self.itos = ["<unk>"] + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, s: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in tokenize_words(s)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

tok = WordTokenizer(text)
print("Vocab size:", tok.vocab_size)

# %% [markdown]
# Sanity check: encoding then decoding a known-in-vocab phrase round-trips.

# %%
sample = "to be or not to be"
ids = tok.encode(sample)
assert tok.decode(ids) == sample, tok.decode(ids)
print("round-trip ok:", ids, "->", tok.decode(ids))

# %% [markdown]
# ## Train / validation split
#
# We hold out the last 10% to measure generalization honestly.

# %%
ids_all = tok.encode(text)
n = len(ids_all)
split = int(0.9 * n)
train_ids = ids_all[:split]
val_ids = ids_all[split:]
print(f"tokens: {n} (train {len(train_ids)}, val {len(val_ids)})")
```

- [ ] **Step 2: Run as script to verify it passes (the test)**

Run:
```bash
source .venv/bin/activate
MPLBACKEND=Agg python notebooks/01_data_and_bag_of_words.py
```
Expected: exits 0; downloads data on first run; prints vocab size; the round-trip `assert` holds; prints token counts.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_data_and_bag_of_words.py data/.gitkeep
git commit -m "feat: notebook 01 part A — data download & word tokenizer"
```

---

### Task 4: Notebook 01 (part B) — Bag-of-Words baseline, training, perplexity, order-blindness

**Files:**
- Modify: `notebooks/01_data_and_bag_of_words.py` (append cells)
- Create (generated): `notebooks/01_data_and_bag_of_words.ipynb`
- Create: `assets/phase1_metrics.json`
- Create: `assets/01_bow_loss.png`

**Interfaces:**
- Consumes (from Task 3, same file): `tok`, `train_ids`, `val_ids`, `device`-style usage.
- Produces: `assets/phase1_metrics.json` with key `"bow_val_perplexity"` (float) — consumed by notebook 02 (Task 5). Also defines `CONTEXT` (int, context window in words) reused conceptually by notebook 02.

- [ ] **Step 1: Append BoW dataset + model + the order-blindness expectation**

Append to `notebooks/01_data_and_bag_of_words.py`:

```python
# %% [markdown]
# ## The Bag-of-Words representation
#
# To predict the next word we look at the previous `CONTEXT` words — but we
# represent them as an **unordered count vector** of length `vocab_size`. The
# position of each word is discarded; only *how many times* each appears remains.

# %%
CONTEXT = 8

def make_bow_dataset(ids: list[int], context: int, vocab_size: int, limit: int):
    # returns X: (N, vocab_size) float counts, Y: (N,) next-word ids
    xs, ys = [], []
    step = max(1, (len(ids) - context - 1) // limit)  # subsample for speed
    for i in range(0, len(ids) - context - 1, step):
        window = ids[i:i + context]
        vec = torch.zeros(vocab_size)
        for w in window:
            vec[w] += 1.0
        xs.append(vec)
        ys.append(ids[i + context])
    return torch.stack(xs), torch.tensor(ys)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

Xtr, Ytr = make_bow_dataset(train_ids, CONTEXT, tok.vocab_size, limit=20000)
Xva, Yva = make_bow_dataset(val_ids, CONTEXT, tok.vocab_size, limit=4000)
Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)
print("BoW train X:", tuple(Xtr.shape), "val X:", tuple(Xva.shape))

# %% [markdown]
# The model is a single linear layer: count vector -> next-word logits. This is
# multinomial logistic regression — about as simple as a learnable model gets.

# %%
class BoWModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        return self.fc(x)

bow = BoWModel(tok.vocab_size).to(device)
print("BoW params:", sum(p.numel() for p in bow.parameters()))

# %% [markdown]
# ## Train it
#
# Plain full-batch-ish training with AdamW and cross-entropy.

# %%
def evaluate(model, X, Y) -> float:
    model.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(X), Y)
    model.train()
    return float(loss)

opt = torch.optim.AdamW(bow.parameters(), lr=1e-2)
EPOCHS = 60
losses = []
for epoch in range(EPOCHS):
    logits = bow(Xtr)
    loss = nn.functional.cross_entropy(logits, Ytr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss))

val_loss = evaluate(bow, Xva, Yva)
bow_val_ppl = float(torch.exp(torch.tensor(val_loss)))
print(f"BoW final train loss {losses[-1]:.3f} | val loss {val_loss:.3f} | val perplexity {bow_val_ppl:.1f}")

# %%
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("BoW training loss")
plt.tight_layout()
plt.savefig("assets/01_bow_loss.png", dpi=120)
plt.show()

# %% [markdown]
# ## The punchline: order is gone
#
# Two different orderings of the same words produce the **identical** BoW vector,
# so the model gives the **identical** prediction. A real language needs order —
# this is the measured flaw we'll fix with attention.

# %%
phrase_a = tok.encode("the king is dead")
phrase_b = list(reversed(phrase_a))

def bow_vec(ids):
    v = torch.zeros(tok.vocab_size, device=device)
    for w in ids:
        v[w] += 1.0
    return v

va, vb = bow_vec(phrase_a), bow_vec(phrase_b)
assert torch.equal(va, vb), "BoW vectors should be identical regardless of order"

with torch.no_grad():
    pa = bow(va.unsqueeze(0))
    pb = bow(vb.unsqueeze(0))
assert torch.allclose(pa, pb), "predictions should be identical regardless of order"
print("Confirmed: '", tok.decode(phrase_a), "' and '", tok.decode(phrase_b),
      "' give identical predictions. Order is invisible to BoW.")

# %% [markdown]
# ## Save the baseline number
#
# Notebook 02 will try to beat this perplexity with dense embeddings.

# %%
os.makedirs("assets", exist_ok=True)
with open("assets/phase1_metrics.json", "w") as f:
    json.dump({"bow_val_perplexity": bow_val_ppl,
               "bow_params": sum(p.numel() for p in bow.parameters()),
               "context": CONTEXT, "vocab_size": tok.vocab_size}, f, indent=2)
print("Saved baseline:", bow_val_ppl)
```

- [ ] **Step 2: Run as script to verify it passes (the test)**

Run:
```bash
source .venv/bin/activate
MPLBACKEND=Agg python notebooks/01_data_and_bag_of_words.py
```
Expected: exits 0; both order-blindness `assert`s hold; prints a finite BoW val perplexity; writes `assets/phase1_metrics.json` and `assets/01_bow_loss.png`.

- [ ] **Step 3: Render to notebook**

Run:
```bash
jupytext --to notebook --execute notebooks/01_data_and_bag_of_words.py -o notebooks/01_data_and_bag_of_words.ipynb
```
Expected: produces `notebooks/01_data_and_bag_of_words.ipynb` without error.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_data_and_bag_of_words.py notebooks/01_data_and_bag_of_words.ipynb assets/phase1_metrics.json assets/01_bow_loss.png
git commit -m "feat: notebook 01 part B — BoW baseline, perplexity, order-blindness demo"
```

---

### Task 5: Notebook 02 — Embeddings beat BoW; switch to char-level + batching

**Files:**
- Create: `notebooks/02_embeddings.py`
- Create (generated): `notebooks/02_embeddings.ipynb`
- Create: `assets/02_embedding_loss.png`

**Interfaces:**
- Consumes: `assets/phase1_metrics.json` (key `bow_val_perplexity`) from Task 4; the same data/tokenizer approach (re-declared self-contained in this notebook).
- Produces: demonstrates `MeanEmbeddingModel`; a char-level tokenizer (`stoi`/`itos`) and a `get_batch(split)` returning `(xb, yb)` tensors of shape `(B, T)` — the batching convention Phase 2 builds on.

- [ ] **Step 1: Write the embedding-vs-BoW cells**

Create `notebooks/02_embeddings.py`:

```python
# %% [markdown]
# # 02 — Embeddings (and why they win)
#
# BoW used a sparse count vector: one slot per vocab word, mostly zeros, and no
# notion that "king" and "queen" are related. A **learned embedding** maps each
# word to a small dense vector. Same task, same prediction head — we only swap
# the representation, then compare perplexity head-to-head with notebook 01.

# %%
import os
import re
import json
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# %% [markdown]
# Rebuild the same word-level tokenizer and dataset as notebook 01 (self-contained).

# %%
def tokenize_words(s: str) -> list[str]:
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

VOCAB_SIZE = 2000
CONTEXT = 8

counts = Counter(tokenize_words(text))
itos = ["<unk>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
stoi = {w: i for i, w in enumerate(itos)}
vocab_size = len(itos)
unk = stoi["<unk>"]

def encode(s): return [stoi.get(w, unk) for w in tokenize_words(s)]

ids_all = encode(text)
split = int(0.9 * len(ids_all))
train_ids, val_ids = ids_all[:split], ids_all[split:]

def make_dataset(ids, context, limit):
    xs, ys = [], []
    step = max(1, (len(ids) - context - 1) // limit)
    for i in range(0, len(ids) - context - 1, step):
        xs.append(ids[i:i + context])
        ys.append(ids[i + context])
    return torch.tensor(xs), torch.tensor(ys)

Xtr, Ytr = make_dataset(train_ids, CONTEXT, 20000)
Xva, Yva = make_dataset(val_ids, CONTEXT, 4000)
Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)
print("context windows:", tuple(Xtr.shape))

# %% [markdown]
# ## The mean-embedding model
#
# Look up an embedding for each of the `CONTEXT` words, average them, and project
# to next-word logits. Note: we keep it order-blind on purpose (averaging) so the
# *only* thing that changes vs BoW is sparse-counts -> dense-learned vectors.

# %%
EMB_DIM = 64

class MeanEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):            # x: (B, T) int ids
        e = self.emb(x)              # (B, T, E)
        pooled = e.mean(dim=1)       # (B, E) — order-blind
        return self.fc(pooled)       # (B, vocab)

model = MeanEmbeddingModel(vocab_size, EMB_DIM).to(device)
emb_params = sum(p.numel() for p in model.parameters())
print("Embedding-model params:", emb_params)

# %% [markdown]
# ## Train (same recipe as BoW)

# %%
def evaluate(m, X, Y):
    m.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(m(X), Y)
    m.train()
    return float(loss)

opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
losses = []
for epoch in range(60):
    logits = model(Xtr)
    loss = nn.functional.cross_entropy(logits, Ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))

emb_val_loss = evaluate(model, Xva, Yva)
emb_val_ppl = float(torch.exp(torch.tensor(emb_val_loss)))
print(f"Embedding val perplexity {emb_val_ppl:.1f}")

plt.figure(figsize=(6, 4))
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("train loss")
plt.title("Mean-embedding training loss"); plt.tight_layout()
plt.savefig("assets/02_embedding_loss.png", dpi=120); plt.show()

# %% [markdown]
# ## Head-to-head with the BoW baseline

# %%
with open("assets/phase1_metrics.json") as f:
    bow_metrics = json.load(f)
bow_val_ppl = bow_metrics["bow_val_perplexity"]
bow_params = bow_metrics["bow_params"]

print(f"BoW       : perplexity {bow_val_ppl:7.1f} | params {bow_params:,}")
print(f"Embeddings: perplexity {emb_val_ppl:7.1f} | params {emb_params:,}")
assert emb_val_ppl < bow_val_ppl, "embeddings should achieve lower perplexity than BoW"
assert emb_params < bow_params, "embeddings should use far fewer parameters than BoW"
print("Embeddings win: lower perplexity AND fewer parameters.")

# %% [markdown]
# ## But embeddings are *still* order-blind
#
# Averaging is permutation-invariant, so reordering the context gives the same
# prediction — exactly BoW's flaw. Dense vectors fixed *representation*, not
# *order*. Order is what **attention** (notebook 03) finally gives us.

# %%
a = torch.tensor([encode("the king is dead")[:CONTEXT]], device=device)
b = torch.tensor([list(reversed(encode("the king is dead")[:CONTEXT]))], device=device)
with torch.no_grad():
    pa, pb = model(a), model(b)
assert torch.allclose(pa, pb, atol=1e-5), "mean-pooling is order-blind"
print("Confirmed: reordering the context does not change the prediction.")

# %% [markdown]
# ## Switching to characters for the transformer
#
# From here on the model is character-level: a tiny, fully transparent vocab
# (~65 symbols), no `<unk>`, no vocabulary cap. We tensorize the whole corpus
# and define the batching helper Phase 2 will reuse.

# %%
chars = sorted(set(text))
cstoi = {c: i for i, c in enumerate(chars)}
citos = {i: c for c, i in cstoi.items()}
char_vocab_size = len(chars)
print("char vocab:", char_vocab_size)

cdata = torch.tensor([cstoi[c] for c in text], dtype=torch.long)
cn = int(0.9 * len(cdata))
ctrain, cval = cdata[:cn], cdata[cn:]

BLOCK = 64   # context length in characters
BATCH = 32

def get_batch(split: str):
    d = ctrain if split == "train" else cval
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    xb = torch.stack([d[i:i + BLOCK] for i in ix])
    yb = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return xb.to(device), yb.to(device)

xb, yb = get_batch("train")
assert xb.shape == (BATCH, BLOCK) and yb.shape == (BATCH, BLOCK)
print("char batch:", tuple(xb.shape), "-> targets", tuple(yb.shape))
print("Ready for attention in notebook 03.")
```

- [ ] **Step 2: Run as script to verify it passes (the test)**

Run:
```bash
source .venv/bin/activate
MPLBACKEND=Agg python notebooks/02_embeddings.py
```
Expected: exits 0; prints both perplexities; the `emb_val_ppl < bow_val_ppl` and `emb_params < bow_params` asserts hold; the order-blind assert holds; char-batch shape assert holds.

- [ ] **Step 3: Render to notebook**

Run:
```bash
jupytext --to notebook --execute notebooks/02_embeddings.py -o notebooks/02_embeddings.ipynb
```
Expected: produces `notebooks/02_embeddings.ipynb` without error.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_embeddings.py notebooks/02_embeddings.ipynb assets/02_embedding_loss.png
git commit -m "feat: notebook 02 — embeddings beat BoW; switch to char-level batching"
```

---

## Self-Review

**Spec coverage (Phase 1 scope = notebooks 00–02):**
- Setup/env + device convention → Task 1, Task 2. ✓
- Data load/explore + auto-download swappable → Task 3. ✓
- Word-level tokenization for BoW → Task 3. ✓
- Bag-of-Words runnable baseline + perplexity + order-blindness demo → Task 4. ✓
- Embeddings beat BoW (fair: same task/head, only representation changes) → Task 5. ✓
- Embeddings still order-blind → motivates attention → Task 5. ✓
- Switch to char-level + tensorize + batching for the pipeline ahead → Task 5. ✓
- Self-contained notebooks, no shared package, committed metrics interface → Conventions + Task 4/5. ✓
- Inline sanity-checks as tests (round-trip, order-blindness, perplexity, shapes) → Tasks 3–5. ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete, runnable cell content. ✓

**Type/name consistency:** `pick_device` (nb00) vs inline `device` selection (nb01/02) — both use the same MPS-else-CPU rule, intentionally re-declared for self-containment; documented. `WordTokenizer.encode/decode/vocab_size`, `make_bow_dataset`, `BoWModel`, `MeanEmbeddingModel`, `get_batch` are each defined where used. `assets/phase1_metrics.json` key `bow_val_perplexity` written in Task 4, read in Task 5 — consistent. `CONTEXT=8`, `VOCAB_SIZE=2000` consistent across notebooks 01 and 02. ✓
