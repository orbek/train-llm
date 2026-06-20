# Build an LLM from Scratch

![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-v2.12.1-blue?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)

A progressive, notebook-driven course that builds a modern decoder-only language
model from raw text all the way up to a Mixture-of-Experts variant — entirely
from scratch in PyTorch (no HuggingFace), sized to train on a laptop in minutes.

Every new method earns its place by beating a **measured** number from the step
before:

> Bag-of-Words → embeddings → attention → modern components → training →
> generation → tuning → BPE → Mixture-of-Experts

Each notebook teaches step by step in plain language, with runnable code,
inline plots, and `assert`-based sanity checks that double as lessons.

## The curriculum

| #  | Notebook | What you build |
|----|----------|----------------|
| 00 | Setup & tour | environment check, auto device selection, the roadmap |
| 01 | Data & Bag-of-Words | word-level tokenizer; a BoW next-word baseline (the order-blind starting point) |
| 02 | Embeddings | dense embeddings that beat BoW with far fewer parameters; word geometry |
| 03 | Attention | scaled dot-product → causal mask → multi-head → grouped-query attention (GQA) |
| 04 | Modern components | RoPE, RMSNorm, SwiGLU — each vs the older idea it replaces |
| 05 | Assembling the model | tour of the full Llama-style decoder block (imported from model.py), weight tying, param counts |
| 06 | Training | the training loop, loss curves, checkpointing |
| 07 | Evaluation & generation | perplexity, sampling (greedy/temperature/top-k/top-p), KV-cache |
| 08 | Tuning | learning-rate warmup + cosine decay, hyperparameter sweeps |
| 09 | BPE tokenizer | Byte-Pair Encoding from scratch vs char-level (≈2× compression) |
| 10 | Mixture-of-Experts | top-k routing + load balancing; the capstone |

The reusable model lives in [`model.py`](model.py) (a small Llama-style GPT:
GQA + RoPE + RMSNorm + SwiGLU + weight tying + KV-cache), tested by
[`test_model.py`](test_model.py). Notebooks 05–07 import it; the rest build
their pieces from scratch in-cell.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
# .venv\Scripts\activate.bat       # Windows (cmd)
pip install -r requirements.txt
```

(Optionally use [`uv`](https://github.com/astral-sh/uv): `uv venv && uv pip install -r requirements.txt` — faster.)

### Hardware

The notebooks automatically detect and use the best available compute engine —
no manual configuration needed:

- **NVIDIA GPU (CUDA)** on Windows or Linux — used if available.
- **Apple Silicon GPU (MPS)** on macOS (M1–M4) — used if CUDA is not present.
- **CPU** — universal fallback; works on any machine, just slower for training.

> **Windows / Linux NVIDIA users:** the default `pip install torch` may install a
> CPU-only build. If PyTorch does not detect your GPU, install a CUDA-enabled
> build from <https://pytorch.org>.

## How to run

Notebooks live in `notebooks/` as paired files: a jupytext `.py` source (the
version-controlled source of truth) and a generated `.ipynb`. Open the `.ipynb`
in JupyterLab:

```bash
jupyter lab
```

Or run a notebook headlessly as a script:

```bash
python notebooks/01_data_and_bag_of_words.py
```

To regenerate a rendered `.ipynb` from its `.py` source:

```bash
jupytext --to notebook --execute --set-kernel python3 notebooks/<file>.py -o notebooks/<file>.ipynb
```

The Shakespeare dataset auto-downloads on first run. To train on your own text,
drop a `.txt` file in `data/` and point the data cell at it.

## Results

By notebook 07 the model — a ~9.4M-parameter character-level transformer —
trains in a few minutes and generates recognizable Shakespeare, e.g.:

```
ROMEO:
The sea of the sea that would not see the sea
That thou hast seen the sea of the sea
```

(The committed training run is capped for fast notebook rendering; raise
`max_iters` in notebook 06 for sharper samples.)

## Status

- [x] Phase 1 — Foundations (notebooks 00–02)
- [x] Phase 2 — The working transformer (notebooks 03–07)
- [x] Phase 3 — Advanced: tuning, BPE, MoE (notebooks 08–10)

All three phases complete: the course runs end-to-end from a Bag-of-Words
baseline to a Mixture-of-Experts transformer that generates text.
