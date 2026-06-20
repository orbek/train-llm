# Build an LLM from Scratch

![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-v2.12.1-blue?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)

A progressive, notebook-driven course that builds a modern decoder-only
language model from raw text up to a Mixture-of-Experts variant — entirely
from scratch in PyTorch (no HuggingFace), sized to train on a Mac in minutes.

Each new method earns its place by beating a **measured** number from the
step before: Bag-of-Words → embeddings → attention → modern components → MoE.

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

The notebooks automatically detect and use the best available compute engine — no manual configuration needed:

- **NVIDIA GPU (CUDA)** on Windows or Linux — used if available.
- **Apple Silicon GPU (MPS)** on macOS (M1/M2/M3/M4) — used if CUDA is not present.
- **CPU** — universal fallback; works on any machine, just slower for training.

> **Windows/Linux NVIDIA users:** the default `pip install torch` may install a CPU-only build. If PyTorch does not detect your GPU, install a CUDA-enabled build from https://pytorch.org.

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

Notebooks use the standard `python3` kernelspec (no custom registration needed). To regenerate a rendered `.ipynb` from its `.py` source:

```bash
jupytext --to notebook --execute --set-kernel python3 notebooks/<file>.py -o notebooks/<file>.ipynb
```

## Status

- [x] Phase 1 — Foundations (notebooks 00–02)
- [x] Phase 2 — The working transformer (notebooks 03–07)
- [x] Phase 3 — Advanced: tuning, BPE, MoE (notebooks 08–10)

All three phases complete: the course runs end-to-end from a Bag-of-Words baseline
to a Mixture-of-Experts transformer that generates Shakespeare.

See `docs/superpowers/specs/2026-06-20-llm-from-scratch-design.md` for the full design.
