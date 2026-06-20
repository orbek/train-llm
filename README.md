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

Notebooks use the standard `python3` kernelspec (no custom registration needed). To regenerate a rendered `.ipynb` from its `.py` source:

```bash
jupytext --to notebook --execute --set-kernel python3 notebooks/<file>.py -o notebooks/<file>.ipynb
```

## Status

- [ ] Phase 1 — Foundations (notebooks 00–02) — in progress
- [ ] Phase 2 — The working transformer (notebooks 03–07)
- [ ] Phase 3 — Advanced: tuning, BPE, MoE (notebooks 08–10)

See `docs/superpowers/specs/2026-06-20-llm-from-scratch-design.md` for the full design.
