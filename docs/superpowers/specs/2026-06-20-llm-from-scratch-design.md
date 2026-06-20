# Build an LLM from Scratch — Design Spec

**Date:** 2026-06-20
**Status:** Approved (design), pending implementation plan

## Summary

A progressive, notebook-driven educational project that builds a modern
decoder-only language model from raw text up to a Mixture-of-Experts (MoE)
variant — entirely from scratch in PyTorch, with no high-level modeling
libraries (no HuggingFace `transformers`). Every component is implemented and
explained in place, and each new technique is introduced *against* the older
alternative it replaces, justified by a **measured** improvement (loss /
perplexity) over the previous step.

The project trains a tiny-but-coherent model on a Mac (Apple Silicon MPS or
CPU) in minutes, so the learner sees the full pipeline end-to-end quickly and
can iterate.

## Goals

- **Learn deeply** — understand every piece by implementing it by hand.
- **Clean teachable code** — numbered, self-contained notebooks others can follow.
- **A real working model** — actually train and generate text, not just toy snippets.

## Non-Goals (YAGNI)

- No distributed / multi-GPU training.
- No HuggingFace `transformers`, `accelerate`, or similar high-level libraries.
- No production serving / API.
- No frontier-scale data or model sizes. Everything is sized for a single Mac.

## Constraints & Environment

- **Hardware:** the user's Mac. PyTorch with MPS backend, CPU fallback.
- **Python:** 3.11+.
- **Dependencies:** `torch`, `numpy`, `matplotlib`, `tqdm`, `jupyterlab`.
  `tiktoken` optional (for comparison only in the BPE notebook).
- **Setup:** plain `venv` + `requirements.txt`. `uv` noted in the README as a
  faster alternative.
- **No internet at train time** beyond a one-time auto-download of the dataset.

## Narrative Spine

The defining principle: **each method earns its place by beating a number from
the step before.**

```
Bag-of-Words (order-blind, sparse)
  → Embeddings (order-blind, dense, fewer params, lower perplexity)
    → Attention (order-aware)
      → Modern components (RoPE / RMSNorm / SwiGLU vs their predecessors)
        → Mixture-of-Experts (more capacity at similar active compute)
```

Every transition is motivated by a concrete, measured weakness of the prior
approach, shown live in the notebook.

## Dataset

- **TinyShakespeare** (~1MB plain text), auto-downloaded to `data/shakespeare.txt`.
- Designed so swapping in any other `.txt` is trivial (single path / loader cell).

## Tokenization Story

Three tokenizers appear, each at the point where it teaches the most:

1. **Word-level** (notebooks 01–02): split on whitespace/punctuation. Used for
   the Bag-of-Words and embedding lessons, because "bag of *words*" lands with
   its real name and is the most intuitive setting.
2. **Char-level** (notebook 02 onward): tiny transparent vocab (~65 chars), no
   UNK/OOV handling needed. This is what the transformer trains on. Chosen so
   the model and its vocabulary stay fully legible.
3. **BPE from scratch** (notebook 09): Byte-Pair Encoding implemented by hand,
   retrain, and compare against char-level. `tiktoken` shown only as a
   reference point.

## Model Architecture (Llama-style, decoder-only)

- Causal multi-head self-attention.
- **RoPE** rotary positional embeddings (taught vs learned absolute positions).
- **RMSNorm**, pre-norm placement (taught vs LayerNorm + post-norm).
- **SwiGLU** feed-forward (taught vs GELU MLP).
- Weight-tied input embedding / output projection.
- **KV-cache** for fast autoregressive generation.
- **MoE capstone (notebook 10):** replace the SwiGLU MLP with `N` experts + a
  top-k router and a load-balancing auxiliary loss.

### Config

A single `Config` dataclass holds every hyperparameter. Two presets:

- **`nano`** — ~1M params, tiny context, for instant debugging / overfit tests.
- **`default`** — ~6 layers, ~6 heads, embedding dim ~384, context length ~256,
  char vocab ~65 → roughly 10–15M params. The "real" training run; trains in
  minutes on MPS.

## Repository Layout

```
notebooks/        00_ through 10_  (the curriculum)
data/             shakespeare.txt (auto-downloaded, gitignored)
checkpoints/      saved model weights (gitignored)
assets/           generated plots / diagrams
requirements.txt
README.md         overview + how to run
.gitignore
```

Notebooks are **largely self-contained** (code lives in cells, read-and-run).
A small shared helper file is introduced only if duplication becomes genuinely
painful across notebooks; the default is self-containment.

## Curriculum (Notebook Sequence)

| #  | Notebook | Covers |
|----|----------|--------|
| 00 | Setup & tour | venv, `torch` + MPS check, project map, how to run |
| 01 | Data & Bag-of-Words baseline | load/explore Shakespeare; word-level tokenize; build BoW count vectors; train `BoW → linear → next-word`; measure val perplexity; **order-blindness demo** ("the cat sat" == "sat the cat") |
| 02 | Embeddings | dense learned embeddings vs sparse one-hot; same task/head, swap representation: `mean(token embeddings) → linear → logits`; beats BoW with fewer params; show it *still* ignores order → motivate attention; switch to char-level; tensorize + batch for the main pipeline |
| 03 | Attention from scratch | intuition → scaled dot-product → causal mask → single head → multi-head, step by step; visualize attention weights |
| 04 | Modern components | RoPE (vs learned positions), RMSNorm (vs LayerNorm), SwiGLU (vs GELU) — each implemented and contrasted |
| 05 | Assembling the model | pre-norm transformer block; stack blocks; full decoder-only model; weight tying; parameter count |
| 06 | Training loop | cross-entropy loss; AdamW; **overfit-one-batch sanity test**; full training run; loss curves; checkpointing |
| 07 | Evaluation & generation | train/val loss; perplexity; sampling (greedy, temperature, top-k, top-p); KV-cache; verify cached == non-cached output |
| 08 | Tuning | learning-rate warmup + cosine decay; model-size / context-length sweeps; which knobs matter and why |
| 09 | BPE tokenizer | Byte-Pair Encoding from scratch; retrain; compare vs char-level; `tiktoken` as reference |
| 10 | Mixture-of-Experts (capstone) | swap MLP for experts + top-k router; load-balancing loss; compare capacity vs active compute |

This covers the full requested journey: dataset, bag-of-words, embedding,
tensorizing, training loop, evaluation, tuning, testing — plus modern
transformer methods and MoE.

## Testing Philosophy

Because the project is notebook-based, "testing" means **inline sanity checks
that double as lessons**:

- Shape assertions at each stage (catch wiring bugs early).
- The classic **overfit-a-single-batch** test (proves the model can learn).
- Gradient-flow / no-NaN checks.
- KV-cache correctness: cached generation must match non-cached output.

## Fair-Comparison Methodology

For BoW vs embeddings (notebooks 01–02), the comparison holds the task and the
prediction head **constant** and changes **only the representation**:

- Same Shakespeare next-token task.
- Same final linear output layer.
- Same loss (cross-entropy → perplexity).

So any perplexity difference is attributable purely to the representation. The
same discipline — change one thing, measure the effect — carries through the
later architectural lessons where practical.

## Open Questions / Future Extensions

- Optional: a word-level vs char-level perplexity discussion when introducing BPE.
- Optional: dropout / weight decay ablations in the tuning notebook.
- Possible later sub-project: scaling the `default` config up on a rented GPU
  (explicitly out of scope here).
