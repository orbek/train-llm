# %% [markdown]
# # 00 — Setup & Tour
#
# Welcome! You are about to build a modern language model — the same kind of architecture
# that powers tools like ChatGPT — completely from scratch using Python and PyTorch.
#
# This first notebook does two things:
# 1. **Checks that your environment is ready** — the right Python version, PyTorch installed,
#    and your hardware detected.
# 2. **Maps out the journey ahead** — a quick tour of all 10 notebooks so you know where
#    we are going before we start.
#
# No machine-learning knowledge is assumed. If you know basic Python, you are ready.
# We will explain every new idea — math, jargon, and all — as it comes up.

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
# Neural networks do a *lot* of arithmetic — millions of multiplications per second.
# Modern hardware has dedicated chips that can do this arithmetic in parallel, much
# faster than an ordinary CPU.
#
# In PyTorch, the word **device** refers to the piece of hardware where a computation
# runs. The two you will encounter in this course are:
#
# - **CPU** — your computer's main processor. Always available, but slower for ML.
# - **MPS** (Metal Performance Shaders) — the GPU built into Apple Silicon Macs
#   (M1/M2/M3/M4 chips). "MPS" is Apple's name for the programming interface that
#   lets PyTorch talk to that GPU. GPUs run many operations at once, which makes
#   training dramatically faster.
#
# The `pick_device()` function below makes this choice automatically:
# 1. It asks PyTorch whether MPS is available on this machine
#    (`torch.backends.mps.is_available()`).
# 2. If yes, it returns `torch.device("mps")` — we will use the Apple GPU.
# 3. If not (e.g., you are on a Linux machine or an Intel Mac), it falls back to
#    `torch.device("cpu")` — perfectly fine for these notebooks, just a bit slower.
#
# Every later notebook reuses this exact helper, so you will only ever see one line
# to set the device: `device = pick_device()`.

# %%
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

# %% [markdown]
# ## Sanity check — a tiny matrix multiplication
#
# Before trusting a device with real training, we run a quick smoke test.
#
# A **tensor** is PyTorch's core data structure — think of it as a multi-dimensional
# array of numbers, like a spreadsheet extended to any number of dimensions.
# `torch.randn(3, 3)` creates a 3×3 tensor filled with random numbers drawn from a
# standard normal distribution (mean 0, standard deviation 1).
#
# The `@` operator performs **matrix multiplication** (matmul): it combines two matrices
# according to the standard rules of linear algebra, producing a new matrix.
# Matrix multiplication is *the* fundamental operation in neural networks — every layer
# is essentially a big matmul — so if it works correctly on our chosen device, we are
# good to go.
#
# `assert y.shape == (3, 3)` checks that the result has the right shape.
# A 3×3 matrix multiplied by another 3×3 matrix must produce a 3×3 matrix; if the
# shape were wrong, something would have gone badly off-course.

# %%
x = torch.randn(3, 3, device=device)
y = x @ x
assert y.shape == (3, 3)
print("Matmul on", device, "ok. Sum =", float(y.sum()))

# %% [markdown]
# ## The journey ahead
#
# Here is the full roadmap. Each notebook builds on the one before it, and each step
# introduces exactly one new idea so you can see clearly what it adds. By notebook 10
# you will have a working Mixture-of-Experts transformer — trained by you, understood
# by you.
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
# Each step beats a *measured* number from the one before — you will not just be told
# that the newer technique is better, you will watch the metrics improve in real time.
# On to notebook 01!

# %%
print("Environment looks good. Continue to 01_data_and_bag_of_words.")
