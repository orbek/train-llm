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
