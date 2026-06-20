# %% [markdown]
# # 04 — Modern Components: RoPE, RMSNorm, and SwiGLU
#
# Welcome back! In notebook 03 we built attention from scratch — scaled dot-product
# attention, the causal mask, multi-head attention, and grouped-query attention.
#
# This notebook zooms in on three **modern architectural choices** that distinguish
# recent large language models (Llama 2/3, Mistral, Gemma, etc.) from the original
# GPT-2/BERT-era transformers. Each component is taught *against the older thing it
# replaced*, so you understand the "why" before the "how":
#
# | Modern component | Replaces | Key benefit |
# |---|---|---|
# | **RMSNorm** | LayerNorm | Simpler, faster — drops mean-centering and bias |
# | **RoPE** | Learned absolute positions | Encodes *relative* distance; generalizes to longer sequences |
# | **SwiGLU** | Plain GELU MLP | Gated nonlinearity; empirically stronger at same parameter count |
#
# Like all other notebooks in this series, these are **pedagogical** versions written for
# clarity. The canonical production implementations will live in `model.py` (Task 4).

# %% [markdown]
# ## A note on working directories
#
# Jupyter runs a notebook from the folder the notebook lives in (`notebooks/`), not from
# the project root. That means paths like `data/` and `assets/` would silently resolve
# to `notebooks/data/` and `notebooks/assets/` — the wrong place. The cell below walks
# up the directory tree until it finds the project root (identified by `requirements.txt`)
# and changes the kernel's working directory there. After this cell runs, every path in
# the notebook is relative to the project root, no matter how you launched Jupyter.

# %%
import os
# Walk up to the repo root so relative paths (data/, assets/, checkpoints/) resolve
# no matter which directory the notebook kernel was launched from.
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():          # reached filesystem root; stop
        break
    os.chdir(parent)
print("Working directory:", os.getcwd())

# %% [markdown]
# ## Imports and reproducibility
#
# We import `torch` and `torch.nn` for tensors and neural-network modules, and
# `torch.nn.functional` (`F`) for activation functions like `silu`.
#
# `torch.manual_seed(1337)` fixes the random number generator so every run of this
# notebook produces the same numbers. `device` selects MPS (Apple Silicon GPU) if
# available, otherwise CPU.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("device:", device)
# Note: these small teaching cells run on CPU for clarity; the full model (later notebooks) uses the detected device.

# %% [markdown]
# ---
# ## Part 1 — RMSNorm vs LayerNorm
#
# ### Why normalization matters
#
# When a neural network processes data through many layers, the **activations** (the
# numbers flowing through the network) can grow very large or shrink very small. Large
# activations cause gradients to explode during backpropagation; tiny activations cause
# gradients to vanish. Either way, training becomes unstable or impossibly slow.
#
# **Normalization** solves this by rescaling activations at every layer to a predictable
# range. Crucially, it does this *without* modifying the network's ability to learn —
# a set of **learned scale parameters** (and sometimes biases) lets the model undo the
# normalization if needed.
#
# ### LayerNorm: the classic approach
#
# **Layer normalization** (Ba et al., 2016) works on each sample independently:
#
# 1. Compute the **mean** across the feature dimension: `μ = mean(x)`.
# 2. Compute the **standard deviation**: `σ = std(x)`.
# 3. **Subtract the mean and divide by std**: `x̂ = (x - μ) / (σ + ε)`.
# 4. Apply learned **scale** `γ` (weight) and **shift** `β` (bias): output = `γ · x̂ + β`.
#
# The subtraction of the mean is called **mean-centering**. It ensures the output has
# zero mean. The bias `β` allows the output mean to be non-zero if the model learns that.
#
# ### RMSNorm: drop mean-centering and bias
#
# **Root Mean Square Normalization** (Zhang & Sennrich, 2019) simplifies LayerNorm in
# two ways:
#
# 1. **No mean-centering**: skip the `x - μ` step entirely.
# 2. **No bias**: skip the `+ β` shift.
#
# Instead, it only divides by the **root mean square (RMS)** of the activations:
#
# ```
# RMS(x) = sqrt( mean(x²) )
# output  = x / RMS(x) · γ
# ```
#
# In PyTorch, `torch.rsqrt(a)` computes `1 / sqrt(a)` efficiently, so:
# ```
# output = x * rsqrt( mean(x²) + ε ) * γ
# ```
#
# **Why does this work?** The mean-centering in LayerNorm was mainly a historical
# convention from batch normalization. Empirically, removing it costs very little in
# model quality while making the operation noticeably faster — especially important
# when you normalize at every single transformer layer.

# %%
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# %% [markdown]
# ### Assert: output has unit RMS per row
#
# With `weight = ones` (the initialized default), RMSNorm should produce outputs
# where the RMS of each row is approximately 1.0.
#
# We verify: for each row of the output, compute `mean(output²)`. Because
# `output = x / RMS(x) · 1`, we get `output² = x² / RMS(x)²`, and
# `mean(output²) = mean(x²) / mean(x²) = 1`. The `atol=1e-2` tolerance
# accounts for the small `eps` added for numerical stability.

# %%
torch.manual_seed(1337)
dim = 64
norm = RMSNorm(dim)
x = torch.randn(8, dim)
out = norm(x)
rms_out = out.pow(2).mean(-1)
assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-2), \
    f"RMSNorm unit-RMS assert failed: {rms_out}"
print("RMSNorm assert passed — per-row RMS:", rms_out[:3].tolist())

# %% [markdown]
# ### LayerNorm vs RMSNorm: a side-by-side comparison
#
# Let's compare the two on the same input to see that they produce similar
# (but not identical) outputs. The key practical difference is speed: RMSNorm
# does less arithmetic per forward pass, and does not maintain a bias parameter.

# %%
torch.manual_seed(42)
x_cmp = torch.randn(4, dim)

layernorm = nn.LayerNorm(dim)
rmsnorm   = RMSNorm(dim)

out_ln  = layernorm(x_cmp)
out_rms = rmsnorm(x_cmp)

print(f"LayerNorm parameters: weight {layernorm.weight.shape}, bias {layernorm.bias.shape}")
print(f"RMSNorm  parameters: weight {rmsnorm.weight.shape}  (no bias)")
print(f"LayerNorm mean of first row: {out_ln[0].mean().item():.6f}  (≈0, mean-centered)")
print(f"RMSNorm  mean of first row: {out_rms[0].mean().item():.6f}  (not forced to 0)")

# %% [markdown]
# ---
# ## Part 2 — RoPE vs Learned Absolute Positions
#
# ### Absolute position embeddings: the old way
#
# In the original transformer (Vaswani et al., 2017) and in GPT-2, each token's
# embedding gets a **position embedding** added to it before the first attention layer.
# The simplest version is a **learned absolute position embedding**: a lookup table
# `nn.Embedding(max_seq_len, n_embd)` that maps each integer position index (0, 1, 2,
# ...) to a learned vector, which is added to the token embedding.
#
# **The problem**: these embeddings are assigned independently per position. The
# embedding for position 5 has no built-in mathematical relationship to position 10.
# The model must *learn* from data that position 10 is 5 steps away from position 5.
# This is inefficient and doesn't generalize beyond the training sequence length.
#
# ### RoPE: Rotary Position Encoding
#
# **Rotary Position Encoding** (Su et al., 2021) takes a completely different approach.
# Instead of *adding* a position vector to embeddings, RoPE **rotates** the query and
# key vectors before the attention dot product.
#
# The key insight: if you rotate two vectors by angles proportional to their positions,
# then their dot product depends only on the **difference** in their positions, not on
# the absolute positions themselves. This is the **relative position** property.
#
# Concretely, a query at position `p` and a key at position `p+k` always produce the
# same dot product as a query at position `q` and a key at position `q+k`, for any
# choice of `p` and `q` (as long as `k` — the offset — is the same).
#
# This means the model learns "5 steps apart feels like this" rather than "position 5
# interacts with position 10 in this specific way". It generalizes much better.
#
# ### How the rotation works
#
# Each head's dimension (`head_dim`) is divided into pairs: (dim 0, dim 1),
# (dim 2, dim 3), ..., (dim d-2, dim d-1). Each pair gets rotated by a different
# **frequency** determined by the pair index `i`:
#
# ```
# θᵢ = 1 / theta^(2i / head_dim)    where theta = 10000 (base frequency)
# ```
#
# For position `p`, pair `i` is rotated by angle `p · θᵢ`. Lower pairs (small `i`)
# rotate slowly (low frequency, captures long-range context). Higher pairs (large `i`)
# rotate fast (high frequency, captures local context). This multi-scale structure is
# similar to how sinusoidal encodings work.
#
# The rotation pairs each element in the **first half** of the vector with the
# matching element in the **second half**: element `i` is paired with element
# `i + head_dim/2`, and that pair is rotated by a position-dependent angle:
# ```
# pair (a, b) → (a·cos - b·sin,  a·sin + b·cos)
# ```
# We do all pairs at once with the "rotate-half" trick — split `x` into halves
# `[A | B]` and return `[-B | A]`:
# ```
# rotated = x * cos + rotate_half(x) * sin
# rotate_half([e0, e1, e2, e3]) = [-e2, -e3, e0, e1]   # halves, not adjacent pairs
# ```
# (This is the "chunked" RoPE variant: pairs are (first-half, second-half), not
# neighbouring elements. `cos`/`sin` repeat each half so both members of a pair
# share the same angle.)
#
# The `cos` and `sin` values are **precomputed** for all positions and all pairs,
# giving tensors of shape `(seq_len, head_dim)`.

# %%
def build_rope(head_dim, seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)            # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, head_dim)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    return x * cos + rotate_half(x) * sin

# %% [markdown]
# ### Understanding `build_rope`
#
# Let's break down what `build_rope` computes:
#
# - `inv_freq`: one frequency per pair of dimensions. For `head_dim=16`, there are 8
#   pairs. Each frequency is `1 / 10000^(2i/16)` for `i` in `{0,1,...,7}`.
# - `torch.outer(t, inv_freq)`: compute `position × frequency` for every combination
#   of position and frequency pair. Shape: `(seq_len, head_dim/2)`.
# - `torch.cat([freqs, freqs], dim=-1)`: duplicate along the feature dimension to get
#   shape `(seq_len, head_dim)`. This is because the same angle applies to both
#   `x1` (via cos) and `-x2`/`x1` (via sin) in the rotation.
# - Return `cos` and `sin` of these angles — the precomputed rotation factors.

# %%
head_dim = 16
seq_len = 64
cos, sin = build_rope(head_dim, seq_len)
print(f"cos shape: {tuple(cos.shape)}   (seq_len={seq_len}, head_dim={head_dim})")
print(f"sin shape: {tuple(sin.shape)}")
print(f"cos[0, :4] (position 0, first 4 dims): {cos[0, :4].tolist()}")   # all ones at pos 0
print(f"cos[1, :4] (position 1, first 4 dims): {cos[1, :4].tolist()}")

# %% [markdown]
# ### Assert: the relative-position property
#
# This is the key property that makes RoPE powerful. Let's verify it directly.
#
# We pick two content vectors (same query `q_vec`, same key `k_vec`) and rotate them
# at two different pairs of absolute positions that share the **same offset `k=5`**:
#
# - Pair 1: query at position `p=10`, key at position `p+k=15`.
# - Pair 2: query at position `q=30`, key at position `q+k=35`.
#
# After applying RoPE, the dot products `(q_at_p · k_at_pk)` and `(q_at_q · k_at_qk)`
# must be **equal** (to within floating-point tolerance `atol=1e-4`).
#
# This is the mathematical guarantee that RoPE encodes *relative*, not absolute, position.
#
# **Implementation note**: `apply_rope` expects `cos`/`sin` of shape `(T, head_dim)`
# and `x` of shape `(B, n_head, T, head_dim)`. When rotating a single position, we
# slice `cos[p:p+1]` which has shape `(1, head_dim)`. This broadcasts correctly
# against `x` of shape `(B, n_head, 1, head_dim)` — no reshape or unsqueeze needed.

# %%
torch.manual_seed(1337)
head_dim = 16
seq_len = 64
cos, sin = build_rope(head_dim, seq_len)

# Fixed content vectors (same content, different positions)
q_vec = torch.randn(1, 1, 1, head_dim)  # (B, n_head, 1, head_dim)
k_vec = torch.randn(1, 1, 1, head_dim)

k = 5   # fixed offset
p = 10  # first absolute base position
q = 30  # second absolute base position (p != q)

# Apply RoPE at positions (p, p+k) — select single-position cos/sin slices
q_at_p   = apply_rope(q_vec, cos[p:p+1], sin[p:p+1])
k_at_pk  = apply_rope(k_vec, cos[p+k:p+k+1], sin[p+k:p+k+1])
dot_p    = (q_at_p * k_at_pk).sum().item()

q_at_q   = apply_rope(q_vec, cos[q:q+1], sin[q:q+1])
k_at_qk  = apply_rope(k_vec, cos[q+k:q+k+1], sin[q+k:q+k+1])
dot_q    = (q_at_q * k_at_qk).sum().item()

print(f"RoPE relative-position assert:")
print(f"  dot at positions ({p}, {p+k}) = {dot_p:.6f}")
print(f"  dot at positions ({q}, {q+k}) = {dot_q:.6f}")
assert abs(dot_p - dot_q) < 1e-4, \
    f"RoPE relative-position FAILED: dot_p={dot_p:.6f}, dot_q={dot_q:.6f}, diff={abs(dot_p-dot_q):.2e}"
print("RoPE relative-position assert PASSED")

# %% [markdown]
# ### RoPE vs absolute positions: a summary
#
# | Property | Learned absolute | RoPE |
# |---|---|---|
# | Encoding method | Add a lookup vector | Rotate Q and K |
# | Position relationship | Learned independently | Relative by construction |
# | Extrapolation beyond training length | Poor | Better (with variants like YaRN) |
# | Extra parameters | `max_seq_len × n_embd` | None (computed, not learned) |
# | Where applied | Token embedding input | Inside each attention head |
#
# Modern models (Llama 2/3, Mistral, Gemma) all use RoPE or a variant of it.

# %% [markdown]
# ---
# ## Part 3 — SwiGLU vs Plain GELU MLP
#
# ### The feed-forward layer in a transformer
#
# Every transformer block contains two sub-layers: (1) attention, and (2) a
# **feed-forward network (MLP)**. The MLP applies a nonlinear transformation to each
# token independently — there is no interaction between positions in this layer.
#
# The classical MLP used in GPT-2 is simple:
# 1. **Up-projection**: linear layer that expands the embedding dimension by 4×.
# 2. **Nonlinearity**: GELU activation (smooth approximation of ReLU).
# 3. **Down-projection**: linear layer that contracts back to the original dimension.
#
# ```
# output = W_down( GELU( W_up(x) ) )
# ```
#
# ### SwiGLU: the gated variant
#
# **SwiGLU** (Shazeer, 2020) replaces the single up-projection with **two** parallel
# up-projections:
#
# - `w1(x)`: the **gate** — passed through SiLU activation.
# - `w3(x)`: the **value** — kept as-is (no activation).
#
# These are multiplied element-wise before the down-projection:
# ```
# output = W_down( SiLU( W1(x) ) * W3(x) )
# ```
#
# The **gate** `SiLU(W1(x))` learns to *control how much* of each dimension of the
# value `W3(x)` passes through. Dimensions where the gate is near zero are suppressed;
# dimensions where the gate is near 1 pass through freely. This gives the network a
# dynamic, content-dependent filtering mechanism — richer than a fixed nonlinearity.
#
# ### Why SiLU (Swish)?
#
# **SiLU** (Sigmoid-Weighted Linear Unit, also called Swish) is defined as:
# ```
# SiLU(x) = x * sigmoid(x)
# ```
# Properties that make it attractive:
# - **Smooth**: unlike ReLU, it is differentiable everywhere (no hard zero at x=0).
# - **Self-gating**: the sigmoid factor acts as a soft gate on the linear term.
# - **Slightly negative for x < 0**: unlike ReLU (which is exactly 0 for x < 0), SiLU
#   has a small negative region, which can improve gradient flow.
#
# ### Hidden dimension in SwiGLU
#
# Standard MLPs use `4 × n_embd` hidden units. SwiGLU uses two up-projections, so to
# keep the total parameter count similar, the hidden dimension is reduced to
# approximately `(8/3) × n_embd` and rounded up to a multiple of 64 for hardware
# efficiency. The formula `64 * (((int(8/3 * n_embd)) + 63) // 64)` does this rounding.

# %%
class SwiGLU(nn.Module):
    def __init__(self, n_embd, hidden=None, dropout=0.0):
        super().__init__()
        if hidden is None:
            hidden = 64 * (((int(8/3 * n_embd)) + 63) // 64)   # ~8/3·n_embd, rounded to 64
        self.w1 = nn.Linear(n_embd, hidden, bias=False)   # gate
        self.w3 = nn.Linear(n_embd, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, n_embd, bias=False)   # down
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# %% [markdown]
# ### Assert: output shape equals input shape
#
# The MLP is a **residual block**: its output is added back to the input (the residual
# stream). This means the output shape must exactly match the input shape `(B, T, C)`.
# We verify this invariant holds regardless of the hidden dimension chosen.

# %%
torch.manual_seed(1337)
B, T, C = 2, 8, 128
mlp = SwiGLU(n_embd=C)
x = torch.randn(B, T, C)
out = mlp(x)
assert out.shape == x.shape, f"SwiGLU shape mismatch: {out.shape} != {x.shape}"
print(f"SwiGLU assert passed — output shape: {out.shape}")

# %% [markdown]
# ### SwiGLU vs plain GELU MLP: a comparison
#
# Let's inspect the hidden dimension and parameter counts of each.

# %%
class GELUMlp(nn.Module):
    """Classic 4× MLP with GELU activation, as used in GPT-2."""
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=True)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

n_embd = 128
gelu_mlp = GELUMlp(n_embd)
swiglu   = SwiGLU(n_embd)

gelu_params   = sum(p.numel() for p in gelu_mlp.parameters())
swiglu_params = sum(p.numel() for p in swiglu.parameters())
swiglu_hidden = swiglu.w1.out_features

print(f"GELU MLP   — hidden: {4*n_embd}, params: {gelu_params:,}")
print(f"SwiGLU MLP — hidden: {swiglu_hidden}, params: {swiglu_params:,}")
print(f"SwiGLU uses ~{swiglu_params/gelu_params:.2f}× as many parameters as GELU MLP")
print(f"(SwiGLU has 3 weight matrices, GELU has 2 — balanced by smaller hidden dim)")

# %% [markdown]
# ## Summary
#
# In this notebook we built three modern transformer components from scratch, each
# taught against its predecessor:
#
# | Component | Key idea | Why it's better |
# |---|---|---|
# | **RMSNorm** | Divide by RMS only — no mean subtraction, no bias | Fewer ops, fewer params, same empirical quality |
# | **RoPE** | Rotate Q and K by position-dependent angles | Relative position by construction; no extra parameters |
# | **SwiGLU** | Two up-projections: gate × value, then down | Richer gating than a single fixed activation |
#
# These three components — along with the grouped-query attention from notebook 03 —
# are the defining architectural choices of the Llama family and most modern open-weight
# LLMs. The production implementations go into `model.py` in the next task.
#
# **Next:** notebook 05 will assemble all these pieces into a complete transformer block
# and train it on real text.
