# %% [markdown]
# # 08 — Hyperparameter Tuning: LR Schedule & Small Sweeps
#
# In notebooks 01–07 we built a GPT from scratch, trained it, and evaluated it.
# Now we take a step back and ask: how do we choose *good* settings?
#
# This notebook teaches two ideas:
#
# 1. **Learning-rate scheduling** — instead of a fixed learning rate, we ramp it up
#    at the start and then decay it smoothly. This often gives significantly better
#    final loss for free.
# 2. **Hyperparameter sweeps** — systematically trying a small grid of values to see
#    which ones work best.
#
# Both ideas are standard practice in every modern LLM training run.
#
# > **Note on speed:** every training run in this notebook is deliberately tiny —
# > ~300 iterations on the nano model — so the whole notebook finishes in about
# > 1–2 minutes. Real sweeps run for hours or days; the patterns here are the same.

# %% [markdown]
# ## Working directory
#
# Jupyter runs notebooks from the `notebooks/` folder. The cell below walks up the
# directory tree until it finds the project root (marked by `requirements.txt`) and
# changes the working directory there, so `from model import ...` resolves correctly
# no matter how you launched Jupyter.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print("Working directory:", os.getcwd())

# %% [markdown]
# ## Imports, device, and seed

# %%
import math
import time
import torch
import matplotlib.pyplot as plt

from model import GPT, NANO_CONFIG, get_device

torch.manual_seed(1337)
device = get_device()
print("Using device:", device)

os.makedirs("assets", exist_ok=True)

# %% [markdown]
# ---
# ## Part 1 — Char-level data pipeline
#
# We reuse the same character-level data pipeline from notebooks 01 and 06.
# The Shakespeare text is loaded (or downloaded if missing), encoded as integers,
# and split 90/10 into train and validation sets.
#
# **Vocabulary:** the set of unique characters in the text — roughly 65 for Shakespeare.
#
# **Encoding:** each character → a unique integer index.
#
# **Block size:** `block_size` consecutive tokens form one training example.
# The nano model uses `block_size=64` (much smaller than the full model's 256).

# %%
DATA_PATH = "data/shakespeare.txt"

if not os.path.exists(DATA_PATH):
    import urllib.request
    os.makedirs("data", exist_ok=True)
    print("Downloading shakespeare.txt …")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH,
    )
    print("Downloaded.")

with open(DATA_PATH, "r") as f:
    text = f.read()

# Use a subset of the text to keep sweep training fast
SUBSET = 200_000            # characters (~19% of full corpus)
text = text[:SUBSET]

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Text subset  : {len(text):,} chars")
print(f"Vocab size   : {vocab_size} unique characters")
print(f"Train tokens : {len(train_data):,}  |  Val tokens: {len(val_data):,}")

BLOCK_SIZE  = NANO_CONFIG.block_size   # 64
BATCH_SIZE  = 32

def get_batch(split):
    """Sample a random (input, target) batch from train or val data."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(f"Batch shape  : x={tuple(xb.shape)}, y={tuple(yb.shape)}")

# %% [markdown]
# ---
# ## Part 2 — What is a hyperparameter?
#
# A **hyperparameter** is any setting you choose *before* training that is not itself
# learned by gradient descent.  Examples:
#
# | Hyperparameter | Typical range | Effect |
# |---|---|---|
# | Learning rate (`lr`) | 1e-4 … 1e-2 | Size of each gradient step |
# | Batch size | 16 … 2048 | Gradient noise vs. memory |
# | Context length (`block_size`) | 64 … 8192 | Memory vs. long-range understanding |
# | Model depth / width | 2–96 layers | Capacity |
#
# In this notebook we focus on the **learning rate**, the single most important
# hyperparameter for training neural networks.

# %% [markdown]
# ---
# ## Part 3 — Learning-rate scheduling
#
# ### Why not a fixed learning rate?
#
# - **Too high at the start** → loss spikes, training can diverge.
# - **Too high at the end** → the optimizer keeps overshooting the minimum; it never
#   fully converges.
# - **Too low throughout** → training is needlessly slow.
#
# The solution: start small (so early updates are stable), ramp up to the peak
# learning rate, then slowly decay back down so the model can settle into a sharp
# minimum.
#
# ### Linear warmup + cosine decay
#
# The most popular schedule in LLM training (used in GPT-3, LLaMA, etc.) has
# three phases:
#
# 1. **Linear warmup** (0 → `warmup_iters`): the learning rate grows linearly from
#    nearly 0 to `max_lr`.
# 2. **Cosine decay** (`warmup_iters` → `lr_decay_iters`): the learning rate follows
#    the smooth shape of a cosine curve, falling from `max_lr` down to `min_lr`.
# 3. **Floor** (beyond `lr_decay_iters`): the learning rate stays at `min_lr` forever.
#
# The **cosine** function is used because it starts decaying slowly (so the model can
# explore), then speeds up in the middle, then slows again near the end (so the model
# can settle).  It is smoother than a step decay.

# %%
def get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    """
    Learning-rate schedule: linear warmup → cosine decay → floor.

    Parameters
    ----------
    it              : current iteration number (0-based)
    warmup_iters    : number of warmup steps
    lr_decay_iters  : iteration at which decay ends (floor begins)
    max_lr          : peak learning rate (reached at end of warmup)
    min_lr          : minimum learning rate (floor)

    Returns
    -------
    float : the learning rate for iteration `it`
    """
    # Phase 1 — linear warmup
    # At it=0 we return max_lr/warmup_iters (nearly 0 for large warmup_iters).
    # At it=warmup_iters-1 we return max_lr * warmup_iters/warmup_iters = max_lr.
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters

    # Phase 3 — flat floor (after decay is complete)
    if it > lr_decay_iters:
        return min_lr

    # Phase 2 — cosine decay
    # ratio goes from 0 (start of decay) to 1 (end of decay).
    # math.cos(pi * ratio) goes from 1 → -1, so coeff goes from 1 → 0.
    # Interpolating: min_lr + 1*(max_lr - min_lr) → min_lr + 0*(max_lr - min_lr)
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1 → 0
    return min_lr + coeff * (max_lr - min_lr)

# %% [markdown]
# ### Quick sanity check
#
# Before plotting, let's verify the three phases with asserts.
# **Assert** means "crash immediately if this is false" — they double as
# runnable documentation of our expectations.

# %%
warm, decay, hi, lo = 100, 1000, 1e-3, 1e-4

# 1. Warmup: lr at it=0 is strictly less than lr at it=warmup (the peak)
assert get_lr(0,        warm, decay, hi, lo) < get_lr(warm, warm, decay, hi, lo), \
    "LR should increase during warmup"

# 2. Floor: after lr_decay_iters the lr stays at min_lr (within floating-point noise)
assert abs(get_lr(decay + 50, warm, decay, hi, lo) - lo) < 1e-12, \
    "LR should be exactly min_lr past the decay horizon"

# 3. Peak: lr at end of warmup should not exceed max_lr (it equals it, modulo float)
assert get_lr(warm, warm, decay, hi, lo) <= hi + 1e-9, \
    "LR should not exceed max_lr"

print("All get_lr asserts PASSED")
print(f"  lr at it=0        : {get_lr(0,         warm, decay, hi, lo):.6f}  (start of warmup)")
print(f"  lr at it=warmup   : {get_lr(warm,       warm, decay, hi, lo):.6f}  (peak, = max_lr)")
print(f"  lr at it=midpoint : {get_lr(550,        warm, decay, hi, lo):.6f}  (mid cosine decay)")
print(f"  lr at it=decay    : {get_lr(decay,      warm, decay, hi, lo):.6f}  (end of decay)")
print(f"  lr at it=decay+50 : {get_lr(decay + 50, warm, decay, hi, lo):.6f}  (floor)")

# %% [markdown]
# ### Visualising the schedule
#
# The plot below shows what `get_lr` returns at every iteration for a typical
# 300-iteration nano run.  Notice:
#
# - The steep linear ramp in the first 30 iterations (warmup).
# - The smooth S-shaped cosine fall from iteration 30 to 270.
# - The flat floor after iteration 270.

# %%
TOTAL_ITERS  = 300
WARMUP_ITERS = 30
DECAY_ITERS  = 270
MAX_LR       = 3e-3
MIN_LR       = 3e-4

iters = list(range(TOTAL_ITERS + 30))   # a bit past decay to show the floor
lrs   = [get_lr(i, WARMUP_ITERS, DECAY_ITERS, MAX_LR, MIN_LR) for i in iters]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(iters, lrs, color="#2563EB", linewidth=2)
ax.axvline(WARMUP_ITERS, color="#DC2626", linestyle="--", alpha=0.7, label=f"end of warmup (it={WARMUP_ITERS})")
ax.axvline(DECAY_ITERS,  color="#16A34A", linestyle="--", alpha=0.7, label=f"end of decay  (it={DECAY_ITERS})")
ax.axhline(MAX_LR, color="gray", linestyle=":", alpha=0.5, label=f"max_lr = {MAX_LR:.0e}")
ax.axhline(MIN_LR, color="gray", linestyle="-.", alpha=0.5, label=f"min_lr = {MIN_LR:.0e}")
ax.set_xlabel("Iteration")
ax.set_ylabel("Learning rate")
ax.set_title("Linear warmup + cosine decay schedule")
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("assets/08_lr_schedule.png", dpi=120)
plt.show()
plt.close()
print("Saved assets/08_lr_schedule.png")

# %% [markdown]
# ---
# ## Part 4 — Learning-rate sweep
#
# ### What is a sweep?
#
# A **sweep** (or grid search) means training the same model multiple times, each time
# with a different value of one hyperparameter, and comparing the results.
# It is the most direct way to answer "which learning rate works best for this task?"
#
# ### What is overfitting?
#
# **Overfitting** happens when a model memorises the training data instead of learning
# the underlying pattern.  Its train loss keeps falling but its *validation loss*
# (on held-out data) starts rising.  We always measure *validation loss* when comparing
# hyperparameter choices, because we care about generalisation.
#
# ### The sweep
#
# We try four learning rates: `[1e-2, 3e-3, 1e-3, 3e-4]`.
#
# - `1e-2` (very high) — likely to diverge or be unstable.
# - `3e-3` (high) — often a sweet spot for small models.
# - `1e-3` (medium) — the classic safe default.
# - `3e-4` (low) — stable but potentially slow; may underfit in only 300 iters.
#
# Each run uses the warmup+cosine schedule from Part 3 with `max_lr` set to the
# candidate value and `min_lr = max_lr / 10`.
#
# > **Tiny sweeps are for learning.** In practice, sweeps run for thousands of
# > iterations with larger models.  The pattern — train, eval, compare — is identical.

# %%
@torch.no_grad()
def estimate_val_loss(model, eval_iters=30):
    """Average val loss over eval_iters random batches (no gradient)."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        xb, yb = get_batch("val")
        _, loss, _ = model(xb, targets=yb)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def train_nano(lr, iters=300):
    """
    Train a fresh nano GPT for `iters` steps with peak learning rate `lr`.

    Uses linear warmup (10% of iters) + cosine decay + AdamW.
    Returns the final validation loss.
    """
    # Build a fresh model — important! each sweep run starts from scratch.
    # Use dataclasses.replace to make a COPY of NANO_CONFIG with our vocab size;
    # assigning `cfg = NANO_CONFIG` would alias (and mutate) the shared default.
    import dataclasses
    cfg = dataclasses.replace(NANO_CONFIG, vocab_size=vocab_size)
    model = GPT(cfg).to(device)

    warmup = max(1, iters // 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    t0 = time.time()
    for it in range(iters):
        # Update learning rate according to the schedule
        current_lr = get_lr(it, warmup, iters, lr, lr / 10)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        xb, yb = get_batch("train")
        _, loss, _ = model(xb, targets=yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = estimate_val_loss(model)
    elapsed = time.time() - t0
    print(f"  lr={lr:.0e}  |  val_loss={val_loss:.4f}  |  {elapsed:.1f}s")
    return val_loss

# %%
# Run the sweep — this takes ~1 minute on CPU, ~30s on MPS
LR_CANDIDATES = [1e-2, 3e-3, 1e-3, 3e-4]
val_losses = []

print("LR sweep (300 iters per run, nano model):")
for lr in LR_CANDIDATES:
    torch.manual_seed(1337)   # same init for fair comparison
    vl = train_nano(lr, iters=300)
    val_losses.append(vl)

print("\nResults:")
for lr, vl in zip(LR_CANDIDATES, val_losses):
    marker = " <-- best" if vl == min(val_losses) else ""
    print(f"  lr={lr:.0e}  val_loss={vl:.4f}{marker}")

best_lr  = LR_CANDIDATES[val_losses.index(min(val_losses))]
best_val = min(val_losses)
print(f"\nBest lr={best_lr:.0e}  val_loss={best_val:.4f}")

# %% [markdown]
# ### Assert: the learning rate actually matters
#
# If all four runs gave the same loss, the sweep would be useless — and would suggest
# a bug (e.g. all runs share the same model state).  We assert that at least some
# variation exists.

# %%
assert min(val_losses) < max(val_losses), \
    "lr sweep should produce different val losses — check that each run starts fresh"
print(f"Sweep assert PASSED  (best={min(val_losses):.4f}, worst={max(val_losses):.4f})")

# %% [markdown]
# ### Visualising the sweep

# %%
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#DC2626" if v == min(val_losses) else "#93C5FD" for v in val_losses]
bars = ax.bar([f"{lr:.0e}" for lr in LR_CANDIDATES], val_losses, color=colors, edgecolor="white")

# Add value labels on each bar
for bar, v in zip(bars, val_losses):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Peak learning rate")
ax.set_ylabel("Final validation loss (lower = better)")
ax.set_title("LR sweep — 300-iter nano GPT (red = best)")
ax.set_ylim(0, max(val_losses) * 1.15)
plt.tight_layout()
plt.savefig("assets/08_lr_sweep.png", dpi=120)
plt.show()
plt.close()
print("Saved assets/08_lr_sweep.png")

# %% [markdown]
# ### Interpreting the results
#
# - **Very high lr (1e-2):** gradient steps are too large; the model may diverge
#   or oscillate, giving a high or erratic loss.
# - **Sweet spot (typically 3e-3 for nano):** the model converges quickly and finds
#   a good minimum in just 300 steps.
# - **Low lr (3e-4):** the model is stable but barely moves in 300 steps —
#   it *underfits* because it hasn't had enough effective gradient signal.
#
# The best lr depends on model size, batch size, and the number of iterations.
# Larger models typically need smaller learning rates.

# %% [markdown]
# ---
# ## Part 5 — Brief model-size note
#
# Learning rate is the most impactful single knob, but **model size** matters too.
# Here is a quick comparison between the nano config and a half-width variant.
#
# | Config | n_embd | Params | Notes |
# |--------|--------|--------|-------|
# | NANO_CONFIG (default) | 128 | ~0.2M | baseline |
# | half-nano | 64 | ~0.06M | fewer parameters → lower capacity |
#
# Rather than run a second training loop (which would add ~30s to the notebook),
# here is a parameter count comparison.

# %%
from copy import deepcopy
from model import GPTConfig

# Nano baseline
nano_cfg = deepcopy(NANO_CONFIG)
nano_cfg.vocab_size = vocab_size
nano_model = GPT(nano_cfg).to(device)
nano_params = nano_model.num_params()

# Half-width variant
half_cfg = GPTConfig(
    vocab_size  = vocab_size,
    block_size  = NANO_CONFIG.block_size,
    n_layer     = NANO_CONFIG.n_layer,
    n_head      = NANO_CONFIG.n_head,
    n_kv_head   = NANO_CONFIG.n_kv_head,
    n_embd      = NANO_CONFIG.n_embd // 2,   # 64 instead of 128
    dropout     = 0.0,
)
half_model = GPT(half_cfg).to(device)
half_params = half_model.num_params()

print(f"Nano     n_embd={nano_cfg.n_embd}  params={nano_params/1e3:.1f}K")
print(f"Half     n_embd={half_cfg.n_embd}   params={half_params/1e3:.1f}K")
print(f"Size ratio: {nano_params / half_params:.1f}x")

# Clean up to free memory
del nano_model, half_model

# %% [markdown]
# ### Which knobs matter most?
#
# In rough order of impact for small models:
#
# 1. **Learning rate** — the most sensitive; a 10× change can make or break training.
# 2. **Number of training steps** — more compute almost always helps.
# 3. **Model size** (depth × width) — more parameters → higher capacity, but also
#    slower training and more risk of overfitting on small data.
# 4. **Batch size** — larger batches give smoother gradient estimates but need a
#    proportionally higher lr (the *linear scaling rule*).
# 5. **Context length** — longer context helps tasks with long-range dependencies.
# 6. **Regularisation** (dropout, weight decay) — prevents overfitting when data is
#    limited relative to model size.

# %% [markdown]
# ---
# ## Part 6 — Practical tuning advice
#
# ### The tuning workflow
#
# 1. **Start with defaults:** `lr=3e-4`, AdamW, warmup 1–5% of total steps, cosine
#    decay to `lr/10`.  These work well for most small models.
# 2. **Do a cheap lr sweep** (like this notebook) with a fraction of the budget to
#    identify a good order of magnitude.
# 3. **Scale up gradually:** once you've found a good lr, increase steps and model
#    size — each time checking that val loss continues to decrease.
# 4. **Watch the train/val gap:** a small gap means the model is generalising; a large
#    and growing gap means it is **overfitting** — add dropout, reduce model size, or
#    get more data.
# 5. **Never tune on the test set:** always reserve a held-out split for final
#    evaluation only.
#
# ### Why these sweeps are tiny
#
# A real LLM training run might sweep over 10–50 hyperparameter combinations, each
# trained for tens of thousands of steps on hundreds of GPUs.  The patterns in this
# notebook — warmup, cosine decay, val-loss comparison — are identical; only the scale
# differs.
#
# > **Next step:** notebook 09 builds a **Byte-Pair Encoding (BPE)** tokenizer from
# > scratch — a smarter way to turn text into tokens than the character-level scheme
# > we have used so far.

# %%
print("Notebook 08 complete.")
print(f"  Sweep best lr   : {best_lr:.0e}  (val_loss={best_val:.4f})")
print(f"  assets/08_lr_schedule.png  written : {os.path.exists('assets/08_lr_schedule.png')}")
print(f"  assets/08_lr_sweep.png     written : {os.path.exists('assets/08_lr_sweep.png')}")
