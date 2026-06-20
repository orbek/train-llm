# %% [markdown]
# # 07 — Evaluation, Sampling Strategies & KV-Cache Generation
#
# In notebooks 03–06 we built a GPT from scratch and trained it on Shakespeare.
# Now we load the best checkpoint and explore what the trained model can do:
#
# - **Perplexity**: a principled measure of how well the model predicts text.
# - **Sampling strategies**: greedy, temperature, top-k, top-p (nucleus) — how each
#   setting balances quality against diversity.
# - **KV-cache**: a simple trick that makes generation much faster by reusing
#   previously computed attention keys and values.
#
# By the end you will understand how to evaluate and interact with a trained language
# model and why the KV-cache is the workhorse behind fast inference.

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
import time
import torch

from model import GPT, sample

torch.manual_seed(42)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("device:", device)

# %% [markdown]
# ---
# ## Part 1 — Load the checkpoint and rebuild the model
#
# Notebook 06 saved the best-validation-loss model to `checkpoints/model.pt`.
# The checkpoint is a plain Python dict with four keys:
#
# | Key | What's inside |
# |-----|---------------|
# | `model_state` | the model's weights (a `state_dict`) |
# | `config` | a `GPTConfig` dataclass with all hyperparameters |
# | `stoi` | character → integer mapping (the tokeniser) |
# | `itos` | integer → character mapping (the detokeniser) |
#
# We rebuild the model from `config` (so the architecture is identical to training),
# then load the weights with `load_state_dict`. Finally we define `encode`/`decode`
# helpers that convert between strings and integer tensors.

# %%
CKPT_PATH = "checkpoints/model.pt"

if not os.path.exists(CKPT_PATH):
    print("=" * 60)
    print("Checkpoint not found at", CKPT_PATH)
    print("Please run notebook 06 first to train and save the model.")
    print("=" * 60)
    sys.exit(1)

# weights_only=False is required because the checkpoint stores a GPTConfig dataclass
# (not just tensors). This is safe here because we wrote the file ourselves in
# notebook 06. Never load untrusted checkpoints with weights_only=False.
checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
config  = checkpoint["config"]
stoi    = checkpoint["stoi"]
itos    = checkpoint["itos"]

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

print(f"Loaded checkpoint from {CKPT_PATH}")
print(f"Config     : {config}")
print(f"Params     : {model.num_params()/1e6:.2f}M")
print(f"Vocab size : {config.vocab_size}  |  block_size: {config.block_size}")

# %% [markdown]
# ---
# ## Part 2 — Perplexity on the validation set
#
# ### What is perplexity?
#
# **Perplexity (PPL)** is the standard metric for language models. Intuitively it
# answers: *"On average, how many equally-likely choices does the model consider at
# each token?"*
#
# Mathematically it is the exponentiated average cross-entropy loss over a corpus:
#
# $$\text{PPL} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_i\right)$$
#
# where $\mathcal{L}_i = -\log p(\text{correct token}_i)$.
#
# - PPL = 1 → model is perfectly certain (impossible in practice).
# - PPL = vocab_size → model assigns equal probability to every token (random).
# - PPL drops during training as the model learns to predict text better.
#
# A val loss of ~1.47 (from training) corresponds to PPL = exp(1.47) ≈ 4.35:
# the model narrows down the next character to about 4 plausible candidates on average.

# %%
# Reconstruct the val split the same way notebook 06 did.
DATA_PATH = "data/shakespeare.txt"
with open(DATA_PATH, "r") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
val_data = data[n:]

print(f"Val tokens : {len(val_data):,}")

@torch.no_grad()
def estimate_val_loss(model, val_data, block_size, batch_size=32, eval_iters=100):
    """Average cross-entropy loss on the validation set."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack([val_data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix]).to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    return sum(losses) / len(losses)

val_loss = estimate_val_loss(model, val_data, config.block_size)
val_ppl  = torch.exp(torch.tensor(val_loss)).item()

print(f"\nVal loss   : {val_loss:.4f}")
print(f"Val PPL    : {val_ppl:.2f}")
print(f"(PPL = exp({val_loss:.4f}) — the model considers ~{val_ppl:.1f} candidates per character)")

# %% [markdown]
# ---
# ## Part 3 — Sampling strategies
#
# Once the model produces logits (raw scores over the vocabulary) we must decide
# *which* token to pick next. Different strategies trade off quality against diversity.
#
# ### Greedy decoding
# Always pick the highest-probability token. Deterministic and coherent, but tends to
# repeat itself and never "takes risks". Formally: `next = argmax(logits)`.
#
# ### Temperature scaling
# Divide the logits by a temperature scalar $T$ before converting to probabilities.
#
# - $T < 1$ (e.g. 0.8): makes the distribution *sharper* — the model prefers its
#   top choices more strongly. More focused, less creative.
# - $T = 1$: unchanged raw probabilities from the model.
# - $T > 1$: flatter distribution — the model picks unusual tokens more often.
#   More creative but more likely to be nonsensical.
# - $T \to 0$: equivalent to greedy.
#
# ### Top-k sampling
# Keep only the top-*k* most probable tokens; set all others to $-\infty$ (zero
# probability). Then sample from that truncated distribution. Prevents picking
# very unlikely tokens while still allowing diversity.
#
# ### Top-p (nucleus) sampling
# Keep the smallest set of top tokens whose cumulative probability exceeds *p*.
# The "nucleus" size adapts dynamically: when the model is confident, the nucleus
# is small; when uncertain, it grows. Often gives the best quality–diversity balance.

# %%
PROMPT = "ROMEO:"
MAX_NEW = 200

# Total length = prompt length + new tokens; must be ≤ block_size for cached gen.
prompt_ids = encode(PROMPT)
print(f"Prompt     : {repr(PROMPT)} ({len(prompt_ids)} tokens)")
print(f"Max total  : {len(prompt_ids) + MAX_NEW} tokens  (block_size = {config.block_size})")
assert len(prompt_ids) + MAX_NEW <= config.block_size, \
    "prompt + new tokens exceeds block_size — reduce MAX_NEW"

def make_seed():
    return torch.tensor([prompt_ids], dtype=torch.long, device=device)

def generate_and_print(label, temperature, top_k=None, top_p=None):
    torch.manual_seed(42)
    out = model.generate(make_seed(), MAX_NEW, temperature=temperature,
                         top_k=top_k, top_p=top_p, use_cache=False)
    text_out = decode(out[0].tolist())
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print('='*60)
    print(text_out)

# %%
generate_and_print("Greedy  (temperature=0.0)", temperature=0.0)
generate_and_print("Temperature 0.8", temperature=0.8)
generate_and_print("Top-k=50  (temperature=1.0)", temperature=1.0, top_k=50)
generate_and_print("Top-p=0.9  (nucleus, temperature=1.0)", temperature=1.0, top_p=0.9)

# %% [markdown]
# **What did you notice?**
#
# - **Greedy** produces the most mechanical-looking output. Every run is identical.
# - **Temperature=0.8** is slightly more varied; still quite coherent.
# - **Top-k=50** opens up many possibilities but guards against the very tail of
#   unlikely characters.
# - **Top-p=0.9** is the most adaptive: when the model is sure (e.g. right after a
#   capital letter starting a name) the nucleus is tiny; when uncertain the nucleus
#   expands. Many practitioners find nucleus sampling most natural-sounding.

# %% [markdown]
# ---
# ## Part 4 — KV-cache: correctness check + speed comparison
#
# ### What is a KV-cache?
#
# During generation the model processes one new token at a time. Naively it re-reads
# the *entire* context from scratch at each step, which is wasteful: the attention keys
# and values for tokens 1…t-1 are the same every time we generate token t.
#
# The **KV-cache** stores those keys and values after computing them. On the next step
# only the *new* token is processed; its query attends over the cached keys/values
# rather than recomputing them. This turns per-step cost from O(t²) to O(t) in
# attention and delivers significant wall-clock speedups on longer sequences.
#
# ### Correctness matters
#
# A KV-cache is just an optimisation — it must not change the *result*. Our first
# check is a byte-identical comparison between cached and non-cached greedy generation.
# If they differ, the cache implementation has a bug.

# %%
seed = torch.tensor([[stoi[c] for c in "ROMEO:"]], device=device)
NEW_TOKENS = 200   # prompt(6) + 200 = 206 ≤ block_size(256) ✓

# --- correctness ---
out_cached = model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=True)
out_plain  = model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=False)

assert torch.equal(out_cached, out_plain), "KV-cache must not change the output"
print("KV-cache correctness assert PASSED — cached output is byte-identical to non-cached")
print()
print(decode(out_cached[0].tolist()))

# %%
# --- speed comparison ---
REPS = 5

t0 = time.perf_counter()
for _ in range(REPS):
    model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=True)
t_cached = (time.perf_counter() - t0) / REPS

t0 = time.perf_counter()
for _ in range(REPS):
    model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=False)
t_plain = (time.perf_counter() - t0) / REPS

speedup = t_plain / t_cached
print(f"With KV-cache   : {t_cached*1000:.1f} ms/generation")
print(f"Without KV-cache: {t_plain*1000:.1f} ms/generation")
print(f"Speedup         : {speedup:.2f}x  (cached is {speedup:.2f}x faster)")

# %% [markdown]
# The KV-cache speedup grows with sequence length and model depth — on larger models
# and longer contexts the gain is dramatic. Here, with a 6-layer model and 200 new
# tokens, we already see a meaningful improvement.

# %% [markdown]
# ---
# ## What we built across Phase 2
#
# Over notebooks 03–07 we built a modern Llama-style transformer from scratch:
#
# | Notebook | Topic |
# |---|---|
# | 03 | Scaled dot-product attention, causal masking, multi-head |
# | 04 | Modern components: RoPE, RMSNorm, SwiGLU, GQA |
# | 05 | Full GPT assembly: pre-norm blocks, weight tying, KV-cache |
# | 06 | Training loop, AdamW, LR schedule, checkpointing |
# | 07 | Perplexity, sampling strategies, KV-cache demo |
#
# The model is a ~10M-parameter character-level transformer trained on Shakespeare —
# a scaled-down but architecturally faithful cousin of modern LLMs.
#
# ## What's next — Phase 3 teaser
#
# Phase 3 picks up where training ends:
#
# - **BPE tokenisation**: swap char-level for byte-pair encoding; larger vocabulary,
#   shorter sequences, dramatically better sample quality.
# - **Instruction fine-tuning / RLHF**: teach the model to follow instructions rather
#   than just complete text.
# - **Mixture of Experts (MoE)**: route each token through a subset of "expert" FFN
#   blocks — same compute budget, much larger effective parameter count.
# - **Scaling laws**: how loss, data, and compute interact as you scale up.
#
# The techniques you learned in Phase 2 are the foundation for all of these.
