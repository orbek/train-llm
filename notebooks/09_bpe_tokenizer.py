# %% [markdown]
# # 09 — BPE Tokenizer from Scratch
#
# In notebooks 01–08 we used **character-level tokenisation**: each token was a
# single character, giving us a vocabulary of ~65 symbols.  That is simple, but it
# has two big drawbacks:
#
# 1. **Long sequences.** Every word is split into many characters. "Shakespeare" → 11
#    tokens.  The model must use its limited context window to learn that those 11
#    characters *belong together*.
# 2. **Wasted capacity.** The model spends effort predicting individual letters instead
#    of learning higher-level patterns like grammar and meaning.
#
# **Byte-Pair Encoding (BPE)** is the classic fix.  It builds a *subword* vocabulary:
# tokens can be whole words, word-pieces like `"ing"` or `"the"`, or individual bytes.
# The key idea: find the most frequent pair of adjacent tokens and merge them into a
# single new token.  Repeat until you reach the desired vocabulary size.
#
# BPE is used in GPT-2, GPT-3, GPT-4, LLaMA, Mistral and most modern LLMs.
# In this notebook we build it entirely from scratch so you can see every step.
#
# **What you will learn:**
# - Why char-level is inefficient and how subword tokenisation helps.
# - The two core helpers: `get_stats` (count pairs) and `merge` (replace a pair).
# - How `BPETokenizer.train` / `.encode` / `.decode` work, step by step.
# - How to measure **compression ratio**: tokens saved vs. char-level.
# - How `tiktoken` (OpenAI's fast BPE library) compares to ours.
# - How to train the nano GPT on BPE tokens end-to-end.

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
import dataclasses
import time
import torch
import matplotlib.pyplot as plt

from model import GPT, NANO_CONFIG

torch.manual_seed(42)
# Auto-detect the best compute engine: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

os.makedirs("assets", exist_ok=True)

# %% [markdown]
# ---
# ## Part 1 — Load the Shakespeare data
#
# We load (or download) Shakespeare and take the first **200 000 characters** as our
# BPE training corpus.  That is large enough for interesting merges but small enough
# to train in a few seconds.
#
# We also set aside a **held-out chunk** (characters 200 000–210 000) that was *not*
# seen during BPE training, to measure compression on unseen text.

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
    full_text = f.read()

TRAIN_CHARS = 200_000
HELD_OUT_CHARS = 10_000

train_text   = full_text[:TRAIN_CHARS]
held_out     = full_text[TRAIN_CHARS : TRAIN_CHARS + HELD_OUT_CHARS]

print(f"Full corpus    : {len(full_text):,} chars")
print(f"BPE training   : {len(train_text):,} chars  (first {TRAIN_CHARS:,})")
print(f"Held-out chunk : {len(held_out):,} chars  (for compression comparison)")

# %% [markdown]
# ---
# ## Part 2 — Why subword tokenisation?
#
# ### The char-level problem
#
# Imagine encoding "the throne" at the character level:
#
#     t  h  e     t  h  r  o  n  e   → 10 tokens
#
# The model sees 10 separate tokens and must discover — purely from gradient descent —
# that those 10 characters often co-occur and form a meaningful unit.
#
# ### The word-level problem
#
# At the other extreme, we could use whole words as tokens.  That gives a very short
# sequence, but:
#
# - The vocabulary would need to contain *every* word (100 000+), making the embedding
#   table huge.
# - Any word not seen in training is **OOV** (out-of-vocabulary) and must be replaced
#   with a generic `<UNK>` token, losing information.
# - Morphological variants ("run", "runs", "running") each need separate entries.
#
# ### The BPE sweet spot
#
# BPE starts from bytes (0–255), so it is **never OOV** — any text can be encoded.
# It then learns to merge frequent pairs into single tokens:
#
# | Step | Sequence |
# |------|----------|
# | Start | `t h e   t h r o n e` |
# | Merge (`t`,`h`) → `th` | `th e   th r o n e` |
# | Merge (`th`,`e`) → `the` | `the   the r o n e` |
# | Merge (`the`,` `) → `the ` | `the  the r o n e` |
#
# Common substrings like `"the "`, `"ing"`, `"ou"`, `" th"` become single tokens.
# The result: **shorter sequences** (better compression) **and** a vocabulary that
# handles any input without OOV.

# %% [markdown]
# ---
# ## Part 3 — The two core helpers
#
# BPE needs just two simple functions.
#
# ### `get_stats(ids)` — count adjacent pairs
#
# Given a list of integer token IDs, count how many times each adjacent pair appears.
#
# **Tiny example:**
# ```
# ids = [84, 104, 101, 32, 84, 104, 114]  # "The The" as ASCII bytes
# stats = get_stats(ids)
# # → {(84,104): 2, (104,101): 1, (101,32): 1, (32,84): 1, (104,114): 1}
# # Pair (84,104) = ('T','h') appears twice — the most frequent!
# ```
#
# ### `merge(ids, pair, idx)` — replace every occurrence of a pair with a new token
#
# Walk through `ids` left-to-right.  Whenever you see `pair = (a, b)` at consecutive
# positions, replace both with the new token `idx`.  Otherwise copy the token as-is.
#
# **Tiny example:**
# ```
# ids  = [84, 104, 101, 32, 84, 104, 114]
# pair = (84, 104)   # the most frequent pair
# idx  = 256         # first new token ID (256 = first slot past the 256 bytes)
# result = merge(ids, pair, idx)
# # → [256, 101, 32, 256, 114]   — 7 tokens shrank to 5
# ```

# %%
def get_stats(ids):
    """Count frequency of every adjacent pair in `ids`."""
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the new token `idx`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# --- tiny worked example ---
print("=== get_stats worked example ===")
demo_ids = [84, 104, 101, 32, 84, 104, 114]   # 'T','h','e',' ','T','h','r'
demo_stats = get_stats(demo_ids)
print(f"  ids    : {demo_ids}")
print(f"  stats  : {dict(sorted(demo_stats.items(), key=lambda x: -x[1]))}")
best_pair = max(demo_stats, key=demo_stats.get)
print(f"  most frequent pair: {best_pair}  = {(chr(best_pair[0]), chr(best_pair[1]))}")

print()
print("=== merge worked example ===")
merged = merge(demo_ids, best_pair, 256)
print(f"  ids before: {demo_ids}  (len={len(demo_ids)})")
print(f"  ids after : {merged}   (len={len(merged)})  — token 256 replaces {best_pair}")

# %% [markdown]
# ---
# ## Part 4 — The BPETokenizer class
#
# Now we wrap the two helpers into a proper tokenizer with three methods:
#
# ### `train(text, vocab_size)`
# 1. Encode `text` as UTF-8 bytes → a list of integers 0–255.  These are the 256
#    **base tokens** (one per byte value).  Every possible byte is already in the vocab,
#    so we will never be OOV.
# 2. Repeat `vocab_size - 256` times:
#    - Find the most frequent adjacent pair with `get_stats`.
#    - Assign it a new token ID (256, 257, 258, …).
#    - Replace every occurrence of the pair in `ids` with the new token (`merge`).
#    - Record the merge rule in `self.merges` and the token's byte sequence in
#      `self.vocab`.
#
# ### `encode(text)`
# Convert text to UTF-8 bytes, then apply merge rules in the order they were learned
# (earlier merges have smaller IDs, so we prioritise by the merge ID via `min(...)`).
#
# ### `decode(ids)`
# Each token ID maps to a byte sequence in `self.vocab`.  Concatenate them and decode
# as UTF-8.  (`errors="replace"` handles the rare case of split multi-byte characters.)

# %%
class BPETokenizer:
    def __init__(self):
        self.merges = {}   # (int, int) → int  :  which pair maps to which new token
        self.vocab  = {}   # int → bytes        :  what bytes does each token represent

    def train(self, text, vocab_size):
        """Learn BPE merge rules from `text` until the vocabulary has `vocab_size` tokens."""
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"
        ids = list(text.encode("utf-8"))                   # start from raw bytes
        self.vocab  = {i: bytes([i]) for i in range(256)} # 256 base tokens
        self.merges = {}

        n_merges = vocab_size - 256
        for i in range(n_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)   # most frequent pair
            idx  = 256 + i                     # new token ID
            ids  = merge(ids, pair, idx)       # collapse all occurrences
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        print(f"Trained BPE: {len(self.vocab)} tokens  |  {len(self.merges)} merges learned")
        print(f"  sequence length: {len(text):,} chars → {len(ids):,} tokens  "
              f"({len(text)/len(ids):.2f} chars/token on train text)")

    def encode(self, text):
        """Encode `text` → list of integer token IDs."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Apply merges in training order: the pair whose merge ID is smallest
            # was learned first — apply it first (greedy left-to-right BPE).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break   # no more applicable merges
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """Decode list of token IDs → string."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

# %% [markdown]
# ---
# ## Part 5 — Train BPETokenizer on Shakespeare
#
# We train on the first 200 000 characters with `vocab_size=512`.
# That means 512 − 256 = **256 merge rules** will be learned.
# After training we inspect a sample of the learned merges to see what common
# substrings were discovered.

# %%
VOCAB_SIZE = 512
bpe = BPETokenizer()

t0 = time.time()
bpe.train(train_text, VOCAB_SIZE)
elapsed = time.time() - t0
print(f"Training time: {elapsed:.1f}s")

# %% [markdown]
# ### Inspecting learned merges
#
# The first merges learned are the most frequent pairs in the data.  For Shakespeare
# we expect common English substrings: spaces before letters, `"th"`, `"he"`, `"the"`,
# `"ou"`, `"er"`, etc.

# %%
print("First 15 learned merges (pair → new token):")
for k, (pair, idx) in enumerate(list(bpe.merges.items())[:15]):
    merged_str = bpe.vocab[idx].decode("utf-8", errors="replace")
    a_str = bpe.vocab[pair[0]].decode("utf-8", errors="replace")
    b_str = bpe.vocab[pair[1]].decode("utf-8", errors="replace")
    print(f"  merge {k+1:3d}: {repr(a_str)} + {repr(b_str)} → {repr(merged_str)}  (id={idx})")

# %% [markdown]
# ---
# ## Part 6 — Round-trip correctness assert
#
# Before using the tokenizer, we must verify that `decode(encode(text)) == text` for
# any text.  If this fails, we have a bug.
#
# This is the fundamental property every tokenizer must satisfy.

# %%
sample = "To be, or not to be, that is the question."

encoded = bpe.encode(sample)
decoded = bpe.decode(encoded)

print(f"Original : {repr(sample)}")
print(f"Encoded  : {encoded}")
print(f"Decoded  : {repr(decoded)}")
print(f"Token IDs: {len(encoded)} tokens for {len(sample)} chars  "
      f"({len(sample)/len(encoded):.2f} chars/token)")

assert decoded == sample, f"Round-trip FAILED!\n  got: {repr(decoded)}"
print("\nRound-trip assert PASSED ✓")

# %% [markdown]
# ---
# ## Part 7 — Compression comparison
#
# ### What is compression ratio?
#
# **Compression ratio** = number of char-level tokens ÷ number of BPE tokens.
# A ratio of 2.5 means BPE uses 2.5× fewer tokens than char-level to represent the
# same text — so the model's context window covers 2.5× more characters.
#
# We measure on the **held-out chunk** (text the tokenizer was *not* trained on) to
# get a realistic estimate of compression on new data.
#
# - **Char-level token count** = number of characters (each char is one token).
# - **BPE token count** = `len(bpe.encode(held_out_text))`.
#
# We also create a bar chart saved to `assets/09_compression.png`.

# %%
char_tokens = len(held_out)                # char-level: 1 token per character
bpe_tokens  = len(bpe.encode(held_out))   # BPE token count

compression_ratio = char_tokens / bpe_tokens
chars_per_token   = char_tokens / bpe_tokens   # same thing, different framing

print(f"Held-out chunk : {char_tokens:,} chars")
print(f"Char-level     : {char_tokens:,} tokens  (1 per char)")
print(f"BPE            : {bpe_tokens:,} tokens")
print(f"Compression    : {compression_ratio:.2f}×  ({chars_per_token:.2f} chars/token)")
print(f"Tokens saved   : {char_tokens - bpe_tokens:,}  ({100*(1 - bpe_tokens/char_tokens):.1f}% reduction)")

# Assert BPE actually compresses (our key correctness check)
assert bpe_tokens < char_tokens, \
    f"BPE should use fewer tokens than char-level, got {bpe_tokens} >= {char_tokens}"
print("\nCompression assert PASSED ✓  (BPE uses fewer tokens than char-level)")

# %% [markdown]
# ### Visualising the compression

# %%
fig, ax = plt.subplots(figsize=(7, 4))

labels = ["Char-level\n(1 token/char)", f"BPE\n(vocab={VOCAB_SIZE})"]
values = [char_tokens, bpe_tokens]
colors = ["#93C5FD", "#2563EB"]

bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, v + 50,
        f"{v:,}", ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_ylabel("Number of tokens (lower = better)")
ax.set_title(
    f"Compression: BPE vs char-level on {HELD_OUT_CHARS:,}-char held-out chunk\n"
    f"BPE compression ratio = {compression_ratio:.2f}×  ({chars_per_token:.2f} chars/token)"
)
ax.set_ylim(0, char_tokens * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("assets/09_compression.png", dpi=120)
plt.show()
plt.close()
print("Saved assets/09_compression.png")

# %% [markdown]
# ---
# ## Part 8 — tiktoken reference (optional)
#
# [tiktoken](https://github.com/openai/tiktoken) is OpenAI's production BPE
# implementation — the same tokenizer used by GPT-2, GPT-3.5, and GPT-4.
# It uses the same BPE algorithm we built above, but implemented in Rust for speed
# and trained on a much larger corpus with a much larger vocabulary (50 257 for GPT-2,
# 100 277 for GPT-4).
#
# We load the GPT-2 encoding and compare it to our from-scratch tokenizer on the same
# sample text.  Note how GPT-2's larger vocabulary achieves even better compression.
#
# This cell is **import-guarded**: if `tiktoken` is not installed, it prints a skip
# message and the notebook continues normally.

# %%
try:
    import tiktoken
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    gpt2_tokens_sample  = enc_gpt2.encode(sample)
    gpt2_tokens_heldout = enc_gpt2.encode(held_out)
    print(f"tiktoken GPT-2 vocab size      : {enc_gpt2.n_vocab:,}")
    print(f"Sample '{sample[:30]}...'")
    print(f"  our BPE   : {len(bpe.encode(sample))} tokens  (vocab={VOCAB_SIZE})")
    print(f"  GPT-2 BPE : {len(gpt2_tokens_sample)} tokens  (vocab={enc_gpt2.n_vocab:,})")
    print(f"Held-out chunk ({HELD_OUT_CHARS:,} chars)")
    print(f"  char-level: {char_tokens} tokens")
    print(f"  our BPE   : {bpe_tokens} tokens  (compression={compression_ratio:.2f}×)")
    print(f"  GPT-2 BPE : {len(gpt2_tokens_heldout)} tokens  "
          f"(compression={char_tokens/len(gpt2_tokens_heldout):.2f}×)")
except ImportError:
    print("tiktoken not installed — skipping reference comparison.")
    print("  Install with: pip install tiktoken>=0.5")

# %% [markdown]
# ### What this tells us
#
# Our from-scratch BPE and GPT-2's BPE use the **identical algorithm** — the only
# differences are:
#
# | | Ours | GPT-2 (tiktoken) |
# |---|---|---|
# | Vocabulary size | 512 | 50 257 |
# | Training corpus | Shakespeare (~200k chars) | Internet-scale data |
# | Implementation | Pure Python | Rust (much faster) |
#
# With a larger vocabulary, GPT-2 achieves a higher compression ratio because it can
# represent longer common substrings as single tokens.  LLaMA and newer models use
# SentencePiece BPE with similarly large vocabularies (32 000–128 000 tokens).

# %% [markdown]
# ---
# ## Part 9 — End-to-end: train the nano GPT on BPE tokens
#
# Now we use our BPETokenizer to tokenize the training data, then train the nano GPT
# on those BPE token IDs.
#
# **Why does BPE help the model?**  BPE tokens are denser — one BPE token represents
# more text than one character.  So the same `block_size` (context window) covers
# more actual text, giving the model more useful context to predict from.
#
# **Setup:**
# - Vocabulary size = 512 (our BPE vocab).
# - We override `NANO_CONFIG.vocab_size` to 512 using `dataclasses.replace` (a safe
#   copy — we never mutate the shared `NANO_CONFIG` object).
# - block_size = 64 (NANO_CONFIG default).
# - ~300 training iterations so the notebook stays under 2 minutes.
# - We record the loss at iteration 0 (before any learning) and at the final iteration,
#   then assert that the loss dropped.

# %%
# --- BPE-encode a subset of Shakespeare for training ---
BPE_SUBSET = full_text[:TRAIN_CHARS]   # same 200k chars we trained BPE on

print("Encoding text with BPE …")
t0 = time.time()
bpe_ids = bpe.encode(BPE_SUBSET)
print(f"Encoded {len(BPE_SUBSET):,} chars → {len(bpe_ids):,} BPE tokens  "
      f"({time.time()-t0:.1f}s)  compression={len(BPE_SUBSET)/len(bpe_ids):.2f}×")

# Train / val split (90/10)
bpe_data = torch.tensor(bpe_ids, dtype=torch.long)
n_split  = int(0.9 * len(bpe_data))
bpe_train = bpe_data[:n_split]
bpe_val   = bpe_data[n_split:]
print(f"BPE train: {len(bpe_train):,} tokens  |  val: {len(bpe_val):,} tokens")

# %%
# --- nano GPT config for BPE ---
BLOCK_SIZE = NANO_CONFIG.block_size   # 64
BATCH_SIZE = 32

bpe_cfg = dataclasses.replace(NANO_CONFIG, vocab_size=VOCAB_SIZE)
print(f"BPE nano config: vocab={bpe_cfg.vocab_size}, block={bpe_cfg.block_size}, "
      f"layers={bpe_cfg.n_layer}, embd={bpe_cfg.n_embd}")

def get_bpe_batch(split):
    """Sample a random (input, target) batch from BPE token data."""
    d = bpe_train if split == "train" else bpe_val
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

# %%
# --- training loop ---
TRAIN_ITERS = 300

torch.manual_seed(42)
nano_bpe = GPT(bpe_cfg).to(device)
optimizer = torch.optim.AdamW(nano_bpe.parameters(), lr=3e-3, weight_decay=0.1)

nano_bpe.train()
losses = []
t_start = time.time()

for it in range(TRAIN_ITERS):
    xb, yb = get_bpe_batch("train")
    _, loss, _ = nano_bpe(xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    losses.append(loss_val)

    if it == 0 or (it + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"  iter {it+1:3d}/{TRAIN_ITERS}  loss={loss_val:.4f}  ({elapsed:.1f}s)")

initial_loss = losses[0]
final_loss   = losses[-1]

print(f"\nInitial loss : {initial_loss:.4f}")
print(f"Final loss   : {final_loss:.4f}")
print(f"Loss drop    : {initial_loss - final_loss:.4f}")

# %% [markdown]
# ### Assert: loss must decrease
#
# After 300 gradient steps the model must be doing better than random — its loss must
# be strictly lower than at iteration 0.

# %%
assert final_loss < initial_loss, (
    f"Training should reduce loss: initial={initial_loss:.4f}, final={final_loss:.4f}"
)
print(f"Loss-drop assert PASSED ✓  ({initial_loss:.4f} → {final_loss:.4f})")

# %% [markdown]
# ---
# ## Summary
#
# | What we built | Key result |
# |---|---|
# | `get_stats` | count adjacent pair frequencies in O(n) |
# | `merge` | replace a pair with a new token in O(n) |
# | `BPETokenizer.train` | 256 merge rules learned from 200k chars |
# | Round-trip | `decode(encode(text)) == text` ✓ |
# | Compression | BPE uses fewer tokens than char-level (see numbers below) |
# | tiktoken reference | same algorithm, bigger vocab, Rust speed |
# | Nano-on-BPE | loss dropped over 300 iters (see numbers below) |
#
# (Markdown cells can't interpolate Python variables, so the exact figures are
# printed by the code cell that follows.)
#
# BPE is one of the most impactful engineering decisions in modern LLMs.  The same
# algorithm you just built is running inside every GPT model.

# %%
# Fill in the summary table with actual numbers
summary_ratio = len(held_out) / len(bpe.encode(held_out))
print("=== Notebook 09 complete ===")
print(f"  Vocab size          : {VOCAB_SIZE}")
print(f"  Merges learned      : {len(bpe.merges)}")
print(f"  Round-trip assert   : PASSED")
print(f"  Compression ratio   : {summary_ratio:.2f}×")
print(f"  BPE tokens saved    : {char_tokens - bpe_tokens:,}  ({100*(1-bpe_tokens/char_tokens):.1f}%)")
print(f"  Nano loss drop      : {initial_loss:.4f} → {final_loss:.4f}")
print(f"  assets/09_compression.png written: {os.path.exists('assets/09_compression.png')}")
