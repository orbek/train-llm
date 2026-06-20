# %% [markdown]
# # 01 — Data & a Bag-of-Words Baseline
#
# Welcome to the first notebook! We are going to build the **simplest possible** next-word
# predictor and measure exactly how good — and how limited — it is.
#
# That model is called a **Bag-of-Words (BoW)** model. It looks at the previous few words
# and tries to guess what word comes next. It will turn out to be mediocre, and the
# reason it fails will be obvious: it completely ignores the *order* of words. Measuring
# that failure precisely is the whole point — everything that comes later in the course
# (embeddings, attention, transformers) is a direct answer to the shortcomings we uncover
# here.
#
# By the end of this notebook you will have:
# - Downloaded and inspected a real text dataset (Shakespeare).
# - Turned raw text into numbers a neural network can process.
# - Built a tiny linear model and trained it.
# - Measured its quality with a metric called **perplexity**.
# - Proved, with a concrete assertion, that word order is invisible to the model.

# %%
import os
import re
import json
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)

# %% [markdown]
# ## Step 1 — Getting the data
#
# We need text to learn from. We will use the **TinyShakespeare** dataset — about
# 1 MB of William Shakespeare's plays concatenated together. It is a classic benchmark
# in language-model research because it is small enough to train on a laptop yet rich
# enough to expose real language patterns.
#
# The cell below downloads the file once and saves it to `data/shakespeare.txt`. If you
# run the notebook a second time, it skips the download and reads from the cached copy.
# After loading, we print the first 250 characters so you can see what the raw text
# looks like.

# %%
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"

if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    print("Downloading TinyShakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("Characters:", len(text))
print("----- sample -----")
print(text[:250])

# %% [markdown]
# ## Step 2 — Turning text into numbers (tokenization)
#
# A neural network can't read letters or words — it only does math on numbers. So our
# first job is to convert the raw Shakespeare text into a list of integers. That
# translation process is called **tokenization**, and each piece we translate is called
# a **token**.
#
# Because this notebook is about the Bag-of-**Words**, we treat each word (and each
# punctuation mark) as one token. We need two things:
# 1. A rule for chopping text into words (a *tokenizer function*).
# 2. A two-way dictionary mapping each unique word ↔ a unique id number (a *vocabulary*).
#
# ### The chopping rule: `tokenize_words`
#
# Before we can number words, we need to split a string into a list of word-sized
# pieces. That is what `tokenize_words` does. Let's walk through it line by line.
#
# - `s.lower()` makes every character lowercase before we do anything else.  This means
#   `"The"` and `"the"` will be treated as the same word, so we don't waste vocabulary
#   slots on capitalization variants.
# - `re.findall(pattern, s.lower())` scans the lowercased string from left to right and
#   returns every substring that matches the pattern — as a Python list.
# - The **pattern** itself has two halves joined by `|` (which means "or" in a
#   *regular expression*):
#   - `[a-z]+` — "one or more letters in a row". This grabs whole words like `hello`
#     or `wherefore`.
#   - `[^a-z\s]` — "a single character that is not a letter (`[^a-z]`) and not a
#     whitespace character (`\s`)". This grabs standalone punctuation marks such as
#     `,`, `.`, `!`, `?`, `;`, and `'`.
# - Spaces are not matched by either half, so they are silently dropped (they are just
#   separators between tokens, not tokens themselves).
#
# Concrete example:
# ```
# tokenize_words("Hello, world!")
# # → ['hello', ',', 'world', '!']
# ```
# Punctuation is kept as its own token because it carries real meaning (a `?` changes
# the interpretation of a sentence). Spaces are gone — we don't need them once the list
# is built.

# %%
def tokenize_words(s: str) -> list[str]:
    # words and standalone punctuation become tokens
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

# %% [markdown]
# ### The word ↔ number dictionary: `WordTokenizer`
#
# Now that we can chop text into word-pieces, we need to assign each unique piece an
# integer id so the model can work with numbers. That is what `WordTokenizer` does.
#
# We also cap the vocabulary at a manageable size. The full Shakespeare text contains
# around 25,000 distinct words — many of them rare (`"prithee"`, `"forsooth"`, …). A
# model whose vocabulary is that large would need many more parameters and much more
# training. Instead we keep only the **most common 1,999 words** and introduce a special
# token called `<unk>` (short for "unknown") that stands in for *any* word we dropped.
# That gives us a tidy vocabulary of exactly 2,000 entries.
#
# Here is what each part of the class does:
#
# **`__init__` (constructor) — build the vocabulary**
# 1. Run `tokenize_words` on the full corpus to get every token.
# 2. Use Python's `Counter` to count how often each unique token appears.
# 3. Take the 1,999 most frequent words (`max_vocab - 1` because slot 0 is reserved).
# 4. Prepend `<unk>` at position 0. This guarantees that any unseen word maps to id 0.
# 5. Build `itos` ("**i**nteger-**to**-**s**tring") — a plain list where `itos[3]`
#    gives you the word at index 3.
# 6. Build `stoi` ("**s**tring-**to**-**i**nteger") — the reverse dictionary where
#    `stoi["the"]` gives you the integer id for `"the"`.
#
# **`encode` — text → list of ids**
# Call `tokenize_words` on the input, then look up each token in `stoi`. Any token not
# in the vocabulary gets the `<unk>` id (0).
#
# **`decode` — list of ids → text**
# Look up each id in `itos` and join the results with spaces. Note: the spaces won't
# perfectly reconstruct the original punctuation spacing, but it gives a readable
# approximation.
#
# Concrete example:
# ```
# tok.encode("to be or not to be")
# # → [12, 7, 43, 91, 12, 7]   (ids will vary by run)
# tok.decode([12, 7, 43, 91, 12, 7])
# # → "to be or not to be"
# ```

# %%
VOCAB_SIZE = 2000  # cap for a tractable BoW model

class WordTokenizer:
    def __init__(self, corpus: str, max_vocab: int = VOCAB_SIZE):
        counts = Counter(tokenize_words(corpus))
        most_common = [w for w, _ in counts.most_common(max_vocab - 1)]
        self.itos = ["<unk>"] + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, s: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in tokenize_words(s)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

tok = WordTokenizer(text)
print("Vocab size:", tok.vocab_size)

# %% [markdown]
# ### Sanity check: round-trip encoding
#
# Before we trust the tokenizer, let's verify that encoding a phrase and then decoding
# it gives back exactly what we started with. We call this a **round-trip check**.
#
# We pick `"to be or not to be"` because every word in it is common enough to be in the
# vocabulary (none should map to `<unk>`). The `assert` statement will crash loudly if
# the round-trip fails — a useful safety net that catches bugs early.

# %%
sample = "to be or not to be"
ids = tok.encode(sample)
assert tok.decode(ids) == sample, tok.decode(ids)
print("round-trip ok:", ids, "->", tok.decode(ids))

# %% [markdown]
# ## Step 3 — Train / validation split
#
# We have one long sequence of token ids. We need to split it into two parts:
#
# - **Training set** — what the model learns from (the first 90%).
# - **Validation set** — a held-out portion the model never trains on (the last 10%).
#
# Evaluating on data the model has *never seen during training* is the honest way to
# measure whether it has learned general patterns or just memorised the training text.
# This is one of the most important concepts in all of machine learning:
# **never evaluate on your training data**.
#
# We use the last 10% (rather than a random shuffle) because the text is a continuous
# stream — randomly shuffling would mix training and validation examples in a way that
# could let the model "peek" at nearby context from the validation set.

# %%
ids_all = tok.encode(text)
n = len(ids_all)
split = int(0.9 * n)
train_ids = ids_all[:split]
val_ids = ids_all[split:]
print(f"tokens: {n} (train {len(train_ids)}, val {len(val_ids)})")

# %% [markdown]
# ## Step 4 — The Bag-of-Words representation
#
# Here is the core idea of this model: given the previous `CONTEXT` words, predict the
# next word.
#
# But instead of feeding those `CONTEXT` words to the model in order (word 1, word 2,
# …, word 8), we throw away the order information entirely. We represent the context
# window as a single **count vector** of length `vocab_size`. Each slot in the vector
# says: "how many times did word #i appear in the context window?" The *positions* of
# those words are discarded.
#
# Why would anyone do this? Historically, count vectors were easy to compute and gave
# surprisingly decent results for tasks like spam detection and sentiment analysis.
# Here we use it deliberately as a *controlled failure* — we want to see exactly how
# much accuracy we lose by throwing order away.
#
# ### `make_bow_dataset`
#
# This function slides a window of width `context` across the token id list and builds
# training examples:
#
# - `window = ids[i : i + context]` — the `context` ids just before position `i+context`.
# - `vec = torch.zeros(vocab_size)` — start with all zeros.
# - For each token id `w` in the window, increment `vec[w]` by 1. After the loop,
#   `vec[w]` holds the count of how many times word `w` appeared in the window.
# - `ys.append(ids[i + context])` — the *target* is the very next token after the window.
#
# The `step` variable sub-samples the data for speed: if the dataset is very long we
# don't use every single overlapping window, just every `step`-th one, giving us at
# most `limit` examples.
#
# The result is:
# - `X`: shape `(N, vocab_size)` — N count vectors, one per example.
# - `Y`: shape `(N,)` — N target word ids, one per example.

# %%
CONTEXT = 8

def make_bow_dataset(ids: list[int], context: int, vocab_size: int, limit: int):
    # returns X: (N, vocab_size) float counts, Y: (N,) next-word ids
    xs, ys = [], []
    step = max(1, (len(ids) - context - 1) // limit)  # subsample for speed
    for i in range(0, len(ids) - context - 1, step):
        window = ids[i:i + context]
        vec = torch.zeros(vocab_size)
        for w in window:
            vec[w] += 1.0
        xs.append(vec)
        ys.append(ids[i + context])
    return torch.stack(xs), torch.tensor(ys)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

Xtr, Ytr = make_bow_dataset(train_ids, CONTEXT, tok.vocab_size, limit=20000)
Xva, Yva = make_bow_dataset(val_ids, CONTEXT, tok.vocab_size, limit=4000)
Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)
print("BoW train X:", tuple(Xtr.shape), "val X:", tuple(Xva.shape))

# %% [markdown]
# ## Step 5 — The linear model
#
# With our count vectors ready, we need a model that can turn them into predictions.
# We use the simplest learnable model that exists: a **single linear layer**.
#
# ### What is a linear layer?
#
# A linear layer (also called a **fully-connected** or **dense** layer) computes:
#
# ```
# output = input @ W + b
# ```
#
# where `W` is a matrix of learnable **weights** and `b` is a vector of learnable
# **biases**. Here, `input` is our count vector of size `vocab_size` and `output` is
# also of size `vocab_size` — one raw score per word.
#
# Those raw output scores are called **logits**. The word comes from "log-odds": they
# are not probabilities yet (they can be negative or greater than 1), but they encode
# the model's raw belief about how likely each word is. Higher logit = more likely.
# Later, the cross-entropy loss (described below) internally converts logits to
# probabilities using the **softmax** function.
#
# This architecture — a single linear layer from input features to class scores — is
# also known as **multinomial logistic regression**. It is about as simple as a
# supervised learning model can get.
#
# ### `BoWModel`
#
# Our model has exactly one layer: `nn.Linear(vocab_size, vocab_size)`. The `forward`
# method just passes the input through that layer and returns the logits. No activation
# function, no hidden layers — just a single matrix multiply.
#
# The total number of parameters is `vocab_size × vocab_size + vocab_size`
# (weights + biases). With `vocab_size = 2000` that is about 4 million parameters —
# not huge, but enough to learn something.

# %%
class BoWModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        return self.fc(x)

bow = BoWModel(tok.vocab_size).to(device)
print("BoW params:", sum(p.numel() for p in bow.parameters()))

# %% [markdown]
# ## Step 6 — Training the model
#
# Training a neural network means adjusting its weights so that it makes better
# predictions. Let's define the key concepts before looking at the code.
#
# ### Cross-entropy loss
#
# We need a number that says "how wrong is the model right now?" That number is the
# **loss** (also called the **cost**). For classification problems like predicting the
# next word, the standard choice is **cross-entropy loss**.
#
# Intuitively: the model outputs a probability distribution over the entire vocabulary
# (via softmax applied to the logits). Cross-entropy measures how much probability the
# model assigned to the *correct* word. If the model was confident and right, the loss
# is low. If the model was confident and wrong, the loss is high.
#
# Formally, for a single example with correct word id `y` and logits `z`:
# ```
# loss = -log( softmax(z)[y] )
# ```
# `nn.functional.cross_entropy` handles the softmax + log + negation in one numerically
# stable call — you just pass raw logits and the target ids.
#
# ### AdamW optimizer
#
# After computing the loss, we compute gradients (via `loss.backward()`) — numbers that
# say "which direction should each weight move to reduce the loss?" Then the
# **optimizer** actually updates the weights.
#
# **AdamW** is a popular, well-tuned variant of gradient descent. It adapts the
# learning rate for each parameter individually and includes **weight decay** (a small
# penalty that keeps weights from growing too large, which helps prevent overfitting).
# `lr=1e-2` (0.01) is the **learning rate** — it controls the step size of each update.
#
# ### Epochs
#
# One **epoch** is one full pass over the training data. We train for `EPOCHS = 60`
# epochs. Each iteration: compute logits → compute loss → compute gradients → update
# weights.
#
# ### `evaluate`
#
# This helper function computes the loss on a given dataset *without* updating any
# weights. We call `model.eval()` to disable dropout and batch-norm updates (not
# relevant here, but good practice), and wrap the forward pass in `torch.no_grad()` to
# skip gradient tracking (faster, less memory). After evaluating we call `model.train()`
# to put the model back in training mode.

# %%
def evaluate(model, X, Y) -> float:
    model.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(X), Y)
    model.train()
    return float(loss)

opt = torch.optim.AdamW(bow.parameters(), lr=1e-2)
EPOCHS = 60
losses = []
for epoch in range(EPOCHS):
    logits = bow(Xtr)
    loss = nn.functional.cross_entropy(logits, Ytr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

val_loss = evaluate(bow, Xva, Yva)
bow_val_ppl = float(torch.exp(torch.tensor(val_loss)))
print(f"BoW final train loss {losses[-1]:.3f} | val loss {val_loss:.3f} | val perplexity {bow_val_ppl:.1f}")

# %%
os.makedirs("assets", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("BoW training loss")
plt.tight_layout()
plt.savefig("assets/01_bow_loss.png", dpi=120)
plt.show()

# %% [markdown]
# ## Step 7 — Interpreting the result: perplexity
#
# The validation loss is a useful number but it is hard to interpret intuitively.
# **Perplexity** is a more human-readable version of the same information.
#
# Informally: perplexity is roughly *how many words the model is "choosing between" on
# average* when making each prediction. A perplexity of 264 means the model is about
# as uncertain as if it were randomly picking from 264 equally likely candidates —
# even though the vocabulary has 2,000 words.
#
# Formally:
# ```
# perplexity = exp(cross_entropy_loss)
# ```
#
# - A **perfect** model that always predicts the correct word with 100% confidence
#   would have perplexity = 1.
# - A **random** model that spreads probability equally over all 2,000 words would
#   have perplexity = 2,000.
# - Our BoW model lands somewhere in between — lower is better.
#
# The perplexity around **264** tells us our BoW model has learned *something* (it is
# well below random), but it is still far from good. The next notebooks will try to
# push this number down substantially by using richer representations.

# %% [markdown]
# ## Step 8 — The punchline: order is gone
#
# Here we prove — not just claim — that the Bag-of-Words representation is blind to
# word order.
#
# The phrase `"the king is dead"` and its reversal `"dead is king the"` contain
# exactly the same words. Therefore their BoW count vectors are **identical**. And
# because the model is just `output = input @ W + b`, identical inputs produce
# identical outputs — the model gives the *same next-word prediction* for both phrases,
# even though they mean completely different things.
#
# This is not a bug in our implementation — it is a fundamental property of the
# Bag-of-Words approach. Any model that collapses a sequence into an unordered count
# loses all positional information by design.
#
# The two `assert` statements below check this formally:
# - First assert: the count vectors are equal (same word counts, same vector).
# - Second assert: the model output logits are equal (same predictions).
#
# This measured failure is exactly why we need sequence-aware models. Everything that
# follows in this course — recurrent networks, self-attention, transformers — is an
# answer to the question "how do we preserve word order?"

# %%
phrase_a = tok.encode("the king is dead")
phrase_b = list(reversed(phrase_a))

def bow_vec(ids):
    v = torch.zeros(tok.vocab_size, device=device)
    for w in ids:
        v[w] += 1.0
    return v

va, vb = bow_vec(phrase_a), bow_vec(phrase_b)
assert torch.equal(va, vb), "BoW vectors should be identical regardless of order"

with torch.no_grad():
    pa = bow(va.unsqueeze(0))
    pb = bow(vb.unsqueeze(0))
assert torch.allclose(pa, pb), "predictions should be identical regardless of order"
print("Confirmed: '", tok.decode(phrase_a), "' and '", tok.decode(phrase_b),
      "' give identical predictions. Order is invisible to BoW.")

# %% [markdown]
# ## Step 9 — Save the baseline metric
#
# We record the validation perplexity (and a few other numbers) to a JSON file. The
# next notebook will load this file to compare its model against ours. This is a simple
# form of **experiment tracking** — keeping a record of what each model achieved so we
# can tell whether a new approach is actually better.
#
# The numbers saved:
# - `bow_val_perplexity` — the headline metric we want future models to beat.
# - `bow_params` — parameter count, so we can make fair comparisons.
# - `context` — the window size used, so later notebooks use the same setting.
# - `vocab_size` — the vocabulary size, same reason.

# %%
os.makedirs("assets", exist_ok=True)
with open("assets/phase1_metrics.json", "w") as f:
    json.dump({"bow_val_perplexity": bow_val_ppl,
               "bow_params": sum(p.numel() for p in bow.parameters()),
               "context": CONTEXT, "vocab_size": tok.vocab_size}, f, indent=2)
print("Saved baseline:", bow_val_ppl)
