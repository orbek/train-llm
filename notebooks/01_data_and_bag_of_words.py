# %% [markdown]
# # 01 — Data & a Bag-of-Words Baseline
#
# Before any neural net, we build the simplest model that could possibly work:
# a **Bag-of-Words** next-word predictor. It will be mediocre — and crucially,
# it throws away word order. That measured failure motivates everything later.

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
# ## Get the data
#
# TinyShakespeare (~1MB). Downloaded once and cached in `data/`.

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
# ## Word-level tokenization
#
# "Bag of **words**" is about words, so here we tokenize on words. We lowercase,
# split words and punctuation, cap the vocabulary to the most common words, and
# map everything else to `<unk>`.

# %%
def tokenize_words(s: str) -> list[str]:
    # words and standalone punctuation become tokens
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

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
# Sanity check: encoding then decoding a known-in-vocab phrase round-trips.

# %%
sample = "to be or not to be"
ids = tok.encode(sample)
assert tok.decode(ids) == sample, tok.decode(ids)
print("round-trip ok:", ids, "->", tok.decode(ids))

# %% [markdown]
# ## Train / validation split
#
# We hold out the last 10% to measure generalization honestly.

# %%
ids_all = tok.encode(text)
n = len(ids_all)
split = int(0.9 * n)
train_ids = ids_all[:split]
val_ids = ids_all[split:]
print(f"tokens: {n} (train {len(train_ids)}, val {len(val_ids)})")
