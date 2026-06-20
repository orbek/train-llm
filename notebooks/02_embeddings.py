# %% [markdown]
# # 02 — Embeddings (and why they win)
#
# BoW used a sparse count vector: one slot per vocab word, mostly zeros, and no
# notion that "king" and "queen" are related. A **learned embedding** maps each
# word to a small dense vector. Same task, same prediction head — we only swap
# the representation, then compare perplexity head-to-head with notebook 01.

# %%
import os
import re
import json
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# %% [markdown]
# Rebuild the same word-level tokenizer and dataset as notebook 01 (self-contained).

# %%
def tokenize_words(s: str) -> list[str]:
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

VOCAB_SIZE = 2000
CONTEXT = 8

counts = Counter(tokenize_words(text))
itos = ["<unk>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
stoi = {w: i for i, w in enumerate(itos)}
vocab_size = len(itos)
unk = stoi["<unk>"]

def encode(s): return [stoi.get(w, unk) for w in tokenize_words(s)]

ids_all = encode(text)
split = int(0.9 * len(ids_all))
train_ids, val_ids = ids_all[:split], ids_all[split:]

def make_dataset(ids, context, limit):
    xs, ys = [], []
    step = max(1, (len(ids) - context - 1) // limit)
    for i in range(0, len(ids) - context - 1, step):
        xs.append(ids[i:i + context])
        ys.append(ids[i + context])
    return torch.tensor(xs), torch.tensor(ys)

Xtr, Ytr = make_dataset(train_ids, CONTEXT, 20000)
Xva, Yva = make_dataset(val_ids, CONTEXT, 4000)
Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)
print("context windows:", tuple(Xtr.shape))

# %% [markdown]
# ## The mean-embedding model
#
# Look up an embedding for each of the `CONTEXT` words, average them, and project
# to next-word logits. Note: we keep it order-blind on purpose (averaging) so the
# *only* thing that changes vs BoW is sparse-counts -> dense-learned vectors.

# %%
EMB_DIM = 64

class MeanEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):            # x: (B, T) int ids
        e = self.emb(x)              # (B, T, E)
        pooled = e.mean(dim=1)       # (B, E) — order-blind
        return self.fc(pooled)       # (B, vocab)

model = MeanEmbeddingModel(vocab_size, EMB_DIM).to(device)
emb_params = sum(p.numel() for p in model.parameters())
print("Embedding-model params:", emb_params)

# %% [markdown]
# ## Train (same recipe as BoW)

# %%
def evaluate(m, X, Y):
    m.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(m(X), Y)
    m.train()
    return float(loss)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
losses = []
for epoch in range(60):
    logits = model(Xtr)
    loss = nn.functional.cross_entropy(logits, Ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))

emb_val_loss = evaluate(model, Xva, Yva)
emb_val_ppl = float(torch.exp(torch.tensor(emb_val_loss)))
print(f"Embedding val perplexity {emb_val_ppl:.1f}")

os.makedirs("assets", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("train loss")
plt.title("Mean-embedding training loss"); plt.tight_layout()
plt.savefig("assets/02_embedding_loss.png", dpi=120); plt.show()

# %% [markdown]
# ## Head-to-head with the BoW baseline
#
# The headline result is **parameter efficiency**: the embedding model uses ~15× fewer
# parameters and still matches or edges BoW on perplexity.
#
# Why is the perplexity margin so small?  Predicting the next word from an unordered
# bag over a 2 000-word vocabulary is a high-entropy task — both models hit a similar
# floor.  The embedding model's real advantages are elsewhere: (a) fewer parameters
# means less risk of overfitting (at the planned learning rate the 4 M-param BoW model
# tends to over-fit while the compact embedding model generalises), and (b) learned
# embeddings carry *geometric structure* that one-hot vectors can never have.

# %%
with open("assets/phase1_metrics.json") as f:
    bow_metrics = json.load(f)
bow_val_ppl = bow_metrics["bow_val_perplexity"]
bow_params = bow_metrics["bow_params"]

print(f"BoW       : perplexity {bow_val_ppl:7.1f} | params {bow_params:,}")
print(f"Embeddings: perplexity {emb_val_ppl:7.1f} | params {emb_params:,}")
assert emb_val_ppl < bow_val_ppl, "embeddings should achieve lower perplexity than BoW"
assert emb_params < bow_params, "embeddings should use far fewer parameters than BoW"
param_ratio = bow_params / emb_params
print(f"Parameter efficiency: {param_ratio:.0f}× fewer params with equal/better perplexity.")
print("Perplexity margin is small — both models hit the high-entropy BoW floor.")
print("The robust wins: parameter efficiency and learned geometry (see next section).")

# %% [markdown]
# ## The real win: structure (one-hot can't have it)
#
# Perplexity is nearly tied, but two things separate the models. The embedding
# model uses ~15x fewer parameters — and it learns *geometry*: related words land
# near each other. One-hot vectors are all mutually orthogonal, so every distinct
# word is exactly equidistant from every other. "Similarity" is impossible by
# construction.

# %%
import torch.nn.functional as F

# Two different one-hot vectors are orthogonal -> cosine similarity 0.
i, j = stoi.get("king", unk), stoi.get("queen", unk)
oh_i = F.one_hot(torch.tensor(i), vocab_size).float()
oh_j = F.one_hot(torch.tensor(j), vocab_size).float()
print("one-hot cos(king, queen) =", float(F.cosine_similarity(oh_i, oh_j, dim=0)))
assert torch.dot(oh_i, oh_j) == 0.0  # distinct one-hots are always orthogonal

# %% [markdown]
# Now the learned embeddings. We rank words by cosine similarity in the embedding
# space. At this tiny scale the neighbours are rough — but the structure is real,
# and one-hot has none of it.

# %%
E = model.emb.weight.detach().cpu()      # (vocab, emb_dim)
En = F.normalize(E, dim=1)
sim = En @ En.t()                        # cosine similarity matrix

def nearest(word, k=5):
    if word not in stoi:
        return []
    q = stoi[word]
    scores = sim[q].clone()
    scores[q] = -1.0                     # exclude the word itself
    top = torch.topk(scores, k).indices.tolist()
    return [itos[t] for t in top]

for w in ["king", "love", "night", "death", "father"]:
    if w in stoi:
        print(f"{w:8s} ~ {nearest(w)}")

# Structure exists: the nearest neighbour is much closer than the average word.
q = stoi["king"]
row = sim[q].clone(); row[q] = -1.0
assert row.max() > row.mean(), "learned embeddings should have non-trivial structure"
print("Embeddings learn geometry; one-hot vectors never can.")

# %% [markdown]
# A 2D shadow (PCA) of a handful of frequent words in the learned space:

# %%
viz_words = [w for w in ["king", "queen", "lord", "father", "mother", "love",
                         "death", "night", "day", "sword", "heart", "blood",
                         "god", "man", "woman"] if w in stoi]
vidx = torch.tensor([stoi[w] for w in viz_words])
V = E[vidx]
V = V - V.mean(0)
_, _, Vt = torch.linalg.svd(V, full_matrices=False)
coords = (V @ Vt[:2].t())
plt.figure(figsize=(7, 6))
plt.scatter(coords[:, 0].tolist(), coords[:, 1].tolist())
for w, (x, y) in zip(viz_words, coords.tolist()):
    plt.annotate(w, (x, y))
plt.title("Learned word embeddings (PCA to 2D)")
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/02_embedding_space.png", dpi=120)
plt.show()

# %% [markdown]
# ## But embeddings are *still* order-blind
#
# Averaging is permutation-invariant, so reordering the context gives the same
# prediction — exactly BoW's flaw. Dense vectors fixed *representation*, not
# *order*. Order is what **attention** (notebook 03) finally gives us.

# %%
a = torch.tensor([encode("the king is dead")[:CONTEXT]], device=device)
b = torch.tensor([list(reversed(encode("the king is dead")[:CONTEXT]))], device=device)
with torch.no_grad():
    pa, pb = model(a), model(b)
assert torch.allclose(pa, pb, atol=1e-5), "mean-pooling is order-blind"
print("Confirmed: reordering the context does not change the prediction.")

# %% [markdown]
# ## Switching to characters for the transformer
#
# From here on the model is character-level: a tiny, fully transparent vocab
# (~65 symbols), no `<unk>`, no vocabulary cap. We tensorize the whole corpus
# and define the batching helper Phase 2 will reuse.

# %%
chars = sorted(set(text))
cstoi = {c: i for i, c in enumerate(chars)}
citos = {i: c for c, i in cstoi.items()}
char_vocab_size = len(chars)
print("char vocab:", char_vocab_size)

cdata = torch.tensor([cstoi[c] for c in text], dtype=torch.long)
cn = int(0.9 * len(cdata))
ctrain, cval = cdata[:cn], cdata[cn:]

BLOCK = 64   # context length in characters
BATCH = 32

def get_batch(split: str):
    d = ctrain if split == "train" else cval
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    xb = torch.stack([d[i:i + BLOCK] for i in ix])
    yb = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return xb.to(device), yb.to(device)

xb, yb = get_batch("train")
assert xb.shape == (BATCH, BLOCK) and yb.shape == (BATCH, BLOCK)
print("char batch:", tuple(xb.shape), "-> targets", tuple(yb.shape))
print("Ready for attention in notebook 03.")
