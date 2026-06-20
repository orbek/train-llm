# Phase 3 — Advanced: Tuning, BPE, MoE (Notebooks 08–10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Finish the course with three advanced notebooks: hyperparameter tuning (LR schedules + sweeps), a Byte-Pair Encoding tokenizer built from scratch, and a Mixture-of-Experts capstone.

**Architecture:** Builds on the merged Phases 1–2. Reuses `model.py` (the source-of-truth GPT). Notebook 08 teaches an LR schedule + small sweeps on the `nano` model. Notebook 09 implements BPE from scratch and compares it to char-level (with `tiktoken` as a reference). Notebook 10 (capstone) is self-contained: it builds a Mixture-of-Experts feed-forward layer and a compact MoE model reusing `model.py` components, leaving `model.py` itself stable.

**Tech Stack:** Python 3.11+, PyTorch (MPS/CPU), NumPy, Matplotlib, tqdm, JupyterLab, jupytext, and `tiktoken` (reference only, notebook 09). No HuggingFace.

## Global Constraints

- Python 3.11+; PyTorch with MPS backend, CPU fallback.
- No HuggingFace / high-level modeling libraries. `tiktoken` permitted ONLY in notebook 09, and its use is import-guarded (skip gracefully if absent).
- Notebook source of truth = jupytext percent `.py`; each runs clean as a script (`MPLBACKEND=Agg python notebooks/NN.py`, exit 0, asserts hold) AND renders (`jupytext --to notebook --execute --set-kernel python3 notebooks/NN.py -o notebooks/NN.ipynb`). Commit both.
- **Fast renders:** use `nano`-scale runs (few hundred iterations) for all training in Phase 3 so notebooks render quickly. Each training cell notes how to scale up. NO full ~9.4M retrains.
- **Explanation standard (binding):** markdown teaches step-by-step in plain beginner-friendly language; define jargon inline on first use; split dense code cells so each function/class is explained immediately before it; walk through each with a tiny concrete example. See memory `notebook-explanation-depth`. The plan gives the CODE and teaching beats; the implementer writes the prose.
- **First code cell of every notebook = the repo-root cwd cell** (copy from an existing notebook).
- `model.py` MUST NOT be modified in Phase 3 (it is merged, tested, stable). Notebook 10 reuses its components by import and defines MoE pieces locally.
- Training convention: `loss.item()`; device = MPS else CPU.
- `data/` and `checkpoints/` gitignored (do NOT commit `data/shakespeare.txt` or `*.pt`); `assets/` tracked.

## Plan Conventions

- Repo-root cwd cell (first code cell, every notebook) — copy verbatim from `notebooks/07_eval_and_generation.py` (the `while not os.path.exists("requirements.txt")` walk).
- Test cycle: run as script (asserts hold) then render. All asserts must be genuine and unweakened.
- Data: load `data/shakespeare.txt` if present, else download it once (same URL/pattern as notebook 01) — these notebooks must run on a fresh clone.

---

## File Structure

- Modify: `requirements.txt` — add `tiktoken>=0.5` (reference comparison in nb09).
- Create: `notebooks/08_tuning.py` (+ `.ipynb`), `assets/08_lr_schedule.png`, `assets/08_lr_sweep.png`.
- Create: `notebooks/09_bpe_tokenizer.py` (+ `.ipynb`), `assets/09_compression.png`.
- Create: `notebooks/10_mixture_of_experts.py` (+ `.ipynb`), `assets/10_expert_utilization.png`.

---

### Task 1: Notebook 08 — Tuning (LR schedule + small sweeps)

**Files:**
- Create: `notebooks/08_tuning.py` (+ `.ipynb`)
- Create: `assets/08_lr_schedule.png`, `assets/08_lr_sweep.png`

**Interfaces:**
- Consumes: `model.py` (`GPT`, `NANO_CONFIG`); char data (re-declared, small).
- Produces: a teaching `get_lr` helper (pedagogical; not added to model.py).

- [ ] **Step 1: Write the notebook**

Apply the explanation standard (define: hyperparameter, learning rate, warmup, cosine decay, overfitting, sweep). Cells:
1. Repo-root cwd cell; imports incl. `import math`, `from model import GPT, NANO_CONFIG`; device; `torch.manual_seed(1337)`.
2. Char data: load/download `data/shakespeare.txt`, build `stoi`/`itos`, encode tensor, 90/10 split, `get_batch(split, block_size, batch_size)`.
3. **LR schedule helper** (teach warmup + cosine decay):
```python
def get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    if it < warmup_iters:                       # linear warmup
        return max_lr * (it + 1) / warmup_iters
    if it > lr_decay_iters:                      # after decay: floor
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)
```
   Plot the schedule over the full run → `assets/08_lr_schedule.png`. Asserts (genuine):
```python
warm, decay, hi, lo = 100, 1000, 1e-3, 1e-4
assert get_lr(0, warm, decay, hi, lo) < get_lr(warm, warm, decay, hi, lo)    # warms up
assert abs(get_lr(decay + 50, warm, decay, hi, lo) - lo) < 1e-12             # decays to floor
assert get_lr(warm, warm, decay, hi, lo) <= hi + 1e-9                        # peak ~ max_lr
```
4. **LR sweep** (small, fast): a short training function `train_nano(lr, iters=300)` that trains a fresh `GPT(NANO_CONFIG)` (override vocab_size) on char data with a fixed `lr`, returning final val loss. Run for `lr in [1e-2, 3e-3, 1e-3, 3e-4]`, collect val losses, bar-plot → `assets/08_lr_sweep.png`. Markdown interprets: too-high diverges, too-low underfits, a sweet spot in between. Assert the sweep produced a finite best and that not all lrs are equal:
```python
assert min(val_losses) < max(val_losses)        # lr actually matters
```
5. **Brief model-size note**: a tiny comparison (e.g. nano vs nano with half `n_embd`) for ~200 iters, or a markdown discussion if a second run is too slow — keep total notebook runtime small. Explain which knobs matter most (lr, then model size / context / batch).
6. Markdown: practical tuning advice + a note that these sweeps are tiny for speed; scale iters/model up for real tuning.

- [ ] **Step 2: Run as script** (`MPLBACKEND=Agg python notebooks/08_tuning.py`, exit 0, asserts hold, both pngs written). **Step 3: Render.** **Step 4: Commit**
```bash
git add notebooks/08_tuning.py notebooks/08_tuning.ipynb assets/08_lr_schedule.png assets/08_lr_sweep.png
git commit -m "feat: notebook 08 — tuning (LR warmup+cosine schedule, small sweeps)"
```

---

### Task 2: Notebook 09 — BPE tokenizer from scratch

**Files:**
- Modify: `requirements.txt` (add `tiktoken>=0.5`)
- Create: `notebooks/09_bpe_tokenizer.py` (+ `.ipynb`)
- Create: `assets/09_compression.png`

**Interfaces:**
- Consumes: `model.py` (`GPT`, `NANO_CONFIG`) for the brief end-to-end demo; char data.
- Produces: a `BPETokenizer` (pedagogical).

- [ ] **Step 1: Add tiktoken to requirements**

Append `tiktoken>=0.5` to `requirements.txt`. Install it: `pip install "tiktoken>=0.5"`. (The notebook still import-guards it.)

- [ ] **Step 2: Write the notebook**

Apply the explanation standard (define: subword, byte, merge, vocabulary, compression ratio, OOV). Cells:
1. Repo-root cwd cell; imports; device; seed.
2. Markdown: why subword? char-level sequences are long; word-level has huge vocab + OOV. BPE sits in between: start from bytes, repeatedly merge the most frequent adjacent pair into a new token.
3. **The two helpers** (each explained, with a tiny worked example):
```python
def get_stats(ids):
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
```
4. **`BPETokenizer`** (explain `train`, `encode`, `decode` step by step):
```python
class BPETokenizer:
    def __init__(self):
        self.merges = {}                                   # (int,int) -> int
        self.vocab = {}                                    # int -> bytes

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```
5. **Train it** on a subset of Shakespeare for speed (e.g. first ~200_000 chars) with `vocab_size = 512`. Print a few learned merges (decode the merged tokens to show they form common substrings like `" th"`, `"the"`, `"ou"`). Round-trip assert:
```python
sample = "To be, or not to be, that is the question."
assert bpe.decode(bpe.encode(sample)) == sample
```
6. **Compression comparison** vs char-level: for a held-out chunk, compute char-level token count (= number of chars) vs BPE token count; report chars-per-token (BPE should be > 1). Bar-plot sequence lengths → `assets/09_compression.png`. Assert BPE compresses:
```python
assert bpe_tokens < char_tokens          # BPE uses fewer tokens than char-level
```
7. **tiktoken reference** (import-guarded):
```python
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    print("tiktoken gpt2 vocab:", enc.n_vocab, "tokens for sample:", len(enc.encode(sample)))
except ImportError:
    print("tiktoken not installed — skipping reference comparison")
```
   Markdown: our from-scratch BPE is the same idea GPT-2/Llama tokenizers use, just smaller and simpler.
8. **Brief end-to-end demo**: encode a subset with BPE, train `GPT(NANO_CONFIG)` (override `vocab_size = 512`, `block_size` small) for ~300 iters, confirm loss drops. Assert `final_loss < initial_loss`. Markdown notes BPE shortens sequences so the same context window covers more text.

- [ ] **Step 3: Run as script** (exit 0, all asserts hold, png written). **Step 4: Render.** **Step 5: Commit**
```bash
git add requirements.txt notebooks/09_bpe_tokenizer.py notebooks/09_bpe_tokenizer.ipynb assets/09_compression.png
git commit -m "feat: notebook 09 — BPE tokenizer from scratch + char-level comparison"
```

---

### Task 3: Notebook 10 — Mixture-of-Experts (capstone)

**Files:**
- Create: `notebooks/10_mixture_of_experts.py` (+ `.ipynb`)
- Create: `assets/10_expert_utilization.png`

**Interfaces:**
- Consumes: `model.py` (`RMSNorm`, `CausalSelfAttention`, `build_rope`, `GPTConfig`); char data.
- Produces: `MoEFeedForward` and a compact MoE GPT (pedagogical, in-notebook). `model.py` is NOT modified.

- [ ] **Step 1: Write the notebook**

Apply the explanation standard (define: expert, router/gating, top-k routing, sparse activation, load balancing, active vs total parameters). Cells:
1. Repo-root cwd cell; imports incl. `import torch.nn.functional as F`, `from model import RMSNorm, CausalSelfAttention, build_rope, GPTConfig`; device; seed.
2. Markdown: the idea — instead of one big MLP per layer, have `N` smaller expert MLPs and a router that sends each token to only its top-`k` experts. Total parameters grow with `N`, but compute per token stays ~constant (only `k` experts run). That's "sparse" scaling.
3. **A single expert** (small SwiGLU):
```python
class Expert(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = 64 * (((int(8 / 3 * n_embd)) + 63) // 64)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```
4. **The MoE feed-forward** with top-k routing + load-balancing aux loss (explain each step):
```python
class MoEFeedForward(nn.Module):
    def __init__(self, n_embd, n_experts=4, top_k=2, dropout=0.0):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd, dropout) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                  # (N, C)
        probs = F.softmax(self.gate(xf), dim=-1)              # (N, n_experts)
        topv, topi = probs.topk(self.top_k, dim=-1)          # (N, top_k)
        topv = topv / topv.sum(-1, keepdim=True)             # renormalize the kept weights
        out = torch.zeros_like(xf)
        for slot in range(self.top_k):
            idx = topi[:, slot]                               # which expert (N,)
            w = topv[:, slot].unsqueeze(-1)                   # its weight (N,1)
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](xf[mask])
        # Switch-style load-balancing aux loss: encourage uniform routing.
        importance = probs.mean(0)                            # mean gate prob per expert
        top1 = probs.argmax(-1)
        load = torch.bincount(top1, minlength=self.n_experts).float() / xf.size(0)
        aux = self.n_experts * (importance * load).sum()
        return out.reshape(B, T, C), aux
```
5. **Shape + aux asserts** on a dummy batch:
```python
moe = MoEFeedForward(64, n_experts=4, top_k=2)
y, aux = moe(torch.randn(2, 8, 64))
assert y.shape == (2, 8, 64)
assert torch.isfinite(aux) and aux.item() >= 0
```
6. **A compact MoE model** reusing model.py components (RMSNorm + CausalSelfAttention + RoPE), with `Block` swapping the MLP for `MoEFeedForward`, accumulating aux losses:
```python
class MoEBlock(nn.Module):
    def __init__(self, config, n_experts, top_k):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.moe = MoEFeedForward(config.n_embd, n_experts, top_k, config.dropout)
    def forward(self, x, cos, sin):
        a, _ = self.attn(self.attn_norm(x), cos, sin, None)
        x = x + a
        m, aux = self.moe(self.mlp_norm(x))
        return x + m, aux

class MoEGPT(nn.Module):
    def __init__(self, config, n_experts=4, top_k=2):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([MoEBlock(config, n_experts, top_k) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T].to(x.device), self.rope_sin[:T].to(x.device)
        aux_total = 0.0
        for block in self.blocks:
            x, aux = block(x, cos, sin)
            aux_total = aux_total + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total
```
7. **Train a nano MoE briefly** on char data (small `GPTConfig` like NANO dims, ~300 iters). Total loss = `loss + 0.01 * aux_total`. Track per-expert utilization across a batch. Plot a bar chart of tokens routed per expert → `assets/10_expert_utilization.png`.
8. **Asserts** (genuine): training reduces loss; load balancing keeps experts in use (no dead/dominant expert):
```python
assert final_loss < initial_loss
# utilization is the fraction of tokens whose top-1 expert == e, measured on a batch
assert utilization.min() > 0.05 and utilization.max() < 0.95   # not collapsed
```
9. **Active vs total params**: print total params (all experts) vs active params per token (≈ embedding + attention + `top_k`/`n_experts` of the expert params). Markdown: this is how frontier models scale capacity cheaply.
10. Closing markdown: course recap (BoW → embeddings → attention → modern components → training → generation → tuning → BPE → MoE) and where to go next.

- [ ] **Step 2: Run as script** (`MPLBACKEND=Agg python notebooks/10_mixture_of_experts.py`, exit 0, all asserts hold, png written). **Step 3: Render.** **Step 4: Commit**
```bash
git add notebooks/10_mixture_of_experts.py notebooks/10_mixture_of_experts.ipynb assets/10_expert_utilization.png
git commit -m "feat: notebook 10 — Mixture-of-Experts capstone (top-k routing, load balancing)"
```

---

## Self-Review

**Spec coverage (Phase 3 = notebooks 08–10):**
- Tuning: LR warmup+cosine schedule + small sweeps, which knobs matter → Task 1. ✓
- BPE from scratch, compare vs char-level, tiktoken reference, brief end-to-end → Task 2. ✓
- MoE capstone: experts + top-k router + load-balancing aux loss, utilization, active-vs-total params, model.py untouched → Task 3. ✓
- Fast nano-scale renders; explanation standard; cwd cell; gitignore compliance → Global Constraints, all tasks. ✓

**Placeholder scan:** No TBD/TODO; complete code for `get_lr`, `BPETokenizer`, `MoEFeedForward`/`MoEGPT`; teaching beats specified; prose to the documented standard. ✓

**Type/name consistency:** `get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr)`; `BPETokenizer.train/encode/decode` with `get_stats`/`merge`; `MoEFeedForward(n_embd, n_experts, top_k, dropout) -> (out, aux)`; `MoEGPT.forward -> (logits, loss, aux_total)`; reuses `RMSNorm`/`CausalSelfAttention`/`build_rope`/`GPTConfig` from `model.py` (unchanged). ✓
