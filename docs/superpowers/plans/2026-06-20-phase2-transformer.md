# Phase 2 — The Working Transformer (Notebooks 03–07) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete modern decoder-only transformer from scratch (attention → modern components → assembled model → training → generation), trained on char-level Shakespeare to produce recognizable text.

**Architecture:** Llama-style decoder-only model: causal attention with Grouped-Query Attention (GQA), RoPE rotary positions, RMSNorm (pre-norm), SwiGLU MLP, weight-tied embeddings, KV-cache for generation. The model lives in a single source-of-truth `model.py`; notebook 05 teaches it and notebooks 06–07 import it. Notebooks 03–04 teach the components in cells first.

**Tech Stack:** Python 3.11+, PyTorch (MPS/CPU), NumPy, Matplotlib, tqdm, JupyterLab, jupytext. No HuggingFace.

## Global Constraints

- Python 3.11+; PyTorch with MPS backend, CPU fallback.
- No HuggingFace / high-level modeling libraries; no `tiktoken` in Phase 2.
- **Notebook source of truth = jupytext percent `.py`**; each notebook must run clean as a script (`MPLBACKEND=Agg python notebooks/NN.py`, exit 0, asserts hold) AND render to `.ipynb` (`jupytext --to notebook --execute --set-kernel python3 notebooks/NN.py -o notebooks/NN.ipynb`). Commit both.
- **`model.py` is the single source of truth** for `GPTConfig`, `RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `Block`, `GPT`, and the RoPE/sampling helpers. Notebooks 06–07 import from it. Notebooks 03–04 build *teaching* versions of components in cells (these may duplicate ideas — that is intentional teaching, not a defect). Notebook 05 imports and teaches `model.py` (does not redefine it).
- Small per-notebook code (char tokenizer, `get_batch`) stays re-declared per notebook (it is short).
- **Explanation standard (binding, from Phase 1):** markdown must teach, not just name concepts — plain beginner-friendly language; define jargon inline on first use; split dense code cells so each function/class is explained immediately before it; walk through each function/class step by step with a tiny concrete worked example. See memory `notebook-explanation-depth`. This plan specifies the CODE and the teaching beats; the implementer writes the full prose to this standard.
- **Working-directory cell:** every notebook's FIRST code cell is the repo-root snippet (below) so `data/`/`assets/`/`checkpoints/` resolve regardless of the kernel's launch dir.
- **Training convention:** extract scalar losses with `loss.item()`. Comment any deviation from a plan-specified hyperparameter inline.
- `data/` and `checkpoints/` gitignored (do NOT commit `data/shakespeare.txt` or `checkpoints/*.pt`); `assets/` tracked.
- Device convention: prefer MPS, else CPU.

## Plan Conventions

- **Repo-root cell** (the standard first code cell for EVERY notebook, including the retrofit of 00–02):
```python
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
```
- **Test cycle per notebook:** run as script (asserts hold) then render. **Test cycle for `model.py`:** `python test_model.py` exits 0 with all asserts passing.
- Default config (the "full" model) and `nano` preset are defined once in `model.py` (Task 4).

---

## File Structure

- Create: `model.py` — source-of-truth model (config, components, GPT, helpers). ~230 lines.
- Create: `test_model.py` — runnable assert-based tests for `model.py`.
- Modify: `notebooks/00_setup_and_tour.py`, `01_..py`, `02_..py` — prepend the repo-root cell; re-render. (Retrofit of the cwd fix.)
- Create: `notebooks/03_attention.py` (+ `.ipynb`) — scaled dot-product attention, causal mask, multi-head, GQA, attention visualization.
- Create: `notebooks/04_modern_components.py` (+ `.ipynb`) — RoPE, RMSNorm, SwiGLU, each vs its predecessor.
- Create: `notebooks/05_assembling_the_model.py` (+ `.ipynb`) — teach `model.py`, instantiate nano + full, param count, forward-shape + overfit-one-batch sanity.
- Create: `notebooks/06_training.py` (+ `.ipynb`) — char data, training loop, full training run, loss curves, checkpoint to `checkpoints/`.
- Create: `notebooks/07_eval_and_generation.py` (+ `.ipynb`) — perplexity, sampling (greedy/temperature/top-k/top-p), KV-cache + cached==non-cached verification, sample generations.
- Create (gitignored, not committed): `checkpoints/model.pt`, `assets/*` plots (assets committed).

---

### Task 1: Repo-root cwd convention (retrofit notebooks 00–02)

**Files:**
- Modify: `notebooks/00_setup_and_tour.py`, `notebooks/01_data_and_bag_of_words.py`, `notebooks/02_embeddings.py` (prepend the repo-root cell as the first code cell, with a short markdown explanation before it)
- Regenerate: the three `.ipynb`

**Interfaces:**
- Consumes: nothing.
- Produces: the standard repo-root cell pattern that notebooks 03–07 will also use.

- [ ] **Step 1: Add a markdown + code cell pair at the top of each of the three notebooks**

After the title markdown cell, insert a markdown cell explaining (plain language): "Jupyter runs a notebook from its own folder, so before anything else we step up to the project's top folder. That way paths like `data/` point to the right place no matter how you launched Jupyter." Then insert the repo-root code cell (from Plan Conventions) as the first executable cell.

- [ ] **Step 2: Verify each runs as a script and the stray-dir bug is gone**

Run (from repo root): `MPLBACKEND=Agg python notebooks/00_setup_and_tour.py` then `01_...` then `02_...`. Expected: all exit 0, all asserts still hold. Then verify the cwd fix from inside notebooks/:
```bash
cd notebooks && MPLBACKEND=Agg python 02_embeddings.py && cd ..
ls notebooks/data notebooks/assets 2>/dev/null && echo "STRAY DIRS (bad)" || echo "no stray dirs (good)"
```
Expected: "no stray dirs (good)" — the script chdir'd to repo root, so data/assets resolved there. (If a stray dir appears, the cwd cell is wrong.) Clean up any test artifacts.

- [ ] **Step 3: Re-render the three notebooks**

```bash
for n in 00_setup_and_tour 01_data_and_bag_of_words 02_embeddings; do
  jupytext --to notebook --execute --set-kernel python3 notebooks/$n.py -o notebooks/$n.ipynb
done
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/00_setup_and_tour.py notebooks/00_setup_and_tour.ipynb notebooks/01_data_and_bag_of_words.py notebooks/01_data_and_bag_of_words.ipynb notebooks/02_embeddings.py notebooks/02_embeddings.ipynb
git commit -m "feat: repo-root cwd cell in notebooks 00-02 (fixes stray-dir on Jupyter launch)"
```

---

### Task 2: Notebook 03 — Attention from scratch (incl. GQA)

**Files:**
- Create: `notebooks/03_attention.py` (+ `.ipynb`)
- Create: `assets/03_attention_weights.png`

**Interfaces:**
- Consumes: char-level batching idea from notebook 02 (re-declared here, small).
- Produces: teaching versions of `scaled_dot_product_attention`, `MultiHeadAttention`, `GroupedQueryAttention` (these are pedagogical; the canonical ones live in `model.py`, Task 4).

- [ ] **Step 1: Write the notebook**

Cells (apply the explanation standard to all markdown — define: query/key/value, dot-product similarity, scaling by √head_dim, softmax, causal mask, head, GQA):
1. Repo-root cell.
2. Imports + device + `torch.manual_seed(1337)`.
3. Markdown: the intuition — attention lets each token look at previous tokens and pull in a weighted mix of their information; order finally matters.
4. **Single-head scaled dot-product attention** (teach, then code):
```python
def scaled_dot_product_attention(q, k, v, causal=True):
    # q,k,v: (B, T, head_dim)
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)   # (B, T, T)
    if causal:
        T = scores.size(-1)
        mask = torch.tril(torch.ones(T, T, device=scores.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
```
   - Assert causal property: with `causal=True`, the attention `weights` are lower-triangular (no weight on future positions):
```python
B, T, d = 1, 5, 8
q = torch.randn(B, T, d); k = torch.randn(B, T, d); v = torch.randn(B, T, d)
out, w = scaled_dot_product_attention(q, k, v, causal=True)
assert torch.allclose(w.triu(1), torch.zeros_like(w), atol=1e-6)  # nothing above the diagonal
```
5. **Multi-head attention** (teach why multiple heads; code a `MultiHeadAttention` nn.Module that splits n_embd into n_head heads, runs SDPA per head, concatenates, output-projects). Assert output shape `(B, T, n_embd)`.
6. **Grouped-Query Attention** (teach: queries keep `n_head` heads but keys/values use fewer `n_kv_head` heads, shared across groups — saves memory/KV-cache size in big models). Code a `GroupedQueryAttention` module using `repeat_interleave` on k,v. Assert: shape `(B,T,n_embd)`; and that with `n_kv_head == n_head` it has the same parameter shapes as plain MHA (n_rep == 1).
7. **Visualize** attention: run one head on a short char sequence, plot the `weights` matrix with matplotlib (`imshow`), save to `assets/03_attention_weights.png`, `plt.show()`. Markdown interprets the triangular pattern.

- [ ] **Step 2: Run as script** — `MPLBACKEND=Agg python notebooks/03_attention.py` (exit 0, asserts hold, png written).
- [ ] **Step 3: Render** — jupytext to `.ipynb`.
- [ ] **Step 4: Commit**
```bash
git add notebooks/03_attention.py notebooks/03_attention.ipynb assets/03_attention_weights.png
git commit -m "feat: notebook 03 — attention from scratch (SDPA, causal, multi-head, GQA, viz)"
```

---

### Task 3: Notebook 04 — Modern components (RoPE, RMSNorm, SwiGLU)

**Files:**
- Create: `notebooks/04_modern_components.py` (+ `.ipynb`)

**Interfaces:**
- Produces: teaching versions of `build_rope`, `apply_rope`, `RMSNorm`, `SwiGLU` (canonical versions in `model.py`, Task 4).

- [ ] **Step 1: Write the notebook**

Apply the explanation standard. Three sections, each "modern thing vs the older thing it replaces":
1. Repo-root cell, imports, device, seed.
2. **RMSNorm vs LayerNorm.** Teach what normalization does (keeps activations at a stable scale so training is smooth), then that RMSNorm drops the mean-centering and bias for speed. Code:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```
   Assert: output has ~unit RMS per row: for random input, `x.pow(2).mean(-1)` of the normalized (weight=1) output ≈ 1.0 (atol 1e-2).
3. **RoPE vs learned absolute positions.** Teach: instead of adding a position vector, RoPE *rotates* each query/key by an angle proportional to its position, so the dot-product between two tokens depends on their *relative* distance. Code `build_rope`, `rotate_half`, `apply_rope`:
```python
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
```
   Assert the **relative-position property**: rotate two vectors at positions p and p+k; the dot product equals that of the same two vectors at positions q and q+k (same offset → same similarity). Concretely build small q,k at different absolute offsets and assert dot products match (atol 1e-4).
4. **SwiGLU vs a plain GELU MLP.** Teach: a normal MLP is up-project → activation → down-project; SwiGLU adds a "gate" branch (`silu(W1 x) * (W3 x)`) that lets the network modulate what passes through. Code:
```python
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
```
   Assert output shape equals input shape.

- [ ] **Step 2: Run as script** (exit 0, asserts hold). **Step 3: Render.** **Step 4: Commit**
```bash
git add notebooks/04_modern_components.py notebooks/04_modern_components.ipynb
git commit -m "feat: notebook 04 — RoPE, RMSNorm, SwiGLU vs their predecessors"
```

---

### Task 4: `model.py` + `test_model.py` (source-of-truth GPT)

**Files:**
- Create: `model.py`
- Create: `test_model.py`

**Interfaces:**
- Produces (imported by notebooks 05–07): `GPTConfig`, `NANO_CONFIG`, `DEFAULT_CONFIG`, `GPT` (with `.forward(idx, targets=None, kv_caches=None, pos=0) -> (logits, loss, presents)` and `.generate(idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True) -> idx`), `RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `Block`, `build_rope`, `apply_rope`, `sample`.

- [ ] **Step 1: Write `model.py`** (TDD: write `test_model.py` first in Step 2's spirit, but the code is fully specified — create the module):

```python
"""Source-of-truth model for the course: a small Llama-style decoder-only
transformer (GQA + RoPE + RMSNorm + SwiGLU + weight tying + KV-cache).
Notebook 05 explains every piece; notebooks 06-07 import from here."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2        # GQA: #key/value heads (must divide n_head)
    n_embd: int = 384
    dropout: float = 0.2
    rope_theta: float = 10000.0


# The "full" ~10-15M model (committed training run) and a tiny debug preset.
DEFAULT_CONFIG = GPTConfig()
NANO_CONFIG = GPTConfig(block_size=64, n_layer=3, n_head=4, n_kv_head=2,
                        n_embd=128, dropout=0.0)


def build_rope(head_dim, seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)      # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 64 * (((int(8 / 3 * config.n_embd)) + 63) // 64)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_k, past_v = kv_cache
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)

        if self.n_rep > 1:                              # GQA: expand kv heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = past_len == 0                       # prefill is causal; single-step isn't
        if not is_causal:
            assert T == 1, "cached generation feeds one token at a time"
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y)), present


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, kv_cache=None):
        a, present = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight       # weight tying
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # subtract tied head (shares storage with tok_emb)
        n = sum(p.numel() for p in self.parameters())
        return n - self.lm_head.weight.numel()

    def forward(self, idx, targets=None, kv_caches=None, pos=0):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[pos:pos + T].to(x.device)
        sin = self.rope_sin[pos:pos + T].to(x.device)
        presents = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, present = block(x, cos, sin, cache)
            presents.append(present)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, use_cache=True):
        was_training = self.training
        self.eval()
        if use_cache:
            assert idx.size(1) + max_new_tokens <= self.config.block_size, \
                "cached generation supports up to block_size total tokens"
            kv_caches = [(None, None) for _ in range(self.config.n_layer)]
            logits, _, presents = self(idx, kv_caches=kv_caches, pos=0)
            kv_caches = presents
            pos = idx.size(1)
            for _ in range(max_new_tokens):
                nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat([idx, nxt], dim=1)
                logits, _, presents = self(nxt, kv_caches=kv_caches, pos=pos)
                kv_caches = presents
                pos += 1
        else:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _, _ = self(idx_cond)
                nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat([idx, nxt], dim=1)
        if was_training:
            self.train()
        return idx


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0.0:                              # greedy / deterministic
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    if top_p is not None:
        s_logits, s_idx = torch.sort(logits, descending=True)
        probs = F.softmax(s_logits, dim=-1).cumsum(dim=-1)
        remove = probs - F.softmax(s_logits, dim=-1) > top_p
        s_logits = s_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, s_idx, s_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

- [ ] **Step 2: Write `test_model.py` (the test) and run it**

```python
import torch
from model import GPT, GPTConfig, NANO_CONFIG

torch.manual_seed(0)


def test_forward_shapes():
    cfg = NANO_CONFIG
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = m(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.dim() == 0 and loss.item() > 0


def test_causal_no_future_leak():
    cfg = NANO_CONFIG
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base, _, _ = m(idx)
        changed = idx.clone(); changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
        alt, _, _ = m(changed)
    # changing the LAST token must not change logits at earlier positions
    assert torch.allclose(base[:, :-1], alt[:, :-1], atol=1e-5)


def test_gqa_reduces_to_mha():
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4,
                    n_kv_head=4, n_embd=64, dropout=0.0)   # n_kv_head == n_head
    m = GPT(cfg)
    assert m.blocks[0].attn.n_rep == 1
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _, _ = m(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_weight_tying():
    m = GPT(NANO_CONFIG)
    assert m.tok_emb.weight.data_ptr() == m.lm_head.weight.data_ptr()


def test_generate_length():
    cfg = NANO_CONFIG
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = m.generate(idx, max_new_tokens=10, temperature=1.0, top_k=5)
    assert out.shape == (1, 14)


def test_kv_cache_matches_no_cache():
    cfg = NANO_CONFIG
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    a = m.generate(idx.clone(), max_new_tokens=20, temperature=0.0, use_cache=True)
    b = m.generate(idx.clone(), max_new_tokens=20, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "cached and non-cached greedy generation must match"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL TESTS PASSED")
```

Run: `python test_model.py`. Expected: prints `ok: ...` for each test then `ALL TESTS PASSED`. If `test_kv_cache_matches_no_cache` fails, the KV-cache/RoPE-position logic is wrong — fix `forward`'s `pos` handling, do not weaken the test.

- [ ] **Step 3: Commit**
```bash
git add model.py test_model.py
git commit -m "feat: model.py — Llama-style GPT (GQA, RoPE, RMSNorm, SwiGLU, KV-cache) + tests"
```

---

### Task 5: Notebook 05 — Assembling & teaching the model

**Files:**
- Create: `notebooks/05_assembling_the_model.py` (+ `.ipynb`)

**Interfaces:**
- Consumes: `model.py` (Task 4).
- Produces: nothing new (teaches the assembled model; later notebooks import `model.py` directly).

- [ ] **Step 1: Write the notebook**

Apply the explanation standard. The notebook IMPORTS `model.py` and walks through it (do not redefine the classes). Cells:
1. Repo-root cell; imports incl. `from model import GPT, GPTConfig, NANO_CONFIG, DEFAULT_CONFIG`; device; seed.
2. Markdown: recap — we built attention (03) and the modern components (04); now we stack them into a full model. Explain the **pre-norm transformer block** (normalize → attention → add residual; normalize → SwiGLU → add residual) and why residual connections + pre-norm make deep nets trainable. Reference `Block` in `model.py`.
3. Markdown: explain **weight tying** (the input embedding and output projection share one matrix — fewer params, often better) pointing at the `self.tok_emb.weight = self.lm_head.weight` line.
4. Instantiate both presets, print parameter counts:
```python
nano = GPT(NANO_CONFIG)
full = GPT(DEFAULT_CONFIG)
print(f"nano params: {nano.num_params()/1e6:.2f}M")
print(f"full params: {full.num_params()/1e6:.2f}M")
assert 9e6 < full.num_params() < 16e6      # the "10-15M" target
```
5. **Forward-shape check** on a dummy batch (assert logits shape `(B,T,vocab)` and a scalar loss).
6. **Overfit-one-batch sanity test** (the canonical lesson — a correct model can memorize a single batch to near-zero loss):
```python
m = GPT(NANO_CONFIG).to(device)
xb = torch.randint(0, NANO_CONFIG.vocab_size, (4, 16), device=device)
yb = torch.randint(0, NANO_CONFIG.vocab_size, (4, 16), device=device)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
for _ in range(300):
    _, loss, _ = m(xb, targets=yb)
    opt.zero_grad(); loss.backward(); opt.step()
print("overfit loss:", loss.item())
assert loss.item() < 0.5      # memorized the single batch -> model wiring is correct
```
   Markdown explains why this test is the first thing to run on any new model.

- [ ] **Step 2: Run as script** (exit 0, asserts hold). **Step 3: Render.** **Step 4: Commit**
```bash
git add notebooks/05_assembling_the_model.py notebooks/05_assembling_the_model.ipynb
git commit -m "feat: notebook 05 — assemble & teach the full GPT model"
```

---

### Task 6: Notebook 06 — Training loop & the real training run

**Files:**
- Create: `notebooks/06_training.py` (+ `.ipynb`)
- Create: `assets/06_loss_curve.png`
- Create (gitignored, NOT committed): `checkpoints/model.pt`

**Interfaces:**
- Consumes: `model.py`.
- Produces: a trained checkpoint at `checkpoints/model.pt` (a dict with `model_state`, `config`, `stoi`, `itos`) consumed by notebook 07.

- [ ] **Step 1: Write the notebook**

Apply the explanation standard (define: epoch/iteration, batch, cross-entropy, AdamW, learning rate, gradient, train vs val loss, checkpoint). Cells:
1. Repo-root cell; imports incl. `from model import GPT, DEFAULT_CONFIG`; device; seed.
2. **Char-level data** (re-declared, small): load `data/shakespeare.txt`, build `stoi`/`itos`, encode to a tensor, 90/10 split, `get_batch(split)` returning `(B, block_size)` tensors on `device`. Set `config = DEFAULT_CONFIG` but override `vocab_size = len(stoi)`.
3. **Loss estimation helper** (averaged over a few batches, under `torch.no_grad()` + `model.eval()`):
```python
@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss, _ = model(xb, targets=yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```
4. **Training loop.** AdamW (lr=3e-4, weight_decay=0.1), `max_iters` set to fit a wall-clock budget (target ≤ ~8 min on MPS; start with `max_iters = 3000`, `eval_interval = 500`, `batch_size = 32`). Record train/val loss at each eval. Use `loss.item()`. Track best val loss.
   - **Wall-clock safety:** if running this as a script would exceed a single command timeout, the implementer runs the script in the background and polls until it finishes (it writes the checkpoint + png on completion). Document at the top of the cell: "Increase `max_iters` for better samples; this committed run is capped for time."
5. **Plot** train/val loss curves → `assets/06_loss_curve.png`, `plt.show()`.
6. **Checkpoint**:
```python
import os
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state": model.state_dict(), "config": config,
            "stoi": stoi, "itos": itos}, "checkpoints/model.pt")
print("saved checkpoints/model.pt")
```
7. **Asserts:** training reduced the loss meaningfully:
```python
assert history["val"][-1] < history["val"][0] - 0.3   # val loss clearly dropped
assert os.path.exists("checkpoints/model.pt")
```

- [ ] **Step 2: Run as script** — `MPLBACKEND=Agg python notebooks/06_training.py` (run in background + poll if it would exceed the command timeout). Expected: exit 0, asserts hold, `checkpoints/model.pt` and `assets/06_loss_curve.png` written. Record the final train/val loss.
- [ ] **Step 3: Render** the `.ipynb` (the render re-runs training; ensure the capped `max_iters` keeps render time reasonable — render in background if needed).
- [ ] **Step 4: Commit** (do NOT commit the checkpoint — it is gitignored)
```bash
git add notebooks/06_training.py notebooks/06_training.ipynb assets/06_loss_curve.png
git commit -m "feat: notebook 06 — training loop + real training run + checkpoint"
```

---

### Task 7: Notebook 07 — Evaluation & generation (with KV-cache)

**Files:**
- Create: `notebooks/07_eval_and_generation.py` (+ `.ipynb`)

**Interfaces:**
- Consumes: `model.py`; `checkpoints/model.pt` from Task 6.

- [ ] **Step 1: Write the notebook**

Apply the explanation standard (define: perplexity, greedy vs sampling, temperature, top-k, top-p/nucleus, KV-cache). Cells:
1. Repo-root cell; imports incl. `from model import GPT, sample`; device; seed.
2. **Load the checkpoint**: rebuild the model from saved `config`, `load_state_dict`, recover `stoi`/`itos`; define `encode`/`decode` helpers. If `checkpoints/model.pt` is missing, print a clear instruction to run notebook 06 first and stop.
3. **Perplexity** on the val split (`exp(mean cross-entropy)`); markdown defines it.
4. **Sampling strategies**: generate from a seed string (e.g., `"\n"` or `"ROMEO:"`) with several settings and print each sample:
   - greedy (`temperature=0.0`)
   - `temperature=0.8`
   - `temperature=1.0, top_k=50`
   - `temperature=1.0, top_p=0.9`
   Markdown explains how each changes the output's diversity/quality.
5. **KV-cache demo + correctness check** (the canonical Phase 2 test): greedy-generate the same continuation with and without the cache and assert identical output; then time both to show the cache is faster:
```python
seed = torch.tensor([[stoi[c] for c in "ROMEO:"]], device=device)
out_cached = model.generate(seed.clone(), 200, temperature=0.0, use_cache=True)
out_plain  = model.generate(seed.clone(), 200, temperature=0.0, use_cache=False)
assert torch.equal(out_cached, out_plain), "KV-cache must not change the output"
print(decode(out_cached[0].tolist()))
```
   (Keep total length ≤ `block_size`.) Optionally print wall-clock timings for cached vs non-cached.
6. Closing markdown: what we built across Phase 2, and a teaser for Phase 3 (tuning, BPE, MoE).

- [ ] **Step 2: Run as script** (exit 0; the KV-cache equality assert holds). Requires `checkpoints/model.pt` from Task 6 to exist.
- [ ] **Step 3: Render** the `.ipynb`.
- [ ] **Step 4: Commit**
```bash
git add notebooks/07_eval_and_generation.py notebooks/07_eval_and_generation.ipynb
git commit -m "feat: notebook 07 — evaluation, sampling, and KV-cache generation"
```

---

## Self-Review

**Spec coverage (Phase 2 scope = notebooks 03–07 + the cwd retrofit + Phase-2 decisions):**
- Attention from scratch (SDPA, causal, multi-head) + **GQA implemented** → Task 2, Task 4. ✓
- Modern components RoPE / RMSNorm / SwiGLU each vs predecessor → Task 3, Task 4. ✓
- Assembled pre-norm model, weight tying, param count (~10–15M **full** config) → Task 4, Task 5. ✓
- Training loop, overfit-one-batch sanity, full training run, loss curves, checkpointing → Task 5 (overfit), Task 6. ✓
- Evaluation, perplexity, sampling (greedy/temp/top-k/top-p), **KV-cache + cached==non-cached verification** → Task 7. ✓
- **`model.py` single source of truth**, notebooks 06–07 import it → Tasks 4–7, Global Constraints. ✓
- **dropout** configurable → `GPTConfig`, Task 4. ✓
- **repo-root chdir cell** in every notebook incl. retrofit of 00–02 → Task 1, Plan Conventions. ✓
- Explanation standard applied to all new notebooks → every notebook task. ✓
- Real tests for the core module (`test_model.py`) → Task 4. ✓

**Placeholder scan:** No TBD/TODO. Core code (`model.py`, `test_model.py`) is complete; notebook tasks specify complete code cells + teaching beats, with prose written to the documented explanation standard (not a placeholder — a binding convention). ✓

**Type/name consistency:** `GPT.forward` returns `(logits, loss, presents)` consistently; `generate(...)` signature matches between `model.py` and notebook 07 usage; `GPTConfig`/`NANO_CONFIG`/`DEFAULT_CONFIG`, `build_rope`/`apply_rope`/`rotate_half`, `sample` names consistent across Tasks 4–7; checkpoint dict keys (`model_state`/`config`/`stoi`/`itos`) written in Task 6 and read in Task 7. `n_kv_head` divides `n_head` (asserted). ✓
