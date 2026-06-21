# Build an LLM from Scratch

![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-v2.12.1-blue?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)

A progressive, notebook-driven course that builds a modern decoder-only language model from raw text all the way up to a Mixture-of-Experts variant — entirely from scratch in PyTorch (no HuggingFace), sized to train on a laptop in minutes.

## The Learning Journey

```mermaid
graph LR
  A[Bag-of-Words] --> B[Embeddings]
  B --> C[Attention]
  C --> D[Modern Components<br/>RoPE/RMSNorm/SwiGLU]
  D --> E[Llama-style Block]
  E --> F[Full Transformer Training]
  F --> G[Mixture-of-Experts]
```

Each new method earns its place by beating a **measured** number from the step before. Each notebook teaches step by step in plain language, with runnable code, inline plots, and `assert`-based sanity checks that act as mathematical "grade" on your implementation.

## The Curriculum

| #  | Notebook | What you build |
|----|----------|----------------|
| 00 | Setup & tour | environment check, auto device selection, the roadmap |
| 01 | Data & Bag-of-Words | word-level tokenizer; a BoW next-word baseline (the order-blind starting point) |
| 02 | Embeddings | dense embeddings that beat BoW with far fewer parameters; word geometry |
| 03 | Attention | scaled dot-product $\rightarrow$ causal mask $\rightarrow$ multi-head $\rightarrow$ grouped-query attention (GQA) |
| 04 | Modern components | RoPE, RMSNorm, SwiGLU — each vs the older idea it replaces |
| 05 | Assembling the model | tour of the full Llama-style decoder block (imported from `model.py`), weight tying, param counts |
| 06 | Training | the training loop, loss curves, checkpointing |
| 07 | Evaluation & generation | perplexity, sampling (greedy/temperature/top-k/top-p), KV-cache |
| 08 | Tuning | learning-rate warmup + cosine decay, hyperparameter sweeps |
| 09 | BPE tokenizer | Byte-Pair Encoding from scratch vs char-level ($\approx$2x compression) |
| 10 | Mixture-of-Experts | top-k routing + load balancing; the capstone |

## Architecture Overview

The reusable model lives in [`model.py`](model.py). It implements a modern Llama-style architecture:

*   **Positional Embeddings:** Rotary Positional Embeddings (RoPE)
*   **Normalization:** RMSNorm (Root Mean Square Layer Normalization)
*   **Activation Function:** SwiGLU
*   **Attention Mechanism:** Grouped-Query Attention (GQA)
*   **Inference Optimization:** KV-Caching for efficient generation

## Setup

### Prerequisites
*   Python 3.10+
*   `pip` or `uv`
*   `jupyterlab` and `jupytext`

### Installation
```bash
# Using standard pip
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
pip install -r requirements.txt

# OR using uv (recommended for speed)
uv venv && uv pip install -r requirements.txt
```

### Hardware Support
The notebooks automatically detect and use the best available compute engine:
- **NVIDIA GPU (CUDA)** on Windows or Linux — preferred for training.
- **Apple Silicon GPU (MPS)** on macOS (M1–M5) — high performance on Mac.
- **CPU** — universal fallback; works on any machine, but significantly slower for training.

> **Windows / Linux NVIDIA users:** Ensure you have a CUDA-enabled PyTorch build installed from [pytorch.org](https://pytorch.org). The default `pip install torch` may only provide CPU support.

## How to Run & Learn

### Running Notebooks
Notebooks live in `notebooks/` as paired files: a jupytext `.py` source (the version-controlled truth) and a generated `.ipynb`. Open the `.ipynb` in JupyterLab for an interactive experience:

```bash
jupyter lab
```

Alternatively, run a notebook headlessly:
```bash
python notebooks/01_data_and_bag_of_words.py
```

### Running on Google Colab
You can also run this project directly in a [Google Colab](https://colab.research.google.com/) notebook. 

To get started, open a new Colab notebook and run the following cell:

```python
!git clone https://github.com/orbek/train-llm.git
%cd train-llm
!pip install -r requirements.txt
```

**Tip:** For best performance during training, go to `Runtime` $\rightarrow$ `Change runtime type` and select **T4 GPU**.

### How to Learn Effectively
*   **Observe the Plots:** Each step includes inline plots to visualize how embeddings or attention patterns change as you add complexity.
*   **The "Assert" Check:** Every notebook contains `assert` statements that validate your implementation against mathematical ground truths. **If an assertion fails, stop and investigate!** It means your code isn't behaving like a real LLM component should.
*   **Modify & Break:** The best way to learn is to change a hyperparameter (e.g., the number of attention heads or the learning rate) and observe how it impacts the loss curves in Notebook 06.

## Results
By notebook 07, you will have a ~9.4M-parameter model that generates recognizable Shakespearean text:

```text
ROMEO:
The sea of the sea that would not see the sea
That thou hast seen the sea of the sea
```

*(Note: The committed training run is capped for fast notebook rendering; raise `max_iters` in notebook 06 for sharper samples.)*

