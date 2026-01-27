# KAN (refactor) — Gated Matching Pursuit + Symbolic Regression

This repository is a **refactor / re-organization of the popular `pykan` package** (Kolmogorov–Arnold Networks), with a focus on **fast + stable symbolic regression** using **gated operator layers** and a **Matching-Pursuit-style** symbolic conversion workflow.

It implements the methods described in the attached paper *“Fast Stable Symbolic Regression in KANs via Gated Matching Pursuit”*.  
(See the paper for the conceptual details: gated operator mixtures, in-context greedy selection, top-k pruning, and discretization.)

---

## What’s inside

- `kan/` — the library code (KAN / MultKAN, layers, utilities)
- `kan_example.py` — a runnable example showing:
  - dataset creation
  - training a KAN with a **GatedSymbolicLayer**
  - optional pruning / gate inspection
  - **greedy in-context symbolic regression**
  - printing a final symbolic formula

---

## Setup

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .\.venv\Scripts\activate  # Windows PowerShell
````

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

**Notes**

* `torch` may require a platform-specific install (CPU/CUDA). If `pip install torch` fails or installs the wrong build, install PyTorch following the official instructions for your platform, then re-run `pip install -r requirements.txt`.
* `openai` / `ollama` appear in `requirements.txt`; they are not required to run the basic `kan_example.py` path unless you wire them into your own experiments.

---

## Run the example

From the repository root:

```bash
python kan_example.py
```

### Optional: simplify the printed expression

The script supports a `--simplify` flag to request simplification of the produced symbolic expression:

```bash
python kan_example.py --simplify
```

What the script does (high-level):

* Builds a toy dataset for: `f(x, y) = exp(sin(pi*x) + y^2)`
* Creates a KAN with an operator library like: `['0','1','x','x^2','exp','log','sqrt','tanh','sin','abs']`
* Trains with gating regularizers (entropy + L1-style terms) to push gates toward sparse / near-discrete choices
* Optionally prunes and checks gate statistics
* Runs greedy in-context symbolic regression (a Matching Pursuit–style refinement) restricted to the **top-k gated candidates**
* Prints the resulting symbolic formula

---

## Typical workflow for your own experiment

1. Define your target function / dataset (or load your data)
2. Choose an operator library (`atom_names`)
3. Train with gating enabled (entropy/L1 regularization + optional top-k pruning)
4. Discretize / refine via greedy in-context symbolic regression
5. Export / inspect the final symbolic formula

---

## Troubleshooting

* **Import errors**: make sure you’re running from the repo root and your venv is activated.
* **Slow training**: reduce `steps`, reduce `grid`, reduce hidden width, or prune more aggressively (`gate_top_k`).
* **Unstable / messy formulas**: increase sparsity pressure (`gating_entropy`, `gating_l1`), prune earlier/more, or restrict the operator library.

