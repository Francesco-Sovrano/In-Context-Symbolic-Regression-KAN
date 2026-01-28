# KAN (refactor) — Gated Matching Pursuit + Symbolic Regression

This repository is a refactor / re-organization of the popular **`pykan`** package (Kolmogorov–Arnold Networks), with a focus on **fast + stable symbolic regression** using:

- **gated operator layers** (train-time sparsity toward discrete symbolic atoms), and
- a **Matching-Pursuit-style** greedy symbolic conversion workflow.

It implements the methods described in *“Fast Stable Symbolic Regression in KANs via Gated Matching Pursuit”* (see the paper for conceptual details).

---

## Repository layout

- `symbolic_kan/` — the library code:
  - `MultKAN` / `KAN` (KAN is an alias for MultKAN)
  - gated symbolic layers (`GatedSymbolicLayer`)
  - symbolic regression utilities (`baseline_symbolic_regression`, `greedy_symbolic_regression`, etc.)
  - the operator library registry (`SYMBOLIC_LIB`) and helpers (`add_symbolic`, safe ops, dataset utils)

- `example_simple.py` — end-to-end demo:
  - build a toy dataset for `f(x, y) = exp(sin(pi*x) + y^2)`
  - train + prune a KAN
  - run one of the supported symbolic regression modes
  - print the final symbolic formula

- `example_logical_features.py` — custom symbolic primitives demo:
  - shows how to register **new symbolic functions** (e.g. step / smooth sign / smooth ReLU)
  - uses `add_symbolic(...)` to extend the operator library used by symbolic regression / gated layers

- `model/` — **auto-generated checkpoints** (created by default by `MultKAN` when training).  
  You can delete it safely; see “Checkpointing” below.

---

## Setup

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .\.venv\Scripts\activate  # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

Notes:

- `torch` may require a platform-specific install (CPU/CUDA). If `pip install torch` installs the wrong build, install PyTorch via the official instructions for your platform, then re-run `pip install -r requirements.txt`.
- `requirements.txt` includes **`pykan`**. The current `example_simple.py` imports `kan.*` (from `pykan`) for optional comparisons, so you either need `pykan` installed **or** you can remove that import if you don’t use the legacy path.

---

## Run `example_simple.py`

From the repository root, you **must** pick a symbolic regression mode via `--symbolic_regression_method`:

```bash
# 1) Baseline: post-hoc symbolic fitting (fastest, simplest)
python example_simple.py --symbolic_regression_method baseline

# 2) Greedy matching pursuit: post-hoc greedy symbolic selection (often better)
python example_simple.py --symbolic_regression_method greedy_matching_pursuit --simplify

# 3) Gated + greedy matching pursuit: train with gated symbolic atoms, then greedy selection (most interpretable)
python example_simple.py --symbolic_regression_method gated_greedy_matching_pursuit --simplify
```

### What you’ll see

- dataset tensor shapes
- training / pruning logs
- a summary from the selected symbolic regression routine
- the final exported formula from `model.symbolic_formula(...)`

### Useful knobs (common)

- `--width 2 5 1` : network width (list)
- `--grid 20` and `--grid_range -1 1` : spline grid and input range
- `--steps 500`, `--lr 1e-2`, `--lamb 1e-2` : training hyperparameters
- `--prune_iters 1`, `--node_th 0.1`, `--edge_th 0.0` : pruning controls
- `--simplify / --no-simplify` : simplify the exported SymPy expression
- gated mode only: `--gating_entropy`, `--gating_l1`, `--gate_top_k_start`, `--gate_top_k_min`

---

## Checkpointing (`./model/`)

`MultKAN` defaults to `auto_save=True` and `ckpt_path="./model"`, so training will create checkpoint files under `./model/` automatically.

If you don’t want checkpoints, set `auto_save=False` when constructing the model, or change `ckpt_path`:

```python
from symbolic_kan.MultKAN import KAN

model = KAN(
    width=[2, 5, 1],
    grid=20,
    grid_range=[-1, 1],
    auto_save=False,          # disable
    # ckpt_path="./my_ckpts", # or redirect
)
```

---

## Extending the symbolic operator library with `add_symbolic`

The symbolic regression code and gated symbolic layers both rely on a global registry `SYMBOLIC_LIB` (in `symbolic_kan/utils.py`).
To add your own primitive so it can be referenced by name in `lib=[...]` / `atom_names=[...]`, use:

```python
from symbolic_kan.utils import add_symbolic
import sympy as sp
import torch

# Example: a smooth step gate (useful as a differentiable “if” primitive)
def smooth_step(z, k=20.0):
    return torch.sigmoid(k * z)

add_symbolic(
    name="step",
    fun=lambda z: smooth_step(z),                 # torch implementation (Tensor -> Tensor)
    c=1,                                          # optional complexity cost
    sympy_fun=lambda z: sp.Function("step")(z),   # how it should appear in SymPy output
)
```

After registering, you can use the new primitive by name:

```python
lib = ["0", "x", "x^2", "abs", "step"]  # "step" is now valid
```

### Using custom primitives in practice

**A) Post-hoc symbolic regression** (no gated layers):

```python
summary = model.greedy_symbolic_regression(dataset, lib=lib, top_k_gates=3, steps=100, lr=1e-2)
formula = model.symbolic_formula(simplify=True)
```

**B) Train-time gated symbolic selection** (gated layers enabled):

```python
from symbolic_kan.MultKAN import KAN

model = KAN(
    width=[2, 5, 1],
    grid=20,
    grid_range=[-1, 1],
    atom_names=lib,   # <-- enables gated symbolic layers over your (extended) library
)
```

### See a full working example

`example_logical_features.py` demonstrates this pattern end-to-end by registering logical / piecewise-style primitives:

- `step` (smooth gate)
- `relu` (smooth hinge)
- `sgn_smooth` (smooth sign)

…and then using them in `safe_lib = [...]` passed into `greedy_symbolic_regression(...)`.

Run it from the repo root:

```bash
python example_logical_features.py --simplify
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'kan'`**  
  Install `pykan` (included in `requirements.txt`) or remove the unconditional `from kan.MultKAN import ...` import from `example_simple.py`.

- **Slow training**  
  Reduce `--steps`, reduce `--grid`, reduce hidden width, or prune more aggressively (especially in gated mode with `--gate_top_k_*`).

- **Messy / unstable formulas**  
  Increase sparsity pressure (`--gating_entropy`, `--gating_l1`), prune earlier/more, or restrict `lib` to a smaller operator set.
