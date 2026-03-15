# KAN (refactor) — Reproducible OFAT Ablation for the Paper: In-Context Symbolic Regression for Robustness-Improved Kolmogorov-Arnold Networks

This repository contains a refactored / reorganized implementation of **Kolmogorov–Arnold Networks (KANs)** with support for:

- standard post-hoc symbolic extraction (**AutoSym-style baseline**),
- **greedy in-context symbolic regression**,
- **gated operator layers** with greedy refinement,
- and the **one-factor-at-a-time (OFAT) robustness pipeline** used in the paper.

The main script used to reproduce the paper runs is:

- **`ablation.py`**

This script runs the five pipelines compared in the paper over multiple Feynman datasets and OFAT configurations, and writes one row per method-run to a CSV file.

---

## What this repository contains

- `symbolic_kan/` — core library code
  - `MultKAN` / `KAN`
  - symbolic regression utilities
  - gated symbolic layers
  - operator library and helpers

- `ablation.py` — **paper pipeline**
  - runs the OFAT ablation on up to `N` randomly selected Feynman datasets
  - evaluates all five methods for each configuration
  - saves raw run results to CSV

- `example_simple.py` — small end-to-end demo
- `example_logical_features.py` — demo for custom symbolic primitives

---

## Environment setup

### 1) Create and activate a virtual environment

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

### Notes

- `torch` may require a platform-specific installation (CPU / CUDA / Apple Silicon). If needed, install the correct PyTorch build first using the official PyTorch instructions, then run `pip install -r requirements.txt`.
- The paper pipeline uses the local `symbolic_kan` package directly.
- Run all commands from the **repository root**.

---

## Dataset layout

The paper pipeline expects local Feynman datasets (available at https://space.mit.edu/home/tegmark/aifeynman.html) under:

```text
symbolic_kan/datasets/
```

with a variant subdirectory such as:

```text
symbolic_kan/datasets/Feynman_with_units/
```

and, optionally, the equation metadata file:

```text
symbolic_kan/datasets/FeynmanEquations.csv
```

If `--equations_csv` is not passed explicitly, `ablation.py` will look for `symbolic_kan/datasets/FeynmanEquations.csv` automatically.

---

## Main paper pipeline: `ablation.py`

The script evaluates the following five methods:

1. `baseline`
2. `fastkan_baseline`
3. `greedy_matching_pursuit`
4. `fastkan_greedy_matching_pursuit`
5. `gated_greedy_matching_pursuit`

### OFAT factors

The script varies one factor at a time around a fixed reference configuration:

- `width_mid ∈ {5,2; 10,2; 20,2; 50,2; 100,2}`
- `lamb ∈ {1e-4, 1e-3, 1e-2, 1e-1}`
- `prune_iters ∈ {1, 3, 5}`
- `seed ∈ {1, 2, 3}`

### Reference configuration

By default, the reference configuration is:

- `width_mid = 5,2`
- `lamb = 1e-2`
- `prune_iters = 3`
- `seed = 1`

This yields:

- **15 OFAT configurations per dataset**
- **5 methods per configuration**
- therefore **75 runs per dataset**

If `--max_datasets 10` is used, the full run produces:

- **750 runs total**

---

## Important reproducibility details

### Random dataset selection

The datasets are **not selected alphabetically**. The script:

1. lists all datasets in the chosen Feynman variant,
2. shuffles them with `--dataset_select_seed`,
3. takes the first `--max_datasets` datasets.

This means dataset selection is:

- **random**
- **without replacement**
- **fully reproducible** if `--dataset_select_seed` is fixed.

### Per-run randomness

The OFAT `seed` factor controls:

- model initialization,
- NumPy random state,
- and, when `--split_strategy random` is used, the train/test split.

So reproducibility requires fixing both:

- the dataset-selection seed: `--dataset_select_seed`
- the OFAT seed grid: `--seed_grid`

---

## Output format

The script writes **one CSV row per method-run**.

Each row includes, among other fields:

- dataset name
- method name
- OFAT factor
- seed
- `width_mid`
- `lamb`
- `prune_iters`
- train MSE
- test MSE
- predicted symbolic formula
- timing information (if enabled)

If a run fails, the script still appends a row containing the configuration and an `error` field. This is important for transparent reporting of unavailable runs.

---

## Key command-line arguments

### Data and dataset selection

- `--feynman_root` — root dataset directory
- `--feynman_variant` — dataset variant, e.g. `Feynman_with_units`
- `--equations_csv` — optional equation metadata CSV
- `--max_datasets` — how many datasets to use
- `--dataset_select_seed` — random seed controlling dataset subset selection

### Sampling

- `--train_num` — max number of training samples per dataset
- `--test_num` — max number of test samples per dataset
- `--split_strategy` — `random` or `linspace`

### Device

- `--device` — `cpu`, `cuda`, or `mps`

### Reference configuration

- `--baseline_width_mid`
- `--baseline_lamb`
- `--baseline_prune_iters`
- `--baseline_seed`

### OFAT grids

- `--width_mid_grid`
- `--lamb_grid`
- `--prune_iters_grid`
- `--seed_grid`

### Training

- `--grid`
- `--lr`
- `--steps`
- `--reg_metric`
- `--node_th`
- `--edge_th`

### Gated method

- `--gating_entropy`
- `--gating_l1`
- `--top_k_gates`
- `--gate_top_k_start`
- `--regression_policy`

### Output

- `--output_csv`
- `--append / --no-append`
- `--timing / --no-timing`
- `--simplify / --no-simplify`

---

## Reproducing the paper run

To make runs reproducible, keep the following fixed:

1. the code version / commit,
2. the Python environment,
3. the dataset files,
4. `--dataset_select_seed`,
5. all OFAT grids,
6. the device type,
7. the output of failed runs as well as successful runs.

To reproduce exactly the paper results, run:

```bash
python ablation.py \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
  --device mps \
  --output_csv results/ablation_ofat_paper10.csv \
  --datasets \
    feynman_I_9_18 \
    feynman_I_10_7 \
    feynman_I_12_1 \
    feynman_I_12_4 \
    feynman_I_13_4 \
    feynman_I_34_1 \
    feynman_II_6_15a \
    feynman_II_6_15b \
    feynman_II_21_32 \
    feynman_II_34_29a
  --append
```

For CPU, replace:

```bash
--device mps
```

with:

```bash
--device cpu
```

For CUDA, replace it with:

```bash
--device cuda
```

If you want a random but reproducible subset instead of an explicit list:

```bash
python ablation.py \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
  --device mps \
  --output_csv results/ablation_random_10ds.csv \
  --max_datasets 10 \
  --dataset_select_seed 123 \
  --append
```

---

## Notes on interpretation

The CSV generated by `ablation.py` contains the raw runs used for the paper analysis.

In the paper:

- **seed sensitivity** is computed from the rows where `ofat_factor = seed`, using the fixed reference configuration,
- **OFAT sensitivity** is computed by aggregating rows where `ofat_factor ∈ {width_mid, lamb, prune_iters}`.

These are different summaries and should not be interpreted as the same quantity.

---

## Troubleshooting

### No datasets found

Check that the selected variant exists, for example:

```text
symbolic_kan/datasets/Feynman_with_units/
```

### Wrong Torch build

Install the correct PyTorch wheel for your platform first, then reinstall the remaining requirements.

### Long runtime

This pipeline is intentionally large. To reduce runtime for quick tests:

- lower `--max_datasets`
- reduce `--steps`
- use fewer OFAT values
- run on `cuda` or `mps` if available

### Interrupted runs

The script appends rows progressively. If interrupted, previously completed runs remain stored in the CSV.

---

## Optional demos

These are not part of the paper pipeline, but remain useful for quick sanity checks:

```bash
python example_simple.py
python example_logical_features.py --simplify
```

---

## Citation / paper context

This codebase supports the experiments reported in the KAN symbolic regression paper, including the robustness analysis based on OFAT sweeps and seed sensitivity.
