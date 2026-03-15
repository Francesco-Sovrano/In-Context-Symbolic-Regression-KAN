# In-Context Symbolic Regression for Robustness-Improved Kolmogorov-Arnold Networks

This repository is the replication package for the paper **"In-Context Symbolic Regression for Robustness-Improved Kolmogorov-Arnold Networks"**.

It contains:

- a refactored KAN implementation in `symbolic_kan/`
- the main experimental driver used for the paper, `ablation.py`
- lightweight examples for sanity checks and smaller runs

The main entry point for reproducing the paper experiments is:

- **`ablation.py`**

`ablation.py` evaluates five symbolic-regression pipelines on selected Feynman datasets and writes one CSV row per method/configuration run.

---

## Repository layout

```text
.
├── ablation.py
├── example_feynman.py
├── example_simple.py
├── requirements.txt
└── symbolic_kan/
```

### Main files

- `symbolic_kan/` — core KAN implementation and symbolic-regression utilities
- `ablation.py` — main paper pipeline for the OFAT robustness study
- `example_feynman.py` — example run on a local Feynman dataset
- `example_simple.py` — small synthetic end-to-end demo
- `requirements.txt` — Python dependencies

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

- `torch` may require a platform-specific install for CPU, CUDA, or Apple Silicon. If needed, install the correct PyTorch build first, then run `pip install -r requirements.txt`.
- Run all commands from the repository root.
- The paper experiments use the local `symbolic_kan` package directly.

---

## Getting the Feynman data

The repository **does not bundle the Feynman benchmark files**. To reproduce the paper runs, download them separately (available at [https://space.mit.edu/home/tegmark/aifeynman.html](https://space.mit.edu/home/tegmark/aifeynman.html)) and place them under `symbolic_kan/datasets/`.

The official AI Feynman repository and documentation both point users to the **Feynman Symbolic Regression Database** for benchmark data. The database is hosted on Max Tegmark's MIT page.

### What you need

At minimum, download:

- the **`Feynman_with_units/`** dataset directory
- **`FeynmanEquations.csv`**

### How to find and download `Feynman_with_units`

1. Open the official AI Feynman repository or documentation page: [https://space.mit.edu/home/tegmark/aifeynman.html](https://space.mit.edu/home/tegmark/aifeynman.html).
2. Follow the link to the **Feynman Symbolic Regression Database**.
3. Download the benchmark dataset archive from that page.
4. Extract the archive locally.
5. Inside the extracted contents, locate:
   - `Feynman_with_units/`
   - `FeynmanEquations.csv`
6. Create the local dataset directory expected by this repository:

```bash
mkdir -p symbolic_kan/datasets
```

7. Copy the downloaded files into that directory so the layout becomes:

```text
symbolic_kan/datasets/
├── FeynmanEquations.csv
└── Feynman_with_units/
    ├── I.10.7
    ├── I.12.1
    ├── I.12.4
    └── ...
```

### Quick verification

Check that the dataset is visible from the repository root:

```bash
ls symbolic_kan/datasets/Feynman_with_units | head
```

You should see filenames such as `I.10.7`, `I.12.1`, and similar equation identifiers.

### Optional variants

`ablation.py` also supports these variants if you have downloaded them:

- `Feynman_without_units`
- `bonus_with_units`
- `bonus_without_units`

If `--equations_csv` is not passed explicitly, `ablation.py` looks for:

```text
symbolic_kan/datasets/FeynmanEquations.csv
```

---

## Quick sanity checks

### Synthetic example

```bash
python example_simple.py --symbolic_regression_method gated_greedy_matching_pursuit
```

### Feynman example

After downloading the dataset, you can run a single Feynman problem with:

```bash
python example_feynman.py \
  --symbolic_regression_method gated_greedy_matching_pursuit \
  --feynman_name I.26.2 \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --device cpu
```

---

## Main paper pipeline: `ablation.py`

The script compares five methods:

1. `baseline`
2. `fastkan_baseline`
3. `greedy_matching_pursuit`
4. `fastkan_greedy_matching_pursuit`
5. `gated_greedy_matching_pursuit`

### OFAT factors

The one-factor-at-a-time (OFAT) sweep varies one factor around a fixed reference configuration:

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

This produces:

- **15 OFAT configurations per dataset**
- **5 methods per configuration**
- therefore **75 runs per dataset**

With `--max_datasets 10`, the full study produces **750 runs**.

---

## Reproducing the paper run

To reproduce the paper tables and figures as closely as possible, keep the following fixed:

1. code version / commit
2. Python environment
3. downloaded dataset files
4. dataset selection or explicit dataset list
5. OFAT grids
6. device type
7. failed runs as well as successful runs

### Exact dataset list used in the paper run

```bash
python ablation.py \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
  --device cpu \
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
```

Replace `--device cpu` with `--device cuda` or `--device mps` if appropriate for your machine.

### Reproducible random subset

If you want a random but reproducible subset instead of the fixed list above:

```bash
python ablation.py \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
  --device cpu \
  --output_csv results/ablation_random_10ds.csv \
  --max_datasets 10 \
  --dataset_select_seed 123 
```

Dataset selection is:

- random
- without replacement
- reproducible when `--dataset_select_seed` is fixed

---

## Important reproducibility details

### Dataset selection

If `--datasets` is supplied, `ablation.py` uses exactly that explicit list.

Otherwise, it:

1. lists datasets in the selected Feynman variant
2. shuffles them using `--dataset_select_seed`
3. takes the first `--max_datasets`

### Per-run randomness

The OFAT `seed` factor controls:

- model initialization
- NumPy random state
- and, when `--split_strategy random` is used, the train/test split

To make results reproducible, fix both:

- `--dataset_select_seed`
- `--seed_grid`

---

## Output format

`ablation.py` writes **one CSV row per method-run**.

Each row includes configuration and result fields such as:

- dataset name
- filename
- target formula (when available)
- method name
- OFAT factor
- seed
- `width_mid`
- `lamb`
- `prune_iters`
- train MSE
- test MSE
- predicted symbolic formula
- timing information when enabled

If a run fails, the script still appends a row with the configuration and an `error` field. This is intentional and should be preserved for transparent reporting.

---

## Key command-line arguments

### Data and dataset selection

- `--feynman_root`
- `--feynman_variant`
- `--equations_csv`
- `--datasets`
- `--max_datasets`
- `--dataset_select_seed`

### Sampling

- `--train_num`
- `--test_num`
- `--split_strategy`

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
- `--timing / --no-timing`
- `--simplify / --no-simplify`

---

## Notes on interpretation

The CSV generated by `ablation.py` contains the raw runs used for the paper analysis.

In the paper:

- **seed sensitivity** is computed from rows where `ofat_factor = seed`, while all other hyperparameters stay at the reference configuration
- **OFAT sensitivity** is computed by aggregating rows where `ofat_factor ∈ {width_mid, lamb, prune_iters}`

These are different summaries and should not be interpreted as the same quantity.

---

## Troubleshooting

### No datasets found

Check that this directory exists:

```text
symbolic_kan/datasets/Feynman_with_units/
```

### `FeynmanEquations.csv` not found

Either:

- place it at `symbolic_kan/datasets/FeynmanEquations.csv`, or
- pass its path explicitly with `--equations_csv`

### Wrong Torch build

Install the correct PyTorch wheel for your platform first, then reinstall the remaining requirements.

### Long runtime

This pipeline is intentionally large. For quick tests:

- lower `--max_datasets`
- reduce `--steps`
- use fewer OFAT values
- run on `cuda` or `mps` if available

### Interrupted runs

The script appends rows progressively. If interrupted, previously completed runs remain in the CSV.

---

## Citation

If you use this repository, please cite the associated paper. A BibTeX entry is provided below.

```bibtex
@inproceedings{sovrano2026incontext,
  author    = {Francesco Sovrano and Lidia Losavio and Giulia Vilone and Marc Langheinrich},
  title     = {In-Context Symbolic Regression for Robustness-Improved Kolmogorov-Arnold Networks},
  booktitle = {eXplainable Artificial Intelligence. 4th World Conference on eXplainable Artificial Intelligence},
  year      = {2026},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  url       = {https://xaiworldconference.com/2026/}
}
```

If you also discuss the benchmark origin, cite the original AI Feynman work and the Feynman Symbolic Regression Database used to distribute the datasets.
