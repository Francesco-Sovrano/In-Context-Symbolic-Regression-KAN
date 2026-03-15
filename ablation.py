# ablation_ofat_10ds_5methods.py
#
# OFAT ablation on first N Feynman datasets, running 5 methods per config.
#
# Methods (matching example_feynman.py):
#   baseline
#   fastkan_baseline
#   greedy_matching_pursuit
#   fastkan_greedy_matching_pursuit
#   gated_greedy_matching_pursuit
#
# OFAT factors:
#   width_mid   in {[5,2],[10,2],[20,2],[50,2],[100,2]}
#   lamb        in {1e-4, 1e-3, 1e-2, 1e-1}
#   prune_iters in {1, 3, 5}
#   seed        in {1, 2, 3}
#
# Total configs per dataset = 5 + 4 + 3 + 3 = 15 (OFAT)
# Total runs per dataset     = 15 * 5 = 75
#
# Example:
#   python3 ablation_ofat_10ds_5methods.py \
#     --feynman_root symbolic_kan/datasets \
#     --feynman_variant Feynman_with_units \
#     --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
#     --device mps \
#     --output_csv ablation_ofat_10ds_5methods.csv \
#     --max_datasets 10

import argparse
import time
from contextlib import contextmanager
import os
import glob
import numpy as np

import torch
import torch.nn.functional as F

from symbolic_kan.MultKAN import KAN, GatedSymbolicLayer

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# -------------------------
# CLI
# -------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="OFAT ablation on first N Feynman datasets; run 5 methods per config; save one row per method-run."
    )

    # Data
    p.add_argument("--feynman_root", type=str, default="symbolic_kan/datasets")
    p.add_argument(
        "--feynman_variant",
        type=str,
        default="Feynman_with_units",
        choices=["Feynman_without_units", "Feynman_with_units", "bonus_without_units", "bonus_with_units"],
    )
    p.add_argument("--equations_csv", type=str, default=None)
    p.add_argument("--max_datasets", type=int, default=10)

    # Sampling
    p.add_argument("--train_num", type=int, default=2000)
    p.add_argument("--test_num", type=int, default=1000)
    p.add_argument("--split_strategy", choices=["random", "linspace"], default="random")

    # Device
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    # Fixed training knobs
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--reg_metric", choices=["node_backward", "edge_backward", "edge_forward_spline_u"], default="edge_backward")

    # Pruning knobs (thresholds fixed; prune_iters varies in OFAT)
    p.add_argument("--node_th", type=float, default=0.1)
    p.add_argument("--edge_th", type=float, default=0.0)
    p.add_argument("--gate_top_k_start", type=int, default=10)

    # OFAT baseline center
    p.add_argument("--baseline_width_mid", type=str, default="5,2")
    p.add_argument("--baseline_lamb", type=float, default=1e-2)
    p.add_argument("--baseline_prune_iters", type=int, default=3)
    p.add_argument("--baseline_seed", type=int, default=1)

    # OFAT grids
    p.add_argument("--width_mid_grid", nargs="+", type=str, default=["5,2", "10,2", "20,2", "50,2", "100,2"])
    p.add_argument("--lamb_grid", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument("--prune_iters_grid", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--seed_grid", nargs="+", type=int, default=[1, 2, 3])

    # Gated best fixed hyperparams (your found optimum)
    p.add_argument("--gating_entropy", type=float, default=1e-3)
    p.add_argument("--gating_l1", type=float, default=1e-2)
    p.add_argument("--top_k_gates", type=int, default=5)
    p.add_argument("--regression_policy", type=str, default="worst",
                   choices=["best", "worst", "ltr", "rtl", "random"])

    # Output
    p.add_argument("--output_csv", type=str, default="ablation_ofat_10ds_5methods.csv")
    p.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


# -------------------------
# Helpers
# -------------------------
def parse_mid(mid: str):
    mid = mid.strip().replace(" ", "")
    a, b = mid.split(",")
    return [int(a), int(b)]


def _feynman_cli_to_filename(ds_name: str) -> str:
    if ds_name.lower().startswith("feynman_"):
        ds_name = ds_name[len("feynman_") :]
    parts = ds_name.split("_")
    return ".".join(parts)


def list_local_feynman_dataset_names(feynman_root: str, variant: str):
    base_dir = os.path.join(feynman_root, variant)
    if not os.path.isdir(base_dir):
        return []
    files = sorted(glob.glob(os.path.join(base_dir, "*")))
    names = []
    for fp in files:
        bn = os.path.basename(fp)
        if bn.startswith("."):
            continue
        names.append("feynman_" + bn.replace(".", "_"))
    return names


def load_local_feynman_dataset_as_kan(
    ds_name: str,
    feynman_root: str,
    variant: str,
    device: str,
    dtype=torch.float32,
    train_cap: int = 4000,
    test_cap: int = 2000,
    seed: int = 0,
    split_strategy: str = "linspace",
):
    filename = _feynman_cli_to_filename(ds_name)
    path = os.path.join(feynman_root, variant, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] < 2:
        raise ValueError(f"Dataset {ds_name} has <2 columns (need X + y). File: {path}")

    X = data[:, :-1].astype(np.float32)
    y = data[:, -1:].astype(np.float32)
    N = int(X.shape[0])

    n_tr = int(min(train_cap, N))
    n_te = int(min(test_cap, max(0, N - n_tr)))

    if split_strategy == "linspace":
        tr_idx = np.unique(np.round(np.linspace(0, N - 1, n_tr)).astype(int))
        te_all = np.unique(np.round(np.linspace(0, N - 1, n_tr + n_te)).astype(int))
        te_idx = te_all[~np.isin(te_all, tr_idx)][:n_te]
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        tr_idx = perm[:n_tr]
        te_idx = perm[n_tr : n_tr + n_te] if n_te > 0 else np.array([], dtype=int)

    train_input = torch.from_numpy(X[tr_idx]).to(device=device, dtype=dtype)
    train_label = torch.from_numpy(y[tr_idx]).to(device=device, dtype=dtype)

    if n_te > 0:
        test_input = torch.from_numpy(X[te_idx]).to(device=device, dtype=dtype)
        test_label = torch.from_numpy(y[te_idx]).to(device=device, dtype=dtype)
    else:
        test_input = torch.empty((0, X.shape[1]), device=device, dtype=dtype)
        test_label = torch.empty((0, 1), device=device, dtype=dtype)

    return {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": test_input,
        "test_label": test_label,
        "N_total": N,
        "N_train": int(train_input.shape[0]),
        "N_test": int(test_input.shape[0]),
    }


def _maybe_device_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


@contextmanager
def timed_block(label, timings, enabled=True):
    if not enabled:
        yield
        return
    _maybe_device_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _maybe_device_sync()
        dt = time.perf_counter() - t0
        timings[label] = timings.get(label, 0.0) + dt


def safe_predict(model, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        yhat = model.predict(x) if hasattr(model, "predict") else model(x)
    if isinstance(yhat, (list, tuple)):
        yhat = yhat[0]
    if yhat.ndim == 1:
        yhat = yhat.view(-1, 1)
    return yhat


def mse_loss(model, x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    yhat = safe_predict(model, x)
    return float(F.mse_loss(yhat, y).item())


def load_feynman_equations_map(equations_csv_path: str):
    if equations_csv_path is None or not os.path.isfile(equations_csv_path):
        return {}
    if _HAS_PANDAS:
        df = pd.read_csv(equations_csv_path)
        cols = [c.lower() for c in df.columns]
        name_col = df.columns[0]
        for cand in ["filename", "name", "equation", "id"]:
            if cand in cols:
                name_col = df.columns[cols.index(cand)]
                break
        formula_col = df.columns[-1]
        for cand in ["formula", "feynman", "tex", "equation", "rhs", "lhs", "target", "output"]:
            if cand in cols:
                formula_col = df.columns[cols.index(cand)]
                break
        out = {}
        for _, row in df.iterrows():
            k = str(row[name_col]).strip()
            v = str(row[formula_col]).strip()
            if k and k != "nan":
                out[k] = v
        return out
    return {}


def append_results_row(output_csv: str, row: dict, append: bool = True):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if _HAS_PANDAS:
        df_row = pd.DataFrame([row])
        if (not append) or (not os.path.isfile(output_csv)):
            df_row.to_csv(output_csv, index=False)
        else:
            df_row.to_csv(output_csv, mode="a", header=False, index=False)
        return

    import csv as _csv
    write_header = (not append) or (not os.path.isfile(output_csv))
    with open(output_csv, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def feynman_lib_full():
    return [
        "0", "1",
        "x",
        "x^2", "x^3", "x^4", "x^5",
        "1/x", "1/x^2", "1/x^3",
        "sqrt", "1/sqrt(x)",
        "log", "exp",
        "sin", "cos", "tan", "tanh",
        "abs", "sgn",
        "arctan", "arcsin", "arccos",
        "arctanh",
        "gaussian",
    ]


def build_ofat_configs(args):
    base = dict(
        seed=int(args.baseline_seed),
        width_mid=str(args.baseline_width_mid),
        lamb=float(args.baseline_lamb),
        prune_iters=int(args.baseline_prune_iters),
    )
    cfgs = []

    for w in args.width_mid_grid:
        c = dict(base)
        c["ofat_factor"] = "width_mid"
        c["width_mid"] = str(w)
        cfgs.append(c)

    for l in args.lamb_grid:
        c = dict(base)
        c["ofat_factor"] = "lamb"
        c["lamb"] = float(l)
        cfgs.append(c)

    for p in args.prune_iters_grid:
        c = dict(base)
        c["ofat_factor"] = "prune_iters"
        c["prune_iters"] = int(p)
        cfgs.append(c)

    for s in args.seed_grid:
        c = dict(base)
        c["ofat_factor"] = "seed"
        c["seed"] = int(s)
        cfgs.append(c)

    # add a stable id for grouping (dataset+config index is ok)
    for idx, c in enumerate(cfgs, start=1):
        c["config_idx"] = idx

    return cfgs


# -------------------------
# Run one method on one dataset/config
# -------------------------
def run_one(method: str, ds_name: str, cfg: dict, args, equations_map: dict):
    lib = feynman_lib_full()
    timings = {}
    wall_t0 = time.perf_counter()

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    with timed_block("dataset_load", timings, enabled=args.timing):
        dataset = load_local_feynman_dataset_as_kan(
            ds_name,
            feynman_root=args.feynman_root,
            variant=args.feynman_variant,
            device=args.device,
            seed=seed,
            train_cap=args.train_num,
            test_cap=args.test_num,
            split_strategy=args.split_strategy,
        )

    Xtr = dataset["train_input"]
    n_in = int(Xtr.shape[1])

    mid = parse_mid(cfg["width_mid"])
    width = [n_in, mid, 1]

    x_min = float(torch.min(Xtr).item())
    x_max = float(torch.max(Xtr).item())
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    f_range = [x_min, x_max]

    kan_kwargs = dict(
        width=width,
        grid=args.grid,
        grid_range=f_range,
        seed=seed,
    )

    # mirror your method logic:
    if method == "gated_greedy_matching_pursuit":
        kan_kwargs["atom_names"] = lib
    else:
        if "fastkan" in method:
            kan_kwargs["numeric_atom_configs"] = {"radial_bf": {"num_grids": args.grid}}
        else:
            kan_kwargs["numeric_atom_configs"] = {"bspline": {"num_grids": args.grid, "degree": 3}}

    with timed_block("model_init", timings, enabled=args.timing):
        model = KAN(**kan_kwargs)

    # training options (mirror your script: fit_initial without lamb, then prune/refit with lamb)
    training_options = dict(
        optimizer="Adam",
        lr=float(args.lr),
        steps=int(args.steps),
        reg_metric=args.reg_metric,
        gating_entropy=float(args.gating_entropy),
        gating_l1=float(args.gating_l1),
    )

    with timed_block("fit_initial", timings, enabled=args.timing):
        model.fit(dataset, **training_options)

    training_options["lamb"] = float(cfg["lamb"])

    # prune + refit rounds
    prune_iters = int(cfg["prune_iters"])
    if prune_iters > 0:
        gate_top_k_pruning_delta = (args.gate_top_k_start - args.top_k_gates) // max(1, prune_iters)
        for i in range(prune_iters):
            top_k = max(args.top_k_gates, args.gate_top_k_start - (i + 1) * gate_top_k_pruning_delta)
            with timed_block(f"prune_{i}", timings, enabled=args.timing):
                model = model.prune(node_th=args.node_th, edge_th=args.edge_th, gate_top_k=top_k)
            with timed_block(f"refit_{i}", timings, enabled=args.timing):
                model.fit(dataset, **training_options)

    # final fit with lamb=0
    training_options["lamb"] = 0.0
    with timed_block("fit_final", timings, enabled=args.timing):
        model.fit(dataset, **training_options)

    # symbolic regression post-pass
    pred_formula_str = None
    if method in ("baseline", "fastkan_baseline"):
        with timed_block("symbolic_regression", timings, enabled=args.timing):
            _ = model.baseline_symbolic_regression(lib=lib, weight_simple=0)

    elif method in ("greedy_matching_pursuit", "fastkan_greedy_matching_pursuit", "gated_greedy_matching_pursuit"):
        symbolic_training_options = dict(training_options)
        symbolic_training_options["steps"] = 100
        symbolic_training_options["lamb"] = 0.0
        with timed_block("symbolic_regression", timings, enabled=args.timing):
            _ = model.greedy_symbolic_regression(
                dataset,
                lib=lib,
                top_k_gates=int(args.top_k_gates),
                policy=str(args.regression_policy if method == "gated_greedy_matching_pursuit" else "best"),
                **symbolic_training_options,
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    # polish
    with timed_block("fit_final_polish", timings, enabled=args.timing):
        model.fit(dataset, **training_options)

    with timed_block("export_formula", timings, enabled=args.timing):
        symbolic_formula = model.symbolic_formula(simplify=args.simplify)
        if symbolic_formula:
            try:
                pred_formula_str = str(symbolic_formula[0][0])
            except Exception:
                pred_formula_str = str(symbolic_formula)

    with timed_block("loss_eval", timings, enabled=args.timing):
        train_mse = mse_loss(model, dataset["train_input"], dataset["train_label"])
        test_mse = mse_loss(model, dataset["test_input"], dataset["test_label"])

    total_wall = time.perf_counter() - wall_t0
    feynman_filename = _feynman_cli_to_filename(ds_name)
    target_formula = equations_map.get(feynman_filename, None)

    row = {
        "dataset": ds_name,
        "filename": feynman_filename,
        "target_formula": target_formula,
        "method": method,

        # OFAT grouping
        "config_idx": int(cfg["config_idx"]),
        "ofat_factor": str(cfg["ofat_factor"]),

        # ablated knobs
        "seed": int(cfg["seed"]),
        "width_mid": str(cfg["width_mid"]),
        "lamb": float(cfg["lamb"]),
        "prune_iters": int(cfg["prune_iters"]),

        # fixed knobs
        "device": args.device,
        "split_strategy": args.split_strategy,
        "grid": int(args.grid),
        "lr": float(args.lr),
        "steps": int(args.steps),
        "reg_metric": args.reg_metric,
        "node_th": float(args.node_th),
        "edge_th": float(args.edge_th),
        "gate_top_k_start": int(args.gate_top_k_start),

        # gated fixed best (logged always)
        "gating_entropy": float(args.gating_entropy),
        "gating_l1": float(args.gating_l1),
        "top_k_gates": int(args.top_k_gates),
        "regression_policy": str(args.regression_policy),

        # results
        "predicted_formula": pred_formula_str,
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),

        # sizes
        "N_total": dataset.get("N_total"),
        "N_train": dataset.get("N_train"),
        "N_test": dataset.get("N_test"),

        "timing_total_wall_s": float(total_wall),
    }
    return row


def main():
    args = get_args()
    print(args)

    if args.equations_csv is None:
        candidate = os.path.join(args.feynman_root, "FeynmanEquations.csv")
        args.equations_csv = candidate if os.path.isfile(candidate) else None

    equations_map = load_feynman_equations_map(args.equations_csv)
    if args.equations_csv:
        print(f"[INFO] Loaded equations map: {args.equations_csv} (rows={len(equations_map)})")
    else:
        print("[WARN] equations_csv not found; target_formula will be empty.")

    dataset_names = list_local_feynman_dataset_names(args.feynman_root, args.feynman_variant)
    if not dataset_names:
        raise RuntimeError(f"No datasets found in {os.path.join(args.feynman_root, args.feynman_variant)}")
    dataset_names = dataset_names[: int(args.max_datasets)]
    print(f"[INFO] Using first {len(dataset_names)} datasets.")

    if (not args.append) and os.path.isfile(args.output_csv):
        os.remove(args.output_csv)

    methods = [
        "baseline",
        "fastkan_baseline",
        "greedy_matching_pursuit",
        "fastkan_greedy_matching_pursuit",
        "gated_greedy_matching_pursuit",
    ]

    cfgs = build_ofat_configs(args)
    print(f"[INFO] OFAT configs per dataset: {len(cfgs)} (expected 15).")
    print(f"[INFO] Runs per dataset: {len(cfgs)} * {len(methods)} = {len(cfgs) * len(methods)} (expected 75).")

    total_runs = len(dataset_names) * len(cfgs) * len(methods)
    run_idx = 0

    for ds in dataset_names:
        for cfg in cfgs:
            # block of 5 rows: same dataset + same config_idx
            for method in methods:
                run_idx += 1
                print("\n" + "=" * 120)
                print(f"[{run_idx}/{total_runs}] DS={ds} | cfg={cfg['config_idx']:02d} factor={cfg['ofat_factor']}"
                      f" | seed={cfg['seed']} width={cfg['width_mid']} lamb={cfg['lamb']:.1e} prune={cfg['prune_iters']}"
                      f" | method={method}")
                print("=" * 120)

                try:
                    row = run_one(method=method, ds_name=ds, cfg=cfg, args=args, equations_map=equations_map)
                    append_results_row(args.output_csv, row, append=True)
                    print(f"[OK] appended -> {args.output_csv} | test_mse={row['test_mse']:.6g}")

                except KeyboardInterrupt:
                    print("\n[CTRL-C] Interrupted by user. Exiting cleanly.")
                    return

                except Exception as e:
                    fail_row = {
                        "dataset": ds,
                        "filename": _feynman_cli_to_filename(ds),
                        "method": method,
                        "config_idx": int(cfg["config_idx"]),
                        "ofat_factor": str(cfg["ofat_factor"]),
                        "seed": int(cfg["seed"]),
                        "width_mid": str(cfg["width_mid"]),
                        "lamb": float(cfg["lamb"]),
                        "prune_iters": int(cfg["prune_iters"]),
                        "device": args.device,
                        "split_strategy": args.split_strategy,
                        "grid": int(args.grid),
                        "lr": float(args.lr),
                        "steps": int(args.steps),
                        "reg_metric": args.reg_metric,
                        "gating_entropy": float(args.gating_entropy),
                        "gating_l1": float(args.gating_l1),
                        "top_k_gates": int(args.top_k_gates),
                        "regression_policy": str(args.regression_policy),
                        "train_mse": None,
                        "test_mse": None,
                        "predicted_formula": None,
                        "error": repr(e),
                    }
                    append_results_row(args.output_csv, fail_row, append=True)
                    print(f"[ERROR] {repr(e)}")
                    print(f"[INFO] failure row appended -> {args.output_csv}")


if __name__ == "__main__":
    main()
