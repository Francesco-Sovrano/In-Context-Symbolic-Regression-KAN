#!/usr/bin/env python3
"""
Train a KAN on:
    f(x, y) = exp(sin(pi*x) + y^2)

Then run symbolic regression (required) and print a final symbolic formula.

Symbolic regression methods:
  - baseline:
      Post-hoc symbolic fitting; simplest and fastest.
  - greedy_matching_pursuit:
      Post-hoc iterative greedy symbolic selection using the dataset; usually stronger than baseline.
  - gated_greedy_matching_pursuit:
      Greedy selection plus gating built into the model (atom_names=lib), so the network is trained to choose atoms;
      most interpretable / most “symbolic-first”.

FastKAN modality (numeric atoms)
--------------------------------
This script also supports a "fastkan" modality via the *_fastkan methods:
  - baseline_fastkan
  - fastkan_greedy_matching_pursuit

In this script, "fastkan" means:
  - We swap the default spline numeric atom configuration (bspline) for a lighter numeric atom configuration
    (radial_bf), via `numeric_atom_configs`.
  - Everything else is unchanged: same dataset, same pruning loop, same symbolic regression routines.
  - The symbolic library `lib` is still used for the symbolic regression step; fastkan only changes the *numeric*
    basis used during training.

Notes:
  - Gated methods use `atom_names=lib` to instantiate GatedSymbolicLayer(s). The fastkan numeric-atom path is only
    used for non-gated methods in this script.
"""

import argparse
import time
from contextlib import contextmanager

import torch

from symbolic_kan.MultKAN import KAN, GatedSymbolicLayer
#from symbolic_kan import create_dataset
#from symbolic_kan.utils import list_feynman_dataset_names, load_pmlb_dataset_as_kan

from symbolic_kan.MultKAN import KAN as BaseKAN

import os
import glob
import numpy as np
def _feynman_cli_to_filename(ds_name: str) -> str:
	"""
	Converte 'feynman_I_6_2a' -> 'I.6.2a'
	"""
	if ds_name.lower().startswith("feynman_"):
		ds_name = ds_name[len("feynman_"):]
	parts = ds_name.split("_")
	return ".".join(parts)

def list_local_feynman_dataset_names(feynman_root: str, variant: str = "Feynman_without_units"):
	"""
	Ritorna lista in formato 'feynman_I_6_2a' a partire dai file dentro:
	  feynman_root/variant/*
	dove i file si chiamano tipo 'I.6.2a'
	"""
	base_dir = os.path.join(feynman_root, variant)
	if not os.path.isdir(base_dir):
		return []

	files = sorted(glob.glob(os.path.join(base_dir, "*")))
	names = []
	for fp in files:
		bn = os.path.basename(fp)
		# evita file nascosti o non-dataset
		if bn.startswith("."):
			continue
		# esempio bn: "I.6.2a" -> "feynman_I_6_2a"
		names.append("feynman_" + bn.replace(".", "_"))
	return names

def load_local_feynman_dataset_as_kan(ds_name: str, feynman_root: str, variant: str,
									  train_ratio: float = 0.8, device: str = "cpu",
									  seed: int = 0, dtype=torch.float32):
	"""
	Legge un file Feynman locale e ritorna dict compatibile con KAN.fit:
	  train_input, train_label, test_input, test_label

	Formato file: colonne numeriche whitespace-separated, ultima colonna = y, resto = X.
	"""
	filename = _feynman_cli_to_filename(ds_name)  # 'I.6.2a'
	path = os.path.join(feynman_root, variant, filename)
	if not os.path.isfile(path):
		raise FileNotFoundError(f"File non trovato: {path}")

	data = np.loadtxt(path)  # shape (N, D)
	if data.ndim == 1:
		data = data.reshape(-1, 1)
	if data.shape[1] < 2:
		raise ValueError(f"Dataset {ds_name} ha meno di 2 colonne (serve X + y). File: {path}")

	# X = data[:, :-1].astype(np.float32)
	# y = data[:, -1:].astype(np.float32)

	# # split riproducibile
	# # rng = np.random.default_rng(seed)
	# # N = X.shape[0]
	# # idx = rng.permutation(N)
	# # n_tr = int(train_ratio * N)
	# # tr_idx = idx[:n_tr]
	# # te_idx = idx[n_tr:]
	# # split riproducibile (sampling fisso)
	# rng = np.random.default_rng(seed)
	# N = X.shape[0]
	# idx = rng.permutation(N)

	# train_cap = 4000
	# test_cap  = 2000

	# tr_idx = idx[:min(train_cap, N)]
	# te_start = len(tr_idx)
	# te_idx = idx[te_start:te_start + min(test_cap, N - te_start)]


	# train_input = torch.from_numpy(X[tr_idx]).to(device=device, dtype=dtype)
	# train_label = torch.from_numpy(y[tr_idx]).to(device=device, dtype=dtype)
	# test_input  = torch.from_numpy(X[te_idx]).to(device=device, dtype=dtype)
	# test_label  = torch.from_numpy(y[te_idx]).to(device=device, dtype=dtype)

	X = data[:, :-1].astype(np.float32)
	y = data[:, -1:].astype(np.float32)

	rng = np.random.default_rng(seed)
	N = X.shape[0]

	train_cap = 4000
	test_cap  = 2000

	n_tr = min(train_cap, N)
	n_te = min(test_cap, max(0, N - n_tr))

	# campiona indici train random
	tr_idx = rng.choice(N, size=n_tr, replace=False)

	# campiona indici test random dai rimanenti (no overlap)
	remaining = np.setdiff1d(np.arange(N), tr_idx, assume_unique=False)
	te_idx = rng.choice(remaining, size=n_te, replace=False)

	train_input = torch.from_numpy(X[tr_idx]).to(device=device, dtype=dtype)
	train_label = torch.from_numpy(y[tr_idx]).to(device=device, dtype=dtype)
	test_input  = torch.from_numpy(X[te_idx]).to(device=device, dtype=dtype)
	test_label  = torch.from_numpy(y[te_idx]).to(device=device, dtype=dtype)


	return {
		"train_input": train_input,
		"train_label": train_label,
		"test_input": test_input,
		"test_label": test_label,
	}



def get_args():
	"""Parse CLI arguments."""
	p = argparse.ArgumentParser(
		description="KAN demo: train on exp(sin(pi*x)+y^2); optional FastKAN numeric atoms; optional gated symbolic layers depending on regression method."
	)

	# BooleanOptionalAction gives you both --simplify and --no-simplify.
	p.add_argument(
		"--simplify",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Simplify the final symbolic formula (default: False).",
	)

	# REQUIRED and constrained to the methods.
	p.add_argument(
		"--symbolic_regression_method",
		required=True,
		choices=[
			"baseline",
			"fastkan_baseline",
			"greedy_matching_pursuit",
			"fastkan_greedy_matching_pursuit",
			"gated_greedy_matching_pursuit",
		],
		help=(
			"Symbolic regression strategy:\n"
			"  baseline:\n"
			"    - Train/prune a standard KAN.\n"
			"    - Run baseline_symbolic_regression(lib=...).\n"
			"    - No gated symbolic layers; no per-edge gate/atom diagnostics.\n"
			"\n"
			"  baseline_fastkan:\n"
			"    - Same as baseline, but uses FastKAN-style numeric atoms (numeric_atom_configs=radial_bf) instead of bspline.\n"
			"    - Intended to reduce per-step compute while keeping the same post-hoc symbolic regression.\n"
			"\n"
			"  greedy_matching_pursuit:\n"
			"    - Train/prune a standard KAN.\n"
			"    - Run greedy_symbolic_regression(dataset, lib=...) (matching-pursuit style greedy selection).\n"
			"    - Still no gated symbolic layers in the model (atom_names is NOT set).\n"
			"\n"
			"  fastkan_greedy_matching_pursuit:\n"
			"    - Same as greedy_matching_pursuit, but uses FastKAN-style numeric atoms (numeric_atom_configs=radial_bf) instead of bspline.\n"
			"\n"
			"  gated_greedy_matching_pursuit:\n"
			"    - Construct KAN with gated symbolic layers by passing atom_names=lib.\n"
			"    - Train/prune with gates, then run greedy_symbolic_regression(dataset, lib=...).\n"
			"    - Enables gate-based inspectability (check_gates, get_symbolic_choice_per_edge) and gate_top_k pruning.\n"
		),
	)

	# Repro / data
	p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
	p.add_argument("--train_num", type=int, default=2000, help="Training samples.")
	p.add_argument("--test_num", type=int, default=1000, help="Test samples.")

	# Model
	#width 5 10, 20, 50, 100
	#width [5,2] [10,2], [20,2], [50,2], [100,2]
	#I.26.2,26,theta1,arcsin(n*sin(theta2)), [2,[5,2],1] una moltiplicazione e un solo innestamento
	p.add_argument(
		"--width",
		nargs="+",
		type=int,
		default=[2, 10, 1],
		help="Network width list, e.g. --width 2 5 1",
	)
	p.add_argument("--grid", type=int, default=20, help="Spline/basis grid size.")
	p.add_argument(
		"--grid_range",
		nargs=2,
		type=float,
		default=[-1.0, 1.0],
		metavar=("MIN", "MAX"),
		help="Input range for all variables (default: -1 1).",
	)

	# Training
	p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
	p.add_argument("--steps", type=int, default=500, help="Steps per fit() call.")
	p.add_argument("--lamb", type=float, default=1e-1, help="Spline L1 regularization.")
	p.add_argument(
		"--gating_entropy",
		type=float,
		default=1e-3,
		help="Entropy regularizer for gate distribution.",
	)
	p.add_argument(
		"--gating_l1",
		type=float,
		default=0,
		help="L1 regularizer for gating weights/masks.",
	)
	p.add_argument(
		"--reg_metric",
		choices=["node_backward", "edge_backward", "edge_forward_spline_u"],
		default="edge_backward",
		help="Regularization metric (avoid edge_forward_spline_n for gated symbolic layers).",
	)

	# Pruning
	p.add_argument(
		"--prune_iters",
		type=int,
		default=2,
		help="How many prune+refit rounds to run (default: 1).",
	)
	#p.add_argument("--node_th", type=float, default=0.1, help="Node pruning threshold.")
	p.add_argument("--node_th", type=float, default=0.1, help="Node pruning threshold.")
	p.add_argument("--edge_th", type=float, default=0.0, help="Edge pruning threshold.")
	p.add_argument(
		"--gate_top_k_start",
		type=int,
		default=6,
		help="Initial top-k gates kept per edge during pruning.",
	)
	p.add_argument(
		"--gate_top_k_min",
		type=int,
		default=3,
		help="Minimum top-k gates kept per edge during pruning.",
	)

	# Timing
	p.add_argument(
		"--timing",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Measure and print wall-clock time per phase (default: True). Use --no-timing to disable.",
	)
	# p.add_argument(
	# 	"--pmlb_name",
	# 	type=str,
	# 	default=None,
	# 	help="Se specificato, esegue SOLO questo dataset PMLB (es: feynman_I_6_2a). Se None, loop su tutti i Feynman."
	# )

	p.add_argument("--feynman_name", type=str, default=None,
			   help="Se specificato, esegue SOLO questo dataset (es: feynman_I_6_2a).")
	p.add_argument("--feynman_root", type=str, default="symbolic_kan/datasets",
				help="Cartella che contiene le sottocartelle Feynman_* (default: symbolic_kan/datasets).")
	p.add_argument("--feynman_variant", type=str, default="Feynman_without_units",
				choices=["Feynman_without_units", "Feynman_with_units", "bonus_without_units", "bonus_with_units"],
				help="Quale collezione Feynman usare.")

	p.add_argument(
		"--max_datasets",
		type=int,
		default=0,
		help="Se >0, limita il numero di dataset Feynman processati (utile per test veloci)."
	)
	p.add_argument(
		"--train_ratio",
		type=float,
		default=0.8,
		help="Frazione training nello split random (default: 0.8)."
	)
	#change to cpu
	p.add_argument(
		"--device",
		type=str,
		default="mps",
		choices=["cpu", "cuda", "mps"],
		help="Device per torch (default: cpu)."
	)


	return p.parse_args()


def _maybe_device_sync():
	"""
	Synchronize accelerator for accurate timing (CUDA and/or MPS), if available.

	Why both?
	- CUDA kernels are async -> need torch.cuda.synchronize()
	- MPS kernels are async -> need torch.mps.synchronize() (if present)
	"""
	# CUDA
	if torch.cuda.is_available():
		torch.cuda.synchronize()

	# MPS (Apple Silicon)
	mps_backend = getattr(torch.backends, "mps", None)
	if mps_backend is not None and mps_backend.is_available():
		if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
			torch.mps.synchronize()


@contextmanager
def timed_block(label, timings, enabled=True):
	"""
	Time a code block (wall-clock).
	Accumulates into `timings[label]` and prints per-block duration.
	"""
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
		print(f"[TIMER] {label}: {dt:.3f}s")


def print_timing_summary(timings, total_wall=None):
	if not timings:
		return
	print("\n=== Timing summary (seconds) ===")
	for k in sorted(timings.keys()):
		print(f"{k:35s} {timings[k]:10.3f}")
	if total_wall is not None:
		print(f"{'TOTAL_WALL':35s} {total_wall:10.3f}")
	else:
		print(f"{'TOTAL_TIMED':35s} {sum(timings.values()):10.3f}")


def check_gates(model):
	"""
	Inspect every GatedSymbolicLayer in model.act_fun and print:

	  - tensor dims (O, I, K)
	  - total edges (O*I)
	  - active edges (mask>0)
	  - per-atom mean probability, argmax counts (all edges and active-only)

	This is a sanity check: sum_k argmax-counts must equal O*I.
	"""
	for layer_idx, layer in enumerate(getattr(model, "act_fun", [])):
		if not isinstance(layer, GatedSymbolicLayer):
			continue

		with torch.no_grad():
			logits = layer.gate_logits  # [O, I, K]
			O, I, K = logits.shape

			probs = torch.softmax(logits, dim=-1)  # [O, I, K]
			best = logits.argmax(dim=-1)  # [O, I]

			# KAN convention: mask has shape [in_dim, out_dim] i.e. [I, O]
			mask = layer.mask  # [I, O]
			active = (mask > 0).T  # [O, I] boolean

			total_edges = O * I
			active_edges = int(active.sum().item())

			counts_all = [int((best == k).sum().item()) for k in range(K)]
			sum_counts = sum(counts_all)

			print(f"\n=== GatedSymbolicLayer {layer_idx} ===")
			print(f"gate_logits shape: O={O}, I={I}, K={K}")
			print(f"total edges (O*I): {total_edges}")
			print(f"active edges (mask>0): {active_edges}")
			print(f"sum_k #argmax_edges_k: {sum_counts}")

			if sum_counts != total_edges:
				print("!!! MISMATCH: argmax counts do not cover all edges")
			else:
				print("argmax coverage OK")

			mean_probs = probs.mean(dim=(0, 1))  # [K]

			atom_names = getattr(layer, "atom_names", [f"atom_{k}" for k in range(K)])
			print(f"{'atom':>10}  {'mean_p':>10}  {'#argmax_all':>13}  {'#argmax_active':>15}")
			for k_idx, name in enumerate(atom_names):
				mp = float(mean_probs[k_idx].item())
				c_all = counts_all[k_idx]
				c_active = int(((best == k_idx) & active).sum().item())
				print(f"{str(name):>10}  {mp:10.6f}  {c_all:13d}  {c_active:15d}")


# def main():
# 	args = get_args()
# 	print(args)

# 	use_old_kan_package = False  # set True only if you want to force using `kan.MultKAN.KAN` instead of `symbolic_kan.MultKAN.KAN`

# 	timings = {}
# 	wall_t0 = time.perf_counter()

# 	# Reproducibility (best-effort; KAN may also set internal seeds)
# 	torch.manual_seed(args.seed)

# 	# Target function and dataset
# 	# def f(x):
# 	# 	# x shape: [N, 2], returns [N, 1]
# 	# 	return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

# 	# f_range = [float(args.grid_range[0]), float(args.grid_range[1])]
# 	# with timed_block("dataset_create", timings, enabled=args.timing):
# 	# 	dataset = create_dataset(
# 	# 		f,
# 	# 		ranges=f_range,
# 	# 		n_var=2,
# 	# 		train_num=args.train_num,
# 	# 		test_num=args.test_num,
# 	# 	)
# 	#print(dataset["train_input"].shape, dataset["train_label"].shape)
# 		# --- PMLB / Feynman dataset list ---
# 	if args.pmlb_name is not None:
# 		dataset_names = [args.pmlb_name]
# 	else:
# 		dataset_names = list_feynman_dataset_names()
# 		if args.max_datasets and args.max_datasets > 0:
# 			dataset_names = dataset_names[:args.max_datasets]

# 	print(f"Found {len(dataset_names)} Feynman datasets to run.")

# 	# Symbol library (always used for symbolic regression; also used as atom_names only for gated_greedy_matching_pursuit)
# 	lib = ["0", "x", "x^2", "exp", "log", "sqrt", "tanh", "sin", "abs"]

# 	# Build model
# 	kan_kwargs = dict(
# 		width=args.width,
# 		grid=args.grid,
# 		grid_range=f_range,
# 		seed=args.seed,
# 	)

# 	if "gated" in args.symbolic_regression_method:
# 		# Gated symbolic layers live inside the model, and will select among `lib` atoms.
# 		kan_kwargs["atom_names"] = lib
# 	else:
# 		# Numeric atom configs:
# 		# - default: bspline
# 		# - fastkan: radial_bf (lighter numeric basis in this script)
# 		if "fastkan" in args.symbolic_regression_method:
# 			kan_kwargs["numeric_atom_configs"] = {
# 				"radial_bf": {"num_grids": args.grid},
# 			}
# 		else:
# 			kan_kwargs["numeric_atom_configs"] = {
# 				"bspline": {"num_grids": args.grid, "degree": 3},
# 			}

# 	with timed_block("model_init", timings, enabled=args.timing):
# 		if use_old_kan_package:
# 			model = BaseKAN(**kan_kwargs)
# 		else:
# 			model = KAN(**kan_kwargs)

# 	# Shared training options (KAN.fit expects these keys)
# 	training_options = dict(
# 		lr=args.lr,
# 		steps=args.steps,
# 		lamb=args.lamb,
# 		reg_metric=args.reg_metric,
# 	)

# 	if use_old_kan_package:
# 		training_options.update(dict(
# 			opt="Adam",
# 		))
# 	else:
# 		training_options.update(dict(
# 			optimizer="Adam",
# 			gating_entropy=args.gating_entropy,
# 			gating_l1=args.gating_l1,
# 		))

# 	# 1) Initial training
# 	with timed_block("fit_initial", timings, enabled=args.timing):
# 		model.fit(dataset, **training_options)

# 	# 2) Prune + refit rounds
# 	gate_top_k_pruning_delta = (args.gate_top_k_start-args.gate_top_k_min)//args.prune_iters
# 	for i in range(args.prune_iters):
# 		top_k = max(args.gate_top_k_min, args.gate_top_k_start - (i+1)*gate_top_k_pruning_delta)

# 		with timed_block(f"prune_round_{i}_prune", timings, enabled=args.timing):
# 			if use_old_kan_package:
# 				model = model.prune(
# 					node_th=args.node_th,
# 					edge_th=args.edge_th,
# 				)
# 			else:
# 				model = model.prune(
# 					node_th=args.node_th,
# 					edge_th=args.edge_th,
# 					gate_top_k=top_k,
# 				)

# 		with timed_block(f"prune_round_{i}_fit", timings, enabled=args.timing):
# 			model.fit(dataset, **training_options)

# 		if "atom_names" in kan_kwargs:
# 			check_gates(model)
# 			# print(model.get_symbolic_choice_per_edge())

# 	# 3) Symbolic regression post-pass (REQUIRED by argparse)
# 	if "baseline" in args.symbolic_regression_method:
# 		with timed_block("symbolic_regression", timings, enabled=args.timing):
# 			if use_old_kan_package:
# 				summary = model.auto_symbolic(lib=lib, weight_simple=0)
# 			else:
# 				summary = model.baseline_symbolic_regression(lib=lib, weight_simple=0)
# 		print("baseline_symbolic_regression:", summary)

# 	elif "greedy_matching_pursuit" in args.symbolic_regression_method:
# 		# Commonly run with lamb=0 for cleaner symbolic selection.
# 		symbolic_training_options = dict(training_options)
# 		symbolic_training_options["steps"] = 100
# 		symbolic_training_options["lamb"] = 0

# 		with timed_block("symbolic_regression", timings, enabled=args.timing):
# 			summary = model.greedy_symbolic_regression(
# 				dataset,
# 				lib=lib,
# 				top_k_gates=3,
# 				**symbolic_training_options,
# 			)
# 		print("greedy_symbolic_regression:", summary)

# 	# 4) Final polish: train with lamb=0 before exporting formula
# 	training_options["lamb"] = 0
# 	with timed_block("fit_final_polish", timings, enabled=args.timing):
# 		model.fit(dataset, **training_options)

# 	if "atom_names" in kan_kwargs:
# 		check_gates(model)
# 		# print(model.get_symbolic_choice_per_edge())

# 	# 5) Export symbolic formula
# 	with timed_block("export_symbolic_formula", timings, enabled=args.timing):
# 		if use_old_kan_package:
# 			symbolic_formula = model.symbolic_formula()
# 		else:
# 			symbolic_formula = model.symbolic_formula(simplify=args.simplify)
# 	if symbolic_formula:
# 		print("Symbolic formula:", symbolic_formula[0][0])

# 	if args.timing:
# 		total_wall = time.perf_counter() - wall_t0
# 		print_timing_summary(timings, total_wall=total_wall)


# if __name__ == "__main__":
# 	main()


# def main():
# 	args = get_args()
# 	print(args)

# 	use_old_kan_package = False  # True solo se vuoi forzare kan.MultKAN.KAN

# 	# --- PMLB / Feynman dataset list ---
# 	if args.pmlb_name is not None:
# 		dataset_names = [args.pmlb_name]
# 	else:
# 		dataset_names = list_feynman_dataset_names()
# 		if args.max_datasets and args.max_datasets > 0:
# 			dataset_names = dataset_names[:args.max_datasets]

# 	print(f"Found {len(dataset_names)} Feynman datasets to run.")

# 	# --- Symbol library: Feynman-only (superset ragionevole) ---
# 	# Deve matchare i nomi in SYMBOLIC_LIB (utils.py).
# 	lib = [
# 		"0", "1", "x",
# 		"x^2", "x^3", "x^4", "x^5",
# 		"1/x", "1/x^2", "1/x^3",
# 		"sqrt", "1/sqrt(x)",
# 		"log", "exp",
# 		"sin", "cos", "tan", "tanh",
# 		"abs", "sgn",
# 		"arctan", "arcsin", "arccos", "arctanh",
# 		"gaussian",
# 	]

# 	# Loop su dataset
# 	for ds_name in dataset_names:
# 		print("\n" + "=" * 80)
# 		print(f"DATASET: {ds_name}")
# 		print("=" * 80)

# 		# per-dataset timings + wall clock
# 		timings = {}
# 		wall_t0 = time.perf_counter()

# 		# Reproducibility (per dataset)
# 		torch.manual_seed(args.seed)

# 		# 1) Load dataset from PMLB and convert to KAN dict
# 		with timed_block("dataset_load_pmlb", timings, enabled=args.timing):
# 			dataset = load_pmlb_dataset_as_kan(
# 				ds_name,
# 				train_ratio=args.train_ratio,
# 				device=args.device,
# 				seed=args.seed,
# 			)

# 		print(dataset["train_input"].shape, dataset["train_label"].shape)

# 		# 2) Auto width and grid_range from dataset
# 		Xtr = dataset["train_input"]
# 		n_in = Xtr.shape[1]

# 		width = list(args.width)
# 		if width[0] != n_in:
# 			width[0] = n_in
# 		if width[-1] != 1:
# 			width[-1] = 1

# 		x_min = float(torch.min(Xtr).item())
# 		x_max = float(torch.max(Xtr).item())
# 		if x_min == x_max:
# 			x_min, x_max = x_min - 1.0, x_max + 1.0
# 		f_range = [x_min, x_max]

# 		# 3) Build model kwargs
# 		kan_kwargs = dict(
# 			width=width,
# 			grid=args.grid,
# 			grid_range=f_range,
# 			seed=args.seed,
# 		)

# 		if "gated" in args.symbolic_regression_method:
# 			# Gated symbolic layers inside model
# 			kan_kwargs["atom_names"] = lib
# 		else:
# 			# Numeric atoms configs (fastkan vs default)
# 			if "fastkan" in args.symbolic_regression_method:
# 				kan_kwargs["numeric_atom_configs"] = {
# 					"radial_bf": {"num_grids": args.grid},
# 				}
# 			else:
# 				kan_kwargs["numeric_atom_configs"] = {
# 					"bspline": {"num_grids": args.grid, "degree": 3},
# 				}

# 		with timed_block("model_init", timings, enabled=args.timing):
# 			if use_old_kan_package:
# 				model = BaseKAN(**kan_kwargs)
# 			else:
# 				model = KAN(**kan_kwargs)

# 		# 4) Training options
# 		training_options = dict(
# 			lr=args.lr,
# 			steps=args.steps,
# 			lamb=args.lamb,
# 			reg_metric=args.reg_metric,
# 		)

# 		if use_old_kan_package:
# 			training_options.update(dict(opt="Adam"))
# 		else:
# 			training_options.update(dict(
# 				optimizer="Adam",
# 				gating_entropy=args.gating_entropy,
# 				gating_l1=args.gating_l1,
# 			))

# 		# 5) Initial training
# 		with timed_block("fit_initial", timings, enabled=args.timing):
# 			model.fit(dataset, **training_options)

# 		# 6) Prune + refit rounds
# 		if args.prune_iters > 0:
# 			gate_top_k_pruning_delta = (args.gate_top_k_start - args.gate_top_k_min) // max(1, args.prune_iters)
# 			for i in range(args.prune_iters):
# 				top_k = max(args.gate_top_k_min, args.gate_top_k_start - (i + 1) * gate_top_k_pruning_delta)

# 				with timed_block(f"prune_round_{i}_prune", timings, enabled=args.timing):
# 					if use_old_kan_package:
# 						model = model.prune(node_th=args.node_th, edge_th=args.edge_th)
# 					else:
# 						model = model.prune(node_th=args.node_th, edge_th=args.edge_th, gate_top_k=top_k)

# 				with timed_block(f"prune_round_{i}_fit", timings, enabled=args.timing):
# 					model.fit(dataset, **training_options)

# 				if "atom_names" in kan_kwargs:
# 					check_gates(model)

# 		# 7) Symbolic regression post-pass
# 		if "baseline" in args.symbolic_regression_method:
# 			with timed_block("symbolic_regression", timings, enabled=args.timing):
# 				if use_old_kan_package:
# 					summary = model.auto_symbolic(lib=lib, weight_simple=0)
# 				else:
# 					summary = model.baseline_symbolic_regression(lib=lib, weight_simple=0)
# 			print("baseline_symbolic_regression:", summary)

# 		elif "greedy_matching_pursuit" in args.symbolic_regression_method:
# 			symbolic_training_options = dict(training_options)
# 			symbolic_training_options["steps"] = 100
# 			symbolic_training_options["lamb"] = 0

# 			with timed_block("symbolic_regression", timings, enabled=args.timing):
# 				summary = model.greedy_symbolic_regression(
# 					dataset,
# 					lib=lib,
# 					top_k_gates=3,
# 					**symbolic_training_options,
# 				)
# 			print("greedy_symbolic_regression:", summary)

# 		# 8) Final polish: lamb=0 then export formula
# 		training_options["lamb"] = 0
# 		with timed_block("fit_final_polish", timings, enabled=args.timing):
# 			model.fit(dataset, **training_options)

# 		if "atom_names" in kan_kwargs:
# 			check_gates(model)

# 		with timed_block("export_symbolic_formula", timings, enabled=args.timing):
# 			if use_old_kan_package:
# 				symbolic_formula = model.symbolic_formula()
# 			else:
# 				symbolic_formula = model.symbolic_formula(simplify=args.simplify)

# 		if symbolic_formula:
# 			print("Symbolic formula:", symbolic_formula[0][0])
# 		else:
# 			print("Symbolic formula: <None>")

# 		# 9) timing summary per dataset
# 		if args.timing:
# 			total_wall = time.perf_counter() - wall_t0
# 			print_timing_summary(timings, total_wall=total_wall)


# if __name__ == "__main__":
# 	main()

#NUOVO FYNMAN
def main():
	args = get_args()
	print(args)

	use_old_kan_package = False  # True solo se vuoi forzare kan.MultKAN.KAN

	# --- Local Feynman dataset list ---
	if args.feynman_name is not None:
		dataset_names = [args.feynman_name]
	else:
		base_dir = os.path.join(args.feynman_root, args.feynman_variant)
		if not os.path.isdir(base_dir):
			raise RuntimeError(f"Cartella non trovata: {base_dir}")

		# file tipo 'I.6.2a' -> name 'feynman_I_6_2a'
		files = sorted(glob.glob(os.path.join(base_dir, "*")))
		dataset_names = []
		for fp in files:
			bn = os.path.basename(fp)
			if bn.startswith("."):
				continue
			dataset_names.append("feynman_" + bn.replace(".", "_"))

		if args.max_datasets and args.max_datasets > 0:
			dataset_names = dataset_names[:args.max_datasets]

	print(f"Found {len(dataset_names)} Feynman datasets to run.")
	if len(dataset_names) == 0:
		raise RuntimeError(f"Nessun dataset trovato in {os.path.join(args.feynman_root, args.feynman_variant)}")

	# --- Symbol library: Feynman-only (superset ragionevole) ---
	# Deve matchare i nomi in SYMBOLIC_LIB (utils.py).
	lib = [
		#"0", "1", 
		"x",
		"x^2", "x^3", "x^4", "x^5",
		"1/x", "1/x^2", "1/x^3",
		"sqrt", "1/sqrt(x)",
		"log", "exp",
		"sin", "cos", "tan", "tanh",
		#"abs", "sgn",
		#"arctan", "arcsin", "arccos", 
		# "arctanh",
		#"gaussian",
	]

	#lib = ["0", "x", "x^2", "exp", "log", "sqrt", "tanh", "sin", "abs"]

	# Loop su dataset
	for ds_name in dataset_names:
		print("\n" + "=" * 80)
		print(f"DATASET: {ds_name}")
		print("=" * 80)

		# per-dataset timings + wall clock
		timings = {}
		wall_t0 = time.perf_counter()

		# Reproducibility (per dataset)
		torch.manual_seed(args.seed)

		# 1) Load dataset from LOCAL FEYNMAN and convert to KAN dict
		with timed_block("dataset_load_feynman_local", timings, enabled=args.timing):
			dataset = load_local_feynman_dataset_as_kan(
				ds_name,
				feynman_root=args.feynman_root,
				variant=args.feynman_variant,
				train_ratio=args.train_ratio,
				device=args.device,
				seed=args.seed,
			)

		print(dataset["train_input"].shape, dataset["train_label"].shape)

		# 2) Auto width and grid_range from dataset
		Xtr = dataset["train_input"]
		n_in = Xtr.shape[1]

		width = list(args.width)
		if width[0] != n_in:
			width[0] = n_in
		if width[-1] != 1:
			width[-1] = 1

		x_min = float(torch.min(Xtr).item())
		x_max = float(torch.max(Xtr).item())
		if x_min == x_max:
			x_min, x_max = x_min - 1.0, x_max + 1.0
		f_range = [x_min, x_max]

		# 3) Build model kwargs
		kan_kwargs = dict(
			width=width,
			grid=args.grid,
			grid_range=f_range,
			seed=args.seed,
		)

		if "gated" in args.symbolic_regression_method:
			# Gated symbolic layers inside model
			kan_kwargs["atom_names"] = lib
		else:
			# Numeric atoms configs (fastkan vs default)
			if "fastkan" in args.symbolic_regression_method:
				kan_kwargs["numeric_atom_configs"] = {
					"radial_bf": {"num_grids": args.grid},
				}
			else:
				kan_kwargs["numeric_atom_configs"] = {
					"bspline": {"num_grids": args.grid, "degree": 3},
				}

		with timed_block("model_init", timings, enabled=args.timing):
			if use_old_kan_package:
				model = BaseKAN(**kan_kwargs)
			else:
				model = KAN(**kan_kwargs)

		# 4) Training options
		training_options = dict(
			lr=args.lr,
			steps=args.steps,
			lamb=args.lamb,
			reg_metric=args.reg_metric,
		)

		if use_old_kan_package:
			training_options.update(dict(opt="Adam"))
		else:
			training_options.update(dict(
				optimizer="Adam",
				gating_entropy=args.gating_entropy,
				gating_l1=args.gating_l1,
			))

		# 5) Initial training
		with timed_block("fit_initial", timings, enabled=args.timing):
			model.fit(dataset, **training_options)

		# 6) Prune + refit rounds
		if args.prune_iters > 0:
			gate_top_k_pruning_delta = (args.gate_top_k_start - args.gate_top_k_min) // max(1, args.prune_iters)
			for i in range(args.prune_iters):
				top_k = max(args.gate_top_k_min, args.gate_top_k_start - (i + 1) * gate_top_k_pruning_delta)

				with timed_block(f"prune_round_{i}_prune", timings, enabled=args.timing):
					if use_old_kan_package:
						model = model.prune(node_th=args.node_th, edge_th=args.edge_th)
					else:
						model = model.prune(node_th=args.node_th, edge_th=args.edge_th, gate_top_k=top_k)

				with timed_block(f"prune_round_{i}_fit", timings, enabled=args.timing):
					model.fit(dataset, **training_options)

				if "atom_names" in kan_kwargs:
					check_gates(model)

		# 7) Symbolic regression post-pass
		if "baseline" in args.symbolic_regression_method:
			with timed_block("symbolic_regression", timings, enabled=args.timing):
				if use_old_kan_package:
					summary = model.auto_symbolic(lib=lib, weight_simple=0)
				else:
					summary = model.baseline_symbolic_regression(lib=lib, weight_simple=0)
			print("baseline_symbolic_regression:", summary)

		elif "greedy_matching_pursuit" in args.symbolic_regression_method:
			symbolic_training_options = dict(training_options)
			symbolic_training_options["steps"] = 100
			symbolic_training_options["lamb"] = 0

			with timed_block("symbolic_regression", timings, enabled=args.timing):
				summary = model.greedy_symbolic_regression(
					dataset,
					lib=lib,
					top_k_gates=3,
					**symbolic_training_options,
				)
			print("greedy_symbolic_regression:", summary)

		# 8) Final polish: lamb=0 then export formula
		training_options["lamb"] = 0
		with timed_block("fit_final_polish", timings, enabled=args.timing):
			model.fit(dataset, **training_options)

		if "atom_names" in kan_kwargs:
			check_gates(model)

		with timed_block("export_symbolic_formula", timings, enabled=args.timing):
			if use_old_kan_package:
				symbolic_formula = model.symbolic_formula()
			else:
				symbolic_formula = model.symbolic_formula(simplify=args.simplify)

		if symbolic_formula:
			print("Symbolic formula:", symbolic_formula[0][0])
		else:
			print("Symbolic formula: <None>")

		# 9) timing summary per dataset
		if args.timing:
			total_wall = time.perf_counter() - wall_t0
			print_timing_summary(timings, total_wall=total_wall)


if __name__ == "__main__":
	main()