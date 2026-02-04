# python3 example_feynman.py --symbolic_regression_method gated_greedy_matching_pursuit --feynman_name I.26.2

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
	p.add_argument('--top_k_gates', type=int, default=3)
	p.add_argument(
		"--width",
		nargs="+",
		type=int,
		default=[2, [5, 2], 1],
		help="Network width list, e.g. --width 2 5 1",
	)
	p.add_argument("--grid", type=int, default=20, help="Spline/basis grid size.")

	# Training
	p.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
	p.add_argument("--steps", type=int, default=500, help="Steps per fit() call.")
	p.add_argument("--lamb", type=float, default=1e-2, help="Spline L1 regularization.")
	p.add_argument(
		"--gating_entropy",
		type=float,
		default=1e-2,
		help="Entropy regularizer for gate distribution.",
	)
	p.add_argument(
		"--gating_l1",
		type=float,
		default=1e-2,
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
		default=5,
		help="How many prune+refit rounds to run (default: 1).",
	)
	#p.add_argument("--node_th", type=float, default=0.1, help="Node pruning threshold.")
	p.add_argument("--node_th", type=float, default=0.1, help="Node pruning threshold.")
	p.add_argument("--edge_th", type=float, default=0.0, help="Edge pruning threshold.")
	p.add_argument(
		"--gate_top_k_start",
		type=int,
		default=10,
		help="Initial top-k gates kept per edge during pruning.",
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
	p.add_argument("--feynman_root", type=str, default="data",
				help="Cartella che contiene le sottocartelle Feynman_* (default: data).")
	p.add_argument("--feynman_variant", type=str, default="Feynman_with_units",
				choices=["Feynman_without_units", "Feynman_with_units", "bonus_without_units", "bonus_with_units"],
				help="Quale collezione Feynman usare.")

	p.add_argument(
		"--max_datasets",
		type=int,
		default=0,
		help="Se >0, limita il numero di dataset Feynman processati (utile per test veloci)."
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


def _feynman_cli_to_filename(ds_name: str) -> str:
	"""
	Converte 'feynman_I_6_2a' -> 'I.6.2a'
	"""
	if ds_name.lower().startswith("feynman_"):
		ds_name = ds_name[len("feynman_"):]
	parts = ds_name.split("_")
	return ".".join(parts)

def list_local_feynman_dataset_names(feynman_root: str, variant: str = "Feynman_with_units"):
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

def load_local_feynman_dataset_as_kan(
	ds_name: str,
	feynman_root: str,
	variant: str,
	device: str = "cpu",
	dtype=torch.float32,
	train_cap: int = 4000,
	test_cap: int = 2000,
	**args
):
	filename = _feynman_cli_to_filename(ds_name)
	path = os.path.join(feynman_root, variant, filename)
	if not os.path.isfile(path):
			raise FileNotFoundError(f"File non trovato: {path}")

	data = np.loadtxt(path)
	if data.ndim == 1:
			data = data.reshape(-1, 1)
	if data.shape[1] < 2:
			raise ValueError(f"Dataset {ds_name} ha meno di 2 colonne (serve X + y). File: {path}")

	X = data[:, :-1].astype(np.float32)
	y = data[:, -1:].astype(np.float32)

	N = X.shape[0]
	n_tr = min(train_cap, N)
	n_te = min(test_cap, max(0, N - n_tr))

	# indici equispaziati (deterministici)
	# train: copre tutto [0..N-1]
	tr_idx = np.unique(np.round(np.linspace(0, N - 1, n_tr)).astype(int))

	# test: prende altri punti equispaziati e rimuove overlap
	# (se N è piccolo, potrebbe ridursi un po' il numero effettivo)
	te_all = np.unique(np.round(np.linspace(0, N - 1, n_tr + n_te)).astype(int))
	te_idx = te_all[~np.isin(te_all, tr_idx)][:n_te]

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
				device=args.device,
				seed=args.seed,
				train_cap = args.train_num, test_cap  = args.test_num
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
			# lamb=args.lamb,
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
		training_options['lamb'] = args.lamb

		# 6) Prune + refit rounds
		if args.prune_iters > 0:
			gate_top_k_pruning_delta = (args.gate_top_k_start - args.top_k_gates) // max(1, args.prune_iters)
			for i in range(args.prune_iters):
				top_k = max(args.top_k_gates, args.gate_top_k_start - (i + 1) * gate_top_k_pruning_delta)

				with timed_block(f"prune_round_{i}_prune", timings, enabled=args.timing):
					if use_old_kan_package:
						model = model.prune(node_th=args.node_th, edge_th=args.edge_th)
					else:
						model = model.prune(node_th=args.node_th, edge_th=args.edge_th, gate_top_k=top_k)

				with timed_block(f"prune_round_{i}_fit", timings, enabled=args.timing):
					model.fit(dataset, **training_options)

				if "atom_names" in kan_kwargs:
					check_gates(model)

		training_options['lamb'] = 0
		with timed_block("fit_final", timings, enabled=args.timing):
			model.fit(dataset, **training_options)

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
					top_k_gates=args.top_k_gates,
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