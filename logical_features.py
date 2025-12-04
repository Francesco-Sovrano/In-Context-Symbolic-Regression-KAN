from kan import *
from kan.utils import add_symbolic, _safe_log, _safe_recip
import torch
import sympy as sp
import re
import json
from numbers import Real
import math
import argparse

################################################################
def get_args():
	p = argparse.ArgumentParser()
	p.add_argument(
		"--max_n",
		type=int,
		default=25000,
		help="Maximum N (default: 25000)",
	)
	p.add_argument(
		"--simplify",
		dest="simplify",
		action="store_true",
		help="Simplify output (default: True). Use --simplify to enable.",
	)
	p.add_argument(
		"--mape_threshold",
		type=float,
		default=0.05,
		help="Stop searching for a solution only if MAPE < mape_threshold (default: 0.05%)",
	)
	return p.parse_args()

args = get_args()
print(args)
MAX_N = args.max_n
SIMPLIFY = args.simplify
MAPE_THRESHOLD = args.mape_threshold

EPS = 1e-8

# ---------------------------------------------------------------------
# Synthetic logical-style toy dataset (no primes, no number theory)
# ---------------------------------------------------------------------
# 2D inputs in (0.1, 1.0); output uses nested if–then–else and a modulus.
torch.manual_seed(0)

#################################################################
#################################################################
# # Modulus term (true modulo, not the smooth symbolic one)
# mod_term = torch.remainder(x2, 0.25)  # in [0, 0.25)
# # Ground-truth rule:
# # if x1 > 0.7:
# #     if mod_term > 0.125:
# #         y = 2.0 + 1.5*x1 + 0.5*mod_term + 0.1*x2
# #     else:
# #         y = 1.5 + 0.8*x1 + mod_term
# # else:
# #     if x2 > 0.4:
# #         y = 1.0 + 0.3*x1*x2 + mod_term
# #     else:
# #         y = 0.5 + 0.2*x1 + 0.1*x2 + 2.0*mod_term
# #
# # All branches are strictly positive, so optional log-scaling is still valid.
# cond_outer = x1 > 0.7
# cond_inner_high = mod_term > 0.125
# cond_inner_lowx2 = x2 > 0.4
# y = torch.empty_like(x1)
# # Branch 1: x1 > 0.7 and mod_term > 0.125
# mask1 = cond_outer & cond_inner_high
# y[mask1] = 2.0 + 1.5 * x1[mask1] + 0.5 * mod_term[mask1] + 0.1 * x2[mask1]
# # Branch 2: x1 > 0.7 and mod_term <= 0.125
# mask2 = cond_outer & (~cond_inner_high)
# y[mask2] = 1.5 + 0.8 * x1[mask2] + mod_term[mask2]
# # Branch 3: x1 <= 0.7 and x2 > 0.4
# mask3 = (~cond_outer) & cond_inner_lowx2
# y[mask3] = 1.0 + 0.3 * x1[mask3] * x2[mask3] + mod_term[mask3]
# # Branch 4: x1 <= 0.7 and x2 <= 0.4
# mask4 = (~cond_outer) & (~cond_inner_lowx2)
# y[mask4] = 0.5 + 0.2 * x1[mask4] + 0.1 * x2[mask4] + 2.0 * mod_term[mask4]
#################################################################
#################################################################

# f = (0.5*x0 + 0.1*x1) + step(x0 - c) * (1.5*x0 + 0.9*x1)
# f = lambda x: torch.where(x[:,[0]] > 0.6, 2.0 * x[:,[0]] + x[:,[1]], 0.5 * x[:,[0]] + 0.1 * x[:,[1]])
f = lambda x: 2.0 * x[:,[0]] + x[:,[1]]*x[:,[0]]
f_range=[-1,1]
dataset = create_dataset(f, n_var=2, ranges=f_range, train_num=2000, test_num=1000)

#############################################################

################################################################
# Symbolic primitives well-suited for piecewise-linear rules
################################################################

LOGICAL_STEEPNESS = 20.0   # sharper than 10, but still differentiable


def smooth_step(z, k: float = LOGICAL_STEEPNESS):
	"""
	Smooth step in [0,1] approximating 1_{z>0}.
	z is generic: z = x - c, z = a*x1 + b*x2 + c, etc.
	"""
	return torch.sigmoid(k * z)


def smooth_sign(z, k: float = LOGICAL_STEEPNESS):
	"""
	Smooth sign-like function in [-1,1]: 2*step(z)-1.
	"""
	s = smooth_step(z, k=k)
	return 2.0 * s - 1.0


def smooth_relu(z, k: float = LOGICAL_STEEPNESS):
	"""
	Smooth ReLU: z * smooth_step(z). Behaves like max(0,z) but smooth.
	"""
	return z * smooth_step(z, k=k)


# --- Register primitives -----------------------------------------------------

# Step gate ~ 1_{z>0}
add_symbolic(
	'step',
	lambda z: smooth_step(z),
	c=1,
	sympy_fun=lambda z: sp.Function('step')(z)  # no Piecewise here
)

# Smooth ReLU hinge
add_symbolic(
	'relu',
	lambda z: smooth_relu(z),
	c=1,
	sympy_fun=lambda z: sp.Function('relu')(z)
)

# Smooth sign
add_symbolic(
	'sgn_smooth',
	lambda z: smooth_sign(z),
	c=2,
	sympy_fun=lambda z: sp.Function('sgn_smooth')(z)  # any smooth sign-like
)

safe_lib = [
	# constants and simple polynomials
	'0',
	'x',
	'x^2',
	'abs',

	# our logical / piecewise primitives
	'step',
	'relu',
	'sgn_smooth',
]

print('Available symbols:', safe_lib)

################################################################

def print_results(model, dataset, transform_fn=None, k=50):
	with torch.no_grad():
		y_hat = model(dataset['test_input'])
		if transform_fn is not None:
			y_hat = transform_fn(y_hat)
			y_true = transform_fn(dataset['test_label'])
		else:
			y_true = dataset['test_label']

		mape = torch.mean(torch.abs((y_true - y_hat) / y_true)) * 100.0
		print(f"MAPE = {mape:.6f}%")

		y = y_true
		pairs_preview = []
		for y_true_i, y_hat_i in list(zip(y_true[:k].tolist(), y_hat[:k].tolist())):
			pairs_preview.append({
				'label': float(y_true_i[0]),
				'pred': float(y_hat_i[0]),
			})

		print(f"test label + prediction (first {k}):", json.dumps(pairs_preview, indent=4))
		return mape

def bad_mape(x):
	# invalid = None, non-numeric, NaN, or ±inf
	return (
		x is None
		or not isinstance(x, Real)
		or not math.isfinite(float(x))
	)

i = 0
mape = float('inf')
symbolic_formula = None
while not symbolic_formula or bad_mape(mape) or mape >= MAPE_THRESHOLD:
	# model
	model = KAN(
		width=[2, [5,5], 1], 
		grid=10, 
		seed=i,
		grid_range=f_range,
	)

	for _ in range(2):
		# train the model
		model.fit(dataset, opt="Adam", lr=1e-2, steps=1000, lamb=1e-2);
		model = model.prune(node_th=0.1, edge_th=0)

	# model = model.refine(30)
	model.fit(dataset, opt="Adam", lr=1e-2, steps=5000)

	summary = model.auto_symbolic_robust_greedy(
		dataset,       # evaluation set
		lib=safe_lib,
		min_edge_score=None,
		mode="backward",
		weight_simple=0,
		lr=1e-2,
		steps=100,
		lamb=1e-2,
		node_th=0.5, edge_th=0.1,
		min_r2=0.9
	)
	print('auto_symbolic_robust_greedy:', summary)

	symbolic_formula = model.symbolic_formula(simplify=SIMPLIFY)
	if symbolic_formula:
		symbolic_formula = symbolic_formula[0][0]
		print('Symbolic formula:', re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', str(symbolic_formula)))

	model.fit(dataset, opt="Adam", lr=1e-2, steps=500, lamb=1e-1)
	model = model.prune(node_th=0.1, edge_th=0)
	model.fit(dataset, opt="Adam", lr=1e-2, steps=500)

	symbolic_formula = model.symbolic_formula(simplify=SIMPLIFY)
	if symbolic_formula:
		symbolic_formula = symbolic_formula[0][0]
		print('Symbolic formula:', re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', str(symbolic_formula)))
		if str(symbolic_formula) == 'nan':
			symbolic_formula = None

	# mape = print_results(model, dataset, k=50)
	i += 1
	break
