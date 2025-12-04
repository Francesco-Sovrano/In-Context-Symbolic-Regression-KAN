from kan import *
import torch
import argparse

################################################################
def get_args():
	p = argparse.ArgumentParser()
	p.add_argument(
		"--simplify",
		dest="simplify",
		action="store_true",
		help="Simplify output (default: True). Use --simplify to enable.",
	)
	return p.parse_args()

args = get_args()
print(args)
SIMPLIFY = args.simplify

# if torch.backends.mps.is_available():
# 	device = torch.device("mps")        # Apple GPU via Metal
# elif torch.cuda.is_available():
# 	device = torch.device("cuda")       # For rare NVIDIA-eGPU setups
# else:
# 	device = torch.device("cpu")
# print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
f_range = [-1,1]
dataset = create_dataset(f, ranges=f_range, n_var=2, train_num=2000, test_num=1000)
print(dataset['train_input'].shape, dataset['train_label'].shape)

lib = ['0', 'x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']

# plot KAN at initialization
model = KAN(
	width=[2, [5,3], 1], 
	grid=50, 
	grid_range=f_range,
	atom_names=lib,
	seed=0,
)
# model(dataset['train_input']);
# model.plot(beta=100)

def check_gates(model):
	"""
	For every GatedSymbolicLayer:
	  - prints dims
	  - checks that sum_k #edges_where_argmax=k == O * I
	  - shows active edges (mask>0) vs total edges
	  - prints per-atom mean prob + argmax counts
	"""
	for layer_idx, layer in enumerate(model.act_fun):
		if not isinstance(layer, GatedSymbolicLayer):
			continue

		with torch.no_grad():
			logits = layer.gate_logits        # [O, I, K]
			O, I, K = logits.shape
			probs = torch.softmax(logits, dim=-1)  # [O, I, K]
			best  = logits.argmax(dim=-1)          # [O, I]

			# masks (KAN convention: mask [in_dim, out_dim])
			mask = layer.mask                      # [I, O]
			active = (mask > 0).T                  # [O, I] boolean: active edges only

			total_edges = O * I
			active_edges = int(active.sum().item())

			# counts per atom
			counts = []
			for k in range(K):
				c = int((best == k).sum().item())
				counts.append(c)

			sum_counts = sum(counts)

			print(f"\n=== GatedSymbolicLayer {layer_idx} ===")
			print(f"gate_logits shape: O={O}, I={I}, K={K}")
			print(f"total edges (O*I): {total_edges}")
			print(f"active edges (mask>0): {active_edges}")
			print(f"sum_k #argmax_edges_k: {sum_counts}")

			# This MUST always hold if argmax sees every edge
			if sum_counts != total_edges:
				print("!!! MISMATCH: some edges do not contribute to any argmax count")
			else:
				print("argmax coverage OK: each (j,i) has exactly one winning atom")

			# Per-atom stats
			mean_probs = probs.mean(dim=(0, 1))  # [K]

			print(f"{'atom':>8}  {'mean_p':>10}  {'#argmax_all':>13}  {'#argmax_active':>15}")
			for k_idx, name in enumerate(layer.atom_names):
				mp = float(mean_probs[k_idx].item())
				c_all = counts[k_idx]
				c_active = int(((best == k_idx) & active).sum().item())
				print(f"{name:>8}  {mp:10.6f}  {c_all:13d}  {c_active:15d}")


for _ in range(5):
	# train the model
	model.fit(dataset, opt="Adam", lr=1e-2, steps=1000, lamb=1e-2);
	model = model.prune(node_th=0.1, edge_th=0)
	# check_gates(model)
	print(model.get_symbolic_choice_per_edge())

# mode = "auto" # "manual"
# if mode == "manual":
# 	# manual mode
# 	model.fix_symbolic(0,0,0,'sin');
# 	model.fix_symbolic(0,1,0,'x^2');
# 	model.fix_symbolic(1,0,0,'exp');
# elif mode == "auto":
# 	# model.auto_symbolic(lib=lib, weight_simple=0)
# 	summary = model.auto_symbolic_robust_greedy(
# 		dataset,       # evaluation set
# 		lib=lib,
# 		min_edge_score=None,         # or e.g. 1e-3 to stop earlier
# 		mode="backward",             # or "ols"
# 		weight_simple=0,
# 		# verbose=1,
# 		lr=1e-2,
# 		steps=100,
# 		# lamb=1e-2,
# 		# node_th=0.5, edge_th=0.1,
# 	)
# 	print('auto_symbolic_robust_greedy:', summary)

# model.fit(dataset, opt="Adam", lr=1e-2, steps=2500, gating_entropy=1e-3, gating_l1=1e-1)
model.fit(dataset, opt="LBFGS", lr=1e-1, steps=500)
print(model.get_symbolic_choice_per_edge())
# check_gates(model)

symbolic_formula = model.symbolic_formula(simplify=SIMPLIFY)
if symbolic_formula:
	symbolic_formula = symbolic_formula[0][0]
	print('Symbolic formula:', re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', str(symbolic_formula)))
