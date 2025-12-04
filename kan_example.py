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

# plot KAN at initialization
model = KAN(
	width=[2, [5,3], 1], 
	grid=50, 
	grid_range=f_range,
	seed=0,
)
# model(dataset['train_input']);
# model.plot(beta=100)

for _ in range(2):
	# train the model
	model.fit(dataset, opt="Adam", lr=1e-2, steps=1000, lamb=1e-2);
	model = model.prune(node_th=0.1, edge_th=0)

model.fit(dataset, opt="Adam", lr=1e-2, steps=2500);

mode = "auto" # "manual"

if mode == "manual":
	# manual mode
	model.fix_symbolic(0,0,0,'sin');
	model.fix_symbolic(0,1,0,'x^2');
	model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
	# automatic mode
	lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
	# model.auto_symbolic(lib=lib, weight_simple=0)
	summary = model.auto_symbolic_robust_greedy(
		dataset,       # evaluation set
		lib=lib,
		min_edge_score=None,         # or e.g. 1e-3 to stop earlier
		mode="backward",             # or "ols"
		weight_simple=0,
		# verbose=1,
		lr=1e-2,
        steps=100,
        lamb=1e-2,
        node_th=0.5, edge_th=0.1,
	)
	print('auto_symbolic_robust_greedy:', summary)

model.fit(dataset, opt="Adam", lr=1e-2, steps=500, lamb=1e-1)
model = model.prune(node_th=0.1, edge_th=0)
model.fit(dataset, opt="Adam", lr=1e-2, steps=500)

symbolic_formula = model.symbolic_formula(simplify=SIMPLIFY)
if symbolic_formula:
	symbolic_formula = symbolic_formula[0][0]
	print('Symbolic formula:', re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', str(symbolic_formula)))
