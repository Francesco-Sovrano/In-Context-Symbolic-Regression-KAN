import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy
import yaml
from sympy.utilities.lambdify import lambdify
import torch.nn.functional as F
import re
import math
from math import isfinite
from math import log

# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)

# singularity protection functions
f_inv = lambda x, y_th: ((x_th := 1/y_th), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x) * (torch.abs(x) >= x_th))
f_inv2 = lambda x, y_th: ((x_th := 1/y_th**(1/2)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**2) * (torch.abs(x) >= x_th))
f_inv3 = lambda x, y_th: ((x_th := 1/y_th**(1/3)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**3) * (torch.abs(x) >= x_th))
f_inv4 = lambda x, y_th: ((x_th := 1/y_th**(1/4)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**4) * (torch.abs(x) >= x_th))
f_inv5 = lambda x, y_th: ((x_th := 1/y_th**(1/5)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**5) * (torch.abs(x) >= x_th))
# sqrt with odd extension: linear inside, sqrt outside; matches at |x| = x_th
f_sqrt = lambda x, y_th: (
	(x_th := y_th**2),
	(y_th / x_th) * x * (torch.abs(x) < x_th)
	+ torch.sqrt(torch.abs(x)) * torch.sign(x) * (torch.abs(x) >= x_th)
)
f_power1d5 = lambda x, y_th: (None, torch.abs(x)**1.5)
f_invsqrt = lambda x, y_th: ((x_th := 1/y_th**2), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th))
f_log = lambda x, y_th: (
	(x_th := torch.exp(-y_th)),
	(-y_th) * (torch.abs(x) < x_th)
	+ torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th)
)
# piecewise-safe tan: returns (delta, value) to match the other functions
f_tan = lambda x, y_th: (
	(delta := torch.pi/2 - torch.arctan(y_th)),
	(
		# map x to [0, π) to measure distance from the asymptote at π/2
		- (y_th / delta) * ((torch.remainder(x, torch.pi) - torch.pi/2))
		  * (torch.abs(torch.remainder(x, torch.pi) - torch.pi/2) < delta)
		+ torch.nan_to_num(torch.tan(torch.remainder(x, torch.pi)))
		  * (torch.abs(torch.remainder(x, torch.pi) - torch.pi/2) >= delta)
	)
)
f_arctanh = lambda x, y_th: (
	(delta := 1 - torch.tanh(y_th) + 1e-4),
	y_th * torch.sign(x) * (torch.abs(x) > 1 - delta)
	+ torch.nan_to_num(torch.atanh(x)) * (torch.abs(x) <= 1 - delta)
)
# Return a consistent 2-tuple; use 1.0 as the domain threshold
f_arcsin = lambda x, y_th: (
	1.0,
	(torch.pi/2) * torch.sign(x) * (torch.abs(x) > 1)
	+ torch.nan_to_num(torch.asin(x)) * (torch.abs(x) <= 1)
)
f_arccos = lambda x, y_th: (
	1.0,
	(torch.pi/2) * (1 - torch.sign(x)) * (torch.abs(x) > 1)
	+ torch.nan_to_num(torch.acos(x)) * (torch.abs(x) <= 1)
)
f_exp = lambda x, y_th: (
	(x_th := torch.log(torch.clamp(y_th, min=torch.finfo(x.dtype).tiny))),
	torch.where(x > x_th, y_th, torch.exp(x))
)


# ---- helpers ---------------------------------------------------------------
def _eps_like(x, mul=1.0, min_eps=1e-8):
	if not x.is_floating_point():
		x = x.float()
	e = mul * torch.finfo(x.dtype).eps
	e = max(float(e), float(min_eps))
	return torch.as_tensor(e, device=x.device, dtype=x.dtype)

def _safe_pos(x, eps=None):
	if eps is None: eps = _eps_like(x, 1024.0)
	eps = eps.to(x.dtype) if torch.is_tensor(eps) else torch.as_tensor(eps, device=x.device, dtype=x.dtype)
	# Always >= eps, smooth everywhere
	return torch.sqrt(x * x + eps * eps)

def _safe_log(x, eps=None):
	r = torch.log(_safe_pos(x, eps=eps))
	r = torch.nan_to_num(r, nan=0.0, posinf=1e5, neginf=-1e5)
	# return _safe_clamp(r, -1e5, 1e5)
	return r

def _safe_invpow(x, k, eps=None):
	"""
	Smooth surrogate for 1 / x^k.
	Even k -> positive; odd k -> keeps sign(x).
	"""
	if eps is None: eps = _eps_like(x, 1024.0)
	ax = _safe_pos(x.abs(), eps)
	denom = ax.pow(k) + (eps ** k)
	if k % 2 == 0:
		r = 1.0 / denom
	else:
		r = _safe_sign(x) * (1.0 / denom)
	# r = _safe_clamp(r, 1e5, -1e5)
	return r

def _safe_recip(x, eps=None):
	if eps is None: eps = _eps_like(x, 1024.0)
	x = x.to(dtype=torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
	eps = eps.to(x.dtype) if torch.is_tensor(eps) else torch.as_tensor(eps, device=x.device, dtype=x.dtype)
	r = x / (x * x + eps * eps)
	# return _safe_clamp(r, -1e5, 1e5)
	return r

def _safe_sign(x, k=64.0):
	"""Smooth sign via tanh(kx); k controls sharpness."""
	return torch.tanh(torch.as_tensor(k, dtype=x.dtype, device=x.device) * x)

def _safe_sqrt(x, eps=None):
	if eps is None: eps = _eps_like(x, 1024.0)
	return torch.sqrt(_safe_pos(x, eps=eps))  # safe_pos already squares, smooth

def _safe_invsqrt(x):
	return _safe_recip(_safe_sqrt(x))

def _softplus_stable(z: torch.Tensor, beta: float = 64.0) -> torch.Tensor:
	compute_dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
	t = (z.to(compute_dtype) * float(beta))
	out = (torch.log1p(torch.exp(-torch.abs(t))) + torch.clamp(t, min=0.0)) / float(beta)
	return out.to(z.dtype)


def _safe_clamp(x: torch.Tensor, lo, hi, *, beta: float = 64.0, nan_fill: float = 0.0) -> torch.Tensor:
	"""
	Smooth clamp approximation using stable softplus, with guards for NaNs/Infs.

	  two-sided: y ≈ lo + sp(x-lo) - sp(x-hi)
	  one-sided: y ≈ lo + sp(x-lo)    or    y ≈ x - sp(x-hi)

	Additionally:
	- NaNs -> nan_fill
	- For two-sided: +inf -> hi, -inf -> lo  (prevents inf-inf)
	"""
	x = torch.nan_to_num(x, nan=nan_fill)  # keep +/-inf for now

	lo_t = torch.as_tensor(lo, device=x.device, dtype=x.dtype) if lo is not None else None
	hi_t = torch.as_tensor(hi, device=x.device, dtype=x.dtype) if hi is not None else None

	if (lo_t is not None) and (hi_t is not None):
		lo_t, hi_t = torch.minimum(lo_t, hi_t), torch.maximum(lo_t, hi_t)

		# Snap infinities to bounds to avoid inf-inf in the smooth expression.
		x = torch.where(torch.isposinf(x), hi_t, x)
		x = torch.where(torch.isneginf(x), lo_t, x)

		# Saturate outside bounds via where; smooth only inside.
		inside = (x > lo_t) & (x < hi_t)
		smooth_inside = lo_t + _softplus_stable(x - lo_t, beta=beta) - _softplus_stable(x - hi_t, beta=beta)
		return torch.where(x <= lo_t, lo_t, torch.where(x >= hi_t, hi_t, torch.where(inside, smooth_inside, x)))

	elif lo_t is not None:
		# Lower-bound only
		x = torch.where(torch.isneginf(x), lo_t, x)
		return torch.where(x <= lo_t, lo_t, lo_t + _softplus_stable(x - lo_t, beta=beta))

	elif hi_t is not None:
		# Upper-bound only
		x = torch.where(torch.isposinf(x), hi_t, x)
		return torch.where(x >= hi_t, hi_t, x - _softplus_stable(x - hi_t, beta=beta))

	else:
		return x

def _safe_exp(x):
	dtype = x.dtype if x.is_floating_point() else torch.float32
	finfo = torch.finfo(dtype)
	hi = math.log(float(finfo.max))
	# optional: lo = math.log(float(finfo.tiny))  # if you care about underflow
	# x = _safe_clamp(x, -hi, hi)
	return torch.exp(x)

def _safe_tan(x):
	return torch.sin(x) * _safe_recip(torch.cos(x))

def _safe_arcsin(x: torch.Tensor) -> torch.Tensor:
	return torch.arcsin(_safe_clamp(x, -1.0, 1.0))


def _safe_arccos(x: torch.Tensor) -> torch.Tensor:
	return torch.arccos(_safe_clamp(x, -1.0, 1.0))

def _safe_arctanh(x: torch.Tensor, eps: float = None):
	# Make eps dtype-aware if not provided (important for fp16/bf16!)
	if eps is None:
		# A slightly larger margin than finfo.eps helps for low precision
		eps = float(8.0 * torch.finfo(x.dtype).eps) if x.is_floating_point() else 1e-6
	return torch.atanh(_safe_clamp(x, -1.0 + eps, 1.0 - eps))

# ---- your derivative/threshold helpers (assumed pre-defined) ---------------
# f_inv, f_inv2, f_inv3, f_inv4, f_inv5, f_sqrt, f_power1d5, f_invsqrt, f_exp, f_log, f_tan, f_arcsin, f_arccos, f_arctanh
# (leaving as-is; only replacing the runtime torch lambdas)

# ---- NaN-proof library -----------------------------------------------------
SYMBOLIC_LIB = {
	'0':           (lambda x: x*0, lambda x: x*0,       0, None),
	'1':           (lambda x: x*0+1, lambda x: x*0+1,       0, None),
	'x':           (lambda x: x,           lambda x: x,                 0, None),
	
	'x^2':         (lambda x: x**2,        lambda x: x**2,              2, None),
	'x^3':         (lambda x: x**3,        lambda x: x**3,              3, None),
	'x^4':         (lambda x: x**4,        lambda x: x**4,              4, None),
	'x^5':         (lambda x: x**5,        lambda x: x**5,              5, None),

	'1/x':   (lambda x: _safe_invpow(x, 1),  lambda x: 1/x,     1, f_inv),
	'1/x^2': (lambda x: _safe_invpow(x, 2),  lambda x: 1/x**2,  2, f_inv2),
	'1/x^3': (lambda x: _safe_invpow(x, 3),  lambda x: 1/x**3,  3, f_inv3),
	'1/x^4': (lambda x: _safe_invpow(x, 4),  lambda x: 1/x**4,  4, f_inv4),
	'1/x^5': (lambda x: _safe_invpow(x, 5),  lambda x: 1/x**5,  5, f_inv5),

	'sqrt':        (lambda x: _safe_sqrt(x),       lambda x: sympy.sqrt(x),     2, f_sqrt),
	# 'x^0.5':       (lambda x: _safe_sqrt(x),       lambda x: sympy.sqrt(x),     2, f_sqrt),
	'x^1.5': (lambda x: x.abs()**1.5, lambda x: sympy.Abs(x)**sympy.Rational(3,2), 4, f_power1d5),

	'1/sqrt(x)':   (lambda x: _safe_invsqrt(x),    lambda x: 1/sympy.sqrt(x),   2, f_invsqrt),
	# '1/x^0.5':     (lambda x: _safe_invsqrt(x),    lambda x: 1/sympy.sqrt(x),   2, f_invsqrt),

	'exp':         (lambda x: _safe_exp(x),        lambda x: sympy.exp(x),      2, f_exp),
	'log':         (lambda x: _safe_log(x),        lambda x: sympy.log(x),      2, f_log),

	'abs':         (lambda x: x.abs(),    lambda x: sympy.Abs(x),  3, None),
	'sin':         (lambda x: torch.sin(x),    lambda x: sympy.sin(x),  2, None),
	'cos':         (lambda x: torch.cos(x),    lambda x: sympy.cos(x),  2, None),
	'tan':         (lambda x: _safe_tan(x),    lambda x: sympy.tan(x),  3, f_tan),
	'tanh':        (lambda x: torch.tanh(x),   lambda x: sympy.tanh(x), 3, None),

	'sgn':         (lambda x: _safe_sign(x),   lambda x: sympy.sign(x), 3, None),

	'arcsin':      (lambda x: _safe_arcsin(x),     lambda x: sympy.asin(x),     4, f_arcsin),
	'arccos':      (lambda x: _safe_arccos(x),     lambda x: sympy.acos(x),     4, f_arccos),
	'arctan':      (lambda x: torch.arctan(x),     lambda x: sympy.atan(x),     4, None),
	'arctanh':     (lambda x: _safe_arctanh(x),    lambda x: sympy.atanh(x),    4, f_arctanh),

	'gaussian':    (lambda x: _safe_exp(-x**2),    lambda x: sympy.exp(-x**2), 3, lambda x, y_th: f_exp(-x**2, y_th)),
	# 'cosh':      (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5),
	# 'sigmoid':   (lambda x: torch.sigmoid(x), sympy.Function('sigmoid'), 4),
	# 'relu':      (lambda x: torch.relu(x), relu),
}

def create_dataset(f, 
				   n_var=2, 
				   f_mode = 'col',
				   ranges = [-1,1],
				   train_num=1000, 
				   test_num=1000,
				   normalize_input=False,
				   normalize_label=False,
				   device='cpu',
				   seed=0):
	'''
	create dataset
	
	Args:
	-----
		f : function
			the symbolic formula used to create the synthetic dataset
		ranges : list or np.array; shape (2,) or (n_var, 2)
			the range of input variables. Default: [-1,1].
		train_num : int
			the number of training samples. Default: 1000.
		test_num : int
			the number of test samples. Default: 1000.
		normalize_input : bool
			If True, apply normalization to inputs. Default: False.
		normalize_label : bool
			If True, apply normalization to labels. Default: False.
		device : str
			device. Default: 'cpu'.
		seed : int
			random seed. Default: 0.
		
	Returns:
	--------
		dataset : dic
			Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
						dataset['test_input'], dataset['test_label']
		 
	Example
	-------
	>>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
	>>> dataset = create_dataset(f, n_var=2, train_num=100)
	>>> dataset['train_input'].shape
	torch.Size([100, 2])
	'''

	np.random.seed(seed)
	torch.manual_seed(seed)

	if len(np.array(ranges).shape) == 1:
		ranges = np.array(ranges * n_var).reshape(n_var,2)
	else:
		ranges = np.array(ranges)
		
	
	train_input = torch.zeros(train_num, n_var)
	test_input = torch.zeros(test_num, n_var)
	for i in range(n_var):
		train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
		test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
				
	if f_mode == 'col':
		train_label = f(train_input)
		test_label = f(test_input)
	elif f_mode == 'row':
		train_label = f(train_input.T)
		test_label = f(test_input.T)
	else:
		print(f'f_mode {f_mode} not recognized')
		
	# if has only 1 dimension
	if len(train_label.shape) == 1:
		train_label = train_label.unsqueeze(dim=1)
		test_label = test_label.unsqueeze(dim=1)
		
	def normalize(data, mean, std):
			return (data-mean)/std
			
	if normalize_input == True:
		mean_input = torch.mean(train_input, dim=0, keepdim=True)
		std_input = torch.std(train_input, dim=0, keepdim=True)
		train_input = normalize(train_input, mean_input, std_input)
		test_input = normalize(test_input, mean_input, std_input)
		
	if normalize_label == True:
		mean_label = torch.mean(train_label, dim=0, keepdim=True)
		std_label = torch.std(train_label, dim=0, keepdim=True)
		train_label = normalize(train_label, mean_label, std_label)
		test_label = normalize(test_label, mean_label, std_label)

	dataset = {}
	dataset['train_input'] = train_input.to(device)
	dataset['test_input'] = test_input.to(device)

	dataset['train_label'] = train_label.to(device)
	dataset['test_label'] = test_label.to(device)

	return dataset

@torch.no_grad()
def _ols_cd(F, Y, ridge=1e-8):
	"""
	Solve [ [sum F^2, sum F],
			[sum F,   N   ] ] [c, d]^T = [sum F*Y, sum Y]^T
	Returns c, d as scalars on F's device/dtype.
	"""
	N = F.numel()
	if N == 0:
		return torch.tensor(0., device=Y.device, dtype=Y.dtype), torch.mean(Y)

	sF2 = torch.sum(F * F)
	sF  = torch.sum(F)
	sFY = torch.sum(F * Y)
	sY  = torch.sum(Y)

	A = torch.stack([
		torch.stack([sF2 + ridge, sF], dim=0),
		torch.stack([sF,         torch.tensor(float(N), device=F.device, dtype=F.dtype)], dim=0)
	], dim=0)
	b = torch.stack([sFY, sY], dim=0)

	try:
		sol = torch.linalg.solve(A, b)
		c, d = sol[0], sol[1]
	except RuntimeError:
		# extreme degeneracy fallback
		c = torch.tensor(0., device=F.device, dtype=F.dtype)
		d = torch.mean(Y)
	return c, d

def old_fit_params(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, verbose=True, device='cpu'):
	'''
	fit a, b, c, d such that
	
	.. math::
		|y-(cf(ax+b)+d)|^2
		
	is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.
	
	Args:
	-----
		x : 1D array
			x values
		y : 1D array
			y values
		fun : function
			symbolic function
		a_range : tuple
			sweeping range of a
		b_range : tuple
			sweeping range of b
		grid_num : int
			number of steps along a and b
		iteration : int
			number of zooming in
		verbose : bool
			print extra information if True
		device : str
			device
		
	Returns:
	--------
		a_best : float
			best fitted a
		b_best : float
			best fitted b
		c_best : float
			best fitted c
		d_best : float
			best fitted d
		r2_best : float
			best r2 (coefficient of determination)
	
	Example
	-------
	>>> num = 100
	>>> x = torch.linspace(-1,1,steps=num)
	>>> noises = torch.normal(0,1,(num,)) * 0.02
	>>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
	>>> fit_params(x, y, torch.sin)
	r2 is 0.9999727010726929
	(tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
	'''
	# fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
	# sweep a and b, choose the best fitted model   
	for _ in range(iteration):
		a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
		b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
		a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
		post_fun = fun(a_grid[None,:,:] * x[:,None,None] + b_grid[None,:,:])
		x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
		y_mean = torch.mean(y, dim=[0], keepdim=True)
		numerator = torch.sum((post_fun - x_mean)*(y-y_mean)[:,None,None], dim=0)**2
		denominator = torch.sum((post_fun - x_mean)**2, dim=0)*torch.sum((y - y_mean)[:,None,None]**2, dim=0)
		r2 = numerator/(denominator+1e-4)
		r2 = torch.nan_to_num(r2)
		
		
		best_id = torch.argmax(r2)
		a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number
		
		
		if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
			if _ == 0 and verbose==True:
				print('Best value at boundary.')
			if a_id == 0:
				a_range = [a_[0], a_[1]]
			if a_id == grid_number - 1:
				a_range = [a_[-2], a_[-1]]
			if b_id == 0:
				b_range = [b_[0], b_[1]]
			if b_id == grid_number - 1:
				b_range = [b_[-2], b_[-1]]
			
		else:
			a_range = [a_[a_id-1], a_[a_id+1]]
			b_range = [b_[b_id-1], b_[b_id+1]]
			
	a_best = a_[a_id]
	b_best = b_[b_id]
	post_fun = fun(a_best * x + b_best)
	r2_best = r2[a_id, b_id]
	
	if verbose == True:
		print(f"r2 is {r2_best}")
		if r2_best < 0.9:
			print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

	post_fun = torch.nan_to_num(post_fun)
	reg = LinearRegression().fit(post_fun[:,None].detach().cpu().numpy(), y.detach().cpu().numpy())
	c_best = torch.from_numpy(reg.coef_)[0].to(device)
	d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
	return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def fit_params(
	x, y, fun=None, *, funs=None,
	steps=200, lr=1e-1, restarts=5, verbose=True, use_lbfgs=True,
	device='cpu', dtype=None, eps=1e-12, topk=12,
	# regularization for (a,b); (c,d) get tiny ridge through OLS
	reg_type="elasticnet", lam=1e-3, centers=(1.0, 0.0), weights=(1.0, 1.0),
	# seeding / domain handling
	grid_scales=(1e-2, 1e-1, 1.0, 10.0, 100.0),
	grid_bspan_std=2.0, domain_penalty=1e6, seed_random_scale=0.25
):
	"""
	Fit y ≈ c * fun(a * x + b) + d by optimizing only (a,b).
	c,d are solved in closed form at every loss eval.
	"""

	assert (fun is not None) or (funs is not None), "Provide `fun` or `funs`."

	# ---- move + sanitize inputs
	if not torch.is_tensor(x): x = torch.as_tensor(x)
	if not torch.is_tensor(y): y = torch.as_tensor(y)
	x = x.detach().to(device)
	y = y.detach().to(device)
	if dtype is None: dtype = x.dtype
	x = x.to(dtype).reshape(-1).contiguous()
	y = y.to(dtype).reshape(-1).contiguous()

	N = x.numel()
	if N < 4:
		raise ValueError("Need at least 4 data points.")

	# Prepare regularization targets for (a,b)
	centers_ab = torch.tensor(centers, device=device, dtype=dtype)  # (a,b) centers
	weights_ab = torch.tensor(weights, device=device, dtype=dtype)  # (a,b) weights

	def _reg_ab(a, b):
		# if reg_type is None:
		#     return 0
		ab = torch.stack([a, b])
		diff = ab - centers_ab
		if reg_type == "l2":
			return lam * N * torch.sum(weights_ab * diff.pow(2))
		elif reg_type == "l1":
			return lam * N * torch.sum(weights_ab * diff.abs())
		elif reg_type == "elasticnet":
			return lam * N * (0.5 * torch.sum(weights_ab * diff.abs()) +
							  0.5 * torch.sum(weights_ab * diff.pow(2)))
		else:
			raise ValueError("reg_type must be 'l2', 'l1', or 'elasticnet'")

	def _loss_for_fun(fun_callable, a, b, need_pred=False):
		z = a * x + b
		f = fun_callable(z)

		valid = torch.isfinite(f)
		valid_frac = valid.float().mean()
		invalid_frac = 1.0 - valid_frac

		if valid_frac == 0:
			# All invalid: huge penalty; dummy outputs
			penalty = domain_penalty * N
			y_hat = torch.zeros_like(y)
			return penalty + _reg_ab(a,b), y_hat, torch.tensor(0., device=device, dtype=dtype), torch.tensor(0., device=device, dtype=dtype)

		# Solve for c,d on the valid subset
		Fv, Yv = f[valid], y[valid]
		c, d = _ols_cd(Fv, Yv, ridge=1e-8)

		# Build full prediction (fill invalid with mean prediction to keep shapes consistent)
		y_hat = c * f + d

		if need_pred:
			# optionally fill invalids (example: with mean of valid predictions)
			if (~valid).any():
				y_hat = y_hat.clone()
				y_hat[~valid] = torch.mean(y_hat[valid])
				
		# Penalize invalid domain proportionally to scale of the residuals
		scale = torch.var(y) + eps
		penalty = domain_penalty * (invalid_frac ** 2) * (float(N)) * scale

		resid = (y_hat - y)
		sse = torch.sum((resid[valid]) ** 2)  # only where model is defined

		loss = sse + penalty + _reg_ab(a, b)

		if need_pred:
			return loss, y_hat, c, d
		return loss, None, c, d

	# ---- smart seed generator for (a,b) using a coarse grid and OLS for (c,d)
	def _seed_grid(fun_callable):
		x_med = torch.median(x)
		x_std = torch.std(x) + 1e-12
		base_bs = torch.linspace(-grid_bspan_std, grid_bspan_std, steps=5, device=device, dtype=dtype) * x_std

		seeds = []
		for s in grid_scales:
			for sign in (1.0, -1.0):
				a0 = torch.tensor(sign * s, device=device, dtype=dtype)
				for off in base_bs:
					b0 = -a0 * x_med + off
					loss, _, c0, d0 = _loss_for_fun(fun_callable, a0, b0, need_pred=False)
					seeds.append((float(loss), float(a0), float(b0), float(c0), float(d0)))

		seeds.sort(key=lambda t: t[0])
		# Add a few random jitters around the best seeds
		best = seeds[:max(4, topk//2)]
		rng = torch.Generator(device=device)
		for _ in range(restarts):
			for L, a0, b0, c0, d0 in best:
				aj = a0 * (1.0 + seed_random_scale * torch.randn((), generator=rng, device=device, dtype=dtype).item())
				bj = b0 + seed_random_scale * x_std.item() * torch.randn((), generator=rng, device=device, dtype=dtype).item()
				loss, _, c1, d1 = _loss_for_fun(fun_callable, torch.tensor(aj, device=device, dtype=dtype),
												torch.tensor(bj, device=device, dtype=dtype), need_pred=False)
				seeds.append((float(loss), float(aj), float(bj), float(c1), float(d1)))
		seeds.sort(key=lambda t: t[0])
		return seeds[:topk]

	# ---- optimizer over (a,b)
	def _optimize_ab(fun_callable, a0, b0):
		a = torch.nn.Parameter(torch.tensor(a0, device=device, dtype=dtype))
		b = torch.nn.Parameter(torch.tensor(b0, device=device, dtype=dtype))

		if use_lbfgs:
			optimizer = torch.optim.LBFGS([a, b], lr=lr, max_iter=steps, line_search_fn='strong_wolfe')
			def closure():
				optimizer.zero_grad(set_to_none=True)
				loss, _, _, _ = _loss_for_fun(fun_callable, a, b, need_pred=False)
				loss.backward()
				return loss
			try:
				optimizer.step(closure)
			except RuntimeError:
				# fall back to Adam a little if LS fails
				opt2 = torch.optim.Adam([a, b], lr=min(lr, 0.1))
				prev = float("inf"); patience=20; stall=0
				for _ in range(min(steps, 200)):
					opt2.zero_grad(set_to_none=True)
					loss, _, _, _ = _loss_for_fun(fun_callable, a, b, need_pred=False)
					loss.backward(); opt2.step()
					cur = float(loss.detach())
					if cur > prev - 1e-9:
						stall += 1
						if stall >= patience: break
					else:
						stall = 0; prev = cur
		else:
			optimizer = torch.optim.Adam([a, b], lr=lr)
			for _ in range(steps):
				optimizer.zero_grad(set_to_none=True)
				loss, _, _, _ = _loss_for_fun(fun_callable, a, b, need_pred=False)
				loss.backward()
				optimizer.step()

		with torch.no_grad():
			final_loss, y_hat, c, d = _loss_for_fun(fun_callable, a, b, need_pred=True)
		return (float(final_loss), float(a.detach()), float(b.detach()),
				float(c.detach()), float(d.detach()), y_hat)

	# ---- try one or many funs
	candidates = funs if (funs is not None) else [fun]
	best_all = None

	for fidx, fcall in enumerate(candidates):
		seeds = _seed_grid(fcall)

		best_fun_loss = None
		best_pack = None
		for L, a0, b0, _, _ in seeds:
			cur = _optimize_ab(fcall, a0, b0)
			if (best_fun_loss is None) or (cur[0] < best_fun_loss):
				best_fun_loss = cur[0]
				best_pack = cur  # (loss, a, b, c, d, y_hat)

		# final metrics for this fun
		loss_f, a_f, b_f, c_f, d_f, y_hat = best_pack
		with torch.no_grad():
			valid = torch.isfinite(y_hat)
			y_mean_v = torch.mean(y[valid])
			ss_tot = torch.sum((y[valid] - y_mean_v).pow(2))
			ss_res = torch.sum((y[valid] - y_hat[valid]).pow(2))
			r2 = 1.0 - (ss_res / (ss_tot + eps))

			r2 = torch.clamp(r2, min=-1e6)
			r2 = torch.nan_to_num(r2, nan=-1e6, posinf=-1e6, neginf=-1e6)

			# # AICc (k=4 parameters: a,b,c,d)
			# k = 4
			# sse = ss_res
			# aic = float(N) * log(float(sse / (valid.sum().item() + eps)) + 1e-12) + 2 * k
			# aicc = aic + (2 * k * (k + 1)) / max(1.0, (N - k - 1))

		pack = {
			"fun": fcall,
			"params": torch.tensor([a_f, b_f, c_f, d_f], device=device, dtype=dtype),
			"r2": r2.to(device=device, dtype=dtype),
			"loss": torch.tensor(loss_f, device=device, dtype=dtype),
			# "aicc": aicc,
		}
		if (best_all is None) or (pack["r2"] > best_all["r2"]):
			best_all = pack

	if verbose:
		print(f"r2 is {float(best_all['r2']):.6f}")
		if best_all["r2"] < 0.9:
			print("r2 is not very high; check if the chosen function matches your data.")

	return best_all["params"], best_all['r2'], best_all['loss']



# @torch.no_grad()
# def _prefilter_seeds(fun, x, y, seeds, k=6, device='cpu', dtype=None):
#     """Rank seeds by quick SSE (no grads), return top-k tuples."""
#     if dtype is None: dtype = x.dtype
#     scores = []
#     for (a0, b0, c0, d0) in seeds:
#         a = torch.tensor(a0, device=device, dtype=dtype)
#         b = torch.tensor(b0, device=device, dtype=dtype)
#         c = torch.tensor(c0, device=device, dtype=dtype)
#         d = torch.tensor(d0, device=device, dtype=dtype)
#         y_hat = c * fun(a * x + b) + d
#         y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
#         sse = torch.sum((y_hat - y).pow(2))
#         val = float(sse)
#         val = val if isfinite(val) else float("inf")
#         scores.append((val, (a0, b0, c0, d0)))
#     scores.sort(key=lambda t: t[0])
#     k = min(k, len(scores))
#     return [tpl for _, tpl in scores[:k]]

# def fit_params(
#     x, y, fun,
#     *, steps=200, lr=0.5, restarts=5, verbose=True, use_lbfgs=False,
#     device='cpu', dtype=None, eps=1e-12, topk=None,
#     # regularization defaults (can be overridden via **loss_kwargs)
#     **loss_kwargs
# ):
#     """
#     Fit y ≈ c * fun(a * x + b) + d by optimizing (a,b,c,d).
#     Returns: params[4], r2, loss
#     """
#     # ---- move + sanitize inputs
#     if not torch.is_tensor(x): x = torch.as_tensor(x)
#     if not torch.is_tensor(y): y = torch.as_tensor(y)
#     x = x.detach().to(device)
#     y = y.detach().to(device)
#     if dtype is None: dtype = x.dtype
#     x = x.to(dtype).reshape(-1).contiguous()
#     y = y.to(dtype).reshape(-1).contiguous()

#     # ---- constants prepared once
#     N = x.numel()
#     centers = loss_kwargs.pop("centers", (1.0, 0.0, 1.0, 0.0))
#     weights = loss_kwargs.pop("weights", (1.0, 1.0, 1.0, 1.0))
#     reg_type = loss_kwargs.pop("reg_type", "elasticnet")
#     lam = loss_kwargs.pop("lam", 1e-2)

#     centers_t = torch.tensor(centers, device=device, dtype=dtype)
#     weights_t = torch.tensor(weights, device=device, dtype=dtype)

#     # ---- loss (returns scalar loss; y_hat only when asked)
#     def _loss(a, b, c, d, need_pred=False):
#         y_hat = c * fun(a * x + b) + d
#         # y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)

#         resid = y_hat - y
#         sse = torch.sum(resid * resid)

#         abcd = torch.stack([a, b, c, d])
#         diff = abcd - centers_t
#         if reg_type == "l2":
#             reg = lam * N * torch.sum(weights_t * diff.pow(2))
#         elif reg_type == "l1":
#             reg = lam * N * torch.sum(weights_t * diff.abs())
#         elif reg_type == "elasticnet":
#             reg = lam * N * (0.5 * torch.sum(weights_t * diff.abs()) +
#                              0.5 * torch.sum(weights_t * diff.pow(2)))
#         else:
#             raise ValueError("reg_type must be 'l2', 'l1', or 'elasticnet'")
#         if need_pred:
#             return sse + reg, y_hat
#         return sse + reg

#     # ---- seeds
#     seeds = [
#         (1., 0., 1., 0.),
#         (-1., 0., -1., 0.),
#         (10., 0., 1., 0.),
#         (-10., 0., -1., 0.),
#         (100., 0., 100., 0.),
#         (-100., 0., -100., 0.),
#     ]
	
#     # ---- quick prefilter: only optimize best few seeds
#     if topk:
#         seeds = _prefilter_seeds(fun, x, y, seeds, k=topk, device=device, dtype=dtype)

#     # ---- track best solution
#     best_loss = None
#     best_params = None

#     # ---- optimize per-seed
#     for a0, b0, c0, d0 in seeds:
#         # parameters as independent scalars (tiny problem -> LBFGS shines)
#         a = torch.nn.Parameter(torch.tensor(a0, device=device, dtype=dtype))
#         b = torch.nn.Parameter(torch.tensor(b0, device=device, dtype=dtype))
#         c = torch.nn.Parameter(torch.tensor(c0, device=device, dtype=dtype))
#         d = torch.nn.Parameter(torch.tensor(d0, device=device, dtype=dtype))

#         if use_lbfgs:
#             optimizer = torch.optim.LBFGS([a, b, c, d], lr=lr, max_iter=steps, line_search_fn='strong_wolfe')

#             def closure():
#                 optimizer.zero_grad(set_to_none=True)
#                 loss = _loss(a, b, c, d, need_pred=False)
#                 loss.backward()
#                 return loss

#             try:
#                 optimizer.step(closure)
#             except RuntimeError:
#                 # rare LBFGS line-search failures -> fallback to Adam a few steps
#                 opt2 = torch.optim.Adam([a, b, c, d], lr=min(lr, 0.1))
#                 prev = float("inf")
#                 patience, stall = 20, 0
#                 for _ in range(min(steps, 200)):
#                     opt2.zero_grad(set_to_none=True)
#                     loss = _loss(a, b, c, d)
#                     loss.backward()
#                     opt2.step()
#                     cur = float(loss.detach())
#                     if cur > prev - 1e-9:
#                         stall += 1
#                         if stall >= patience:
#                             break
#                     else:
#                         stall = 0
#                         prev = cur
#         else:
#             opt2 = torch.optim.Adam([a, b, c, d], lr=lr)
#             prev = float("inf")
#             patience, stall = 25, 0
#             for _ in range(steps):
#                 opt2.zero_grad(set_to_none=True)
#                 loss = _loss(a, b, c, d)
#                 loss.backward()
#                 opt2.step()
#                 cur = float(loss.detach())
#                 # early stop on plateau
#                 if cur > prev - 1e-9:
#                     stall += 1
#                     if stall >= patience:
#                         break
#                 else:
#                     stall = 0
#                     prev = cur

#         with torch.no_grad():
#             final_loss = float(_loss(a, b, c, d))
#             if (best_loss is None) or (final_loss < best_loss):
#                 best_loss = final_loss
#                 best_params = torch.stack([a.detach(), b.detach(), c.detach(), d.detach()])

#     # ---- final metrics
#     with torch.no_grad():
#         a_best, b_best, c_best, d_best = best_params
#         _, y_hat = _loss(a_best, b_best, c_best, d_best, need_pred=True)
#         # R^2 (self-contained to avoid external dependency)
#         y_mean = torch.mean(y)
#         ss_tot = torch.sum((y - y_mean).pow(2))
#         ss_res = torch.sum((y - y_hat).pow(2))
#         r2 = 1.0 - (ss_res / (ss_tot + eps))
#         r2 = torch.clamp(r2, min=-1e6, max=1.0)  # guard degenerate cases
#         r2 = torch.nan_to_num(r2, nan=0.0, posinf=1, neginf=-1e8)

#     if verbose:
#         print(f"r2 is {float(r2):.6f}")
#         if r2 < 0.9:
#             print("r2 is not very high; check if the chosen function matches your data.")

#     return best_params, r2, torch.tensor(best_loss, device=device, dtype=dtype)




def sparse_mask(in_dim, out_dim):
	'''
	get sparse mask
	'''
	in_coord = torch.arange(in_dim) * 1/in_dim + 1/(2*in_dim)
	out_coord = torch.arange(out_dim) * 1/out_dim + 1/(2*out_dim)

	dist_mat = torch.abs(out_coord[:,None] - in_coord[None,:])
	in_nearest = torch.argmin(dist_mat, dim=0)
	in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1,0)
	out_nearest = torch.argmin(dist_mat, dim=1)
	out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1,0)
	all_connection = torch.cat([in_connection, out_connection], dim=0)
	mask = torch.zeros(in_dim, out_dim)
	mask[all_connection[:,0], all_connection[:,1]] = 1.
	
	return mask


def add_symbolic(name, fun, c=1, fun_singularity=None, sympy_fun=None):
	if sympy_fun is None:
		sympy_fun = sympy.Function(f'{name}')
	SYMBOLIC_LIB[name] = (fun, sympy_fun, c, fun_singularity)
	
  
def ex_round(ex1, n_digit):
	'''
	rounding the numbers in an expression to certain floating points
	
	Args:
	-----
		ex1 : sympy expression
		n_digit : int
		
	Returns:
	--------
		ex2 : sympy expression
	
	Example
	-------
	>>> from symbolic_kan.utils import *
	>>> from sympy import *
	>>> input_vars = a, b = symbols('a b')
	>>> expression = 3.14534242 * exp(sin(pi*a) + b**2) - 2.32345402
	>>> ex_round(expression, 2)
	'''
	ex2 = ex1
	for a in sympy.preorder_traversal(ex1):
		if isinstance(a, sympy.Float):
			ex2 = ex2.subs(a, round(a, n_digit))
	return ex2


def augment_input(orig_vars, aux_vars, x):
	'''
	augment inputs
	
	Args:
	-----
		orig_vars : list of sympy symbols
		aux_vars : list of auxiliary symbols
		x : inputs
		
	Returns:
	--------
		augmented inputs
	
	Example
	-------
	>>> from symbolic_kan.utils import *
	>>> from sympy import *
	>>> orig_vars = a, b = symbols('a b')
	>>> aux_vars = [a + b, a * b]
	>>> x = torch.rand(100, 2)
	>>> augment_input(orig_vars, aux_vars, x).shape
	'''
	# if x is a tensor
	if isinstance(x, torch.Tensor):
		
		aux_values = torch.tensor([]).to(x.device)

		for aux_var in aux_vars:
			func = lambdify(orig_vars, aux_var,'numpy') # returns a numpy-ready function
			aux_value = torch.from_numpy(func(*[x[:,[i]].numpy() for i in range(len(orig_vars))]))
			aux_values = torch.cat([aux_values, aux_value], dim=1)
			
		x = torch.cat([aux_values, x], dim=1)

	# if x is a dataset
	elif isinstance(x, dict):
		x['train_input'] = augment_input(orig_vars, aux_vars, x['train_input'])
		x['test_input'] = augment_input(orig_vars, aux_vars, x['test_input'])
		
	return x


def batch_jacobian(func, x, create_graph=False, mode='scalar'):
	'''
	jacobian
	
	Args:
	-----
		func : function or model
		x : inputs
		create_graph : bool
		
	Returns:
	--------
		jacobian
	
	Example
	-------
	>>> from symbolic_kan.utils import batch_jacobian
	>>> x = torch.normal(0,1,size=(100,2))
	>>> model = lambda x: x[:,[0]] + x[:,[1]]
	>>> batch_jacobian(model, x)
	'''
	# x in shape (Batch, Length)
	def _func_sum(x):
		return func(x).sum(dim=0)
	if mode == 'scalar':
		return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph)[0]
	elif mode == 'vector':
		return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

def batch_hessian(model, x, create_graph=False):
	'''
	hessian
	
	Args:
	-----
		func : function or model
		x : inputs
		create_graph : bool
		
	Returns:
	--------
		jacobian
	
	Example
	-------
	>>> from symbolic_kan.utils import batch_hessian
	>>> x = torch.normal(0,1,size=(100,2))
	>>> model = lambda x: x[:,[0]]**2 + x[:,[1]]**2
	>>> batch_hessian(model, x)
	'''
	# x in shape (Batch, Length)
	jac = lambda x: batch_jacobian(model, x, create_graph=True)
	def _jac_sum(x):
		return jac(x).sum(dim=0)
	return torch.autograd.functional.jacobian(_jac_sum, x, create_graph=create_graph).permute(1,0,2)


def create_dataset_from_data(inputs, labels, train_ratio=0.8, device='cpu'):
	'''
	create dataset from data
	
	Args:
	-----
		inputs : 2D torch.float
		labels : 2D torch.float
		train_ratio : float
			the ratio of training fraction
		device : str
		
	Returns:
	--------
		dataset (dictionary)
	
	Example
	-------
	>>> from symbolic_kan.utils import create_dataset_from_data
	>>> x = torch.normal(0,1,size=(100,2))
	>>> y = torch.normal(0,1,size=(100,1))
	>>> dataset = create_dataset_from_data(x, y)
	>>> dataset['train_input'].shape
	'''
	num = inputs.shape[0]
	train_id = np.random.choice(num, int(num*train_ratio), replace=False)
	test_id = list(set(np.arange(num)) - set(train_id))
	dataset = {}
	dataset['train_input'] = inputs[train_id].detach().to(device)
	dataset['test_input'] = inputs[test_id].detach().to(device)
	dataset['train_label'] = labels[train_id].detach().to(device)
	dataset['test_label'] = labels[test_id].detach().to(device)
	
	return dataset


def get_derivative(model, inputs, labels, derivative='hessian', loss_mode='pred', reg_metric='w', lamb=0., lamb_l1=1., lamb_entropy=0.):
	'''
	compute the jacobian/hessian of loss wrt to model parameters
	
	Args:
	-----
		inputs : 2D torch.float
		labels : 2D torch.float
		derivative : str
			'jacobian' or 'hessian'
		device : str
		
	Returns:
	--------
		jacobian or hessian
	'''
	def get_mapping(model):

		mapping = {}
		name = 'model1'

		keys = list(model.state_dict().keys())
		for key in keys:

			y = re.findall(".[0-9]+", key)
			if len(y) > 0:
				y = y[0][1:]
				x = re.split(".[0-9]+", key)
				mapping[key] = name + '.' + x[0] + '[' + y + ']' + x[1]


			y = re.findall("_[0-9]+", key)
			if len(y) > 0:
				y = y[0][1:]
				x = re.split(".[0-9]+", key)
				mapping[key] = name + '.' + x[0] + '[' + y + ']'

		return mapping

	
	#model1 = copy.deepcopy(model)
	model1 = model.copy()
	mapping = get_mapping(model)
   
	# collect keys and shapes
	keys = list(model.state_dict().keys())
	shapes = []

	for params in model.parameters():
		shapes.append(params.shape)


	# turn a flattened vector to model params
	def param2statedict(p, keys, shapes):

		new_state_dict = {}

		start = 0
		n_group = len(keys)
		for i in range(n_group):
			shape = shapes[i]
			n_params = torch.prod(torch.tensor(shape))
			new_state_dict[keys[i]] = p[start:start+n_params].reshape(shape)
			start += n_params

		return new_state_dict
	
	def differentiable_load_state_dict(mapping, state_dict, model1):

		for key in keys:
			if mapping[key][-1] != ']':
				exec(f"del {mapping[key]}")
			exec(f"{mapping[key]} = state_dict[key]")
			

	# input: p, output: output
	def get_param2loss_fun(inputs, labels):

		def param2loss_fun(p):

			p = p[0]
			state_dict = param2statedict(p, keys, shapes)
			# this step is non-differentiable
			#model.load_state_dict(state_dict)
			differentiable_load_state_dict(mapping, state_dict, model1)
			if loss_mode == 'pred':
				pred_loss = torch.mean((model1(inputs) - labels)**2, dim=(0,1), keepdim=True)
				loss = pred_loss
			elif loss_mode == 'reg':
				reg_loss = model1.get_reg(reg_metric=reg_metric, lamb_l1=lamb_l1, lamb_entropy=lamb_entropy) * torch.ones(1,1)
				loss = reg_loss
			elif loss_mode == 'all':
				pred_loss = torch.mean((model1(inputs) - labels)**2, dim=(0,1), keepdim=True)
				reg_loss = model1.get_reg(reg_metric=reg_metric, lamb_l1=lamb_l1, lamb_entropy=lamb_entropy) * torch.ones(1,1)
				loss = pred_loss + lamb * reg_loss
			return loss

		return param2loss_fun
	
	fun = get_param2loss_fun(inputs, labels)    
	p = model2param(model)[None,:]
	if derivative == 'hessian':
		result = batch_hessian(fun, p)
	elif derivative == 'jacobian':
		result = batch_jacobian(fun, p)
	return result

def model2param(model):
	'''
	turn model parameters into a flattened vector
	'''
	p = torch.tensor([]).to(model.device)
	for params in model.parameters():
		p = torch.cat([p, params.reshape(-1,)], dim=0)
	return p
