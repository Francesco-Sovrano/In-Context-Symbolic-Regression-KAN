from kan import *
from kan.utils import add_symbolic, SYMBOLIC_LIB, _safe_log, _safe_recip, _safe_pos, _safe_abs, _safe_sign
import torch
from scipy import special
import numpy as np
import random
import sympy as sp
# from sympy import log as s_log
# from sympy import Ei as s_Ei
# from sympy import Function
from sympy import S
import torch.nn.functional as F
import re
import json
from numbers import Real
import math
import argparse

torch.manual_seed(0); random.seed(0); np.random.seed(0)
torch.use_deterministic_algorithms(False)  # KAN splines can benefit from non-strict determinism
# torch.set_default_dtype(torch.float64)

def get_args():
	p = argparse.ArgumentParser()
	p.add_argument(
		"--max-n",
		type=int,
		default=25000,
		help="Maximum N (default: 25000)",
	)
	# Booleans default to True, with matching --no- flags to turn them off
	p.add_argument(
		"--log-scaled",
		dest="log_scaled",
		action="store_true",
	)
	p.add_argument(
		"--simplify",
		dest="simplify",
		action="store_true",
		# default=True,
		help="Simplify output (default: True). Use --no-simplify to disable.",
	)
	# p.add_argument(
	# 	"--no-simplify",
	# 	dest="simplify",
	# 	action="store_false",
	# 	help=argparse.SUPPRESS,
	# )
	p.add_argument(
		"--mape-threshold",
		type=float,
		default=0.05,
		help="Stop searching for a solution only if MAPE < mape_threshold (default: 0.05%)",
	)
	return p.parse_args()

args = get_args()
print(args)
MAX_N = args.max_n
LOG_SCALED = args.log_scaled
SIMPLIFY = args.simplify
MAPE_THRESHOLD = args.mape_threshold

GAMMA = 0.5772156649015328606  # Euler–Mascheroni
EPS = 1e-8


# print('Predefined available symbols:', sorted(SYMBOLIC_LIB.keys()))

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.backends.cuda.matmul.allow_tf32 = True
# elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(device)

# --- Data: read primes ---
with open('primes.txt', "r") as f:
	prime_list = [int(line.strip()) for line in f if line.strip()][:MAX_N]

x = torch.tensor(prime_list, dtype=torch.float32).view(-1, 1)
y = torch.arange(1, len(prime_list)+1, dtype=torch.float32).view(-1, 1)

# # --- Scale to increase smoothness: log ---
if LOG_SCALED:
	x = torch.log(x)
	y = torch.log(y)

# --- Train/test split with shuffle ---
# N = x.shape[0]
# perm = torch.randperm(N)
# tr = int(0.8 * min(MAX_N,N))
# idx_tr, idx_te = perm[:tr], perm[tr:]
# dataset = {
# 	"train_input": x[idx_tr],
# 	"train_label": y[idx_tr],
# 	"test_input": x[idx_te],
# 	"test_label": y[idx_te],
# }
N = x.shape[0]
tr = int(0.8 * min(MAX_N,N))
dataset = {
	"train_input": x[:tr],
	"train_label": y[:tr],
	"test_input": x[tr:],
	"test_label": y[tr:],
}

########################################################################
# Mark inputs that are out-of-domain or (numerically) at the pole x = 1

# General: 1 / (log(x))^i
def f_invlog_pow(i):
    """
    Even i:    inside -> +y_th (flat cap)
    Odd  i:    inside -> (y_th / t_th) * t  (linear, sign-preserving)
    t = log(x),  t_th = (1 / y_th)^(1/i)
    """
    assert i >= 1
    is_even = (i % 2 == 0)

    def _f(x, y_th):
        tiny = torch.finfo(x.dtype).tiny
        t = torch.log(torch.clamp(x, min=tiny))               # t = log x
        y_pos = torch.clamp(y_th, min=tiny)
        # t_th = (1 / y_th)^(1/i)
        t_th = torch.pow(1.0 / y_pos, 1.0 / float(i))

        cond = torch.isfinite(t) & (torch.abs(t) < t_th)
        outside = 1.0 / torch.pow(t, i)                       # safe because |t| >= t_th on ~cond

        if is_even:
            inside = y_th                                     # flat cap to +y_th
        else:
            inside = (y_th / t_th) * t                        # linear ramp hitting ±y_th at ±t_th

        val = torch.where(cond, inside, outside)
        return t_th, val

    return _f

# x/(log x)^i — scaled version: match x*y_th at boundary
def f_x_invlog_pow(i: int):
    assert i >= 1
    is_even = (i % 2 == 0)
    def _f(x, y_th):
        tiny = torch.finfo(x.dtype).tiny
        t = torch.log(torch.clamp(x, min=tiny))
        y_pos = torch.clamp(y_th, min=tiny)
        t_th = torch.pow(1.0 / y_pos, 1.0 / float(i))
        cond = torch.isfinite(t) & (torch.abs(t) < t_th)
        outside = x / torch.pow(t, i)
        inside  = x * (y_th if is_even else (y_th / t_th) * t)
        return t_th, torch.where(cond, inside, outside)
    return _f

# 1 / (log(x) - c)
def f_invlog_shift(c):
    def _f(x, y_th):
        tiny = torch.finfo(x.dtype).tiny
        t = torch.log(torch.clamp(x, min=tiny)) - c
        t_th = 1.0 / torch.clamp(y_th, min=tiny)
        cond = torch.isfinite(t) & (torch.abs(t) < t_th)
        val = torch.where(cond, (y_th / t_th) * t, 1.0 / t)
        return t_th, val
    return _f

# x/(log x - c) (i = 1 shifted)
def f_x_invlog_shift(c: float):
    def _f(x, y_th):
        tiny = torch.finfo(x.dtype).tiny
        t = torch.log(torch.clamp(x, min=tiny)) - c
        t_th = 1.0 / torch.clamp(y_th, min=tiny)
        cond = torch.isfinite(t) & (torch.abs(t) < t_th)
        outside = x / t
        inside  = x * (y_th / t_th) * t
        return t_th, torch.where(cond, inside, outside)
    return _f

# 1 / (log(x) - 1 - 1/log(x))
def f_invlog_minus1_minus_invlog(x, y_th):
    tiny = torch.finfo(x.dtype).tiny
    t = torch.log(torch.clamp(x, min=tiny))
    inv_t = torch.where(torch.abs(t) > 0, 1.0 / t,
                        torch.sign(t) * (1.0 / tiny))
    z = t - 1.0 - inv_t
    z_th = 1.0 / torch.clamp(y_th, min=tiny)
    cond = torch.isfinite(z) & (torch.abs(z) < z_th)
    val = torch.where(cond, (y_th / z_th) * z, 1.0 / z)
    return z_th, val

# x/(log x - 1 - 1/log x) (i = 1 on z := t - 1 - 1/t)
def f_x_invlog_minus1_minus_invlog(x, y_th):
    tiny = torch.finfo(x.dtype).tiny
    t = torch.log(torch.clamp(x, min=tiny))
    inv_t = torch.where(torch.abs(t) > 0, 1.0 / t, torch.sign(t) * (1.0 / tiny))
    z = t - 1.0 - inv_t
    z_th = 1.0 / torch.clamp(y_th, min=tiny)
    cond = torch.isfinite(z) & (torch.abs(z) < z_th)
    outside = x / z
    inside  = x * (y_th / z_th) * z
    return z_th, torch.where(cond, inside, outside)

# ---------- helpers used below ----------
def _lambertw_principal(y, iters=6):
    """Principal Lambert W for y>=0 via Newton; vectorized and torch-safe."""
    tiny = torch.finfo(y.dtype).tiny
    y = torch.clamp(y, min=tiny)
    # piecewise init: good on (0,∞)
    w = torch.where(y < 1.0, y, torch.log(y) - torch.log(torch.clamp(torch.log(y), min=tiny)))
    for _ in range(iters):
        ew = torch.exp(w)
        f  = w * ew - y
        den = torch.clamp(ew * (w + 1.0), min=tiny)
        w = w - f / den
    return w

# ---------- log^i ----------
def f_log_pow(i):
    """
    For u = log x, cap when u <= -u_th with u_th = y_th^(1/i).
    Inside value = (-1)^i * y_th so that |log^i| ≤ y_th at boundary.
    """
    assert i >= 1
    def _f(x, y_th):
        tiny = torch.finfo(x.dtype).tiny
        u_th = torch.pow(torch.clamp(y_th, min=tiny), 1.0/float(i))
        x_th = torch.exp(-u_th)
        cond = x <= x_th  # includes x<=0 (domain guard)
        outside = torch.pow(_safe_log(x), i)   # defined since we use _safe_log
        sgn = -1.0 if (i % 2) else 1.0
        inside = sgn * y_th
        return x_th, torch.where(cond, inside, outside)
    return _f

# ---------- loglog^i ----------
def f_loglog_pow(i):
    """
    For t=log x, u=log t. Cap when u <= -u_th with u_th = y_th^(1/i),
    i.e. when t <= exp(-u_th). Inside value = (-1)^i * y_th.
    """
    assert i >= 1
    def _f(x, y_th):
        tiny = torch.finfo(x.dtype).tiny
        t = torch.log(torch.clamp(x, min=tiny))                 # t = log x
        u_th = torch.pow(torch.clamp(y_th, min=tiny), 1.0/float(i))
        t_th = torch.exp(-u_th)
        cond = t <= t_th
        outside = torch.pow(torch.log(torch.clamp(t, min=tiny)), i)  # (log log x)^i
        sgn = -1.0 if (i % 2) else 1.0
        inside = sgn * y_th
        return t_th, torch.where(cond, inside, outside)
    return _f

# ---------- loglog/log ----------
def f_loglog_over_log(x, y_th):
    """
    g(x) = log(log x)/log x. Near t=log x -> 0+,
    choose t_th by solving |log t_th| / t_th = y_th  =>  t_th = W(y_th)/y_th,
    where W is principal Lambert W. Inside value = -y_th (correct sign).
    """
    tiny = torch.finfo(x.dtype).tiny
    t = torch.log(torch.clamp(x, min=tiny))                     # t = log x
    y = torch.clamp(y_th, min=tiny)
    w = _lambertw_principal(y)                                  # W(y_th)
    t_th = torch.clamp(w / y, min=tiny)                         # threshold in t-space
    cond = t <= t_th
    tt = torch.clamp(t, min=tiny)
    outside = torch.log(tt) / tt                                # log t / t
    inside  = -y_th
    return t_th, torch.where(cond, inside, outside)

########################################################################

safe_lib = [
	'1/sqrt(x)', '1/x', '1/x^2', '1/x^3', '1/x^4', '1/x^5', 
	'arccos', 'arcsin', 'arctan', 'arctanh', 'cos', 'sin', 'tan', 'tanh',
	'exp', 'gaussian', 'log', 'sgn', 'sqrt',
	'0', 
	'abs', 'x', 'x^2', 'x^3', 'x^4', 'x^5', 
]

# Families: 1/log^k and x/log^k  (k = 2..5)
for i in range(1, 6):
    k = '1/log' + (f'^{i}' if i != 1 else '')
    add_symbolic(
        k,
        lambda x: _safe_recip(_safe_log(x)**i),
        c=i,
        fun_singularity=f_invlog_pow(i),
        sympy_fun=lambda x: 1/(sp.log(x)**i)
    )
    safe_lib.append(k)

for i in range(1, 6):
    k = 'x/log' + (f'^{i}' if i != 1 else '')
    add_symbolic(
        k,
        lambda x: x * _safe_recip(_safe_log(x)**i),
        c=i,
        fun_singularity=f_x_invlog_pow(i),
        sympy_fun=lambda x: x/(sp.log(x)**i)
    )
    safe_lib.append(k)

# loglog^i, i = 1..3  (add singularities)
for i in range(1, 4):
    k = 'loglog' + (f'^{i}' if i != 1 else '')
    add_symbolic(
        k,
        lambda x: _safe_log(_safe_log(x))**i,
        c=i,
        fun_singularity=f_loglog_pow(i),
        sympy_fun=lambda x: sp.log(sp.log(x))**i
    )
    safe_lib.append(k)

# log^i, i = 2..3  (add singularities)
for i in range(2, 3+1):
    k = 'log' + (f'^{i}' if i != 1 else '')
    add_symbolic(
        k,
        lambda x: _safe_log(x)**i,
        c=i,
        fun_singularity=f_log_pow(i),
        sympy_fun=lambda x: sp.log(x)**i
    )
    safe_lib.append(k)

# Ratio loglog/log — use custom handler that matches the boundary exactly
add_symbolic(
    'loglog/log',
    lambda x: _safe_log(_safe_log(x)) * _safe_recip(_safe_log(x)),
    c=3,
    fun_singularity=f_loglog_over_log,
    sympy_fun=lambda x: sp.log(sp.log(x))/sp.log(x)
)

# Shifted poles: use x-scaled handler where numerator is x
add_symbolic('1/(log-1)',
    lambda x: _safe_recip(_safe_log(x) - 1),
    c=2,
    fun_singularity=f_invlog_shift(1.0),
    sympy_fun=lambda x: 1/(sp.log(x) - 1)
)
add_symbolic('x/(log-1)',
    lambda x: x * _safe_recip(_safe_log(x) - 1),
    c=2,
    fun_singularity=f_x_invlog_shift(1.0),
    sympy_fun=lambda x: x/(sp.log(x) - 1)
)

add_symbolic('1/(log-1-1/log)',
    lambda x: _safe_recip(_safe_log(x) - 1 - (1/_safe_log(x))),
    c=3,
    fun_singularity=f_invlog_minus1_minus_invlog,
    sympy_fun=lambda x: 1/(sp.log(x) - 1 - (1/sp.log(x)))
)
add_symbolic('x/(log-1-1/log)',
    lambda x: x * _safe_recip(_safe_log(x) - 1 - (1/_safe_log(x))),
    c=3,
    fun_singularity=f_x_invlog_minus1_minus_invlog,       # <-- x-scaled version
    sympy_fun=lambda x: x/(sp.log(x) - 1 - (1/sp.log(x)))
)

safe_lib += [
	'loglog/log',
	'1/(log-1)', 'x/(log-1)',
	'1/(log-1-1/log)', 'x/(log-1-1/log)',
]

# --- Modulus operators -------------------------------------------------------
# Exact (non-smooth) modulus and smooth (differentiable) modulus via Fourier series.
# 'modulus' acts like x mod 1; KAN can learn any period/phase via affine transforms.
# 'modulus_smooth' is a C^∞ approximation (differentiable everywhere).

# def _modulus(x, p=1.0, phi=0.0, eps=EPS):
#     """Exact wrap: returns values in [phi, phi + p) for p>0."""
#     p_t  = torch.clamp(torch.as_tensor(p,  dtype=x.dtype, device=x.device).abs(), min=eps)
#     phi_t = torch.as_tensor(phi, dtype=x.dtype, device=x.device)
#     return torch.remainder(x - phi_t, p_t) + phi_t

def _frac_smooth(x, K=7):
    """Smooth fractional part ≈ x mod 1 using truncated Fourier series.
       Returns in (0,1). Larger K -> closer to true sawtooth; K=7 is a good default.
    """
    # Ensure broadcast over last dimension; keep original shape on return
    x_exp = x.unsqueeze(-1)            # [..., 1]
    k = torch.arange(1, K+1, device=x.device, dtype=x.dtype).view(1, -1)  # [1, K]
    terms = torch.sin(2*math.pi*k*x_exp) / (math.pi*k)  # [..., K]
    y = 0.5 - terms.sum(dim=-1)        # [...]
    return y

def _modulus_smooth(x, p=1.0, phi=0.0, K=7, eps=EPS):
    """Smooth wrap: p*frac_smooth((x-phi)/p) + phi."""
    p_t   = torch.clamp(torch.as_tensor(p,   dtype=x.dtype, device=x.device).abs(), min=eps)
    phi_t = torch.as_tensor(phi, dtype=x.dtype, device=x.device)
    z = (x - phi_t) / p_t
    return p_t * _frac_smooth(z, K=K) + phi_t

# ---- Symbolic registrations (unary x -> ...) --------------------------------
# Smooth modulus operator with period 1 (differentiable surrogate).
K_DEFAULT = 7
# Period-1 in its *argument t*. In x, the learnable period is 1/|a|.
add_symbolic(
    'modulus_smooth',
    lambda t: _modulus_smooth(t, p=1.0, phi=0.0, K=K_DEFAULT),  # smooth numeric path
    c=3,
    fun_singularity=None,
    sympy_fun=lambda t: sp.Mod(t, 1)  # exact symbolic Mod
)
safe_lib.append('modulus_smooth')


# ---- Exponential integral Ei(u) approximations ----
def _ei_series(u, K=40, eps=EPS):
	"""
	Near-0 series for Ei(u).
	Uses log|u| in the analytic form, guarded as log(max(|u|, eps)).
	"""
	s = torch.zeros_like(u)

	# Precompute 1/k and k! up to K
	k = torch.arange(1, K + 1, device=u.device, dtype=u.dtype)
	inv_k = k.reciprocal()
	fact = torch.cumprod(k, dim=0)  # k!
	# Start term = u^k with k=1 initially -> we accumulate inv_k/fact weighting
	term = u.clone()
	for i in range(K):
		s = s + term * (inv_k[i] / fact[i])
		term = term * u

	logabsu = _safe_log(torch.abs(u), eps)
	return torch.as_tensor(GAMMA, dtype=u.dtype, device=u.device) + logabsu + s


def _ei_asymp(u, M=8, eps=EPS):
	"""
	Asymptotic: Ei(u) ≈ e^u / u * (1 + 1/u + 2!/u^2 + ... + M!/u^M)
	Guard 1/u and powers thereof.
	"""
	invu = _safe_recip(u, eps)            # safe 1/u
	series = torch.ones_like(u)           # k=0 term
	fact = torch.ones_like(u)
	pow_invu = torch.ones_like(u)
	for k in range(1, M + 1):
		fact = fact * k                   # k!
		pow_invu = pow_invu * invu
		series = series + fact * pow_invu

	return torch.exp(u) * series * invu

def _expi_torch(u, eps=EPS):
	"""
	Piecewise Ei(u): series near 0, asymptotic away from 0.
	"""
	thr = 2.0 if u.dtype == torch.float32 else 4.0
	out = torch.empty_like(u)
	m = (torch.abs(u) <= thr)
	if m.any():
		out[m] = _ei_series(u[m], eps=eps)
	if (~m).any():
		# asymptotic is intended for large positive u; for large negative u it returns tiny (OK).
		out[~m] = _ei_asymp(u[~m], eps=eps)
	return out

# ---- Logarithmic integral li(x) and offset Li(x) ----

def li_torch(x, eps=EPS):
	"""
	Principal value li(x) = Ei(log x).
	Domain-handled with safe log; avoids NaNs for x<=0 by clamping at eps.
	"""
	return _expi_torch(_safe_log(x, eps), eps=eps)

_LI2_CACHE = {}
def _li2(dtype, device, eps):
	key = (dtype, device, eps)
	if key not in _LI2_CACHE:
		_LI2_CACHE[key] = li_torch(torch.tensor(2.0, dtype=dtype, device=device), eps)#.detach()
	return _LI2_CACHE[key]


# def Li_torch(x, eps=EPS):
# 	return li_torch(x, eps) - _li2(x.dtype, x.device, eps)

def f_li_singularity(x, y_th):
    """
    Near x=1, li(x) = Ei(log x) ~ γ + log|log x| + O(1).
    Choose t = log x. Boundary at |t| = e^{-y_th} so that log|t| = -y_th.
    Inside: cap to γ - y_th (matches the leading term at the boundary).
    """
    tiny = torch.finfo(x.dtype).tiny
    t = torch.log(torch.clamp(x, min=tiny))                     # t = log x
    y_pos = torch.clamp(y_th, min=tiny)
    t_th = torch.exp(-y_pos)                                    # |t| boundary
    cond = torch.abs(t) < t_th

    gamma_t = torch.as_tensor(GAMMA,            # Euler–Mascheroni
                              dtype=x.dtype, device=x.device)
    inside  = gamma_t - y_th
    outside = li_torch(x)

    return t_th, torch.where(cond, inside, outside)

# ---- Möbius cache for Riemann R(x) ----
_MU_CACHE = {}
def _mu_vec(K, dtype, device):
	key = (K, dtype, device)
	if key not in _MU_CACHE:
		mu_list = [int(sp.mobius(k)) for k in range(1, K + 1)]
		_MU_CACHE[key] = torch.tensor(mu_list, dtype=dtype, device=device)#.detach()
	return _MU_CACHE[key]


def riemann_R_torch(x, K=8, eps=EPS):
	"""
	R(x) ≈ sum_{k=1..K} μ(k)/k · li(x^{1/k})
	Uses x_pos = clamp(x, eps) to avoid negatives/zero; avoids NaNs in powers/logs.
	"""
	x_pos = torch.clamp(x, eps)  # ensure positive base for pow/log
	mu = _mu_vec(K, x.dtype, x.device)

	s = torch.zeros_like(x_pos)
	for k in range(1, K + 1):
		muk = int(mu[k - 1].item())  # tensor -> python int (safe for control flow)
		if muk == 0:
			continue
		term = li_torch(torch.pow(x_pos, 1.0 / k), eps)
		s = s + (muk / float(k)) * term

	return s

def f_R_singularity(K=8):
    """
    Cap R(x) near x=1 using the truncated leading asymptotic that matches
    the boundary at |log x| = e^{-y_th}.
    """
    def _f(x, y_th):
        tiny  = torch.finfo(x.dtype).tiny
        t     = torch.log(torch.clamp(x, min=tiny))             # t = log x
        y_pos = torch.clamp(y_th, min=tiny)
        t_th  = torch.exp(-y_pos)
        cond  = torch.abs(t) < t_th

        # Möbius weights
        mu = _mu_vec(K, x.dtype, x.device)                      # shape [K], dtype=x.dtype
        k  = torch.arange(1, K+1, device=x.device, dtype=x.dtype)
        w  = mu / k                                             # μ(k)/k

        S1 = w.sum()                                            # Σ μ(k)/k
        S2 = (-w * torch.log(k)).sum()                          # -Σ μ(k)/k * log k
        gamma_t = torch.as_tensor(GAMMA, dtype=x.dtype, device=x.device)

        inside  = S1 * (gamma_t - y_th) + S2                    # scalar-broadcast
        outside = riemann_R_torch(x, K=K)

        return t_th, torch.where(cond, inside, outside)
    return _f

# Riemann R, li, Li, and basics
add_symbolic('R', 
	lambda x: torch.where(torch.abs(_safe_log(x)) > EPS, riemann_R_torch(x), 1e12), 
	c=2, 
	fun_singularity=f_R_singularity(K=8)
)
add_symbolic('li', 
	lambda x: torch.where(torch.abs(_safe_log(x)) > EPS, li_torch(x), 1e12), 
	c=2, 
	fun_singularity=f_li_singularity
)
# add_symbolic('Li', lambda x: torch.where(torch.abs(_safe_log(x)) > EPS, Li_torch(x), 1e12), c=4, fun_singularity=f_invlog)

safe_lib += [
	'R',
	'li', 
	# 'Li',
]


print('Available symbols:', safe_lib)
# print('Predefined available symbols:', sorted(SYMBOLIC_LIB.keys()))
################################################################

def print_results(model, dataset, transform_fn=None, k=50):
	with torch.no_grad():
		y_hat = model(dataset['test_input']).detach().cpu()
		y = dataset['test_label'].detach().cpu()
		if transform_fn:
			y = transform_fn(y)
			y_hat = transform_fn(y_hat)
		y_hat = torch.round(y_hat)
		y = torch.round(y)
		mse = torch.mean((y_hat - y)**2).item()
		mape = 100*(torch.mean(torch.abs((y_hat - y) / y))).item()
		print(f"Test MSE: {mse:.6g} | MAPE: {mape:.3f}%")
		# show a few label/prediction pairs nicely
		pairs_preview = list(zip(y[:k].tolist(), y_hat[:k].tolist()))
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
	model = KAN(width=[1, 64, 1], grid=5, k=3, seed=i)
	model.fit(dataset, 
		opt="Adam", 
		lr=1e-3 if LOG_SCALED else 1e-1, 
		steps=1000 if LOG_SCALED else 5000, 
		lamb=1e-2 if LOG_SCALED else 1, 
		lamb_entropy=10.
	)
	model = model.prune(node_th=1e-2, edge_th=0)
	model.fit(dataset, 
		opt="Adam", 
		lr=1e-3 if LOG_SCALED else 1e-2, 
		steps=10000 if LOG_SCALED else 20000, 
		lamb=0 if LOG_SCALED else 3e-1
	)
	# model = model.prune()
	# model.fit(dataset, opt="Adam", lr=1e-3, steps=500, lamb=0)

	print_results(model, dataset, transform_fn=torch.exp if LOG_SCALED else None, k=50)

	# model.auto_symbolic(lib=safe_lib, weight_simple=0)
	summary = model.auto_symbolic_robust_greedy(
		dataset,       # evaluation set
		lib=safe_lib,
		min_edge_score=None,         # or e.g. 1e-3 to stop earlier
		mode="backward",             # or "ols"
		weight_simple=0,
		# verbose=1,
		lr=1e-3,
		steps=200,
		lamb=0,
		min_r2=0.9
	)
	print('auto_symbolic_robust_greedy:', summary)
	

	symbolic_formula = model.symbolic_formula(simplify=SIMPLIFY)
	if symbolic_formula:
		symbolic_formula = symbolic_formula[0][0]
		print('Symbolic formula:', re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', str(symbolic_formula)))

	model.fit(dataset, 
		opt="Adam", 
		lr=1e-3, 
		steps=10000, # if LOG_SCALED else 10000,  
		lamb=0
	)

	# model.fit(dataset, opt="Adam", lr=1e-4, steps=250, lamb=1, lamb_entropy=1e-1)
	symbolic_formula = model.symbolic_formula(simplify=SIMPLIFY)
	if symbolic_formula:
		symbolic_formula = symbolic_formula[0][0]
		print('Symbolic formula:', re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', str(symbolic_formula)))
		if str(symbolic_formula) == 'nan':
			symbolic_formula = None
	# else:
	# 	model.unfix_symbolic_all()

	mape = print_results(model, dataset, transform_fn=torch.exp if LOG_SCALED else None, k=50)
	i+=1
