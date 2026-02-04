import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .utils import SYMBOLIC_LIB
from typing import *

class BSplineBasisFunction(nn.Module):
	"""
	Clamped uniform B-spline basis (Cox–de Boor), robust for training.

	Fix:
	- If train_grid=True, degree-0 bases must be differentiable w.r.t. knots.
	  Hard comparisons kill knot gradients.
	- Avoid .item() in clamp.
	"""

	def __init__(
		self,
		grid_min: float = -2.0,
		grid_max: float = 2.0,
		num_grids: int = 8,
		degree: int = 3,
		train_grid: bool = True,
		min_step: float = 1e-3,
		soft_box_steepness: float = 80.0,  # only used when train_grid=True
	):
		super().__init__()
		assert num_grids >= degree + 1, "num_grids must be >= degree + 1"
		assert grid_max > grid_min

		self.num_grids = int(num_grids)
		self.degree = int(degree)
		self.train_grid = bool(train_grid)
		self.min_step = float(min_step)
		self.soft_box_steepness = float(soft_box_steepness)

		p = self.degree
		G = self.num_grids
		n_interior = G - p + 1

		interior = torch.linspace(grid_min, grid_max, n_interior)  # includes endpoints
		left_pad  = interior[0].repeat(p)
		right_pad = interior[-1].repeat(p)
		knots0 = torch.cat([left_pad, interior, right_pad], dim=0)  # length = G + p + 1

		if not train_grid:
			self.register_buffer("knots_fixed", knots0)
			self.knot_base = None
			self.knot_deltas = None
		else:
			self.knot_base = nn.Parameter(knots0[:1].clone())  # scalar start
			d0 = knots0[1:] - knots0[:-1]
			inv = torch.log(torch.expm1(torch.clamp(d0 - self.min_step, min=1e-6)))
			self.knot_deltas = nn.Parameter(inv)

	def _knots(self) -> torch.Tensor:
		if not self.train_grid:
			return self.knots_fixed
		steps = F.softplus(self.knot_deltas) + self.min_step
		return torch.cat([self.knot_base, self.knot_base + torch.cumsum(steps, dim=0)], dim=0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: [..., I]
		return: [..., I, G]
		"""
		t = self._knots().to(dtype=x.dtype, device=x.device)
		p = self.degree
		G = self.num_grids

		# Differentiable clamp bounds (no .item())
		low = t[p]
		high = t[-p-1]
		xc = x.clamp(min=low, max=high)
		x_e = xc[..., None]  # [..., I, 1]

		# --- Degree-0 bases ---
		# If knots are trainable, use a differentiable "soft box":
		#   1_{[t_i, t_{i+1})}(x) ≈ sigmoid(k(x - t_i)) - sigmoid(k(x - t_{i+1}))
		if self.train_grid:
			k = self.soft_box_steepness
			left = torch.sigmoid(k * (x_e - t[:-1]))   # [..., I, T-1]
			right = torch.sigmoid(k * (x_e - t[1:]))   # [..., I, T-1]
			N = (left - right).clamp_min(0.0)
		else:
			N = ((x_e >= t[:-1]) & (x_e < t[1:])).to(x.dtype)
			# include right boundary at very end
			N[..., -1] = torch.maximum(N[..., -1], (x_e[..., 0] == t[-1]).to(x.dtype))

		# --- Cox–de Boor recursion ---
		for d in range(1, p + 1):
			m_prev = N.shape[-1]
			m_new = m_prev - 1

			t_i     = t[:m_new]
			t_i_d   = t[d:d + m_new]
			t_i1    = t[1:m_new + 1]
			t_i_d1  = t[d + 1:d + 1 + m_new]

			den1 = (t_i_d - t_i)
			den2 = (t_i_d1 - t_i1)

			N_left  = N[..., :m_new]
			N_right = N[..., 1:m_new + 1]

			den1_b = den1.view(*([1] * (x_e.ndim - 1)), -1)
			den2_b = den2.view(*([1] * (x_e.ndim - 1)), -1)

			w1 = torch.where(
				den1_b != 0,
				(x_e - t_i) / den1_b,
				torch.zeros_like(x_e),
			)
			w2 = torch.where(
				den2_b != 0,
				(t_i_d1 - x_e) / den2_b,
				torch.zeros_like(x_e),
			)

			N = w1 * N_left + w2 * N_right

		return N[..., :G]



class SplineLinear(nn.Linear):
	def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
		self.init_scale = init_scale
		super().__init__(in_features, out_features, bias=False, **kw)

	def reset_parameters(self) -> None:
		nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class StepBasisFunction(nn.Module):
	def __init__(
		self,
		grid_min: float = -2.,
		grid_max: float = 2.,
		num_grids: int = 8,
		steepness: float = 20.,      # higher -> sharper step
		train_grid: bool = True,
		train_steepness: bool = True,
	):
		super().__init__()

		grid = torch.linspace(grid_min, grid_max, num_grids)
		self.grid = nn.Parameter(grid, requires_grad=train_grid)

		self.log_k = nn.Parameter(
			torch.tensor(float(steepness)).log(),
			requires_grad=train_steepness,
		)

	@property
	def k(self):
		return self.log_k.exp()

	def forward(self, x):
		# x: [..., I] -> [..., I, G]
		# basis_j(x) = sigmoid(k * (x - c_j))
		return torch.sigmoid(self.k * (x[..., None] - self.grid))

class RadialBasisFunction(nn.Module):
	def __init__(
		self,
		grid_min: float = -2.,
		grid_max: float = 2.,
		num_grids: int = 8,
		denominator: float = None,      # larger -> smoother
		train_grid: bool = True,
		train_denominator: bool = True
	):
		super().__init__()

		# learnable grid
		grid = torch.linspace(grid_min, grid_max, num_grids)
		self.grid = nn.Parameter(grid, requires_grad=train_grid)

		# learnable (positive) denominator, in log-space
		if denominator is None:
			denominator = (grid_max - grid_min) / (num_grids - 1)

		denom_tensor = torch.tensor(float(denominator), dtype=torch.float32)
		self.log_denominator = nn.Parameter(
			denom_tensor.log(),
			requires_grad=train_denominator
		)

	@property
	def denominator(self) -> torch.Tensor:
		# always positive
		return self.log_denominator.exp()

	def forward(self, x):
		# x: [..., I] -> [..., I, G]
		den = self.denominator
		return torch.exp(-((x[..., None] - self.grid) / den) ** 2)

class RationalBasisFunction(nn.Module):
	"""
	Rational basis used by RationalKANLayer.

	For each basis index b we learn two polynomials:
	  P_b(x) = sum_{k=0}^{deg_num}   alpha[b,k] * x^k
	  Q_b(x) = sum_{k=0}^{deg_den}   beta[b,k]  * x^k

	and define the scalar basis function
	  phi_b(x) = P_b(x) / (1 + |Q_b(x)|)

	This is evaluated element-wise on x and returns [..., I, num_bases].
	"""

	def __init__(
		self,
		num_bases: int = 8,
		degree_numerator: int = 3,
		degree_denominator: int = 2,
		**args
	):
		super().__init__()
		# num_bases = num_grids
		self.grid = nn.Parameter(
			torch.linspace(-1., 1., num_bases),
			requires_grad=False,
		)

		assert num_bases > 0
		assert degree_numerator >= 0
		assert degree_denominator >= 0

		self.num_bases = num_bases
		self.deg_num = degree_numerator
		self.deg_den = degree_denominator

		# coefficients for P_b and Q_b
		# shapes: [B, deg+1], where B = num_bases
		self.alpha = nn.Parameter(torch.empty(num_bases, degree_numerator + 1))
		self.beta  = nn.Parameter(torch.empty(num_bases, degree_denominator + 1))

		self.reset_parameters()

	def reset_parameters(self) -> None:
		# Small initialization so activations stay in a reasonable range
		nn.init.trunc_normal_(self.alpha, mean=0.0, std=0.2)
		nn.init.trunc_normal_(self.beta,  mean=0.0, std=0.2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: [..., I]
		return: [..., I, num_bases]
		"""
		x = x.to(self.alpha.dtype)

		# powers: x^0, x^1, ..., x^K
		max_deg = max(self.deg_num, self.deg_den)
		powers = [torch.ones_like(x)]
		for _ in range(1, max_deg + 1):
			powers.append(powers[-1] * x)
		x_powers = torch.stack(powers, dim=-1)      # [..., I, K+1]

		# numerator P_b(x)
		num_terms = x_powers[..., : self.deg_num + 1]              # [..., I, deg_num+1]
		num = torch.einsum('...id,bd->...ib', num_terms, self.alpha)  # [..., I, B]

		# denominator 1 + |Q_b(x)|
		den_terms = x_powers[..., : self.deg_den + 1]
		den_poly = torch.einsum('...id,bd->...ib', den_terms, self.beta)  # [..., I, B]
		den = 1.0 + den_poly.abs()

		return num / den

NUMERIC_ATOM_FNS = {
	"step_bf": StepBasisFunction,
	"radial_bf": RadialBasisFunction,
	"rational_bf": RationalBasisFunction,
	"bspline": BSplineBasisFunction,
}

def masked_softmax_stable(logits, mask, dim=-1, eps=1e-8):
	# logits: [..., K], mask: same shape (bool)
	mask = mask.to(torch.bool)

	# detect rows where everything is masked
	all_masked = (~mask).all(dim=dim, keepdim=True)

	# set invalid to -inf, but make "all-masked" rows finite before softmax
	logits = logits.masked_fill(~mask, float("-inf"))
	logits = logits.masked_fill(all_masked, 0.0)

	# do softmax in fp32 for stability, then cast back
	probs = F.softmax(logits, dim=dim, dtype=torch.float32).to(logits.dtype)

	# zero out invalid and all-masked rows, then renormalize
	probs = probs.masked_fill(~mask, 0.0)
	denom = probs.sum(dim=dim, keepdim=True)
	probs = torch.where(denom > 0, probs / (denom + eps), torch.zeros_like(probs))
	return probs

class GatedSymbolicLayer(nn.Module):
	"""
	Symbolic + numeric layer with differentiable gating over SYMBOLIC_LIB atoms
	plus optional numeric atoms (step basis, RBF, ...).

	For symbolic atoms k:
		phi_{j,i,k}(x) = d_{j,i,k} + c_{j,i,k} * f_k(a_{j,i,k} * x + b_{j,i,k})

	For numeric atoms k:
		phi_{j,i,k}(x) = g_{j,i,k}(x_i)
	where g_{j,i,k} is an independent numeric module per edge (j,i)
	(no a,b,c,d wrapper).
	"""

	# names of built-in numeric atom *types* we know how to construct
	numeric_layers = list(NUMERIC_ATOM_FNS.keys())

	def __init__(
		self,
		input_dim,
		output_dim,
		atom_names,
		*,
		init_atom_bias = 0.0,
		symbolic_scale = 1.0,
		use_base_update = True,
		base_activation = F.silu,
		# fn_name -> kwargs, e.g. {"step_bf": {...}, "rbf": {...}}
		numeric_atom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
		**args,
	):
		super().__init__()

		self.in_dim = input_dim
		self.out_dim = output_dim
		self.base_atom_names = list(atom_names)          # atoms from SYMBOLIC_LIB
		self.symbolic_scale = symbolic_scale
		self.init_atom_bias = init_atom_bias

		self.use_base_update = use_base_update
		if use_base_update:
			self.base_activation = base_activation
			# bias=False so we can decompose per edge like FastKANLayer
			self.base_linear = nn.Linear(input_dim, output_dim, bias=False)

		# ---- numeric atoms configuration ----
		if numeric_atom_configs is not None:
			self._numeric_atom_configs = copy.deepcopy(numeric_atom_configs)
		else:
			self._numeric_atom_configs = {}

		self.numeric_atoms = nn.ModuleDict()
		self.numeric_coefs = nn.ParameterDict()

		numeric_names = []
		for fn_name, cfg in self._numeric_atom_configs.items():
			cfg = cfg.copy()
			basis_cls = NUMERIC_ATOM_FNS[fn_name]
			basis = basis_cls(**cfg)                     # e.g. StepBasisFunction

			self.numeric_atoms[fn_name] = basis          # one per atom type, not per edge

			# Determine grid size G from the basis output
			with torch.no_grad():
				dummy = torch.zeros(1, self.in_dim, device=self.gate_logits.device if hasattr(self, "gate_logits") else "cpu")
				G = basis(dummy).shape[-1]               # basis(dummy): [1, in_dim, G]

			# Per-edge coefficients: [out_dim, in_dim, G]
			coef = nn.Parameter(torch.empty(self.out_dim, self.in_dim, G))
			# Reasonable init; you can tweak scale if you want
			nn.init.trunc_normal_(coef, mean=0.0, std=0.1)
			self.numeric_coefs[fn_name] = coef

			numeric_names.append(fn_name)

		# Final atom name list (symbolic + numeric)
		self.atom_names = list(self.base_atom_names) + numeric_names
		self.num_atoms = len(self.atom_names)

		# simple LN over inputs for "preacts"
		self.layernorm = nn.LayerNorm(input_dim)

		# mask [in_dim, out_dim] in KAN convention
		self.mask = nn.Parameter(
			torch.ones(input_dim, output_dim),
			requires_grad=False,
		)

		# gate logits: [out_dim, in_dim, K]
		self.gate_logits = nn.Parameter(
			torch.zeros(self.out_dim, self.in_dim, self.num_atoms)
		)
		with torch.no_grad():
			self.gate_logits += init_atom_bias  # typically 0

		# mask over atoms per edge; 1 = active, 0 = pruned
		self.register_buffer(
			"gate_mask",
			torch.ones(self.out_dim, self.in_dim, self.num_atoms)
		)
		self._gate_mask_hook = None

		# per-edge per-atom affine params: [out_dim, in_dim, K, 4] = (a,b,c,d)
		if self.num_atoms > 0:
			self.affine = nn.Parameter(
				torch.zeros(self.out_dim, self.in_dim, self.num_atoms, 4)
			)
			with torch.no_grad():
				# default = identity-ish: a=1, b=0, c=1, d=0
				self.affine[..., 0] = 1.0  # a
				self.affine[..., 1] = 0.0  # b
				self.affine[..., 2] = 1.0  # c
				self.affine[..., 3] = 0.0  # d
		else:
			self.affine = None

		# --- fake KAN-style meta fields so MultKAN doesn't break ---
		self.out_dim_sum = output_dim
		self.out_dim_mult = 0

		# dummy "grid" & k so things like refine()/plot don't crash
		self.grid = nn.Parameter(
			torch.linspace(-1.0, 1.0, 2),
			requires_grad=False
		)
		self.k = 3

		# scale_* used in their reg() / plotting
		self.scale_base = nn.Parameter(torch.ones(output_dim))
		self.scale_sp   = nn.Parameter(torch.ones(output_dim))

		# dummy spline coeffs, just to satisfy coef-based regularizer
		self._dummy_coef = nn.Parameter(
			torch.zeros(self.out_dim * self.in_dim, 1),
			requires_grad=False
		)
		self.comp_log_s = torch.nn.Parameter(torch.zeros(self.out_dim, self.in_dim, self.num_atoms))


	# ------------------------------------------------------------------
	# KAN-ish API bits
	# ------------------------------------------------------------------
	@property
	def coef(self) -> torch.Tensor:
		"""Dummy coefficient tensor just so reg() doesn't blow up."""
		return self._dummy_coef

	@torch.no_grad()
	def update_grid_from_samples(self, acts: torch.Tensor):
		return

	@torch.no_grad()
	def initialize_grid_from_parent(self, parent_layer, parent_acts: torch.Tensor):
		return

	# ------------------------------------------------------------------
	# core symbolic computation
	# ------------------------------------------------------------------
	def _symbolic_vals(self, pre, time_benchmark=False):
		"""
		pre: [B, in_dim]

		Returns:
		  sym_vals: [B, out_dim, in_dim, num_atoms]
			= phi_{j,i,k}(pre[:,i]) for all edges and atoms.
		"""
		B, I = pre.shape
		O = self.out_dim
		K = self.num_atoms
		device = pre.device

		if K == 0:
			return torch.zeros(B, O, I, 0, device=device)

		# for symbolic atoms: pre -> a,b,c,d wrapper
		pre_exp = pre[:, None, :, None]   # [B,1,I,1]

		a = self.affine[..., 0]  # [O,I,K]
		b = self.affine[..., 1]
		c = self.affine[..., 2]
		d = self.affine[..., 3]

		a_b = a.unsqueeze(0)     # [1,O,I,K]
		b_b = b.unsqueeze(0)
		c_b = c.unsqueeze(0)
		d_b = d.unsqueeze(0)

		arg = a_b * pre_exp + b_b       # [B,O,I,K]
		sym_vals = torch.zeros(B, O, I, K, device=device)

		# ---------------- SYMBOLIC ATOMS ----------------
		num_sym = len(self.base_atom_names)

		for k_idx, name in enumerate(self.base_atom_names):
			arg_k = arg[..., k_idx]             # [B,O,I]
			torch_fun = SYMBOLIC_LIB[name][0]

			v = torch_fun(arg_k)            # broadcasting

			# ensure [B,O,I]
			while v.ndim < 3:
				v = v.unsqueeze(-1)
			v = v.reshape(B, O, I)

			c_k = c_b[..., k_idx]       # [1,O,I]
			d_k = d_b[..., k_idx]

			sv = torch.nan_to_num(d_k + c_k * v, nan=0.0, posinf=1e5, neginf=-1e5)
			sv = sv.clamp(-1e5, 1e5)
			sym_vals[..., k_idx] = sv

		# ---------------- NUMERIC ATOMS ----------------
		if not time_benchmark:
			pre = self.layernorm(pre)   # [B,I]

		offset = num_sym
		for k, name in enumerate(self._numeric_atom_configs.keys()):
			# shared basis for this numeric atom type
			basis = self.numeric_atoms[name](pre)     # [B, in_dim, G]
			coef  = self.numeric_coefs[name]          # [out_dim, in_dim, G]

			# vals[b, o, i] = sum_g basis[b, i, g] * coef[o, i, g]
			atom_vals = torch.einsum("big,oig->boi", basis, coef)

			sym_vals[:, :, :, offset + k] = atom_vals

		sym_vals = self.symbolic_scale * sym_vals
		return sym_vals   # [B,O,I,K]



	# ------------------------------------------------------------------
	# forward: KAN-compatible signature
	# ------------------------------------------------------------------
	def forward(self, x, time_benchmark = True, temperature = None):
		"""
		x: [B, in_dim]

		Returns:
		  x_out            : [B, out_dim]
		  preacts          : [B, out_dim, in_dim]
		  postacts_mixed   : [B, out_dim, in_dim]  (gated per-edge output)
		  postspline_dummy : [B, out_dim, in_dim]  (same as postacts_mixed)
		"""
		if temperature is None:
			temperature = 1.0

		pre = x

		B, I = pre.shape
		O = self.out_dim
		K = self.num_atoms
		device = x.device

		# preacts: broadcast pre per output
		preacts = pre.unsqueeze(1).expand(B, O, I)    # [B,O,I]

		# 1) symbolic + numeric candidates per-edge
		sym_vals = self._symbolic_vals(pre, time_benchmark=time_benchmark)          # [B,O,I,K]

		def compress_asinh_scaled(x, log_s):
			s = torch.exp(log_s).clamp_min(1e-6)   # [O,I,K]
			s = s.unsqueeze(0)                     # [1,O,I,K]
			return s * torch.asinh(x / s)
		sym_vals = compress_asinh_scaled(sym_vals, self.comp_log_s)

		# 2) gating over atoms
		if K == 0:
			edge_out = torch.zeros(B, O, I, device=device)
		else:
			logits = self.gate_logits / float(max(temperature, 1e-3))   # also avoid tiny temp
			mask = self.gate_mask.bool()
			probs = masked_softmax_stable(logits, mask, dim=-1)         # [O,I,K]
			probs = probs.unsqueeze(0)                                  # [1,O,I,K]
			edge_out = (sym_vals * probs).sum(dim=-1)                   # [B,O,I]

		# 2b) optional base_update like FastKANLayer
		if self.use_base_update:
			base_hidden = self.base_activation(x)           # [B,I]
			W_base = self.base_linear.weight               # [O,I]
			base_per_edge = torch.einsum("bi,oi->boi", base_hidden, W_base)
			edge_out = edge_out + base_per_edge            # [B,O,I]

		# 3) apply pruning mask [I,O] -> [O,I]
		edge_out = edge_out * self.mask.T.unsqueeze(0)       # [B,O,I]

		x_out = edge_out.sum(dim=-1)                         # [B,O]

		# For KAN compatibility, we return edge_out twice
		postspline = edge_out

		return x_out, preacts, edge_out, postspline

	@torch.no_grad()
	def prune_gates_topk(
		self,
		k: int,
	):
		"""
		For each edge (i->j), keep only top-k atoms (by current logits) and
		prune the rest by setting gate_mask=0 for them.

		"""

		O, I, K = self.gate_logits.shape
		device = self.gate_logits.device

		# build index sets once
		symbolic_idxs = list(range(K))

		if len(symbolic_idxs) <= k:
			# nothing to prune on symbolic side
			return

		mask = self.gate_mask.clone()

		logits = self.gate_logits.detach()

		for j in range(O):
			for i in range(I):
				# if that edge is pruned at all, skip
				if self.mask[i, j] <= 0:
					continue

				# scores only on symbolic atoms
				scores = logits[j, i, symbolic_idxs]  # [S]

				# if all zero/very small, we can still pick top-k
				k_eff = min(k, scores.shape[0])
				topk = torch.topk(scores, k_eff, dim=-1).indices  # indices into symbolic_idxs

				keep_sym = {symbolic_idxs[idx.item()] for idx in topk}

				# zero out all symbolic atoms except keep_sym
				for a_idx in symbolic_idxs:
					if a_idx not in keep_sym:
						mask[j, i, a_idx] = 0.0

		# update buffer
		self.gate_mask.copy_(mask)

		# (optional) mask gradients so pruned logits never move again
		if self._gate_mask_hook is not None:
			try:
				self._gate_mask_hook.remove()
			except Exception:
				pass

		def _mask_grad(grad):
			# grad: [O,I,K]
			return grad * self.gate_mask

		self._gate_mask_hook = self.gate_logits.register_hook(_mask_grad)


	# ------------------------------------------------------------------
	# utilities for pruning / swapping / reading choices
	# ------------------------------------------------------------------
	@torch.no_grad()
	def get_symbolic_choices(self):
		"""
		Returns:
		  dict[(i, j)] = best_atom_name
		"""
		choices = {}
		if self.num_atoms == 0:
			return choices

		best = self.gate_logits.argmax(dim=-1)  # [O,I]
		O, I = best.shape

		for j in range(O):
			for i in range(I):
				idx = int(best[j, i].item())
				name = self.atom_names[idx]
				choices[(i, j)] = name
		return choices

	def gating_regularizer(
		self,
		entropy_weight = 1e-3,
		l1_weight = 0.0,
	):
		"""
		Encourage gates to become somewhat sharp + keep logits bounded.

		R = entropy_weight * mean_edge_entropy + l1_weight * mean(|logits|)
		"""
		if self.num_atoms == 0:
			return torch.zeros((), device=self.gate_logits.device)

		logits = self.gate_logits      # [O,I,K]
		probs  = F.softmax(logits, dim=-1)

		eps = 1e-8
		ent = -(probs * (probs.clamp_min(eps).log())).sum(dim=-1)  # [O,I]
		mean_ent = ent.mean()

		l1 = logits.abs().mean()

		return entropy_weight * mean_ent + l1_weight * l1

	@torch.no_grad()
	def get_subset(self, in_ids, out_ids):
		"""
		For pruning: return a new layer restricted to selected inputs/outputs.
		"""
		in_ids  = torch.as_tensor(in_ids, dtype=torch.long)
		out_ids = torch.as_tensor(out_ids, dtype=torch.long)

		new = GatedSymbolicLayer(
			input_dim=in_ids.numel(),
			output_dim=out_ids.numel(),
			atom_names=self.base_atom_names,
			init_atom_bias=self.init_atom_bias,
			symbolic_scale=self.symbolic_scale,
			use_base_update=self.use_base_update,
			base_activation=self.base_activation,
			numeric_atom_configs=self._numeric_atom_configs,
		).to(self.mask.device)

		# base_update weights (if enabled)
		if self.use_base_update:
			new.base_linear.weight.copy_(self.base_linear.weight[out_ids][:, in_ids])

		# mask
		new.mask.copy_(self.mask[in_ids][:, out_ids])

		# layernorm
		new.layernorm.weight.copy_(self.layernorm.weight[in_ids])
		new.layernorm.bias.copy_(self.layernorm.bias[in_ids])

		# gate logits & affine
		if self.num_atoms > 0:
			new.gate_logits.copy_(self.gate_logits[out_ids][:, in_ids, :])
			new.affine.copy_(self.affine[out_ids][:, in_ids, :, :])
			new.gate_mask.copy_(self.gate_mask[out_ids][:, in_ids, :])

		# --- numeric atoms: shared basis + per-edge coefficients ---
		for name, basis in self.numeric_atoms.items():
			new_basis = new.numeric_atoms[name]
			new_basis.load_state_dict(basis.state_dict())

			coef     = self.numeric_coefs[name]        # [old_O, old_I, G]
			new_coef = new.numeric_coefs[name]         # [new_O, new_I, G]
			new_coef.data.copy_(coef.data[out_ids][:, in_ids, :])

		# meta
		new.out_dim_sum = out_ids.numel()
		new.out_dim_mult = 0

		return new

	@torch.no_grad()
	def swap(self, i1: int, i2: int, mode: str = 'in'):
		"""
		Swap neurons along input ('in') or output ('out') dimension for visualization.
		"""
		if mode not in ("in", "out"):
			raise ValueError("mode must be 'in' or 'out'")

		if mode == "in":
			# mask [I,O]
			self.mask[[i1, i2], :] = self.mask[[i2, i1], :]

			# gate & affine
			self.gate_logits[:, [i1, i2], :] = self.gate_logits[:, [i2, i1], :]
			if self.num_atoms > 0:
				self.affine[:, [i1, i2], :, :] = self.affine[:, [i2, i1], :, :]

			# gate_mask should follow the same permutation
			self.gate_mask[:, [i1, i2], :] = self.gate_mask[:, [i2, i1], :]

			# layernorm params
			lnw = self.layernorm.weight.data
			lnb = self.layernorm.bias.data
			lnw[[i1, i2]] = lnw[[i2, i1]]
			lnb[[i1, i2]] = lnb[[i2, i1]]

			# numeric coefs: swap along input index
			for name, coef in self.numeric_coefs.items():   # [O, I, G]
				coef.data[:, [i1, i2], :] = coef.data[:, [i2, i1], :]

			# base_update: swap input dimension
			if self.use_base_update:
				self.base_linear.weight[:, [i1, i2]] = self.base_linear.weight[:, [i2, i1]]

		else:  # mode == "out"
			self.mask[:, [i1, i2]] = self.mask[:, [i2, i1]]

			self.gate_logits[[i1, i2], :, :] = self.gate_logits[[i2, i1], :, :]
			if self.num_atoms > 0:
				self.affine[[i1, i2], :, :, :] = self.affine[[i2, i1], :, :, :]

			# gate_mask follow outputs as well
			self.gate_mask[[i1, i2], :, :] = self.gate_mask[[i2, i1], :, :]

			# keep scale_* in sync with outputs (for consistency)
			sb = self.scale_base.data
			sp = self.scale_sp.data
			sb[[i1, i2]] = sb[[i2, i1]]
			sp[[i1, i2]] = sp[[i2, i1]]

			# numeric coefs: swap along output index
			for name, coef in self.numeric_coefs.items():   # [O, I, G]
				coef.data[[i1, i2], :, :] = coef.data[[i2, i1], :, :]

			# base_update: swap output dimension
			if self.use_base_update:
				self.base_linear.weight[[i1, i2], :] = self.base_linear.weight[[i2, i1], :]
