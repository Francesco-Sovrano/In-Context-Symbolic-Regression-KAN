import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .utils import SYMBOLIC_LIB
from typing import *

class SplineLinear(nn.Linear):
	def __init__(self, in_features: int, out_features: int,
				 init_scale: float = 0.1, **kw) -> None:
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

class FastKANLayer(nn.Module):
	"""
	Drop-in numeric layer for MultKAN with:
	- RBF spline + optional base_update (MLP-like residual)
	- KAN-compatible API.
	"""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		grid_min: float = -2.,
		grid_max: float = 2.,
		num_grids: int = 5,
		use_base_update: bool = True,
		base_activation = F.silu,
		spline_weight_init_scale: float = 0.1,
		train_grid: bool = True,
	) -> None:
		super().__init__()

		self.in_dim = input_dim
		self.out_dim = output_dim
		self.num_grids = num_grids

		self.layernorm = nn.LayerNorm(input_dim)
		self.rbf = RadialBasisFunction(
			grid_min=grid_min,
			grid_max=grid_max,
			num_grids=num_grids,
			train_grid=train_grid,
			train_denominator=train_denominator,
			# train_steepness=train_steepness # optional
		)
		self.spline_linear = SplineLinear(
			input_dim * num_grids, output_dim,
			init_scale=spline_weight_init_scale
		)

		self.use_base_update = use_base_update
		if use_base_update:
			self.base_activation = base_activation
			# bias=False so we can decompose base term per edge
			self.base_linear = nn.Linear(input_dim, output_dim, bias=False)

		# mask in KAN convention: [in_dim, out_dim]
		self.mask = nn.Parameter(
			torch.ones(input_dim, output_dim),
			requires_grad=False
		)

		# meta fields used by MultKAN
		self.out_dim_sum = output_dim
		self.out_dim_mult = 0       # this layer itself has no mult slots
		self.k = 3                  # dummy; just to keep reg / plotting happy
		self.scale_base = nn.Parameter(torch.ones(output_dim))
		self.scale_sp   = nn.Parameter(torch.ones(output_dim))

	# ---------- properties used elsewhere ----------

	@property
	def grid(self):
		return self.rbf.grid

	@property
	def coef(self) -> torch.Tensor:
		"""
		For reg(): shape [num_edges, num_grids] = [out_dim * in_dim, num_grids]
		from spline weights only (no base_update).
		"""
		W = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
		return W.reshape(self.out_dim * self.in_dim, self.num_grids)

	# ---------- main forward (KAN-compatible) ----------

	def forward(self, x: torch.Tensor, time_benchmark: bool = True):
		"""
		x: [B, in_dim]

		Returns:
		  x_numerical        : [B, out_dim]
		  preacts            : [B, out_dim, in_dim]
		  postacts_numerical : [B, out_dim, in_dim]
		  postspline         : [B, out_dim, in_dim]
		"""

		if not time_benchmark:
			pre = self.layernorm(x)   # [B, I]
		else:
			pre = x                   # [B, I]

		basis = self.rbf(pre)        # [B, I, G]
		B = x.shape[0]

		# spline term: weight [O, I*G] -> [O, I, G]
		W_spline = self.spline_linear.weight.view(
			self.out_dim, self.in_dim, self.num_grids
		)                            # [O, I, G]
		postspline = torch.einsum('big,oig->boi', basis, W_spline)   # [B, O, I]

		# base_update as per original FastKAN, but expanded per-edge
		if self.use_base_update:
			base_hidden = self.base_activation(x)         # [B, I]
			W_base = self.base_linear.weight              # [O, I]
			base_per_edge = torch.einsum('bi,oi->boi', base_hidden, W_base)
			postspline = postspline + base_per_edge       # [B, O, I]

		# apply pruning mask: [I, O] -> [O, I] -> broadcast
		postspline = postspline * self.mask.T.unsqueeze(0)  # [B, O, I]

		x_numerical = postspline.sum(dim=-1)              # [B, O]
		postacts_numerical = postspline                   # [B, O, I]

		# preacts: same pre for each output
		preacts = pre.unsqueeze(1).expand(B, self.out_dim, self.in_dim)  # [B, O, I]

		return x_numerical, preacts, postacts_numerical, postspline

	# ---------- grid adaptation used by MultKAN.update_grid / refine ----------

	@torch.no_grad()
	def update_grid_from_samples(self, acts: torch.Tensor):
		"""
		acts: [B, in_dim] -> update self.rbf.grid using sample quantiles
		(still fine even if grid is learnable; this just overwrites it)
		"""
		if acts is None:
			return

		device = self.grid.device
		x = acts.to(device).reshape(-1)
		if x.numel() == 0:
			return

		x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

		if x.numel() < self.num_grids:
			gmin, gmax = x.min(), x.max()
			if gmin == gmax:
				gmin, gmax = gmin - 1., gmax + 1.
			new_grid = torch.linspace(gmin, gmax, self.num_grids, device=device)
		else:
			qs = torch.linspace(0., 1., self.num_grids, device=device)
			new_grid = torch.quantile(x, qs)

		self.rbf.grid.copy_(new_grid)

	@torch.no_grad()
	def initialize_grid_from_parent(self,
									parent_layer: "FastKANLayer",
									parent_acts: torch.Tensor):
		"""
		Used by MultKAN.initialize_grid_from_another_model(...)
		"""
		if hasattr(parent_layer, "rbf") and \
		   parent_layer.rbf.grid.shape == self.rbf.grid.shape:
			self.rbf.grid.copy_(parent_layer.rbf.grid)
		else:
			# Fallback: adapt from parent's activations
			self.update_grid_from_samples(parent_acts)

	# ---------- structural ops: used by prune / expand / swap ----------

	@torch.no_grad()
	def get_subset(self,
				   in_ids: torch.Tensor,
				   out_ids: torch.Tensor) -> "FastKANLayer":
		"""
		Return a *new* FastKANLayer restricted to given in/out indices.
		Shapes follow KAN's convention:
		  in_ids  over input_dim
		  out_ids over output_dim (subnodes before mult)
		"""
		in_ids  = torch.as_tensor(in_ids, dtype=torch.long)
		out_ids = torch.as_tensor(out_ids, dtype=torch.long)

		new = FastKANLayer(
			input_dim=in_ids.numel(),
			output_dim=out_ids.numel(),
			grid_min=float(self.grid.min().item()),
			grid_max=float(self.grid.max().item()),
			num_grids=self.num_grids,
			use_base_update=self.use_base_update,
			base_activation=getattr(self, "base_activation", F.silu),
			spline_weight_init_scale=self.spline_linear.init_scale,
		).to(self.mask.device)

		# copy grid
		new.rbf.grid.copy_(self.rbf.grid)

		# spline weights
		W_old = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
		W_new = new.spline_linear.weight.view(new.out_dim, new.in_dim, new.num_grids)
		W_new.copy_(W_old[out_ids][:, in_ids, :])
		new.spline_linear.weight.copy_(W_new.view(new.out_dim, new.in_dim * new.num_grids))

		# base weights
		if self.use_base_update:
			new.base_linear.weight.copy_(self.base_linear.weight[out_ids][:, in_ids])

		# mask
		new.mask.copy_(self.mask[in_ids][:, out_ids])

		# layernorm
		new.layernorm.weight.copy_(self.layernorm.weight[in_ids])
		new.layernorm.bias.copy_(self.layernorm.bias[in_ids])

		# "scale_*" regs follow outputs
		new.scale_base.copy_(self.scale_base[out_ids])
		new.scale_sp.copy_(self.scale_sp[out_ids])

		# out_dim_sum/mult will get overwritten by MultKAN after pruning
		new.out_dim_sum = out_ids.numel()
		new.out_dim_mult = 0

		return new

	@torch.no_grad()
	def swap(self, i1: int, i2: int, mode: str = 'in'):
		"""
		Swap neurons along input ('in') or output ('out') dimension.
		Used by MultKAN.swap / auto_swap.
		"""
		if mode not in ("in", "out"):
			raise ValueError("mode must be 'in' or 'out'")

		if mode == "in":
			# mask [I, O]
			self.mask[[i1, i2], :] = self.mask[[i2, i1], :]

			# spline weights: [O, I, G]
			W = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
			W[:, [i1, i2], :] = W[:, [i2, i1], :]
			self.spline_linear.weight.copy_(W.view(self.out_dim, self.in_dim * self.num_grids))

			# base weights: [O, I]
			if self.use_base_update:
				self.base_linear.weight[:, [i1, i2]] = \
					self.base_linear.weight[:, [i2, i1]]

			# layernorm params
			lnw = self.layernorm.weight.data
			lnb = self.layernorm.bias.data
			lnw[[i1, i2]] = lnw[[i2, i1]]
			lnb[[i1, i2]] = lnb[[i2, i1]]

		else:  # mode == 'out'
			# mask [I, O]
			self.mask[:, [i1, i2]] = self.mask[:, [i2, i1]]

			W = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
			W[[i1, i2], :, :] = W[[i2, i1], :, :]
			self.spline_linear.weight.copy_(W.view(self.out_dim, self.in_dim * self.num_grids))

			if self.use_base_update:
				self.base_linear.weight[[i1, i2], :] = \
					self.base_linear.weight[[i2, i1], :]

			# scale_base/scale_sp follow outputs
			sb = self.scale_base.data
			sp = self.scale_sp.data
			sb[[i1, i2]] = sb[[i2, i1]]
			sp[[i1, i2]] = sp[[i2, i1]]

class FastKAN(nn.Module):
	def __init__(
		self,
		layers_hidden: List[int],
		grid_min: float = -2.,
		grid_max: float = 2.,
		num_grids: int = 8,
		use_base_update: bool = True,
		base_activation = F.silu,
		spline_weight_init_scale: float = 0.1,
	) -> None:
		super().__init__()
		self.layers = nn.ModuleList([
			FastKANLayer(
				in_dim, out_dim,
				grid_min=grid_min,
				grid_max=grid_max,
				num_grids=num_grids,
				use_base_update=use_base_update,
				base_activation=base_activation,
				spline_weight_init_scale=spline_weight_init_scale,
			) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
		])

	def forward(self, x):
		for layer in self.layers:
			x, *_ = layer(x)   # keep only x_numerical
		return x

NUMERIC_ATOM_FNS = {
	"step_bf": StepBasisFunction,
	"radial_bf": RadialBasisFunction,
	"rational_bf": RationalBasisFunction,
}

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
		input_dim: int,
		output_dim: int,
		atom_names: List[str],
		*,
		init_atom_bias: float = 0.0,
		symbolic_scale: float = 1.0,
		# fn_name -> kwargs, e.g. {"step_bf": {...}, "rbf": {...}}
		numeric_atom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
		**args,
	):
		super().__init__()

		self.in_dim = input_dim
		self.out_dim = output_dim
		self.base_atom_names = list(atom_names)          # atoms from SYMBOLIC_LIB
		self.symbolic_scale = symbolic_scale

		# ---- numeric atoms configuration ----
		if numeric_atom_configs is not None:
			self._numeric_atom_configs = copy.deepcopy(numeric_atom_configs)
		else:
			self._numeric_atom_configs = {}

		# For each numeric fn_name we create:
		#   - one shared basis module (self.numeric_atoms[fn_name])
		#   - per-edge coefficients in self.numeric_coefs[fn_name] with shape [out_dim, in_dim, G]
		self.numeric_atoms = nn.ModuleDict()
		# fn_name -> per-edge coefficients over the basis grid
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
		# these are ONLY USED for symbolic atoms, ignored for numeric ones
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
	def _symbolic_vals(self, pre: torch.Tensor) -> torch.Tensor:
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

			# crude domain fixes for logs/sqrts
			if "log" in name.lower():
				arg_local = arg_k.abs() + 1e-3
			elif "sqrt" in name.lower():
				arg_local = arg_k.abs()
			else:
				arg_local = arg_k

			v = torch_fun(arg_local)            # broadcasting

			# ensure [B,O,I]
			while v.ndim < 3:
				v = v.unsqueeze(-1)
			v = v.reshape(B, O, I)

			v = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)

			c_k = c_b[..., k_idx]       # [1,O,I]
			d_k = d_b[..., k_idx]

			sym_vals[..., k_idx] = d_k + c_k * v

		# ---------------- NUMERIC ATOMS ----------------
		offset = num_sym
		for k, name in enumerate(self._numeric_atom_configs.keys()):
			# shared basis for this numeric atom type
			basis = self.numeric_atoms[name](pre)     # [B, in_dim, G]
			coef  = self.numeric_coefs[name]          # [out_dim, in_dim, G]

			# vals[b, o, i] = sum_g basis[b, i, g] * coef[o, i, g]
			atom_vals = torch.einsum("big,oig->boi", basis, coef)

			sym_vals[:, :, :, offset + k] = atom_vals

		sym_vals = torch.nan_to_num(sym_vals, nan=0.0, posinf=1e3, neginf=-1e3)
		sym_vals = self.symbolic_scale * sym_vals
		return sym_vals   # [B,O,I,K]


	# ------------------------------------------------------------------
	# forward: KAN-compatible signature
	# ------------------------------------------------------------------
	def forward(
		self,
		x: torch.Tensor,
		time_benchmark: bool = True,
		temperature: Optional[float] = None,
	):
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

		if not time_benchmark:
			pre = self.layernorm(x)   # [B,I]
		else:
			pre = x                   # [B,I]

		B, I = pre.shape
		O = self.out_dim
		K = self.num_atoms
		device = x.device

		# preacts: broadcast pre per output
		preacts = pre.unsqueeze(1).expand(B, O, I)    # [B,O,I]

		# 1) symbolic + numeric candidates per-edge
		sym_vals = self._symbolic_vals(pre)          # [B,O,I,K]

		# 2) gating over atoms
		if K == 0:
			edge_out = torch.zeros(B, O, I, device=device)
		else:
			logits = self.gate_logits / float(temperature)   # [O,I,K]
			# apply pruning mask – disabled atoms get -inf logits
			masked_logits = logits.masked_fill(self.gate_mask == 0, float('-inf'))

			probs  = F.softmax(masked_logits, dim=-1).unsqueeze(0)  # [1,O,I,K]
			edge_out = (sym_vals * probs).sum(dim=-1)               # [B,O,I]

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
		symbolic_only: bool = True,
	):
		"""
		For each edge (i->j), keep only top-k atoms (by current logits) and
		prune the rest by setting gate_mask=0 for them.

		If symbolic_only=True, only consider atoms from base_atom_names
		(i.e. SYMBOLIC_LIB) and do not count numeric atoms (e.g. step_bf/rbf)
		towards the top-k.
		"""

		O, I, K = self.gate_logits.shape
		device = self.gate_logits.device

		# build index sets once
		if symbolic_only:
			symbolic_idxs = [
				idx for idx, name in enumerate(self.atom_names)
				if name in self.base_atom_names
			]
		else:
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
	def get_symbolic_choices(self) -> Dict[Tuple[int, int], str]:
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
		entropy_weight: float = 1e-3,
		l1_weight: float = 0.0,
	) -> torch.Tensor:
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
	def get_subset(self, in_ids: torch.Tensor, out_ids: torch.Tensor):
		"""
		For pruning: return a new layer restricted to selected inputs/outputs.
		"""
		in_ids  = torch.as_tensor(in_ids, dtype=torch.long)
		out_ids = torch.as_tensor(out_ids, dtype=torch.long)

		new = GatedSymbolicLayer(
			input_dim=in_ids.numel(),
			output_dim=out_ids.numel(),
			atom_names=self.base_atom_names,
			init_atom_bias=0.0,
			symbolic_scale=self.symbolic_scale,
			numeric_atom_configs=self._numeric_atom_configs,
		).to(self.mask.device)

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
			# basis is shared across all edges ⇒ just copy the entire module
			new_basis = new.numeric_atoms[name]
			new_basis.load_state_dict(basis.state_dict())

			# subset the coefficient tensor: [out_dim, in_dim, G] -> [new_out, new_in, G]
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
