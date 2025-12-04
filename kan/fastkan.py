import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SplineLinear(nn.Linear):
	def __init__(self, in_features: int, out_features: int,
				 init_scale: float = 0.1, **kw) -> None:
		self.init_scale = init_scale
		super().__init__(in_features, out_features, bias=False, **kw)

	def reset_parameters(self) -> None:
		nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class HingeBasisFunction(nn.Module):
	def __init__(
		self,
		grid_min=-2., grid_max=2., num_grids=8,
		train_grid=True,
	):
		super().__init__()
		grid = torch.linspace(grid_min, grid_max, num_grids)
		self.grid = nn.Parameter(grid, requires_grad=train_grid)

	def forward(self, x):
		# basis_j(x) = max(0, x - c_j)
		return F.relu(x[..., None] - self.grid)

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
		self.rbf = StepBasisFunction(
			grid_min=grid_min,
			grid_max=grid_max,
			num_grids=num_grids,
			train_grid=train_grid,
			# train_denominator=train_denominator,
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

class OpFastKANLayer(nn.Module):
	"""
	FastKAN-style layer with learned ADD vs MUL per output node.

	Numeric API matches KAN / FastKAN:
	  forward(x) -> (x_numerical, preacts, postacts_numerical, postspline)

	Differences:
	  - aggregation is not always sum over inputs; instead,
		for each output j we learn a gate over:
		  add: sum_i postspline[..., j, i]
		  mul: prod_i postspline[..., j, i]
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
		ops: Sequence[str] = ("add", "mul"),
		use_gumbel: bool = False,
		gumbel_tau: float = 1.0,
	) -> None:
		super().__init__()

		self.in_dim  = input_dim
		self.out_dim = output_dim
		self.num_grids = num_grids

		self.layernorm = nn.LayerNorm(input_dim)
		self.rbf = StepBasisFunction(
			grid_min=grid_min,
			grid_max=grid_max,
			num_grids=num_grids,
			train_grid=train_grid,
		)
		self.spline_linear = SplineLinear(
			input_dim * num_grids, output_dim,
			init_scale=spline_weight_init_scale
		)

		self.use_base_update = use_base_update
		if use_base_update:
			self.base_activation = base_activation
			self.base_linear = nn.Linear(input_dim, output_dim, bias=False)

		# pruning mask: [in_dim, out_dim]
		self.mask = nn.Parameter(
			torch.ones(input_dim, output_dim),
			requires_grad=False
		)

		# meta fields used by MultKAN / KAN
		self.out_dim_sum  = output_dim
		self.out_dim_mult = 0
		self.k = 3
		self.scale_base = nn.Parameter(torch.ones(output_dim))
		self.scale_sp   = nn.Parameter(torch.ones(output_dim))

		# ---- operator gating: add vs mul ----
		self.ops = ops
		self.num_ops = 2
		# one logit vector per output node: [out_dim, 2]
		self.op_logits = nn.Parameter(
			torch.zeros(output_dim, self.num_ops)
		)

		self.use_gumbel = use_gumbel
		self.gumbel_tau = gumbel_tau

		self.last_op_gates = None  # for inspection

	# ---------- properties ----------

	@property
	def grid(self):
		return self.rbf.grid

	@property
	def coef(self) -> torch.Tensor:
		W = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
		return W.reshape(self.out_dim * self.in_dim, self.num_grids)

	# ---------- operator utilities ----------

	@torch.no_grad()
	def get_op_choice(self, hard: bool = True):
		"""
		Returns a list of 'add'/'mul' for each output node.

		hard=True: argmax over softmax(logits)
		"""
		if hard:
			probs = F.softmax(self.op_logits, dim=-1)
			idx = probs.argmax(dim=-1)        # [out_dim]
		else:
			idx = self.op_logits.argmax(dim=-1)

		ops = []
		for k in idx.tolist():
			if k == 0:
				ops.append("add")
			else:
				ops.append("mul")
		return ops

	def _aggregate(self, postspline: torch.Tensor) -> torch.Tensor:
		"""
		postspline: [B, O, I]

		Returns:
		  y: [B, O]
		"""
		B, O, I = postspline.shape

		# candidate aggregations
		agg_add = postspline.sum(dim=-1)               # [B, O]
		# small eps so near-zero edges don't kill everything
		eps = 1e-6
		agg_mul = torch.prod(postspline + eps, dim=-1) # [B, O]

		# gates over [add, mul]: [O, 2]
		logits = self.op_logits
		if self.use_gumbel:
			gates = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
		else:
			gates = F.softmax(logits, dim=-1)

		self.last_op_gates = gates.detach()

		# y[b,o] = gates[o,0]*agg_add[b,o] + gates[o,1]*agg_mul[b,o]
		y = (gates[:, 0].unsqueeze(0) * agg_add +
			 gates[:, 1].unsqueeze(0) * agg_mul)
		return y

	# ---------- main forward ----------

	def forward(self, x: torch.Tensor, time_benchmark: bool = True):
		"""
		x: [B, in_dim]

		Returns:
		  x_numerical        : [B, out_dim]
		  preacts            : [B, out_dim, in_dim]
		  postacts_numerical : [B, out_dim, in_dim]
		  postspline         : [B, out_dim, in_dim]
		"""
		B = x.shape[0]

		if not time_benchmark:
			pre = self.layernorm(x)   # [B, I]
		else:
			pre = x                   # [B, I]

		basis = self.rbf(pre)        # [B, I, G]

		# spline term
		W_spline = self.spline_linear.weight.view(
			self.out_dim, self.in_dim, self.num_grids
		)                            # [O, I, G]
		postspline = torch.einsum('big,oig->boi', basis, W_spline)   # [B, O, I]

		# base_update
		if self.use_base_update:
			base_hidden = self.base_activation(x)         # [B, I]
			W_base = self.base_linear.weight              # [O, I]
			base_per_edge = torch.einsum('bi,oi->boi', base_hidden, W_base)
			postspline = postspline + base_per_edge       # [B, O, I]

		# pruning mask
		postspline = postspline * self.mask.T.unsqueeze(0)  # [B, O, I]

		# aggregate according to learned operator
		x_numerical = self._aggregate(postspline)          # [B, O]

		# compat
		postacts_numerical = postspline
		preacts = pre.unsqueeze(1).expand(B, self.out_dim, self.in_dim)

		return x_numerical, preacts, postacts_numerical, postspline

	# ---------- grid-adaptation & structural ops ----------

	@torch.no_grad()
	def update_grid_from_samples(self, acts: torch.Tensor):
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
									parent_layer: "OpFastKANLayer",
									parent_acts: torch.Tensor):
		if hasattr(parent_layer, "rbf") and \
		   parent_layer.rbf.grid.shape == self.rbf.grid.shape:
			self.rbf.grid.copy_(parent_layer.rbf.grid)
		else:
			self.update_grid_from_samples(parent_acts)

	@torch.no_grad()
	def get_subset(self,
				   in_ids: torch.Tensor,
				   out_ids: torch.Tensor) -> "OpFastKANLayer":
		in_ids  = torch.as_tensor(in_ids, dtype=torch.long)
		out_ids = torch.as_tensor(out_ids, dtype=torch.long)

		new = OpFastKANLayer(
			input_dim=in_ids.numel(),
			output_dim=out_ids.numel(),
			grid_min=float(self.grid.min().item()),
			grid_max=float(self.grid.max().item()),
			num_grids=self.num_grids,
			use_base_update=self.use_base_update,
			base_activation=getattr(self, "base_activation", F.silu),
			spline_weight_init_scale=self.spline_linear.init_scale,
			use_gumbel=self.use_gumbel,
			gumbel_tau=self.gumbel_tau,
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

		# scale_* regs
		new.scale_base.copy_(self.scale_base[out_ids])
		new.scale_sp.copy_(self.scale_sp[out_ids])

		# op logits
		new.op_logits.copy_(self.op_logits[out_ids])

		new.out_dim_sum  = out_ids.numel()
		new.out_dim_mult = 0

		return new

	@torch.no_grad()
	def swap(self, i1: int, i2: int, mode: str = 'in'):
		if mode not in ("in", "out"):
			raise ValueError("mode must be 'in' or 'out'")

		if mode == "in":
			self.mask[[i1, i2], :] = self.mask[[i2, i1], :]

			W = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
			W[:, [i1, i2], :] = W[:, [i2, i1], :]
			self.spline_linear.weight.copy_(W.view(self.out_dim, self.in_dim * self.num_grids))

			if self.use_base_update:
				self.base_linear.weight[:, [i1, i2]] = \
					self.base_linear.weight[:, [i2, i1]]

			lnw = self.layernorm.weight.data
			lnb = self.layernorm.bias.data
			lnw[[i1, i2]] = lnw[[i2, i1]]
			lnb[[i1, i2]] = lnb[[i2, i1]]

		else:  # mode == 'out'
			self.mask[:, [i1, i2]] = self.mask[:, [i2, i1]]

			W = self.spline_linear.weight.view(self.out_dim, self.in_dim, self.num_grids)
			W[[i1, i2], :, :] = W[[i2, i1], :, :]
			self.spline_linear.weight.copy_(W.view(self.out_dim, self.in_dim * self.num_grids))

			if self.use_base_update:
				self.base_linear.weight[[i1, i2], :] = \
					self.base_linear.weight[[i2, i1], :]

			sb = self.scale_base.data
			sp = self.scale_sp.data
			sb[[i1, i2]] = sb[[i2, i1]]
			sp[[i1, i2]] = sp[[i2, i1]]

			# op logits follow outputs
			self.op_logits[[i1, i2]] = self.op_logits[[i2, i1]]
