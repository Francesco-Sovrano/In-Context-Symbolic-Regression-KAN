import torch
import torch.nn as nn
import numpy as np
# from .KANLayer import KANLayer
from .fastkan import FastKANLayer as KANLayer, GatedSymbolicLayer
#from .Symbolic_MultKANLayer import *
from .Symbolic_KANLayer import Symbolic_KANLayer
from .LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
#from .MultKANLayer import MultKANLayer
import pandas as pd
from sympy.printing import latex
from sympy import Float as SymFloat
import sympy
import yaml
from .utils import SYMBOLIC_LIB
import contextlib, signal, sympy as sp
import math
import re

def compactify_symbolic_formula(f):
	f = str(f)
	f = re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', f)
	f = re.sub(r'0\.99+\*', '', f)
	f = re.sub(r'1\.0+\*', '', f)
	# small = r'\d(?:\.\d+)?e-(?:0?[4-9]|\d{2,})'  # e-4..e-9, e-04..e-09, e-10, e-11, ...
	# f = re.sub(rf'\s+[+-]\s+{small}\s*[^*/]', '', f)         # ... ± tiny
	# f = re.sub(rf'\s*[^*/]{small}\s+[+-]\s+', '', f)         # tiny ± ...
	return f

@contextlib.contextmanager
def _model_snapshot(model):
	"""
	In-place snapshot/restore of a MultKAN, including:
	  - all parameters/buffers (via state_dict)
	  - training/eval mode & flags
	  - RNG state
	  - cached activations
	  - chained sub-KAN structure (_chained_kan_modules, _chained_kan_meta)

	Assumes you do NOT change the architecture (width, depth, etc.)
	inside the context; only parameters and sub-KAN attachments.
	"""

	# ---------- 1) Save weights/buffers (CPU copy) ----------
	state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

	# ---------- 2) Save training/eval flag + attrs you toggle ----------
	mode       = model.training
	save_act0  = getattr(model, "save_act", False)
	auto_save0 = getattr(model, "auto_save", False)

	# ---------- 3) RNG states for determinism ----------
	torch_state = torch.random.get_rng_state()
	cuda_state  = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
	np_state    = np.random.get_state()
	py_state    = random.getstate()

	# ---------- 4) Cache forward-time buffers ----------
	cache_data        = model.cache_data
	acts              = model.acts[:]              if model.acts              is not None else None
	acts_scale        = model.acts_scale[:]        if model.acts_scale        is not None else None
	acts_scale_spline = model.acts_scale_spline[:] if model.acts_scale_spline is not None else None
	subnode_actscale  = model.subnode_actscale[:]  if model.subnode_actscale  is not None else None
	edge_actscale     = model.edge_actscale[:]     if model.edge_actscale     is not None else None

	# ---------- 5) Snapshot chained-KAN structures ----------
	had_ckm      = hasattr(model, "_chained_kan_modules")
	had_ckm_meta = hasattr(model, "_chained_kan_meta")

	if had_ckm:
		# shallow copy of mapping: keys and module *objects* are kept;
		# we just restore the mapping later to drop any new edges.
		prev_ckm_modules = nn.ModuleDict({
			k: v for k, v in model._chained_kan_modules.items()
		})
	else:
		prev_ckm_modules = None

	prev_ckm_meta = copy.deepcopy(model._chained_kan_meta) if had_ckm_meta else None

	try:
		# everything inside the 'with' happens here
		yield
	finally:
		# ---------- A) Restore / remove chained-KAN modules ----------
		if had_ckm:
			model._chained_kan_modules = nn.ModuleDict({
				k: v for k, v in prev_ckm_modules.items()
			})
		else:
			if hasattr(model, "_chained_kan_modules"):
				delattr(model, "_chained_kan_modules")

		# ---------- B) Restore / remove chained-KAN metadata ----------
		if had_ckm_meta:
			model._chained_kan_meta = prev_ckm_meta
		else:
			if hasattr(model, "_chained_kan_meta"):
				delattr(model, "_chained_kan_meta")

		# ---------- C) Restore weights and mode ----------
		# Architecture hasn't changed, so strict=True is fine
		model.load_state_dict(state, strict=True)
		model.train(mode)
		model.save_act  = save_act0
		model.auto_save = auto_save0

		# ---------- D) Restore RNG states ----------
		torch.random.set_rng_state(torch_state)
		if cuda_state is not None:
			torch.cuda.set_rng_state_all(cuda_state)
		np.random.set_state(np_state)
		random.setstate(py_state)

		# ---------- E) Restore cached forward buffers ----------
		model.cache_data        = cache_data
		model.acts              = acts
		model.acts_scale        = acts_scale
		model.acts_scale_spline = acts_scale_spline
		model.subnode_actscale  = subnode_actscale
		model.edge_actscale     = edge_actscale

class SimplifyTimeout(Exception): pass

@contextlib.contextmanager
def time_limit(seconds: float):
	"""Hard wall-clock timeout using SIGALRM (Unix only)."""
	if seconds is None or seconds <= 0:
		yield; return
	if not hasattr(signal, "setitimer"):   # e.g., Windows
		yield; return                      # can't enforce w/o processes
	prev = signal.getsignal(signal.SIGALRM)
	def _raise(_signum, _frame): raise SimplifyTimeout()
	try:
		signal.signal(signal.SIGALRM, _raise)
		signal.setitimer(signal.ITIMER_REAL, seconds)
		yield
	finally:
		signal.setitimer(signal.ITIMER_REAL, 0)
		signal.signal(signal.SIGALRM, prev)

class MultKAN(nn.Module):
	def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu', atom_names=None, numeric_atom_configs=None, chain_nodes=0, chain_types=None):

		super(MultKAN, self).__init__()

		self.seed = seed
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		### initializeing the numerical front ###

		self.act_fun = []
		self.depth = len(width) - 1
		self.chain_nodes = chain_nodes
		self.chain_types = chain_types

		# multiplicative strength per layer (start at 0 = no mult influence)
		# self.mult_alpha = nn.ParameterList([
		#   nn.Parameter(torch.tensor(0.0), requires_grad=False)
		#   for _ in range(self.depth)
		# ])
		
		#print('haha1', width)
		for i in range(len(width)):
			#print(type(width[i]), type(width[i]) == int)
			if type(width[i]) == int or type(width[i]) == np.int64:
				width[i] = [width[i],0]
				
		#print('haha2', width)
			
		self.width = width
		
		# if mult_arity is just a scalar, we extend it to a list of lists
		# e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
		# in the second hidden layer, 1 mult op has arity 4.
		if isinstance(mult_arity, int):
			self.mult_homo = True # when homo is True, parallelization is possible
		else:
			self.mult_homo = False # when home if False, for loop is required. 
		self.mult_arity = mult_arity

		width_in = self.width_in
		width_out = self.width_out
		
		self.base_fun_name = base_fun
		if base_fun == 'silu':
			base_fun = torch.nn.SiLU()
		elif base_fun == 'identity':
			base_fun = torch.nn.Identity()
		elif base_fun == 'zero':
			base_fun = lambda x: x*0.

		self.k = k
		self.base_fun = base_fun
		
		self.grid = grid
		self.grid_eps = grid_eps
		self.grid_range = grid_range

		self.atom_names = atom_names or []
		self.noise_scale = noise_scale
		self.numeric_atom_configs = numeric_atom_configs

		self.scale_base_mu = scale_base_mu
		self.scale_base_sigma = scale_base_sigma

		self.affine_trainable = affine_trainable
		self.sp_trainable = sp_trainable
		self.sb_trainable = sb_trainable
		
		self.save_act = save_act

		self.auto_save = auto_save
		self.state_id = 0
		self.ckpt_path = ckpt_path
		self.round = round
		
		self.device = device
			
		self._chained_kan_modules = nn.ModuleDict()
		self._chained_kan_meta = {}

		# NEW: per-edge gate logits (one scalar gate per edge)
		# shape for layer l: (width_in[l], width_out[l+1])
		# init them strongly "off" (negative logit)
		self.chained_gate_logits = nn.ParameterList([
			nn.Parameter(torch.full((self.width_in[l], self.width_out[l+1]), -5.0))
			for l in range(self.depth)
		])

		# NEW: pre-build sub-KANs at init if chain_nodes > 0
		# chain_types can be "mul"/"div" per layer or a single string
		if self.chain_nodes and self.chain_nodes > 0 and self.chain_types is not None:
			self._init_all_chained_kan_edges()
		
		for l in range(self.depth):
			# splines
			if isinstance(grid, list):
				grid_l = grid[l]
			else:
				grid_l = grid
				
			if isinstance(k, list):
				k_l = k[l]
			else:
				k_l = k
					
			if self.atom_names is None and not self.numeric_atom_configs:
				sp_batch = KANLayer(
					input_dim=width_in[l],
					output_dim=width_out[l+1],
					grid_min=self.grid_range[0],
					grid_max=self.grid_range[1],
					num_grids=grid_l,
					k=k_l, 
					noise_scale=self.noise_scale, 
					scale_base_mu=self.scale_base_mu, 
					scale_base_sigma=self.scale_base_sigma, 
					# use_base_update=False,
				)
			else:
				sp_batch = GatedSymbolicLayer(
					input_dim=width_in[l],
					output_dim=width_out[l+1],
					atom_names=self.atom_names,
					numeric_atom_configs=self.numeric_atom_configs,
					base_activation=base_fun,
				)

			# sp_batch = KANLayer(in_dim=width_in[l], out_dim=width_out[l+1], num=grid_l, k=k_l, noise_scale=noise_scale, scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, sparse_init=sparse_init)
			self.act_fun.append(sp_batch)

		self.node_bias = []
		self.node_scale = []
		self.subnode_bias = []
		self.subnode_scale = []

		self.node_bias  = nn.ParameterList([
			nn.Parameter(torch.zeros(self.width_in[l + 1]), requires_grad=affine_trainable)
			for l in range(self.depth)
		])
		self.node_scale = nn.ParameterList([
			nn.Parameter(torch.ones (self.width_in[l + 1]), requires_grad=affine_trainable)
			for l in range(self.depth)
		])
		self.subnode_bias  = nn.ParameterList([
			nn.Parameter(torch.zeros(self.width_out[l + 1]), requires_grad=affine_trainable)
			for l in range(self.depth)
		])
		self.subnode_scale = nn.ParameterList([
			nn.Parameter(torch.ones (self.width_out[l + 1]), requires_grad=affine_trainable)
			for l in range(self.depth)
		])
			
		
		self.act_fun = nn.ModuleList(self.act_fun)

		### initializing the symbolic front ###
		self.symbolic_fun = []
		for l in range(self.depth):
			sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l+1])
			self.symbolic_fun.append(sb_batch)

		self.symbolic_fun = nn.ModuleList(self.symbolic_fun)

		# if self.atom_names is not None:
		#   # disable legacy symbolic branch – gating layer is the only source of nonlinearity
		#   with torch.no_grad():
		#       for l in range(self.depth):
		#           self.symbolic_fun[l].mask.zero_()
				
		self.node_scores = None
		self.edge_scores = None
		self.subnode_scores = None
		
		self.cache_data = None
		self.acts = None
		
		self.to(self.device)
		
		if auto_save:
			if first_init:
				if not os.path.exists(ckpt_path):
					# Create the directory
					os.makedirs(ckpt_path)
				print(f"checkpoint directory created: {ckpt_path}")
				print('saving model version 0.0')

				history_path = self.ckpt_path+'/history.txt'
				with open(history_path, 'w') as file:
					file.write(f'### Round {self.round} ###' + '\n')
					file.write('init => 0.0' + '\n')
				self.saveckpt(path=self.ckpt_path+'/'+'0.0')
			else:
				self.state_id = state_id
			
		self.input_id = torch.arange(self.width_in[0],)

	def _clone_chained_kan_structure_from(self, src: "MultKAN"):
		"""
		Recreate the same chained sub-KAN structure that `src` has,
		but only for edges whose (l, i, j) are still valid for the
		current model shape (self.act_fun[l].mask).

		Call this *before* load_state_dict when cloning `src`.
		"""
		# If src has no chained meta, just reset ours to empty
		if not hasattr(src, "_chained_kan_meta") or not src._chained_kan_meta:
			self._chained_kan_modules = nn.ModuleDict()
			self._chained_kan_meta = {}
			return

		# Ensure we have containers
		self._chained_kan_modules = nn.ModuleDict()
		self._chained_kan_meta = {}

		src_has_modules = hasattr(src, "_chained_kan_modules")

		for (l, i, j), meta in src._chained_kan_meta.items():
			# --- 1) layer must exist ---
			if l < 0 or l >= len(self.act_fun):
				continue

			# --- 2) (i, j) must be in-bounds for current mask shape ---
			mask = getattr(self.act_fun[l], "mask", None)
			if mask is None:
				continue
			in_dim, out_dim = mask.shape
			if not (0 <= i < in_dim and 0 <= j < out_dim):
				# stale edge from a previous (wider) architecture – skip it
				continue

			# --- 3) src must actually have child modules for this edge ---
			op = meta.get("op", "mul")
			edge_key_src = meta.get("key", f"l{l}_i{i}_j{j}")
			if (not src_has_modules) or (edge_key_src not in src._chained_kan_modules):
				continue
			num_children = len(src._chained_kan_modules[edge_key_src])

			# --- 4) create child sub-KANs and meta in self ---
			self._init_chained_kan_edge(
				l=l,
				i=i,
				j=j,
				op=op,
				hidden_nodes=num_children,
				verbose=False,
			)

	def _filtered_state_for_clone(self, clone_model: "MultKAN"):
		"""
		Build a filtered version of self.state_dict() that is safe to load
		into `clone_model` (the pruned copy):

		- keep only _chained_kan_modules.* entries for edges that exist in clone_model
		- slice chained_gate_logits.* to match clone_model shapes if needed
		- keep everything else as-is
		"""
		src_state = self.state_dict()
		new_state = {}

		# convenience
		clone_has_sub = hasattr(clone_model, "_chained_kan_modules")
		clone_sub_keys = set(clone_model._chained_kan_modules.keys()) if clone_has_sub else set()

		for k, v in src_state.items():
			# --- 1) parameters of child sub-KANs ---
			if k.startswith("_chained_kan_modules."):
				# keys like: "_chained_kan_modules.l0_i0_j1.0.chained_gate_logits.0"
				parts = k.split(".")
				if len(parts) < 3:
					continue
				edge_key = parts[1]  # e.g. "l0_i0_j1"
				if edge_key not in clone_sub_keys:
					# this sub-KAN edge doesn't exist in the clone => drop it
					continue
				new_state[k] = v
				continue

			# --- 2) top-level chained_gate_logits for the root model ---
			if k.startswith("chained_gate_logits."):
				# k = "chained_gate_logits.<l>"
				try:
					idx = int(k.split(".")[1])
				except Exception:
					continue

				if not hasattr(clone_model, "chained_gate_logits"):
					continue
				if idx >= len(clone_model.chained_gate_logits):
					continue

				dest = clone_model.chained_gate_logits[idx]
				if v.shape == dest.shape:
					new_state[k] = v
				else:
					# slice to overlapping shape if possible
					in_old, out_old = v.shape
					in_new, out_new = dest.shape
					if in_new <= in_old and out_new <= out_old:
						new_state[k] = v[:in_new, :out_new]
					else:
						# can't sensibly map this, so skip
						continue
				continue

			# --- 3) everything else (normal KAN + symbolic params, etc.) ---
			new_state[k] = v

		return new_state


	def _cleanup_stale_chained_kan_meta(self):
		"""
		Remove any entries from _chained_kan_meta whose (l, i, j)
		are no longer valid for the current architecture.
		"""
		if not hasattr(self, "_chained_kan_meta") or not self._chained_kan_meta:
			return

		new_meta = {}
		for (l, i, j), meta in self._chained_kan_meta.items():
			if l < 0 or l >= len(self.act_fun):
				continue
			mask = getattr(self.act_fun[l], "mask", None)
			if mask is None:
				continue
			in_dim, out_dim = mask.shape
			if 0 <= i < in_dim and 0 <= j < out_dim:
				new_meta[(l, i, j)] = meta

		self._chained_kan_meta = new_meta



	def _init_all_chained_kan_edges(self):
		"""
		Create a chained KAN chain for *every* edge (l, i -> j), but
		do NOT rewire the numeric/symbolic masks (we just want the
		modules to exist and be trainable via gating).
		"""
		# chain_types can be a single string or list per layer
		def _layer_op(l):
			if isinstance(self.chain_types, (list, tuple)):
				return self.chain_types[l]
			return self.chain_types

		for l in range(self.depth):
			op = _layer_op(l)
			# normalize op to "mul" / "div"
			if op in ("mul", "multiplication", "*"):
				op_ = "mul"
			elif op in ("div", "division", "/"):
				op_ = "div"
			else:
				raise ValueError(f"Unsupported chain_types[{l}] = {op!r}")
			for i in range(self.width_in[l]):
				for j in range(self.width_out[l+1]):
					self._init_chained_kan_edge(
						l=l,
						i=i,
						j=j,
						op=op_,
						hidden_nodes=self.chain_nodes,
						verbose=False,
						rewire=False,   # <--- KEY: don't touch masks or fun_names
					)

		
	def to(self, device):
		'''
		move the model to device
		
		Args:
		-----
			device : str or device

		Returns:
		--------
			self
			
		Example
		-------
		>>> from kan import *
		>>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		>>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
		>>> model.to(device)
		'''
		super(MultKAN, self).to(device)
		self.device = device
		
		for kanlayer in self.act_fun:
			kanlayer.to(device)
			
		for symbolic_kanlayer in self.symbolic_fun:
			symbolic_kanlayer.to(device)
			
		return self

	@property
	def n_edge(self):
		'''
		the number of active edges
		'''
		depth = len(self.act_fun)
		complexity = 0
		for l in range(depth):
			complexity += torch.sum(self.act_fun[l].mask > 0.)
		return complexity.item()
	
	@property
	def width_in(self):
		'''
		The number of input nodes for each layer
		'''
		width = self.width
		width_in = [width[l][0]+width[l][1] for l in range(len(width))]
		return width_in
		
	@property
	def width_out(self):
		'''
		The number of output subnodes for each layer
		'''
		width = self.width
		if self.mult_homo == True:
			width_out = [width[l][0]+self.mult_arity*width[l][1] for l in range(len(width))]
		else:
			width_out = [width[l][0]+int(np.sum(self.mult_arity[l])) for l in range(len(width))]
		return width_out
	
	@property
	def n_sum(self):
		'''
		The number of addition nodes for each layer
		'''
		width = self.width
		n_sum = [width[l][0] for l in range(1,len(width)-1)]
		return n_sum
	
	@property
	def n_mult(self):
		'''
		The number of multiplication nodes for each layer
		'''
		width = self.width
		n_mult = [width[l][1] for l in range(1,len(width)-1)]
		return n_mult
	
	@property
	def feature_score(self):
		'''
		attribution scores for inputs
		'''
		self.attribute()
		if self.node_scores is None:
			return None
		else:
			return self.node_scores[0]
	
	def log_history(self, method_name): 

		if self.auto_save:

			# save to log file
			#print(func.__name__)
			with open(self.ckpt_path+'/history.txt', 'a') as file:
				file.write(str(self.round)+'.'+str(self.state_id)+' => '+ method_name + ' => ' + str(self.round)+'.'+str(self.state_id+1) + '\n')

			# update state_id
			self.state_id += 1

			# save to ckpt
			self.saveckpt(path=self.ckpt_path+'/'+str(self.round)+'.'+str(self.state_id))
			print('saving model version '+str(self.round)+'.'+str(self.state_id))
	
	def saveckpt(self, path='model'):
		'''
		save the current model to files (configuration file and state file)
		
		Args:
		-----
			path : str
				the path where checkpoints are saved

		Returns:
		--------
			None
			
		Example
		-------
		>>> from kan import *
		>>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		>>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
		>>> model.saveckpt('./mark')
		# There will be three files appearing in the current folder: mark_cache_data, mark_config.yml, mark_state
		'''
	
		model = self
		
		dic = dict(
			width = model.width,
			grid = model.grid,
			k = model.k,
			mult_arity = model.mult_arity,
			base_fun_name = model.base_fun_name,
			affine_trainable = model.affine_trainable,
			grid_eps = model.grid_eps,
			grid_range = model.grid_range,
			sp_trainable = model.sp_trainable,
			sb_trainable = model.sb_trainable,
			state_id = model.state_id,
			auto_save = model.auto_save,
			ckpt_path = model.ckpt_path,
			round = model.round,
			device = str(model.device),
			atom_names = model.atom_names,
			noise_scale = model.noise_scale,
			numeric_atom_configs = model.numeric_atom_configs,
			chain_nodes = model.chain_nodes,
			chain_types = model.chain_types,
			seed = model.seed,
		)

		# --- store chained sub-KAN metadata for reconstruction ---
		if hasattr(self, "_chained_kan_meta") and self._chained_kan_meta:
			bin_meta = []
			for (l, i, j), meta in self._chained_kan_meta.items():
				key = meta.get("key", f"l{l}_i{i}_j{j}")
				op = meta.get("op", "mul")
				num_children = len(self._chained_kan_modules[key])
				bin_meta.append({
					"l": int(l),
					"i": int(i),
					"j": int(j),
					"op": op,
					"num_children": int(num_children),
				})
			dic["_chained_kan_meta_list"] = bin_meta

		for i in range (model.depth):
			dic[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

		with open(f'{path}_config.yml', 'w') as outfile:
			yaml.dump(dic, outfile, default_flow_style=False)

		torch.save(model.state_dict(), f'{path}_state')
		torch.save(model.cache_data, f'{path}_cache_data')
	
	@staticmethod
	def loadckpt(path='model'):
		with open(f'{path}_config.yml', 'r') as stream:
			config = yaml.safe_load(stream)

		state = torch.load(f'{path}_state', map_location='cpu')  # <-- portability

		model_load = MultKAN(
			width=config['width'], grid=config['grid'], k=config['k'],
			seed=config['seed'],
			mult_arity=config['mult_arity'], base_fun=config['base_fun_name'],
			affine_trainable=config['affine_trainable'], grid_eps=config['grid_eps'],
			grid_range=config['grid_range'], sp_trainable=config['sp_trainable'],
			sb_trainable=config['sb_trainable'], state_id=config['state_id'],
			auto_save=config['auto_save'], first_init=False,
			ckpt_path=config['ckpt_path'], round=config['round']+1,
			device=config['device'], atom_names=config['atom_names'], 
			noise_scale=config['noise_scale'],
			numeric_atom_configs=config['numeric_atom_configs'],
			chain_nodes=config['chain_nodes'],
			chain_types=config['chain_types'],
		)

		# --- reconstruct chained sub-KANs from stored meta, if any ---
		bin_meta = config.get("_chained_kan_meta_list", [])
		if bin_meta:
			for entry in bin_meta:
				l = entry["l"]
				i = entry["i"]
				j = entry["j"]
				op = entry["op"]
				num_children = entry["num_children"]
				model_load._init_chained_kan_edge(
					l=l,
					i=i,
					j=j,
					op=op,
					hidden_nodes=num_children,
					verbose=False,
				)

		model_load.load_state_dict(state, strict=True)
		model_load.cache_data = torch.load(f'{path}_cache_data')
		
		depth = len(model_load.width) - 1
		for l in range(depth):
			out_dim = model_load.symbolic_fun[l].out_dim
			in_dim = model_load.symbolic_fun[l].in_dim
			funs_name = config[f'symbolic.funs_name.{l}']
			for j in range(out_dim):
				for i in range(in_dim):
					fun_name = funs_name[j][i]
					model_load.symbolic_fun[l].funs_name[j][i] = fun_name
					model_load.symbolic_fun[l].funs[j][i] = SYMBOLIC_LIB[fun_name][0]
					model_load.symbolic_fun[l].funs_sympy[j][i] = SYMBOLIC_LIB[fun_name][1]
					model_load.symbolic_fun[l].funs_avoid_singularity[j][i] = SYMBOLIC_LIB[fun_name][3]
		return model_load

	def copy(self):
		'''
		deepcopy
		
		Args:
		-----
			path : str
				the path where checkpoints are saved

		Returns:
		--------
			MultKAN
			
		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
		>>> model2 = model.copy()
		>>> model2.act_fun[0].coef.data *= 2
		>>> print(model2.act_fun[0].coef.data)
		>>> print(model.act_fun[0].coef.data)
		'''
		path='copy_temp'
		self.saveckpt(path)
		return KAN.loadckpt(path)
	
	def update_grid_from_samples(self, x):
		for l in range(self.depth):
			self.get_act(x)
			self.act_fun[l].update_grid_from_samples(self.acts[l])
			
	def update_grid(self, x):
		'''
		call update_grid_from_samples. This seems unnecessary but we retain it for the sake of classes that might inherit from MultKAN
		'''
		self.update_grid_from_samples(x)

	def initialize_grid_from_another_model(self, model, x):
		model(x)
		for l in range(self.depth):
			self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

	@property
	def symbolic_enabled(self):
		# Only count legacy Symbolic_KANLayer when it's actually active.
		return any(
			self.symbolic_fun[l].mask.abs().sum().item() > 0
			# and not isinstance(self.act_fun[l], GatedSymbolicLayer)
			for l in range(self.depth)
		)

	def forward(self, x, singularity_avoiding=False, y_th=10.):
		x = x[:,self.input_id.long()]
		assert x.shape[1] == self.width_in[0]
		
		# cache data
		self.cache_data = x
		
		self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
		self.acts_scale = []
		self.acts_scale_spline = []
		self.subnode_actscale = []
		self.edge_actscale = []
		# self.neurons_scale = []

		self.acts.append(x)  # acts shape: (batch, width[l])

		# _symbolic_enabled = self.symbolic_enabled

		for l in range(self.depth):
			
			x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

			# NEW: inject chained sub-KAN contribution for this layer
			x_numerical = self._chained_kan_forward_layer(l, x, x_numerical)

			# ===== normal symbolic part =====
			if self.symbolic_fun[l].mask.abs().sum().item() > 0:
				x_symbolic, postacts_symbolic = self.symbolic_fun[l](
					x, singularity_avoiding=singularity_avoiding, y_th=y_th
				)
			else:
				x_symbolic = 0.
				postacts_symbolic = 0.

			# combine numeric + symbolic + chained-KAN contributions
			x = x_numerical + x_symbolic
			preacts = preacts + x_symbolic
			
			if self.save_act:
				# save subnode_scale
				self.subnode_actscale.append(torch.std(x, dim=0).detach())
			
			# subnode affine transform
			x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
			
			if self.save_act:
				postacts = postacts_numerical + postacts_symbolic

				# self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
				input_range = torch.std(preacts, dim=0) + 0.1
				output_range_spline = torch.std(postacts_numerical, dim=0)  # training: spline only
				output_range = torch.std(postacts, dim=0)  # viz: spline + symbolic

				# save edge_scale
				self.edge_actscale.append(output_range)
				
				self.acts_scale.append((output_range / input_range))
				self.acts_scale_spline.append(output_range_spline / input_range)
			
			# multiplication
			dim_sum  = self.width[l+1][0]
			dim_mult = self.width[l+1][1]

			# Extra idea: reparameterized product: h = ((prod_i (1 + eps * f_i)) - 1) / eps
			if self.mult_homo == True:
				for i in range(self.mult_arity-1):
					if i == 0:
						x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
					else:
						x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
			else:
				# fall back to your existing hetero-arity logic
				x_mult = None
				for j in range(dim_mult):
					acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
					for i in range(self.mult_arity[l+1][j]-1):
						if i == 0:
							x_mult_j = x[:, [acml_id]] * x[:, [acml_id+1]]
						else:
							x_mult_j = x_mult_j * x[:, [acml_id+i+1]]
					x_mult = x_mult_j if x_mult is None else torch.cat([x_mult, x_mult_j], dim=1)

			# if dim_mult > 0:
			#   x = torch.cat([x[:, :dim_sum], x_mult], dim=1)
			#   # # scale multiplicative part by an annealed scalar mult_alpha[l]
			#   # alpha = self.mult_alpha[l]
			#   # x = torch.cat([x[:, :dim_sum], alpha * x_mult], dim=1)
			
			# x = x + self.biases[l].weight
			# node affine transform
			x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
			
			self.acts.append(x.detach())
			
		return x

	def set_mode(self, l, i, j, mode, mask_n=None):
		if mode == "s":
			mask_n = 0.;
			mask_s = 1.
		elif mode == "n":
			mask_n = 1.;
			mask_s = 0.
		elif mode == "sn" or mode == "ns":
			if mask_n is None:
				mask_n = 1.
			else:
				mask_n = mask_n
			mask_s = 1.
		else:
			mask_n = 0.;
			mask_s = 0.

		with torch.no_grad():
			self.act_fun[l].mask[i, j] = mask_n
			self.symbolic_fun[l].mask[j, i] = mask_s

	def fix_symbolic(self, l, i, j, fun_name, verbose=True, random=False, log_history=True, given_params=None):
		r2, loss, params = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random, given_params=given_params)
		self.set_mode(l, i, j, mode="s")
		
		if log_history:
			self.log_history('fix_symbolic')
		return r2, loss, params

	def unfix_symbolic(self, l, i, j, log_history=True):
		'''
		unfix the (l,i,j) activation function.
		'''
		# restore numeric mode (this will re-enable the numeric spline edge)
		self.set_mode(l, i, j, mode="n")

		# clear symbolic name
		self.symbolic_fun[l].funs_name[j][i] = "0"

		# --- NEW: if this edge had a chained KAN attached, forget its metadata ---
		if hasattr(self, "_chained_kan_meta"):
			key = (l, i, j)
			if key in self._chained_kan_meta:
				self._chained_kan_meta.pop(key)

		# NOTE: I’m *not* deleting self._chained_kan_modules here because:
		# - they may be shared across multiple edges, or
		# - you might want to re-use them later.
		# If you ever want full cleanup, you can track per-edge indices instead.

		if log_history:
			self.log_history('unfix_symbolic')


	def unfix_symbolic_all(self, log_history=True):
		'''
		unfix all activation functions.
		'''
		for l in range(len(self.width) - 1):
			for i in range(self.width_in[l]):
				for j in range(self.width_out[l + 1]):
					self.unfix_symbolic(l, i, j, log_history)

	def reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, reg_type="elasticnet"):
		"""
		Numerically robust regularization.
		If anything goes NaN/inf, we clamp it to 0 so _safe_reg() doesn't see NaNs.
		"""

		# --- pick which activity scale to use ---
		if reg_metric == 'edge_forward_spline_n':
			acts_scale = self.acts_scale_spline

		elif reg_metric == 'edge_forward_sum':
			acts_scale = self.acts_scale

		elif reg_metric == 'edge_forward_spline_u':
			acts_scale = self.edge_actscale

		elif reg_metric == 'edge_backward':
			acts_scale = self.edge_scores

		elif reg_metric == 'node_backward':
			acts_scale = self.node_attribute_scores

		else:
			raise Exception(f'reg_metric = {reg_metric} not recognized!')

		reg_ = 0.0

		# --- entropy + L1/L2 over edge/node scores ---
		for l, vec in enumerate(acts_scale):
			# vec: [out_dim, in_dim]
			# Clean up NaN/inf right away
			vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
			vec = vec.abs()

			if vec.numel() == 0:
				continue

			# --- rows ---
			p_row_den = vec.sum(dim=1, keepdim=True)              # (out_dim, 1)
			zero_row = (p_row_den == 0)

			p_row_den_safe = torch.where(zero_row, torch.ones_like(p_row_den), p_row_den)
			p_row = vec / p_row_den_safe                          # (out_dim, in_dim)

			if zero_row.any():
				uniform_val = 1.0 / max(vec.shape[1], 1)
				p_row = torch.where(
					zero_row,
					torch.full_like(p_row, uniform_val),
					p_row,
				)

			p_row = torch.clamp(p_row, min=1e-4, max=1.0)

			# --- columns ---
			p_col_den = vec.sum(dim=0, keepdim=True)              # (1, in_dim)
			zero_col = (p_col_den == 0)

			p_col_den_safe = torch.where(zero_col, torch.ones_like(p_col_den), p_col_den)
			p_col = vec / p_col_den_safe

			if zero_col.any():
				uniform_val = 1.0 / max(vec.shape[0], 1)
				p_col = torch.where(
					zero_col,
					torch.full_like(p_col, uniform_val),
					p_col,
				)

			p_col = torch.clamp(p_col, min=1e-4, max=1.0)

			# final safety: kill any NaNs that might sneak in
			p_row = torch.nan_to_num(p_row, nan=1e-4, posinf=1.0, neginf=1e-4)
			p_col = torch.nan_to_num(p_col, nan=1e-4, posinf=1.0, neginf=1e-4)

			entropy_row = - torch.mean(torch.sum(p_row * torch.log2(p_row), dim=1))
			entropy_col = - torch.mean(torch.sum(p_col * torch.log2(p_col), dim=0))

			reg_ = reg_ + lamb_entropy * (entropy_row + entropy_col)

			# L1 / L2 / elastic-net on vec itself
			if reg_type == "l2":
				l2 = vec.pow(2).sum()
				reg_ = reg_ + lamb_l1 * l2
			elif reg_type == "l1":
				l1 = vec.sum()
				reg_ = reg_ + lamb_l1 * l1
			elif reg_type == "elasticnet":
				l1 = vec.sum()
				l2 = vec.pow(2).sum()
				reg_ = reg_ + lamb_l1 * 0.5 * (l1 + l2)

		# --- spline / coef regularizer ---
		for i in range(len(self.act_fun)):
			layer = self.act_fun[i]
			if not hasattr(layer, "coef") or layer.coef is None:
				continue
			coef = layer.coef
			if coef.numel() == 0:
				continue

			coef = torch.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

			coeff_l1 = torch.sum(torch.mean(torch.abs(coef), dim=1))
			if coef.shape[1] > 1:
				coeff_diff_l1 = torch.sum(
					torch.mean(torch.abs(torch.diff(coef, dim=1)), dim=1)
				)
			else:
				coeff_diff_l1 = 0.0

			reg_ = reg_ + lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

		return reg_

	
	def get_reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, reg_type="elasticnet"):
		'''
		Get regularization. This seems unnecessary but in case a class wants to inherit this, it may want to rewrite get_reg, but not reg.
		'''
		return self.reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, reg_type=reg_type)
	
	def get_params(self):
		"""
		Get parameters as a list (so they can be reused for clipping etc.).
		"""
		return list(self.parameters())
			
	def fit(self, dataset, optimizer="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2.,
		lamb_coef=0., lamb_coefdiff=0., update_grid=False, grid_update_num=10, loss_fn=None, lr=1.,
		start_grid_update_step=-1, stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False,
		in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
		singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_sum', reg_type="elasticnet", display_metrics=None, gating_entropy=0.0, gating_l1=0.0, mult_node_weight_decay=0):

		"""
		Numerically robust training loop (Adam/LBFGS) with safe forward/loss/regularizer,
		gradient clipping, and finite-aware logging.
		"""

		# ---- device & dtype
		dev = getattr(self, "device", "cpu")
		if isinstance(dev, str):
			dev = torch.device(dev)
		self.to(dev)
		dtype = next(self.parameters()).dtype

		# ---- move dataset once
		train_X = dataset['train_input'].to(dev)
		train_y = dataset['train_label'].to(dev)
		test_X  = dataset['test_input'].to(dev)
		test_y  = dataset['test_label'].to(dev)

		# ---- lambda guard
		if lamb == 0.:
			self.save_act = False

		# ---- utils
		def _safe_pred(x):
			y = self.forward(x, singularity_avoiding=singularity_avoiding, y_th=y_th)
			return torch.nan_to_num(y, nan=0, posinf=1e12, neginf=-1e12)

		def _safe_loss(pred, target, fn):
			l = fn(pred, target)
			return torch.nan_to_num(l, nan=0, posinf=1e12, neginf=-1e12)

		def _safe_reg():
			if not self.save_act:
				return torch.zeros((), device=dev, dtype=dtype)

			# existing KAN regularizer (edge/node attribution + spline coeffs)
			if reg_metric == 'edge_backward':
				self.attribute()
			if reg_metric == 'node_backward':
				self.node_attribute()

			r = torch.zeros((), device=dev, dtype=dtype)
			if lamb > 0:
				r = lamb*self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, reg_type=reg_type)

			# extra reg for node-level mult nodes
			mult_reg = torch.zeros((), device=dev, dtype=dtype)
			if mult_node_weight_decay > 0:
				for l in range(self.depth):
					if l + 1 >= len(self.width):
						continue
					width_cfg = self.width[l + 1]
					if not isinstance(width_cfg, (list, tuple)) or len(width_cfg) < 2:
						continue
					num_sum  = int(width_cfg[0])
					num_mult = int(width_cfg[1])
					if num_mult <= 0:
						continue

					for j in range(num_sum, num_sum + num_mult):
						mult_reg = mult_reg + self.node_scale[l][j].pow(2).sum()
						mult_reg = mult_reg + self.node_bias[l][j].pow(2).sum()
						layer = self.act_fun[l]
						if hasattr(layer, "spline_coeffs"):
							coeffs_j = layer.spline_coeffs[:, j, ...]
							mult_reg += coeffs_j.pow(2).sum()
				mult_reg = mult_node_weight_decay * mult_reg

			# add gating regularizer for all GatedSymbolicLayer layers
			gating_r = torch.zeros((), device=dev, dtype=dtype)
			if gating_entropy != 0.0 or gating_l1 != 0.0:
				for layer in self.act_fun:
					if isinstance(layer, GatedSymbolicLayer):
						gating_r += layer.gating_regularizer(
							entropy_weight=gating_entropy,
							l1_weight=gating_l1,
						)

				# NEW: regularizer for chained-SubKAN gates
				if hasattr(self, "chained_gate_logits"):
					for logits in self.chained_gate_logits:
						p = torch.sigmoid(logits)  # (in_dim, out_dim)

						if gating_l1 != 0.0:
							gating_r = gating_r + gating_l1 * p.mean()

						if gating_entropy != 0.0:
							# encourage confident 0/1 gates
							p_clamped = torch.clamp(p, 1e-4, 1 - 1e-4)
							ent = -(
								p_clamped * torch.log2(p_clamped)
								+ (1 - p_clamped) * torch.log2(1 - p_clamped)
							).mean()
							gating_r = gating_r + gating_entropy * ent

			return torch.nan_to_num(r + gating_r + mult_reg, nan=0, posinf=1e12, neginf=-1e12)

		def _safe_sqrt(x):
			return torch.sqrt(torch.clamp(torch.nan_to_num(x, nan=1e12, posinf=1e12, neginf=1e12), min=0.0))

		def _finite_or_raise(name, t):
			if not torch.isfinite(t).all():
				bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].squeeze(-1).tolist()
				raise RuntimeError(f"{name} contains non-finite values at indices {bad}")

		# ---- loss fn
		if loss_fn is None:
			loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
		else:
			loss_fn = loss_fn_eval = loss_fn

		# ---- grid update schedule (robust)
		grid_update_num = max(1, int(grid_update_num))
		stop_grid_update_step = max(0, int(stop_grid_update_step))
		grid_update_freq = max(1, int(stop_grid_update_step / grid_update_num)) if update_grid else steps+1

		# ---- optimizer
		params = list(self.get_params()) # it already includes the params from edge_level_multdiv_kan_modules

		opt_name = optimizer  # save the string
		if opt_name == "Adam":
			optimizer = torch.optim.Adam(params, lr=lr)
		elif opt_name == "LBFGS":
			optimizer = torch.optim.LBFGS(
				params, lr=lr, history_size=10, line_search_fn="strong_wolfe",
				tolerance_grad=1e-32, tolerance_change=1e-32
			)
		else:
			raise ValueError(f"Unknown optimizer: {opt_name}")

		# ---- bookkeeping
		results = {'train_loss': [], 'test_loss': [], 'reg': []}
		if metrics is not None:
			for m in metrics:
				# assume metric(pred, target) -> scalar tensor
				val = m(pred_te, test_y[test_id]).detach().cpu().item()
				results[m.__name__].append(val)

		# ---- batch sizes
		Ntr = train_X.shape[0]; Nte = test_X.shape[0]
		if batch == -1 or batch > Ntr:
			batch_size = Ntr
			batch_size_test = Nte
		else:
			batch_size = int(batch)
			batch_size_test = min(int(batch), Nte)

		# ---- ID samplers
		g = torch.Generator(device='cpu')
		g.manual_seed(self.seed)

		def make_batch_ids():
			tr_id = torch.randperm(Ntr, generator=g)[:batch_size].cpu().numpy()
			te_id = torch.randperm(Nte, generator=g)[:batch_size_test].cpu().numpy()
			return tr_id, te_id

		# initialize ids for LBFGS closure
		train_id, test_id = make_batch_ids()

		# ---- closure (LBFGS)
		def closure():
			nonlocal train_id
			optimizer.zero_grad(set_to_none=True)
			pred = _safe_pred(train_X[train_id])
			tr_loss = _safe_loss(pred, train_y[train_id], loss_fn)
			reg_ = _safe_reg()
			obj = tr_loss + reg_
			_finite_or_raise("objective", obj)
			obj.backward()
			# if optimizer != 'LBFGS': # clipping interferes with LBFGS’s quasi-second-order assumptions
			#   torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
			return obj

		# ---- optional fig dir
		if save_fig and not os.path.exists(img_folder):
			os.makedirs(img_folder)

		# ---- train loop
		pbar = tqdm(range(steps), desc='description', ncols=100)
		for it in pbar:
			# restore save_act on final step if it was originally True
			if it == steps - 1:
				self.save_act = True

			# temporarily force save_act to True while plotting (then restore)
			if save_fig and it % save_fig_freq == 0:
				_save_act_before = self.save_act
				self.save_act = True

			# new minibatches each iter (LBFGS keeps current ids inside closure)
			train_id, test_id = make_batch_ids()

			# optional grid update (only on finite batch to avoid poisoning)
			if (update_grid and it % grid_update_freq == 0 and it < stop_grid_update_step
				and it >= start_grid_update_step):
				with torch.no_grad():
					probe = _safe_pred(train_X[train_id])
					if torch.isfinite(probe).all():
						self.update_grid(train_X[train_id])
					# else: skip this update silently

			# ---- step
			if opt_name == "LBFGS":
				try:
					optimizer.step(closure)
					# recompute for logging
					with torch.no_grad():
						pred_tr = _safe_pred(train_X[train_id])
						train_loss = _safe_loss(pred_tr, train_y[train_id], loss_fn_eval)
						reg_ = _safe_reg()
				except RuntimeError as e:
					raise RuntimeError("LBFGS step failed (non-finite objective/grad). "
									   "Try smaller lr post-prune, or check atoms.") from e
			else:  # Adam
				pred_tr = _safe_pred(train_X[train_id])
				train_loss = _safe_loss(pred_tr, train_y[train_id], loss_fn)
				reg_ = _safe_reg()
				loss = train_loss + reg_
				_finite_or_raise("loss", loss)
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
				optimizer.step()

			# ---- test loss
			with torch.no_grad():
				pred_te = _safe_pred(test_X[test_id])
				test_loss = _safe_loss(pred_te, test_y[test_id], loss_fn_eval)

			# ---- custom metrics
			if metrics is not None:
				for m in metrics:
					results[m.__name__].append(m().item())

			# ---- logging (safe RMSE)
			tr_rmse = float(_safe_sqrt(train_loss).detach().cpu().item())
			te_rmse = float(_safe_sqrt(test_loss).detach().cpu().item())
			reg_val = float(torch.nan_to_num(reg_, nan=1e12, posinf=1e12, neginf=1e12).detach().cpu().item())

			results['train_loss'].append(tr_rmse)
			results['test_loss'].append(te_rmse)
			results['reg'].append(reg_val)

			if it % log == 0:
				if display_metrics is None:
					pbar.set_description(f"| train_loss: {tr_rmse:.2e} | test_loss: {te_rmse:.2e} | reg: {reg_val:.2e} | ")
				else:
					string, data = '', ()
					for metric in display_metrics:
						if metric not in results:
							raise Exception(f'{metric} not recognized')
						string += f' {metric}: %.2e |'
						data += (results[metric][-1],)
					pbar.set_description(string % data)

		self.log_history('fit')
		return results

	def prune_symbolic_gates_topk(self, k: int, layers=None):
		"""
		For each GatedSymbolicLayer in act_fun, prune its symbolic gates
		to top-k per edge.
		"""
		if layers is None:
			layers = range(self.depth)

		for l in layers:
			layer = self.act_fun[l]
			if isinstance(layer, GatedSymbolicLayer):
				layer.prune_gates_topk(k=k)

	def prune_node(self, threshold=1e-2, mode="auto", active_neurons_id=None, log_history=True):
		"""
		Prune nodes (sum + mult) in a way that keeps the multiplicative
		structure and indexing consistent.

		- Decide which nodes in each layer (except input/output) to keep.
		- Convert node selection into subnode indices per layer.
		- Build a new MultKAN and slice all layers accordingly.
		"""

		if self.acts is None:
			self.get_act()

		device = self.device
		depth = self.depth  # number of connection layers

		if active_neurons_id is not None:
			mode = "manual"

		# --- 1) Node-level importance (auto mode) ---
		if mode == "auto":
			self.attribute()  # fills self.node_scores

		# --- 2) Active nodes per node-layer (0..depth) ---
		# active_nodes[l] is a LongTensor of node indices for layer l.
		active_nodes = [None] * (depth + 1)

		# input layer: keep everything (prune_input handles feature pruning)
		active_nodes[0] = torch.arange(self.width_in[0], device=device)

		for l in range(1, depth):
			n_nodes = self.width_in[l]  # = width[l][0] + width[l][1]

			if mode == "auto":
				scores = self.node_scores[l]  # [n_nodes]
				keep = scores > threshold
			else:
				keep = torch.zeros(n_nodes, dtype=torch.bool, device=device)
				keep[active_neurons_id[l - 1]] = True

			ids = torch.nonzero(keep, as_tuple=False).squeeze(-1)
			active_nodes[l] = ids

		# output layer: keep all outputs
		active_nodes[depth] = torch.arange(self.width_in[depth], device=device)

		# --- 3) New width[] and mult_arity from node selection ---
		width_new = [copy.deepcopy(self.width[0])]
		if not self.mult_homo:
			new_mult_arity = [copy.deepcopy(self.mult_arity[0])]
		else:
			new_mult_arity = None

		for l in range(1, depth):
			S_old, M_old = self.width[l]   # [#sum_nodes, #mult_nodes]
			ids = active_nodes[l]

			if ids.numel() == 0:
				sum_ids = torch.empty(0, dtype=torch.long, device=device)
				mult_ids = torch.empty(0, dtype=torch.long, device=device)
			else:
				sum_mask  = ids < S_old
				mult_mask = ~sum_mask

				sum_ids  = ids[sum_mask]            # indices 0..S_old-1
				mult_ids = ids[mult_mask] - S_old   # indices 0..M_old-1

			num_sum  = int(sum_ids.numel())
			num_mult = int(mult_ids.numel())
			width_new.append([num_sum, num_mult])

			if not self.mult_homo:
				old_ar = self.mult_arity[l]  # list length M_old
				ar_new = [old_ar[int(m.item())] for m in mult_ids]
				new_mult_arity.append(ar_new)

		# Last layer: keep original structure (we didn't prune outputs)
		width_new.append(copy.deepcopy(self.width[-1]))
		if not self.mult_homo:
			new_mult_arity.append(copy.deepcopy(self.mult_arity[-1]))

		# --- 4) Active subnodes per connection layer (0..depth-1) ---
		active_subnodes = [None] * depth

		for l in range(depth - 1):       # connection layers 0..depth-2
			layer_nodes = l + 1          # node layer whose subnodes we are selecting
			S_old, M_old = self.width[layer_nodes]

			ids_nodes = active_nodes[layer_nodes]
			ids_set = set(ids_nodes.tolist())

			out_ids = []

			# sum-node subnodes: index = node index
			for j in range(S_old):
				if j in ids_set:
					out_ids.append(j)

			# mult-node subnodes: contiguous ranges after sums
			offset = S_old
			if self.mult_homo:
				ar = self.mult_arity          # int arity
				for j_mult in range(M_old):
					node_idx = S_old + j_mult
					if node_idx in ids_set:
						out_ids.extend(range(offset, offset + ar))
					offset += ar
			else:
				for j_mult, ar in enumerate(self.mult_arity[layer_nodes]):
					node_idx = S_old + j_mult
					if node_idx in ids_set:
						out_ids.extend(range(offset, offset + ar))
					offset += ar

			active_subnodes[l] = torch.as_tensor(out_ids, dtype=torch.long, device=device)

		# Last connection layer: keep all subnodes of final layer
		active_subnodes[depth - 1] = torch.arange(self.width_out[-1], device=device)

		# Optional: masks for compatibility
		self.mask_up = []
		for l in range(depth + 1):
			m = torch.zeros(self.width_in[l], device=device)
			m[active_nodes[l]] = 1.0
			self.mask_up.append(m)

		self.mask_down = []
		for l in range(depth):
			w_out_l = self.width_out[l + 1] if l < depth - 1 else self.width_out[-1]
			m = torch.zeros(w_out_l, device=device)
			m[active_subnodes[l]] = 1.0
			self.mask_down.append(m)

		# Make sure our sub-KAN metadata is consistent with current widths
		self._cleanup_stale_chained_kan_meta()

		# --- 5) Build a new compact model and slice with get_subset ---
		model2 = MultKAN(
			copy.deepcopy(self.width),
			grid=self.grid,
			k=self.k,
			seed=self.seed,
			base_fun=self.base_fun_name,
			mult_arity=self.mult_arity,
			ckpt_path=self.ckpt_path,
			auto_save=True,
			first_init=False,
			state_id=self.state_id,
			round=self.round,
			device=self.device,
			atom_names=self.atom_names,
			noise_scale=self.noise_scale,
			numeric_atom_configs=self.numeric_atom_configs,
			chain_nodes=self.chain_nodes,
		).to(self.device)

		# --- RECONSTRUCT sub-KAN modules so state_dict keys match ---
		model2._clone_chained_kan_structure_from(self)

		# build filtered state_dict that matches model2's structure
		filtered_state = self._filtered_state_for_clone(model2)

		model2.load_state_dict(filtered_state)

		for l in range(depth):
			in_ids  = active_nodes[l]          # input nodes to this layer
			out_ids = active_subnodes[l]       # subnodes of layer (l+1)

			if l < depth - 1:
				# node params correspond to nodes in layer (l+1)
				nodes_next = active_nodes[l + 1]
				num_sum, num_mult = width_new[l + 1]

				with torch.no_grad():
					# node-scale/bias for next layer's nodes
					model2.node_bias[l] = nn.Parameter(
						model2.node_bias[l][nodes_next].detach().clone(),
						requires_grad=model2.node_bias[l].requires_grad,
					)
					model2.node_scale[l] = nn.Parameter(
						model2.node_scale[l][nodes_next].detach().clone(),
						requires_grad=model2.node_scale[l].requires_grad,
					)
					# subnode scale/bias for this layer's subnodes
					model2.subnode_bias[l] = nn.Parameter(
						model2.subnode_bias[l][out_ids].detach().clone(),
						requires_grad=model2.subnode_bias[l].requires_grad,
					)
					model2.subnode_scale[l] = nn.Parameter(
						model2.subnode_scale[l][out_ids].detach().clone(),
						requires_grad=model2.subnode_scale[l].requires_grad,
					)

				# update structural metadata
				model2.width[l + 1] = [num_sum, num_mult]
				model2.act_fun[l].out_dim_sum = num_sum
				model2.act_fun[l].out_dim_mult = num_mult
				model2.symbolic_fun[l].out_dim_sum = num_sum
				model2.symbolic_fun[l].out_dim_mult = num_mult

			# slice numeric & symbolic layers
			model2.act_fun[l] = model2.act_fun[l].get_subset(in_ids, out_ids)
			# note: slice symbolic from *self* (as in your original code)
			model2.symbolic_fun[l] = self.symbolic_fun[l].get_subset(in_ids, out_ids)

			# slice root-level gates to pruned in/out indices
			if hasattr(model2, "chained_gate_logits") and l < len(model2.chained_gate_logits):
				g = model2.chained_gate_logits[l].detach()
				# in_ids are node indices of layer l
				# out_ids are subnode indices of layer (l+1)
				g_new = g[in_ids][:, out_ids]
				model2.chained_gate_logits[l] = nn.Parameter(
					g_new, requires_grad=True
				)

		model2.cache_data = self.cache_data
		model2.acts = None
		model2.width = width_new
		if not self.mult_homo:
			model2.mult_arity = new_mult_arity

		if log_history:
			self.log_history('prune_node')
			model2.state_id += 1

		return model2

	def prune_edge(self, threshold=3e-2, log_history=True):
		'''
		pruning edges

		Args:
		-----
			threshold : float
				if the attribution score of an edge is below the threshold, it is considered dead and will be set to zero.
			
		Returns:
		--------
			pruned network : MultKAN
		'''
		if self.acts is None:
			self.get_act()

		# Ensure attribution / edge_scores are up-to-date
		if self.edge_scores is None:
			self.attribute()
		
		for i in range(len(self.width)-1):
			with torch.no_grad():
				old_mask = self.act_fun[i].mask
				new_mask = ((self.edge_scores[i] > threshold).permute(1,0) * old_mask).float()
				self.act_fun[i].mask.copy_(new_mask)
			
		if log_history:
			self.log_history('prune_edge')
	
	def prune(self, node_th=1e-2, edge_th=0, gate_top_k=0):
		if self.acts is None:
			self.get_act()
		
		# 1) node pruning
		model = self
		if node_th:
			model = model.prune_node(node_th, log_history=False)

		model.forward(model.cache_data)
		model.attribute()

		if edge_th:
			model.prune_edge(edge_th, log_history=False)

		# --- sanity check that something is left ---
		if model.n_edge == 0:
			raise RuntimeError(
				"prune() removed all active edges; model is now empty. "
				"Try using smaller node_th / edge_th, or disable pruning here."
			)

		if gate_top_k:
			model.prune_symbolic_gates_topk(k=gate_top_k)

		model.log_history('prune')
		return model
	
	def remove_edge(self, l, i, j, log_history=True):
		'''
		remove activtion phi(l,i,j) (set its mask to zero)
		'''
		with torch.no_grad():
			# numeric spline / gated numeric edge
			if hasattr(self.act_fun[l], "mask"):
				self.act_fun[l].mask[i, j] = 0.

			# symbolic branch
			if hasattr(self.symbolic_fun[l], "mask"):
				# mask shape is [out_dim, in_dim]
				self.symbolic_fun[l].mask[j, i] = 0.
			if hasattr(self.symbolic_fun[l], "funs_name"):
				self.symbolic_fun[l].funs_name[j][i] = "0"

			# multiplication / division children
			self._delete_chained_kan_edge(l, i, j)
		if log_history:
			self.log_history('remove_edge')


	def remove_node(self, l, i, mode='all', log_history=True):
		# helper to remove all chained KAN edges matching a selector
		def _drop_edges_for_node(layer_idx, cond_fn):
			if not hasattr(self, "_chained_kan_meta"):
				return
			# iterate over a copy because we'll mutate the dict
			for (L, ii, jj) in list(self._chained_kan_meta.keys()):
				if L != layer_idx:
					continue
				if cond_fn(ii, jj):
					self._delete_chained_kan_edge(L, ii, jj)

		if mode == 'down':
			# remove all edges that go into node (l, i) from layer l-1
			with torch.no_grad():
				self.act_fun[l - 1].mask[:, i] = 0.
				self.symbolic_fun[l - 1].mask[i, :].zero_()
			# any edge (l-1, in_idx -> out_idx = i)
			_drop_edges_for_node(l - 1, lambda ii, jj: jj == i)

		elif mode == 'up':
			# remove all edges that go out of node (l, i) to layer l+1
			with torch.no_grad():
				self.act_fun[l].mask[i, :] = 0.
				self.symbolic_fun[l].mask[:, i].zero_()
			# any edge (l, in_idx = i -> out_idx)
			_drop_edges_for_node(l, lambda ii, jj: ii == i)

		else:
			# both directions
			self.remove_node(l, i, mode='up',   log_history=False)
			self.remove_node(l, i, mode='down', log_history=False)

		if log_history:
			self.log_history('remove_node')
			
			
	def attribute(self, l=None, i=None, out_score=None, plot=True):
		if l is not None:
			self.attribute()
			out_score = self.node_scores[l]
	   
		if self.acts is None:
			self.get_act()

		def score_node2subnode(node_score, width, mult_arity, out_dim):

			assert np.sum(width) == node_score.shape[1]
			if isinstance(mult_arity, int):
				n_subnode = width[0] + mult_arity * width[1]
			else:
				n_subnode = width[0] + int(np.sum(mult_arity))

			#subnode_score_leaf = torch.zeros(out_dim, n_subnode).requires_grad_(True)
			#subnode_score = subnode_score_leaf.clone()
			#subnode_score[:,:width[0]] = node_score[:,:width[0]]
			subnode_score = node_score[:,:width[0]]
			if isinstance(mult_arity, int):
				#subnode_score[:,width[0]:] = node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[width[0]:].shape[0], mult_arity).reshape(out_dim,-1)
				# subnode_score = torch.cat([subnode_score, node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[:,width[0]:].shape[1], mult_arity).reshape(out_dim,-1)], dim=1)
				subnode_score = torch.cat(
					[subnode_score,
					 (node_score[:, width[0]:] / mult_arity)[:, :, None]
						 .expand(out_dim, node_score[:,width[0]:].shape[1], mult_arity)
						 .reshape(out_dim, -1)],
					dim=1
				)
			else:
				acml = width[0]
				for i, ar in enumerate(mult_arity):
					if ar <= 0:
						continue
					# divide by ar so the total importance for this mult node is preserved
					s = node_score[:, width[0] + i] / float(ar)   # [out_dim]
					subnode_score = torch.cat(
						[subnode_score,
						 s.unsqueeze(1).expand(out_dim, ar)],     # [out_dim, ar]
						dim=1
					)
					acml += ar
			return subnode_score


		node_scores = []
		subnode_scores = []
		edge_scores = []
		
		l_query = l
		if l is None:
			l_end = self.depth
		else:
			l_end = l

		# back propagate from the queried layer
		out_dim = self.width_in[l_end]
		if out_score is None:
			node_score = torch.eye(out_dim).requires_grad_(True)
		else:
			node_score = torch.diag(out_score).requires_grad_(True)
		node_scores.append(node_score)
		
		device = self.act_fun[0].grid.device

		for l in range(l_end,0,-1):

			# node to subnode 
			if isinstance(self.mult_arity, int):
				subnode_score = score_node2subnode(node_score, self.width[l], self.mult_arity, out_dim=out_dim)
			else:
				mult_arity = self.mult_arity[l]
				#subnode_score = score_node2subnode(node_score, self.width[l], mult_arity)
				subnode_score = score_node2subnode(node_score, self.width[l], mult_arity, out_dim=out_dim)

			subnode_scores.append(subnode_score)
			# subnode to edge
			#print(self.edge_actscale[l-1].device, subnode_score.device, self.subnode_actscale[l-1].device)
			edge_score = torch.einsum('ij,ki,i->kij', self.edge_actscale[l-1], subnode_score.to(device), 1/(self.subnode_actscale[l-1]+1e-4))
			edge_scores.append(edge_score)

			# edge to node
			node_score = torch.sum(edge_score, dim=1)
			node_scores.append(node_score)

		self.node_scores_all = list(reversed(node_scores))
		self.edge_scores_all = list(reversed(edge_scores))
		self.subnode_scores_all = list(reversed(subnode_scores))

		self.node_scores = [torch.mean(l, dim=0) for l in self.node_scores_all]
		self.edge_scores = [torch.mean(l, dim=0) for l in self.edge_scores_all]
		self.subnode_scores = [torch.mean(l, dim=0) for l in self.subnode_scores_all]

		# return
		if l_query is not None:
			if i is None:
				return self.node_scores_all[0]
			else:
				
				# plot
				if plot:
					in_dim = self.width_in[0]
					plt.figure(figsize=(1*in_dim, 3))
					plt.bar(range(in_dim),self.node_scores_all[0][i].cpu().detach().numpy())
					plt.xticks(range(in_dim));

				return self.node_scores_all[0][i]
			
	def node_attribute(self):
		self.node_attribute_scores = []
		for l in range(1, self.depth+1):
			node_attr = self.attribute(l)
			self.node_attribute_scores.append(node_attr)
			
	def feature_interaction(self, l, neuron_th = 1e-2, feature_th = 1e-2):
		dic = {}
		width = self.width_in[l]

		for i in range(width):
			score = self.attribute(l,i,plot=False)

			if torch.max(score) > neuron_th:
				features = tuple(torch.where(score > torch.max(score) * feature_th)[0].detach().numpy())
				if features in dic.keys():
					dic[features] += 1
				else:
					dic[features] = 1

		return dic

	def get_symbolic_choice_per_edge(self):
		choices = {}
		for l, layer in enumerate(self.act_fun):
			if isinstance(layer, GatedSymbolicLayer):
				edge_choices = layer.get_symbolic_choices()
				for (i, j), name in edge_choices.items():
					choices[(l, i, j)] = name
		return choices

	def greedy_symbolic_regression(
		self,
		data,
		*,
		optimizer="Adam",
		lib=None,
		min_edge_score=None,
		layers=None,
		weight_simple=0,
		verbose=1,
		lr=1e-3,
		steps=200,
		lamb=0,
		top_k_gates=1,
		**args
	):
		"""
		Greedy: at each iteration, fix one numeric spline edge to a symbolic atom
		(or to a composite op implemented by a chain of sub-KANs).

		- For simple atoms (e.g. 'x', 'sin', '0', ...): we call fix_symbolic().
		- For 'multiplication' / 'division' and self.chain_nodes > 0:
			* run greedy_symbolic_regression recursively on each child sub-KAN,
			* then fit the parent and use the train loss as score for this choice.
		"""

		eval_input = data["train_input"]
		device = self.device if hasattr(self, "device") else eval_input.device
		X = eval_input.to(device)

		# --- Default atom library -------------------------------------------------
		if lib is None:
			lib = list(SYMBOLIC_LIB.keys())
		# Child KANs must NOT see multiplication/division
		if not self.chain_nodes:
			lib = [f for f in lib if f not in ("multiplication", "division")]

		# --- Layers to consider ---------------------------------------------------
		Ls = list(layers or range(self.depth))

		picks = []
		i_fn = 0
		nothing_left = False

		# -------------------------------------------------------------------------
		# Small helpers to avoid repetition
		# -------------------------------------------------------------------------
		def iter_numeric_edges(layer_scores):
			"""
			Yield (l, scores, num_mask, sym_off) for layers with numeric, non-symbolic edges.
			"""
			for l in Ls:
				scores = layer_scores[l]                       # (out_dim, in_dim)
				num_mask = (self.act_fun[l].mask > 0).T        # (out_dim, in_dim)
				sym_off  = (self.symbolic_fun[l].mask == 0)    # (out_dim, in_dim)
				yield l, scores, num_mask, sym_off

		def build_edge_lib(l, i, j):
			"""
			Build edge-specific library for layer l, in-index i, out-index j.
			Handles gated layers and ensures '0', '1', and (if needed) mult/div are present.
			"""
			# --- build edge-specific library ordering using gating ---
			if isinstance(self.act_fun[l], GatedSymbolicLayer):
				gate_layer = self.act_fun[l]
				logits_edge = gate_layer.gate_logits[j, i]  # [K]
				probs_edge = torch.softmax(logits_edge, dim=-1)
				atom_order = torch.argsort(probs_edge, descending=True).tolist()

				lib_edge = [
					gate_layer.atom_names[k_idx]
					for k_idx in atom_order[:top_k_gates]
				]
				# If any numeric layer appears, fall back to global lib
				if any(x in GatedSymbolicLayer.numeric_layers for x in lib_edge):
					lib_edge = list(lib)
			else:
				lib_edge = list(lib)

			if not lib_edge:
				raise RuntimeError("No symbolic functions to map")

			# Make sure '0' and '1' exist so edges can be "removed" or identity-mapped
			if "0" not in lib_edge:
				lib_edge.append("0")
			if "1" not in lib_edge:
				lib_edge.append("1")

			# For chained KANs, ensure multiplicative ops are available
			if self.chain_nodes > 1:
				if "multiplication" not in lib_edge:
					lib_edge.append("multiplication")
				if "division" not in lib_edge:
					lib_edge.append("division")

			return lib_edge

		def try_fun_on_edge(l, i, j, fun_name, lib):
			"""
			Try mapping edge (l, i, j) to `fun_name` inside a model snapshot,
			fit, and return the final train loss.

			For simple atoms:
				- fix_symbolic() on that edge, then fit().
			For 'multiplication' / 'division' and chain_nodes > 0:
				- temporarily graft sub-KANs on this edge,
				- recursively run greedy_symbolic_regression on each child sub-KAN,
				- then fit() the parent.
			All changes are rolled back by _model_snapshot.
			"""
			with _model_snapshot(self):
				done_something = False
				if fun_name in ("multiplication", "division"):
					if self.chain_nodes > 0:
						self._init_chained_kan_edge(
							l=l,
							i=i,
							j=j,
							hidden_nodes=self.chain_nodes,
							op="mul" if fun_name == "multiplication" else "div",
							verbose=(verbose >= 2),
						)
						# recursively run greedy SR on each child sub-module
						meta = self._chained_kan_meta[(l, i, j)]
						edge_key = meta["key"]
						children = self._chained_kan_modules[edge_key]  # ModuleList
						for sub_module in children:
							sub_module.greedy_symbolic_regression(
								data,
								optimizer=optimizer,
								lib=lib,
								min_edge_score=min_edge_score,
								layers=layers,
								weight_simple=weight_simple,
								verbose=verbose,
								lr=lr,
								steps=steps,
								lamb=lamb,
								top_k_gates=top_k_gates,
							)
						done_something = True
				else:
					# Simple symbolic atom on this edge
					self.fix_symbolic(
						l,
						i,
						j,
						fun_name,
						verbose=(verbose >= 2),
						log_history=False,
					)
					done_something = True

				# Fit the (possibly augmented) model and measure train loss
				if done_something:
					results = self.fit(
						data,
						optimizer=optimizer,
						lr=lr,
						steps=steps,
						lamb=lamb,
					)
			return results["train_loss"][-1]

		def select_best_fun_for_edge(l, i, j):
			"""
			For a single edge (l, i, j), search over its edge-specific library
			and return (best_function, best_loss).
			"""
			lib_edge = build_edge_lib(l, i, j)

			if len(lib_edge) == 1:
				# Only one choice; don't bother fitting
				return lib_edge[0], None

			best_function = None
			best_loss = float("inf")

			for fun_name in lib_edge:
				loss = try_fun_on_edge(l, i, j, fun_name, lib)
				if loss < best_loss:
					best_loss = loss
					best_function = fun_name

			if not best_function:
				# Original code comments "fallback to '0'" but actually used '1'
				best_function = "1"

			return best_function, best_loss

		def zero_out_low_edges(layer_scores, threshold):
			"""
			Map all remaining numeric edges with score < threshold to '1'
			and record them in `picks`.
			"""
			for l2, scores2, num_mask2, sym_off2 in iter_numeric_edges(layer_scores):
				low_mask = (scores2 < threshold) & num_mask2 & sym_off2
				js, is_ = torch.nonzero(low_mask, as_tuple=True)

				for j2, i2 in zip(js.tolist(), is_.tolist()):
					s_edge = float(scores2[j2, i2].item())
					self.fix_symbolic(
						l2,
						int(i2),
						int(j2),
						"1",
						verbose=(verbose >= 2),
						log_history=False,
					)
					picks.append(
						{
							"l": l2,
							"i": int(i2),
							"j": int(j2),
							"fun": "1",
							"loss": None,
							"score": s_edge,
						}
					)

		# -------------------------------------------------------------------------
		# Main greedy loop
		# -------------------------------------------------------------------------
		while not nothing_left:
			i_fn += 1

			# === SCORE ALL CANDIDATE EDGES ===
			self.attribute()
			layer_scores = [s.detach().clone() for s in self.edge_scores]  # each: (out_dim, in_dim)

			# === FIND THE SINGLE BEST EDGE (global across layers) ===
			best = None  # (score, l, i, j)

			for l, scores, num_mask, sym_off in iter_numeric_edges(layer_scores):
				cand = scores.clone()
				cand[~num_mask] = -float("inf")
				cand[~sym_off] = -float("inf")

				# pick best in this layer (no threshold here)
				val = torch.max(cand)
				if torch.isfinite(val):
					j, i = torch.nonzero(cand == val, as_tuple=False)[0]  # first argmax
					s = float(val.item())
					# NOTE: this preserves your original "<" choice (not ">")
					if (best is None) or (s < best[0]):
						best = (s, l, int(i.item()), int(j.item()))

			if best is None:
				if verbose >= 1:
					print("[greedy] No eligible numeric edges remain. Stop.")
				nothing_left = True
				break

			score, l, i, j = best

			# === If best score is below min_edge_score: zero out all remaining low-score edges ===
			if (min_edge_score is not None) and (score < float(min_edge_score)):
				thr = float(min_edge_score)
				if verbose >= 1:
					print(
						f"[greedy] best score {score:.3e} < min_edge_score={thr:.3e}."
						" Mapping remaining low-score edges to '1' and stopping."
					)

				zero_out_low_edges(layer_scores, thr)
				# no more greedy steps after this
				break

			# Otherwise, proceed with standard greedy replacement for this single edge
			if verbose >= 1:
				print(
					f"[greedy] step {i_fn}: pick edge (l={l}, i={i}, j={j}) "
					f"with score={score:.3e}"
				)

			best_function, best_loss = select_best_fun_for_edge(l, i, j)

			# === Commit winner (fallback already handled) ===
			if best_function in ("multiplication", "division") and self.chain_nodes > 0:
				self._init_chained_kan_edge(
					l=l,
					i=i,
					j=j,
					hidden_nodes=self.chain_nodes,
					op="mul" if best_function == "multiplication" else "div",
					verbose=(verbose >= 2),
				)
				meta = self._chained_kan_meta[(l, i, j)]
				edge_key = meta["key"]
				children = self._chained_kan_modules[edge_key]  # ModuleList

				# recurse on children: each sub-KAN runs its own greedy SR
				info_dict = {
					f"sub_{l}_{i}_{j}": sub_module.greedy_symbolic_regression(
						data,
						optimizer=optimizer,
						lib=lib,
						min_edge_score=min_edge_score,
						layers=layers,
						weight_simple=weight_simple,
						verbose=verbose,
						lr=lr,
						steps=steps,
						lamb=lamb,
						top_k_gates=top_k_gates,
					)
					for sub_module in children
				}
			else:
				# Simple symbolic atom commit
				self.fix_symbolic(
					l,
					i,
					j,
					best_function,
					verbose=(verbose >= 2),
					log_history=False,
				)
				info_dict = {}

			info_dict.update({
				"l": l,
				"i": i,
				"j": j,
				"fun": best_function,
				"loss": best_loss,
				"score": score,
			})
			picks.append(info_dict)

			# re-fit after committing this edge
			self.fit(data, optimizer=optimizer, lr=lr, steps=steps, lamb=lamb)
			# no structural pruning: "removal" is via fun='1'

		# housekeeping
		self.log_history("greedy_symbolic_regression")
		return picks


	def _init_chained_kan_edge(
		self,
		l: int,
		i: int,
		j: int,
		op: str,                 # "mul" or "div"
		hidden_nodes: int = 1,
		verbose: bool = False,
		rewire: bool = True,     # <--- NEW
	):
		"""
		Create & graft two 1D KANs on edge (l, i -> j).

		If rewire = True:
		  - disable numeric spline on (l, i, j)
		  - set symbolic_fun fun_name to "multiplication"/"division" and enable mask
		If rewire = False:
		  - only create / register the child KAN modules and metadata;
			numeric + symbolic paths are left untouched.
		"""
		assert op in ("mul", "div")

		edge_key = f"l{l}_i{i}_j{j}"

		# --- instantiate child 1D KANs for THIS edge (if not already) ---
		if edge_key not in self._chained_kan_modules:
			SubKAN = self.__class__  # same class as the parent

			child_list = []
			for h in range(hidden_nodes):
				sub_module = SubKAN(
					width=[1, 1, 1],
					grid=self.grid,
					k=self.k,
					noise_scale=self.noise_scale,
					scale_base_mu=self.scale_base_mu,
					scale_base_sigma=self.scale_base_sigma,
					base_fun=self.base_fun_name,
					affine_trainable=self.affine_trainable,
					grid_eps=self.grid_eps,
					grid_range=self.grid_range,
					sp_trainable=self.sp_trainable,
					sb_trainable=self.sb_trainable,
					seed=self.seed + 1 + h,
					save_act=True,
					sparse_init=False,
					auto_save=False,
					first_init=False,
					ckpt_path=self.ckpt_path,
					state_id=0,
					round=0,
					device=self.device,
					atom_names=self.atom_names,
					numeric_atom_configs=self.numeric_atom_configs,
					chain_nodes=0,  # children do NOT spawn more KANs
					chain_types=None,
				)
				child_list.append(sub_module)

			self._chained_kan_modules[edge_key] = nn.ModuleList(child_list)

		self._chained_kan_meta[(l, i, j)] = {
			"op": op,
			"key": edge_key,
		}

		if verbose:
			print(f"[MultKAN] prepared chained {op} KAN on edge (l={l}, i={i}, j={j})")

		# ---- Only rewire numeric/symbolic edge if requested ----
		if rewire:
			self.act_fun[l].mask[i, j] = 0  # turn off numeric edge

			fun_name = "multiplication" if op == "mul" else "division"
			self.symbolic_fun[l].funs_name[j][i] = fun_name
			self.symbolic_fun[l].mask[j, i] = 1  # enable symbolic edge

	def _chained_kan_forward_layer(self, l: int, x: torch.Tensor, x_numerical: torch.Tensor):
		"""
		Compute additive contribution of all chained sub-KANs in layer l,
		gated by self.chained_gate_logits[l].

		Safely skips any edges whose (i, j) no longer exist after pruning.
		"""
		# no gates or no chains -> nothing to do
		if not hasattr(self, "chained_gate_logits"):
			return x_numerical
		if self.chain_nodes is None or self.chain_nodes <= 0:
			return x_numerical
		if not hasattr(self, "_chained_kan_meta") or not self._chained_kan_meta:
			return x_numerical

		gates = torch.sigmoid(self.chained_gate_logits[l])  # (in_dim_old, out_dim_old)
		# quick exit if everything is basically off
		if torch.max(gates) < 1e-6:
			return x_numerical

		batch, out_dim = x_numerical.shape
		in_dim = x.shape[1]

		contrib = torch.zeros_like(x_numerical)

		for (L, i, j), meta in self._chained_kan_meta.items():
			if L != l:
				continue

			# --- skip edges that no longer exist after pruning ---
			if i < 0 or j < 0:
				continue
			if i >= in_dim or j >= out_dim:
				# this edge's node indices were pruned / reindexed away
				continue

			# gates tensor may still have the old, larger size; guard here too
			if i >= gates.shape[0] or j >= gates.shape[1]:
				continue

			g_ij = gates[i, j]
			if g_ij.item() < 1e-6:
				continue

			edge_key = meta["key"]
			op = meta["op"]
			if not hasattr(self, "_chained_kan_modules"):
				continue
			if edge_key not in self._chained_kan_modules:
				continue

			children = self._chained_kan_modules[edge_key]  # ModuleList

			# 1D input for this edge
			z = x[:, i:i+1]  # shape (B, 1) in the good case; can be (B, 0) if i >= in_dim, but we guarded above
			if z.shape[1] == 0:
				# nothing to feed to the child KAN; this edge is effectively dead
				continue

			# each child is a small KAN: width [1,1,1], so output is (B,1)
			vals = [child(z) for child in children]  # each (B, 1)
			if len(vals) == 0:
				continue

			if op == "div":
				num = vals[0]
				den = torch.ones_like(num)
				for t in vals[1:]:
					den = den * (t + 1e-6)  # avoid exact zeros
				v = num / den
			else:  # "mul"
				v = vals[0]
				for t in vals[1:]:
					v = v * t

			# add gated contribution to output neuron j
			contrib[:, j:j+1] += g_ij * v

		return x_numerical + contrib

	def _delete_chained_kan_edge(self, l: int, i: int, j: int):
		"""
		Remove any multiplication/division sub-models and metadata attached to edge (l, i -> j).
		Safe to call even if nothing is attached.
		"""
		if not hasattr(self, "_chained_kan_meta"):
			return

		key = (l, i, j)
		meta = self._chained_kan_meta.pop(key, None)
		if meta is None:
			return

		edge_key = meta.get("key", None)
		if edge_key is None:
			return

		if hasattr(self, "_chained_kan_modules") and edge_key in self._chained_kan_modules:
			# drop the entire ModuleList of children
			del self._chained_kan_modules[edge_key]


	def symbolic_formula(
		self,
		var=None,
		normalizer=None,
		output_normalizer=None,
		simplify: bool = False,
		compact: bool = True,
	):
		"""
		Return sympy expressions for the network outputs, optionally applying
		input/output normalization and algebraic simplification.

		Supports:
			- legacy Symbolic_KANLayer
			- GatedSymbolicLayer
			- sub-KAN chains in _chained_kan_meta / _chained_kan_modules
		"""
		symbolic_acts = []
		symbolic_acts_premult = []

		# ---------- helpers ----------

		def _sf(v):
			"""Safe float -> SymFloat conversion (no NaN/inf)."""
			fv = float(v.detach().cpu())
			if math.isnan(fv) or math.isinf(fv):
				fv = 0.0
			return SymFloat(fv)

		def _safe_gate_float(val: float) -> float:
			"""Sanitize gate scalars."""
			if not math.isfinite(val):
				return 0.0
			return val

		def _chained_gate_value(l, i, j) -> float:
			"""
			Scalar gate in [0,1] for sub-KAN at edge (l, i -> j).
			If logistic gates are not defined, returns 1.0.
			"""
			if not hasattr(self, "chained_gate_logits"):
				return 1.0
			if l < 0 or l >= len(self.chained_gate_logits):
				return 1.0
			g_l = self.chained_gate_logits[l]
			if i < 0 or j < 0 or i >= g_l.shape[0] or j >= g_l.shape[1]:
				return 1.0
			gate_val = float(torch.sigmoid(g_l[i, j]).detach().cpu())
			return _safe_gate_float(gate_val)

		def _expr_has_bad(ex):
			"""Check if expression has NaN / infinities."""
			return ex.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo)

		def _subkan_expr(l, i, j, z_sym):
			"""
			Symbolic expression for a chain of sub-KANs on edge (l, i -> j),
			or None if no valid chain is attached or expression is ill-posed.
			"""
			if not hasattr(self, "_chained_kan_meta"):
				return None
			meta = self._chained_kan_meta.get((l, i, j))
			if meta is None:
				return None

			edge_key = meta.get("key")
			op_bin = meta.get("op", "mul")  # "mul" or "div"
			if not edge_key:
				return None
			if not hasattr(self, "_chained_kan_modules"):
				return None
			if edge_key not in self._chained_kan_modules:
				return None

			gate_val = _chained_gate_value(l, i, j)
			if gate_val < 1e-6:
				return None  # effectively off

			children = self._chained_kan_modules[edge_key]
			sub_exprs = []
			for sub_model in children:
				sub_formula_list, _ = sub_model.symbolic_formula(
					var=[z_sym],
					normalizer=None,
					output_normalizer=None,
					simplify=False,
					compact=False,
				)
				if sub_formula_list:
					sub_exprs.append(sub_formula_list[0])

			if not sub_exprs:
				return None

			if op_bin == "div":
				num = sub_exprs[0]
				if len(sub_exprs) == 1:
					den = SymFloat(1.0)
				else:
					den = sub_exprs[1]
					for ex_k in sub_exprs[2:]:
						den = den * ex_k
				expr = num / den
			else:  # "mul"
				expr = sub_exprs[0]
				for ex_k in sub_exprs[1:]:
					expr = expr * ex_k

			# Drop pathological sub-KAN expressions
			if _expr_has_bad(expr):
				return None

			gate_val = _safe_gate_float(gate_val)
			return SymFloat(gate_val) * expr

		def _edge_term_gated(l, i, j, x_layer):
			"""
			Symbolic contribution of edge (l, i -> j) when using GatedSymbolicLayer.
			Returns a sympy expression or None.
			"""
			gated = self.act_fun[l]
			# skip dead edges
			if gated.mask[i, j] <= 0:
				return None

			logits_ij = gated.gate_logits[j, i]  # [K]
			best_k = int(torch.argmax(logits_ij).item())
			atom_name = gated.atom_names[best_k]

			# skip purely numeric atoms in final formula
			if atom_name not in gated.base_atom_names:
				return None

			sympy_fun = SYMBOLIC_LIB[atom_name][1]

			a_t = gated.affine[j, i, best_k, 0]
			b_t = gated.affine[j, i, best_k, 1]
			c_t = gated.affine[j, i, best_k, 2]
			d_t = gated.affine[j, i, best_k, 3]

			a = _sf(a_t)
			b = _sf(b_t)
			c = _sf(c_t)
			d = _sf(d_t)

			term = d
			if abs(float(c)) > 1e-12:
				arg = a * x_layer[i] + b
				try:
					val = sympy_fun(arg)
				except Exception as e:
					print("Error in gated symbolic edge (l,i,j):", l, i, j,
						  "atom:", atom_name, e)
					return None
				term = term + c * val

			term = SymFloat(float(gated.symbolic_scale)) * term
			if _expr_has_bad(term):
				return None
			return term

		def _edge_term_symbolic(l, i, j, x_layer):
			"""
			Symbolic contribution of edge (l, i -> j) using legacy
			Symbolic_KANLayer + optional sub-KAN chains.
			"""
			# 1) sub-KAN chain if present
			z_sym = x_layer[i]
			sub_expr = _subkan_expr(l, i, j, z_sym)
			if sub_expr is not None:
				return sub_expr

			# 2) otherwise: standard symbolic edge
			a, b, c, d = [_sf(v) for v in self.symbolic_fun[l].affine[j, i]]
			sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]

			term = d
			if abs(float(c)) > 1e-12:
				arg = a * x_layer[i] + b
				try:
					val = sympy_fun(arg)
				except Exception as e:
					print("Error in symbolic edge (l,i,j):", l, i, j, e)
					return None
				term = term + c * val

			if _expr_has_bad(term):
				return None
			return term

		# ---------- 0) build input symbols ----------
		if var is None:
			x_syms = [sympy.Symbol(f"x_{ii}")
					  for ii in range(1, self.width[0][0] + 1)]
		elif isinstance(var[0], sympy.Expr):
			x_syms = list(var)
		else:
			x_syms = [sympy.symbols(v_) for v_ in var]

		x0 = x_syms

		# optional input normalization
		if normalizer is not None:
			mean = [SymFloat(float(m)) for m in normalizer[0]]
			std  = [SymFloat(float(s)) if float(s) != 0 else SymFloat(1.0)
					for s in normalizer[1]]
			x_syms = [(x_syms[i] - mean[i]) / std[i] for i in range(len(x_syms))]

		symbolic_acts.append(x_syms)

		# ---------- 1) propagate layer by layer ----------
		for l in range(len(self.width_in) - 1):
			num_sum  = self.width[l + 1][0]
			num_mult = self.width[l + 1][1]

			act_layer = self.act_fun[l]

			# op type per subnode if using KANLayer, else default: "add"
			if hasattr(act_layer, "get_op_choice"):
				op_types = act_layer.get_op_choice(hard=True)
			else:
				op_types = ["add"] * self.width_out[l + 1]

			x_layer = symbolic_acts[-1]
			y_subnodes = []

			# ---- per-subnode symbolic expression ----
			use_gated = (
				isinstance(act_layer, GatedSymbolicLayer)
				and self.symbolic_fun[l].mask.abs().sum().item() == 0
			)

			for j in range(self.width_out[l + 1]):
				op = op_types[j] if j < len(op_types) else "add"
				yj = SymFloat(1.0) if op == "mul" else SymFloat(0.0)

				for i in range(self.width_in[l]):
					if use_gated:
						term = _edge_term_gated(l, i, j, x_layer)
					else:
						term = _edge_term_symbolic(l, i, j, x_layer)

					if term is None:
						continue

					yj = yj * term if op == "mul" else yj + term

				# subnode affine
				yj = _sf(self.subnode_scale[l][j]) * yj + _sf(self.subnode_bias[l][j])

				if simplify:
					try:
						with time_limit(getattr(self, "simplify_timeout", 10.0)):
							if not _expr_has_bad(yj):
								yj = sympy.simplify(yj, ratio=1.4)
					except SimplifyTimeout:
						print(f"Simplify timed out for subnode {j}; using unsimplified yj.")

				y_subnodes.append(yj)

			symbolic_acts_premult.append(y_subnodes)

			# ---- multiplicative nodes (same logic as before) ----
			mult_nodes = []
			offset = num_sum
			for k in range(num_mult):
				if isinstance(self.mult_arity, int):
					ar = self.mult_arity
				else:
					ar = self.mult_arity[l + 1][k]

				mult_k = y_subnodes[offset]
				for t in range(1, ar):
					mult_k = mult_k * y_subnodes[offset + t]
				mult_nodes.append(mult_k)
				offset += ar

			# sum-nodes + mult-nodes form the next node layer
			y_nodes = y_subnodes[:num_sum] + mult_nodes

			# node-level affine
			for j in range(self.width_in[l + 1]):
				y_nodes[j] = self.node_scale[l][j] * y_nodes[j] + self.node_bias[l][j]

			symbolic_acts.append(y_nodes)

		# ---------- 2) optional output denormalization ----------
		if output_normalizer is not None:
			out_layer = symbolic_acts[-1]
			means = output_normalizer[0]
			stds  = output_normalizer[1]
			assert len(out_layer) == len(means) == len(stds)
			out_layer = [
				out_layer[i] * stds[i] + means[i]
				for i in range(len(out_layer))
			]
			symbolic_acts[-1] = out_layer

		# ---------- 3) store internals + return output list ----------
		self.symbolic_acts = [list(layer_exprs) for layer_exprs in symbolic_acts]
		self.symbolic_acts_premult = [
			list(layer_exprs) for layer_exprs in symbolic_acts_premult
		]

		# final outputs
		symbolic_formula_list = list(symbolic_acts[-1])

		# final safety: kill any lingering NaN / infinities
		cleaned = []
		for ex in symbolic_formula_list:
			if _expr_has_bad(ex):
				cleaned.append(SymFloat(0.0))
			else:
				cleaned.append(ex)
		symbolic_formula_list = cleaned

		if compact:
			symbolic_formula_list = list(
				map(compactify_symbolic_formula, symbolic_formula_list)
			)
		return symbolic_formula_list, x0




	def expand_depth(self):
		'''
		expand network depth, add an indentity layer to the end. For usage, please refer to tutorials interp_3_KAN_compiler.ipynb.
		
		Args:
		-----
			var : None or a list of sympy expression
				input variables
			normalizer : [mean, std]
			output_normalizer : [mean, std]
			
		Returns:
		--------
			None
		'''
		self.depth += 1

		# add kanlayer, set mask to zero
		dim_out = self.width_in[-1]
		if self.atom_names is None:
			layer = KANLayer(dim_out, dim_out, num_grids=self.grid).to(self.device)
		else:
			layer = GatedSymbolicLayer(dim_out, dim_out, atom_names=self.atom_names, numeric_atom_configs=self.numeric_atom_configs, base_activation=self.base_fun).to(self.device)
		
		# layer = KANLayer(dim_out, dim_out, num=self.grid, k=self.k)
		with torch.no_grad():
			layer.mask.zero_()
		self.act_fun.append(layer)

		self.width.append([dim_out, 0])
		if not self.mult_homo:
			self.mult_arity.append([])

		# add symbolic_kanlayer set mask to one. fun = identity on diagonal and zero for off-diagonal
		layer = Symbolic_KANLayer(dim_out, dim_out)
		with torch.no_grad():
			layer.mask += 1.

		for j in range(dim_out):
			for i in range(dim_out):
				if i == j:
					layer.fix_symbolic(i,j,'x')
				else:
					layer.fix_symbolic(i,j,'0')

		self.symbolic_fun.append(layer)

		self.node_bias.append(torch.nn.Parameter(torch.zeros(dim_out,device=self.device)).requires_grad_(self.affine_trainable))
		self.node_scale.append(torch.nn.Parameter(torch.ones(dim_out,device=self.device)).requires_grad_(self.affine_trainable))
		self.subnode_bias.append(torch.nn.Parameter(torch.zeros(dim_out,device=self.device)).requires_grad_(self.affine_trainable))
		self.subnode_scale.append(torch.nn.Parameter(torch.ones(dim_out,device=self.device)).requires_grad_(self.affine_trainable))

	def expand_width(self, layer_id, n_added_nodes, sum_bool=True, mult_arity=2):
		'''
		expand network width. For usage, please refer to tutorials interp_3_KAN_compiler.ipynb.
		
		Args:
		-----
			layer_id : int
				layer index
			n_added_nodes : init
				the number of added nodes
			sum_bool : bool
				if sum_bool == True, added nodes are addition nodes; otherwise multiplication nodes
			mult_arity : init
				multiplication arity (the number of numbers to be multiplied)
			
		Returns:
		--------
			None
		'''
		def _expand(layer_id, n_added_nodes, sum_bool=True, mult_arity=2, added_dim='out'):
			l = layer_id
			in_dim = self.symbolic_fun[l].in_dim
			out_dim = self.symbolic_fun[l].out_dim
			if sum_bool:

				if added_dim == 'out':
					new = Symbolic_KANLayer(in_dim, out_dim + n_added_nodes)
					old = self.symbolic_fun[l]
					in_id = np.arange(in_dim)
					out_id = np.arange(out_dim + n_added_nodes) 

					for j in out_id:
						for i in in_id:
							new.fix_symbolic(i,j,'0')
					with torch.no_grad():
						new.mask += 1.

					# copy functions (these are python callables / metadata — fine to assign)
					for j in out_id:
						for i in in_id:
							if j > n_added_nodes - 1:
								new.funs[j][i]                   = old.funs[j - n_added_nodes][i]
								new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j - n_added_nodes][i]
								new.funs_sympy[j][i]             = old.funs_sympy[j - n_added_nodes][i]
								new.funs_name[j][i]              = old.funs_name[j - n_added_nodes][i]
								# affine is an nn.Parameter[..., 4]; copy without breaking autograd:
								with torch.no_grad():
									new.affine[j, i].copy_(old.affine[j - n_added_nodes, i])

					# install the layers on the right device
					self.symbolic_fun[l] = new
					if self.atom_names is None:
						self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_nodes, num_grids=self.grid).to(self.device)
					else:
						self.act_fun[l] = GatedSymbolicLayer(in_dim, out_dim + n_added_nodes, atom_names=self.atom_names, base_activation=self.base_fun, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
					# self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_nodes, num=self.grid, k=self.k).to(self.device)
					with torch.no_grad():
						# zero the (learnable/buffer) mask safely
						self.act_fun[l].mask.zero_()

					def _prepend_param_1d(p: torch.nn.Parameter, head: torch.Tensor) -> torch.nn.Parameter:
						# keep dtype/device/reqgrad consistent
						head = head.to(device=p.device, dtype=p.dtype)
						new_t = torch.cat([head, p.detach()], dim=0)
						return torch.nn.Parameter(new_t, requires_grad=p.requires_grad)

					# prepend ones/zeros for the new OUT nodes
					self.node_scale[l]    = _prepend_param_1d(self.node_scale[l],    torch.ones(n_added_nodes))
					self.node_bias[l]     = _prepend_param_1d(self.node_bias[l],     torch.zeros(n_added_nodes))
					self.subnode_scale[l] = _prepend_param_1d(self.subnode_scale[l], torch.ones(n_added_nodes))
					self.subnode_bias[l]  = _prepend_param_1d(self.subnode_bias[l],  torch.zeros(n_added_nodes))

				if added_dim == 'in':
					new = Symbolic_KANLayer(in_dim + n_added_nodes, out_dim)
					old = self.symbolic_fun[l]
					in_id = np.arange(in_dim + n_added_nodes)
					out_id = np.arange(out_dim) 

					for j in out_id:
						for i in in_id:
							new.fix_symbolic(i,j,'0')
					with torch.no_grad():
						new.mask += 1.

					for j in out_id:
						for i in in_id:
							if i > n_added_nodes-1:
								new.funs[j][i] = old.funs[j][i-n_added_nodes]
								new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j][i-n_added_nodes]
								new.funs_sympy[j][i] = old.funs_sympy[j][i-n_added_nodes]
								new.funs_name[j][i] = old.funs_name[j][i-n_added_nodes]
								with torch.no_grad():
									new.affine[j, i].copy_(old.affine[j, i - n_added_nodes])

					self.symbolic_fun[l] = new
					if self.atom_names is None:
						self.act_fun[l] = KANLayer(in_dim + n_added_nodes, out_dim, num_grids=self.grid).to(self.device)
					else:
						self.act_fun[l] = GatedSymbolicLayer(in_dim + n_added_nodes, out_dim, atom_names=self.atom_names, base_activation=self.base_fun, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
					# self.act_fun[l] = KANLayer(in_dim + n_added_nodes, out_dim, num=self.grid, k=self.k)
					with torch.no_grad():
						self.act_fun[l].mask.zero_()


			else:

				if isinstance(mult_arity, int):
					mult_arity = [mult_arity] * n_added_nodes

				if added_dim == 'out':
					n_added_subnodes = np.sum(mult_arity)
					new = Symbolic_KANLayer(in_dim, out_dim + n_added_subnodes)
					old = self.symbolic_fun[l]
					in_id = np.arange(in_dim)
					out_id = np.arange(out_dim + n_added_nodes)

					for j in out_id:
						for i in in_id:
							new.fix_symbolic(i,j,'0')
					with torch.no_grad():        
						new.mask += 1.

					for j in out_id:
						for i in in_id:
							if j < out_dim:
								new.funs[j][i] = old.funs[j][i]
								new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j][i]
								new.funs_sympy[j][i] = old.funs_sympy[j][i]
								new.funs_name[j][i] = old.funs_name[j][i]
								with torch.no_grad():
									new.affine[j, i].copy_(old.affine[j, i])

					self.symbolic_fun[l] = new
					if self.atom_names is None:
						self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_subnodes, num_grids=self.grid).to(self.device)
					else:
						self.act_fun[l] = GatedSymbolicLayer(in_dim, out_dim + n_added_subnodes, atom_names=self.atom_names, base_activation=self.base_fun, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
					# self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_subnodes, num=self.grid, k=self.k)
					with torch.no_grad():
						self.act_fun[l].mask.zero_()

					# --- resize 1D parameters WITHOUT using .data ---
					def _cat_param_1d(param: torch.nn.Parameter, tail: torch.Tensor) -> torch.nn.Parameter:
						# keep dtype/device/grad setting
						t = torch.cat([param.detach(), tail.to(param.device, param.dtype)], dim=0)
						return torch.nn.Parameter(t, requires_grad=param.requires_grad)

					self.node_scale[l]   = _cat_param_1d(self.node_scale[l],   torch.ones (n_added_nodes,     device=self.device))
					self.node_bias[l]    = _cat_param_1d(self.node_bias[l],    torch.zeros(n_added_nodes,     device=self.device))
					self.subnode_scale[l]= _cat_param_1d(self.subnode_scale[l],torch.ones (n_added_subnodes,  device=self.device))
					self.subnode_bias[l] = _cat_param_1d(self.subnode_bias[l], torch.zeros(n_added_subnodes,  device=self.device))

				if added_dim == 'in':
					new = Symbolic_KANLayer(in_dim + n_added_nodes, out_dim)
					old = self.symbolic_fun[l]
					in_id = np.arange(in_dim + n_added_nodes)
					out_id = np.arange(out_dim) 

					for j in out_id:
						for i in in_id:
							new.fix_symbolic(i,j,'0')
					with torch.no_grad():
						new.mask += 1.

					for j in out_id:
						for i in in_id:
							if i < in_dim:
								new.funs[j][i] = old.funs[j][i]
								new.funs_avoid_singularity[j][i] = old.funs_avoid_singularity[j][i]
								new.funs_sympy[j][i] = old.funs_sympy[j][i]
								new.funs_name[j][i] = old.funs_name[j][i]
								with torch.no_grad():
									new.affine[j, i].copy_(old.affine[j, i])

					self.symbolic_fun[l] = new
					if self.atom_names is None:
						self.act_fun[l] = KANLayer(in_dim + n_added_nodes, out_dim, num_grids=self.grid).to(self.device)
					else:
						self.act_fun[l] = GatedSymbolicLayer(in_dim + n_added_nodes, out_dim, atom_names=self.atom_names, base_activation=self.base_fun, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
					# self.act_fun[l] = KANLayer(in_dim + n_added_nodes, out_dim, num=self.grid, k=self.k)
					with torch.no_grad():
						self.act_fun[l].mask.zero_()

		_expand(layer_id-1, n_added_nodes, sum_bool, mult_arity, added_dim='out')
		_expand(layer_id, n_added_nodes, sum_bool, mult_arity, added_dim='in')
		if sum_bool:
			self.width[layer_id][0] += n_added_nodes
		else:
			if isinstance(mult_arity, int):
				mult_arity = [mult_arity] * n_added_nodes

			self.width[layer_id][1] += n_added_nodes
			self.mult_arity[layer_id] += mult_arity
			
	def perturb(self, mag=1.0, mode='non-intrusive'):
		'''
		preturb a network. For usage, please refer to tutorials interp_3_KAN_compiler.ipynb.
		
		Args:
		-----
			mag : float
				perturbation magnitude
			mode : str
				pertubatation mode, choices = {'non-intrusive', 'all', 'minimal'}
			
		Returns:
		--------
			None
		'''
		perturb_bool = {}
		
		if mode == 'all':
			perturb_bool['aa_a'] = True
			perturb_bool['aa_i'] = True
			perturb_bool['ai'] = True
			perturb_bool['ia'] = True
			perturb_bool['ii'] = True
		elif mode == 'non-intrusive':
			perturb_bool['aa_a'] = False
			perturb_bool['aa_i'] = False
			perturb_bool['ai'] = True
			perturb_bool['ia'] = False
			perturb_bool['ii'] = True
		elif mode == 'minimal':
			perturb_bool['aa_a'] = True
			perturb_bool['aa_i'] = False
			perturb_bool['ai'] = False
			perturb_bool['ia'] = False
			perturb_bool['ii'] = False
		else:
			raise Exception('mode not recognized, valid modes are \'all\', \'non-intrusive\', \'minimal\'.')
				
		for l in range(self.depth):
			funs_name = self.symbolic_fun[l].funs_name
			for j in range(self.width_out[l+1]):
				for i in range(self.width_in[l]):
					out_array = list(np.array(self.symbolic_fun[l].funs_name)[j])
					in_array = list(np.array(self.symbolic_fun[l].funs_name)[:,i])
					out_active = len([i for i, x in enumerate(out_array) if x != "0"]) > 0
					in_active = len([i for i, x in enumerate(in_array) if x != "0"]) > 0
					dic = {True: 'a', False: 'i'}
					edge_type = dic[in_active] + dic[out_active]
					
					if l < self.depth - 1 or mode != 'non-intrusive':
					
						if edge_type == 'aa':
							if self.symbolic_fun[l].funs_name[j][i] == '0':
								edge_type += '_i'
							else:
								edge_type += '_a'

						if perturb_bool[edge_type]:
							with torch.no_grad():
								self.act_fun[l].mask[i, j] = mag
							
					if l == self.depth - 1 and mode == 'non-intrusive':
						with torch.no_grad():
							self.act_fun[l].mask[i, j] = 1
							self.act_fun[l].scale_base[i, j] = 0
							self.act_fun[l].scale_sp[i, j] = 0
						
		self.get_act(self.cache_data)
		
		self.log_history('perturb')
							
							
	def module(self, start_layer, chain):
		#chain = '[-1]->[-1,-2]->[-1]->[-1]'
		groups = chain.split('->')
		n_total_layers = len(groups)//2
		#start_layer = 0

		for l in range(n_total_layers):
			current_layer = cl = start_layer + l
			id_in = [int(i) for i in groups[2*l][1:-1].split(',')]
			id_out = [int(i) for i in groups[2*l+1][1:-1].split(',')]

			in_dim = self.width_in[cl]
			out_dim = self.width_out[cl+1]
			id_in_other = list(set(range(in_dim)) - set(id_in))
			id_out_other = list(set(range(out_dim)) - set(id_out))
			with torch.no_grad():
				self.act_fun[cl].mask[np.ix_(id_in_other,id_out)] = 0.
				self.act_fun[cl].mask[np.ix_(id_in,id_out_other)] = 0.
				self.symbolic_fun[cl].mask[np.ix_(id_out,id_in_other)] = 0.
				self.symbolic_fun[cl].mask[np.ix_(id_out_other,id_in)] = 0.
			
		self.log_history('module')
		
	def get_act(self, x=None):
		'''
		collect intermidate activations
		'''
		if isinstance(x, dict):
			x = x['train_input']
		if x is None:
			if self.cache_data is not None:
				x = self.cache_data
			else:
				raise Exception("missing input data x")
		save_act = self.save_act
		self.save_act = True
		self.forward(x)
		self.save_act = save_act    

KAN = MultKAN
