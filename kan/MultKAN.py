import torch
import torch.nn as nn
import numpy as np
from .KANLayer import KANLayer
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
from .spline import curve2coef
from .utils import SYMBOLIC_LIB, fit_params
from .hypothesis import plot_tree
import tempfile
import contextlib, signal, sympy as sp
import math
import re

def compactify_symbolic_formula(f):
	f = str(f)
	f = re.sub(r'(\d+\.\d\d\d\d)\d+', r'\1', f)
	f = re.sub(r'0\.9+\*', '', f)
	f = re.sub(r'1\.0+\*', '', f)
	f = re.sub(r'\s+[+-]\s+\d\.\d+e-[4-9]\d?', '', f)
	f = re.sub(r'\d\.\d+e-[4-9]\d?\s+[+-]\s+', '', f)
	return f

@contextlib.contextmanager
def _model_snapshot(model):
	# ---------- 1) Save weights/buffers (to CPU to save GPU mem) ----------
	state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

	# ---------- 2) Save training/eval flag + attrs you toggle ----------
	mode = model.training
	save_act0  = getattr(model, "save_act", True)
	auto_save0 = getattr(model, "auto_save", True)

	# ---------- 3) RNG states for determinism across candidates ----------
	torch_state = torch.random.get_rng_state()
	cuda_state  = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
	np_state    = np.random.get_state()
	py_state    = random.getstate()

	# ---------- 4) Cache forward-time buffers ----------
	cache_data = getattr(model, "cache_data", None)
	acts = getattr(model, "acts", [])[:]
	acts_premult = getattr(model, "acts_premult", [])[:]
	spline_preacts = getattr(model, "spline_preacts", [])[:]
	spline_postsplines = getattr(model, "spline_postsplines", [])[:]
	spline_postacts = getattr(model, "spline_postacts", [])[:]
	acts_scale = getattr(model, "acts_scale", [])[:]
	acts_scale_spline = getattr(model, "acts_scale_spline", [])[:]
	subnode_actscale = getattr(model, "subnode_actscale", [])[:]
	edge_actscale = getattr(model, "edge_actscale", [])[:]

	# ---------- 5) Track existing binary-KAN structures ----------
	had_bkm = hasattr(model, "_binary_kan_modules")
	if had_bkm:
		# we only need the keys; parameters are restored via load_state_dict
		prev_bkm_keys = set(model._binary_kan_modules.keys())
	else:
		prev_bkm_keys = None

	had_bkm_meta = hasattr(model, "_binary_kan_meta")
	if had_bkm_meta:
		prev_bkm_meta = copy.deepcopy(model._binary_kan_meta)
	else:
		prev_bkm_meta = None

	try:
		# everything inside the 'with' happens here
		yield
	finally:
		# ---------- A) Remove any *new* binary KAN modules ----------
		if hasattr(model, "_binary_kan_modules"):
			if had_bkm:
				# keep only pre-existing keys, drop new ones
				cur_keys = set(model._binary_kan_modules.keys())
				new_keys = cur_keys - prev_bkm_keys
				for k in new_keys:
					del model._binary_kan_modules[k]
			else:
				# there was no _binary_kan_modules before snapshot
				delattr(model, "_binary_kan_modules")

		# ---------- B) Restore / remove binary-KAN metadata ----------
		if had_bkm_meta:
			model._binary_kan_meta = prev_bkm_meta
		else:
			if hasattr(model, "_binary_kan_meta"):
				delattr(model, "_binary_kan_meta")

		# ---------- C) Restore weights and mode ----------
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
	model.cache_data = cache_data
	model.acts = acts
	model.acts_premult = acts_premult
	model.spline_preacts = spline_preacts
	model.spline_postsplines = spline_postsplines
	model.spline_postacts = spline_postacts
	model.acts_scale = acts_scale
	model.acts_scale_spline = acts_scale_spline
	model.subnode_actscale = subnode_actscale
	model.edge_actscale = edge_actscale

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
	def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu', atom_names=None, numeric_atom_configs=None, chain_nodes=0):

		super(MultKAN, self).__init__()

		self.seed = seed
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		### initializeing the numerical front ###

		self.act_fun = []
		self.depth = len(width) - 1
		self.chain_nodes = chain_nodes

		# multiplicative strength per layer (start at 0 = no mult influence)
		# self.mult_alpha = nn.ParameterList([
		# 	nn.Parameter(torch.tensor(0.0), requires_grad=False)
		# 	for _ in range(self.depth)
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
			
		self.grid_eps = grid_eps
		self.grid_range = grid_range

		self.atom_names = atom_names
		self.noise_scale = noise_scale
		self.numeric_atom_configs = numeric_atom_configs
			
		self._binary_kan_modules = nn.ModuleDict()
		self._binary_kan_meta = {}
		
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
					
			self.scale_base_mu = scale_base_mu
			self.scale_base_sigma = scale_base_sigma
			if self.atom_names is None:
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
					base_fun=base_fun
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

		self.grid = grid
		self.k = k
		self.base_fun = base_fun

		### initializing the symbolic front ###
		self.symbolic_fun = []
		for l in range(self.depth):
			sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l+1])
			self.symbolic_fun.append(sb_batch)

		self.symbolic_fun = nn.ModuleList(self.symbolic_fun)

		# if self.atom_names is not None:
		# 	# disable legacy symbolic branch – gating layer is the only source of nonlinearity
		# 	with torch.no_grad():
		# 		for l in range(self.depth):
		# 			self.symbolic_fun[l].mask.zero_()
				
		self.affine_trainable = affine_trainable
		self.sp_trainable = sp_trainable
		self.sb_trainable = sb_trainable
		
		self.save_act = save_act
			
		self.node_scores = None
		self.edge_scores = None
		self.subnode_scores = None
		
		self.cache_data = None
		self.acts = None
		
		self.auto_save = auto_save
		self.state_id = 0
		self.ckpt_path = ckpt_path
		self.round = round
		
		self.device = device
		self.to(device)
		
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

	def initialize_from_another_model(self, another_model, x):
		'''
		initialize from another model of the same width, but their 'grid' parameter can be different. 
		Note this is equivalent to refine() when we don't want to keep another_model
		
		Args:
		-----
			another_model : MultKAN
			x : 2D torch.float

		Returns:
		--------
			self
			
		Example
		-------
		>>> from kan import *
		>>> model1 = KAN(width=[2,5,1], grid=3)
		>>> model2 = KAN(width=[2,5,1], grid=10)
		>>> x = torch.rand(100,2)
		>>> model2.initialize_from_another_model(model1, x)
		'''
		with torch.no_grad():
			another_model(x)  # get activations
			batch = x.shape[0]

			self.initialize_grid_from_another_model(another_model, x)

			for l in range(self.depth):
				spb = self.act_fun[l]
				#spb_parent = another_model.act_fun[l]

				# spb = spb_parent
				preacts = another_model.spline_preacts[l]
				postsplines = another_model.spline_postsplines[l]
				self.act_fun[l].coef.copy_(curve2coef(preacts[:,0,:], postsplines.permute(0,2,1), spb.grid, k=spb.k))
				self.act_fun[l].scale_base.copy_(another_model.act_fun[l].scale_base)
				self.act_fun[l].scale_sp.copy_(another_model.act_fun[l].scale_sp)
				with torch.no_grad():
					self.act_fun[l].mask.copy_(another_model.act_fun[l].mask)

			for l in range(self.depth):
				self.node_bias[l].copy_(another_model.node_bias[l])
				self.node_scale[l].copy_(another_model.node_scale[l])
				
				self.subnode_bias[l].copy_(another_model.subnode_bias[l])
				self.subnode_scale[l].copy_(another_model.subnode_scale[l])

			for l in range(self.depth):
				self.symbolic_fun[l] = another_model.symbolic_fun[l]

			return self.to(self.device)
	
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

	
	def refine(self, new_grid):
		'''
		grid refinement
		
		Args:
		-----
			new_grid : init
				the number of grid intervals after refinement

		Returns:
		--------
			a refined model : MultKAN
			
		Example
		-------
		>>> from kan import *
		>>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		>>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
		>>> print(model.grid)
		>>> x = torch.rand(100,2)
		>>> model.get_act(x)
		>>> model = model.refine(10)
		>>> print(model.grid)
		checkpoint directory created: ./model
		saving model version 0.0
		5
		saving model version 0.1
		10
		'''

		model_new = MultKAN(width=self.width, 
					 grid=new_grid, 
					 k=self.k, 
					 seed=self.seed+1,
					 mult_arity=self.mult_arity, 
					 base_fun=self.base_fun_name, 
					 affine_trainable=self.affine_trainable, 
					 grid_eps=self.grid_eps, 
					 grid_range=self.grid_range, 
					 sp_trainable=self.sp_trainable,
					 sb_trainable=self.sb_trainable,
					 ckpt_path=self.ckpt_path,
					 auto_save=True,
					 first_init=False,
					 state_id=self.state_id,
					 round=self.round,
					 device=self.device,
					 atom_names=self.atom_names,
					 noise_scale=self.noise_scale,
					 numeric_atom_configs=self.numeric_atom_configs,
					 chain_nodes=self.chain_nodes)
			
		model_new.initialize_from_another_model(self, self.cache_data)
		model_new.cache_data = self.cache_data
		model_new.grid = new_grid
		
		self.log_history('refine')
		model_new.state_id += 1
		
		return model_new.to(self.device)
	
	
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
		)

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
	
	def rewind(self, model_id):
		'''
		rewind to an old version
		
		Args:
		-----
			model_id : str
				in format '{a}.{b}' where a is the round number, b is the version number in that round 

		Returns:
		--------
			MultKAN
			
		Example
		-------
		Please refer to tutorials. API 12: Checkpoint, save & load model
		''' 
		self.round += 1
		self.state_id = model_id.split('.')[-1]
		
		history_path = self.ckpt_path+'/history.txt'
		with open(history_path, 'a') as file:
			file.write(f'### Round {self.round} ###' + '\n')

		self.saveckpt(path=self.ckpt_path+'/'+f'{self.round}.{self.state_id}')
		
		print('rewind to model version '+f'{self.round-1}.{self.state_id}'+', renamed as '+f'{self.round}.{self.state_id}')

		return MultKAN.loadckpt(path=self.ckpt_path+'/'+str(model_id))
	
	
	def checkout(self, model_id):
		'''
		check out an old version
		
		Args:
		-----
			model_id : str
				in format '{a}.{b}' where a is the round number, b is the version number in that round 

		Returns:
		--------
			MultKAN
			
		Example
		-------
		Same use as rewind, although checkout doesn't change states
		''' 
		return MultKAN.loadckpt(path=self.ckpt_path+'/'+str(model_id))
	
	def update_grid_from_samples(self, x):
		'''
		update grid from samples
		
		Args:
		-----
			x : 2D torch.tensor
				inputs

		Returns:
		--------
			None
			
		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
		>>> print(model.act_fun[0].grid)
		>>> x = torch.linspace(-10,10,steps=101)[:,None]
		>>> model.update_grid_from_samples(x)
		>>> print(model.act_fun[0].grid)
		''' 
		for l in range(self.depth):
			self.get_act(x)
			self.act_fun[l].update_grid_from_samples(self.acts[l])
			
	def update_grid(self, x):
		'''
		call update_grid_from_samples. This seems unnecessary but we retain it for the sake of classes that might inherit from MultKAN
		'''
		self.update_grid_from_samples(x)

	def initialize_grid_from_another_model(self, model, x):
		'''
		initialize grid from another model
		
		Args:
		-----
			model : MultKAN
				parent model
			x : 2D torch.tensor
				inputs

		Returns:
		--------
			None
			
		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
		>>> print(model.act_fun[0].grid)
		>>> x = torch.linspace(-10,10,steps=101)[:,None]
		>>> model2 = KAN(width=[1,1], grid=10, k=3, seed=0)
		>>> model2.initialize_grid_from_another_model(model, x)
		>>> print(model2.act_fun[0].grid)
		'''
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
		self.acts_premult = []
		self.spline_preacts = []
		self.spline_postsplines = []
		self.spline_postacts = []
		self.acts_scale = []
		self.acts_scale_spline = []
		self.subnode_actscale = []
		self.edge_actscale = []
		# self.neurons_scale = []

		self.acts.append(x)  # acts shape: (batch, width[l])

		# _symbolic_enabled = self.symbolic_enabled

		for l in range(self.depth):
			
			x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
			#print(preacts, postacts_numerical, postspline)

			# ===== MultKAN / DivKAN contribution on this layer =====
			x_multkan = 0.0  # sentinel: stays float if no binary KANs on this layer
			if hasattr(self, "_binary_kan_meta") and hasattr(self, "_binary_kan_modules"):
				for (l_edge, i_edge, j_edge), meta in self._binary_kan_meta.items():
					if l_edge != l:
						continue  # this binary KAN belongs to another layer

					edge_key = meta["key"]
					if edge_key not in self._binary_kan_modules:
						continue

					child_list = self._binary_kan_modules[edge_key]  # ModuleList of sub-KANs

					# input to the 1D KANs for this edge: spline preactivation
					# preacts: [B, out_dim, in_dim] -> scalar [B, 1]
					z = preacts[:, j_edge, i_edge].unsqueeze(-1)  # [B, 1]

					vals = [child(z).squeeze(-1) for child in child_list]  # each [B]
					if len(vals) == 0:
						continue

					if isinstance(x_multkan, float):
						x_multkan = torch.zeros_like(x_numerical)

					op_bin = meta["op"]
					eps = 1e-6

					if op_bin == "div":
						# first child is numerator, rest multiply into denominator
						num = vals[0]
						if len(vals) == 1:
							den = torch.ones_like(num)
						else:
							den = vals[1]
							for v in vals[2:]:
								den = den * v
						edge_val = num / (den.abs() + eps)
					else:  # "mul" or anything else → product
						edge_val = vals[0]
						for v in vals[1:]:
							edge_val = edge_val * v

					# add this edge contribution to the j_edge output unit
					x_multkan[:, j_edge] += edge_val

			# ===== normal symbolic part =====
			if self.symbolic_fun[l].mask.abs().sum().item() > 0:
				x_symbolic, postacts_symbolic = self.symbolic_fun[l](
					x, singularity_avoiding=singularity_avoiding, y_th=y_th
				)
			else:
				x_symbolic = 0.
				postacts_symbolic = 0.

			# combine numeric + symbolic + binary-KAN contributions
			x = x_numerical + x_symbolic
			if not isinstance(x_multkan, float):
				x = x + x_multkan
			
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
				
				self.acts_scale.append((output_range / input_range))  # <-- no .detach()
				self.acts_scale_spline.append(output_range_spline / input_range)
				self.spline_preacts.append(preacts.detach())
				self.spline_postacts.append(postacts.detach())
				self.spline_postsplines.append(postspline.detach())

				self.acts_premult.append(x.detach())
			
			# multiplication
			dim_sum  = self.width[l+1][0]
			dim_mult = self.width[l+1][1]

			if self.mult_homo:
				if dim_mult > 0:
					# tail shape: [batch, dim_mult * mult_arity]
					tail = x[:, dim_sum:]                     # [B, dim_mult * mult_arity]
					B = tail.shape[0]
					arity = self.mult_arity                  # int

					# reshape to [B, dim_mult, arity]: each mult node has "arity" inputs
					tail = tail.view(B, dim_mult, arity)

					# choose epsilon (could be per-layer or global; start with small fixed value)
					eps = getattr(self, "mult_eps", 1)     # or self.mult_eps[l] if per-layer

					# reparameterized product:
					# h = ((prod_i (1 + eps * f_i)) - 1) / eps
					g      = 1.0 + eps * tail               # [B, dim_mult, arity]
					prod_g = g.prod(dim=-1)                 # [B, dim_mult]
					x_mult = (prod_g - 1.0) / eps           # [B, dim_mult]
				else:
					x_mult = None
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

			if dim_mult > 0:
				x = torch.cat([x[:, :dim_sum], x_mult], dim=1)
				# # scale multiplicative part by an annealed scalar mult_alpha[l]
				# alpha = self.mult_alpha[l]
				# x = torch.cat([x[:, :dim_sum], alpha * x_mult], dim=1)
			
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

	def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, verbose=True, random=False, log_history=True, given_params=None):
		'''
		set (l,i,j) activation to be symbolic (specified by fun_name)
		
		Args:
		-----
			l : int
				layer index
			i : int
				input neuron index
			j : int
				output neuron index
			fun_name : str
				function name
			fit_params_bool : bool
				obtaining affine parameters through fitting (True) or setting default values (False)
			a_range : tuple
				sweeping range of a
			b_range : tuple
				sweeping range of b
			verbose : bool
				If True, more information is printed.
			random : bool
				initialize affine parameteres randomly or as [1,0,1,0]
			log_history : bool
				indicate whether to log history when the function is called
		
		Returns:
		--------
			None or r2 (coefficient of determination)
			
		Example 1 
		---------
		>>> # when fit_params_bool = False
		>>> model = KAN(width=[2,5,1], grid=5, k=3)
		>>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=False)
		>>> print(model.act_fun[0].mask.reshape(2,5))
		>>> print(model.symbolic_fun[0].mask.reshape(2,5))
					
		Example 2
		---------
		>>> # when fit_params_bool = True
		>>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=1.)
		>>> x = torch.normal(0,1,size=(100,2))
		>>> model(x) # obtain activations (otherwise model does not have attributes acts)
		>>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=True)
		>>> print(model.act_fun[0].mask.reshape(2,5))
		>>> print(model.symbolic_fun[0].mask.reshape(2,5))
		'''
		if not fit_params_bool:
			r2, loss, params = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random, given_params=given_params)
		else:
			x = self.spline_preacts[l][:, j, i]   # what the edge actually uses as input
			y = self.spline_postacts[l][:, j, i]  # edge output before subnode/node affines
			# x_min, x_max, y_min, y_max = self.get_range(l, i, j)
			r2, loss, params = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, verbose=verbose, given_params=given_params)
			# if mask[i,j] == 0:
			#     r2 = - 1e8
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

		# --- NEW: if this edge had a binary KAN attached, forget its metadata ---
		if hasattr(self, "_binary_kan_meta"):
			key = (l, i, j)
			if key in self._binary_kan_meta:
				self._binary_kan_meta.pop(key)

		# NOTE: I’m *not* deleting self._binary_kan_modules here because:
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

	def get_range(self, l, i, j, verbose=True):
		'''
		Get the input range and output range of the (l,i,j) activation
		
		Args:
		-----
			l : int
				layer index
			i : int
				input neuron index
			j : int
				output neuron index
		
		Returns:
		--------
			x_min : float
				minimum of input
			x_max : float
				maximum of input
			y_min : float
				minimum of output
			y_max : float
				maximum of output
		
		Example
		-------
		>>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
		>>> x = torch.normal(0,1,size=(100,2))
		>>> model(x) # do a forward pass to obtain model.acts
		>>> model.get_range(0,0,0)
		'''
		x = self.spline_preacts[l][:, j, i]
		y = self.spline_postacts[l][:, j, i]
		x_min = torch.min(x).cpu().detach().numpy()
		x_max = torch.max(x).cpu().detach().numpy()
		y_min = torch.min(y).cpu().detach().numpy()
		y_max = torch.max(y).cpu().detach().numpy()
		if verbose:
			print('x range: [' + '%.2f' % x_min, ',', '%.2f' % x_max, ']')
			print('y range: [' + '%.2f' % y_min, ',', '%.2f' % y_max, ']')
		return x_min, x_max, y_min, y_max

	def plot(self, folder="./figures", beta=3, metric='backward', scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None, varscale=1.0):
		'''
		plot KAN
		
		Args:
		-----
			folder : str
				the folder to store pngs
			beta : float
				positive number. control the transparency of each activation. transparency = tanh(beta*l1).
			mask : bool
				If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
			mode : bool
				"supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
			scale : float
				control the size of the diagram
			in_vars: None or list of str
				the name(s) of input variables
			out_vars: None or list of str
				the name(s) of output variables
			title: None or str
				title
			varscale : float
				the size of input variables
			
		Returns:
		--------
			Figure
			
		Example
		-------
		>>> # see more interactive examples in demos
		>>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
		>>> x = torch.normal(0,1,size=(100,2))
		>>> model(x) # do a forward pass to obtain model.acts
		>>> model.plot()
		'''
		global Symbol
		
		if not self.save_act:
			print('cannot plot since data are not saved. Set save_act=True first.')
		
		# forward to obtain activations
		if self.acts is None:
			if self.cache_data is None:
				raise Exception('model hasn\'t seen any data yet.')
			self.forward(self.cache_data)
			
		if metric == 'backward':
			self.attribute()
			
		
		if not os.path.exists(folder):
			os.makedirs(folder)
		# matplotlib.use('Agg')
		depth = len(self.width) - 1
		for l in range(depth):
			w_large = 2.0
			for i in range(self.width_in[l]):
				for j in range(self.width_out[l+1]):
					rank = torch.argsort(self.acts[l][:, i])
					fig, ax = plt.subplots(figsize=(w_large, w_large))

					num = rank.shape[0]

					#print(self.width_in[l])
					#print(self.width_out[l+1])
					symbolic_mask = self.symbolic_fun[l].mask[j][i]
					numeric_mask = self.act_fun[l].mask[i][j]
					if symbolic_mask > 0. and numeric_mask > 0.:
						color = 'purple'
						alpha_mask = 1
					if symbolic_mask > 0. and numeric_mask == 0.:
						color = "red"
						alpha_mask = 1
					if symbolic_mask == 0. and numeric_mask > 0.:
						color = "black"
						alpha_mask = 1
					if symbolic_mask == 0. and numeric_mask == 0.:
						color = "white"
						alpha_mask = 0
						

					if tick == True:
						ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
						ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
						x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
						plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
						plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
					else:
						plt.xticks([])
						plt.yticks([])
					if alpha_mask == 1:
						plt.gca().patch.set_edgecolor('black')
					else:
						plt.gca().patch.set_edgecolor('white')
					plt.gca().patch.set_linewidth(1.5)
					# plt.axis('off')

					plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
					if sample == True:
						plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
					plt.gca().spines[:].set_color(color)

					plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
					plt.close()

		def score2alpha(score):
			return np.tanh(beta * score)

		
		if metric == 'forward_n':
			scores = self.acts_scale
		elif metric == 'forward_u':
			scores = self.edge_actscale
		elif metric == 'backward':
			scores = self.edge_scores
		else:
			raise Exception(f'metric = \'{metric}\' not recognized')
		
		alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
			
		# draw skeleton
		width = np.array(self.width)
		width_in = np.array(self.width_in)
		width_out = np.array(self.width_out)
		A = 1
		y0 = 0.3  # height: from input to pre-mult
		z0 = 0.1  # height: from pre-mult to post-mult (input of next layer)

		neuron_depth = len(width)
		min_spacing = A / np.maximum(np.max(width_out), 5)

		max_neuron = np.max(width_out)
		max_num_weights = np.max(width_in[:-1] * width_out[1:])
		y1 = 0.4 / np.maximum(max_num_weights, 5) # size (height/width) of 1D function diagrams
		y2 = 0.15 / np.maximum(max_neuron, 5) # size (height/width) of operations (sum and mult)

		fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * (y0+z0)))
		# fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

		# -- Transformation functions
		DC_to_FC = ax.transData.transform
		FC_to_NFC = fig.transFigure.inverted().transform
		# -- Take data coordinates and transform them to normalized figure coordinates
		DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
		
		# plot scatters and lines
		for l in range(neuron_depth):
			
			n = width_in[l]
			
			# scatters
			for i in range(n):
				plt.scatter(1 / (2 * n) + i / n, l * (y0+z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
				
			# plot connections (input to pre-mult)
			for i in range(n):
				if l < neuron_depth - 1:
					n_next = width_out[l+1]
					N = n * n_next
					for j in range(n_next):
						id_ = i * n_next + j

						symbol_mask = self.symbolic_fun[l].mask[j][i]
						numerical_mask = self.act_fun[l].mask[i][j]
						if symbol_mask == 1. and numerical_mask > 0.:
							color = 'purple'
							alpha_mask = 1.
						if symbol_mask == 1. and numerical_mask == 0.:
							color = "red"
							alpha_mask = 1.
						if symbol_mask == 0. and numerical_mask == 1.:
							color = "black"
							alpha_mask = 1.
						if symbol_mask == 0. and numerical_mask == 0.:
							color = "white"
							alpha_mask = 0.
						
						plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
						plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
							
							
			# plot connections (pre-mult to post-mult, post-mult = next-layer input)
			if l < neuron_depth - 1:
				n_in = width_out[l+1]
				n_out = width_in[l+1]
				mult_id = 0
				for i in range(n_in):
					if i < width[l+1][0]:
						j = i
					else:
						if i == width[l+1][0]:
							if isinstance(self.mult_arity,int):
								ma = self.mult_arity
							else:
								ma = self.mult_arity[l+1][mult_id]
							current_mult_arity = ma
						if current_mult_arity == 0:
							mult_id += 1
							if isinstance(self.mult_arity,int):
								ma = self.mult_arity
							else:
								ma = self.mult_arity[l+1][mult_id]
							current_mult_arity = ma
						j = width[l+1][0] + mult_id
						current_mult_arity -= 1
						#j = (i-width[l+1][0])//self.mult_arity + width[l+1][0]
					plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0+z0) + y0, (l+1) * (y0+z0)], color='black', lw=2 * scale)

					
					
			plt.xlim(0, 1)
			plt.ylim(-0.1 * (y0+z0), (neuron_depth - 1 + 0.1) * (y0+z0))


		plt.axis('off')

		for l in range(neuron_depth - 1):
			# plot splines
			n = width_in[l]
			for i in range(n):
				n_next = width_out[l + 1]
				N = n * n_next
				for j in range(n_next):
					id_ = i * n_next + j
					im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
					left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
					right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
					bottom = DC_to_NFC([0, l * (y0+z0) + y0/2 - y1])[1]
					up = DC_to_NFC([0, l * (y0+z0) + y0/2 + y1])[1]
					newax = fig.add_axes([left, bottom, right - left, up - bottom])
					# newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
					newax.imshow(im, alpha=alpha[l][j][i])
					newax.axis('off')
					
			  
			# plot sum symbols
			N = n = width_out[l+1]
			for j in range(n):
				id_ = j
				path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/sum_symbol.png"
				im = plt.imread(path)
				left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
				right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
				bottom = DC_to_NFC([0, l * (y0+z0) + y0 - y2])[1]
				up = DC_to_NFC([0, l * (y0+z0) + y0 + y2])[1]
				newax = fig.add_axes([left, bottom, right - left, up - bottom])
				newax.imshow(im)
				newax.axis('off')
				
			# plot mult symbols
			N = n = width_in[l+1]
			n_sum = width[l+1][0]
			n_mult = width[l+1][1]
			for j in range(n_mult):
				id_ = j + n_sum
				path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/mult_symbol.png"
				im = plt.imread(path)
				left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
				right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
				bottom = DC_to_NFC([0, (l+1) * (y0+z0) - y2])[1]
				up = DC_to_NFC([0, (l+1) * (y0+z0) + y2])[1]
				newax = fig.add_axes([left, bottom, right - left, up - bottom])
				newax.imshow(im)
				newax.axis('off')

		if in_vars is not None:
			n = self.width_in[0]
			for i in range(n):
				if isinstance(in_vars[i], sympy.Expr):
					plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'${latex(in_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
				else:
					plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
				
				

		if out_vars is not None:
			n = self.width_in[-1]
			for i in range(n):
				if isinstance(out_vars[i], sympy.Expr):
					plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, f'${latex(out_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
				else:
					plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, out_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')

		if title is not None:
			plt.gcf().get_axes()[0].text(0.5, (y0+z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

			
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
		
			
	def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2.,
		lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,
		start_grid_update_step=-1, stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False,
		in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
		singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', reg_type="elasticnet", display_metrics=None, gating_entropy=0.0, gating_l1=0.0):

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

		# # === NEW: train (a,b,c,d) for active symbolic edges ===
		# # We (re)establish hooks every fit() call so this works after prune/refine/etc.
		# for sb in self.symbolic_fun:
		#     # 1) Make the affine parameters learnable
		#     if hasattr(sb, "affine") and isinstance(sb.affine, torch.nn.Parameter):
		#         sb.affine.requires_grad_(True)

		#         # 2) Only update affine where symbolic mask is ON
		#         #    (mask shape: [out_dim, in_dim] -> broadcast to [out_dim, in_dim, 4])
		#         #    Remove any old hook first (if fit() is called multiple times)
		#         if hasattr(sb, "_affine_hook") and sb._affine_hook is not None:
		#             try:
		#                 sb._affine_hook.remove()
		#             except Exception:
		#                 pass

		#         def _masked_grad(grad, sb_ref=sb):
		#             # grad: [out_dim, in_dim, 4]
		#             with torch.no_grad():
		#                 m = sb_ref.mask.detach().unsqueeze(-1)  # [out_dim, in_dim, 1]
		#             return grad * m

		#         sb._affine_hook = sb.affine.register_hook(_masked_grad)

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
			return torch.nan_to_num(y, nan=1e12, posinf=1e12, neginf=1e12)

		def _safe_loss(pred, target, fn):
			l = fn(pred, target)
			return torch.nan_to_num(l, nan=1e12, posinf=1e12, neginf=1e12)

		def _safe_reg():
			if not self.save_act or lamb == 0.:
				return torch.zeros((), device=dev, dtype=dtype)

			# existing KAN regularizer (edge/node attribution + spline coeffs)
			if reg_metric == 'edge_backward':
				self.attribute()
			if reg_metric == 'node_backward':
				self.node_attribute()
			r = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, reg_type=reg_type)

			# NEW: add gating regularizer for all GatedSymbolicLayer layers
			if gating_entropy != 0.0 or gating_l1 != 0.0:
				for layer in self.act_fun:
					if isinstance(layer, GatedSymbolicLayer):
						r = r + layer.gating_regularizer(
							entropy_weight=gating_entropy,
							l1_weight=gating_l1,
						)

			return torch.nan_to_num(r, nan=1e12, posinf=1e12, neginf=1e12)

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
		params = list(self.get_params())
		# include parameters of grafted binary KANs (MultKAN / DivKAN children)
		if hasattr(self, "_binary_kan_modules"):
			for child_list in self._binary_kan_modules.values():  # each is a ModuleList
				params.extend(child_list.parameters())

		if opt == "Adam":
			optimizer = torch.optim.Adam(params, lr=lr)
		elif opt == "LBFGS":
			optimizer = torch.optim.LBFGS(
				params, lr=lr, history_size=10, line_search_fn="strong_wolfe",
				tolerance_grad=1e-32, tolerance_change=1e-32
			)
		else:
			raise ValueError(f"Unknown optimizer: {opt}")

		# ---- bookkeeping
		results = {'train_loss': [], 'test_loss': [], 'reg': []}
		if metrics is not None:
			for m in metrics:
				results[m.__name__] = []

		# ---- batch sizes
		Ntr = train_X.shape[0]; Nte = test_X.shape[0]
		if batch == -1 or batch > Ntr:
			batch_size = Ntr
			batch_size_test = Nte
		else:
			batch_size = int(batch)
			batch_size_test = min(int(batch), Nte)

		# ---- ID samplers
		rng = np.random.default_rng()
		def make_batch_ids():
			tr_id = rng.choice(Ntr, batch_size, replace=False)
			te_id = rng.choice(Nte, batch_size_test, replace=False)
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
			obj = tr_loss + lamb * reg_
			_finite_or_raise("objective", obj)
			obj.backward()
			torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
			if opt == "LBFGS":
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
				loss = train_loss + lamb * reg_
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

			# ---- figures
			if save_fig and it % save_fig_freq == 0:
				self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title=f"Step {it}", beta=beta)
				plt.savefig(os.path.join(img_folder, f'{it}.jpg'), bbox_inches='tight', dpi=200)
				plt.close()
				self.save_act = _save_act_before  # restore

		self.log_history('fit')
		return results

	def prune_symbolic_gates_topk(self, k: int, layers=None, symbolic_only: bool = True):
		"""
		For each GatedSymbolicLayer in act_fun, prune its symbolic gates
		to top-k per edge.
		"""
		if layers is None:
			layers = range(self.depth)

		for l in layers:
			layer = self.act_fun[l]
			if isinstance(layer, GatedSymbolicLayer):
				layer.prune_gates_topk(k=k, symbolic_only=symbolic_only)


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

		model2.load_state_dict(self.state_dict())

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

		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
		>>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
		>>> dataset = create_dataset(f, n_var=2)
		>>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
		>>> model = model.prune_edge()
		>>> model.plot()
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

		# --- NEW: sanity check that something is left ---
		if model.n_edge == 0:
			raise RuntimeError(
				"prune() removed all active edges; model is now empty. "
				"Try using smaller node_th / edge_th, or disable pruning here."
			)

		if gate_top_k:
			model.prune_symbolic_gates_topk(k=gate_top_k)

		model.log_history('prune')
		return model
	
	def prune_input(self, threshold=1e-2, active_inputs=None, log_history=True):
		"""
		Prune input features based on attribution or a manual list.
		Also updates any MultKAN/DivKAN child KANs on layer 0.
		"""
		if active_inputs is None:
			self.attribute()
			input_score = self.node_scores[0]
			input_mask = input_score > threshold
			print('keep:', input_mask.tolist())
			input_id = torch.where(input_mask == True)[0]
		else:
			input_id = torch.tensor(active_inputs, dtype=torch.long).to(self.device)

		# --- build new model with same hyperparams ---
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
			atom_names=self.atom_names,
			noise_scale=self.noise_scale,
			numeric_atom_configs=self.numeric_atom_configs,
			chain_nodes=self.chain_nodes,
		).to(self.device)

		# copy all parameters, including child KANs (if any)
		model2.load_state_dict(self.state_dict())

		# --- shrink first layer numeric and symbolic front ---
		model2.act_fun[0] = model2.act_fun[0].get_subset(
			input_id,
			torch.arange(self.width_out[1])
		)
		model2.symbolic_fun[0] = self.symbolic_fun[0].get_subset(
			input_id,
			torch.arange(self.width_out[1])
		)

		# --- update binary KAN metadata for layer 0 ---
		# mapping from old input index -> new input index
		input_id_cpu = input_id.detach().cpu().tolist()
		old2new = {old_i: new_i for new_i, old_i in enumerate(input_id_cpu)}

		if hasattr(model2, "_binary_kan_meta") and hasattr(model2, "_binary_kan_modules"):
			new_meta = {}
			new_modules = nn.ModuleDict()

			# first, carry over all existing modules; we'll prune/rename as needed
			for edge_key, modlist in model2._binary_kan_modules.items():
				new_modules[edge_key] = modlist

			# now rebuild meta, remapping layer-0 input indices
			for (L, i_old, j), meta in model2._binary_kan_meta.items():
				edge_key_old = meta.get("key", None)
				if edge_key_old is None:
					continue

				if L == 0:
					# input layer: check if input feature survived
					if i_old not in old2new:
						# this input feature was pruned -> drop its child KANs
						if edge_key_old in new_modules:
							del new_modules[edge_key_old]
						continue

					i_new = old2new[i_old]
					new_edge_key = f"l{L}_i{i_new}_j{j}"

					# move ModuleList under new key if necessary
					if edge_key_old in new_modules and new_edge_key != edge_key_old:
						new_modules[new_edge_key] = new_modules[edge_key_old]
						del new_modules[edge_key_old]

					new_meta[(L, i_new, j)] = {
						"op": meta.get("op", "mul"),
						"key": new_edge_key,
					}
				else:
					# layers > 0: unaffected by input pruning
					new_meta[(L, i_old, j)] = meta
					# keep modules as-is
					# (edge_key_old is already in new_modules)

			model2._binary_kan_meta = new_meta
			model2._binary_kan_modules = new_modules

		# --- fix bookkeeping ---
		model2.cache_data = self.cache_data
		model2.acts = None

		model2.width[0] = [len(input_id), 0]
		model2.input_id = input_id

		if log_history:
			self.log_history('prune_input')
			model2.state_id += 1

		return model2


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

			# MultKAN / DivKAN children
			self._delete_binary_kan_edge(l, i, j)
		if log_history:
			self.log_history('remove_edge')

	def remove_node(self, l, i, mode='all', log_history=True):
		"""
		Remove neuron (l,i) by zeroing masks of incoming/outgoing edges
		and cleaning up any attached MultKAN/DivKAN submodels.
		"""
		# helper to remove all binary KAN edges matching a selector
		def _drop_edges_for_node(layer_idx, cond_fn):
			if not hasattr(self, "_binary_kan_meta"):
				return
			# iterate over a copy because we'll mutate the dict
			for (L, ii, jj) in list(self._binary_kan_meta.keys()):
				if L != layer_idx:
					continue
				if cond_fn(ii, jj):
					self._delete_binary_kan_edge(L, ii, jj)

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
		'''
		get attribution scores

		Args:
		-----
			l : None or int
				layer index
			i : None or int
				neuron index
			out_score : None or 1D torch.float
				specify output scores
			plot : bool
				when plot = True, display the bar show
			
		Returns:
		--------
			attribution scores

		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
		>>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
		>>> dataset = create_dataset(f, n_var=3)
		>>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
		>>> model.attribute()
		>>> model.feature_score
		'''
		# output (out_dim, in_dim)
		
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
		'''
		get feature interaction

		Args:
		-----
			l : int
				layer index
			neuron_th : float
				threshold to determine whether a neuron is active
			feature_th : float
				threshold to determine whether a feature is active
			
		Returns:
		--------
			dictionary

		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
		>>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
		>>> dataset = create_dataset(f, n_var=3)
		>>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
		>>> model.attribute()
		>>> model.feature_interaction(1)
		'''
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

	def suggest_symbolic(self, l, i, j, lib=None, topk=5, verbose=True, r2_loss_fun=lambda x: np.log2(1+1e-5-x), regression_loss_fun=lambda x: 1/np.log2(x), c_loss_fun=lambda x: x, weight_simple = 0):
		r2s = []
		regression_losses = []
		cs = []
		
		if lib is None:
			symbolic_lib = SYMBOLIC_LIB
		else:
			symbolic_lib = {}
			for item in lib:
				symbolic_lib[item] = SYMBOLIC_LIB[item]

		# getting r2 and complexities
		for (name, content) in symbolic_lib.items():
			r2, regression_loss, params = self.fix_symbolic(l, i, j, name, verbose=False, log_history=False)
			# print(name, r2, params)
			if r2 == -1e8: # zero function
				r2s.append(-1e8)
			else:
				r2s.append(r2)
			regression_losses.append(regression_loss)
			self.unfix_symbolic(l, i, j, log_history=False)
			c = content[2]
			cs.append(c)

		r2s = np.array(r2s)
		regression_losses = np.array(regression_losses)
		cs = np.array(cs)
		r2_loss = r2_loss_fun(r2s).astype('float')
		regression_loss = regression_loss_fun(regression_losses).astype('float')
		cs_loss = c_loss_fun(cs)
		
		loss = weight_simple * cs_loss + (1-weight_simple) * r2_loss
			
		# --- tie-break by c_loss when losses are equal ---
		topk = min(topk, len(symbolic_lib))
		order = np.lexsort((cs_loss, regression_losses, loss))   # primary: loss, secondary: c_loss
		sorted_ids = order[:topk]
		# sorted_ids = np.argsort(loss)[:topk]
		# print(sorted_ids)

		r2s = r2s[sorted_ids][:topk]
		cs = cs[sorted_ids][:topk]
		r2_loss = r2_loss[sorted_ids][:topk]
		cs_loss = cs_loss[sorted_ids][:topk]
		loss = loss[sorted_ids][:topk]
		
		topk = np.minimum(topk, len(symbolic_lib))
		
		if verbose == True:
			# print results in a dataframe
			results = {}
			results['function'] = [list(symbolic_lib.items())[sorted_ids[i]][0] for i in range(topk)]
			results['fitting r2'] = r2s[:topk]
			results['r2 loss'] = r2_loss[:topk]
			results['complexity'] = cs[:topk]
			results['complexity loss'] = cs_loss[:topk]
			results['total loss'] = loss[:topk]

			df = pd.DataFrame(results)
			print(df)

		best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
		best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
		best_r2 = r2s[0]
		best_c = cs[0]
			
		return best_name, best_fun, best_r2, best_c

	def auto_symbolic(self, lib=None, verbose=1, weight_simple = 0, r2_threshold=0.0):
		for l in range(len(self.width_in) - 1):
			for i in range(self.width_in[l]):
				for j in range(self.width_out[l + 1]):
					if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
						print(f'skipping ({l},{i},{j}) since already symbolic')
					elif self.symbolic_fun[l].mask[j, i] == 0. and self.act_fun[l].mask[i][j] == 0.:
						_, _, params = self.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
						print(f'fixing ({l},{i},{j}) with 0')
					else:
						name, fun, r2, c = self.suggest_symbolic(l, i, j, lib=lib, verbose=False, weight_simple=weight_simple)
						if r2 >= r2_threshold:
							_, _, params = self.fix_symbolic(l, i, j, name, verbose=verbose > 1, log_history=False)
							if verbose >= 1:
								print(f'fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}, params={params}')
						else:
							print(f'For ({l},{i},{j}) the best fit was {name}, but r^2 = {r2} and this is lower than {r2_threshold}. This edge was omitted, keep training or try a different threshold.')
							
		self.log_history('auto_symbolic')

	def get_symbolic_choice_per_edge(self):
		choices = {}
		for l, layer in enumerate(self.act_fun):
			if isinstance(layer, GatedSymbolicLayer):
				edge_choices = layer.get_symbolic_choices()
				for (i, j), name in edge_choices.items():
					choices[(l, i, j)] = name
		return choices

	def auto_symbolic_robust_greedy(
		self,
		data,
		*,
		optimizer="Adam",
		lib=None,                   # list[str] atoms to try; default: all in SYMBOLIC_LIB, '0' added if missing
		min_edge_score=None,        # if top score < this, remaining edges -> '0'
		layers=None,                # None = all layers; or list[int]
		weight_simple=0,
		verbose=1,                  # 0 quiet, 1 picks only, 2 + details from suggest_symbolic
		lr=1e-3,
		steps=200,
		lamb=0,
		top_k_gates=1,
		**args
	):
		"""
		Greedy: at each iteration, fix the single most important numeric B-spline edge.
		Importance defaults to the model's own backward attribution (fast and stable).

		Behavior of `min_edge_score`:
		  - While the best edge has score >= min_edge_score:
			  -> do normal greedy symbolic replacement on that edge.
		  - Once the best edge has score < min_edge_score:
			  -> set ALL remaining eligible edges with score < min_edge_score to '0'
				 and stop.
		"""

		eval_input = data['train_input']
		device = self.device if hasattr(self, "device") else eval_input.device
		X = eval_input.to(device)

		# Default atom library
		if lib is None:
			lib = list(SYMBOLIC_LIB.keys())
		# we will ensure '0' is present in lib_edge later per edge

		# Child KANs must NOT see MultKAN/DivKAN
		if not self.chain_nodes:
			lib = [f for f in lib if f not in ("MultKAN", "DivKAN")]

		# Which layers to consider
		Ls = range(self.depth) if layers is None else layers

		picks = []
		i_fn = 0
		nothing_left = False
		while not nothing_left:
			i_fn += 1

			# === SCORE ALL CANDIDATE EDGES ===
			self.attribute()
			layer_scores = [s.detach().clone() for s in self.edge_scores]  # each: (out_dim, in_dim)

			# === FIND THE SINGLE BEST EDGE (global across layers) ===
			best = None  # (score, l, i, j)
			for l in Ls:
				scores = layer_scores[l]                       # (out_dim, in_dim)
				# numeric candidates only
				num_mask = (self.act_fun[l].mask > 0).T        # (out_dim, in_dim)
				# not already symbolic
				sym_off  = (self.symbolic_fun[l].mask == 0)    # (out_dim, in_dim)

				cand = scores.clone()
				cand[~num_mask] = -float("inf")
				cand[~sym_off]  = -float("inf")

				# pick best in this layer (no threshold here)
				val = torch.max(cand)
				if torch.isfinite(val):
					j, i = torch.nonzero(cand == val, as_tuple=False)[0]  # first argmax
					s = float(val.item())
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
					print(f"[greedy] best score {score:.3e} < min_edge_score={thr:.3e}."
						  " Mapping remaining low-score edges to '0' and stopping.")

				for l2 in Ls:
					scores2 = layer_scores[l2]
					num_mask2 = (self.act_fun[l2].mask > 0).T
					sym_off2  = (self.symbolic_fun[l2].mask == 0)

					low_mask = (scores2 < thr) & num_mask2 & sym_off2
					js, is_ = torch.nonzero(low_mask, as_tuple=True)

					for j2, i2 in zip(js.tolist(), is_.tolist()):
						s_edge = float(scores2[j2, i2].item())
						self.fix_symbolic(
							l2, int(i2), int(j2), '0',
							fit_params_bool=False,
							verbose=(verbose >= 2),
							log_history=False
						)
						picks.append({
							'l': l2,
							'i': int(i2),
							'j': int(j2),
							'fun': '0',
							'loss': None,
							'score': s_edge
						})

				# no more greedy steps after this
				break

			# Otherwise, proceed with standard greedy replacement for this single edge
			if verbose >= 1:
				print(f"[greedy] step {i_fn}: pick edge (l={l}, i={i}, j={j}) with score={score:.3e}")

			best_function = None
			best_loss = float('inf')

			# --- build edge-specific library ordering using gating ---
			if isinstance(self.act_fun[l], GatedSymbolicLayer):
				gate_layer = self.act_fun[l]
				# logits for this edge (out=j, in=i): [K+1]
				logits_edge = gate_layer.gate_logits[j, i]
				probs_edge  = torch.softmax(logits_edge, dim=-1)

				atom_order = torch.argsort(probs_edge, descending=True).tolist()
				# gated ordering of names that also exist in 'lib'
				lib_edge = [
					gate_layer.atom_names[k_idx]
					for k_idx in atom_order[:top_k_gates]
				]
				if any(map(lambda x: x in GatedSymbolicLayer.numeric_layers, lib_edge)):
					lib_edge = lib
			else:
				# fall back to global lib
				lib_edge = lib

			if not lib_edge:
				raise RuntimeError('No symbolic functions to map')

			# make sure '0' exists so edges can be "removed" symbolically
			if '0' not in lib_edge:
				lib_edge.append('0')
			if '1' not in lib_edge:
				lib_edge.append('1')
			if self.chain_nodes > 1:
				if 'MultKAN' not in lib_edge:
					lib_edge.append('MultKAN')
				if 'DivKAN' not in lib_edge:
					lib_edge.append('DivKAN')

			# --- search over this edge-specific lib ---
			if len(lib_edge) == 1:
				best_function = lib_edge[0]
				best_loss = None
			else:
				for fun_name in lib_edge:
					with _model_snapshot(self):  # snapshot-and-restore around each try
						if 'KAN' in fun_name:
							# x = self.spline_preacts[l][:, j, i]   # what the edge actually uses as input
							# this will *graft* two child KANs on this edge inside `self`
							self._init_binary_kan_edge(
								l=l,
								i=i,
								j=j,
								hidden_nodes=self.chain_nodes,
								op="mul" if fun_name == "MultKAN" else "div",
								verbose=(verbose >= 2),
							)
							self.fit(data, opt=optimizer, lr=lr, steps=steps, lamb=lamb)
						else:
							self.fix_symbolic(
								l, i, j, fun_name,
								fit_params_bool=False,
								verbose=(verbose >= 2),
								log_history=False
							)
						results = self.fit(data, opt=optimizer, lr=lr, steps=steps, lamb=lamb)
						if results['train_loss'][-1] < best_loss:
							best_loss = results['train_loss'][-1]
							best_function = fun_name

			# commit winner (fallback to '0' if anything goes wrong)
			if not best_function:
				best_function = '0'

			if 'KAN' in best_function:
				# graft child KANs for this edge *for real* (outside snapshot)
				self._init_binary_kan_edge(
					l=l,
					i=i,
					j=j,
					hidden_nodes=self.chain_nodes,
					op="mul" if best_function == "MultKAN" else "div",
					verbose=(verbose >= 2),
				)
				self.fit(data, opt=optimizer, lr=lr, steps=steps, lamb=lamb)
				meta = self._binary_kan_meta[(l, i, j)]
				edge_key = meta["key"]
				children = self._binary_kan_modules[edge_key]  # ModuleList
				picks.append({
					f'sub_{l}_{i}_{j}': sub_module.auto_symbolic_robust_greedy(
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
				})
			else:
				self.fix_symbolic(
					l, i, j, best_function,
					fit_params_bool=False,
					verbose=(verbose >= 2),
					log_history=False
				)

			picks.append({
				'l': l,
				'i': i,
				'j': j,
				'fun': best_function,
				'loss': best_loss,
				'score': score
			})

			# re-fit after committing this edge
			self.fit(data, opt=optimizer, lr=lr, steps=steps, lamb=lamb)
			# no structural pruning: "removal" is via fun='0'

		# housekeeping
		self.log_history('auto_symbolic_robust_greedy')
		return picks

	def _init_binary_kan_edge(
		self,
		l: int,
		i: int,
		j: int,
		op: str,                 # "mul" or "div"
		hidden_nodes: int = 1,
		verbose: bool = False,
	):
		"""
		Create & graft two 1D KANs on edge (l, i -> j), combined by
		multiplication (op='mul') or division (op='div').

		After this call:
		  - two child KANs are registered as submodules of `self`
		  - metadata is stored so forward() can use them
		  - the numeric spline for this edge is disabled
		  - the symbolic name at this edge is set to 'MultKAN' or 'DivKAN'
		"""
		assert op in ("mul", "div")

		# unique key for this edge
		edge_key = f"l{l}_i{i}_j{j}"

		# --- instantiate child 1D KANs for THIS edge (if not already) ---
		if edge_key not in self._binary_kan_modules:
			SubKAN = self.__class__  # same class as the parent

			child_list = []
			for h in range(hidden_nodes):
				sub_module = SubKAN(
					width=[1, 1],
					grid=self.grid,
					k=self.k,
					# mult_arity=2,
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
					first_init=True,
					ckpt_path=self.ckpt_path,
					state_id=0,
					round=0,
					device=self.device,
					atom_names=self.atom_names,
					numeric_atom_configs=self.numeric_atom_configs,
					chain_nodes=0, # children don't spawn more MultKANs
				)
				# sub_module.forward(self.cache_data)
				child_list.append(sub_module)

			self._binary_kan_modules[edge_key] = nn.ModuleList(child_list)

		# store operator + key in meta
		self._binary_kan_meta[(l, i, j)] = {
			"op": op,
			"key": edge_key,
		}
		
		if verbose:
			print(f"[MultKAN] grafted binary {op} KAN on edge (l={l}, i={i}, j={j})")

		# --- disable original numeric spline for this edge --------------
		# act_fun uses mask (in_dim, out_dim)^T in your code above
		self.act_fun[l].mask[i, j] = 0  # turn off numeric edge

		# and mark symbolic edge with a special name so that forward()
		# knows to route through the binary KAN instead.
		fun_name = "MultKAN" if op == "mul" else "DivKAN"
		self.symbolic_fun[l].funs_name[j][i] = fun_name
		self.symbolic_fun[l].mask[j, i] = 1  # enable symbolic edge

	def _delete_binary_kan_edge(self, l: int, i: int, j: int):
		"""
		Remove any MultKAN/DivKAN sub-models and metadata attached to edge (l, i -> j).
		Safe to call even if nothing is attached.
		"""
		if not hasattr(self, "_binary_kan_meta"):
			return

		key = (l, i, j)
		meta = self._binary_kan_meta.pop(key, None)
		if meta is None:
			return

		edge_key = meta.get("key", None)
		if edge_key is None:
			return

		if hasattr(self, "_binary_kan_modules") and edge_key in self._binary_kan_modules:
			# drop the entire ModuleList of children
			del self._binary_kan_modules[edge_key]


	def symbolic_formula(self, var=None, normalizer=None, output_normalizer=None, simplify=False, compact=True):
		symbolic_acts = []
		symbolic_acts_premult = []
		x = []

		def ex_round(ex1, n_digit):
			ex2 = ex1
			for a in sympy.preorder_traversal(ex1):
				if isinstance(a, sympy.Float):
					ex2 = ex2.subs(a, round(a, n_digit))
			return ex2

		if var is None:
			for ii in range(1, self.width[0][0] + 1):
				x.append(sympy.Symbol(f'x_{ii}'))
		elif isinstance(var[0], sympy.Expr):
			x = var
		else:
			x = [sympy.symbols(var_) for var_ in var]

		x0 = x

		if normalizer is not None:
			mean = [SymFloat(float(m)) for m in normalizer[0]]
			std  = [SymFloat(float(s)) if float(s) != 0 else SymFloat(1.0) for s in normalizer[1]]
			x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

		symbolic_acts.append(x)

		def _sf(v):
			fv = float(v.detach().cpu())
			if math.isnan(fv) or math.isinf(fv):
				fv = 0.0
			return SymFloat(fv)

		for l in range(len(self.width_in) - 1):
			num_sum = self.width[l + 1][0]
			num_mult = self.width[l + 1][1]

			layer = self.act_fun[l]

			# get op type per subnode if using KANLayer, else default to "add"
			if hasattr(layer, "get_op_choice"):
				op_types = layer.get_op_choice(hard=True)
			else:
				op_types = ["add"] * self.width_out[l+1]

			y = []

			# -------------------------------------------------------------
			# Per-subnode expression y_j
			# -------------------------------------------------------------
			for j in range(self.width_out[l + 1]):
				op = op_types[j] if j < len(op_types) else "add"

				# initialize according to op
				if op == "mul":
					yj = SymFloat(1.0)
				else:
					yj = SymFloat(0.0)

				# ---------------------- choose source ----------------------
				# use gated layer *only* if symbolic layer has not been fixed yet
				use_gated = (
					isinstance(layer, GatedSymbolicLayer)
					and self.symbolic_fun[l].mask.abs().sum().item() == 0
				)

				if use_gated:
					# ---------- CASE 1: use GatedSymbolicLayer ----------
					gated = layer
					O, I = gated.out_dim, gated.in_dim

					for i in range(self.width_in[l]):
						# pruning mask: skip dead edges
						if gated.mask[i, j] <= 0:
							continue

						# hard gate: pick best atom index
						logits_ij = gated.gate_logits[j, i]  # [K]
						best_k = int(torch.argmax(logits_ij).item())
						atom_name = gated.atom_names[best_k]

						# skip numeric atoms in symbolic formula
						if atom_name not in gated.base_atom_names:
							continue

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
							arg = a * x[i] + b
							try:
								val = sympy_fun(arg)
							except Exception as e:
								print('Error in gated symbolic edge (l,i,j):', l, i, j, 'atom:', atom_name, e)
								continue
							term = term + c * val

						term = SymFloat(float(gated.symbolic_scale)) * term

						if op == "mul":
							yj = yj * term
						else:
							yj = yj + term

				else:
					# ---------- CASE 2: use legacy Symbolic_KANLayer ----------
					for i in range(self.width_in[l]):
						fun_name_ij = None
						if hasattr(self.symbolic_fun[l], "funs_name"):
							fun_name_ij = self.symbolic_fun[l].funs_name[j][i]

						# --------- SPECIAL CASE: MultKAN / DivKAN edges ----------
						if fun_name_ij in ("MultKAN", "DivKAN"):
							z = x[i]  # or a fresh sympy symbol tied to this edge

							op_bin = "mul"
							edge_key = None
							if hasattr(self, "_binary_kan_meta"):
								meta = self._binary_kan_meta.get((l, i, j))
								if meta is not None:
									op_bin = meta.get("op", "mul")
									edge_key = meta.get("key", None)

							sub_exprs = []
							if edge_key is not None and hasattr(self, "_binary_kan_modules"):
								if edge_key in self._binary_kan_modules:
									children = self._binary_kan_modules[edge_key]
									for sub_model in children:
										sub_formula_list, _x0 = sub_model.symbolic_formula(
											var=[z],
											normalizer=None,
											output_normalizer=None,
											simplify=False,
											compact=False,
										)
										if len(sub_formula_list) > 0:
											sub_exprs.append(sub_formula_list[0])

							if len(sub_exprs) == 0:
								term = SymFloat(0.0)
							else:
								if op_bin == "div":
									num = sub_exprs[0]
									if len(sub_exprs) == 1:
										den = SymFloat(1.0)
									else:
										den = sub_exprs[1]
										for ex_k in sub_exprs[2:]:
											den = den * ex_k
									term = num / den
								else:
									term = sub_exprs[0]
									for ex_k in sub_exprs[1:]:
										term = term * ex_k

						else:
							# --------- NORMAL symbolic edge logic ----------
							a, b, c, d = [_sf(v) for v in self.symbolic_fun[l].affine[j, i]]
							sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]

							term = d
							if abs(float(c)) > 1e-12:
								arg = a * x[i] + b
								try:
									val = sympy_fun(arg)
								except Exception as e:
									print('Error in symbolic edge (l,i,j):', l, i, j, e)
									continue
								term = term + c * val
						# print(0, term)
						# print(1, a,b,c,d, fun_name_ij)

						# aggregate into y_j according to the node op
						if op == "mul":
							yj = yj * term
						else:
							yj = yj + term


				# subnode affine
				yj = _sf(self.subnode_scale[l][j]) * yj + _sf(self.subnode_bias[l][j])

				if simplify:
					try:
						with time_limit(getattr(self, "simplify_timeout", 10.0)):
							if not yj.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
								y.append(sympy.simplify(yj, ratio=1.4))
							else:
								y.append(yj)
					except SimplifyTimeout:
						print(f"Simplify timed out for subnode {j}; using unsimplified yj.")
						y.append(yj)
				else:
					y.append(yj)

			symbolic_acts_premult.append(y)

			# -------------------------------------------------------------
			# multiplication-node logic (unchanged)
			# -------------------------------------------------------------
			mult = []
			offset = num_sum
			for k in range(num_mult):
				if isinstance(self.mult_arity, int):
					ar = self.mult_arity
				else:
					ar = self.mult_arity[l+1][k]
				mult_k = y[offset]
				for t in range(1, ar):
					mult_k = mult_k * y[offset + t]
				mult.append(mult_k)
				offset += ar

			y = y[:num_sum] + mult

			# node affine
			for j in range(self.width_in[l+1]):
				y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]

			x = y
			symbolic_acts.append(x)

		# -------------------------------------------------------------
		# output normalizer
		# -------------------------------------------------------------
		if output_normalizer is not None:
			output_layer = symbolic_acts[-1]
			means = output_normalizer[0]
			stds = output_normalizer[1]
			assert len(output_layer) == len(means)
			assert len(output_layer) == len(stds)
			output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
			symbolic_acts[-1] = output_layer

		self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
		self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

		symbolic_formula_list = [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))]
		if compact:
			symbolic_formula_list = list(map(compactify_symbolic_formula, symbolic_formula_list))
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
			layer = GatedSymbolicLayer(dim_out, dim_out, atom_names=self.atom_names, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
		
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
						self.act_fun[l] = GatedSymbolicLayer(in_dim, out_dim + n_added_nodes, atom_names=self.atom_names, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
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
						self.act_fun[l] = GatedSymbolicLayer(in_dim + n_added_nodes, out_dim, atom_names=self.atom_names, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
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
						self.act_fun[l] = GatedSymbolicLayer(in_dim, out_dim + n_added_subnodes, atom_names=self.atom_names, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
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
						self.act_fun[l] = GatedSymbolicLayer(in_dim + n_added_nodes, out_dim, atom_names=self.atom_names, numeric_atom_configs=self.numeric_atom_configs).to(self.device)
					
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
		'''
		specify network modules
		
		Args:
		-----
			start_layer : int
				the earliest layer of the module
			chain : str
				specify neurons in the module
			
		Returns:
		--------
			None
		'''
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
		
	def tree(self, x=None, in_var=None, style='tree', sym_th=1e-3, sep_th=1e-1, skip_sep_test=False, verbose=False):
		'''
		turn KAN into a tree
		'''
		if x is None:
			x = self.cache_data
		plot_tree(self, x, in_var=in_var, style=style, sym_th=sym_th, sep_th=sep_th, skip_sep_test=skip_sep_test, verbose=verbose)

	def speed(self, compile=False):
		'''
		turn on KAN's speed mode
		'''
		self.save_act=False
		self.auto_save=False
		if compile == True:
			return torch.compile(self)
		else:
			return self
		
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
		
	def get_fun(self, l, i, j):
		'''
		get function (l,i,j)
		'''
		inputs = self.spline_preacts[l][:,j,i].cpu().detach().numpy()
		outputs = self.spline_postacts[l][:,j,i].cpu().detach().numpy()
		# they are not ordered yet
		rank = np.argsort(inputs)
		inputs = inputs[rank]
		outputs = outputs[rank]
		plt.figure(figsize=(3,3))
		plt.plot(inputs, outputs, marker="o")
		return inputs, outputs
		
		
	def history(self, k='all'):
		'''
		get history
		'''
		with open(self.ckpt_path+'/history.txt', 'r') as f:
			data = f.readlines()
			n_line = len(data)
			if k == 'all':
				k = n_line

			data = data[-k:]
			for line in data:
				print(line[:-1])
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
	
	def evaluate(self, dataset):
		with torch.no_grad():
			yhat = self.forward(dataset['test_input'], singularity_avoiding=False, y_th=1e3)
			yhat = torch.nan_to_num(yhat, nan=1e12, posinf=1e12, neginf=1e12)
			rmse = torch.sqrt(torch.mean((yhat - dataset['test_label'])**2)).item()
		return {'test_loss': rmse, 'n_edge': self.n_edge, 'n_grid': self.grid}

	
	def swap(self, l, i1, i2, log_history=True):
		"""
		Swap neurons i1 and i2 in layer l in a grad-safe way.
		"""
		# swap structural parts in child modules
		self.act_fun[l-1].swap(i1, i2, mode='out')
		self.symbolic_fun[l-1].swap(i1, i2, mode='out')
		self.act_fun[l].swap(i1, i2, mode='in')
		self.symbolic_fun[l].swap(i1, i2, mode='in')

		def swap_(param: torch.nn.Parameter, i1: int, i2: int):
			# in-place swap without breaking autograd bookkeeping
			with torch.no_grad():
				tmp = param[i1].clone()
				param[i1].copy_(param[i2])
				param[i2].copy_(tmp)

		swap_(self.node_scale[l-1],    i1, i2)
		swap_(self.node_bias[l-1],     i1, i2)
		swap_(self.subnode_scale[l-1], i1, i2)
		swap_(self.subnode_bias[l-1],  i1, i2)

		if log_history:
			self.log_history('swap')

			
	@property
	def connection_cost(self):
		
		cc = 0.
		for t in self.edge_scores:
			
			def get_coordinate(n):
				return torch.linspace(0,1,steps=n+1, device=self.device)[:n] + 1/(2*n)

			in_dim = t.shape[0]
			x_in = get_coordinate(in_dim)

			out_dim = t.shape[1]
			x_out = get_coordinate(out_dim)

			dist = torch.abs(x_in[:,None] - x_out[None,:])
			cc += torch.sum(dist * t)

		return cc
	
	def auto_swap_l(self, l):

		num = self.width_in[1]
		for i in range(num):
			ccs = []
			for j in range(num):
				self.swap(l,i,j,log_history=False)
				self.get_act()
				self.attribute()
				cc = self.connection_cost.detach().clone()
				ccs.append(cc)
				self.swap(l,i,j,log_history=False)
			j = torch.argmin(torch.tensor(ccs))
			self.swap(l,i,j,log_history=False)

	def auto_swap(self):
		'''
		automatically swap neurons such as connection costs are minimized
		'''
		depth = self.depth
		for l in range(1, depth):
			self.auto_swap_l(l)
			
		self.log_history('auto_swap')

KAN = MultKAN
