import torch
import torch.nn as nn
import numpy as np
import sympy
from .utils import *



class Symbolic_KANLayer(nn.Module):
	'''
	KANLayer class

	Attributes:
	-----------
		in_dim : int
			input dimension
		out_dim : int
			output dimension
		funs : 2D array of torch functions (or lambda functions)
			symbolic functions (torch)
		funs_avoid_singularity : 2D array of torch functions (or lambda functions) with singularity avoiding
		funs_name : 2D arry of str
			names of symbolic functions
		funs_sympy : 2D array of sympy functions (or lambda functions)
			symbolic functions (sympy)
		affine : 3D array of floats
			affine transformations of inputs and outputs
	'''
	def __init__(self, in_dim=3, out_dim=2, device='cpu'):
		'''
		initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)
		
		Args:
		-----
			in_dim : int
				input dimension
			out_dim : int
				output dimension
			device : str
				device
			
		Returns:
		--------
			self
			
		Example
		-------
		>>> sb = Symbolic_KANLayer(in_dim=3, out_dim=3)
		>>> len(sb.funs), len(sb.funs[0])
		'''
		super(Symbolic_KANLayer, self).__init__()
		self.out_dim = out_dim
		self.in_dim = in_dim
		# self.mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device=device)).requires_grad_(False)
		self.register_buffer("mask", torch.zeros(out_dim, in_dim, device=device))
		# torch
		self.funs = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
		self.funs_avoid_singularity = [[lambda x, y_th: ((), x*0.) for i in range(self.in_dim)] for j in range(self.out_dim)]
		# name
		self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
		# sympy
		self.funs_sympy = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
		### make funs_name the only parameter, and make others as the properties of funs_name?
		
		self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device))
		# c*f(a*x+b)+d
		# optional: init to [a=1,b=0,c=1,d=0] per edge
		# self.affine[..., 0] = 1.0  # a
		# self.affine[..., 1] = 0.0  # b
		# self.affine[..., 2] = 1.0  # c
		# self.affine[..., 3] = 0.0  # d
		
		self.device = device
		self.to(device)
		
	def to(self, device):
		'''
		move to device
		'''
		super(Symbolic_KANLayer, self).to(device)
		self.device = device    
		return self

	def _dump_python_state(self):
		# shallow-copy refs to callables is fine; we just need the same lists back
		return dict(
			funs=[[f for f in row] for row in self.funs],
			funs_avoid=[[f for f in row] for row in self.funs_avoid_singularity],
			funs_sympy=[[f for f in row] for row in self.funs_sympy],
			funs_name=[[s for s in row] for row in self.funs_name],
		)

	def _load_python_state(self, snap):
		self.funs                  = [[f for f in row] for row in snap["funs"]]
		self.funs_avoid_singularity= [[f for f in row] for row in snap["funs_avoid"]]
		self.funs_sympy            = [[f for f in row] for row in snap["funs_sympy"]]
		self.funs_name             = [[s for s in row] for row in snap["funs_name"]]

	def _rebuild_callables_from_names(self):
		# Useful when loading from YAML/funs_name
		for j in range(self.out_dim):
			for i in range(self.in_dim):
				name = self.funs_name[j][i]
				fun, fun_sympy, _, fun_avoid = SYMBOLIC_LIB[name]
				self.funs[j][i] = fun
				self.funs_sympy[j][i] = fun_sympy
				self.funs_avoid_singularity[j][i] = fun_avoid

	
	def forward(self, x, singularity_avoiding=False, y_th=10.):
		'''
		forward
		
		Args:
		-----
			x : 2D array
				inputs, shape (batch, input dimension)
			singularity_avoiding : bool
				if True, funs_avoid_singularity is used; if False, funs is used. 
			y_th : float
				the singularity threshold
			
		Returns:
		--------
			y : 2D array
				outputs, shape (batch, output dimension)
			postacts : 3D array
				activations after activation functions but before being summed on nodes
		
		Example
		-------
		>>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
		>>> x = torch.normal(0,1,size=(100,3))
		>>> y, postacts = sb(x)
		>>> y.shape, postacts.shape
		(torch.Size([100, 5]), torch.Size([100, 5, 3]))
		'''
		
		batch = x.shape[0]
		postacts = []

		_affine = torch.nan_to_num(self.affine, nan=0, posinf=0, neginf=0)
		for i in range(self.in_dim):
			postacts_ = []
			for j in range(self.out_dim):
				a, b, c, d = _affine[j,i]
				if abs(float(c)) <= 1e-12:
					xij = torch.zeros_like(x[:,[i]]) + d
				else:
					if singularity_avoiding and self.funs_avoid_singularity[j][i] is not None:
						xij = c*self.funs_avoid_singularity[j][i](a*x[:,[i]]+ b, torch.tensor(y_th))[1] + d
					else:
						xij = c*self.funs[j][i](a*x[:,[i]] + b) + d
				postacts_.append(self.mask[j][i]*xij)
			postacts.append(torch.stack(postacts_))

		postacts = torch.stack(postacts)
		postacts = postacts.permute(2,1,0,3)[:,:,:,0]
		y = torch.sum(postacts, dim=2)
		
		return y, postacts
		
		
	def get_subset(self, in_id, out_id):
		sbb = Symbolic_KANLayer(in_dim=len(in_id), out_dim=len(out_id), device=self.device)

		# tensors/buffers
		with torch.no_grad():
			sbb.mask.copy_(self.mask[out_id][:, in_id])
			sbb.affine.copy_(self.affine[out_id][:, in_id])

		# python-side callables and names
		sbb.funs              = [[self.funs[j][i]                   for i in in_id] for j in out_id]
		sbb.funs_avoid_singularity = [[self.funs_avoid_singularity[j][i] for i in in_id] for j in out_id]
		sbb.funs_sympy        = [[self.funs_sympy[j][i]             for i in in_id] for j in out_id]
		sbb.funs_name         = [[self.funs_name[j][i]              for i in in_id] for j in out_id]
		return sbb

	def old_fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10,10), b_range=(-10,10), verbose=True):
		'''
		fix an activation function to be symbolic
		
		Args:
		-----
			i : int
				the id of input neuron
			j : int 
				the id of output neuron
			fun_name : str
				the name of the symbolic functions
			x : 1D array
				preactivations
			y : 1D array
				postactivations
			a_range : tuple
				sweeping range of a
			b_range : tuple
				sweeping range of a
			verbose : bool
				print more information if True
			
		Returns:
		--------
			r2 (coefficient of determination)
			
		Example 1
		---------
		>>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
		>>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
		>>> sb.fix_symbolic(2,1,'sin')
		>>> print(sb.funs_name)
		>>> print(sb.affine)
		
		Example 2
		---------
		>>> # when x & y are provided, old_fit_params() is called to find the best fit coefficients
		>>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
		>>> batch = 100
		>>> x = torch.linspace(-1,1,steps=batch)
		>>> noises = torch.normal(0,1,(batch,)) * 0.02
		>>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
		>>> sb.fix_symbolic(2,1,'sin',x,y)
		>>> print(sb.funs_name)
		>>> print(sb.affine[1,2,:].data)
		'''
		if isinstance(fun_name,str):
			fun = SYMBOLIC_LIB[fun_name][0]
			fun_sympy = SYMBOLIC_LIB[fun_name][1]
			fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
			self.funs_sympy[j][i] = fun_sympy
			self.funs_name[j][i] = fun_name
			
			if x == None or y == None:
				#initialzie from just fun
				self.funs[j][i] = fun
				self.funs_avoid_singularity[j][i] = fun_avoid_singularity
				if random == False:
					self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.], device=self.device)
				else:
					self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
				return None
			else:
				#initialize from x & y and fun
				params, r2 = old_fit_params(x,y,fun, a_range=a_range, b_range=b_range, verbose=verbose, device=self.device)
				self.funs[j][i] = fun
				self.funs_avoid_singularity[j][i] = fun_avoid_singularity
				self.affine.data[j][i] = params
				return r2
		else:
			# if fun_name itself is a function
			fun = fun_name
			fun_sympy = fun_name
			self.funs_sympy[j][i] = fun_sympy
			self.funs_name[j][i] = "anonymous"

			self.funs[j][i] = fun
			self.funs_avoid_singularity[j][i] = fun
			if random == False:
				self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.], device=self.device)
			else:
				self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
			return None
	
	def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, verbose=True, given_params=None):
		if isinstance(fun_name,str):
			fun = SYMBOLIC_LIB[fun_name][0]
			fun_sympy = SYMBOLIC_LIB[fun_name][1]
			fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
			self.funs_sympy[j][i] = fun_sympy
			self.funs_name[j][i] = fun_name

			if given_params is not None:
				with torch.no_grad():
					self.affine[j, i].copy_(torch.as_tensor(given_params.detach(), device=self.affine.device, dtype=self.affine.dtype))
			
			if x == None or y == None:
				#initialzie from just fun
				self.funs[j][i] = fun
				self.funs_avoid_singularity[j][i] = fun_avoid_singularity
				with torch.no_grad():
					if random == False:
						params = torch.tensor([1.,0.,1.,0.], device=self.device)
						self.affine[j, i].copy_(params)
					else:
						params = torch.rand(4, device=self.device) * 2 - 1
						self.affine[j, i].copy_(params)
				return None, None, params
			else:
				self.funs[j][i] = fun
				self.funs_avoid_singularity[j][i] = fun_avoid_singularity
				#initialize from x & y and fun
				try:
					params, r2, loss = fit_params(x,y,fun, verbose=verbose, device=self.device)
					print(f'Fitting {fun_name}: R2={r2}, loss={loss}, params={params}')
					params = torch.nan_to_num(params, nan=0.0, posinf=1e12, neginf=-1e12)
					# if torch.abs(params[0]) < 1e-6 or torch.abs(params[2]) < 1e-6:
					#     raise Exception(f'Params are too low for {fun_name} (r2: {r2}): {params}')
					with torch.no_grad():
						self.affine[j, i].copy_(params.detach())
				except Exception as e:
					with torch.no_grad():
						if random == False:
							params = torch.tensor([1.,0.,1.,0.], device=self.device)
							self.affine[j, i].copy_(params)
						else:
							params = torch.rand(4, device=self.device) * 2 - 1
							self.affine[j, i].copy_(params)
					r2 = -1e8
					print(f'Cannot run {fun_name}:',e)
				return r2, loss, params
		else:
			# if fun_name itself is a function
			fun = fun_name
			fun_sympy = fun_name
			self.funs_sympy[j][i] = fun_sympy
			self.funs_name[j][i] = "anonymous"

			self.funs[j][i] = fun
			self.funs_avoid_singularity[j][i] = fun
			with torch.no_grad():
				if random == False:
					params = torch.tensor([1.,0.,1.,0.], device=self.device)
					self.affine[j, i].copy_(params)
				else:
					params = torch.rand(4, device=self.device) * 2 - 1
					self.affine[j, i].copy_(params)
			return None, None, params
		
	def swap(self, i1, i2, mode = 'in'):
		"""
		Swap input columns (mode='in') or output rows (mode='out') for all symbolic structures.
		Tables are shaped [out_dim][in_dim]; tensors are [out_dim, in_dim, ...].
		"""
		if i1 == i2:
			return
		if mode not in ('in', 'out'):
			raise ValueError("mode must be 'in' or 'out'")

		J, I = self.out_dim, self.in_dim
		n = I if mode == 'in' else J
		if not (0 <= i1 < n and 0 <= i2 < n):
			raise IndexError(f"swap indices out of range for mode={mode}: {i1}, {i2}, n={n}")

		# --- swap Python tables (list-of-lists) ---
		# All are [out][in]
		def _swap_cols(tbl):
			for j in range(J):
				tbl[j][i1], tbl[j][i2] = tbl[j][i2], tbl[j][i1]

		def _swap_rows(tbl):
			tbl[i1], tbl[i2] = tbl[i2], tbl[i1]

		tables = [self.funs, self.funs_name, self.funs_sympy, self.funs_avoid_singularity]
		if mode == 'in':
			for tbl in tables:
				_swap_cols(tbl)
			dim = 1
		else:
			for tbl in tables:
				_swap_rows(tbl)
			dim = 0

		with torch.no_grad():
			def _swap_tensor(t):
				idx = torch.arange(t.size(dim), device=t.device)
				idx[i1], idx[i2] = idx[i2].clone(), idx[i1].clone()
				t.copy_(t.index_select(dim, idx))  # preserve storage/param identity

			_swap_tensor(self.affine)
			_swap_tensor(self.mask)

