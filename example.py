import pickle # for saving model class
import time # for timing
import itertools as it # for smarter loops
from types import SimpleNamespace # for simpler dicts
import numba # for optimization
import numpy as np # for numerical data

from vfi import linear_interp as linear_interp # for linear interpolation
from vfi import optimize_1d as optimize_1d # for optimization

############
# 1. model #
############

# fundamental model class

class model():

	# called when an instance is created, e.g. model = example.model('name')
	def __init__(self,name='baseline',load=False):
		
		self.name = name
		if load:
			self.load(name)
		else:
			self.par,self.sol = self.setup()

	def setup(self):

		# a. model parameter dictionary
		par = SimpleNamespace()

		settings(par)
		create(par)
		
		# b. solution dictionary
		sol = dict()

		print('model setup done')
		return par,sol

	def save(self): # save the model parameters and solution
		
		# a. parameters
		tic = time.time()
		with open(f'data/{self.name}_par.file', 'wb') as f:
			pickle.dump(self.par, f, pickle.HIGHEST_PROTOCOL)
		toc = time.time()
		print(f'parameters saved in {toc-tic:.2f} secs')

		# b. solution
		tic = time.time()
		np.savez(f'data/{self.name}.npz', **(self.sol))
		toc = time.time()		
		print(f'solution saved in {toc-tic:.2f} secs')

	def load(self,name): # load the model parameters and solution

		# a. parameters
		tic = time.time()
		with open(f'data/{name}_par.file', 'rb') as f:
			self.par = pickle.load(f)
		toc = time.time()
		print(f'parameters loaded in {toc-tic:.2f} secs')

		# b. solution
		self.sol = dict()
		with np.load(f'data/{self.name}.npz') as data:
			for key in data.files:
				self.sol[key] = data[key]
		print(f'solution loaded in {toc-tic:.2f} secs')

	def solve(self,do_print=True):

		self.par.do_print = do_print
		allocate_sol(self.par,self.sol)
		solve(self.par,self.sol)

############
# 2. setup #
############

# define the model parameters
def settings(par):

	par.T = 20 # planning horizon

	# a. preferences
	par.beta = 0.96 # discount factor

	par.phi = 0.10 # weight on durables
	par.gamma = 2 # substitution
	par.rho = 2 # curvature
	par.bubar = 0.1 # floor on durables

	par.varphi = 1 # strenght of bequest motive
	par.qubar = 1 # luxiness of bequst motive
	par.vartheta = 2 # curvature
	
	# b. budget
	par.r = 0.03 # interest rate
	par.delta = 0.1 # depreciation of durables
	par.tau = 0.1 # transaction cost
	par.omega = 1.0 # wage rate
	par.l_set = [0.5,1.0,1.5] # labor productivity
	par.trans_prob_l = [[0.80,0.15,0.05],
						[0.10,0.80,0.10],
						[0.05,0.15,0.80]] # transition probabilities for l

	# c. grids (length, minimum, maximum, curvature)
	par.Nm = 100
	par.m_min = 0
	par.m_max = 12
	par.m_phi = 1.2

	par.Nn = 80
	par.n_min = 0
	par.n_max = 8
	par.n_phi = 1.2

	par.Na = 200
	par.a_min = 0
	par.a_max = 10
	par.a_phi = 1.2

	par.Nb = 100
	par.b_min = 0
	par.b_max = 8
	par.b_phi = 1.2

	par.Nx = 150
	par.x_min = 0
	par.x_max = 15
	par.x_phi = 1.2

# create grids and make various assertions
def create(par):

	# a. grids
	def nonlinspace(min_val,max_val,num,phi): # phi up, more points close to min_val
		x = np.zeros(num)
		x[0] = min_val
		for i in range(1,num):
			x[i] = x[i-1] + (max_val-x[i-1]) / (num-i)**phi
		return x

	par.grid_m = nonlinspace(par.m_min,par.m_max,par.Nm,par.m_phi)
	par.grid_n = nonlinspace(par.n_min,par.n_max,par.Nn,par.n_phi)
	par.grid_a = nonlinspace(par.a_min,par.a_max,par.Na,par.a_phi)
	par.grid_b = nonlinspace(par.b_min,par.b_max,par.Nb,par.b_phi)
	par.grid_x = nonlinspace(par.x_min,par.x_max,par.Nx,par.x_phi)

	# b. labor productivity
	par.Nl = len(par.l_set)
	par.trans_prob_l = np.array(par.trans_prob_l)
	assert np.all(np.sum(par.trans_prob_l,axis=1) == 1) # probabilities must sum to one in each row

	# c. utility functions (assertions)
	assert par.phi > 0 and par.phi < 1
	assert par.gamma > 1
	assert par.rho > 1
	assert par.qubar > 0
	assert par.varphi > 0
	assert par.vartheta > 1

	return par

def allocate_sol(par,sol):

	par.w_shape = (par.Nl,par.Nb,par.Na)
	sol['inv_w'] = np.zeros((par.T,*par.w_shape))
	
	par.keep_shape = (par.Nl,par.Nn,par.Nm)
	sol['inv_v_keep'] = np.zeros((par.T,*par.keep_shape))
	sol['c_keep'] = np.zeros((par.T,*(par.keep_shape)))

	par.adj_shape = (par.Nl,par.Nx)
	sol['inv_v_adj'] = np.zeros((par.T,*par.adj_shape))
	sol['b_adj'] = np.zeros((par.T,*par.adj_shape))

##########################
# 3. auxilarry functions #
##########################

# practical note on @numba.nijt: 
# make the funciton much faster, but require that
# 1) the inputs are either scalar or 1-dim numpy arrays
# 2) does not change
# debugging -> uncomment @numba.nijt

# # negative inverse transformation
@numba.njit
def transform(x):
	if not x == 0:
		return -1/x
	else:
		return -1e16

# utility function
@numba.njit
def u_func(b,c,phi,gamma,rho,bubar):
	if b > 0 and c > 0:
		agg = phi*(b+bubar)**(1-gamma)+(1-phi)*c**(1-gamma)
		ces = agg**(1/(1-gamma))
		return ces**(1-rho)/(1-rho)
	else:
		return -1e16
		
# bequest utility function
@numba.njit
def nu_func(a,b,qubar,varphi,vartheta):	
	if a+b > -qubar:
		agg = a+b+qubar
		return varphi*agg**(1-vartheta)/(1-vartheta)
	else:
		return -1e16

############
# 4. solve #
############

# back-wards induction
def solve(par,sol):

	tic = time.time()

	for t in reversed(range(par.T)): # par.T-1,par.T-2,...,0

		if par.do_print: print(f't = {t:2d}')

		# a. post-decision value function
		if t == par.T-1:
			compute_W_last_period(t,par,sol)
		else:
			compute_W_gateway(t,par,sol)

		# b. value function keepers
		solve_keep_gateway(t,par,sol)
			
		# c. value function adjusters
		solve_adj_gateway(t,par,sol)
	
	toc = time.time()
	checksum = np.mean(sol['inv_v_adj'])
	print(f'model solved in {toc-tic:.1f} secs (checksum = {checksum})')

##########################################
# 4. solve: post-decision value function #
##########################################

# compute post-decision value function in t = par.T-1
def compute_W_last_period(t,par,sol):

	tic = time.time()		
	
	for (i_l,_l),(i_b,b),(i_a,a) in it.product(	enumerate(par.l_set),
												enumerate(par.grid_b),
												enumerate(par.grid_a)):

		index = (t,i_l,i_b,i_a)
		w = nu_func(a,b,par.qubar,par.varphi,par.vartheta)
		sol['inv_w'][index] = -1.0/w

	toc = time.time()		

	if par.do_print: print(f' w found ({toc-tic:3.2f} secs)')

# compute post-decision value function for t < par.T-1
def compute_W_gateway(t,par,sol): 

	tic = time.time()

	# a. interpolators for adj and keep in next period
	grids = [par.grid_n,par.grid_m] # list of grids 
	values = [sol['inv_v_keep'][t+1,0,:,:]] # list of values, proto type with i_l = 0
	s1 = linear_interp.create_interpolator_dict(grids,values,par.Na) # dict for setting up interpolator

	grids = [par.grid_x] # list of grids 
	values = [sol['inv_v_adj'][t+1,0,:]] # list of values, proto type with i_l = 0
	s2 = linear_interp.create_interpolator_dict(grids,values,par.Na) # dict for setting up interpolator

	#. b. inv_w_* continuation value CONDITIONAL on adj/keep and l in next period
	inv_w_keep = np.empty(par.Nl*par.Nl*par.Nb*par.Na) 
	inv_w_adj = np.empty(par.Nl*par.Nl*par.Nb*par.Na) 

	inv_v_plus_keep = sol['inv_v_keep'][t+1,:,:,:].ravel()
	inv_v_plus_adj = sol['inv_v_adj'][t+1,:,:].ravel()	

	# c. compute in parallel loop
	compute_W_loop(	inv_w_keep,inv_w_adj,
					par.Nl,par.Nb,par.Na,par.l_set,par.grid_a,par.grid_b,par.r,par.delta,par.tau,par.omega,
					s1.x,s1.dimx,s1.Nx,inv_v_plus_keep,s1.dimy,s1.Ny,
					s2.x,s2.dimx,s2.Nx,inv_v_plus_adj,s2.dimy,s2.Ny)

	# d. find max over adj and keep
	inv_w_max = np.fmax(inv_w_keep,inv_w_adj).reshape(par.Nl,par.Nl,par.Nb,par.Na)
	w_max = -1.0/inv_w_max

	#. e. find expected average (over l_plus dimension)
	for i_l_plus in range(par.Nl):
		for i_l in range(par.Nl):
			w_max[i_l_plus,i_l,:,:] *= par.trans_prob_l[i_l,i_l_plus]
	w = np.sum(w_max,axis=0)

	# f. save
	sol['inv_w'][t,:,:,:] = -1.0/w 

	toc = time.time()
	if par.do_print: print(f' w found ({toc-tic:3.2f} secs)')

@numba.njit(parallel=True)
def compute_W_loop(	inv_w_keep,inv_w_adj,
					Nl,Nb,Na,l_set,grid_a,grid_b,r,delta,tau,omega,
		 			s1_x,s1_dimx,s1_Nx,s1_y,s1_dimy,s1_Ny,
		 			s2_x,s2_dimx,s2_Nx,s2_y,s2_dimy,s2_Ny):

	for i_b in numba.prange(Nb): # loop done in paralle

		# allocate containers
		xi_keep = np.empty(2)
		xi_adj = np.empty(1)
		m_plus_vec = np.empty(Na)
		x_plus_vec = np.empty(Na)

		# loops over l_plus
		for i_l_plus in range(Nl):

			# interpolators (numba JIT compiled)
			interp_inv_v_keep = linear_interp.interpolator(s1_x,s1_dimx,s1_Nx,s1_y[i_l_plus*s1_Ny:],s1_dimy,s1_Ny,Na)
			interp_inv_v_adj = linear_interp.interpolator(s2_x,s2_dimx,s2_Nx,s2_y[i_l_plus*s2_Ny:],s2_dimy,s2_Ny,Na)

			# loop over l
			for i_l in range(Nl):

				index = i_l_plus*Nl*Nb*Na + i_l*Nb*Na + i_b*Na + 0 # linear index

				# a. next-period states
				n_plus = (1-delta)*grid_b[i_b]
				for i_a in range(Na):
					m_plus_vec[i_a] = (1+r)*grid_a[i_a] + omega*l_set[i_l_plus]
					x_plus_vec[i_a] = m_plus_vec[i_a] + (1-tau)*n_plus
		
				# b. evaluate keep
				xi_keep[0] = n_plus
				xi_keep[1] = m_plus_vec[0]
				inv_w_keep[index:(index+Na)] = interp_inv_v_keep.evaluate_monotone(xi_keep,m_plus_vec) 

				# c. evaluate adj
				xi_adj[0] = x_plus_vec[0]
				inv_w_adj[index:(index+Na)] = interp_inv_v_adj.evaluate_monotone(xi_keep,x_plus_vec) 
			
##################
# 5. solve: keep #
##################

# gateway
def solve_keep_gateway(t,par,sol): 

	tic = time.time()

	# a. interpolator
	grids = [par.grid_b,par.grid_a] # list of grids 
	values = [sol['inv_w'][t,0,:,:]] # list of values, proto type with i_l = 0
	s = linear_interp.create_interpolator_dict(grids,values,1) # dict for setting up interpolator

	#. b. output
	inv_w = sol['inv_w'][t,:,:,:].ravel()
	inv_v = sol['inv_v_keep'][t,:,:,:].ravel()
	c = sol['c_keep'][t,:,:,:].ravel()

	# c. solve in parallel loop
	solve_keep_loop(inv_v,c,
		par.Nl,par.Nn,par.Nm,par.beta,par.phi,par.gamma,par.rho,par.bubar,
		par.grid_n,par.grid_m,s.x,s.dimx,s.Nx,inv_w,s.dimy,s.Ny)

	toc = time.time()
	if par.do_print: print(f' v keep found ({toc-tic:.2f} secs)')

# solve keep problem across all states (parallel)
@numba.njit(parallel=True)
def solve_keep_loop(inv_v,c,
					Nl,Nn,Nm,beta,phi,gamma,rho,bubar,grid_n,grid_m,
					s_x,s_dimx,s_Nx,s_y,s_dimy,s_Ny):

	for i_n in numba.prange(Nn): # done in parallel
		
		# loop l
		for i_l in range(Nl):

			# i. interpolatator
			interp_inv_w = linear_interp.interpolator(s_x,s_dimx,s_Nx,s_y[i_l*s_Ny:],s_dimy,s_Ny,1)

			# loop m
			for i_m in range(Nm):
				
				# ii. states
				n = grid_n[i_n]
				m = grid_m[i_m]
				
				# iii. solve
				index = i_l*Nn*Nm + i_n*Nm + i_m # linear index
				solve_keep(inv_v[index:],c[index:],n,m,beta,phi,gamma,rho,bubar,interp_inv_w)

# objective function for keep problem
@numba.njit
def solve_keep_obj(c,m,n,beta,phi,gamma,rho,bubar,interp_inv_w):

	# a. penalty
	if c <= 1e-16: # penalty, too low c
		c = 1e-16
		penalty = 1000*(1e-16-c)
	elif c >= m: # penalty, too high c
		c = m
		penalty = 1000*(c-m)

	# b. utility
	u = u_func(n,c,phi,gamma,rho,bubar)
	
	# c. continuation value
	xi = np.empty(2)
	xi[0] = n 			
	xi[1] = m - c			
	inv_w = interp_inv_w.evaluate(xi)[0]
	w = transform(inv_w)
	
	# d. value of choice
	value_of_choice = u + beta*w

	return -value_of_choice + penalty

# optimizer for keep problem (numba JIT compilled)
optimizer_keep = optimize_1d.create_optimizer(solve_keep_obj)

# solve keep problem for specific state
@numba.njit
def solve_keep(inv_v,c,n,m,beta,phi,gamma,rho,bubar,interp_inv_w):

	if m <= 0: # u = -inf

		c[0] = 0
		inv_v[0] = 0
	
	else: # -inf < u < 0
		
		# i. optimizer
		c_low = min(1e-8,m/4)
		c_high = m
		tol = 1e-5
		c_optimal = optimizer_keep(c_low,c_high,tol,m,n,beta,phi,gamma,rho,bubar,interp_inv_w)
		
		# ii. save
		c[0] = c_optimal
		inv_v[0] = transform(-solve_keep_obj(c_optimal,m,n,beta,phi,gamma,rho,bubar,interp_inv_w))

####################
# 6. solve: adjust #
####################

# gateway
def solve_adj_gateway(t,par,sol): 

	tic = time.time()

	# a. interpolator
	grids = [par.grid_n,par.grid_m] # list of grids 
	values = [sol['inv_v_keep'][t,0,:,:]] # list of values, proto type with i_l = 0
	s = linear_interp.create_interpolator_dict(grids,values,1) # dict for setting up interpolator

	#. b. output
	inv_v_keep = sol['inv_v_keep'][t,:,:,:].ravel()
	inv_v_adj = sol['inv_v_adj'][t,:,:].ravel()
	b = sol['b_adj'][t,:,:].ravel()

	# c. solve in parallel loop
	solve_adj_loop(	inv_v_adj,b,
					par.Nl,par.Nx,par.grid_x,
					s.x,s.dimx,s.Nx,inv_v_keep,s.dimy,s.Ny)

	toc = time.time()
	if par.do_print: print(f' v adj found ({toc-tic:.2f} secs)')

# solve keep problem across all states (parallel)
@numba.njit(parallel=True)
def solve_adj_loop(	inv_v,b,
					Nl,Nx,grid_x,
					s_x,s_dimx,s_Nx,s_y,s_dimy,s_Ny):

	for i_l in numba.prange(Nl): # parallized

		# interpolatator
		interp_inv_v_keep = linear_interp.interpolator(s_x,s_dimx,s_Nx,s_y[i_l*s_Ny:],s_dimy,s_Ny,1)

		for i_x in range(Nx):
			
			# i. states
			x = grid_x[i_x]
			
			# ii. solve
			index = i_l*Nx + i_x # linear index
			solve_adj(inv_v[index:],b[index:],x,interp_inv_v_keep)

# objective function for adj problem
@numba.njit
def solve_adj_obj(b,x,interp_inv_v_keep):

	# a. starting value
	if b <= 1e-16: # penalty, too low c
		b = 1e-16
		penalty = 1000*(1e-16-b)
	elif b >= x: # penalty, too high c
		b = x
		penalty = 1000*(b-x)

	# c. value
	xi = np.empty(2)
	xi[0] = b 			
	xi[1] = x - b			
	inv_value_of_choice = interp_inv_v_keep.evaluate(xi)[0]
	
	return -inv_value_of_choice + penalty

# optimizer for adjust problem (numba JIT compilled)
optimizer_adj = optimize_1d.create_optimizer(solve_adj_obj)

# solve adjust problem for specific state
@numba.njit
def solve_adj(inv_v,b,x,interp_inv_v_keep):

	if x <= 0: # u = -inf

		b[0] = 0
		inv_v[0] = 0
	
	else: # -inf < u < 0
		
		# i. optimizer
		b_low = min(1e-8,x/4)
		b_high = x
		tol = 1e-5
		b_optimal = optimizer_adj(b_low,b_high,tol,x,interp_inv_v_keep)
		
		# ii. save
		b[0] = b_optimal
		inv_v[0] = -solve_adj_obj(b_optimal,x,interp_inv_v_keep)
