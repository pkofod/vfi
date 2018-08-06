import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import vfi.linear_interp as linear_interp

def setup(Nxi):

    # a. grids
    grid_x = np.linspace(0,1,100)
    grid_y = np.linspace(0,1,200)
    grid_z = np.linspace(0,1,250)

    # b. values
    def f(x,y,z):
        return x**2 + y + z

    def g(x,y,z):
        return x**2 + y + z**2

    xx, yy, zz = np.meshgrid(grid_x,grid_y,grid_z,indexing='ij')
    ff = f(xx,yy,zz)
    gg = g(xx,yy,zz)

    # c. interpolators
    fhat = linear_interp.create_interpolator(grids=[grid_x,grid_y,grid_z],values=[ff],Nxi=Nxi)
    ghat = linear_interp.create_interpolator(grids=[grid_x,grid_y,grid_z],values=[gg],Nxi=Nxi)
    fghat = linear_interp.create_interpolator(grids=[grid_x,grid_y,grid_z],values=[ff,gg],Nxi=Nxi)

    # d. scipy interpolators
    fhat_scipy = RegularGridInterpolator([grid_x,grid_y,grid_z], ff, method='linear',bounds_error=False,fill_value=None)
    ghat_scipy = RegularGridInterpolator([grid_x,grid_y,grid_z], gg, method='linear',bounds_error=False,fill_value=None)

    return fhat,ghat,fghat,fhat_scipy,ghat_scipy

def single(loop_length,best_of=5):

    print(f'loop length \t {loop_length}')

    # a. setup
    fhat,_ghat,_fghat,fhat_scipy,_ghat_scipy = setup(1)

    # b. xi
    xi = np.random.uniform(size=(loop_length,3))

    # c. time
    r = 0
    fhat.evaluate(xi[r,:]) # to cache jit

    res = np.zeros(loop_length)
    time_linear_interp = np.inf
    for _i in range(best_of):
        tic = time.time()
        for r in range(loop_length):
            res[r:] = fhat.evaluate(xi[r,:])[0]
        toc = time.time()
        time_linear_interp = np.fmin(time_linear_interp,toc-tic)

    print(f'linear_interp \t {time_linear_interp:.2f} secs')

    # d. time scipy
    res_scipy = np.zeros(loop_length)
    time_scipy = np.inf
    for _i in range(best_of):    
        tic = time.time()
        for r in range(loop_length):
            res_scipy[r] = fhat_scipy(xi[r,:])
        toc = time.time()
        time_scipy = np.fmin(time_scipy,toc-tic)

    print(f'scipy \t\t {time_scipy:.2f} secs'.format())

    print(f'speed-up \t {time_scipy/time_linear_interp:.1f}\n')

    # e. assertions
    assert(np.allclose(res,res_scipy))

def vector(Nxi,best_of=5):

    print('vector length \t {}'.format(Nxi))
    
    # a. setup
    _fhat,_ghat,fghat,fhat_scipy,ghat_scipy = setup(Nxi)

    # b. xi
    xi_vec = np.linspace(0,1,Nxi)
    
    xi_base = np.zeros(3)
    xi_base[0:2] = np.random.uniform(size=(2,))
    xi_base[2] = xi_vec[0]

    xi = np.zeros((Nxi,3))
    xi[:,0:2] = xi_base[0:2]
    xi[:,2] = xi_vec
    
    # c. time
    fghat.evaluate_monotone(xi_base,xi_vec) # to cache jit
    time_linear_interp = np.inf
    for _i in range(best_of):
        tic = time.time()
        fg_res = fghat.evaluate_monotone(xi_base,xi_vec)
        toc = time.time()
        time_linear_interp = np.fmin(time_linear_interp,toc-tic)
    print(f'linear_interp \t {time_linear_interp:.2f} secs')

    # c. time (no monotonicity)
    fghat.evaluate(xi.ravel()) # to cache jit
    time_linear_interp_nomono = np.inf
    for _i in range(best_of):
        tic = time.time()
        fg_res_nomono = fghat.evaluate(xi.ravel())
        toc = time.time()
        time_linear_interp_nomono = np.fmin(time_linear_interp_nomono,toc-tic)

    print(f'linear_interp \t {time_linear_interp_nomono:.2f} secs (no monotonicity)')

    # d. time scipy
    time_scipy = np.inf
    for _i in range(best_of):    
        tic = time.time()
        f_res_scipy = fhat_scipy(xi)
        g_res_scipy = ghat_scipy(xi)
        toc = time.time()
        time_scipy = np.fmin(time_scipy,toc-tic)

    print(f'scipy \t\t {time_scipy:.2f} secs'.format())

    print(f'speed-up \t {time_scipy/time_linear_interp:.1f}')
    print(f'speed-up \t {time_scipy/time_linear_interp_nomono:.1f} (no monotonicity)\n')

    # e. assertions
    assert(np.allclose(fg_res[:Nxi],f_res_scipy))
    assert(np.allclose(fg_res[Nxi:],g_res_scipy))
    assert(np.allclose(fg_res_nomono[:Nxi],f_res_scipy))
    assert(np.allclose(fg_res_nomono[Nxi:],g_res_scipy))