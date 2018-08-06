import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from vfi import linear_interp as linear_interp 

def all(model):

    widgets.interact(all_,
                     model=widgets.fixed(model),
                     t=widgets.Dropdown(description='t', options=list(range(model.par.T)), value=0),
                     i_l=widgets.Dropdown(description='i_l', options=list(range(model.par.Nl)), value=0),
                     )

def all_(model,t,i_l):

    w_and_discrete(model,t,i_l)
    keep(model,t,i_l)
    adj(model,t,i_l)

def w_and_discrete(model,t,i_l):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_w = fig.add_subplot(1,2,1,projection='3d')
    ax_discrete = fig.add_subplot(1,2,2)

    # c. plot w
    b,a = np.meshgrid(par.grid_a, par.grid_b)
    ax_w.plot_surface(b,a,sol['inv_w'][t,i_l,:,:],cmap=cm.viridis)
    ax_w.set_title(f'inverse $w$ ($t = {t}$, $l = {par.l_set[i_l]:.1f}$)',pad=10)
    
    # d. details w
    ax_w.set_xlabel('$a_t$')
    ax_w.set_xlim([par.a_min,par.a_max])
    ax_w.set_ylabel('$b_t$')
    ax_w.set_ylim([par.b_min,par.b_max])
    ax_w.invert_xaxis()

    # e. interpolation
    n,m = np.meshgrid(par.grid_n,par.grid_m,indexing='ij')
    x = m + (1-par.tau)*n
    
    grids = [par.grid_x]
    values = [sol['inv_v_adj'][t,i_l,:]]
    interp_inv_v_adj = linear_interp.create_interpolator(grids,values,x.size)

    inv_v_adj = interp_inv_v_adj.evaluate(x.ravel()).reshape((par.Nn,par.Nm))

    # f. best discrete choice
    I = inv_v_adj > sol['inv_v_keep'][t,i_l,:,:]

    y = n[I].ravel()
    x = m[I].ravel()
    ax_discrete.scatter(x,y,s=2,label='adjust')
    
    x = m[~I].ravel()
    y = n[~I].ravel()
    ax_discrete.scatter(x,y,s=2,label='keep')
        
    ax_discrete.set_title(f'optimal discrete choice ($t = {t}$, $l = {par.l_set[i_l]:.1f}$)',pad=10)

    legend = ax_discrete.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax_discrete.grid(True)
    ax_discrete.set_xlabel('$m_t$')
    ax_discrete.set_xlim([par.m_min,par.m_max])
    ax_discrete.set_ylabel('$n_t$')
    ax_discrete.set_ylim([par.n_min,par.n_max])
    
    plt.show()
    
def keep(model,t,i_l):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_c = fig.add_subplot(1,2,1,projection='3d')
    ax_v = fig.add_subplot(1,2,2,projection='3d')

    n,m = np.meshgrid(par.grid_m, par.grid_n)

    # c. plot consumption
    ax_c.plot_surface(n,m,sol['c_keep'][t,i_l,:,:],cmap=cm.viridis)
    ax_c.set_title(f'$c^{{keep}}$ ($t = {t}$, $l = {par.l_set[i_l]:.1f}$)',pad=10)

    # d. plot value function
    ax_v.plot_surface(n,m,sol['inv_v_keep'][t,i_l,:,:],cmap=cm.viridis)
    ax_v.set_title(f'inverse $v^{{keep}}$ ($t = {t}$, $l = {par.l_set[i_l]:.1f}$)',pad=10)

    # e. details
    for ax in [ax_c,ax_v]:

        ax.set_xlabel('$m_t$')
        ax.set_xlim([par.m_min,par.m_max])
        ax.set_ylabel('$n_t$')
        ax.set_ylim([par.n_min,par.n_max])
        ax.invert_xaxis()

    plt.show()

def adj(model,t,i_l):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_b = fig.add_subplot(1,2,1)
    ax_v = fig.add_subplot(1,2,2)
    
    # c. plot consumption
    ax_b.plot(par.grid_x,sol['b_adj'][t,i_l,:],lw=2)
    ax_b.set_title(f'$b^{{adj}}$ ($t = {t}$, $l = {par.l_set[i_l]:.1f}$)',pad=10)

    # d. plot value function
    ax_v.plot(par.grid_x,sol['inv_v_adj'][t,i_l,:],lw=2)
    ax_v.set_title(f'inverse $v^{{adj}}$ ($t = {t}$, $l = {par.l_set[i_l]:.1f}$)',pad=10)

    # e. details
    for ax in [ax_b,ax_v]:
        ax.grid(True)
        ax.set_xlabel('$x_t$')
        ax.set_xlim([par.x_min,par.x_max])

    plt.show()