import numpy as np
import matplotlib.pyplot
import tqdm
from scipy import sparse as sp
import numba

#%%
@numba.njit
def laplacian(N, dx):
    A = np.zeros((N**2,N**2))
    for i1 in range(N):
        for i2 in range(N):
            for j1 in range(N):
                for j2 in range(N):
                    k = N*i1+j1
                    l = N*i2+j2
                    if k==l:
                        A[k,l] = -4
                    if abs(i1-i2)==1 and j1==j2:
                        A[k,l] = 1
                    if abs(j1-j2)==1 and i1==i2:
                        A[k,l] = 1
    return A/dx**2

#%%

def accretion_disc(N, x_start, x_end, t_start, t_end, dt, v):
    x = np.linspace(x_start, x_end, N)
    t = np.arange(t_start, t_end, dt)
    dx = np.diff(x)[0]
    ux_init = np.zeros(N**2)
    uy_init = np.zeros(N**2)
    
    
    
    L = lil_matrix(laplacian(N, dx))
    A1 = sp.identity(N**2)-v*dt*L
    ux_no = sp.solve(A1, ux_init)
    uy_no = sp.solve(A1, uy_init)
    
    u_adv = 