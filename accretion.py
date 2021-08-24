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
    
    
    
    L = sp.lil_matrix(laplacian(N, dx))
    A1 = sp.identity(N**2)-v*dt*L
    visc_x = sp.solve(A1, ux_init)
    visc_y = sp.solve(A1, uy_init)
    
    def advection_term(ux, uy, dt, N, x):
        ux_mat = np.reshape(ux, (N, N))
        uy_mat = np.reshape(uy, (N, N))
        x_mat, y_mat = np.meshgrid(x, x)
        x_adv = x_mat - ux_mat*dt
        y_adv = y_mat - uy_mat*dt
        ux_new = np.zeros((N, N))
        uy_new = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                x1_index, x2_index = np.argsort(np.abs(x - x_adv[i, j]))[0:2]
                y1_index, y2_index = np.argsort(np.abs(x - y_adv[i, j]))[0:2]
                
                w11 = abs((x2_index - x_adv[i, j])*(y2_index - y_adv[i, j]) / 
                          ((x2_index - x1_index[i, j])*(y2_index - y1_index)))
                w12 = abs((x2_index - x_adv[i, j])*(y1_index - y_adv[i, j]) / 
                          ((x2_index - x1_index[i, j])*(y2_index - y1_index)))
                w21 = abs((x1_index - x_adv[i, j])*(y2_index - y_adv[i, j]) / 
                          ((x2_index - x1_index[i, j])*(y2_index - y1_index)))
                w22 = abs((x1_index - x_adv[i, j])*(y1_index - y_adv[i, j]) / 
                          ((x2_index - x1_index[i, j])*(y2_index - y1_index)))
                
                ux_new[i, j] = (w11 * ux_mat[x1_index, y1_index] + 
                                w12 * ux_mat[x1_index, y2_index] + 
                                w21 * ux_mat[x2_index, y1_index] + 
                                w22 * ux_mat[x2_index, y2_index])
                uy_new[i, j] = (w11 * uy_mat[x1_index, y1_index] + 
                                w12 * uy_mat[x1_index, y2_index] + 
                                w21 * uy_mat[x2_index, y1_index] + 
                                w22 * uy_mat[x2_index, y2_index])
        return ux_new, uy_new