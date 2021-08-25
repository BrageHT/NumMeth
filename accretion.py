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

def x_diff(N, dx):
    A = np.kron(np.eye(N, N), np.eye(N, N, 1) - np.eye(N, N, -1))
    return A/(2*dx)

def y_diff(N, dx):
    A = np.eye(N**2, N**2, N) - np.eye(N**2, N**2, -N)
    return A/(2*dx)
bb = y_diff(4, 1)
cc = x_diff(4, 1)
#%%
@numba.njit
def accretion_disc(N, x_start, x_end, t_start, t_end, dt, v, ite):
    x = np.linspace(x_start, x_end, N)
    t = np.arange(t_start, t_end, dt)
    dx = x[1] - x[0]
    ux_init = np.zeros(N**2)
    uy_init = np.zeros(N**2)
    
    def advection_term(ux, uy):
        ux_mat = np.reshape(ux, (N, N))
        uy_mat = np.reshape(uy, (N, N))
        x_mat, y_mat = np.meshgrid(x, x)
        x_adv = x_mat - ux_mat*dt
        y_adv = y_mat - uy_mat*dt
        ux_new = np.zeros((N, N))
        uy_new = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                x_approx = (x_adv[i, j] - x_start)/dx
                y_approx = (y_adv[i, j] - x_start)/dx
                x1_index = int(x_approx)
                x2_index = int(np.ceil(x_approx))
                y1_index = int(y_approx)
                y2_index = int(np.ceil(y_approx))
                
                w11 = abs((x[x2_index] - x_adv[i, j])*(x[y2_index] - y_adv[i, j]) / 
                          ((x[x2_index] - x[x1_index])*(x[y2_index] - x[y1_index])))
                w12 = abs((x[x2_index] - x_adv[i, j])*(x[y1_index] - y_adv[i, j]) / 
                          ((x[x2_index] - x[x1_index])*(x[y2_index] - x[y1_index])))
                w21 = abs((x[x1_index] - x_adv[i, j])*(x[y2_index] - y_adv[i, j]) / 
                          ((x[x2_index] - x[x1_index])*(x[y2_index] - x[y1_index])))
                w22 = 1 - w11 - w12 - w21
                
                ux_new[i, j] = (w11 * ux_mat[x1_index, y1_index] + 
                                w12 * ux_mat[x1_index, y2_index] + 
                                w21 * ux_mat[x2_index, y1_index] + 
                                w22 * ux_mat[x2_index, y2_index])
                uy_new[i, j] = (w11 * uy_mat[x1_index, y1_index] + 
                                w12 * uy_mat[x1_index, y2_index] + 
                                w21 * uy_mat[x2_index, y1_index] + 
                                w22 * uy_mat[x2_index, y2_index])
        return np.flatten(ux_new), np.flatten(uy_new)
    
    L = laplacian(N, dx)
    
    del_x = x_diff(N, dx)
    del_y = y_diff(N, dx)
    
    A1 = np.identity(N**2)-v*dt*L
    A2 = L*dt
    
    inv_A1 = np.linalg.inv(A1)
    inv_A2 = np.linalg.inv(A2)
    
    ux = np.zeros((ite, N, N))
    uy = np.zeros((ite, N, N))
    
    ux[0] = np.reshape(ux_init, (N, N))
    uy[0] = np.reshape(uy_init, (N, N))
    
    for i in tqdm(range(ite)):
        
        visc_x = inv_A1 @ ux_init
        visc_y = inv_A1 @ uy_init
        
        adv_x, adv_y = advection_term(visc_x, visc_y)
        
        pres_term = del_x @ adv_x + del_x @ adv_y
        
        p = inv_A2 @ pres_term
        
        ux_init = adv_x - dt * del_x @ p
        uy_init = adv_y - dt * del_y @ p
        
        ux[i+1] = np.reshape(ux_init, (N, N))
        uy[i+1] = np.reshape(uy_init, (N, N))
    
    return ux, uy