import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

#%%

hx, hy = np.meshgrid(np.linspace(-20,20,71), np.linspace(-20,20,71))

u = -5*hy/np.sqrt(hy**2 + hx**2+0.00001)
z = 5*hx /np.sqrt(hy**2 + hx**2+0.00001)


#%%
# @numba.njit
def accretion_disc_dirichlet(N, x_start, x_end, dt, v, ite):
    x = np.linspace(x_start, x_end, N)
    dx = x[1] - x[0]
    
    # ux_init = u.flatten()
    # uy_init = z.flatten()
    ux_init = np.zeros(N**2)
    uy_init = np.zeros(N**2)
    
    r = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            r[i, j] = np.sqrt((x[i])**2+(x[j])**2)
    fx = np.array([[-0.1*x[i]/(r[i,j]**3+0.001) for i in range(N)] for j in range(N)])
    fy = np.array([[-0.1*x[j]/(r[i,j]**3+0.001) for i in range(N)] for j in range(N)])
    
    fxx = fx.flatten()
    fyy = fy.flatten()
    
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
                if np.isnan(x_adv[i,j]):
                    ux_new[i,j] = 0
                elif np.isnan(y_adv[i,j]):
                    uy_new[i,j] = 0
                else:
                    x_approx = (x_adv[i, j] - x_start)/dx
                    y_approx = (y_adv[i, j] - x_start)/dx
                    x1_index = int(x_approx)
                    x2_index = x1_index +1
                    y1_index = int(y_approx)
                    y2_index = y1_index +1
                    
                    try:
                        
                        # print((x1_index, x2_index), (y1_index, y2_index))
                        # print(x_adv[i,j],y_adv[i,j])
                        
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
                    except IndexError:
                        ux_new[i, j] = 0
                        uy_new[i, j] = 0
                        
        return ux_new.flatten(), uy_new.flatten()
    
    L = laplacian(N, dx)
    
    del_x = x_diff(N, dx)
    del_y = y_diff(N, dx)
    
    A1 = np.identity(N**2)-v*dt*L
    A1[0:N] = 0
    A1[-N:] = 0
    for i in range(N):
        A1[i, i] = 1
        A1[-(i+1), -(i+1)] = 1
    for i in range(N-2):
        A1[(i+1)*N] = 0
        A1[(i+1)*N, (i+1)*N] = 1
        A1[(i+2)*N-1] = 0
        A1[(i+2)*N-1, (i+2)*N-1] = 1
    
    A2 = L*dt
    
    # A2[0:N] = 0
    # A2[-N:] = 0
    # for i in range(N):
    #     A2[i, i] = 1
    #     A2[-(i+1), -(i+1)] = 1
    # for i in range(N-2):
    #     A2[(i+1)*N] = 0
    #     A2[(i+1)*N, (i+1)*N] = 1
    #     A2[(i+2)*N-1] = 0
    #     A2[(i+2)*N-1, (i+2)*N-1] = 1
    
    # A2[0:N] = 0
    # A2[-N:] = 0
    # for i in range(N):
    #     A2[i,i] = -1.5/dx
    #     A2[i,i+N] = 2/dx
    #     A2[i,i+2*N] = -0.5/dx
        
    #     A2[-N+i,-N+i] = 1.5/dx
    #     A2[-N+i,-2*N+i] = -2/dx
    #     A2[-N+i,-3*N+i] = 0.5/dx
    # for i in range(N-2):
    #     A2[(i+1)*N] = 0
    #     A2[(i+2)*N-1] = 0
    #     A2[(i+1)*N, (i+1)*N] = -1.5/dx
    #     A2[(i+1)*N, (i+1)*N+1] = 2/dx
    #     A2[(i+1)*N, (i+1)*N+2] = -0.5/dx
        
    #     A2[(i+2)*N-1, (i+2)*N-1] = 1.5/dx
    #     A2[(i+2)*N-1, (i+2)*N-2] = -2/dx
    #     A2[(i+2)*N-1, (i+2)*N-3] = 0.5/dx
    # A2[N//2] = 0
    # A2[N//2, N//2] = 1
    
    inv_A1 = np.linalg.inv(A1)
    inv_A2 = np.linalg.inv(A2)
    
    ux = np.zeros((ite+1, N, N))
    uy = np.zeros((ite+1, N, N))
    
    ux[0] = np.reshape(ux_init, (N, N))
    uy[0] = np.reshape(uy_init, (N, N))
    
    ps = np.zeros((ite, N, N))
    
    for i in tqdm(range(ite)):
        
        visc_x = inv_A1 @ (ux_init + fxx*dt)
        visc_y = inv_A1 @ (uy_init + fyy*dt)
        
        adv_x, adv_y = advection_term(visc_x, visc_y)
        
        pres_term = del_x @ adv_x + del_x @ adv_y
        
        pres_term[0:N] = 0
        pres_term[-N:] = 0
        # pres_term[N**2//2] = 1000
        for j in range(N-2):
            pres_term[(j+1)*N] = 0
            pres_term[(j+2)*N-1] = 0
        # pres_term[N**2//2] = 1
        
        
        p = inv_A2 @ pres_term
        
        ps[i] = np.reshape(p, (N, N))
        
        ux_init = adv_x - dt * del_x @ p
        uy_init = adv_y - dt * del_y @ p
        
        ux[i+1] = np.reshape(ux_init, (N, N))
        uy[i+1] = np.reshape(uy_init, (N, N))
    
    return x, ux, uy, ps, fxx, fyy

#%%

trial = accretion_disc_dirichlet(71, -20, 20, 0.001, 0.2, 50)

#%%
from IPython.display import clear_output


plt.close('all')
xx, yy = np.meshgrid(trial[0], trial[0])

fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
plt.plot(0,0, 'o')
t = range(len(trial[1]))

for i in t:
    ax.cla()
    ax.quiver(xx,yy,trial[1][i],trial[2][i])
    clear_output(wait = True)
    plt.pause(0.05)
    
#%%
plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
t1 = range(len(trial[3]))
mi = np.min(trial[3])
ma = np.max(trial[3])

for i in t1:
    ax1.cla()
    ax1.imshow(trial[3][i], vmin = mi, vmax = ma, cmap='jet')
    clear_output(wait = True)
    plt.pause(0.1)

#%%
plt.close('all')
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1,1,1)

for i in t:
    ax2.cla()
    ax2.streamplot(xx,yy,trial[1][i],trial[2][i])
    clear_output(wait = True)
    plt.pause(0.1)
    
#%% 

def lmao(N, dx, dt):
    
    L = laplacian(N, dx)
    A2 = L*dt
    
    # A2[0:N] = 0
    # A2[-N:] = 0
    # for i in range(N):
    #     A2[i, i] = 1
    #     A2[-(i+1), -(i+1)] = 1
    # for i in range(N-2):
    #     A2[(i+1)*N] = 0
    #     A2[(i+1)*N, (i+1)*N] = 1
    #     A2[(i+2)*N-1] = 0
    #     A2[(i+2)*N-1, (i+2)*N-1] = 1
    
    A2[0:N] = 0
    A2[-N:] = 0
    for i in range(N):
        A2[i,i] = -1.5/dx
        A2[i,i+N] = 2/dx
        A2[i,i+2*N] = -0.5/dx
        
        A2[-N+i,-N+i] = 1.5/dx
        A2[-N+i,-2*N+i] = -2/dx
        A2[-N+i,-3*N+i] = 0.5/dx
    for i in range(N-2):
        A2[(i+1)*N] = 0
        A2[(i+2)*N-1] = 0
        A2[(i+1)*N, (i+1)*N] = -1.5/dx
        A2[(i+1)*N, (i+1)*N+1] = 2/dx
        A2[(i+1)*N, (i+1)*N+2] = -0.5/dx
        
        A2[(i+2)*N-1, (i+2)*N-1] = 1.5/dx
        A2[(i+2)*N-1, (i+2)*N-2] = -2/dx
        A2[(i+2)*N-1, (i+2)*N-3] = 0.5/dx
    
    return A2

A = lmao(3, 1, 1)

#%%

def accretion_disc_neumann(N, x_start, x_end, dt, v, ite):
    x = np.linspace(x_start, x_end, N)
    dx = x[1] - x[0]
    
    ux_init = u.flatten()
    uy_init = z.flatten()
    # ux_init = np.zeros(N**2)
    # uy_init = np.zeros(N**2)
    
    r = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            r[i, j] = np.sqrt((x[i])**2+(x[j])**2)
    fx = np.array([[-0.1*x[i]/(r[i,j]**3+0.001) for i in range(N)] for j in range(N)])
    fy = np.array([[-0.1*x[j]/(r[i,j]**3+0.001) for i in range(N)] for j in range(N)])
    
    fxx = fx.flatten()
    fyy = fy.flatten()
    
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
                if np.isnan(x_adv[i,j]):
                    ux_new[i,j] = 0
                elif np.isnan(y_adv[i,j]):
                    uy_new[i,j] = 0
                else:
                    x_approx = (x_adv[i, j] - x_start)/dx
                    y_approx = (y_adv[i, j] - x_start)/dx
                    x1_index = int(x_approx)
                    x2_index = x1_index +1
                    y1_index = int(y_approx)
                    y2_index = y1_index +1
                    
                    try:
                        
                        # print((x1_index, x2_index), (y1_index, y2_index))
                        # print(x_adv[i,j],y_adv[i,j])
                        
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
                    except IndexError:
                        ux_new[i, j] = 0
                        uy_new[i, j] = 0
                        
        return ux_new.flatten(), uy_new.flatten()
    
    L = laplacian(N, dx)
    
    del_x = x_diff(N, dx)
    del_y = y_diff(N, dx)
    
    A1 = np.identity(N**2)-v*dt*L
    A1[0:N] = 0
    A1[-N:] = 0
    for i in range(N):
        A1[i, i] = 1
        A1[-(i+1), -(i+1)] = 1
    for i in range(N-2):
        A1[(i+1)*N] = 0
        A1[(i+1)*N, (i+1)*N] = 1
        A1[(i+2)*N-1] = 0
        A1[(i+2)*N-1, (i+2)*N-1] = 1
    
    A2 = L*dt
    
    A2[0:N] = 0
    A2[-N:] = 0
    for i in range(N):
        A2[i, i] = 1
        A2[-(i+1), -(i+1)] = 1
    for i in range(N-2):
        A2[(i+1)*N] = 0
        A2[(i+1)*N, (i+1)*N] = 1
        A2[(i+2)*N-1] = 0
        A2[(i+2)*N-1, (i+2)*N-1] = 1
    
    A2[0:N] = 0
    A2[-N:] = 0
    for i in range(N):
        A2[i,i] = -1.5/dx
        A2[i,i+N] = 2/dx
        A2[i,i+2*N] = -0.5/dx
        
        A2[-N+i,-N+i] = 1.5/dx
        A2[-N+i,-2*N+i] = -2/dx
        A2[-N+i,-3*N+i] = 0.5/dx
    for i in range(N-2):
        A2[(i+1)*N] = 0
        A2[(i+2)*N-1] = 0
        A2[(i+1)*N, (i+1)*N] = -1.5/dx
        A2[(i+1)*N, (i+1)*N+1] = 2/dx
        A2[(i+1)*N, (i+1)*N+2] = -0.5/dx
        
        A2[(i+2)*N-1, (i+2)*N-1] = 1.5/dx
        A2[(i+2)*N-1, (i+2)*N-2] = -2/dx
        A2[(i+2)*N-1, (i+2)*N-3] = 0.5/dx
    # A2[N//2] = 0
    # A2[N//2, N//2] = 1
    
    inv_A1 = np.linalg.inv(A1)
    inv_A2 = np.linalg.inv(A2)
    
    ux = np.zeros((ite+1, N, N))
    uy = np.zeros((ite+1, N, N))
    
    ux[0] = np.reshape(ux_init, (N, N))
    uy[0] = np.reshape(uy_init, (N, N))
    
    ps = np.zeros((ite, N, N))
    
    for i in tqdm(range(ite)):
        
        visc_x = inv_A1 @ (ux_init + fxx*dt)
        visc_y = inv_A1 @ (uy_init + fyy*dt)
        
        adv_x, adv_y = advection_term(visc_x, visc_y)
        
        pres_term = del_x @ adv_x + del_x @ adv_y
        
        pres_term[0:N] = 0
        pres_term[-N:] = 0
        for j in range(N-2):
            pres_term[(j+1)*N] = 0
            pres_term[(j+2)*N-1] = 0
        # pres_term[N**2//2] = 1
        
        
        p = inv_A2 @ pres_term
        
        ps[i] = np.reshape(p, (N, N))
        
        ux_init = adv_x - dt * del_x @ p
        uy_init = adv_y - dt * del_y @ p
        
        ux[i+1] = np.reshape(ux_init, (N, N))
        uy[i+1] = np.reshape(uy_init, (N, N))
    
    return x, ux, uy, ps, fxx, fyy

#%%

trial2 = accretion_disc_neumann(71, -10, 10, 0.001, 0.2, 50)

#%%

plt.close('all')
fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
plt.plot(0,0, 'o')
t = range(len(trial[1]))

for i in t:
    ax.cla()
    ax.quiver(xx,yy,trial2[1][i],trial2[2][i])
    clear_output(wait = True)
    plt.pause(0.05)
    
#%%

plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
t1 = range(len(trial2[3]))
mi = np.min(trial2[3])
ma = np.max(trial2[3])

for i in t1:
    ax1.cla()
    ax1.imshow(trial2[3][i], vmin = mi, vmax = ma, cmap='jet')
    clear_output(wait = True)
    plt.pause(0.1)
    
#%%
plt.close('all')
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1,1,1)

for i in t:
    ax2.cla()
    ax2.streamplot(xx,yy,trial[1][i],trial[2][i])
    clear_output(wait = True)
    plt.pause(0.1)
    