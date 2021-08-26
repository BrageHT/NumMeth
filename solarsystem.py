# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:06:13 2021

@author: brage
"""
from tqdm import tqdm
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp
from IPython.display import clear_output
#%%
"""
Defining classes
"""

class Planet:
    def __init__(self,mass,pos,vel):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        
#%%



#%%

# @jit
def merge(planet,r,list_mask):
    """
    Calculating distance between planets and merging if close enough,
    and setting (highest) mask index of planet to zero.
    """
    
    close, _ = np.where(np.abs(r - 1/2) < 1/2) # planet a & b within 1 unit distance
    close_star = np.where(np.abs(r[:,0] - 3/2) < 3/2) # larger merge radius for star (3 unit dist)
    # print(close_star)
    
    
    if len(close_star) > 0 and np.any(close_star) == np.any(np.where(list_mask == 1)):
        close = np.array([0,*close_star[0]])
        # print(close)
        
    if len(close) != 0 and np.any(close) == np.any(np.where(list_mask == 1)):
        # conservation of momentum
        m_final = planet[close[0]].mass + planet[close[1]].mass
        v_final = (planet[close[0]].vel*planet[close[0]].mass + 
                   planet[close[1]].vel*planet[close[1]].mass) / m_final
        
        planet[close[0]].mass = m_final
        planet[close[0]].vel = v_final
        
        planet[close[1]].mass = 0
        planet[close[1]].vel = 0
        # print(*close)
        list_mask[close[1]] = 0
        list_mask[close[0]] = 1
        # print(f"Planets {close[0]} and  {close[1]} merged! ")
        
        
# @jit
def timestep(planet,dt,first_run):
    """
    Explicit Euler for timestepping newtonian gravity
    """
    
    # Initializing and creating matricies before the first timestep
    
    if first_run:
        N = len(planet)
        r = np.zeros((N,N)) # r from low to high planet no.
        r_x = np.zeros((N,N)) # same
        r_y = np.zeros((N,N)) # same
        cos_theta = np.zeros((N,N))
        sin_theta = np.zeros((N,N))
        a = np.zeros((2,N))
        first_run = False
        
        # global list_mask
        # list_mask = np.ones(N)
        
    for i in range(N):
        if list_mask[i]:
            for j in range(N):
                if list_mask[j]:
                    if i != j: # and i < j:
                        
                        r_x[i,j] = planet[i].pos[0] - planet[j].pos[0]
                        r_y[i,j] = planet[i].pos[1] - planet[j].pos[1]
                        r[i,j] = np.sqrt(r_x[i,j]**2 + r_y[i,j]**2)
                        cos_theta[i,j] = r_x[i,j]/r[i,j]
                        sin_theta[i,j] = r_y[i,j]/r[i,j]
        
                        
                        a[:,i] += -planet[j].mass/r[i,j]**2 * np.array( [ cos_theta[i,j], sin_theta[i,j] ] )


    for i in range(N):
        if list_mask[i]:
            planet[i].vel += a[:,i]*dt
            planet[i].pos += planet[i].vel
    

    merge(planet,r,list_mask)


        
#%%
# @jit
def random_init(N,mass_range,pos_range):
    planets = []
    
    # Star in origo
    star_mass = 500
    planets.append(Planet(star_mass,(0.0,0.0),(0.0,0.0)))
    
    # creating N-1 planetesimals
    for i in range(N-1):
        
        pos = np.random.uniform(*pos_range,2)
        dist = np.sqrt(pos[0]**2 + pos[1]**2)
        
        #calculating v_int for a (somewhat) stable orbit
        vel = np.sqrt(star_mass/dist)/10 * np.array([pos[1]/dist, pos[0]/dist] ) # sine and cos
        # print(vel)
        planets.append(Planet(np.random.uniform(*mass_range), pos, vel))
        
        
    return planets
    

#%%
plt.close("all")

fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
zoom = 30

#Defining initial conditions
planets_array = random_init(50,(0.1,5),(-zoom,zoom))
first_run = True
list_mask = np.ones(len(planets_array))


for i in tqdm(range(1000)):
    
    timestep(planets_array,0.01,first_run)
    ax.cla()
    # ax.legend()
    list_mask[0] = 1
    # print(planets_array[0].vel)
    for j in range(len(planets_array)):
        if list_mask[j]:
            ax.plot(*planets_array[j].pos,marker='o', markersize=2*np.sqrt(planets_array[j].mass/np.pi),scalex=False,scaley=False)
            # if j == 0:
            #     ax.legend()
            #     ax.plot(*planets[j].pos,'o',scalex=False,scaley=False,label='Star')
            # else:
            #     ax.plot(*planets[j].pos,'o',scalex=False,scaley=False)

    ax.set(xlim=(-zoom,zoom), ylim=(-zoom,zoom))
    clear_output(wait = True)
    plt.pause(0.05)
    


#%%
