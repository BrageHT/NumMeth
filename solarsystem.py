# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:06:13 2021

@author: brage
"""

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
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        
#%%



#%%


def merge(planet,r,list_mask):
    """
    Calculating distance between planets and merging if close enough,
    and setting (highest) mask index of planet to zero.
    """
    close, _ = np.where(np.abs(r - 1/2) < 1/2) # planet a & b within 1 unit distance
    if len(close) != 0: # and close.any() != list_except.any():
        # print(close)
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
        # print(f"Planets {close[0]} and  {close[1]} merged! ")
        # print([planet[close[0]].mass, timestep(planets,0.1,first_run)])
        

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
            planet[i].vel = a[:,i]*dt
            planet[i].pos += planet[i].vel
    

    merge(planet,r,list_mask)


        

    

#%%
plt.close("all")

sun = Planet(1,(2.0,2.0),(0.02,0.0))
earth = Planet(1,(0.0,0.0),(0.0,-0.05)) #1/333000
other = Planet(1,(5.0,5.0), (0.0,0.05))
other2 = Planet(1, (-4.0,3.0),(0.06,0.0))

planets = [earth,sun,other,other2]
# planets = [earth,sun]

fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
zoom = 15


first_run = True
list_mask = np.ones(len(planets))


for i in range(300):
    
    timestep(planets,0.1,first_run)
    ax.cla()
    
    for j in range(len(planets)):
        if list_mask[j]:
            ax.plot(*planets[j].pos,'o',scalex=False,scaley=False)

    ax.set(xlim=(-zoom,zoom), ylim=(-zoom,zoom))
    clear_output(wait = True)
    plt.pause(0.05)
    


#%%
