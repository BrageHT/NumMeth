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

#defining diff eq

def timestep2(planet,dt):
    N = len(planet)
    
    r = np.zeros((N,N)) # r from low to high planet no.
    r_x = np.zeros((N,N)) # same
    r_y = np.zeros((N,N)) # same
    cos_theta = np.zeros((N,N))
    sin_theta = np.zeros((N,N))
    a = np.zeros((2,N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                
                r_x[i,j] = planet[i].pos[0] - planet[j].pos[0]
                r_y[i,j] = planet[i].pos[1] - planet[j].pos[1]
                r[i,j] = np.sqrt(r_x[i,j]**2 + r_y[i,j]**2)
                cos_theta[i,j] = r_x[i,j]/r[i,j]
                sin_theta[i,j] = r_y[i,j]/r[i,j]

                
                a[:,i] += -planet[j].mass/r[i,j]**2 * np.array( [ cos_theta[i,j], sin_theta[i,j] ] )
                # print(a[:,i])
                
        planet[i].vel = a[:,i]*dt
        planet[i].pos += planet[i].vel
        # print(planet[2].pos)

        

    

#%%
plt.close("all")

sun = Planet(1,(5.0,-10.0),(0.02,0.0))
earth = Planet(1,(-5.0,-5.0),(0.0,-0.05)) #1/333000
other = Planet(1,(10.0,5.0), (0.0,0.05))
other2 = Planet(1, (-12.0,3.0),(0.06,0.0))

planets = [earth,sun,other,other2]
fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
zoom = 20

for i in range(1000):
    
    ax.cla()
    ax.plot(*earth.pos,'o',scalex=False,scaley=False)
    ax.plot(*sun.pos,'o',scalex=False,scaley=False)
    ax.plot(*other.pos,'o',scalex=False,scaley=False)
    ax.plot(*other2.pos,'o',scalex=False,scaley=False)
    ax.set(xlim=(-zoom,zoom), ylim=(-zoom,zoom))
    clear_output(wait = True)
    plt.pause(0.05)
    timestep2(planets,0.5)


#%%
