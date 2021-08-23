# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:06:13 2021

@author: brage
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
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

# def f(t,vec):
aa = []
t = np.array([i*0.01 for i in range(100)])
def timestep(planet1,planet2,dt):
    
    A = planet1.pos[0] - planet2.pos[0]
    B = planet1.pos[1] - planet2.pos[1]
    dist = np.array([A,B])
    value = -np.array( [np.sign(A)/dist[0]**2, np.sign(B)/dist[1]**2] )
    a1 = planet2.mass * value
    a2 = planet1.mass * value
    
    planet1.vel += a1*dt
    planet2.vel += a2*dt
    
    planet1.pos += planet1.vel
    planet2.pos += planet2.vel
    aa.append(a1[1])
    print( a1,a2 )

def timestep2(planet1,planet2,dt):
    
    dist = np.abs((planet1.pos[0] - planet2.pos[0])**2 + (planet1.pos[1] - planet2.pos[1])**2)
    vec = np.array([planet1.pos[0] - planet2.pos[0], planet1.pos[1] - planet2.pos[1]])
    
    r = (dist + vec)
    a1 = -2*planet2.mass/(r)
    a2 = 2*planet1.mass/(r)
    
    planet1.vel += a1*dt
    planet2.vel += a2*dt
    
    planet1.pos += planet1.vel
    planet2.pos += planet2.vel
    

#%%
plt.close("all")
sun = Planet(1,(0.0,0.0),(0.0,0.0))
earth = Planet(1/333000,(10.0,10.0),(0.0,-0.1))

fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
zoom = 10
for i in range(200):
    
    ax.cla()
    ax.plot(*earth.pos,'o',scalex=False,scaley=False)
    ax.plot(*sun.pos,'o',scalex=False,scaley=False)
    ax.set(xlim=(-zoom,zoom), ylim=(-zoom,zoom))
    clear_output(wait = True)
    plt.pause(0.05)
    timestep2(earth,sun,0.1)
# for _ in range(10):

# print(earth.pos,sun.pos)

#%%
plt.plot(t,aa)