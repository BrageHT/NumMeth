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

def timestep2(planet,dt):
    
    
    r_x = planet[0].pos[0] - planet[1].pos[0]
    r_y = planet[0].pos[1] - planet[1].pos[1]
    r = np.sqrt(r_x**2 + r_y**2)
    cos_theta = r_x/r
    sin_theta = r_y/r
    # try:
    ang = np.array( [cos_theta, sin_theta] )
    # finally:
    #     ang = np.array([cos_theta, 0])
        
    a1 = -planet[1].mass/(r**2) * ang
    a2 = planet[0].mass/(r**2) * ang
    # print (planet2.vel)
    planet[0].vel += a1*dt
    planet[1].vel += a2*dt
    
    planet[0].pos += planet[0].vel
    planet[1].pos += planet[1].vel
    

#%%
plt.close("all")
sun = Planet(1,(0.0,0.0),(0.0,0.0))
earth = Planet(1,(-10.0,-10.0),(0.0,-0.05)) #1/333000
planets = [earth,sun]
fig = plt.figure(0)
ax = fig.add_subplot(1,1,1)
zoom = 15

for i in range(400):
    
    ax.cla()
    ax.plot(*earth.pos,'o',scalex=False,scaley=False)
    ax.plot(*sun.pos,'o',scalex=False,scaley=False)
    ax.set(xlim=(-zoom,zoom), ylim=(-zoom,zoom))
    clear_output(wait = True)
    plt.pause(0.05)
    timestep2(planets,0.1)
# for _ in range(10):

# print(earth.pos,sun.pos)

#%%
plt.plot(t,aa)