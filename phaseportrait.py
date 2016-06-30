# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:30:06 2016

@author: GURURAJK
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

x=np.linspace(-1,3,5) #general case
#x=np.linspace(-2,2,5)/np.sqrt(2) #Special case - zero mean unit variance
#x=np.ones(10) #Special case - zero variance
y=3*x+5+np.random.normal(0,0.1,len(x))
sx=np.sum(x)
sx2=np.sum(np.square(x))
sy=np.sum(y)
sxy=np.sum(x*y)
def f(Y,t):
    a1,a2=Y
    #return [-sx2*a1-sx*a2+sxy,-sx*a1-len(x)*a2+sy]
    return [-16*a1-8*a2+3,-8*a1-4*a2+1]
y1=np.linspace(-15,15,100)    
y2=np.linspace(-15,15,100)

t=0

Y1,Y2=np.meshgrid(y1,y2)

u,v=np.zeros(Y1.shape),np.zeros(Y2.shape)

NI,NJ=Y1.shape

for i in range(NI):
    for j in range(NJ):
        xx=Y1[i,j]
        yy=Y2[i,j]
        yprime=f([xx,yy],t)
        u[i,j]=yprime[0]/np.sqrt(np.square(yprime[0])+np.square(yprime[1]))
        v[i,j]=yprime[1]/np.sqrt(np.square(yprime[0])+np.square(yprime[1]))
Q=plt.quiver(Y1,Y2,u,v,color='b',headwidth=1,headlength=3)
plt.savefig('plot1.png',dpi=1000)
y0=[[-10,-10],[-5,-10],[0,-10],[5,-10],[10,-10],[-10,10],[-5,10],[0,10],[5,10],[10,10],[-10,-5],[-10,0],[-10,5],[-10,10],[10,-5],[10,0],[10,5],[10,10]]
tspan=np.linspace(0,100,100)
for ya in y0:
    ys=odeint(f,ya,tspan)
    plt.plot(ys[:,0],ys[:,1],'r-')
    plt.plot(ys[-1,0],ys[-1,1],'o')
plt.savefig('plot2.png',dpi=1000)
plt.show()