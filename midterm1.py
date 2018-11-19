# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:27:27 2018

@author: bhavesh

Importing the numerical analysis libraries
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve,minimize
import pandas as pd
import matplotlib.tri as mptri
from matplotlib.ticker import AutoMinorLocator 
import scipy.sparse as sp
from scipy.interpolate import interp1d,splrep,splder,splev
from scipy.integrate import solve_bvp  #verify the FEM solution
##################################################################
rc('font',**{'family':'lmodern','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
matplotlib.rcParams['lines.linewidth']=3
rc('xtick',labelsize=18)
rc('ytick',labelsize=18)
mx=AutoMinorLocator(10)
my=AutoMinorLocator(10)
##################################################################

"""
Radial Return Mapping Algorithm Documentation:
    Problem 1, Variables and shapes

nNp1    : numpy array: (3,3)
mod_nNp1: float (=la.norm(nNp1,2))
sNp1    : numpy array: (3,3)
sNp1t   : numpy array: (3,3)
Tf      : float 
"""

class params():
    def __init__(self,Tf,E,nu):
        self.mu=E/(2*(1+nu))
        lam=2*nu*self.mu/(1-2*nu)
        self.kap=lam+2./3*self.mu
        self.Ko=35.
        self.Kp=2.5
        self.Hp=4.5
        self.eps1=10.
        self.eps2=5.
        self.A=np.diag([1.,1,3/5])
        self.B=np.array([[0.,4,0],[4,1,0],[0,0,5/3]])
        self.dt=0.05
        self.T=np.linspace(0.,Tf,int(Tf/self.dt)+1)
        Ee=np.eye(3)
        self.Eye=0.5*(np.einsum('ik,jl->ijkl',Ee,Ee)+np.einsum('il,jk->ijkl',Ee,Ee))

Tf=5.;E=15.;nu=0.25
pm=params(Tf,E,nu)

def f(eta,K):
    return la.norm(eta)-(2/3.)**0.5*K

#def main():
#    Initialization of the required matrices
sig=np.zeros((3,3,pm.T.size))
alph=np.zeros(pm.T.size)
q=sig.copy();
Cep=np.zeros((3,3,3,3,pm.T.size));
Cep[:,:,:,:,0]=pm.kap*np.einsum('ij,kl->ijkl',np.eye(3),np.eye(3))+2*pm.mu*(pm.Eye
   -1./3*np.einsum('ij,kl->ijkl',np.eye(3),np.eye(3)))
ep=np.zeros((3,3))
for i in range(1,pm.T.size):
    t=pm.T[i]; epsNp1=pm.A*pm.eps1*t+pm.B*pm.eps2*np.sin(t);
    eNp1=epsNp1 - 1./3*np.eye(3)*np.einsum('ii',epsNp1)
    sNp1=2*pm.mu*(eNp1-ep)
    eta=sNp1-q[:,:,i-1]; xi=eta/la.norm(eta)
    if f(eta,pm.Ko+pm.Kp*alph[i-1]) <=0.:
        q[:,:,i] = q[:,:,i-1]
        alph[i]=alph[i-1]
        dgam=0.
        sig[:,:,i]=sNp1+pm.kap*np.einsum('ii',epsNp1)*np.eye(3)-2*pm.mu*dgam*xi
        thta=1.-2*pm.mu*dgam/la.norm(eta); thtab=0.#(1.+1/(3*pm.mu)*(pm.Kp+pm.Hp))**(-1) + thta - 1.#0.
        Cep[:,:,:,:,i] = pm.kap*np.einsum('ij,kl->ijkl',np.eye(3),
           np.eye(3)) + 2*pm.mu*thta*(pm.Eye-1./3*np.einsum('ij,kl->ijkl',
                 np.eye(3),np.eye(3)))-2*pm.mu*thtab*np.einsum('ij,kl->ijkl',xi,xi)
    else:
        dgam=3*f(eta,pm.Ko+pm.Kp*alph[i-1])/(2.*(pm.Hp+pm.Kp+3*pm.mu))
        alph[i] = alph[i-1]+(2./3)**0.5*dgam
        q[:,:,i] = q[:,:,i-1]+ 2./3*pm.Hp*dgam*xi
        ep += dgam*xi
        sig[:,:,i] = sNp1+pm.kap*np.einsum('ii',epsNp1)*np.eye(3)-2*pm.mu*dgam*xi
        thta=1.-2*pm.mu*dgam/la.norm(eta); thtab=(1.+1/(3*pm.mu)*(pm.Kp+pm.Hp))**(-1) + thta - 1.
        Cep[:,:,:,:,i] = pm.kap*np.einsum('ij,kl->ijkl',np.eye(3),
           np.eye(3)) + 2*pm.mu*thta*(pm.Eye-1./3*np.einsum('ij,kl->ijkl',
                 np.eye(3),np.eye(3)))-2*pm.mu*thtab*np.einsum('ij,kl->ijkl',xi,xi)  
#    print(dgam)

plt.figure(figsize=(10,10))
plt.plot(pm.T,alph,label=r'$\alpha (t)$')
plt.xlabel(r'Time ($t$)',fontsize=18)
plt.ylabel(r'$\bf\alpha(t)$',fontsize=18)
plt.legend(loc=0,fontsize=18)
ax=plt.gca()
ax.xaxis.set_minor_locator(mx)
ax.yaxis.set_minor_locator(my)
plt.grid(True)
plt.savefig('alphP1.eps')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(pm.T,q[0,0,:],label=r'$q_{11}$')
plt.plot(pm.T,q[0,1,:],label=r'$q_{12}$')
plt.plot(pm.T,q[1,1,:],label=r'$q_{22}$')
plt.plot(pm.T,q[2,2,:],label=r'$q_{33}$')
plt.xlabel(r'Time ($t$)',fontsize=20)
plt.ylabel(r'${\bf q}(t)$',fontsize=20)
plt.legend(loc=0,fontsize=20)
ax=plt.gca()
ax.xaxis.set_minor_locator(mx)
ax.yaxis.set_minor_locator(my)
plt.grid(True)
plt.savefig('qplotP1.eps')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(pm.T,sig[0,0,:],label=r'$\sigma_{11}$')
plt.plot(pm.T,sig[0,1,:],label=r'$\sigma_{12}$')
plt.plot(pm.T,sig[1,1,:],label=r'$\sigma_{22}$')
plt.plot(pm.T,sig[2,2,:],label=r'$\sigma_{33}$')
plt.xlabel(r'Time ($t$)',fontsize=20)
plt.ylabel(r'${\bf \sigma}(t)$',fontsize=20)
plt.legend(loc=0,fontsize=20)
ax=plt.gca()
ax.xaxis.set_minor_locator(mx)
ax.yaxis.set_minor_locator(my)
plt.grid(True)
plt.savefig('sigplotP1.eps')
plt.close()     
        
plt.figure(figsize=(10,10))
plt.plot(pm.T,Cep[0,0,0,0,:],label=r'$\mathcal{C}_{1111}$')
plt.plot(pm.T,Cep[1,1,1,1,:],label=r'$\mathcal{C}_{2222}$')
#plt.plot(pm.T,Cep[2,2,2,2,:],label=r'$\mathcal{C}_{3333}$')
plt.plot(pm.T,Cep[0,1,0,1,:],label=r'$\mathcal{C}_{1212}$')
plt.xlabel(r'Time ($t$)',fontsize=20)
plt.ylabel(r'${\mathcal{C}_{ijkl}} (t)$',fontsize=20)
plt.legend(loc=0,fontsize=20)
ax=plt.gca()
ax.xaxis.set_minor_locator(mx)
ax.yaxis.set_minor_locator(my)
plt.grid(True)    
plt.savefig('CijklP1.eps')
plt.close()     