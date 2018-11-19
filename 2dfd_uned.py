# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:24:23 2018

@author: bshri_etybhvn

fixme: formualte the jacobian and rhs 
     : shape functions (--check)
     : gauss points (--check for quads, triangles not checked!)
     : assembly
     : material derivatives3
     : check quadrature rule for 2D quads   (--seems to work, not checked!)
     : tensor (kronecker) product in case of shape functions  (--check for quads)
     : total lagrangian formulation (not implemented yet !)
     : automatically assign dofs and their values from coordinates (line 226)
     : add a function for external force calculation (--line 280)
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
from scipy.interpolate import griddata
##################################################################
rc('font',**{'family':'lmodern','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
rc('xtick',labelsize=22)
rc('ytick',labelsize=22)
matplotlib.rcParams['xtick.major.pad']=10
mx=AutoMinorLocator(10)
my=AutoMinorLocator(10)
##################################################################
class geometry() : #1D geometry
    def __init__(self,Eltype):
        self.tolNR=1.e-8
        if Eltype[0]=='L':
            self.A=8.e-1
            self.B=1.
            ne=10
            self.nLnodes=ne+1
            self.nQnodes=2*ne+1
            self.mu=1.e5
            kap=1.e1*self.mu 
            self.lam=kap+2.*self.mu/3
            self.Po=2.5e5*np.linspace(0,1,100)
            self.epS=2.*np.linspace(1,20,100)
        elif Eltype[0]=='Q':
            self.xlength=2.
            self.ylength=2.
            self.nx=1
            self.ny=1
            self.nDim=2    #No. of dof per node, it is essentially the dimension of the problem
            self.thck=1.
            self.nSteps=10
            self.mu=40.
            kap=1.e1*self.mu 
            self.lam=40.
            self.tolNR=1.e-15
            self.maxiter=20

def meshgenerate():
    xs=0.;ys=0.;
    xe=float(format(xs+geom.xlength+geom.xlength/geom.nx,'.15f'))
#    print('xe=',xe)
    ye=float(format(ys+geom.ylength+geom.ylength/geom.ny,'.15f'))
    stepx=float(format(geom.xlength/geom.nx,'.15f'))
#    print(stepx)
    stepy=float(format(geom.ylength/geom.ny,'.15f'))
    mesh=np.mgrid[xs:xe:stepx,ys:ye:stepy].reshape(2,-1).T
#    print(mesh)
#    Connectivity
    col1=np.hstack((np.arange(geom.ny*i+(i+1),(geom.ny+1)*(i+1),1) for i in range(geom.nx)))
    connectivity=np.vstack((col1,col1+geom.ny+1,col1+1,col1+geom.ny+2)).T-1
    return {'msh':mesh,
            'connv':connectivity}
                   
class GPXi():
    def __init__(self,ordr):
        from numpy.polynomial.legendre import leggauss  #Gauss-Legendre Quadrature for 1D (proxy 2D quads -- check, 3D hex -- not checked)
        self.xi=leggauss(ordr)[0]    #nodes
        self.wght=leggauss(ordr)[1]  #weights

class basis():  # defined on the canonical element (1D : [-1,1], 2D (Q): [-1,1] x [-1,1] )
    def __init__(self,eltype,deg):
        from sympy import Symbol,diff,Array,lambdify,tensorproduct,Matrix,flatten
        if eltype=='L':                                  #L: 1D FE
            z=Symbol('z')
            if deg==2.:     # denotes the number of nodes 
                N=1/2*Array([1-z,1+z])
                dfN=diff(N,z)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
            elif deg==3.:
                N=1/2*Array([z*(z-1),2*(1+z)*(1-z),z*(1+z)])
                dfN=diff(N,z)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
            else:
                raise Exception('Element type not implemented yet')
        elif eltype=='Q':                                #Q: 2D FE : Node-numbering <-- "tensor-product" starting from bottom left corner
            if deg==4.:
                xi=Symbol('xi');eta=Symbol('eta')
                arr1=1/2*Array([1-eta,1+eta]);arr2=1/2*Array([1-xi,1+xi])
                N=tensorproduct(arr1,arr2)
                dfN=Matrix(flatten(diff(N,xi))).col_join(Matrix(flatten(diff(N,eta))))
#                print(N)
                self.Ns=lambdify((xi,eta),flatten(N),'numpy')
                self.dN=lambdify((xi,eta),dfN,'numpy')
            elif deg==9.:
                xi=Symbol('xi');eta=Symbol('eta')
                arr1=Array([eta*(eta-1)/2,(1-eta**2),eta*(eta+1)/2]);arr2=Array([xi*(xi-1)/2,(1-xi**2),xi*(xi+1)/2])
                N=tensorproduct(arr1,arr2);
                dfN=Matrix(flatten(diff(N,xi))).col_join(Matrix(flatten(diff(N,eta))))
                print(N)
                self.Ns=lambdify((xi,eta),flatten(N),'numpy')
                self.dN=lambdify((xi,eta),dfN,'numpy')

class DWDIi():               
    def __init__(self,ndim):
        from sympy import Symbol,diff,lambdify,log,transpose,Matrix,flatten
        I1=Symbol('I1');I2=Symbol('I2');J=Symbol('J');
        W = 1/2*geom.mu*(I1-3)-geom.mu*log(J)+geom.lam/2*(J-1)**2     #change W here to include the modified Neo-Hookean
        dWdI1=diff(W,I1);
        dWdI2=diff(W,I2);
        dWdJ=diff(W,J);
        d2WdI12=diff(dWdI1,I1);
        d2WdJ2=diff(dWdJ,J);
        if ndim==2:
            f12=Symbol('f12');f11=Symbol('f11');f22=Symbol('f22');f21=Symbol('f21')
            f=Matrix([f11,f12,f21,f22]);
            dWdI1=dWdI1.subs(I1,transpose(f).dot(f))                                            #substituting I1, in terms of 
            d2WdI12=d2WdI12.subs(I1,transpose(f).dot(f))
#            dWdI2.subs(0.5*((transpose(f).dot(f)+f[0]**2)**2 
#                            - (f[1]**2 + f[0]**2)**2 
#                            + 2*f[0]**2*(f[1] 
#                            + f[2])**2 
#                            + (f[0]**2 
#                               + f[2]**2)**2 ))           # cannot get expression of I2 directly in terms of vector representation of F
            dWdJ=dWdJ.subs(J,f[0]*f[3]-f[1]*f[2])
            d2WdJ2=d2WdJ2.subs(J,f[0]*f[3]-f[1]*f[2])
            self.DWDI1=lambdify(f,dWdI1,'numpy')
            self.DWDJ=lambdify(f,dWdJ,'numpy')            #output the derivative of invariants at the given F (input) as lambda function
            self.D2WDI12=lambdify(f,d2WdI12,'numpy')
            self.D2WDJ2=lambdify(f,d2WdJ2,'numpy')
def locmat(nodes,de):                     #local stiffness (jacobian) and force (residual) over the reference element
    """
    Storing the Gauss-points, local basis-functions, local gradients, and global gradients. 
    Forming the B-matrix using kron (trick -- check notes!) 
    nodes --- all xs, followed by all ys, Nshp needed for updated lagrangian in the future (not in TL (?))
    """
#    print(nodes.shape)
#    print(de.shape)
    Xi=np.tile(GP.xi,OrdGauss)
    Eta=np.repeat(GP.xi,OrdGauss)
    dof=de.reshape(de.size,-1,1).repeat(len(Xi),axis=-1)  #arranging dof for (dot) product with B (len(Xi) and not len(GP.xi)) !!!
    Wg=np.outer(GP.wght,GP.wght).flatten() 
    Nshp=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),B.Ns(Xi,Eta))      #kron has to be taken on nDim (and not OrdGauss)
#    print(np.array(B.Ns(Xi,Eta)).shape)
    gDshpL=np.array(B.dN(Xi,Eta)).reshape(geom.nDim,int(Eltype[1]),-1)                #local derivatives
    Je=np.einsum('ilk,lj->ijk',gDshpL,nodes.reshape(geom.nDim,-1).T)                  #computing the jacobian
    detJ=(Je[0,0,:]*Je[1,1,:]-Je[0,1,:]*Je[1,0,:])
#    print(detJ)
#    Jeinv=1/detJ*np.array([[Je[1,1,:],-Je[0,1,:]],[-Je[1,0,:],Je[0,0,:]]])              #avoid computing inverse on a loop (--check ?)
    Jeinv=1/detJ*np.einsum('ijk->jik',Je)
    gDshpG=np.einsum('ilk,ljk->ijk',Jeinv,gDshpL)                                     #global derivatives 
    Bmat=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),gDshpG)            
#    B1=Bmat.copy()
    gradU=np.einsum('ilk,ljk->ijk',Bmat,dof)                                          #remember that gradU is never symmetric !!!
#    print(gradU)
    """
    Computing the deformation gradient (F12,F11,F22).T = B*de, and first piola (S) --> (S12,S11,S22), 
    Multiplying by the Gauss-weights, and calculating the element residual
    """
    F=gradU+np.array([1,0,0,1.]).reshape(-1,1,1).repeat(len(Xi),axis=-1)      #computing F as (F11,F11,F22) shape: 3x1xNGP
    detF=F[0]*F[3]-F[1]*F[2]    
    WpI1=dWdIi.DWDI1(*F)
    WpJ=dWdIi.DWDJ(*F)
    WppI1=dWdIi.D2WDI12(*F)
    WppJ=dWdIi.D2WDJ2(*F)
    Finv=np.array([F[3],-F[1],-F[1],F[0]])/detF                               #avoid computing inverse on the loop for the deformation gradient    
#    Helpful variables:     
    S=WpI1*2*F+(WpJ*detF).reshape(1,1,-1)*Finv[np.array([0,2,1,3],int)]       #notice the swap of axes for transpose
    fac=Wg*detJ*geom.thck
    S*=fac                                                                    #multiplying S by the determinant of the jacobian, thickness, and gauss-weights
    res=np.einsum('lik,ljk->ij',Bmat,S)                                       #double contraction along axis 1 and 2 (of B)
    """
    Computing the Consistent Tangent:D= B^T *C *B    <-- Cijkl, check notes
    Cijkl = 4*W''_(I1) Fij Fkl + 2 W'_(I1) delik deljl + J**2*W''_(J) F-1ji F-1lk +J*W'_(J) F-1ji F-1lk - J W'_(J) F-1jk F-1li 
    F = (F11,F12,f21,F22).T
    """
    F11=F[0];F12=F[1];F21=F[2];F22=F[3]
    #This C does not have minor symmetry (relates S to F) , only major symmetry                                             
    
    C1111=4*WppI1*F11*F11+2*WpI1+detF**2*WppJ*Finv[0]**2                                             #scalar addition to multi-dimensional array (--check??) 
    C1112=4*WppI1*F11*F12+detF**2*WppJ*Finv[0]*Finv[2]
    C1121=4*WppI1*F11*F21+detF**2*WppJ*Finv[0]*Finv[1]
    C1122=4*WppI1*F11*F22+detF**2*WppJ*Finv[0]*Finv[3]+detF*WpJ*Finv[0]*Finv[3]-detF*WpJ*Finv[1]*Finv[2]
    C1212=4*WppI1*F12*F12+2*WpI1+detF**2*WppJ*Finv[2]**2
    C1221=4*WppI1*F12*F21+detF**2*WppJ*Finv[2]*Finv[1]+detF*WpJ*(Finv[2]*Finv[1] -Finv[3]*Finv[0])
    C1222=4*WppI1*F12*F22+detF**2*WppJ*Finv[2]*Finv[3]
    C2121=4*WppI1*F21*F21+2*WpI1+detF**2*WppJ*Finv[1]**2
    C2122=4*WppI1*F21*F22+detF**2*WppJ*Finv[2]*Finv[3]+detF*WpJ*(Finv[2]*Finv[3] -Finv[3]*Finv[1])
    C2222=4*WppI1*F22*F22+2*WpI1+detF**2*WppJ*Finv[3]**2
    
    C1111=C1111.flatten()
    C1112=C1112.flatten()
    C1121=C1121.flatten()
    C1122=C1122.flatten()
    C1212=C1212.flatten()
    C1221=C1221.flatten()
    C1222=C1222.flatten()
    C2121=C2121.flatten()
    C2122=C2122.flatten()
    C2222=C2222.flatten()
    
    C=np.array([[C1111,C1112,C1121,C1122],
                [C1112,C1212,C1221,C1222],
                [C1121,C1221,C2121,C2122],
                [C1122,C1222,C2122,C2222]])
    
    D=np.einsum('lik,lpk,pjk->ij',Bmat,C,Bmat)                                #Check the multiplication once for a simple case!
#    print(res)
    return {'K':D,
            'F':res.flatten(),
            'Stress':S,
            'DefGrad':F,
            'InptGlobal':np.einsum('ilj,l->ij',Nshp,nodes),
            'NGP':len(Xi)}

  

Eltype='Q4'
OrdGauss=2           #No. of Gauss-points (in 2D: # of points in each direction counted the same way as local nodes)
geom=geometry(Eltype)
B=basis(Eltype[0],float(Eltype[1]))
GP=GPXi(OrdGauss) 
dWdIi=DWDIi(geom.nDim)
meshxy=meshgenerate()['msh']
conVxy=meshgenerate()['connv']
dof=1.e9*np.ones(meshxy.size) #initializing dofs (displacement of nodes)
intpt=([])
def assembly(disp):
    globK=0.*np.eye(disp.size)
    globF=np.zeros(disp.size)
    for i in range(len(conVxy)):
        elnodes=conVxy[i]
        globdof=np.array([2*elnodes,2*elnodes+1]).flatten()#.T.flatten()
#        print(globdof)
        nodexy=meshxy[elnodes]
        locdisp=disp[globdof]
#        print(locdisp)
        globK[np.ix_(globdof,globdof)] += locmat(nodexy.T.flatten(),locdisp)['K']
#        print(globK)
        globF[globdof] += locmat(nodexy.T.flatten(),locdisp)['F']
        strs=locmat(nodexy.T.flatten(),locdisp)['Stress']
        DG=locmat(nodexy.T.flatten(),locdisp)['DefGrad']
#        calculate strains and integration point coordinates
        ngp=locmat(nodexy.T.flatten(),locdisp)['NGP']
        Strn=(np.einsum('lik,ljk->ijk',DG.reshape(geom.nDim,geom.nDim,-1),DG.reshape(geom.nDim,geom.nDim,-1))-np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1).repeat(ngp,axis=-1))/2
        intpt.append(locmat(nodexy.T.flatten(),locdisp)['InptGlobal'])
    return {'Jac':globK,
            'rhs':globF,
            'S':strs.reshape(geom.nDim,geom.nDim,-1),
            'F':DG.reshape(geom.nDim,geom.nDim,-1),
            'E':Strn,
            'IntP':intpt}
        
#def bcassign(nodes):
#    dofpres=np.array([[1.,1,1,0.02*geom.ylength],
#                      [0.,1,1,0.02*geom.ylength]])    #xcoor,ycoor,dir (x:0, y:1),val 
    
prescribed_dofs=np.array([[0,0.],
                          [1,0],
                          [5,0],
                          [3,0.02*geom.ylength],
                          [7,0.02*geom.ylength]])

dof[(prescribed_dofs[:,0]).astype(int)]=0.
fdof=dof==1.e9                          #free dofs flags: further initialization to zeros needed only for the first step 
nfdof=np.invert(fdof)                   #fixed dofs flags
dof[fdof]=0.
Ks=assembly(dof.reshape(-1,2).T.flatten())['Jac'] 
dofstore=np.zeros(dof.shape)
Fs=assembly(dof.reshape(-1,2).T.flatten())['rhs']
intpt1=assembly(dof)['IntP'][0].T
DfGrn=([]);Strs=([]);LagStrain=([]);
for i in range(10):
    dof[(prescribed_dofs[:,0]).astype(int)]=(i+1)/(geom.nSteps)*prescribed_dofs[:,1]
    Ks1=assembly(dof)['Jac']
    Fs1=assembly(dof)['rhs'] 
    normres=la.norm(Fs1[fdof],2)
    iterNR=0;
    while normres >= geom.tolNR:# and iterNR<=geom.maxiter:        
        dof[fdof] += la.solve(Ks1[np.ix_(fdof,fdof)],-Fs1[fdof])    #external force add (-- not required here, only for this case though)
        Ks1=assembly(dof)['Jac']
        Fs1=assembly(dof)['rhs']
        normres=la.norm(Fs1[fdof],2)
        print('Iter: {}'.format(iterNR))
        iterNR += 1
    DfGrn.append(assembly(dof)['F'])
    Strs.append(assembly(dof)['S'])
    LagStrain.append(assembly(dof)['E'])
    dofstore=np.vstack((dofstore,dof))

DfGrn=np.array(DfGrn);LagStrain=np.array(LagStrain);Strs=np.array(Strs)
#plt.figure()
#from scipy.interpolate import interp2d
#iobd=interp2d(meshxy.T.flatten()[:4],
#             meshxy.T.flatten()[4:],
#             dof[np.array([1,3,5,7])])
#plt.contourf(iobd(meshxy.T.flatten()[:4],
#             meshxy.T.flatten()[4:]),extent=(0,2,0,2))
#plt.colorbar()