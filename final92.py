# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:24:23 2018

@author: bhavesh

Updated Lagrangian Finite Elasto-plasticity 
4-noded quad elements 
fixme:
    - implement finite deformation using box 9.1 (Simo-Hughes)
    - extrapolate stresses from gauss points to nodes 
    - general framework to export a vtk file from solution 
    - the invariants are those of Febar and not Fe, and hence premultiply 
      accordingly by J**(-2/3)    (Je=J)--> only for this case (partI)
    
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
class geometry() : #nD geometry (quads -- check!, hex -- check!)
    def __init__(self,Eltype,Tf,E,nu):
        self.tolNR=1.e-7
        self.maxiter=100
        self.xtol=0.
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
            self.xlength=1.
            self.ylength=1.
            self.nx=1
            self.ny=1
            self.nDim=2    #No. of dof per node, it is essentially the dimension of the problem
            self.thck=1.
            self.nSteps=100
            self.mu=40.
            kap=1.e1*self.mu 
            self.lam=40.
        elif Eltype[0]=='H':
            self.xlength=3.
            self.ylength=3.
            self.zlength=3.
            self.nx=1
            self.ny=1
            self.nz=1
            self.nDim=3    #No. of dof per node, it is essentially the dimension of the problem
            self.nSteps=100
            self.mu=0.412
            self.lam=self.mu*(E-2*self.mu)/(3*self.mu-E)
            self.kap=self.lam+2./3*self.mu
            self.Ko=12000000000000.
            self.Kp=900. 
            self.Hp=2.5
            self.NGp=8    # number of gauss point, remember to change this if you change integration order 
            self.Sf=1                #Scale factor for deformed plot
            Ee=np.eye(3)
            self.Eye=0.5*(np.einsum('ik,jl->ijkl',Ee,Ee)+np.einsum('il,jk->ijkl',Ee,Ee))
            self.T=np.linspace(0.,Tf,self.nSteps+1)
            self.bta=0.2                        # strain factor
            self.C1=0.165
            self.C2=0.0
            

def meshgn():                  # specify starting point, and step size to generate uniform grid of meshpoints and connectivity
    xs=0.;ys=0.;zs=0.;
    xe=float(format(xs+geom.xlength+geom.xlength/geom.nx,'.15f'))
    ye=float(format(ys+geom.ylength+geom.ylength/geom.ny,'.15f'))
    ze=float(format(zs+geom.zlength+geom.zlength/geom.nz,'.15f'))
    stepx=float(format(geom.xlength/geom.nx,'.15f'))
    stepy=float(format(geom.ylength/geom.ny,'.15f'))
    stepz=float(format(geom.zlength/geom.nz,'.15f'))
    mesh=np.einsum('ijkl->ikjl',np.mgrid[xs:xe:stepx,ys:ye:stepy,zs:ze:stepz]).reshape(3,-1).T
#    connectivity
    col1=np.hstack(((np.arange(geom.nz*i+(i+1),(geom.nz+1)*(i+1),1) for i in range(geom.nx)))) #Layer 1 on the first face
    col1=np.hstack((( (geom.nz+1)*(geom.nx+1) )*i+col1 for i in range(geom.ny)))
    conn=np.vstack((col1,
                    col1+1,
                    col1+geom.nz+1,
                    col1+geom.nz+2,
                    col1+(geom.nz+1)*(geom.nx+1),
                    col1+1+(geom.nz+1)*(geom.nx+1),
                    col1+2+(geom.nz+1)*(geom.nx+1),
                    col1+3+(geom.nz+1)*(geom.nx+1))).T-1   #stack scalar sum to generate connectivity
    return {'msh':mesh,
            'con':conn
            }
             
class GPXi():
    def __init__(self,ordr):
        from numpy.polynomial.legendre import leggauss  #Gauss-Legendre Quadrature for 1D (proxy 2D quads -- check, 3D hex -- not checked)
        self.xi=leggauss(ordr)[0]    #nodes
        self.wght=leggauss(ordr)[1]  #weights

class basis():  # defined on the canonical element (1D : [-1,1], 2D (Q): [-1,1] x [-1,1], 3D (H): [-1,1]^3 )
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
                N=tensorproduct(arr1,arr2);
                dfN=Matrix(flatten(diff(N,xi))).col_join(Matrix(flatten(diff(N,eta))))
                self.Ns=lambdify((xi,eta),flatten(N),'numpy')
                self.dN=lambdify((xi,eta),dfN,'numpy')
            elif deg==9.:
                xi=Symbol('xi');eta=Symbol('eta')
                arr1=Array([eta*(eta-1)/2,(1-eta**2),eta*(eta+1)/2]);arr2=Array([xi*(xi-1)/2,(1-xi**2),xi*(xi+1)/2])
                N=tensorproduct(arr1,arr2);
                dfN=Matrix(flatten(diff(N,xi))).col_join(Matrix(flatten(diff(N,eta))))
                self.Ns=lambdify((xi,eta),flatten(N),'numpy')
                self.dN=lambdify((xi,eta),dfN,'numpy')
        elif eltype=='H':
            if deg==8.:
                xi=Symbol('xi');eta=Symbol('eta');rho=Symbol('rho')
                arr1=1/2*Array([1-eta,1+eta]);arr2=1/2*Array([1-xi,1+xi]);arr3=1/2*Array([1-rho,1+rho])
                N=tensorproduct(arr1,arr2,arr3);
                dfN=Matrix(flatten(diff(N,xi))).col_join(Matrix(flatten(diff(N,eta)))).col_join(Matrix(flatten(diff(N,rho))))
                self.Ns=lambdify((xi,eta,rho),flatten(N),'numpy')
                self.dN=lambdify((xi,eta,rho),dfN,'numpy')
        else:
            raise Exception('Only 1D, 2D and 3D continuum elements implemented')

class DWDIi():            # the substitution changes for a 3D element       
    def __init__(self,ndim):
        from sympy import Symbol,diff,lambdify,log,transpose,Matrix
        I1=Symbol('I1');J=Symbol('J');I2=Symbol('I2')
        W = geom.C1*(J**(-2./3)*I1-3)+geom.C2*(J**(-4./3)*I2-3.)+geom.kap/2.*(1/2.*(J**2-1.)-log(J))     #change W here to include the modified Neo-Hookean
        dWdI1=diff(W,I1);dWdI2=diff(W,I2);dWdJ=diff(W,J);
        d2WdI12=diff(dWdI1,I1);d2WdI2dI1=diff(dWdI1,I2);d2WdJdI1=diff(dWdI1,J);
        d2WdI1dI2=diff(dWdI2,I1);d2WdI22=diff(dWdI2,I2);d2WdJdI2=diff(dWdI2,J);
        d2WdI1dJ=diff(dWdJ,I1);d2WdI2dJ=diff(dWdJ,I2);d2WdJ2=diff(dWdJ,J);
        if ndim==2 or ndim==3:
            f11=Symbol('f11');f12=Symbol('f12');f13=Symbol('f13');
            f21=Symbol('f21');f22=Symbol('f22');f23=Symbol('f23');
            f31=Symbol('f31');f32=Symbol('f32');f33=Symbol('f33')
            f=Matrix([f11,f12,f13,f21,f22,f23,f31,f32,f33]);
            I1subs=f[0]**2 + f[1]**2 + f[2]**2 + f[3]**2 + f[4]**2 + f[5]**2 + f[6]**2 + f[7]**2 + f[8]**2;
            I2subs=0.5*(I1subs**2-((f[0]**2 + f[3]**2 + f[6]**2)**2 + 2*(f[0]*f[1] + f[3]*f[4] + f[6]*f[7])**2 + (f[1]**2 + f[4]**2 + f[7]**2)**2 + 2*(f[0]*f[2] + f[3]*f[5] + f[6]*f[8])**2 + 2*(f[1]*f[2] + f[4]*f[5] + f[7]*f[8])**2 + (f[2]**2 + f[5]**2 + f[8]**2)**2));
            Jsubs=f[0]*(f[4]*f[8]-f[7]*f[5])-f[1]*(f[3]*f[8]-f[6]*f[5])+f[2]*(f[3]*f[7]-f[6]*f[4])

            dWdI1=dWdI1.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])                                           #substituting I1, in terms of 
            dWdI2=dWdI2.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            dWdJ=dWdJ.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI12=d2WdI12.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI12=d2WdI12.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI2dI1=d2WdI2dI1.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdJdI1=d2WdJdI1.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI1dI2=d2WdI1dI2.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI22=d2WdI22.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdJdI2=d2WdJdI2.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI1dJ=d2WdI1dJ.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdI2dJ=d2WdI2dJ.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])
            d2WdJ2=d2WdJ2.subs([(I1,I1subs),(I2,I2subs),(J,Jsubs)])

#            Convert to lambda functions

            self.DWDI1=lambdify(f,dWdI1,'numpy')
            self.DWDI2=lambdify(f,dWdI2,'numpy')            #output the derivative of invariants at the given F (input) as lambda function
            self.DWDJ=lambdify(f,dWdJ,'numpy')
            
            self.D2WDI12=lambdify(f,d2WdI12,'numpy')
            self.D2WDI2DI1=lambdify(f,d2WdI2dI1,'numpy')
            self.D2WDJDI1=lambdify(f,d2WdJdI1,'numpy')
            self.D2WDI1DI2=lambdify(f,d2WdI1dI2,'numpy')
            self.D2WDI22=lambdify(f,d2WdI22,'numpy')
            self.D2WDJDI2=lambdify(f,d2WdJdI2,'numpy')
            self.D2WDI1DJ=lambdify(f,d2WdI1dJ,'numpy')
            self.D2WDI2DJ=lambdify(f,d2WdI2dJ,'numpy')
            self.D2WDJ2=lambdify(f,d2WdJ2,'numpy')

def locmat(nodes,de,qe,alphe,eplast):                     #local stiffness (jacobian) and force (residual) over the reference element
    """
    Storing the Gauss-points, local basis-functions, local gradients, and global gradients. 
    Forming the B-matrix using kron (trick -- check notes!) 
    nodes --- all xs, followed by all ys followed by all zs
    de --- all dx, followed by all dy, followed by all dz
    """
    Xi=np.tile(np.repeat(GP.xi,OrdGauss),OrdGauss)
    Eta=np.repeat(GP.xi.T,2*OrdGauss).flatten()
    Rho=np.tile(GP.xi,2*OrdGauss)                         #Generating Gauss-Points through numpy (--check with jupyter notebook)
    dof=de.reshape(de.size,-1,1).repeat(len(Xi),axis=-1)  #arranging dof for (dot) product with B (len(Xi) and not len(GP.xi)) !!!
    Wg=np.repeat(GP.wght,2*OrdGauss)
    Nshp=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),B.Ns(Xi,Eta,Rho))      #kron has to be taken on the nDim (and not OrdGauss)
    gDshpL=np.array(B.dN(Xi,Eta,Rho)).reshape(geom.nDim,int(Eltype[1]),-1)                #local derivatives
    
    Je=np.einsum('ilk,lj->ijk',gDshpL,nodes.reshape(geom.nDim,-1).T)                 #computing the jacobian (remains the same, even for 3D ? -- check ?)
    detJ=np.dstack(la.det(Je[:,:,i]) for i in range(Xi.size))        # 1x1xNgP       # try making it faster by removing the generator
    Jeinv=np.dstack(la.inv(Je[:,:,i]) for i in range(Xi.size))       # 3x3xNgP       #avoid computing inverse on a loop (--check ?)
    gDshpG=np.einsum('ilk,ljk->ijk',Jeinv,gDshpL)                                    #global derivatives  (remains the same, even for 3D ? )
    Bmat=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),gDshpG)            
    gradU=np.einsum('ilk,ljk->ijk',Bmat,dof)                         # 9x1xNgP                   #remember that gradU is never symmetric !!!
    """
    Computing the deformation gradient (F11,F12,F13...,F33).T = B*de, and first piola (S) --> (S11,S12,S13,.....,S33), 
    Multiplying by the Gauss-weights, and calculating the element residual (res)
    """
    F=gradU+np.eye(geom.nDim).reshape(-1,1,1).repeat(len(Xi),axis=-1)      #convert to 3x3xNgP and the take det(F) and inv(F)
    detF=np.dstack(la.det(F.reshape(geom.nDim,geom.nDim,-1)[:,:,i] )for i in range(Xi.size))    # 1x1xNgP 
    
#    Calculating the derivatives of W wrt invariants (for stress)
    WpI1=dWdIi.DWDI1(*F)
    WpI2=dWdIi.DWDI2(*F)
    WpJ=dWdIi.DWDJ(*F)
    
#    Calculating the derivatives of W wrt invariants (for modulus tangent)
    WppI1=dWdIi.D2WDI12(*F)
    WppI2I1=dWdIi.D2WDI2DI1(*F)
    WppJI1=dWdIi.D2WDJDI1(*F)
    WppI1I2=dWdIi.D2WDI1DI2(*F)
    WppI2=dWdIi.D2WDI22(*F)
    WppJI2=dWdIi.D2WDJDI2(*F)
    WppI1J=dWdIi.D2WDI1DJ(*F)
    WppI2J=dWdIi.D2WDI2DJ(*F)
    WppJ=dWdIi.D2WDJ2(*F)
    Finv=(np.dstack(la.inv(F.reshape(geom.nDim,geom.nDim,-1)[:,:,i]) for i in range(Xi.size))).reshape(-1,1,geom.NGp)                #avoid computing inverse on the loop for the deformation gradient    
    F3b3=F.reshape(geom.nDim,geom.nDim,-1); Finv3b3=Finv.reshape(geom.nDim,geom.nDim,-1)  #reshaping F and Finv for multiplication
    I1f=np.einsum('ijk,ijk->k',F3b3,F3b3)        #calculating the first invariant at GP
    
    FinvT=np.einsum('ijk->jik',Finv.reshape(geom.nDim,geom.nDim,-1))
    FFt=(np.einsum('iqg,qjg->ijg',np.einsum('ipg,qpg->iqg',F3b3,F3b3),F3b3)).reshape(-1,1,geom.NGp)
    S=WpI1*2*F+(WpJ*detF)*FinvT.reshape(-1,1,geom.NGp)+2*WpI2*(I1f*F-FFt) - (2*geom.mu*eplast).reshape(-1,1,geom.NGp) #[np.array([0,3,6,1,4,7,2,5,8],int)]       #notice the swap of axes for transpose (of the inverse)
    """
    RADIAL RETURN MAPPING ALGORITHM
    Checking if the stress-state is admissible, f(....) <= 0
    Check for the yield surface, the backstress and the plastic strain
    Compute Cep_ijkl separately
    """
    epl=eplast.copy()
    eye3d=np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1).repeat(geom.NGp,axis=-1)
    epSS=0.5*(F3b3+np.einsum('ijk->jik',F3b3)-2.*eye3d)
    edev=epSS-1./3*np.einsum('iik->k',epSS)*eye3d  #    print(epSS[:,:,0])
    strial=2*geom.mu*(edev-eplast)                 # trial deviator
    sEff=strial-qe
    ftrial = la.norm(sEff,axis=(0,1))-(2./3)**0.5*(geom.Ko+geom.Kp*alphe)   # Yield-surface boundary (NGp)
    plsTidx = ftrial > 0.    # Boolean array to keep track of plasticity(True-Plastic, False-Elastic!)
#    Assume elastic response and then check later on
    dgam=np.zeros(geom.NGp)   
    nNp1=np.zeros((geom.nDim,geom.nDim,geom.NGp))
    thta=np.ones(geom.NGp)
    thtab=np.zeros(geom.NGp)
    dgam[plsTidx]=3*ftrial[plsTidx]/(2*(geom.Hp+geom.Kp+3*geom.mu))    #evolve dgam only at the yielded gauss points
    nNp1[:,:,plsTidx]=(sEff[:,:,plsTidx]/la.norm(sEff[:,:,plsTidx],axis=(0,1)))
#    S = (strial+geom.kap*np.einsum('iik->k',epSS)*eye3d - 2*geom.mu*dgam*nNp1).reshape(-1,1,geom.NGp)

    thta[plsTidx]=1.-2.*geom.mu*dgam[plsTidx]/la.norm(sEff[:,:,plsTidx],axis=(0,1))
    thtab[plsTidx]=((1.+1/(3*geom.mu)*(geom.Kp+geom.Hp))**(-1) + thta[plsTidx] - 1.).copy()
#    S[:,:,plsTidx] = ((strial + geom.kap*np.einsum('iik->k',epSS)*eye3d - 2*geom.mu*(dgam*nNp1)).reshape(-1,1,geom.NGp))[:,:,plsTidx]
    alphe+= (2./3)**0.5*dgam
    qe+= 2./3*geom.Hp*dgam*nNp1            #update qe in place and store in n+1 
    epl += dgam*nNp1
#    S = (strial+geom.kap*np.einsum('iik->k',epSS)*eye3d - 2*geom.mu*dgam*nNp1).reshape(-1,1,geom.NGp)
    fac=Wg*detJ
    res=np.einsum('lik,ljk->ij',Bmat,fac*S)             #double contraction along axis 1 and 2 (of B)
    """
    Computing the Consistent Tangent: B^T *C *B    <-- Cijkl, check notes
    Cijkl = 4*W''_(I1) Fij Fkl + 2 W'_(I1) delik deljl + J**2*W''(J) F-1ji F-1lk +J*W'(J) F-1ji F-1lk - J W'_(J) F-1jk F-1li 
    """
#    Helpful variables:     
    #This C does not have minor symmetry (relates S to F) , only major symmetry                                             
#    
    Bta=2*(I1f*F3b3-np.einsum('iqk,qjk->ijk',np.einsum('ipk,qpk->iqk',F3b3,F3b3),F3b3))       #check with hand 
    DFDWDI1=WppI1*2*F3b3+WppI2I1*Bta+WppJI1*detF*np.einsum('ijk->jik',Finv3b3)
    DFDWDI2=WppI1I2*2*F3b3+WppI2*Bta+WppJI2*detF*np.einsum('ijk->jik',Finv3b3)
    DFDWDJ=WppI1J*2*F3b3+WppI2J*Bta+WppJ*detF*np.einsum('ijk->jik',Finv3b3)
    
    IDGp = (np.einsum('ik,jl->ijkl',np.eye(geom.nDim),np.eye(geom.nDim))[:,:,:,:,np.newaxis]).repeat(geom.NGp,axis=-1)
#    EyeGp = geom.Eye[:,:,:,:,np.newaxis].repeat(geom.NGp,axis=-1)    

    C=(np.einsum('ijg,klg->ijklg',2*F3b3,DFDWDI1)+2*WpI1*IDGp+np.einsum('ijg,klg->ijklg',Bta,DFDWDI2)
    +2*WpI2*(2*np.einsum('ijg,klg->ijklg',F3b3,F3b3)+I1f*IDGp
            -np.einsum('ilg,kjg->ijklg',F3b3,F3b3)
            -np.einsum('ljg,ikg->ijklg',np.einsum('qlg,qjg->ljg',F3b3,F3b3),(np.eye(geom.nDim)[:,:,np.newaxis]).repeat(geom.NGp,axis=-1))
            -np.einsum('ikg,jlg->ijklg',np.einsum('ipg,kpg->ikg',F3b3,F3b3),(np.eye(geom.nDim)[:,:,np.newaxis]).repeat(geom.NGp,axis=-1)))
    +detF*np.einsum('jig,klg->ijklg',Finv3b3,DFDWDJ)+WpJ*(detF*np.einsum('jig,lkg->ijklg',Finv3b3,Finv3b3)
    -detF*np.einsum('jkg,lig->ijklg',Finv3b3,Finv3b3))).reshape(9,9,-1,order='A')
    Cauch=(1./detF*np.einsum('ikg,jkg->ijg',S.reshape(3,3,-1),F3b3)).reshape(-1,1,geom.NGp)
#    print(WppI1)
#
#    Some useful variables and shapes of previous variables: 
#    nNp1 = 3x3xNGp
#    dgam = NGp     (hoping that scalar multiplication along axis3 works!)
#    thta = NGp
#    thtab= NGp
#    la.norm(sEff,2,axis=(0,1)) = NGp 
#    Now start doing einsum to generate Cep, by first introducing a 
#    new axis at the end that keeps track of Gauss-points for the above
#    n-d arrays
#    IdGp  : del ij del kl x NGp
#    EyeGP : Eye extended along 5th dimension (fourth order identity tensor repeat along fifth dimension)
#    
    
#    C=(2*geom.mu*EyeGp+geom.lam*IDGp).reshape(9,9,-1,order='A')
#    Cep=(geom.kap*IDGp+2*geom.mu*thta*(EyeGp-1./3*IDGp) - 2.*geom.mu*thtab*np.einsum('ijg,klg->ijklg',nNp1,nNp1)).reshape(9,9,-1,order='A')
    Cep=C.copy()
    Cep1111=Cep[0,0,0];Cep1212=Cep[1,1,0];Cep2222=Cep[4,4,0];Cep3333=Cep[-1,-1,0]

    D=np.einsum('lik,lpk,pjk->ij',Bmat,Cep*fac,Bmat)                                #Check the multiplication once for a simple case!
    IntpG=np.einsum('ilj,l->ij',Nshp,nodes)
    return D,res.flatten(),Cauch,F,IntpG,qe,alphe,epl,Cep1111,Cep2222,Cep1212,Cep3333

Eltype='H8'
OrdGauss=2           #No. of Gauss-points (in 2D: # of points in each direction counted the same way as local nodes)
Tf = 1.0
globE=1.199
globnu=0.25
geom=geometry(Eltype,Tf,globE,globnu)
B=basis(Eltype[0],float(Eltype[1]))
GP=GPXi(OrdGauss) 
dWdIi=DWDIi(geom.nDim)

meshxyz=meshgn()['msh']
conVxyz=meshgn()['con']

dof=1.e9*np.ones(meshxyz.size) #initializing dofs (displacement of nodes)

def assembly(disp,q,alph,Ep):
    globK=0.*np.eye(disp.size)
    globF=np.zeros(disp.size) 
    strs=np.zeros((len(conVxyz),9,1,geom.NGp))
    Strn=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp))
    DG=np.zeros((len(conVxyz),9,1,geom.NGp))
    epsStrn=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp)) #geom.nDim=3 in this case
    for i in range(len(conVxyz)):
        elnodes=conVxyz[i]
        ep=Ep[i]
        qe=q[i]
        alphe=alph[i]
        globdof=np.array([3*elnodes,3*elnodes+1,3*elnodes+2]).flatten()#.T.flatten()  : gets the elemental dofs in the order u1.....,w3
        nodexy=meshxyz[elnodes]
        locdisp=disp[globdof]
        elemK,elemF,strs[i],DG[i],gipt,q[i],alph[i],Ep[i],C1111,C2222,C1212,C3333=locmat(nodexy.T.flatten(),locdisp,qe,alphe,ep)
        globK[np.ix_(globdof,globdof)] += elemK
        globF[globdof] += elemF  
        intpt.append(gipt)
#        Calculate strains and integration point coordinates
        Strn[i]=(np.einsum('lik,ljk->ijk',DG.reshape(geom.nDim,geom.nDim,-1),DG.reshape(geom.nDim,geom.nDim,-1))-np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1).repeat(8,axis=-1))/2
        epsStrn[i]=(np.einsum('ijk->jik',DG.reshape(geom.nDim,geom.nDim,-1))+DG.reshape(geom.nDim,geom.nDim,-1)-2*np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1).repeat(8,axis=-1))/2
        strs=strs.reshape(len(conVxyz),geom.nDim,geom.nDim,-1)    
        DG=DG.reshape(len(conVxyz),geom.nDim,geom.nDim,-1)
    return globK,globF,strs,DG,Strn,epsStrn,intpt,q,Ep,alph,C1111,C1212,C2222,C3333       

prescribed_dofs=np.array([[0,0.],
                          [1,0],
                          [2,0],
                          [4,0],
                          [5,0],
                          [6,0],
                          [7,0],
                          [8,0],
                          [10,0],
                          [11,0],
                          [12,0],
#                          [13,0],
                          [14,0],
#                          [16,0],
                          [17,0],
                          [18,0],
#                          [19,0],
                          [20,0.],
#                          [22,0],
                          [23,0],
                          [3,geom.bta*geom.xlength],
                          [9,geom.bta*geom.xlength],
                          [15,geom.bta*geom.xlength],
                          [21,geom.bta*geom.xlength]])           #apply 2% strain

dof[(prescribed_dofs[:,0]).astype(int)]=0.
fdof=dof==1.e9                          #free dofs flags: further initialization to zeros needed only for the first step 
nfdof=np.invert(fdof)                   #fixed dofs flags
dof[fdof]=0.
lineardof=dof.copy()
DfGrn=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp,geom.nSteps+1))                  # Deformation Gradient
Strs=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp,geom.nSteps+1))                   # First-PK Stress
LagStrain=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp,geom.nSteps+1))              # Lagrangian Strain
epss=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp,geom.nSteps+1))
intpt=([])
qGlobal=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp,geom.nSteps+1))    #check dimensions done (@)
alphGlobal=np.zeros((len(conVxyz),geom.NGp,geom.nSteps+1))                     #just a scalar
Eplastic=np.zeros((len(conVxyz),geom.nDim,geom.nDim,geom.NGp,geom.nSteps+1))                     #initialize the plastic strain
_,_,_,_,_,_,intpt1,_,_,_,_,_,_,_=assembly(dof,qGlobal[:,:,:,:,0],alphGlobal[:,:,0],Eplastic[:,:,:,:,0])    #global gauss-point locations (?)
intpt1=intpt1[0].T
dofstore=np.zeros(dof.shape);
Ks3,_,_,_,_,_,_,_,_,_,_,_,_,_=assembly(dof,qGlobal[:,:,:,:,0],alphGlobal[:,:,0],Eplastic[:,:,:,:,0])  #linear stiffness (check!)

#Initialize residuals to empty lists
res4=([]);res10=([]);res25=([]);res50=([]);res80=([])
Cep1111=np.zeros(geom.T.size);
Cep1111[0]=geom.kap+geom.C1*8./3
Cep1212=np.zeros(geom.T.size);
Cep1212[0]=2*geom.C1
Cep2222=np.zeros(geom.T.size);
Cep2222[0]=geom.kap+geom.C1*8./3
Cep1313=np.zeros(geom.T.size);
Cep1313[0]=2*geom.C1
Cep3333=np.zeros(geom.T.size);
Cep3333[0]=geom.kap+geom.C1*8./3
#Load-Steps
for istep in range(geom.nSteps):
    print('Step: {}'.format(istep+1))
    dof[(prescribed_dofs[:,0]).astype(int)]=(istep+1)/(geom.nSteps)*prescribed_dofs[:,1]
    Ks1,Fs1,strs,dfg,lstrn,epsstrn,_,qglb,epls,alphglb,C1111temp,C1212temp,C2222temp,C3333temp=assembly(dof,qGlobal[:,:,:,:,istep],alphGlobal[:,:,istep],Eplastic[:,:,:,:,istep])
    normres=1;
#    print(la.norm(Fs1[fdof],2))
    if istep==3:
        res4.append(Fs1[fdof])    #zero-based indexing 
    elif istep==9:
        res10.append(Fs1[fdof])
    elif istep==24:
        res25.append(Fs1[fdof])
    elif istep==49:
        res50.append(Fs1[fdof])
    elif istep==79:
        res80.append(Fs1[fdof])
    normres=la.norm(Fs1[fdof],2);
    res0=normres.copy()
    iterNR=0;
#    Newton-Increments
    while normres > geom.tolNR*res0 and iterNR<=geom.maxiter:     
        print('Iter: {}'.format(iterNR+1))
        dof[fdof] += la.solve(Ks1[np.ix_(fdof,fdof)],-Fs1[fdof])    #external force add (-- not required here, only for this case though)
        Ks1,Fs1,strs,dfg,lstrn,epsstrn,_,qglb,epls,alphglb,_,_,_,_=assembly(dof,qGlobal[:,:,:,:,istep],alphGlobal[:,:,istep],Eplastic[:,:,:,:,istep]) 
        if istep==3:
            res4.append(Fs1[fdof])    #zero-based indexing 
        elif istep==9:
            res10.append(Fs1[fdof])
        elif istep==24:
            res25.append(Fs1[fdof])
        elif istep==49:
            res50.append(Fs1[fdof])
        elif istep==79:
            res80.append(Fs1[fdof])
        normres=la.norm(Fs1[fdof],2) 
#        print(la.norm(Fs1[fdof],2))
#        _,_,_,_,_,_,_,qGlobal[:,:,:,:,istep],Eplastic[:,:,:,:,istep],alphGlobal[:,:,istep],_,_,_,_ = assembly(dof,qGlobal[:,:,:,:,istep],alphGlobal[:,:,istep],Eplastic[:,:,:,:,istep])
        iterNR += 1
    dofstore=np.vstack((dofstore,dof))
    Strs[:,:,:,:,istep+1],DfGrn[:,:,:,:,istep+1],LagStrain[:,:,:,:,istep+1],epss[:,:,:,:,istep+1],_,qGlobal[:,:,:,:,istep+1],Eplastic[:,:,:,:,istep+1],alphGlobal[:,:,istep+1],Cep1111[istep+1],Cep1212[istep+1],Cep2222[istep+1],Cep3333[istep+1]=strs,dfg,lstrn,epsstrn,_,qglb,epls,alphglb,C1111temp,C1212temp,C2222temp,C3333temp
###############################################################################
excelWrite=False
#Output the Residual in respective excel sheets
if excelWrite:
    pd.DataFrame(np.array(res4).T).to_excel('res4.xlsx',index=False,header=False)
    pd.DataFrame(np.array(res4).T).to_excel('res10.xlsx',index=False,header=False)
    pd.DataFrame(np.array(res4).T).to_excel('res25.xlsx',index=False,header=False)
    pd.DataFrame(np.array(res4).T).to_excel('res50.xlsx',index=False,header=False)
    pd.DataFrame(np.array(res4).T).to_excel('res80.xlsx',index=False,header=False)
plot_it=True
if plot_it:
    plt.figure(figsize=(10,10))
    plt.plot(DfGrn[0,0,2,-1,:],Strs[0,0,0,-1,:],label=r'$\sigma_{11}$')
    plt.plot(DfGrn[0,0,2,-1,:],Strs[0,-1,-1,-1,:],label=r'$\sigma_{33}$')
    plt.plot(DfGrn[0,0,2,-1,:],Strs[0,1,1,-1,:],label=r'$\sigma_{22}$')
    plt.plot(DfGrn[0,0,2,-1,:],Strs[0,0,-1,-1,:],label=r'$\sigma_{13}$')
    plt.xlabel(r'$\gamma$ (Stretch)',fontsize=20)
    plt.ylabel(r'$\sigma_{ij}$ ',fontsize=20)
    plt.legend(loc=0,fontsize=20)
    plt.legend(loc=0,fontsize=18)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax=plt.gca()
    ax.xaxis.set_minor_locator(mx)
    ax.set_title(r'Stress vs Stretch',fontsize=20)
    ax.yaxis.set_minor_locator(my)
    plt.grid(True)
#    plt.savefig('StrsStrchP4b1m.eps')
#    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(geom.T,Strs[0,0,0,-1,:],label=r'$\sigma_{11}$')
    plt.plot(geom.T,Strs[0,-1,-1,-1,:],label=r'$\sigma_{33}$')
    plt.plot(geom.T,Strs[0,1,1,-1,:],label=r'$\sigma_{22}$')
    plt.plot(geom.T,Strs[0,0,-1,-1,:],label=r'$\sigma_{13}$')
    plt.xlabel(r'$t$ (Time)',fontsize=20)
    plt.ylabel(r'$\sigma_{ij}$ ',fontsize=20)
    plt.legend(loc=0,fontsize=20)
    plt.legend(loc=0,fontsize=18)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax=plt.gca()
    ax.xaxis.set_minor_locator(mx)
    ax.set_title(r'Stress vs time',fontsize=20)
    ax.yaxis.set_minor_locator(my)
    plt.grid(True)
#    plt.savefig('StrsTimeP4b1m.eps')
#    plt.close()
    
###############################################################################
#Visualization using Scipy.interpolate's n-D griddata
#from scipy.interpolate import griddata
#mgD=np.mgrid[-1:1.25:0.25,-1:1.25:0.25,-1:1.25:0.25]
#Strs33=griddata(intpt1,Strs[-1,-1,-1,:],
#                np.einsum('ijkl->ikjl',mgD).reshape(3,-1).T,method='nearest')        #last step stress33 component 
#xx=np.arange(-1.,1.25,0.25);yy=xx.copy()

###############################################################################
vtKwrite=False
if vtKwrite:
    #Create vtk data file for visualization in Paraview
    filename='CIHW3.vtk'
    filenamedef='CIHW3def.vtk'
    filestrsstrn='CIHW31.vtk'
    name='Hex8'
    if name=='Hex8':
        ParaviewID=12
    output_file=open(filename,'w')
    output_filedef=open(filenamedef,'w')
    output_filestrsstrn=open(filestrsstrn,'w')
    
    output_file.write('# vtk DataFile Version 2.0\n')
    output_file.write('%s\n' %name);
    output_file.write('ASCII\n');
    output_file.write('\n');
    #nodes
    output_file.write('DATASET UNSTRUCTURED_GRID\n')
    output_file.write('POINTS %d float\n' %(len(meshxyz))); #list of nodes
    for i in range(len(meshxyz)):
        output_file.write('%15.10f %15.10f %15.10f \n'%( meshxyz[i,0],
                                                         meshxyz[i,1],
                                                         meshxyz[i,2]))
    
    output_file.write('\n');
    output_file.write('CELLS %d %d \n' %(len(conVxyz), (len(conVxyz[0])+1)*len(conVxyz))) #connectivity
    output_file.write('8 %d %d %d %d %d %d %d %d'%(conVxyz[0,0],
                                                   conVxyz[0,2],
                                                   conVxyz[0,6],
                                                   conVxyz[0,4],
                                                   conVxyz[0,1],
                                                   conVxyz[0,3],
                                                   conVxyz[0,7],
                                                   conVxyz[0,5]
                                                   ))
    output_file.write('\n')
    output_file.write('CELL_TYPES %d\n' %(len(conVxyz)))
    a=ParaviewID*np.ones(len(conVxyz),int) 
    output_file.write('\n'.join(map(str,a)))
    output_file.write('\nPOINT_DATA %d\n' %len(meshxyz)) #data fields  
    output_file.write('SCALARS U float 3\n')
    output_file.write('Lookup_table default\n')
    
    for i in range (len(meshxyz)):
    	output_file.write('%12.10f %12.10f %12.10f\n' %(dofstore[-1].reshape(-1,3)[i,0],
                                                        dofstore[-1].reshape(-1,3)[i,1],
                                                        dofstore[-1].reshape(-1,3)[i,2]))
    output_file.write('\n') 
    output_file.close()
    
    # Deformed data
    output_filedef.write('# vtk DataFile Version 2.0\n')
    output_filedef.write('%s\n' %name);
    output_filedef.write('ASCII\n');
    output_filedef.write('\n');
    #nodes
    output_filedef.write('DATASET UNSTRUCTURED_GRID\n')
    output_filedef.write('POINTS %d float\n' %(len(meshxyz))); #list of nodes
    
    for i in range(len(meshxyz)):
        output_filedef.write('%15.10f %15.10f %15.10f \n'%( meshxyz[i,0]+geom.Sf*dofstore[-1].reshape(-1,3)[i,0],
                                                            meshxyz[i,1]+geom.Sf*dofstore[-1].reshape(-1,3)[i,1],
                                                            meshxyz[i,2]+geom.Sf*dofstore[-1].reshape(-1,3)[i,2]))
    
    output_filedef.write('\n');
    output_filedef.write('CELLS %d %d \n' %(len(conVxyz), (len(conVxyz[0])+1)*len(conVxyz))) #connectivity
    output_filedef.write('8 %d %d %d %d %d %d %d %d'%(conVxyz[0,0],
                                                   conVxyz[0,2],
                                                   conVxyz[0,6],
                                                   conVxyz[0,4],
                                                   conVxyz[0,1],
                                                   conVxyz[0,3],
                                                   conVxyz[0,7],
                                                   conVxyz[0,5]
                                                   ))
    output_filedef.write('\n')
    output_filedef.write('CELL_TYPES %d\n' %(len(conVxyz)))
    a=ParaviewID*np.ones(len(conVxyz),int) 
    output_filedef.write('\n'.join(map(str,a)))
    output_filedef.write('\nPOINT_DATA %d\n' %len(meshxyz)) #data fields  
    output_filedef.write('SCALARS Udef float 3\n')
    output_filedef.write('Lookup_table default\n')
    
    for i in range (len(meshxyz)):
    	output_filedef.write('%12.10f %12.10f %12.10f\n' %(dofstore[-1].reshape(-1,3)[i,0],
                                                        dofstore[-1].reshape(-1,3)[i,1],
                                                        dofstore[-1].reshape(-1,3)[i,2]))
    output_filedef.write('\n') 
    
    output_filedef.close()
    
    #Stress-Strain File 
    
    output_filestrsstrn.write('# vtk DataFile Version 2.0\n')
    output_filestrsstrn.write('%s\n' %name);
    output_filestrsstrn.write('ASCII\n');
    output_filestrsstrn.write('\n');
    #nodes
    output_filestrsstrn.write('DATASET UNSTRUCTURED_GRID\n')
    output_filestrsstrn.write('POINTS %d float\n' %(len(meshxyz))); #list of integration point
    for i in range(len(intpt1)):
        output_filestrsstrn.write('%15.10f %15.10f %15.10f \n'%( intpt1[i,0],
                                                         intpt1[i,1],
                                                         intpt1[i,2]))      #write global integration point
    
    output_filestrsstrn.write('\n');
    output_filestrsstrn.write('CELLS %d %d \n' %(len(conVxyz), (len(conVxyz[0])+1)*len(conVxyz))) #connectivity
    output_filestrsstrn.write('8 %d %d %d %d %d %d %d %d'%(conVxyz[0,0],
                                                   conVxyz[0,2],
                                                   conVxyz[0,6],
                                                   conVxyz[0,4],
                                                   conVxyz[0,1],
                                                   conVxyz[0,3],
                                                   conVxyz[0,7],
                                                   conVxyz[0,5]
                                                   ))
    output_filestrsstrn.write('\n')
    output_filestrsstrn.write('CELL_TYPES %d\n' %(len(conVxyz)))
    a=ParaviewID*np.ones(len(conVxyz),int) 
    output_filestrsstrn.write('\n'.join(map(str,a)))
    output_filestrsstrn.write('\nPOINT_DATA %d\n' %len(meshxyz)) #data fields  
    output_filestrsstrn.write('SCALARS E33 float 1\n')
    output_filestrsstrn.write('Lookup_table default\n')
    
    for i in range (len(intpt1)):
    	output_filestrsstrn.write('%12.10f \n' %(LagStrain[0,-1,-1,i,-1]))
    output_filestrsstrn.write('\n') 
    
    #output_filestrsstrn.write('\nPOINT_DATA %d\n' %len(meshxyz)) #data fields  
    output_filestrsstrn.write('SCALARS S33 float 1\n')
    output_filestrsstrn.write('Lookup_table default\n')
    
    for i in range (len(intpt1)):
    	output_filestrsstrn.write('%12.10f \n' %(Strs[0,-1,-1,i,-1]))
    output_filestrsstrn.write('\n') 
    output_filestrsstrn.close()  
