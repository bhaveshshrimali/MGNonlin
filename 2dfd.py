# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:24:23 2018

@author: bshrima2

fixme: 
    line 297: --- assign dirichlet dofs more precisely (based on node sets)
    line 68:  --- fix meshgenerates to incorporate higher order quads 
    line 89 ---   add element_sets to identify corresponding elements in case
                  non-zero Neumann data to be prescribed
"""

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
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
from sympy import Symbol,diff,lambdify,log,transpose,Matrix,flatten,Array,tensorproduct
#from sympy import Symbol,diff,Array,lambdify,tensorproduct,Matrix,flatten

##################################################################
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
rc('xtick',labelsize=18)
rc('ytick',labelsize=18)
matplotlib.rcParams['xtick.major.pad']=10
mx=AutoMinorLocator(10)
my=AutoMinorLocator(10)
##################################################################
class geometry() : #1D geometry
    def __init__(self,Eltype):
        self.tolNR=1.e-6
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
            self.xlength=2
            self.ylength=2
            self.nx=1
            self.ny=1
            nel = (self.nx-1)*(self.ny-1)
            self.NE = nel
            self.nDim=2                                                        #No. of dof per node, it is essentially the dimension of the problem
            self.thck=1.
            self.nSteps=10
            self.mu=40.
            kap=1.e1*self.mu 
            self.lam=40.
#            self.tolNR=1.e-10
            self.maxiter=100
            

def meshgenerate(order):
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

class node_sets_bc():
    def __init__(self,mesh,connectivity):
        ytop = mesh[:,1].max()
        ybottom = mesh[:,1].min()
        xright = mesh[:,0].max()
        xleft = mesh[:,0].min()
        
        select_nodes = np.arange(mesh.shape[0])
#       
        self.nodes_right_idx = select_nodes[abs(mesh[:,0] - xright ) <= 1.e-15]
        self.nodes_left_idx = select_nodes[abs(mesh[:,0] - xleft ) <= 1.e-15]
        self.nodes_top_idx = select_nodes[abs(mesh[:,1] - ytop ) <= 1.e-15]
        self.nodes_bottom_idx = select_nodes[abs(mesh[:,1] - ybottom ) <= 1.e-15]
        
#        print(select_nodes[abs(mesh[:,0] - xright ) <= 1.e-15])
#        print(select_nodes[abs(mesh[:,1] - ybottom ) <= 1.e-15])
        self.nodes_bottom_right_idx = np.intersect1d(select_nodes[abs(mesh[:,0] - xright ) <= 1.e-15],
                                                select_nodes[abs(mesh[:,1] - ybottom ) <= 1.e-15])
        
        self.nodes_top_right_idx = np.intersect1d(select_nodes[abs(mesh[:,0] - xright ) <= 1.e-15],
                                             select_nodes[abs(mesh[:,1] - ytop ) <= 1.e-15])
        
        self.nodes_top_left_idx = np.intersect1d(select_nodes[abs(mesh[:,0] - xleft ) <= 1.e-15],
                                                select_nodes[abs(mesh[:,1] - ytop ) <= 1.e-15])
        
        self.nodes_bottom_left_idx = np.intersect1d(select_nodes[abs(mesh[:,0] - xleft ) <= 1.e-15],
                                                select_nodes[abs(mesh[:,1] - ybottom ) <= 1.e-15])

      
def lag_basis(k,z,Xi):
    n = 1.
    for i in range(len(Xi)):
        if k != i:
            n *= (z-Xi[i])/(Xi[k]-Xi[i])
    return n    
               
class GPXi():
    def __init__(self,ordr):
        from numpy.polynomial.legendre import leggauss  #Gauss-Legendre Quadrature for 1D (proxy 2D quads -- check, 3D hex -- not checked)
        self.xi=leggauss(ordr)[0]    #nodes
        self.wght=leggauss(ordr)[1]  #weights

class basis():  # defined on the canonical element (1D : [-1,1], 2D (Q): [-1,1] x [-1,1] )
    def __init__(self,eltype,deg):
        deg=int(deg)
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
            xi = Symbol('xi',real=True); eta = Symbol('eta',real=True)
            Xi_nodes = np.linspace(-1,1,deg+1)
            arr2 = Array([lag_basis(m,xi,Xi_nodes) for m in range(deg+1)]) 
            arr1 = Array([lag_basis(m,eta,Xi_nodes) for m in range(deg+1)])
#            print(arr2)
            N = tensorproduct(arr1,arr2)
#            print(N)
            dfN = Matrix(flatten(N.diff(xi))).col_join(Matrix(flatten(N.diff(eta)))) 
            self.Ns=lambdify((xi,eta),flatten(N),'numpy')
            self.dN=lambdify((xi,eta),dfN,'numpy')

class DWDIi():               
    def __init__(self,ndim):
        I1=Symbol('I1');J=Symbol('J');
        W = 1/2.*geom.mu*(I1-2.)-geom.mu*log(J)+geom.lam/2.*(J-1.)**2               #change W here to include the modified Neo-Hookean
        dWdI1=W.diff(I1,1)
        dWdJ=W.diff(J,1)
#        print(dWdJ)
        d2WdI12=W.diff(I1,2);
        d2WdJ2=W.diff(J,2);
        if ndim==2:
            f12=Symbol('f12');f11=Symbol('f11');f22=Symbol('f22');f21=Symbol('f21')
            f=Matrix([f11,f12,f21,f22]);
            dWdI1=dWdI1.subs(I1,transpose(f).dot(f)) + 1.e-32 * f[0]                                           #substituting I1, in terms of 
            d2WdI12=d2WdI12.subs(I1,transpose(f).dot(f)) + 1.e-32 * f[0]
#            dWdI2.subs(0.5*((transpose(f).dot(f)+f[0]**2)**2 
#                            - (f[1]**2 + f[0]**2)**2 
#                            + 2*f[0]**2*(f[1] 
#                            + f[2])**2 
#                            + (f[0]**2 
#                               + f[2]**2)**2 ))           # cannot get expression of I2 directly in terms of vector representation of F
            dWdJ=dWdJ.subs(J,f[0]*f[3]-f[1]*f[2]) + 1.e-27 * f[0]
            d2WdJ2=d2WdJ2.subs(J,f[0]*f[3]-f[1]*f[2]) + 1.e-27 * f[0]
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
#    print(nodes.reshape(2,-1).T)
#    print(de.shape)
    Xi=np.tile(GP.xi,OrdGauss)
    Eta=np.repeat(GP.xi,OrdGauss)
#    print(Eta)   
    dofloc=de.reshape(de.size,-1,1).repeat(Xi.shape[0],axis=-1)                              # arranging dof for (dot) product with B (len(Xi) and not len(GP.xi)) !!!
    Wg=np.outer(GP.wght,GP.wght).flatten() 
    Nshp=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),np.array(B.Ns(Xi,Eta)))      # kron has to be taken on nDim (and not OrdGauss)
#    print(np.array(B.dN(Xi,Eta)).shape)
    gDshpL=np.array(B.dN(Xi,Eta)).reshape(geom.nDim,n_nodes_elem,-1)                  # local derivatives
#    print(gDshpL[:,:,2])
    Je=np.einsum('ilk,lj->ijk',gDshpL,nodes.reshape(geom.nDim,-1).T)                  # computing the jacobian
#    print(Je[:,:,0])
    detJ=np.dstack(la.det(Je[:,:,i]) for i in range(Xi.shape[0]))        # 1x1xNgP                  # try making it faster by removing the generator
    Jeinv=np.dstack(la.inv(Je[:,:,i]) for i in range(Xi.shape[0]))       # 3x3xNgP       #avoid computing inverse on a loop (--check ?)
    gDshpG=np.einsum('ilk,ljk->ijk',Jeinv,gDshpL)                                     #global derivatives  (remains the same, even for 3D ? )
    Bmat=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),gDshpG)            
    gradU=np.einsum('ilk,ljk->ijk',Bmat,dofloc)                         # 9x1xNgP                   #remember that gradU is never symmetric !!!
#     print(gradU[:,:,0])
#    print(gDshpG.shape)
    """
    Computing the deformation gradient (F12,F11,F22).T = B*de, and first piola (S) --> (S12,S11,S22), 
    Multiplying by the Gauss-weights, and calculating the element residual
    """
    F=gradU+np.eye(geom.nDim).reshape(-1,1,1).repeat(Xi.shape[0],axis=-1)      #convert to 3x3xNgP and the take det(F) and inv(F)
    F2b2 = F.reshape(geom.nDim,geom.nDim,-1)
    detF = np.dstack(la.det(F2b2[:,:,i] )for i in range(Xi.shape[0]))
    Finv2b2 = np.dstack(la.inv(F2b2[:,:,i] )for i in range(Xi.shape[0]))
    Finv2b2T = np.einsum('ijk->jik',Finv2b2)
    FinvT = Finv2b2T.reshape(-1,1,Xi.shape[0])
    Finv = Finv2b2.reshape(-1,1,Xi.shape[0])
    #    print(gradU[:,:,0])
#    detF = F[0]*F[3]-F[1]*F[2]
#    detF=(F[0]*F[3]-F[1]*F[2]).reshape(1,1,-1)#.repeat(Xi.shape[0],axis=-1)   
    WpI1=dWdIi.DWDI1(*F).squeeze().reshape(1,1,-1)#.repeat(Xi.shape[0],axis=-1)
    WpJ=dWdIi.DWDJ(*F).squeeze().reshape(1,1,-1)#.repeat(Xi.shape[0],axis=-1)
    WppI1=dWdIi.D2WDI12(*F).squeeze().reshape(1,1,-1)#.repeat(Xi.shape[0],axis=-1)
    WppJ=dWdIi.D2WDJ2(*F).squeeze().reshape(1,1,-1)#.repeat(Xi.shape[0],axis=-1)
#    Finv=np.array([F[3],-F[1],-F[2],F[0]])/detF                                 # avoid computing inverse on the loop for the deformation gradient    
#    FinvT = np.array([F[3],-F[2],-F[1],F[0]])/detF
#    print('shape = ',np.array([F[3],-F[2],-F[1],F[0]]).shape)
#    Helpful variables:     
    S=WpI1*2.*F+(WpJ*detF)*FinvT         # notice the swap of axes for transpose
    fac=Wg*detJ*geom.thck
    S*=fac                                                                      # multiplying S by the determinant of the jacobian, thickness, and gauss-weights
    res=np.einsum('lik,ljk->ij',Bmat,S)                                         # double contraction along axis 1 and 2 (of B)
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
    
    D=np.einsum('lik,lpk,pjk->ij',Bmat,C,Bmat)                                  #Check the multiplication once for a simple case!
    IntptGlob = np.einsum('ilj,l->ij',Nshp,nodes)
    return D,res.flatten(),S,F,IntptGlob,Xi.shape[0]

def assembly(disp):
    globK=0.*np.eye(disp.size)
    globF=np.zeros(disp.size)
#    strs = np.zeros((geom.NE,2,2,NGPts))
#    DG = np.zeros((geom.NE,2,2,NGPts))
#    intpt = strs.copy()
    
    for i in range(conVxy.shape[0]):
        elnodes=conVxy[i]
        globdof=np.array([2*elnodes,2*elnodes+1]).flatten()#.T.flatten()
        nodexy=meshxy[elnodes]
        locdisp=disp[globdof]
        kel,fel,strs,DG,intpt,ngp=locmat(nodexy.T.flatten(),locdisp)
#        print(intpt)
        globK[np.ix_(globdof,globdof)] += kel
        globF[globdof] += fel
        
#        calculate strains and integration point coordinates
        Strn=(np.einsum('lik,ljk->ijk',DG.reshape(geom.nDim,geom.nDim,-1),DG.reshape(geom.nDim,geom.nDim,-1))-np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1).repeat(ngp,axis=-1))/2
    strs = strs.reshape(geom.nDim,geom.nDim,-1)
    DG = DG.reshape(geom.nDim,geom.nDim,-1)
    return globK,globF,strs,DG,Strn,intpt

Eltype='Q1'
n_nodes_elem = (int(Eltype[-1])+1)**2
OrdGauss=2                                                                      #No. of Gauss-points (in 2D: # of points in each direction counted the same way as local nodes)
NGPts = int(OrdGauss**2)
geom=geometry(Eltype)
B=basis(Eltype[0],float(Eltype[1]))
GP=GPXi(OrdGauss) 
dWdIi=DWDIi(geom.nDim)
meshxy=meshgenerate(1)['msh']
conVxy=meshgenerate(1)['connv']
identify_nodeBC = node_sets_bc(meshxy,conVxy)
dof=1.e9*np.ones(meshxy.size) #initializing dofs (displacement of nodes)
intpt=([])


dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
#dofs_bottom_left = np.hstack((2*identify_nodeBC.nodes_bottom_left_idx,
#                          2*identify_nodeBC.nodes_bottom_left_idx+1))
dofs_bottom_left_x = 2*identify_nodeBC.nodes_bottom_left_idx
dofs_bottom_right_y = 2*identify_nodeBC.nodes_bottom_idx+1
dofs_bottom = np.hstack((dofs_bottom_left_x,dofs_bottom_right_y)) 
pres_dofs = np.hstack((dofs_top_y,dofs_bottom)).T
num_pres_dof = pres_dofs.size 
dofs_top_yval = .0005*geom.ylength*np.ones(dofs_top_y.size)
pres_dofs_top = np.vstack((dofs_top_y,dofs_top_yval)).T
pres_dofs_bottom = np.vstack((dofs_bottom,0*dofs_bottom)).T
prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))

lineardof = np.zeros(dof.size)
lineardof[(prescribed_dofs[:,0]).astype(int)]=prescribed_dofs[:,1]
dof[(prescribed_dofs[:,0]).astype(int)]=0
fdof=dof==1.e9                          #free dofs flags: further initialization to zeros needed only for the first step 
nfdof=np.invert(fdof)                   #fixed dofs flags
dof[fdof]=0.

#Collect the linear stiffness / force - vector for reference to solve a linear problem
Ks,Fs,_,_,_,Gauss_pt_global = assembly(lineardof)
lineardof[fdof]=la.solve(Ks[np.ix_(fdof,fdof)],-Ks[np.ix_(fdof,nfdof)] @ lineardof[nfdof])
Gauss_pt_global=Gauss_pt_global[0].T
dofstore=np.zeros(dof.shape)

DfGrn=([]);Strs=([]);LagStrain=([])
for i in range(geom.nSteps):
    print('Step: ',i)
    dof[(prescribed_dofs[:,0]).astype(int)]=(i+1)/(geom.nSteps)*prescribed_dofs[:,1]
    Ks1,Fs1,strs,DG,Es,_ = assembly(dof)
    normres0=la.norm(Fs1[fdof],2)
    normres = normres0.copy()
    iterNR=0
    while normres >= geom.tolNR:#* normres0:# and iterNR <= geom.maxiter: 
        print('Iter: {}'.format(iterNR))
        del_dof =  la.solve(Ks1[np.ix_(fdof,fdof)],-Fs1[fdof])                #external force add (-- not required here, only for this case though)
        dof[fdof] += del_dof.copy() 
        Ks1,Fs1,strs,DG,Es,_ = assembly(dof)
        normres=la.norm(Fs1[fdof],2)
#        print(la.norm(Fs1[fdof]))
        iterNR += 1
    Strs.append(strs)
    DfGrn.append(DG)
    LagStrain.append(Es)
    dofstore=np.vstack((dofstore,dof))
dofstore = dofstore.T
DfGrn=np.array(DfGrn);LagStrain=np.array(LagStrain);Strs=np.array(Strs)
#plt.figure(figsize=(8,8))
#plt.tricontourf(meshxy[:,0],meshxy[:,1],dof[np.arange(1,dof.size,2)])
#plt.colorbar()