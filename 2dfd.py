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
import matplotlib
from matplotlib import rc
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve,minimize
import pandas as pd
import matplotlib.tri as mptri
from matplotlib.ticker import AutoMinorLocator,LogLocator 
from scipy.interpolate import interp1d,splrep,splder,splev
from scipy.integrate import solve_bvp  #verify the FEM solution
from scipy.interpolate import griddata
from sympy import Symbol,diff,lambdify,log,transpose,Matrix,flatten,Array,tensorproduct
import pyamg as pmg
#from sympy import Symbol,diff,Array,lambdify,tensorproduct,Matrix,flatten

##################################################################
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
rc('xtick',labelsize=22)
rc('ytick',labelsize=22)
matplotlib.rcParams['xtick.major.pad']=10
sptsy=np.array([i for i in np.linspace(0.1,1,10)])
sptsx=np.array([i for i in np.linspace(0.1,1,10)])

xmintick=AutoMinorLocator(20)
ymintick=AutoMinorLocator(20)
ymintickLog=LogLocator(base=10.0,subs=tuple(sptsy),numticks=len(sptsy)+1)
xmintickLog=LogLocator(base=10.0,subs=tuple(sptsx),numticks=len(sptsx)+1)

##################################################################
def FESolver2D(numelx,numely,problemtype):
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
                self.xlength=10
                self.alph = 1.
                self.ylength=10
                self.nx=numelx
                self.ny=numely
                nel = (self.nx-1)*(self.ny-1)
                self.NE = nel
                self.nDim=2                                                        #No. of dof per node, it is essentially the dimension of the problem
                self.thck=1.
                self.nSteps=10
                self.mu=40.
                kap=1.e1*self.mu 
                self.lam=40.
    #            self.tolNR=1.e-10
                self.maxiter=50
                
    
    def meshgenerate(order):
        xs=0.;ys=0.;
        xe=geom.xlength
#        print('xe=',xe)
        ye=geom.ylength
        stepx=geom.xlength/geom.nx
        stepy=geom.ylength/geom.nx
        mesh=np.mgrid[xs:xe+1.e-15:stepx,ys:ye+1.e-15:stepy].reshape(2,-1).T
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
            W = 1/2./geom.alph*geom.mu*(I1**geom.alph-2.**geom.alph)-geom.mu*(log(J))+geom.lam/2.*(J-1.)**2               #change W here to include the modified Neo-Hookean
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
                dWdJ=dWdJ.subs(J,f[0]*f[3]-f[1]*f[2]) + 1.e-32 * f[0]
                d2WdJ2=d2WdJ2.subs(J,f[0]*f[3]-f[1]*f[2]) + 1.e-32 * f[0]
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
    #    detJ=np.dstack(la.det(Je[:,:,i]) for i in range(Xi.shape[0]))        # 1x1xNgP                  # try making it faster by removing the generator
        detJ=(Je[0,0,:]*Je[1,1,:]-Je[0,1,:]*Je[1,0,:])
        Jeinv=1/detJ*np.array([[Je[1,1,:],-Je[0,1,:]],[-Je[1,0,:],Je[0,0,:]]])
    #    Jeinv=np.dstack(la.inv(Je[:,:,i]) for i in range(Xi.shape[0]))       # 2x2xNgP       #avoid computing inverse on a loop (--check ?)
        gDshpG=np.einsum('ilk,ljk->ijk',Jeinv,gDshpL)                                     #global derivatives  (remains the same, even for 3D ? )
        Bmat=np.kron(np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1),gDshpG)            
        gradU=np.einsum('ilk,ljk->ijk',Bmat,dofloc)                         # 9x1xNgP                   #remember that gradU is never symmetric !!!
    #    print(gradU[:,:,0])
    #    print(gDshpG.shape)
        """
        Computing the deformation gradient (F12,F11,F22).T = B*de, and first piola (S) --> (S12,S11,S22), 
        Multiplying by the Gauss-weights, and calculating the element residual
        """
        F=gradU+np.eye(geom.nDim).reshape(-1,1,1).repeat(Xi.shape[0],axis=-1)      #convert to 2x2xNgP and the take det(F) and inv(F)
    #    F2b2 = F.reshape(geom.nDim,geom.nDim,-1)
        detF=F[0]*F[3]-F[1]*F[2] 
    #    detF = np.dstack(la.det(F2b2[:,:,i] ) for i in range(Xi.shape[0]))
    #    Finv2b2 = np.dstack(la.inv(F2b2[:,:,i] ) for i in range(Xi.shape[0]))
    #    Finv2b2T = np.einsum('jik',Finv2b2)
    #    FinvT = Finv2b2T.reshape(-1,1,Xi.shape[0])
    #    Finv = Finv2b2.reshape(-1,1,Xi.shape[0])
        Finv=np.array([F[3],-F[1],-F[2],F[0]])/detF
        FinvT = Finv[np.array([0,2,1,3],int)]
        #    print(gradU[:,:,0])
    #    detF = F[0]*F[3]-F[1]*F[2]
    #    detF=(F[0]*F[3]-F[1]*F[2]).reshape(1,1,-1)#.repeat(Xi.shape[0],axis=-1)   
        WpI1=dWdIi.DWDI1(*F).squeeze().reshape(-1,1,1)#.repeat(Xi.shape[0],axis=-1)
        WpJ=dWdIi.DWDJ(*F).squeeze().reshape(-1,1,1)#.repeat(Xi.shape[0],axis=-1)
        WppI1=dWdIi.D2WDI12(*F).squeeze().reshape(-1,1,1)#.repeat(Xi.shape[0],axis=-1)
        WppJ=dWdIi.D2WDJ2(*F).squeeze().reshape(-1,1,1)#.repeat(Xi.shape[0],axis=-1)
        WpI1 = np.einsum('kji',WpI1)
        WpJ = np.einsum('kji',WpJ)
        WppI1 = np.einsum('kji',WppI1)
        WppJ = np.einsum('kji',WppJ)
        
        
    #    Finv=np.array([F[3],-F[1],-F[2],F[0]])/detF                                 # avoid computing inverse on the loop for the deformation gradient    
    #    FinvT = np.array([F[3],-F[2],-F[1],F[0]])/detF
    #    print('shape = ',np.array([F[3],-F[2],-F[1],F[0]]).shape)
    #    Helpful variables:     
        detF = detF.reshape(1,1,-1)
    #    S = 2.*np.einsum('ijk,ijk->ijk',)
        S=WpI1*2.*F+(WpJ*detF)*FinvT         # notice the swap of axes for transpose
        fac=Wg*detJ*geom.thck
#        S *= fac  
    #    print(fac.shape)
                                                                        # multiplying S by the determinant of the jacobian, thickness, and gauss-weights
        res=np.einsum('lik,ljk->ij',Bmat,fac*S)                                         # double contraction along axis 1 and 2 (of B)
        """
        Computing the Consistent Tangent:D= B^T *C *B    <-- Cijkl, check notes
        Cijkl = 4*W''_(I1) Fij Fkl + 2 W'_(I1) delik deljl + J**2*W''_(J) F-1ji F-1lk +J*W'_(J) F-1ji F-1lk - J W'_(J) F-1jk F-1li 
        F = (F11,F12,f21,F22).T
        """
        F2b2 = F.reshape(geom.nDim,geom.nDim,-1)
        I1f=np.einsum('ijk,ijk->k',F2b2,F2b2)
        FinvT2b2=np.einsum('ijk->jik',Finv.reshape(geom.nDim,geom.nDim,-1))
        Finv2b2 = np.einsum('jik',FinvT2b2) 
#        F11=F[0];F12=F[1];F21=F[2];F22=F[3]
        #This C does not have minor symmetry (relates S to F) , only major symmetry                                             
        
        WppI2I1 = 0.
        WppJI1 = 0.
        WppI2 = 0.
        WppJI2 = 0.
        WppI1I2 = 0.
        WppI1J = 0.
        WppI2J = 0.
        WpI2 = 0.
        
        
        IDGp = (np.einsum('ik,jl->ijkl',np.eye(geom.nDim),np.eye(geom.nDim))[:,:,:,:,np.newaxis]).repeat(NGPts,axis=-1)
        
        Bta=2*(I1f*F2b2-np.einsum('iqk,qjk->ijk',np.einsum('ipk,qpk->iqk',F2b2,F2b2),F2b2))       #check with hand 
        DFDWDI1=WppI1*2*F2b2+WppI2I1*Bta+WppJI1*detF*np.einsum('ijk->jik',Finv2b2)
        DFDWDI2=WppI1I2*2*F2b2+WppI2*Bta+WppJI2*detF*np.einsum('ijk->jik',Finv2b2)
        DFDWDJ=WppI1J*2*F2b2+WppI2J*Bta+WppJ*detF*np.einsum('ijk->jik',Finv2b2)
        
        C=(np.einsum('ijg,klg->ijklg',2*Finv2b2,DFDWDI1)+2*WpI1*IDGp+np.einsum('ijg,klg->ijklg',Bta,DFDWDI2)
            +2*WpI2*(2*np.einsum('ijg,klg->ijklg',Finv2b2,Finv2b2)+I1f*IDGp
            -np.einsum('ilg,kjg->ijklg',Finv2b2,Finv2b2)
            -np.einsum('ljg,ikg->ijklg',np.einsum('qlg,qjg->ljg',Finv2b2,Finv2b2),(np.eye(geom.nDim)[:,:,np.newaxis]).repeat(NGPts,axis=-1))
            -np.einsum('ikg,jlg->ijklg',np.einsum('ipg,kpg->ikg',Finv2b2,Finv2b2),(np.eye(geom.nDim)[:,:,np.newaxis]).repeat(NGPts,axis=-1)))
            +detF*np.einsum('jig,klg->ijklg',Finv2b2,DFDWDJ)+WpJ*(detF*np.einsum('jig,lkg->ijklg',Finv2b2,Finv2b2)
            -detF*np.einsum('jkg,lig->ijklg',Finv2b2,Finv2b2))).reshape(4,4,-1,order='A')




        
        
        
        
        
        
        
        
#        C1111=4*WppI1*F11*F11+2*WpI1+detF**2*WppJ*Finv[0]**2                                             #scalar addition to multi-dimensional array (--check??) 
#        C1112=4*WppI1*F11*F12+detF**2*WppJ*Finv[0]*Finv[2]
#        C1121=4*WppI1*F11*F21+detF**2*WppJ*Finv[0]*Finv[1]
#        C1122=4*WppI1*F11*F22+detF**2*WppJ*Finv[0]*Finv[3]+detF*WpJ*Finv[0]*Finv[3]-detF*WpJ*Finv[1]*Finv[2]
#        C1212=4*WppI1*F12*F12+2*WpI1+detF**2*WppJ*Finv[2]**2
#        C1221=4*WppI1*F12*F21+detF**2*WppJ*Finv[2]*Finv[1]+detF*WpJ*(Finv[2]*Finv[1] -Finv[3]*Finv[0])
#        C1222=4*WppI1*F12*F22+detF**2*WppJ*Finv[2]*Finv[3]
#        C2121=4*WppI1*F21*F21+2*WpI1+detF**2*WppJ*Finv[1]**2
#        C2122=4*WppI1*F21*F22+detF**2*WppJ*Finv[2]*Finv[3]+detF*WpJ*(Finv[2]*Finv[3] -Finv[3]*Finv[1])
#        C2222=4*WppI1*F22*F22+2*WpI1+detF**2*WppJ*Finv[3]**2
#        
#        C1111=C1111.flatten()
#        C1112=C1112.flatten()
#        C1121=C1121.flatten()
#        C1122=C1122.flatten()
#        C1212=C1212.flatten()
#        C1221=C1221.flatten()
#        C1222=C1222.flatten()
#        C2121=C2121.flatten()
#        C2122=C2122.flatten()
#        C2222=C2222.flatten()
#        
#        C=np.array([[C1111,C1112,C1121,C1122],
#                    [C1112,C1212,C1221,C1222],
#                    [C1121,C1221,C2121,C2122],
#                    [C1122,C1222,C2122,C2222]])
        
        D=np.einsum('lik,lpk,pjk->ij',Bmat,C*fac,Bmat)                                  #Check the multiplication once for a simple case!
        IntptGlob = np.einsum('ilj,l->ij',Nshp,nodes)
        return D,res.flatten(),S.squeeze(),F.squeeze(),IntptGlob,Xi.shape[0]
    
    def assembly(disp):
        globK=sp.lil_matrix(1.e-17*np.eye(disp.size))
        globF=np.zeros(disp.size)
        S_pk = np.zeros((conVxy.shape[0],4,NGPts))    #First PK-Stress
        DefGrad = np.zeros((conVxy.shape[0],4,NGPts))    #Deformation Gradient 
        for i in range(conVxy.shape[0]):
            elnodes=conVxy[i]
            globdof=np.array([2*elnodes,2*elnodes+1]).flatten()#.T.flatten()
#            print(elnodes)
            nodexy=meshxy[elnodes]
            locdisp=disp[globdof]
            kel,fel,S_pk[i,:,:],DefGrad[i,:,:],intpt,ngp=locmat(nodexy.T.flatten(),locdisp)
    #        print(intpt)
            globK[np.ix_(globdof,globdof)] += kel
            globF[globdof] += fel
    #        calculate strains and integration point coordinates
    #        Strn=(np.einsum('lik,ljk->ijk',DG.reshape(geom.nDim,geom.nDim,-1),DG.reshape(geom.nDim,geom.nDim,-1))-np.eye(geom.nDim).reshape(geom.nDim,geom.nDim,-1).repeat(ngp,axis=-1))/2
    #    strs = strs.reshape(geom.nDim,geom.nDim,-1)
    #    DG = DG.reshape(geom.nDim,geom.nDim,-1)
        globK = globK.tocsr().copy()
        return globK,globF,S_pk,DefGrad,intpt

    def assignbc(p_type):
        if p_type == 'UT':
            dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
            dofs_top_yval = 0.5*geom.ylength*np.ones(dofs_top_y.size)
            
            dofs_bottom_left_x = 2*identify_nodeBC.nodes_bottom_left_idx
            dofs_bottom_right_y = 2*identify_nodeBC.nodes_bottom_idx+1
            dofs_bottom = np.hstack((dofs_bottom_left_x,dofs_bottom_right_y)) 
            
            pres_dofs_top = np.vstack((dofs_top_y,dofs_top_yval)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
        elif p_type == 'UT_fixed_base':
            dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
            dofs_top_yval = 0.5*geom.ylength*np.ones(dofs_top_y.size)

            dof_bottom_y = 2*identify_nodeBC.nodes_bottom_idx+1
            dof_bottom_x = 2*identify_nodeBC.nodes_bottom_idx
            dofs_bottom = np.hstack((dof_bottom_x,dof_bottom_y)) 

            pres_dofs_top = np.vstack((dofs_top_y,dofs_top_yval)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0.*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
        elif p_type == 'Simple_Shear':
            dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
            dofs_top_x = 2*identify_nodeBC.nodes_top_idx
            dofs_top = np.hstack((dofs_top_x,dofs_top_y))
            
            dof_bottom_y = 2*identify_nodeBC.nodes_bottom_idx+1
            dof_bottom_x = 2*identify_nodeBC.nodes_bottom_idx
            dofs_bottom = np.hstack((dof_bottom_x,dof_bottom_y)) 
            
            
            dofs_top_yval = 0.*geom.ylength*np.ones(dofs_top_y.size)
            dofs_top_xval = 0.3*geom.xlength*np.ones(dofs_top_x.size)
            dofs_top_val = np.hstack((dofs_top_xval,dofs_top_yval))
            
            
            pres_dofs_top = np.vstack((dofs_top,dofs_top_val)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
            
        elif p_type == 'Pure_Shear':
            dofs_top_x = 2*identify_nodeBC.nodes_top_idx
            
            dof_bottom_y = 2*identify_nodeBC.nodes_bottom_idx+1
            dof_bottom_x = 2*identify_nodeBC.nodes_bottom_idx
            dofs_bottom = np.hstack((dof_bottom_x,dof_bottom_y)) 
            
            
            dofs_top_xval = 0.5*geom.xlength*np.ones(dofs_top_x.size)            
            
            pres_dofs_top = np.vstack((dofs_top_x,dofs_top_xval)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
            
        elif p_type == 'UC_fixed_base':
            dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
            dof_bottom_y = 2*identify_nodeBC.nodes_bottom_idx+1
            dof_bottom_x = 2*identify_nodeBC.nodes_bottom_idx
            dofs_bottom = np.hstack((dof_bottom_x,dof_bottom_y)) 
            dofs_top_yval = -0.5*geom.ylength*np.ones(dofs_top_y.size)
            pres_dofs_top = np.vstack((dofs_top_y,dofs_top_yval)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
        
        elif p_type == 'UC':
            
            dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
            dofs_bottom_left_x = 2*identify_nodeBC.nodes_bottom_left_idx
            dofs_bottom_right_y = 2*identify_nodeBC.nodes_bottom_idx+1
            dofs_bottom = np.hstack((dofs_bottom_left_x,dofs_bottom_right_y)) 
            dofs_top_yval = -0.5*geom.ylength*np.ones(dofs_top_y.size)
            pres_dofs_top = np.vstack((dofs_top_y,dofs_top_yval)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
            
        elif p_type == 'UT_mod':
            dofs_top_y = 2*identify_nodeBC.nodes_top_idx+1
            dofs_top_yval = 0.5*geom.ylength*np.ones(dofs_top_y.size)

            dof_bottom_y = 2*identify_nodeBC.nodes_bottom_idx+1
            mid_node_bottom = identify_nodeBC.nodes_bottom_idx.size//2
            print(identify_nodeBC.nodes_bottom_idx)
            print(identify_nodeBC.nodes_bottom_idx[mid_node_bottom])
            dof_bottom_mid_x = 2*identify_nodeBC.nodes_bottom_idx[mid_node_bottom]
#            dof_bottom_x = 2*identify_nodeBC.nodes_bottom_idx
            dofs_bottom = np.hstack((dof_bottom_mid_x,dof_bottom_y)) 

            pres_dofs_top = np.vstack((dofs_top_y,dofs_top_yval)).T
            pres_dofs_bottom = np.vstack((dofs_bottom,0.*dofs_bottom)).T
            prescribed_dofs = np.vstack((pres_dofs_top,pres_dofs_bottom))
 
        return prescribed_dofs
    
    Eltype='Q1'
    n_nodes_elem = (int(Eltype[-1])+1)**2
    OrdGauss=8                                                                      #No. of Gauss-points (in 2D: # of points in each direction counted the same way as local nodes)
    NGPts = int(OrdGauss**2)
    geom=geometry(Eltype)
    B=basis(Eltype[0],float(Eltype[1]))
    GP=GPXi(OrdGauss) 
    dWdIi=DWDIi(geom.nDim)
    meshxy=meshgenerate(1)['msh']
#    print(meshxy.shape)
    conVxy=meshgenerate(1)['connv']
    identify_nodeBC = node_sets_bc(meshxy,conVxy)
    dof=-np.inf*np.ones(meshxy.size) #initializing dofs (displacement of nodes)
#    intpt=([])
    prescribed_dofs = assignbc(problemtype)
    print(prescribed_dofs.shape)
    lineardof = np.zeros(dof.size)
    lineardof[(prescribed_dofs[:,0]).astype(int)]=prescribed_dofs[:,1]
    dof[(prescribed_dofs[:,0]).astype(int)]=0
    fdof=dof==-np.inf                       #free dofs flags: further initialization to zeros needed only for the first step 
    nfdof=np.invert(fdof)                   #fixed dofs flags
    dof[fdof]=0.
    
    #Collect the linear stiffness / force - vector for reference to solve a linear problem
    Ks,Fs,_,_,Gauss_pt_global = assembly(lineardof)
    lineardof[fdof]=sla.spsolve(Ks[np.ix_(fdof,fdof)],-Ks[np.ix_(fdof,nfdof)] @ lineardof[nfdof])
    Gauss_pt_global=Gauss_pt_global[0].T
    dofstore=np.zeros(dof.shape)
    
    DfGrn=np.zeros((geom.nSteps,conVxy.shape[0],4,NGPts));
    DfGrn[:,:,[0,-1],:]=1.
    Strs=np.zeros((geom.nSteps,conVxy.shape[0],4,NGPts));
    
    residualStep = []
    flag=True
    for i in range(geom.nSteps):
        print('Step: ',i)
        dof[(prescribed_dofs[:,0]).astype(int)]=(i+1)/(geom.nSteps)*prescribed_dofs[:,1]
        Ks1,Fs1,_,_,_ = assembly(dof)
#        print(la.norm(dof,np.inf))
#        print(la.cond(Ks1[np.ix_(fdof,fdof)].todense()))
        normres0=la.norm(Fs1[fdof],2)
        normres = normres0.copy()
        iterNR=0
        resNewton = []
        while normres >= geom.tolNR* normres0 and iterNR <= geom.maxiter: 
    #        print('Iter: {}'.format(iterNR))
            res=[]
            Bvec = np.ones((Ks1[np.ix_(fdof,fdof)].shape[0],1))
            smooth=('energy', {'krylov': 'cg', 'maxiter': 50, 'degree': 8, 'weighting': 'local'})
            ml = pmg.smoothed_aggregation_solver(Ks1[np.ix_(fdof,fdof)], Bvec, strength='evolution', max_coarse=50,
                                           smooth=smooth)
            
    #        ml = pmg.smoothed_aggregation_solver(Ks1[np.ix_(fdof,fdof)],max_levels=20)
            del_dof = ml.solve(-Fs1[fdof],-Fs1[fdof],tol=1.e-9,residuals=res)        #np.random.random(Ks1[np.ix_(fdof,fdof)].shape[0])
#            del_dof =  sla.spsolve(Ks1[np.ix_(fdof,fdof)],-Fs1[fdof])                #external force add (-- not required here, only for this case though)
            dof[fdof] += del_dof.copy() 
            Ks1,Fs1,strs,DG,_ = assembly(dof)
            normres=la.norm(Fs1[fdof],np.inf)
            print('Residual Norm: {}   Iter: {}   LoadStep: {}'.format(normres,iterNR+1,i+1))
            if normres >= 1.e6:
                break
            resNewton.append(res)
    #        print(la.norm(Fs1[fdof]))
            iterNR += 1
        
            
        Strs[i,:,:,:] = strs.copy()
        DfGrn[i,:,:,:] = DG.copy()
    #    LagStrain.append(Es)
        dofstore=np.vstack((dofstore,dof))
        residualStep.append(resNewton)
        if normres >= 1.e6:
            flag==False
            break
    dofstore = dofstore.T
    conv_VTK = conVxy[:,[0,1,3,2]]
    with open('alldata.pkl','wb') as fil:
        pckl.dump(residualStep,fil)
        
    return dofstore,Strs,DfGrn,meshxy,conv_VTK


if __name__ == '__main__':
#    dofstore,Strs,DfGrn,meshxy,conv_VTK = FESolver2D(32,32,'UT_fixed_base')
#    check_norm=la.norm(DfGrn[-1,:,-1,:].flatten()-1.5,2)
    import pickle as pckl
    
    
    N_ELEM_X = 10
    N_ELEM_Y = 10
    BC_TYPE = 'UT_mod'
    W_TYPE = 0
    dofstore,Strs,DfGrn,meshxy,conv_VTK = FESolver2D(N_ELEM_X, N_ELEM_Y, BC_TYPE)

    import os
    FOLDER_NAME = '{0}_{1}_{2}'.format(W_TYPE, N_ELEM_X * N_ELEM_Y, BC_TYPE)
    REL_PATH = os.path.join('data', FOLDER_NAME)
    os.makedirs(os.path.join(os.getcwd(), REL_PATH), exist_ok=True)
    np.savez('data/{0}/all_data'.format(FOLDER_NAME),
        dofstore=dofstore, Strs=Strs, DfGrn=DfGrn, meshxy=meshxy, conv_VTK=conv_VTK)

#DfGrn=np.array(DfGrn);LagStrain=np.array(LagStrain);Strs=np.array(Strs)
#if flag:
#    for idx,resx in enumerate(residualStep):
#        fig,axglob = plt.subplots(1,1,figsize=(8,8))
#        axglob.set_title(r'Load Step: ${0:1d}$'.format(idx+1),fontsize=22)
#        for idxNewton,res_newton in enumerate(resx):
#            axglob.semilogy(res_newton,'-o',label=r'Newton iteration: ${0:1d}$'.format(idxNewton+1))
#        axglob.set_xlabel(r'Iterations',fontsize=22)
#        axglob.set_ylabel(r'Residual Norm',fontsize=22)    
#        axglob.legend(loc=0,fontsize=22)
#        axglob.grid(True,linestyle='--')
#        axglob.yaxis.set_minor_locator(ymintickLog)
#        axglob.xaxis.set_minor_locator(xmintick)
#        fig.tight_layout()
#    fig.savefig('Step'+str(idx+1)+'.png',dpi=400)
#    plt.close()
# Post processing using VTK

#import vtk 






#
#plt.figure(figsize=(8,8))
#plt.tricontourf(meshxy[:,0],meshxy[:,1],dof[np.arange(1,dof.size,2)])
#plt.colorbar()