import numpy as np 
import numpy.linalg as la 
import matplotlib.pyplot as plt
import matplotlib.patches as pts 
from mpl_toolkits.mplot3d import Axes3D
import os 
plt.rc('text',usetex=True)

#RSA Circles Monodisperse 
cdir=os.getcwd()  #current wd

#Parameters 
L = 1.0 
nps=60  #60 circles in JAM2015 paper 
fo = 0.25# volume fraction of the cylinders 
r = L*(fo/nps/np.pi)**(0.5)  #radius of each cylinder 
d = 2.*r  
print('Desired Porosity = ',np.pi*r**2*nps)

# Distance factors 
s1 = 1.1*d
s2 = 0.05*d   

# Mark points 
hgrid=np.mgrid[-L:2*L:L,-L:2*L:L].reshape(1,2,-1) #remember to change 9 (for 3D spheroids, spheres etc.)
#hgrid *= 0 
xs=np.zeros((nps,2)); xs[0]=0.15*np.random.random((2))+0.5  #pre-allocated array for circles (with first circle symmetrically placed ) 
edprtdist=np.zeros((2,2))
# RSA for circles (without additional centres for periodicity)
plt.close()
#ax.add_patch(pts.Circle((xs[0,0],xs[0,1]),r))
for ip in range(1,nps):
    xtemp=np.random.random((2))
    tdis = (xtemp-xs[:ip]).reshape(len(xs[:ip]),2,-1)
    dists=la.norm(tdis+hgrid,2,axis=1)
    edist=abs(xtemp.reshape(-1,1)-np.array([r,L-r],float))  
    while (dists < s1).any() or (edist < s2).any():
        xtemp = np.random.random((2))
        tdis = (xtemp-xs[:ip]).reshape(len(xs[:ip]),2,-1)
        dists=la.norm(tdis+hgrid,2,axis=1)
        edist = abs(xtemp.reshape(-1,1)-np.array([r,L-r],float)) 
    xs[ip]=xtemp.copy()
#    ax.add_patch(pts.Circle((xs[ip,0],xs[ip,1]),r))
xfinal=list(xs)       #each element is an array of shape (2,), generated to add additional circles for periodicity 
k=nps-1

for i in range(nps):
    x1,x2=xfinal[i]
    if x1 < r and x2 >= r:
        k += 1
        xfinal.append(np.array([x1+L,x2]))
        if x2 > L-r:
            k += 1
            xfinal.append(np.array([x1,x2-L]))
            k += 1
            xfinal.append(np.array([x1+L,x2-L]))
        elif x2 <= L-r:
            k += 1
            xfinal.append(np.array([x1,x2])) 
            k += 1
            xfinal.append(np.array([x1+L,x2]))
    elif x1 >=r and x2 < r:
        k += 1
        xfinal.append(np.array([x1,x2+L]))
        if x1 > L-r:
            k += 1
            xfinal.append(np.array([x1-L,x2]))
            k += 1
            xfinal.append(np.array([x1-L,x2+L]))
        elif x1 <= L-r:
            k += 1
            xfinal.append(np.array([x1,x2]))
            k += 1
            xfinal.append(np.array([x1,x2+L]))
    elif x1 >= r and x2 >= r:
        k += 1
        xfinal.append(np.array([x1,x2]))
        if x1 > L-r and x2 <= L-r:
            k += 1
            xfinal.append(np.array([x1-L,x2]))
        elif x1 <= L-r and x2 > L-r:
            k += 1
            xfinal.append(np.array([x1,x2-L]))
        elif x1 > L-r and x2 > L-r: 
            k += 1
            xfinal.append(np.array([x1-L,x2-L]))
    elif x1 < r and x2 < r:
        k += 1
        xfinal.append(np.array([x1+L,x2]))
        k += 1
        xfinal.append(np.array([x1,x2+L]))
        k += 1
        xfinal.append(np.array([x1+L,x2+L]))



print('Number of initial cylinders = ',nps)
print('Number of additional cylinders for periodicity = ',k+1-nps)
print('Total number of cylinders = ',k+1)
xfinal=np.array(xfinal)
fig,ax=plt.subplots(1,1,figsize=(8,8))
ax.set_axis_off()
ax.add_patch(pts.Rectangle((0,0),L,L,fill=True))
for i in range(len(xfinal)):
    ax.add_patch(pts.Circle((xfinal[i,0],xfinal[i,1]),r,facecolor='white'))
#    ax.text(xfinal[i,0],xfinal[i,1],r'{\bf C} '+str(i))
#    ax.grid(True,linestyle='--')
ax.tick_params(which='both',labelbottom=False,labelleft=False)
fig.tight_layout()
ax.set_axis_on()
#fig.axes(True)
plt.savefig('Circles.eps')
#plt.close()
geo_write=False

if geo_write:
    # Writing the geometry file to be meshed by netgen
    cdir=os.getcwd()
    if not os.path.exists(cdir+'/geofiles/'+str(int(100*fo))):
        cdir1=cdir+'/geofiles/'+str(int(100*fo))
        os.makedirs(cdir1)
    os.chdir(cdir+'/geofiles/'+str(int(100*fo)))
    ftext='\nsolid p1=plane (0,0,0;0,0,-1);\nsolid p2=plane (1,1,1;0,0,1);\nsolid p3=plane (0,0,0;0,-1,0);\nsolid p4=plane (1,1,1;0,1,0);\nsolid p5=plane (0,0,0;-1,0,0);\nsolid p6=plane (1,1,1;1,0,0);\nsolid p7=plane (-1,-1,-1;0,0,-1);\nsolid p8=plane (2,2,2;0,0,1);\nsolid p9=plane (-1,-1,-1;0,-1,0);\nsolid p10=plane (2,2,2;0,1,0);\nsolid p11=plane (-1,-1,-1;-1,0,0);\nsolid p12=plane (2,2,2;1,0,0);'
    ftext2='\nsolid cube=p1 and p2 and p3 and p4 and p5 and p6;\nsolid cube2 =p7 and p8 and p9 and p10 and p11 and p12;'
    
    fname=str(int(100*fo))+'cubecylrandom.geo'
    with open(fname,'w') as mfile:
        mfile.write('algebraic3d\n')
        mfile.write(ftext+ftext2)
        for k in range(len(xfinal)):
            mfile.write('\nsolid cyl'+str(k)+' = cylinder('+str(xfinal[k,0])+',-1.5,'+str(xfinal[k,1])+';'
            +str(xfinal[k,0])+',1.5,'+str(xfinal[k,1])+';'
            +str(r)+');\n')
        mfile.write('solid cyls=')
        for j in range(len(xfinal)-1):
            mfile.write('\ncyl'+str(j)+' or')
        mfile.write('\ncyl'+str(len(xfinal)-1)+';\n')
        mfile.write('\nsolid cubeandcyl = cube2 and cyls;\n')
        mfile.write('solid void=cube and not cubeandcyl;\n')
        mfile.write('\ntlo void -transparent -maxh = 0.05;\n')
        mfile.write('identify periodic p1 p2;\nidentify periodic p3 p4;\nidentify periodic p5 p6;\n')
    
    os.chdir(cdir)
        
        