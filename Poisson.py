import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
import time

class Poisson(object):
    '''
    This class contain the five kind of solvers to solve the 2D Poisson equation.
    They are the Jacobi method, Gauss-Seidel (GS), Successive over-relaxation (SoR) (w = 1.5), Steepest descent (SD), Conjugate gradient (CG).
    with the convergence condition '||R||2 < 10^(-6)'. For all solvers, we let the vectors 'live in' 2D grids.
    '''
    def __init__(self, N = 100, nx = 20, ny = 20, w=1.5):   #initialize variables
        self.N = N
        self.nx = nx   
        self.ny = ny
        self.w = w   #relaxiation factor
        self.phi = np.zeros((self.nx+2,self.ny+2))    #add ghost cell to get 22*22 matrix
        self.rho = np.zeros((self.nx+2,self.ny+2))
        for i in range(1,nx+1):
            for j in range(1,nx+1):
                self.rho[i][j] = np.exp(-0.5*((-10.5+i)**2+(-10.5+j)**2))

    '''Jacobi solver use the potentials calculated from the last step'''
    def Jacobi(self):
        self.ti = time.clock()
        self.iteration = 0
        self.R = np.zeros((self.nx, self.ny))
        self.next_phi = np.zeros((self.nx+2,self.ny+2))
        self.norm = []    #add the 2-norm value after each iteration
        while True:
            self.boundary()    #add boundary condition for each iteration
            for i in range(1,self.nx+1):    #update potential values
                for j in range(1,self.ny+1):
                    self.next_phi[i][j] = 0.25*(self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-self.rho[i][j])
            self.phi = self.next_phi
            for i in range(1, self.nx+1):    #get a compact residual matrix
                for j in range(1, self.ny+1):
                    self.R[i-1][j-1] = (self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-4*self.phi[i][j]-self.rho[i][j])
            self.norm.append(self.cal_norm(self.R))    #calculate 2-norm
            self.iteration +=1
            if (self.norm[self.iteration-1] < 10**(-6) or self.iteration == self.N):
                print ('Jacobi solver iteration: %d'%(self.iteration))
                print ('Jacobi last norm: %e'%self.norm[-1])
                break
        self.te = time.clock()            
        print ('Jacobi solver time:\t{0:e}. \t Nr. iteration {1}'.format(self.te - self.ti, self.iteration))

    '''Gauss-Seidel method updates the potential using the ones with one lower index from last step'''
    def GS(self):
        self.ti = time.clock()
        self.iteration = 0
        self.R = np.zeros((self.nx, self.ny))
        self.norm = []
        while True:
            self.boundary()
            for i in range(1,self.nx+1):
                for j in range(1,self.ny+1):
                    self.phi[i][j] = 0.25*(self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-self.rho[i][j])
            for i in range(1, self.nx+1):
                for j in range(1, self.ny+1):
                    self.R[i-1][j-1] = (self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-4*self.phi[i][j]-self.rho[i][j])
            self.norm.append(self.cal_norm(self.R))
            self.iteration +=1
            if (self.norm[self.iteration-1] < 10**(-6) or self.iteration == self.N):
                print ('Gauss-Seidel solver iteration: %d'%(self.iteration))
                print ('Gauss-Seidel solver last norm: %e'%self.norm[-1])
                break
        self.te = time.clock()            
        print ('Gauss-Seidel solver time:\t{0:e}. \t Nr. iteration {1}'.format(self.te - self.ti, self.iteration))
        
    '''Successive over-relaxation (SoR) (w = 1.5), using a relaxation factor w to modify the GS method'''
    def SoR(self):
        self.ti = time.clock()
        self.iteration = 0
        self.R = np.zeros((self.nx, self.ny))
        self.norm = []
        while True:
            self.boundary()
            for i in range(1,self.nx+1):
                for j in range(1,self.ny+1):
                    self.phi[i][j] = (1-self.w)*self.phi[i][j]+0.25*self.w*(self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-self.rho[i][j])
            for i in range(1, self.nx+1):
                for j in range(1, self.ny+1):
                    self.R[i-1][j-1] = (self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-4*self.phi[i][j]-self.rho[i][j])
            self.norm.append(self.cal_norm(self.R))
            self.iteration +=1
            if (self.norm[self.iteration-1] < 10**(-6) or self.iteration == self.N):
                print ('Successive over-relaxation solver iteration: %d'%(self.iteration))
                print ('Successive over-relaxation solver last norm: %e'%self.norm[-1])
                break
        self.te = time.clock()            
        print ('Successive over-relaxation solver time:\t{0:e}. \t Nr. iteration {1}'.format(self.te - self.ti, self.iteration))
        
    '''Steepest descent method, where iteration follows a zig-zag trajectory'''
    def SD(self):        
        self.ti = time.clock()
        self.iteration = 0
        self.R = np.zeros((self.nx+2, self.ny+2))    #the residual matrix
        self.R_20 = np.zeros((self.nx, self.ny))    #for a compact 20*20 residual matrix
        self.V = np.zeros((self.nx+2, self.ny+2))    #product of AR        
        
        for i in range(1,self.nx+1):
            for j in range(1,self.ny+1):
                self.R[i][j] = self.rho[i][j]-(self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-4*self.phi[i][j])
        self.norm = []
        while True:
            self.boundary()
            for i in range(1,self.nx+1):
                for j in range(1,self.ny+1):
                    self.V[i][j] = self.R[i-1][j]+self.R[i+1][j]+self.R[i][j-1]+self.R[i][j+1]-4*self.R[i][j]
            self.alpha = np.dot(np.matrix(self.R).getA1(), np.matrix(self.R).getA1())/np.dot(np.matrix(self.R).getA1(), np.matrix(self.V).getA1())   #perform inner product using matrix to vector 
            self.phi += self.alpha*self.R
            for i in range(1,self.nx+1):   
                for j in range(1,self.ny+1):
                    self.R[i][j] = self.rho[i][j]-(self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-4*self.phi[i][j]) 
                    
            for i in range(1, self.nx+1):
                for j in range(1, self.ny+1):            
                    self.R_20[i-1][j-1] = self.R[i][j]
            self.norm.append(self.cal_norm(self.R_20))
            self.iteration +=1
            
            if (self.norm[self.iteration-1] < 10**(-6) or self.iteration == self.N):
                print ('steepest descent solver iteration: %d'%(self.iteration))
                print ('steepest descent solver last norm: %e'%self.norm[-1])
                break
        self.te = time.clock()            
        print ('steepest descent solver time:\t{0:e}. \t Nr. iteration {1}'.format(self.te - self.ti, self.iteration))
        
    '''Conjugate gradient, combine the steepest descent direction with the direction used in previous step'''
    def CG(self):
        self.ti = time.clock()
        self.iteration = 0
        self.R = np.zeros((self.nx+2, self.ny+2))
        self.new_R = np.zeros((self.nx+2, self.ny+2))
        self.R_20 = np.zeros((self.nx, self.ny))    #for a 20*20 residual matrix
        self.V = np.zeros((self.nx+2, self.ny+2))    #product of AP (previous direction vector/matrix)
        
        for i in range(1,self.nx+1):
            for j in range(1,self.ny+1):
                self.R[i][j] = self.rho[i][j]-(self.phi[i-1][j]+self.phi[i+1][j]+self.phi[i][j-1]+self.phi[i][j+1]-4*self.phi[i][j])
        self.P = self.R
        self.norm = []
        
        while True:
            self.boundary()
            for i in range(1,self.nx+1):
                for j in range(1,self.ny+1):
                    self.V[i][j] = self.P[i-1][j]+self.P[i+1][j]+self.P[i][j-1]+self.P[i][j+1]-4*self.P[i][j]
            self.alpha = np.dot(np.matrix(self.R).getA1(), np.matrix(self.R).getA1())/np.dot(np.matrix(self.P).getA1(), np.matrix(self.V).getA1())
            self.phi += self.alpha*self.P
            self.new_R = self.R-self.alpha*self.V
            self.beta = np.dot(np.matrix(self.new_R).getA1(), np.matrix(self.new_R).getA1())/np.dot(np.matrix(self.R).getA1(), np.matrix(self.R).getA1())
            self.P = self.new_R+self.beta*self.P
            self.R = self.new_R
            
            for i in range(1, self.nx+1):
                for j in range(1, self.ny+1):            
                    self.R_20[i-1][j-1] = self.R[i][j]
            self.norm.append(self.cal_norm(self.R_20))
            self.iteration +=1
            
            if (self.norm[self.iteration-1] < 10**(-6) or self.iteration == self.N):
            #if (self.iteration == self.N):
                print ('Conjugate gradient solver iteration: %d'%(self.iteration))
                print ('Conjugate gradient solver last norm: %e'%self.norm[-1])
                break
        self.te = time.clock()            
        print ('Conjugate gradient solver time:\t{0:e}. \t Nr. iteration {1}'.format(self.te - self.ti, self.iteration))

    '''boundary condition'''
    def boundary(self):
        for j in range(self.nx+2):
            self.phi[0][j] = -self.phi[1][j]
            self.phi[self.nx+1][j] = -self.phi[self.nx][j]
        for i in range(self.nx+2):
            self.phi[i][0] = -self.phi[i][1]
            self.phi[i][self.ny+1] = -self.phi[i][self.ny]

    '''calculate the 2-norm from a 1D vector form'''
    def cal_norm(self, R):
        vector = np.matrix(R).getA1()
        sum = 0
        for i in range(self.nx**2):
            sum += vector[i]**2
        sum = np.sqrt(sum)
        return (sum)
            
    '''plot the potential surface'''
    def plot3d(self, ax, x1,x2,y1,y2):
        self.x = np.linspace(x1,x2,self.nx+2)
        self.y = np.linspace(y1,y2,self.nx+2)
        self.x, self.y = np.meshgrid(self.x, self.y)
        self.sur = ax.plot_surface(self.x, self.y, self.phi, rstride = 1, cstride = 1, cmap = cm.coolwarm)
        ax.set_xlim(x1,x2)
        ax.set_ylim(y1,y2)
        ax.set_xlabel('X',fontsize = 14)
        ax.set_ylabel('Y',fontsize = 14)
        ax.set_zlabel('Potential (Conjugate gradient)',fontsize = 14)
      
    '''plot the 2-norm'''
    def plot_norm(self,ax,color,label):
        self.x = np.linspace(1,self.iteration,self.iteration)
        self.y = self.norm
        ax.plot(self.x, self.y, color+'-', markersize = 7, label = label)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration number', fontsize = 14)
        ax.set_ylabel('$||R||_2$', fontsize = 14)


fig = plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1, projection = '3d')
ax2 = plt.subplot(1,2,2)

cmd =Poisson()
cmd.Jacobi()
cmd.plot_norm(ax2,'r|','Jacobi')
cmd =Poisson()
cmd.GS()
cmd.plot_norm(ax2,'g','Gauss-Seidel')
cmd =Poisson()
cmd.SoR()
cmd.plot_norm(ax2,'m','Successive over-relaxation')
cmd =Poisson()
cmd.SD()
cmd.plot_norm(ax2,'c','Steepest descent')
cmd =Poisson()
cmd.CG()
cmd.plot_norm(ax2,'k','Conjugate gradient')
cmd.plot3d(ax1, -10.5, 10.5, -10.5, 10.5)

plt.legend()
plt.savefig('norm-log.pdf')
