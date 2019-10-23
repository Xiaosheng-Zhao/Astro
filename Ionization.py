import numpy as np
import matplotlib.pyplot as plt

class non_linear(object):
    '''
    This class contains three non-linear equation solver: Bisection, Secant method, Newton-Raphson method.
    In order to solve the ionization equilibrium problem.
    '''
    def __init__(self, N = 100, Tor =10**(-15), T=30, n=10**5, xM=10**(-12)):
        self.N =N
        self.torlence = Tor
        self.T = T
        self.xM = xM
        self.n = n
        self.alpha = 3*10**(-6)*T**(-0.5)
        self.gama = 3*10**(-9)
        self.beta = 3*10**(-11)*T**(-0.5)
        self.zeta = 10**(-17)
        self.xe1 = (-2*self.beta/self.alpha+np.sqrt(4*self.beta**2/self.alpha**2+12*self.zeta/self.alpha/self.n))   #lower boundary of the bracket
        #self.xe1 = 10**(-8)
        #self.xe2 = 10**(-5)
        self.xe2 = 1.   #upper boundary
        self.xe = self.xe1 + 0.01    #initial trial for Newton-Raphson method
        
    def function(self, xe):
        P = xe**3 + self.beta/self.alpha*self.xM*xe**2-self.zeta/self.alpha/self.n*xe - self.zeta*self.beta/self.alpha/self.gama/self.n*self.xM
        return (P)

    def derivative(self, xe):
        dP = 3*xe**2 + 2*self.beta/self.alpha*self.xM*xe-self.zeta/self.alpha/self.n
        return (dP)
    
    def bisection(self):
        self.iteration = 0
        self.delta = []
        while True:
            xem = (self.xe1+self.xe2)/2
            if (self.function(xem) ==0):
                self.xe = xem
                break
            elif (self.function(xem)* self.function(self.xe1)<0):
                self.xe2 = xem
                self.xe1 = self.xe1
            else:
                self.xe1 = xem
                self.xe2 = self.xe2
            self.iteration += 1
            self.delta.append(np.abs(self.xe2-self.xe1))
            if (np.abs(self.xe2-self.xe1)<self.torlence or self.iteration==self.N):
                self.xe = self.xe2
                print ('Bisection solver solution:\t{0:e}. \t Nr. iteration {1}'.format(self.xe, self.iteration))
                print ('Bisection solver last error: %e'%self.delta[-1])
                break

    def secant(self):
        self.iteration = 0
        self.delta = []
        print (self.xe1)
        while True:
            self.xe = self.xe2 - (self.xe2-self.xe1)/(self.function(self.xe2)-self.function(self.xe1))*self.function(self.xe2)
            self.xe1 = self.xe2
            self.xe2 = self.xe
            self.iteration += 1
            self.delta.append(np.abs(self.xe2-self.xe1))
            if (np.abs(self.xe2-self.xe1)<self.torlence or self.iteration==self.N):
                self.xe = self.xe2
                print ('Secant solver solution:\t{0:e}. \t Nr. iteration {1}'.format(self.xe, self.iteration))
                print ('Secant solver last error: %e'%self.delta[-1])
                break            
            
    '''Newton-Raphson method'''
    def NR(self):
        self.iteration = 0
        self.delta = []
        next_xe = 0
        while True:
            next_xe = self.xe - self.function(self.xe)/self.derivative(self.xe)
            self.delta.append(np.abs(next_xe-self.xe))
            self.iteration += 1
            if (np.abs(next_xe-self.xe)<self.torlence or self.iteration==self.N):
                self.xe = next_xe
                print ('Newton-Raphson solver solution:\t{0:e}. \t Nr. iteration {1}'.format(self.xe, self.iteration))
                print ('Newton-Raphson solver last error: %e'%self.delta[-1])
                break  
            self.xe = next_xe
            
    '''plot the residual/error'''
    def plot_residual(self,ax,color,label):
        self.x = np.linspace(1,self.iteration,self.iteration)
        self.y = self.delta
        ax.plot(self.x, self.y, color+'-', markersize = 7, label = label+'({0})'.format(self.iteration))
        ax.set_yscale('log')
        ax.set_xlabel('Iteration number', fontsize = 14)
        ax.set_ylabel('error', fontsize = 14)
            

fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(1,1,1)
cmd = non_linear()
cmd.bisection()
cmd.plot_residual(ax1,'r','Bisection')
cmd = non_linear()
cmd.secant()
cmd.plot_residual(ax1,'m','Secant')
cmd = non_linear()
cmd.NR()
cmd.plot_residual(ax1,'k','Newton-Raphson')
plt.legend()
plt.savefig('non_solver.pdf')
