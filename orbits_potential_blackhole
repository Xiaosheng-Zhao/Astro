from numpy import *
import matplotlib.pyplot as plt
import math 
class SoS(object):
    ''' 
    class SoS plots the orbits and the corresponding surface of section in axisymmetric potentials.
    where:
        R: radius
        Rk: R for kick step
        Pr: momentum for R
        z: z coordinates
        zk: z for kick step
        Pz: momentum for z
        CR: SoC for R
        CPr: SoC for Pr
        E: energy for efficient potential
        v0: circular speed
        q: axial ratios
        Rc: inner radius
        GM: for constant for black hole
        dt, time : time step size and total time 
    '''
    def __init__(self, _R=[0.15], _z=[0], _Pr=[0], _Pz=[0.456], _E=-0.8, _v0=1, _q=0.9,_Rc=0.14, _GM=0.5, _dt=0.001, _time=300):
        self.R, self.z=_R,_z
        self.Pr=_Pr
        self.Rk=[]
        self.Pz=_Pz
        self.zk=[]
        self.CR=[]
        self.CPr=[]
        self.E=_E
        self.v0=_v0
        self.q=_q
        self.Rc=_Rc
        self.GM=_GM
        self.dt=_dt
        self.time= _time
        self.n=int(_time/_dt)
    def cal(self):       # use Leapfrog integrator Method to calculate the orbits of stars and corresponding SoS
        for i in range(self.n):
            if self.z[-1]>-0.001 and self.z[-1]<0.001 and self.Pz[-1]>0:
                self.CR.append(self.R[-1])
                self.CPr.append(self.Pr[-1])
            self.Rk.append(self.R[-1]+0.5*self.dt*self.Pr[-1])
            self.zk.append(self.z[-1]+0.5*self.dt*self.Pz[-1])
            self.Pr.append(self.Pr[-1]+self.dt*(-(0.5*self.v0**2*(2*self.Rk[-1])/(self.Rc+self.Rk[-1]**2 + self.zk[-1]**2/self.q**2)+self.Rk[-1]*self.GM/(self.Rk[-1]**2+ self.zk[-1]**2)**1.5)))
            self.Pz.append(self.Pz[-1]+self.dt*(-(0.5*self.v0**2*(2*self.zk[-1])/self.q**2/(self.Rc+self.Rk[-1]**2 + self.zk[-1]**2/self.q**2)+self.zk[-1]*self.GM/(self.Rk[-1]**2+ self.zk[-1]**2)**1.5)))
            self.R.append(self.Rk[-1]+0.5*self.dt*self.Pr[-1])
            self.z.append(self.zk[-1]+0.5*self.dt*self.Pz[-1])
    def plot_orbits(self,_ax,_style):       # plot the orbits
        _ax.plot(self.R,self.z,'o'+_style,markersize=0.2,label='E= %.2f'%self.E)
        _ax.legend()
    def plot_SoSection(self,_ax,_style):  # plot the SoS
        _ax.plot(self.CR,self.CPr,'o'+_style,markersize=0.5,label='E= %.2f'%self.E)
        _ax.legend()
    
fig=plt.figure(figsize=(10,10))              #set axes
ax1=plt.axes([0.1,0.55,0.35,0.35])
ax2=plt.axes([0.6,0.55,0.35,0.35])
ax3=plt.axes([0.1,0.1,0.35,0.35])
ax4=plt.axes([0.6,0.1,0.35,0.35])

ax1.set_xlabel(r'$x$',fontsize=18)
ax1.set_ylabel(r'$y$',fontsize=18)
ax2.set_xlabel(r'$x$',fontsize=18)
ax2.set_ylabel(r'$v_x$',fontsize=18)
ax3.set_xlabel(r'$x$',fontsize=18)
ax3.set_ylabel(r'$y$',fontsize=18)
ax4.set_xlabel(r'$x$',fontsize=18)
ax4.set_ylabel(r'$v_x$',fontsize=18)

SoS1=SoS([-0.7],[0],[0],[0.456])          #calculate and draw
SoS1.cal()
SoS1.plot_orbits(ax1,'k')
SoS1.plot_SoSection(ax2,'k')
SoS2=SoS([1.8],[-0.08],[0],[-0.489])        #for another set of initial parameters
SoS2.cal()
SoS2.plot_orbits(ax3,'k')
SoS2.plot_SoSection(ax4,'k')
