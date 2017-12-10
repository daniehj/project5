from pylab import *
from random import uniform,seed
from scipy.special import gamma
import time
from numba import autojit,jit

def powerLawFit(x,lam):
	n = 1 + 3*lam/(1-lam)
	an = n**n/gamma(n)
	return an*x**(n-1)*exp(-n*x)

N = 500
m0 = 1
mcc = int(1E3)

trans = 1E5
@jit
def transactions(N,m0,mcc,trans,lambd):
    agent = zeros(shape=(N,mcc))
    agent[:,0] = m0
    
    for col in range(1,mcc):
        agent[:,col] = agent[:,col-1]
        k = 0
        while k < trans:
            i = int(uniform(0,N-1))
            j = int(uniform(0,N-1))
    
            epsilon = uniform(0,1)
                        
            delta_m = (1 - lambd) * (epsilon*agent[j,col] - (1 - epsilon)*agent[i,col])
            agent[i,col] += delta_m
            agent[j,col] -= delta_m
            k+=1
            
    figure(1)
    ylabel(r'$Agents$')
    xlabel(r'$Wealth\ m_0={}$'.format(m0))
    
    dm = 0.01
    bins = arange(0,4.,dm)
    
    y1 = histogram(agent.flatten(), bins = bins , normed = 1)[0]
    x = bins[0:-1]
    plot(x,y1, '.',label = "$\lambda = {}$".format(lambd))
    plot(x,powerLawFit(x,lambd),color = 'black')
    legend()
    grid(True)
    savefig('lambdafitted.png')
    
    return x,y1

t0 = time.clock()

lambd = [0.00,0.25,0.50,0.90]
for lambd in lambd:
    x,y1 = transactions(N,m0,mcc,trans,lambd)
    figure(2)
    loglog(x,y1,'.',label = "$\lambda = {}$".format(lambd))
    plot(x,powerLawFit(x,lambd),color = 'black')
    axis([-1,5,10E-4,20])
    legend()
    grid(True)
    savefig('lambdafittedlog.png')
    
    
t1 = time.clock()
tim = t1-t0
print '''Time for calculating: {:.6}s'''.format(tim)
show()




