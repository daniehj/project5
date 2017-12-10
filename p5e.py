from pylab import *
from random import uniform,seed
from scipy.special import gamma
import time
from numba import autojit,jit

N = 1000
m0 = 1
mcc = int(1E3)
trans = 1E3

def powLawFit(x,lam):
	n = 1 + 3*lam/(1-lam)
	an = n**n/gamma(n)
	return an*x**(n-1)*exp(-n*x)


@jit
def transactions(N,m0,mcc,trans,lambd,a):
    agent = zeros(shape=(N,mcc))
    agent[:,0] = m0

    for col in range(1,mcc):
        agent[:,col] = agent[:,col-1]
        k = 0
        while k < trans:
            i = int(uniform(0,N-1))
            j = int(uniform(0,N-1))
            if agent[i,col]-agent[j,col] == 0:
                p = 1.
            else:
                p = abs((agent[i,col]-agent[j,col])/m0)**(-a)
                #print p
                
                
            check = uniform(0,1)
            if check < p and i != j:
                epsilon = uniform(0,1)

                delta_m = (1 - lambd) * (epsilon*agent[j,col] - (1 - epsilon)*agent[i,col])
           
                agent[i,col] += delta_m
                agent[j,col] -= delta_m 
                k+=1
    
    figure(1)
    ylabel(r'$Agents\ in\ percent\ \%$')
    xlabel(r'$Wealth\ m_0={},\ \lambda={}$'.format(m0,lambd))
    
    dm = 0.01
    mMax = max(agent.flatten())
    bins = arange(0,mMax,dm)
    
    y1 = histogram(agent.flatten(), bins = bins , normed = 100)[0]
    x = bins[0:-1]
    loglog(x,y1, label = r'$\alpha = {}$'.format(a))
    legend()
    
    grid(True)
    
    
    return x,y1#agent


lambd = [0.,0.25]
t0 = time.clock()
alpha = [0.0,0.50,1.00,1.50,2.00]

for alpha in alpha:
    x, y1 = transactions(N,m0,mcc,trans,lambd[0],alpha)
savefig('wealthweightl0.png')
show()
t1 = time.clock()
tim = t1-t0

print '''\nTime for calculations = {:.3}\n'''.format(tim)

figure()
plot(x,y1, '.',label = "$\lambda = 0.50$")
plot(x,powLawFit(x,lambd[0]), label = "gibbs distribution")
show()

propC = 7.
figure()
start = 0
end = -1
pAlpha = linspace(1.8,2.2,3)
loglog(x,y1,label = r'$\alpha = {}$'.format(alpha))
for pAlpha in pAlpha:
    plot(x,x**(-pAlpha)/propC,'--',label = r'$Pareto\ \alpha={}$'.format(pAlpha))
grid(True)
legend()
axis([1E-1,1E1,1E-3,1E1])
savefig('paretoalpha2l0.png')
show()


propC = 3.
pAlpha = linspace(1.8,2.2,3)
t0 = time.clock()
alpha = [0.00,0.50,1.00,1.50,2.00]
for alpha in alpha:
    x,y1 = transactions(N,m0,mcc,trans,lambd[1],alpha)
savefig('wealthweightl0_25.png')
show()
t1 = time.clock()
tim = t1-t0

figure()
start = 0
end = -1
loglog(x,y1,label = r'$\alpha = {}$'.format(alpha))
for pAlpha in pAlpha:
    plot(x,x**(-pAlpha)/propC,'--',label = r'$Pareto\ \alpha={}$'.format(pAlpha))
grid(True)
legend()
axis([1E-1,1E1,1E-3,1E1])
savefig('paretoalpha2l025.png')
show()

print '''\nTime for calculations = {:.3}\n'''.format(tim)

show()