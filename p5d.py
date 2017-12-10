from pylab import *
from random import uniform,seed
from scipy.special import gamma
import time
from numba import autojit,jit


def powLawFit(x,lam):
	n = 1 + 3*lam/(1-lam)
	an = n**n/gamma(n)
	return an*x**(n-1)*exp(-n*x)

#seed(123)

N = 1000
m0 = 1
mcc = int(1E3)
a = 0.0
g = 0.0
trans = 1E3

@jit
def transactions(N,m0,mcc,trans,lambd,a,g):
	'''
	Function for 
	'''
	agent = zeros(shape=(N,mcc))
	agent[:,0] = m0

	c = zeros(shape=(N,N))

	for col in range(1,mcc):
		agent[:,col] = agent[:,col-1]
		k = 0
		while k < trans:
			i = int(uniform(0,N-1))
			j = int(uniform(0,N-1))
			if agent[i,col]-agent[j,col] == 0:
				p = 1.
			else:
				p = 2*((abs((agent[i,col]-agent[j,col])/m0)**(-a))*((c[i,j]+1) + 1)**g)
				c_ = (sum(c)/(mcc*2))
				
			check = uniform(0,1)
			if check < p and i != j:

				epsilon = uniform(0,1)
				delta_m = (1 - lambd) * (epsilon*agent[j,col] - (1 - epsilon)*agent[i,col])
				agent[i,col] += delta_m
				agent[j,col] -= delta_m
				c[i,j]+=1
				c[j,i]+=1
				k+=1
				
	figure(1)
	ylabel(r'$Agents\ in\ percent\ \%$')
	xlabel(r'$Wealth\ m_0={},\ \lambda={}\ \gamma={}$'.format(m0,lambd,g))
	
	dm = 0.01
	mMax = max(agent.flatten())
	bins = arange(0,mMax,dm)
	
	y1 = histogram(agent.flatten(), bins = bins , normed = 100)[0]
	x = bins[0:-1]
	loglog(x,y1, label = r'$\gamma = {}$'.format(g))
	legend()

	grid(True)
	
	return x,y1
	
lmbda = 0
a = 2.
t0 = time.clock()
gama = [0.,1.,2.,3.,4.]
for gama in gama:
	x,y1 = transactions(N,m0,mcc,trans,lmbda,a,gama)
savefig('gammawealthl0a2.png')
t1 = time.clock()
tim = t1-t0
print '''\nTime for calculations = {:.3}\n'''.format(tim)

propC = 1.
pAlpha = a
figure()
start = 0
end = -1
loglog(x,y1,label = r'$\alpha = {}$'.format(a))
#plot(x,x**(-pAlpha)/propC,label = r'$Pareto\ \alpha={}$'.format(pAlpha))
plot(x,powLawFit(x,lmbda))
grid(True)
legend()
savefig('gammapoweralph2.png')
show()


a = 1.
t0 = time.clock()
gama = [0.,1.,2.,3.,4.]
for gama in gama:
	x,y1 = transactions(N,m0,mcc,trans,lmbda,a,gama)
savefig('gammawealthl0a1.png')
t1 = time.clock()
tim = t1-t0
print '''\nTime for calculations = {:.3}\n'''.format(tim)

propC = 1.
pAlpha = a
figure()
start = 0
end = -1
loglog(x,y1,label = r'$\alpha = {}$'.format(a))
#plot(x,x**(-pAlpha)/propC,label = r'$Pareto\ \alpha={}$'.format(pAlpha))
plot(x,powLawFit(x,lmbda))
grid(True)
legend()
savefig('gammapoweralph1.png')
show()
