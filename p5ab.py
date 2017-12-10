from pylab import *
from random import uniform

N = 500
m0 = 1
mcc = int(1E2)

agent = zeros(shape=(N,mcc))
agent[:,0] = m0


trans = 1E2

for col in range(1,mcc):
    agent[:,col] = agent[:,col-1]
    k = 0
    while k < trans:
        i = int(uniform(0,N-1))
        j = int(uniform(0,N-1))
        
        epsilon = uniform(0,1)
        agent[i,col] = epsilon*(agent[i,col] + agent[j,col])
        agent[j,col] = (1 - epsilon)*(agent[i,col] + agent[j,col])
        k += 1
        


dm = 0.01
average = (agent[:,50:]).flatten()
bins = arange(0,1.0,dm)
b = 1/mean(average)
######################################################
## 5a
figure()
hist(average, bins = bins , normed = 1)
ylabel(r'Agents')
xlabel(r'$Wealth\ m(m_0=1)$')
plot(bins,b*exp(-b*bins))
savefig('Histogram5a.png')
show()

#######################################################
## 5b
figure()
x = sort(average)
y = log10(b*exp(-b*x))
model = polyfit(x,y,1)
plot(x,y,'.')
plot(x,polyval(model,x),'r',label = r'$f(x)=%.2fx+%.2f$'%(model[0],model[1]))
legend()
savefig('Polyfit5b.png')
show()