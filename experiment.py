from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import PolynomialFeatures
import scipy
import statsmodels.api as sm
import numpy as np
import sys
import gc

print(sys.argv,'START')

probparams = [
    dict(beta=1.5, sigma=0.05),
    dict(beta=5.5, sigma=0.05),
]

algparams = [
    dict(beta=0.5,  c1=.5, c2=1.),
    dict(beta=1.5,  c1=.5, c2=1.),
    dict(beta=2.5,  c1=.5, c2=1.),
    dict(beta=3.5,  c1=.5, c2=1.),
    dict(beta=4.5,  c1=.5, c2=1.),
    dict(beta=5.5,  c1=.5, c2=1.),
    dict(beta=6.5,  c1=.5, c2=1.),
    dict(beta=7.5,  c1=.5, c2=1.),
    dict(beta=8.5,  c1=.5, c2=1.),
    dict(beta=9.5,  c1=.5, c2=1.),
    dict(beta=10.5, c1=.5, c2=1.),
]

probparami = int(sys.argv[1])
algparami  = int(sys.argv[2])
seed = int(sys.argv[3])

beta0 = probparams[probparami]['beta']
sigma = probparams[probparami]['sigma']

beta = algparams[algparami]['beta']
c1 = algparams[algparami]['c1']
c2 = algparams[algparami]['c2']

X = np.load('experiment_X_%.1f.npy'%beta0)
Ksqrt = np.load('experiment_Ksqrt_%.1f.npy'%beta0)

T,d = X.shape
np.random.seed(seed)
mu0 = Ksqrt.dot(np.random.randn(T))
mu1 = Ksqrt.dot(np.random.randn(T))

del Ksqrt
gc.collect()

eps = sigma*np.random.randn(T)
idx = np.arange(T)
np.random.shuffle(idx)
X = X[idx]
mu0 = mu0[idx]
mu1 = mu1[idx]
pulls = np.zeros(T,'bool')
betafl = int(beta)
Xp = PolynomialFeatures(betafl).fit_transform(X) if betafl>=1 else np.ones((T,1))
M = Xp.shape[1]
for t in range(T):
    if t%2500==0: print(sys.argv,t)
    x  = X[t]
    xp = Xp[t]
    h  = c1*(t+1)**(-1/(2*beta+d))
    mask = (((X-x)**2).sum(1) <= h**2)
    mask1 = (mask & pulls)[:t-1]
    mask0 = (mask & ~pulls)[:t-1]
    if sum(mask1) < 2*M:
        pulls[t] = 1
    elif sum(mask0) < 2*M:
        pulls[t] = 0
    else:
        lm1 = sm.OLS((mu1+eps)[:t-1][mask1], Xp[:t-1][mask1]).fit()
        lm0 = sm.OLS((mu0+eps)[:t-1][mask0], Xp[:t-1][mask0]).fit()
        pred1 = lm1.get_prediction(xp)
        pred0 = lm0.get_prediction(xp)
        CATE = pred1.predicted_mean-pred0.predicted_mean
        CATEse = np.sqrt(pred1.se_mean**2+pred0.se_mean**2)
        alpha = c2/t
        zconf = scipy.stats.norm.ppf(1.-alpha/2.)
        if CATE > zconf*CATEse:
            pulls[t] = 1
        elif CATE < -zconf*CATEse:
            pulls[t] = 0
        else:
            pulls[t] = np.random.randint(2)
regs = np.maximum(mu1,mu0)-mu0-(mu1-mu0)*pulls

np.save('experiment_regs_%d_%d_%d.npy'%(probparami,algparami,seed), regs)

print(sys.argv,'END')
