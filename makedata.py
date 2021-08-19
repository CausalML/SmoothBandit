from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import PolynomialFeatures
import scipy
import numpy as np
import sys

print(sys.argv)

beta = sys.argv[1]

np.random.seed(0)
T = 20000
d = 2
X = np.random.rand(T*d).reshape(T,d)
K = Matern(nu=float(beta),length_scale=0.15)(X)
Ksqrt = scipy.linalg.sqrtm(K)
np.save('experiment_X_%s.npy'%beta,X)
np.save('experiment_Ksqrt_%s.npy'%beta,Ksqrt)

