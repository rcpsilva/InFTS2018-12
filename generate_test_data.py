import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)


f_name = 'series16.pkl'
k1 = 1.0 * Matern(length_scale=1, length_scale_bounds=(1e-1, 10.0), nu=1.3)
k2 = 1.0 * RBF(length_scale=5, length_scale_bounds=(1e-1, 10.0))

# Specify Gaussian Process
gp1 = GaussianProcessRegressor(kernel=k1)
gp2 = GaussianProcessRegressor(kernel=k2)

# Plot prior
X_ = np.linspace(0, 80, 2000)
y_mean, y_std = gp1.predict(X_[:, np.newaxis], return_std=True)
#y_mean, y_std = gp2.predict(X_[:, np.newaxis], return_std=True)

y1 = gp1.sample_y(X_[:, np.newaxis], 1) + 20

y2 = gp2.sample_y(X_[:, np.newaxis], 1) + 20

idxs = np.random.permutation(len(y1))

y = np.concatenate([y1*0.5 + 20, y1])   #np.concatenate([y1, y2])

#y = np.array([y[0] for y in y_samples[:]])
t = np.arange(0, len(y+1), 1)

#y = np.concatenate((y*5 - 75, y + 2.5, y*5 - 75))
#t = np.arange(0, len(y+1), 1)


print(y)

plt.plot(t, y)
#plt.plot(t, y, lw=1)
plt.title("Prior (kernel:  %s)" % k1, fontsize=12)

with open(f_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([t, y], f)


plt.show()
