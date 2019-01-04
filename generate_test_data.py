import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)


kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
# kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

# Specify Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel)

# Plot prior
X_ = np.linspace(0, 80, 1000)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)

y_samples = gp.sample_y(X_[:, np.newaxis], 1) + 20

print(X_)
print(len(y_samples))

y = np.array([y[0] for y in y_samples[:]])
t = np.arange(0, len(y+1), 1)

y = np.concatenate((y*5 - 75, y + 2.5, y*5 - 75))
t = np.arange(0, len(y+1), 1)


print(y)

plt.plot(t, y, lw=1)
plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

with open('series_13.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([t, y], f)


plt.show()
