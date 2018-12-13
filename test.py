import rule_base_management as rbm
from concrete_fts import ConcreteFTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

#Fts order
order = 3

# Gatter sample data
n_samples = 1000

data = np.random.normal(0, 1, n_samples)
data = ss.medfilt(data, 101)
t = np.arange(n_samples)

t_s = 800
train = data[0:(t_s+1)]  # 0:400
test = data[t_s-order:]  # 397:500

# Generate fts
fts = ConcreteFTS(nsets=24, order=order, lb=np.min(data)*1.5, ub=np.max(data)*1.5)
print(fts.partitions)
fts.fit(train)
rbm.print_rule_base(fts.rule_base)

# Forecast
forecast = fts.predict(test)
print(forecast)

plt.plot(t, data)
plt.plot(t[t_s-1:], forecast[:(len(forecast)-order+1)])
plt.show()
