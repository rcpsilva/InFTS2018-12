import rule_base_management as rbm
from concrete_fts import ConcreteFTS
from time_variant_fts import TimeVariantFTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

#Fts order
order = 4
window_size = 100

# Gather sample data
n_samples = 1000


data = np.random.normal(0, 1, n_samples)
data = ss.medfilt(data, 101)
t = np.arange(n_samples)


# Generate fts
fts = TimeVariantFTS(nsets=9, order=order, lb=np.min(data)*1.5, ub=np.max(data)*1.5, window_size=window_size)

# Forecast
forecasts = []
for x in data:
    forecast = fts.predict(x)
    forecasts.append(forecast)

print(forecasts)

plt.plot(t, data)
plt.plot(t[1:], forecasts[:(len(forecasts)-1)])
plt.show()


# plt.plot(t, data)
# plt.plot(t[t_s-1:], forecast[:(len(forecast)-order+1)])
# plt.show()
