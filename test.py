import rule_base_management as rbm
from concrete_fts import ConcreteFTS
import pertinence_funcs as pf
from time_variant_fts import TimeVariantFTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

#Fts order
order = 3
window_size = 100

# Gather sample data
n_samples = 300


x = np.linspace(-np.pi, 5*np.pi, n_samples)
data = 3 + np.sin(x) # + np.random.normal(0, 0.1, len(x))

# data = np.random.normal(0, 1, n_samples)
# data = ss.medfilt(data, 101)
t = np.arange(n_samples)


# # Generate fts
# fts = TimeVariantFTS(nsets=3, order=order, lb=np.min(data)*1.5, ub=np.max(data)*1.5, window_size=window_size)
#
#
# # Forecast
# forecasts = []
# for x in data:
#     forecast = fts.predict(x)
#     forecasts.append(forecast)

data_range = np.max(data) - np.min(data)
cfts = ConcreteFTS(nsets=30, order=order, lb=np.min(data)-data_range*0.2, ub=np.max(data)+data_range*0.2)
cfts.fit(data[:window_size])
cfts_forecast = cfts.predict(data)

pf.plot_pertinence(cfts.partitions)
plt.plot(t, data)
plt.plot(t[order:], cfts_forecast)
#plt.plot(t[1:], forecasts[:(len(forecasts)-1)])
plt.show()


# plt.plot(t, data)
# plt.plot(t[t_s-1:], forecast[:(len(forecast)-order+1)])
# plt.show()
