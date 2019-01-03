from tkinter.dnd import Icon

import rule_base_management as rbm
import pertinence_funcs as pf
import numpy as np
import matplotlib.pyplot as plt
from fts_concrete import ConcreteFTS
from fts_time_variant import TimeVariantFTS
from fts_incremental import IncrementalMuSigmaFTS
from fts_incremental_rule_deletion import IncMuSigmaRuleDeletionFTS
from fts_time_variant_adaptive import TimeVariantAdaptiveFTS
from fts_stream import StreamAdaptiveWindowFTS

import scipy.signal as ss

#Fts order
order = 3
nsets = 9
window_size = 100

# Gather sample data
n_samples = 100


x = np.linspace(-np.pi, 10*np.pi, n_samples)
data = 3 + np.sin(x)  # + np.random.normal(0, 0.1, len(x))
data2 = 3 + np.sin(x) + x
data = np.concatenate((data, data2, data+35))
# data = np.random.normal(0, 1, n_samples)
# data = ss.medfilt(data, 101)
t = np.arange(len(data))


# # Generate fts
# fts = TimeVariantFTS(nsets=3, order=order, lb=np.min(data)*1.5, ub=np.max(data)*1.5, window_size=window_size)
#
#
# # Forecast
# forecasts = []
# for x in data:
#     forecast = fts.predict(x)
#     forecasts.append(forecast)

multiplier = 2.698
data_range = np.max(data) - np.min(data)
cfts = ConcreteFTS(nsets=nsets, order=order, lb=np.mean(data) - multiplier*np.std(data),
                   ub=np.mean(data) + multiplier*np.std(data))
cfts.fit(data[:window_size])
cfts_forecast = cfts.predict(data)


# ifts = IncMuSigmaRuleDeletionFTS(nsets=nsets, order=order, deletion=False, bound_type='mu-sigma')
# ifts = TimeVariantAdaptiveFTS(nsets=nsets, order=order, bound_type='mu-sigma')
ifts = StreamAdaptiveWindowFTS(nsets=nsets, order=order, bound_type='mu-sigma', update_type='translate', deletion=True)
# ifts = TimeVariantAdaptiveFTS(nsets=nsets, order=order, bound_type='min-max')
ifts_forecast = []
samples_so_far = 0
count = 1
for d in data:
    print('{} of {}'.format(count, len(data)))
    count = count+1
    samples_so_far = samples_so_far + 1
    ifts_forecast.append(ifts.predict(d))
    if ifts.partitions:
        plt.clf()
        pf.plot_pertinence(ifts.partitions)
        plt.scatter(t, data, s=10)
        plt.scatter(np.arange(1, samples_so_far+1), ifts_forecast, s=7)
        plt.pause(0.01)

plt.scatter(t[1:], data[:-1], s=3)
# plt.plot(np.arange(order + 1, len(t) + 1), cfts_forecast)
plt.show()


# plt.plot(t, data)
# plt.plot(t[t_s-1:], forecast[:(len(forecast)-order+1)])
# plt.show()
