import pertinence_funcs as pf
import numpy as np
import matplotlib.pyplot as plt
from fts_concrete import ConcreteFTS
from fts_time_variant import TimeVariantFTS
from fts_incremental import IncrementalMuSigmaFTS
from fts_incremental_rule_deletion import IncMuSigmaRuleDeletionFTS
from fts_time_variant_adaptive import TimeVariantAdaptiveFTS
from fts_stream import StreamAdaptiveWindowFTS
import pandas as pd
import pickle
import time
from tqdm import tqdm


# Configurations
order = [1, 2, 3]
nsets = [3, 5, 7, 9]
series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

inc_par_deletion = [True, False]
inc_par_bound_type = ['mu-sigma', 'min-max']


def get_exp_data(_fts, _y, _s, _n, _o, _alg_name, up_type, _del_par, _b_par):

    forecasts = []
    tt = time.process_time()
    for yt in _y:
        forecasts.append(_fts.predict(yt))
    elapsed_time = time.process_time() - tt

    y = np.array(_y)
    forecasts = np.array(forecasts)
    rmse = np.sqrt(np.sum((y[1:] - forecasts[0:len(y)]) ** 2) / len(y))
    mape = np.mean(np.abs((y - forecasts) / y))
    u_metric = np.mean(np.abs((y[0:-2] - y[2:]))) / np.mean(np.abs(y[1:] - forecasts[0:len(y)]))

    di = dict(series=_s,
              algrorithm=_alg_name,
              order=_o,
              nsets=_n,
              f_nsets = fts.nsets,
              update_type=up_type,
              deletion=_del_par,
              window_size=fts.window_size,
              bound_type=_b_par,
              ##################################
              RMSE=rmse,
              MAPE=mape,
              U=u_metric,
              ##################################
              time=elapsed_time)

    return di


df = pd.DataFrame()

for s in tqdm(series):
    f_name = 'series_{}.pkl'.format(s)
    #  Load series
    with open(f_name, 'rb') as f:
        t, y = pickle.load(f)

    for o in tqdm(order):
        for n in tqdm(nsets):

            #  Incremental
            print('incremental')
            for del_par in inc_par_deletion:
                for b_par in inc_par_bound_type:

                    del_str = 'delon' if del_par else 'deloff'
                    alg_name = 'fts_nw_{}_{}'.format(del_str, b_par)

                    fts = IncMuSigmaRuleDeletionFTS(nsets=n, order=o, deletion=del_par, bound_type=b_par)

                    d = get_exp_data(fts, y, s, n, o, alg_name, 'translate', del_par, b_par)

                    df = pd.DataFrame(d).append(df, ignore_index=True)
                    df.to_csv('test.csv')

            #  Sliding Window
            print('sliding window')
            for del_par in inc_par_deletion:
                for b_par in inc_par_bound_type:
                    del_str = 'delon' if del_par else 'deloff'
                    alg_name = 'fts_sw_{}_{}'.format(del_str, b_par)

                    fts = StreamAdaptiveWindowFTS(nsets=n, order=o, bound_type=b_par, update_type='retrain',
                                                          deletion=del_par)

                    d = get_exp_data(fts, y, s, n, o, alg_name, 'retrain', del_par, b_par)

                    df = pd.DataFrame(d).append(df, ignore_index=True)
                    df.to_csv('test.csv')