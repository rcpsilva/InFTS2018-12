from builtins import len
from builtins import dict
from builtins import open

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
import scipy.stats as ss

# Configurations

series = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

order = [3, 2, 1]
nsets = [3, 5, 7, 9]

slid_wind_par_window_size = [None, 100, 400]

inc_par_deletion = [True, False]
inc_par_bound_type = ['mu-sigma', 'min-max']
inc_par_translation_threshold = [0.5, 0.25, 0.1]


def get_exp_data(_fts, _y, _s, _n, _o, _alg_name, up_type, _del_par, _b_par, translation_threshold=-1):

    forecasts = []
    current_window_size = []
    tt = time.process_time()
    for yt in _y:
        forecasts.append(_fts.predict(yt))
        current_window_size.append(len(_fts.window))

    elapsed_time = time.process_time() - tt

    if len(np.shape(forecasts)) > 1:
        _y = [temp[0] for temp in _y]

    _y = np.array(_y)
    forecasts = np.array(forecasts)

    rmse = np.sqrt(np.sum((_y[1:] - forecasts[:-1]) ** 2) / len(forecasts))
    mape = np.mean(np.abs((_y[1:] - forecasts[:-1])/_y[1:]))
    corr_p = ss.pearsonr(_y[1:], forecasts[:-1])
    corr_s = ss.spearmanr(_y[1:], forecasts[:-1])
    u_metric = np.mean(np.abs((_y[0:-2] - _y[2:]))) / np.mean(np.abs(_y[1:] - forecasts[0:-1]))

    di = dict(series=[_s],
              algrorithm=[_alg_name],
              order=[_o],
              nsets=[_n],
              f_nsets=[fts.nsets],
              update_type=[up_type],
              deletion=[_del_par],
              window_size=[np.mean(current_window_size)],
              bound_type=[_b_par],
              translation_threshold=[translation_threshold],
              ##################################
              pearson=[corr_p[0]],
              spearman=[corr_s[0]],
              RMSE=[rmse],
              MAPE=[mape],
              U=[u_metric],
              ##################################
              time=[elapsed_time])

    return di


# df = pd.read_csv('test_corr_2019-01-15.csv')  # pd.DataFrame()
# print(df.head())
df = pd.DataFrame()
fdataname = 'test_corr_2019-03-24.csv'

for s in tqdm(series):
    f_name = 'data_sets/series_{}.pkl'.format(s)
    #  Load series
    with open(f_name, 'rb') as f:
        t, y = pickle.load(f)

    # Treat inputs
    if len(y.shape) > 1 and y.shape[0] > 1:
        y = [_y[0] for _y in y]
    else:
        y = [_y for _y in y]

    for o in tqdm(order):
        for n in tqdm(nsets):

            #  Incremental
            # for del_par in inc_par_deletion:
            #     for b_par in inc_par_bound_type:
            #         for par_tt in inc_par_translation_threshold:
            #
            #             del_str = 'delon' if del_par else 'deloff'
            #             alg_name = 'inc_{}_{}'.format(del_str, b_par)
            #
            #             fts = IncMuSigmaRuleDeletionFTS(nsets=n, order=o, deletion=del_par, bound_type=b_par,
            #                                             translation_threshold=par_tt)
            #
            #             d = get_exp_data(fts, y, s, n, o, alg_name, 'translate', del_par, b_par,
            #                              translation_threshold=par_tt)
            #
            #             df = pd.DataFrame(d).append(df, ignore_index=True)
            #             df.to_csv(fdataname)

            #  Sliding Window
            for window_size in slid_wind_par_window_size:
                for del_par in [False]:
                    for b_par in inc_par_bound_type:
                        del_str = 'delon' if del_par else 'deloff'
                        alg_name = 'sw_{}_{}'.format(del_str, b_par)

                        fts = StreamAdaptiveWindowFTS(nsets=n, order=o, bound_type=b_par, update_type='retrain',
                                                              deletion=del_par, max_window_size=window_size)

                        d = get_exp_data(fts, y, s, n, o, alg_name, 'retrain', del_par, b_par)

                        df = pd.DataFrame(d).append(df, ignore_index=True)
                        df.to_csv(fdataname)

            # Fixed window
            # for window_size in slid_wind_par_window_size[1:]:
            #     for b_par in inc_par_bound_type:
            #         for del_par in [False]:
            #
            #             del_str = 'delon' if del_par else 'deloff'
            #             alg_name = 'tv_{}_{}'.format(del_str, b_par)
            #
            #             fts = TimeVariantFTS(nsets=n, order=o, window_size=window_size, bound_type=b_par)
            #             d = get_exp_data(fts, y, s, n, o, alg_name, 'retrain', del_par, b_par)
            #
            #             df = pd.DataFrame(d).append(df, ignore_index=True)
            #             df.to_csv(fdataname)
