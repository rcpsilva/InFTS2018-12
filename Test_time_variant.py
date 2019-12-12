from builtins import len
from builtins import dict
from builtins import open

import pertinence_funcs as pf
import numpy as np
import matplotlib.pyplot as plt
from fts_concrete import ConcreteFTS
from fts_time_variant import TimeVariantFTS
from fts_incremental import FinalIncrementalFTS
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

window_size = [300, 250, 150, 100, 75, 50, 25, 10, 5]

bound_type = ['mu-sigma', 'min-max']
partitionner = ['knn', 'uniform']

fdataname = 'final_time-variant_2019-05-20.csv'

dfile = open(fdataname, 'a+')
dfile.write('series,algorithm,order,nsets,bound_type,partitioner,window_size,pearson,spearman,RMSE,MAPE,U,time\n')
dfile.close()


dfile = open(fdataname, 'a+')
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

            for bt in bound_type:
                for p in partitionner:
                    for w in window_size:
                        #  Instantiate fts
                        fts = TimeVariantFTS(n, o, w, bound_type=bt, partitionner=p)

                        # Get forecasts for the whole series
                        forecasts = []
                        current_window_size = []
                        tt = time.process_time()

                        for yt in y:
                            forecasts.append(fts.predict(yt))
                            current_window_size.append(len(fts.window))

                        elapsed_time = time.process_time() - tt

                        if len(np.shape(forecasts)) > 1:
                            y = [temp[0] for temp in y]

                        y = np.array(y)
                        forecasts = np.array(forecasts)

                        rmse = np.sqrt(np.sum((y[1:] - forecasts[:-1]) ** 2) / len(forecasts))
                        mape = np.mean(np.abs((y[1:] - forecasts[:-1]) / y[1:]))
                        corr_p = ss.pearsonr(y[1:], forecasts[:-1])
                        corr_s = ss.spearmanr(y[1:], forecasts[:-1])
                        u_metric = 1/(np.mean(np.abs((y[0:-1] - y[1:]))) / np.mean(np.abs(y[1:] - forecasts[0:-1])))

                        # Save data
                        # 'series,algorithm,order,nsets,bound_type,partitioner,window_size,pearson,spearman,RMSE,MAPE,U,time,\n')
                        row = '{},TVFTS,{},{},{},{},{},{},{},{},{},{},{}\n'.format(s, o, fts.nsets, bt, p, w, corr_p[0], corr_s[0], rmse, mape, u_metric, elapsed_time)
                        dfile.write(row)

dfile.close()


