from FTS import FTS
import rule_base_management as rbm
import partition_utilities as pu
import pertinence_funcs as pf
import numpy as np
from forecast_funcs import forecast_weighted_average_t_sets


class TimeVariantAdaptiveFTS(FTS):
    """ Implements a time variant FTS for Streaming data with adaptive window size

    """

    def __init__(self, nsets, order, bound_type='min-max'):

        self.nsets = nsets
        self.order = order
        self.lb = 0
        self.ub = 0
        self.fuzzy_sets = np.arange(self.nsets)
        self.rule_base = []  # rbm.init_rule_base(self.fuzzy_sets, self.order)
        self.window = []  # Stores the last "order" data
        self.partitions = []  # pu.generate_t_partitions(nsets, lb, ub)
        self.alpha_cut = 0
        self.window_size = order+1

        self.r1_forecast = None
        self.c_forecast = None
        self.i1_forecast = None
        self.sigma_multiplier = 2.698
        self.bound_type = bound_type

    def fit(self, data):
        # Resets the rule base
        self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)

        # Regenerate partitions
        data_range = np.max(data) - np.min(data)

        if self.bound_type == 'min-max':
            self.partitions = pu.generate_t_partitions(self.nsets, np.min(data) - data_range*0.5,
                                                    np.max(data) + data_range*0.5)
        else:  # self.bound_type == 'mu-sigma'
            self.partitions = pu.generate_t_partitions(self.nsets, np.mean(data) - np.std(data) * self.sigma_multiplier,
                                                       np.mean(data) + np.std(data) * self.sigma_multiplier)

        # Populate the rule base
        for i in range(len(data)-self.order):
            window = data[i:(i+self.order+1)]

            # Get pertinences
            pertinence_list = pf.t_pertinence_list(window, self.partitions)

            # Build rule
            rule = rbm.generate_rule(pertinence_list)

            # Add rule to the rule base
            rbm.add_rule(self.rule_base, rule, self.nsets, self.order)

    def predict(self, x):

        if len(self.window) < self.window_size:  # If there is not enough data, use persistence
            self.window.append(x)
            return x
        else:
            error_list = []
            if self.r1_forecast and self.i1_forecast and self.c_forecast:
                error_list.append(np.abs(self.i1_forecast - x))
                error_list.append(np.abs(self.c_forecast - x))
                error_list.append(np.abs(self.r1_forecast - x))
                champion = np.argmin(error_list)
            elif self.i1_forecast and self.c_forecast:
                error_list.append(np.abs(self.i1_forecast - x))
                error_list.append(np.abs(self.c_forecast - x))
                champion = np.argmin(error_list)
            else:
                champion = 0

            # Remove one window
            r1_window = self.window[:]
            r1_window.pop(0)
            r1_window.pop(1)
            r1_window.append(x)

            if len(r1_window) > self.window_size:
                self.fit(r1_window)
                self.r1_forecast = forecast_weighted_average_t_sets(r1_window[len(r1_window)-self.order:],
                                                                    self.rule_base, self.alpha_cut, self.partitions,
                                                                    self.nsets, self.order)

            # Constant size window
            c_window = self.window[:]
            c_window.pop(0)
            c_window.append(x)

            self.fit(c_window)
            self.c_forecast = forecast_weighted_average_t_sets(c_window[len(c_window)-self.order:], self.rule_base,
                                                               self.alpha_cut, self.partitions, self.nsets, self.order)

            # Increase one window
            i1_window = self.window[:]
            i1_window.append(x)

            self.fit(c_window)
            self.i1_forecast = forecast_weighted_average_t_sets(i1_window[len(i1_window)-self.order:], self.rule_base,
                                                                self.alpha_cut, self.partitions, self.nsets, self.order)

            if champion == 2:
                forecast = self.r1_forecast
                self.window = r1_window[:]
            elif champion == 0:
                forecast = self.i1_forecast
                self.window = i1_window[:]
            else:
                forecast = self.c_forecast
                self.window = c_window[:]

            return forecast
