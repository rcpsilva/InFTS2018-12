from FTS import FTS
import rule_base_management as rbm
import partition_utilities as pu
import pertinence_funcs as pf
import numpy as np
from forecast_funcs import forecast_weighted_average_t_sets


class TimeVariantFTS(FTS):
    """ Implements a time variant FTS for Streaming data

    """

    def __init__(self, nsets, order, window_size, bound_type='min-max'):

        self.nsets = nsets
        self.order = order
        self.fuzzy_sets = np.arange(self.nsets)
        self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)
        self.window = []  # Stores the last "order" data
        self.partitions = []
        self.alpha_cut = 0
        self.window_size = window_size

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
            self.window.pop(0)
            self.window.append(x)
            self.fit(self.window)
            forecast = forecast_weighted_average_t_sets(self.window[len(self.window)-self.order:], self.rule_base,
                                                        self.alpha_cut, self.partitions, self.nsets, self.order)
            return forecast
