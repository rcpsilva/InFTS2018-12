from FTS import FTS
import rule_base_management as rbm
import partition_utilities as pu
import pertinence_funcs as pf
import numpy as np
from forecast_funcs import forecast_weighted_average_t_sets
from pertinence_funcs import fuzzify_x_list_t


class IncrementalMuSigmaFTS(FTS):
    """ Implements an incremental FTS for Streaming data

    """

    def __init__(self, nsets, order, bound_type):

        self.nsets = nsets
        self.order = order
        self.lb = None
        self.ub = None
        self.fuzzy_sets = np.arange(self.nsets)
        self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)
        self.window = []  # Stores the last "order" data
        self.partitions = None
        self.alpha_cut = 0
        self.window_size = order+1

        self.mu = 0
        self.sigma = 0
        self.n = 0  # Number of samples
        self.sigma_multiplier = 2.698

        self.min_val = 0
        self.max_val = 0

        self.bound_type = bound_type

    def fit(self, data):

        if self.bound_type == 'min-max':
            data_range = self.max_val - self.min_val
            lb = self.min_val - data_range*0.5
            ub = self.max_val + data_range * 0.5
        else:  # self.bound_type == 'mu_sigma'
            lb = self.mu - self.sigma * self.sigma_multiplier
            ub = self.mu + self.sigma * self.sigma_multiplier

        if self.partitions:
            old_partitions = self.partitions[:]
        else:
            old_partitions = pu.generate_t_partitions(self.nsets,lb,ub)

        # 1) Compute the new partitions
        self.partitions = pu.generate_t_partitions(self.nsets,lb,ub)

        # 2) Verify the pertinence of the old sets centers with respect to the new partitions
        old_centers = [p[1] for p in old_partitions]
        f = fuzzify_x_list_t(old_centers, self.partitions)

        # 3) Compute the final set of partitions
        up_partitions = self.partitions + [old_partitions[i] for i in range(len(f)) if f[i] < 0]
        up_partitions = sorted(up_partitions, key=lambda n_p: n_p[1])
        self.partitions = up_partitions

        # 4) Compute the mappings required to update the rule base
        map_old_new = fuzzify_x_list_t(old_centers, self.partitions)

        # 5) Update rules
        self.rule_base = rbm.update_rule_base(self.rule_base, map_old_new, np.arange(len(self.partitions)), self.order)

        # 6) Add new rule
        # Get pertinences
        pertinence_list = pf.t_pertinence_list(data, self.partitions)

        # Build rule
        rule = rbm.generate_rule(pertinence_list)

        # Add rule to the rule base
        rbm.add_rule(self.rule_base, rule, self.nsets, self.order)

    def predict(self, x):

        if len(self.window) < self.window_size:  # If there is not enough data, persist
            self.window.append(x)
            self.n = self.n+1
            self.update_mu_and_sigma(x)
            self.update_min_and_max(x)
            return x
        else:
            self.window.pop(0)
            self.window.append(x)

            self.n = self.n + 1
            self.update_mu_and_sigma(x)
            self.update_min_and_max(x)

            self.fit(self.window)
            forecast = forecast_weighted_average_t_sets(self.window[len(self.window) - self.order:], self.rule_base,
                                                        self.alpha_cut, self.partitions, self.nsets, self.order)
            return forecast

    def update_mu_and_sigma(self, x):

        # Update mean
        old_mu = self.mu
        self.mu = old_mu + (x - old_mu) / self.n

        # Update standard deviation
        s = (self.sigma ** 2 * self.n) + (x - old_mu)*(x-self.mu)
        self.sigma = np.sqrt(s / self.n)

    def update_min_and_max(self, x):

        if x < self.min_val:
            self.min_val = x

        if x > self.max_val:
            self.max_val = x