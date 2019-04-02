from FTS import FTS
import rule_base_management as rbm
import partition_utilities as pu
import pertinence_funcs as pf
import numpy as np
from forecast_funcs import forecast_weighted_average_t_sets
from pertinence_funcs import fuzzify_x_list_t
from forecast_funcs import find_unappropriate_rules


class StreamAdaptiveWindowFTS(FTS):
    """ Implements a time variant FTS for Streaming data with adaptive window size

    """

    def __init__(self, nsets, order, bound_type='min-max', update_type='retrain', deletion=False, max_window_size=None):

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
        self.bound_type = bound_type  # ('min-max'/'mu-sigma')
        self.update_type = update_type  # ('retrain'/'translate')
        self.deletion = deletion
        self.last_forecast = None

        self.max_window_size = max_window_size

    def fit(self, data):
        if self.update_type == 'retrain':
            self.retrain(data)
        elif self.update_type == 'translate':
            self.translate(data)

    def predict(self, x):

        if len(self.window) < self.window_size:  # If there is not enough data, use persistence
            self.window.append(x)
            self.last_forecast = x
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

            if self.deletion:
                self.clean_up(x)

            # Remove one window
            if len(self.window) > 2:
                r1_window = self.window[:]
                r1_window.pop(0)
                r1_window.pop(1)
                r1_window.append(x)

                if len(r1_window) > self.window_size:
                    self.fit(r1_window)
                    self.r1_forecast = forecast_weighted_average_t_sets(r1_window[len(r1_window)-self.order:],
                                                                        self.rule_base, self.alpha_cut, self.partitions,
                                                                        self.nsets, self.order)
            else:
                self.r1_forecast = x

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
                if self.max_window_size:
                    if len(self.window) > self.max_window_size:
                        self.window = self.window[1:]
            else:
                forecast = self.c_forecast
                self.window = c_window[:]

            self.last_forecast = forecast

            return forecast

    def translate(self, data):

        if not self.rule_base:
            self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)

        if self.bound_type == 'min-max':
            data_range = np.max(data) - np.min(data)
            lb = np.min(data) - data_range * 0.5
            ub = np.max(data) + data_range * 0.5
        else:  # self.bound_type == 'mu_sigma'
            lb = np.mean(data) - np.std(data) * self.sigma_multiplier
            ub = np.mean(data) + np.std(data) * self.sigma_multiplier

        if self.partitions:
            old_partitions = self.partitions[:]
        else:
            old_partitions = pu.generate_t_partitions(self.nsets, lb, ub)

        # 1) Compute the new partitions
        self.partitions = pu.generate_t_partitions(self.nsets, lb, ub)

        # 2) Verify the pertinence of the old sets centers with respect to the new partitions
        old_centers = [p[1] for p in old_partitions]
        f = fuzzify_x_list_t(old_centers, self.partitions)

        # 3) Compute the final set of partitions
        up_partitions = self.partitions + [old_partitions[i] for i in range(len(f)) if f[i] < 0]
        up_partitions = sorted(up_partitions, key=lambda n_p: n_p[1])
        self.partitions = up_partitions
        self.nsets = len(self.partitions)

        # 4) Compute the mappings required to update the rule base
        map_old_new = fuzzify_x_list_t(old_centers, self.partitions)

        # 5) Update rules
        self.rule_base = rbm.update_rule_base(self.rule_base, map_old_new, np.arange(len(self.partitions)), self.order)

        # 6) Add new rule
        # Get pertinences
        pertinence_list = pf.t_pertinence_list(data[(len(data)-self.order-1):], self.partitions)

        # Build rule
        rule = rbm.generate_rule(pertinence_list)

        # Add rule to the rule base
        rbm.add_rule(self.rule_base, rule, self.nsets, self.order)

    def retrain(self, data):
        # Resets the rule base
        self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)

        # Regenerate partitions
        data_range = np.max(data) - np.min(data)

        if self.bound_type == 'min-max':
            self.partitions = pu.generate_t_partitions(self.nsets, np.min(data) - data_range * 0.1,
                                                       np.max(data) + data_range * 0.1)
        else:  # self.bound_type == 'mu-sigma'
            self.partitions = pu.generate_t_partitions(self.nsets, np.mean(data) - np.std(data) * self.sigma_multiplier,
                                                       np.mean(data) + np.std(data) * self.sigma_multiplier)

        # Populate the rule base
        for i in range(len(data) - self.order):
            window = data[i:(i + self.order + 1)]

            # Get pertinences
            pertinence_list = pf.t_pertinence_list(window, self.partitions)

            # Build rule
            rule = rbm.generate_rule(pertinence_list)

            # Add rule to the rule base
            rbm.add_rule(self.rule_base, rule, self.nsets, self.order)

    def clean_up(self, x):
        if self.deletion:
            # Check if the model got the last linguistic value right
            if self.partitions and (fuzzify_x_list_t([x], self.partitions)[0] !=
                                    fuzzify_x_list_t([self.last_forecast], self.partitions)[0]):
                # Otherwise, find and remove the unappropriate rules
                un_rules = find_unappropriate_rules(self.window[(len(self.window)-self.order):], self.alpha_cut,
                                                    self.partitions, self.nsets, self.order)
                for u_r in un_rules:
                    self.rule_base[1][u_r] = set()
