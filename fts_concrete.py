from FTS import FTS
import rule_base_management as rbm
import partition_utilities as pu
import pertinence_funcs as pf
import numpy as np
from forecast_funcs import forecast_weighted_average_t_sets


class ConcreteFTS(FTS):

    def __init__(self, nsets, order, lb, ub):

        self.nsets = nsets
        self.order = order
        self.lb = lb
        self.ub = ub
        self.fuzzy_sets = np.arange(self.nsets)
        self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)
        self.partitions = pu.generate_t_partitions(nsets, lb, ub)
        self.alpha_cut = 0

    def fit(self, data):

        for i in range(len(data)-self.order):
            window = data[i:(i+self.order+1)]

            # Get pertinences
            pertinence_list = pf.t_pertinence_list(window, self.partitions)

            # Build rule
            rule = rbm.generate_rule(pertinence_list)

            # Add rule to the rule base
            rbm.add_rule(self.rule_base, rule, self.nsets, self.order)

    def predict(self, x):
        predictions = []
        for i in range(len(x)-self.order):
            predictions.append(forecast_weighted_average_t_sets(x[i:(i+self.order)], self.rule_base, self.alpha_cut,
                                                                self.partitions, self.nsets, self.order))
        return predictions
