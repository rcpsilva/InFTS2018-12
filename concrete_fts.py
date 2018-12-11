import FTS
import rule_base_management as rbm
import partition_utilities as pu
import pertinence_funcs as pf
import numpy as np


class ConcreteFTS:

    def __init__(self, nsets, order, lb, ub):

        self.nsets = nsets
        self.order = order
        self.fuzzy_sets = np.arange(self.nsets)
        self.rule_base = rbm.init_rule_base(self.fuzzy_sets, self.order)
        self.window = []  # Stores the last "order" data
        self.partitions = pu.generate_t_partitions(nsets, lb, ub)

    def fit(self, data):

        # Get pertinences
        pertinence_list = []

        for x in data:
            pertinence = []
            for p in self.partitions:
                pertinence.append(pf.t_pertinence(x, p))
            pertinence_list.append(pertinence)

        # Build rule
        rule = rbm.generate_rule(pertinence_list)

        # Add rule to the rule base

        rbm.add_rule(self.rule_base, rule, self.nsets, self.order)

    def predict(self):
        pass


fts = ConcreteFTS(3, 1, 1, 4)
print(fts.partitions)
fts.fit([1, 2])
rbm.print_rule_base(fts.rule_base)