import pertinence_funcs as pf
from itertools import product
import numpy as np


def forecast_weighted_average_t_sets(x, rule_base, alpha_cut, partitions, nsets, order):
    """ Generates triangular and equally spaced partitions

    Args:
        x:
        rule_base:
        partitions:
        nsets:
        order:
    Returns:
        Forecast value
    """

    if len(x) < order:
        print('Input size less than Order. No forecast returned')
        return None

    # Find matching rules
    pertinence_list = pf.t_pertinence_list(x, partitions)

    l_sets = []
    l_ps = []
    for p in pertinence_list:
        x_sets = []  # List of valid sets (pertinence > 0)
        x_ps = []  # List of pertinences with respect to the valid sets
        for i in range(len(p)):
            if p[i] > alpha_cut:
                x_sets.append(i)
                x_ps.append(p[i])
        l_sets.append(x_sets)
        l_ps.append(x_ps)

    # Find rules with pertinence > 0
    rules = list(product(*l_sets))

    # print(rules)

    consequents = []
    for r in rules:
        #  Find rule in the rule base and compute the consequent
        #  Find rule index
        index = np.dot([nsets ** o for o in np.arange(order - 1, -1, -1)], np.array(r))

        c = [rule_base[0][index][len(rule_base[0][index]) - 1]] \
            if not rule_base[1][index] else rule_base[1][index]

        c = [partitions[i][1] for i in c]

        consequents.append(np.mean(c))

    pertinences = [np.prod(e) for e in list(product(*l_ps))]

    return np.dot(consequents, pertinences) / np.sum(pertinences)
