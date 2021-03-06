import pertinence_funcs as pf
from itertools import product
import numpy as np
import warnings


def find_inappropriate_rules(x, alpha_cut, partitions, nsets, order):
    un_rules = []

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

    for r in rules:
        #  Find rule in the rule base
        #  Find rule index
        index = np.dot([nsets ** o for o in np.arange(order - 1, -1, -1)], np.array(r))
        un_rules.append(index)

    return un_rules


def forecast_weighted_average_t_sets(x, rule_base, alpha_cut, partitions, nsets, order):
    """ Produces a forecast value given an input x and the FTS parameters

    Args:
        x:
        rule_base:
        alpha_cut:
        partitions:
        nsets:
        order:
    Returns:
        Forecast value
    """

    if len(x) < order:
        warnings.warn('Input size less than Order. No forecast returned')
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

    #print(rules)

    consequents = []
    for r in rules:
        #  Find rule in the rule base and compute the consequent
        #  Find rule index
        index = np.dot([nsets ** o for o in np.arange(order - 1, -1, -1)], np.array(r))

        c = [rule_base[0][index][len(rule_base[0][index]) - 1]] \
            if not rule_base[1][index] else rule_base[1][index]

        #  print('--------------------')
        #  print('{} -> {}'.format(rule_base[0][index], c))
        c = [partitions[i][1] for i in c]
        #  print(c)

        consequents.append(np.mean(c))
        #  print(rule_base[1][index])

    pertinences = [np.prod(e) for e in list(product(*l_ps))]

    forecast = x[len(x) - 1]
    if consequents:
        forecast = np.dot(consequents, pertinences) / np.sum(pertinences)

    return forecast
