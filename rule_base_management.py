import numpy as np
import itertools


def fuzzify_max_pertinence(pertinence):
    """ Fuzzify values based on  maximum pertinence

    Args:
        pertinence: list of pertinences
    Returns:
        Fuzzified value
    """

    return np.argmax(pertinence)


def generate_rule(pertinence):
    """ Generates a rule given a list of prtinence lists

    Args:
       pertinence: list of lists of pertinence (len(pertinence) = order+1)
    Returns:
       rule: Fuzzy logic rule
    """
    antecedent = [fuzzify_max_pertinence(p) for p in pertinence[0:(len(pertinence)-1)]]
    consequent = fuzzify_max_pertinence(pertinence[len(pertinence)-1])
    rule = [antecedent, consequent]

    return rule


def init_rule_base(fuzzy_sets, order):
    """ Generates a rule given a list of prtinence lists

    Args:
        fuzzy_sets: list of fuzzy_sets
        order: fuzzy time series order
    Returns:
        rule_base: rule base (all possible antecedents + empty consequents)
    """

    antecedents = list(itertools.product(fuzzy_sets, repeat=order))
    consequents = []
    for i in range(len(antecedents)):
        consequents.append(set())

    rule_base = [antecedents, consequents]

    return rule_base


def add_rule(rule_base, rule, nsets, order):
    """ Generates a rule given a list of prtinence lists

    Args:
        rule_base: Rule base
        rule: The rule to added rule[0] = antencedent , rule[1] = consequent
        nsets: Number of fuzzy sets
        order: FTS order
    Returns:
        rule_base: rule base (all possible antecedents + updated consequents)
    """
    antecedents = rule_base[0]
    consequents = rule_base[1]

    # Find rule index
    index = np.dot([nsets ** o for o in np.arange(order - 1, -1, -1)], np.array(rule[0]))

    # Update consequent
    consequents[index].add(rule[1])

    # Return the updated rule base
    rule_base = [antecedents, consequents]  # Not sure if this is needed
    return rule_base


def update_rule_base(rule_base,map_old_new, fuzzy_sets, order):

    new_rule_base = init_rule_base(fuzzy_sets, order)

    # For each rule in in rule_base, find the corresponding rule in new_rule_base
    # Copy and Translate consequent

    return new_rule_base


def print_rule_base(rule_base):
    """ Prints a rule base

    Args:
        rule_base: Rule base
    Returns:

    """

    antecedents = rule_base[0]
    consequents = rule_base[1]

    for i in range(len(antecedents)):
        if consequents[i]:
            s = ''
            for set_id in antecedents[i]:
                s = s + 'A{} '.format(set_id)

            s = s + '-> '

            for set_id in consequents[i]:
                s = s + 'A{} '.format(set_id)

            print(s)
