import numpy as np
import matplotlib.pyplot as plt


def t_pertinence(x, partition):
    """ Computes the pertinence of x with respect to a triangular fuzzy set given parameterized by partition

    Args:
        x: point
        partition: triangular partition parameters
    Returns:
        pertinence: pertinence values
    """

    pertinence = 0
    if x < partition[0] or x > partition[2]:
        pertinence = 0
    elif x == partition[1]:
        pertinence = 1
    elif partition[0] <= x < partition[1]:
        pertinence = (x-partition[0])/(partition[1]-partition[0])
    elif partition[1] < x <= partition[2]:
        pertinence = (x - partition[2]) / (partition[1] - partition[2])

    return pertinence


def t_pertinence_list(lx, partitions):
    """ Generates triangular and equally spaced partitions

    Args:
        lx: point
        partitions: list triangular fuzzy sets partitions
    Returns:
        pertinence_list: stores the pertinences of each x in ls with respect to each set in partitions
    """

    pertinence_list = []
    for x in lx:
        pertinence = []
        for p in partitions:
            pertinence.append(t_pertinence(x, p))
        pertinence_list.append(pertinence)

    return pertinence_list


def fuzzify_x_list_t(x, partitions):
    """ Fuzzifies a list of points given a set o triangular partitions

        Args:
            x: list point
            partitions: list triangular fuzzy sets partitions
        Returns:
            f_list: list of fuzzified values
    """

    f_list = []

    for xi in x:
        p_list = t_pertinence_list([xi], partitions)
        if np.max(p_list[0]) > 0.5:
            f_list.append(np.argmax(p_list[0]))
        else:
            f_list.append(-1)  # No fuzzy set represents the value

    return f_list


def plot_pertinence(partitions):

    x = np.arange(min(list(map(min, partitions))), max(list(map(max, partitions))), 0.01)
    for i in range(len(partitions)):
        y = [t_pertinence(xi, partitions[i]) * 20 - 20 for xi in x]
        plt.plot(y, x)

