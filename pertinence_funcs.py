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
