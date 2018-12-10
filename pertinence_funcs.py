def t_pertinence(x, partition):
    """ Generates triangular and equally spaced partitions

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
