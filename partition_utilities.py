
def generate_t_partitions(n, lb, ub):
    """ Generates triangular and equally spaced partitions

    Args:
        n: number of partitions
        lb: lower bound for the partitions
        ub: upper bound for the partitions
    Returns:
        partitions: list of triangular partitions
    """

    delta = (ub-lb)/n
    partitions = [[(i*delta+lb)-delta, i*delta+lb, (i*delta+lb)+delta] for i in range(n)]

    return partitions


def generate_t_partitions_from_centers(centers):
    """ Generates triangular partitions from a list of centers

    Args:
        centers: centers of each partition
    Returns:
        partitions: list of triangular partitions
    """

    partitions = [[centers[i-1], centers[i], centers[i+1]] for i in range(1, len(centers)-1)]
    partitions = [[centers[0] - (centers[1]-centers[0]), centers[0], centers[1]]] + partitions
    partitions.append([centers[len(centers)-2], centers[len(centers)-1], centers[len(centers)-1] +
                       (centers[len(centers)-1]-centers[len(centers)-2])])

    return partitions

