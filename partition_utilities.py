import numpy as np

def generate_t_partitions(n, lb, ub):
    """ Generates triangular and equally spaced partitions

    Args:
        n: number of partitions
        lb: lower bound for the partitions
        ub: upper bound for the partitions
    Returns:
        partitions: list of triangular partitions
    """

    delta = (ub-lb)/(n-1)
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


def generate_t_partitions_knn(data, n, lb, ub):
    """ Generates triangular and equally spaced partitions

    Args:
        data:
        n: number of partitions
        lb: lower bound for the partitions
        ub: upper bound for the partitions
    Returns:
        partitions: list of triangular partitions
    """
    delta = (ub-lb)/(n-1)
    centers = [i*delta+lb for i in range(n)]
    samples_per_cluster = np.ones(n)

    for x in data:
        #  Find the closest center
        closest = np.argmin([np.abs(x - c) for c in centers])
        #  Update the center
        centers[closest] = ((centers[closest] * samples_per_cluster[closest]) + x) / (samples_per_cluster[closest] + 1)
        samples_per_cluster[closest] += 1

    partitions = generate_t_partitions_from_centers(centers)

    return partitions
