import numpy as np


def euclidian_dist(x_known, x_unknown):
    num_pred = x_unknown.shape[0]
    num_data = x_known.shape[0]

    dists = np.zeros((num_pred, num_data))

    for i in range(num_pred):
        for j in range(num_data):
            dists[i, j] = np.sqrt(np.sum((x_known[j] - x_unknown[i]) ** 2))

    return dists
