import numpy as np


def k_nearest_labels(diststance, y_known, k):
    num_pred = diststance.shape[0]
    n_nearest = []

    for j in range(num_pred):
        dst = diststance[j]
        closest_y = y_known[dst.argsort()[:k]]
        n_nearest.append(closest_y)
    return np.asarray(n_nearest)
