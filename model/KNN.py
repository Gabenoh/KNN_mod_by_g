import numpy as np
import pandas as pd

from utils import euclidian_dist
from utils import k_nearest_labels


class KNearest_Neighbours(object):

    def __init__(self, k):
        self.k = k
        self.test_set_x = None
        self.train_set_x = None
        self.train_set_y = None

    def fit(self, train_set_x, train_set_y):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

    def predict(self, test_set_x):

        self.test_set_x = test_set_x
        self.distans = euclidian_dist(self.train_set_x, self.test_set_x)
        self.knl = k_nearest_labels(self.distans, self.train_set_y, self.k)
        predictions = []

        for i in self.knl:
            predictions.append(round(np.mean(i)))

        return predictions
