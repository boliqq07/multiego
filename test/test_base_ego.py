import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from multiego.base_ego import BaseEgo
from multiego.ego import search_space


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        parameters = {'C': [0.1, 1, 10]}
        model = GridSearchCV(SVR(), parameters)
        ###

        X, y = load_boston(return_X_y=True)
        X = X[:, :5]  # (简化计算，示意)
        searchspace_list = [
            np.arange(0.01, 1, 0.2),
            np.array([0, 20, 30]),
            np.arange(1, 10, 2),
            np.array([0, 1]),
            np.arange(0.4, 0.6, 0.1),
        ]

        searchspace = search_space(*searchspace_list)
        self.y = y
        self.searchspace = searchspace

    def test_something(self):
        mean_std = np.random.random((self.searchspace.shape[0], 1000))
        be = BaseEgo()

        try:
            rank = be.egosearch(self.y, self.searchspace, mean_std, rankway="ego", return_type="pd", flexibility=20)

        except AssertionError:
            pass
        try:
            rank = be.egosearch(self.y, self.searchspace, mean_std, rankway="ego", return_type="pd", flexibility=100)
        except UserWarning:
            pass


if __name__ == '__main__':
    unittest.main()
