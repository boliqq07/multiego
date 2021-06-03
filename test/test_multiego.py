import unittest
# -*- coding: utf-8 -*-

# @Time    : 2020/11/25 14:50
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np

from multiego.base_multiplyego import search_space, BaseMultiplyEgo


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        parameters = {'C': [0.1, 1, 10]}

        model = GridSearchCV(SVR(), parameters)
        parameters = {'alpha': [0.1, 1, 10]}
        model2 = GridSearchCV(Lasso(), parameters)

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
        self.model1 =model
        self.model2 =model2

        self.y = np.concatenate([y.reshape(-1,1),2*y.reshape(-1,1)],axis=1)
        self.searchspace = searchspace

    def test_something(self):
        me = BaseMultiplyEgo(n_jobs=1)
        mean_std = [np.random.random((self.searchspace.shape[0], 2)),np.random.random((self.searchspace.shape[0], 2))]
        predict_y_all = np.random.random((self.searchspace.shape[0],1000, 2))
        rank = me.egosearch(self.y, self.searchspace, meanandstd_all=mean_std, predict_y_all= predict_y_all )
        rank = me.egosearch(self.y, self.searchspace, meanandstd_all=mean_std, predict_y_all= predict_y_all )


if __name__ == '__main__':
    unittest.main()

