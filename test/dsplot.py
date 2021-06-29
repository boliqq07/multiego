# -*- coding: utf-8 -*-

# @Time    : 2021/6/28 20:35
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from mgetool.exports import Store
from mgetool.show import BasePlot
from multiego.base_multiplyego import BaseMultiplyEgo

from multiego.base_ego import BaseEgo
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle

from multiego.ego import Ego

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    import numpy as np
    from multiego.multiplyego import search_space, MultiplyEgo
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    #####model2#####
    parameters = {'C': [1, 10]}
    model2 = GridSearchCV(SVR(), parameters)

    me = BaseMultiplyEgo()

    st=Store()

    np.random.seed(0)

    n=3

    y=np.random.random(size=(100,n))

    me.pareto_front_point(y,sign=None)

    yall = np.random.random(size=(200,1, n))
    yall = yall+np.random.random(size=(200,  1000 , n))/20

    re = me.rank(y=y,predict_y_all=yall)
