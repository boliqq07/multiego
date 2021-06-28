# -*- coding: utf-8 -*-

# @Time    : 2020/11/2 17:49
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    import numpy as np
    from multiego.ego import search_space, Ego
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    #####model1#####
    model = SVR()
    ###

    #####model2#####
    parameters = {'C': [0.1, 1, 10]}
    model = GridSearchCV(SVR(), parameters)
    ###

    X, y = load_boston(return_X_y=True)
    X = X[:, :5]  # (简化计算，示意)
    searchspace_list = [
        np.arange(0.01, 1, 0.1),
        np.array([0, 20, 30, 50, 70, 90]),
        np.arange(1, 10, 1),
        np.array([0, 1]),
        np.arange(0.4, 0.6, 0.02),
    ]
    searchspace = search_space(*searchspace_list)
    #
    me = Ego(regclf=model, searchspace=searchspace, X=X, y=y, n_jobs=6)

    re = me.egosearch(flexibility=10)
