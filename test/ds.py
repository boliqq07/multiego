# -*- coding: utf-8 -*-

# @Time    : 2021/6/28 20:35
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

from mgetool.show import BasePlot
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

    #####model1#####
    ###

    #####model2#####
    parameters = {'C': [1, 10]}
    model2 = GridSearchCV(SVR(), parameters)
    ###

    X, y = load_boston(return_X_y=True)
    X = X[:, :5]  # (简化计算，示意)
    np.random.seed(0)
    y = np.concatenate((y.reshape(-1, 1), (y.reshape(-1, 1) * (1 + np.random.random()) + np.random.random())), axis=1)
    searchspace_list = [
        np.arange(0.5, 1, 0.1),
        np.array([0, 20, ]),
        np.arange(3.9, 4.1, 0.01),
        np.array([0, 1]),
        np.arange(0.5, 0.55, 0.02),
    ]

    X, y = shuffle(X, y, random_state=5)

    nor = Normalizer()
    # X = nor.fit_transform(X)

    searchspace = search_space(*searchspace_list)

    #
    # # gd = GridSearchCV(KNeighborsRegressor(),param_grid=[{"n_neighbors":[2,3,4,5,6,7],"leaf_size":[10,20,30]}],scoring="neg_mean_absolute_error")
    gd = GridSearchCV(Lasso(), param_grid=[{"alpha": [1, 0.1, 0.01, 0.001]}], scoring="neg_mean_absolute_error")
    # # gd = GridSearchCV(SVR(),param_grid= {'C': [1e7,1e8,1e9],"epsilon":[0.1,0.01,1,10],},
    # #                   scoring="neg_mean_absolute_error",n_jobs=10,verbose=2)
    #
    gd.fit(X, y[:, 0])
    ts = gd.predict(X)
    bp = BasePlot()
    plt = bp.scatter(y[:, 0], ts)
    plt.show()

    # lr = SVR(C=1e7)
    # lr.fit(X, y[:, 0])
    # score= lr.score(X,y[:,0])

    me = MultiplyEgo(regclf=[Lasso(alpha=0.01), Lasso(alpha=0.01)], searchspace = searchspace, X=X, y=y,
                     number=50,n_jobs=6)  # 没什么用，只是需要全searchspace，最后的表格能对齐

    re = me.rank(fraction=1500,flexibility=20)  # 3 . 用合并的均值方差算EI