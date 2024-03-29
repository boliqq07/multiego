# -*- coding: utf-8 -*-

# @Time    : 2020/11/25 14:50
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from multiego.base_multiplyego import BaseMultiplyEgo

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    import numpy as np
    from multiego.multiplyego import MultiplyEgo, search_space
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    #####model1#####
    model1 = SVR()
    ###

    #####model2#####
    parameters = {'C': [1, 10]}
    model2 = GridSearchCV(SVR(), parameters)
    ###
    # 模型应当提前调整好参数

    X, y = load_boston(return_X_y=True)
    X = X[:, :5]  # (简化计算，示意)
    y = np.concatenate((y.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    searchspace_list = [
        np.arange(0.01, 1, 0.1),
        np.array([0, 20, 30, 50, 70, 90]),
        np.arange(1, 10, 1),
        np.array([0, 1]),
        np.arange(0.4, 0.6, 0.02),
    ]

    searchspace = search_space(*searchspace_list)

    spilt = 3  # 1. 在内存没有超的情况下，数字越小越好（>2）,这里划分3部分空间,
    sps = np.array_split(searchspace, spilt)

    ms = []
    pre = []
    for i in range(spilt):  # 2 .每一部分空间算出预测值,均值方差，然后合并

        me = MultiplyEgo(regclf=[model1, model2], searchspace=sps[i], X=X, y=y, number=50, n_jobs=5)
        me.fit()
        # ms.append(me.mean_std_all)
        pre.append(me.predict_y_all)

    pre = np.concatenate(pre, axis=0)
    me = BaseMultiplyEgo(strategy="mean")
    re = me.rank(y=y,predict_y_all=pre,fraction=1000,flexibility=[20,20] )  # 3 . 用合并的均值方差算EI
