# -*- coding: utf-8 -*-

# @Time    : 2020/11/25 14:50
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from tqdm import tqdm

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    import numpy as np
    from multiego.ego import search_space, Ego
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    #####model1#####
    # model = SVR()
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

    spilt = 5  # 1. 划分分空间,在内存没有超的情况下，数字越小越好（>2）
    sps = np.array_split(searchspace, spilt)

    ms = []

    for i in tqdm(range(spilt), desc="空间"):  # 2 .每一部分空间算出均值方差，然后合并

        me = Ego(regclf=model, searchspace=sps[i], X=X, y=y, number=50, n_jobs=6)

        pre = me.fit()
        msi = me.meanandstd(pre)

        ms.append(msi)

    ms = np.concatenate(ms, axis=0)

    me = Ego(regclf=model, searchspace=searchspace, X=X, y=y)  # 没什么用，只是需要全searchspace，最后的表格能对齐
    re = me.egosearch(meanstd=ms)  # 3 . 用合并的均值方差算EI
