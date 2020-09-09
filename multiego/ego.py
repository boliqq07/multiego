# -*- coding: utf-8 -*-

# @Time    : 2020/9/8 23:51
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-

"""
Created on Sun Jan 28 15:24:10 2018

@author: ww
"""

import numpy as np
import sklearn.utils
import sklearn
from mgetool.tool import parallelize
from scipy import stats
from sklearn.utils import check_array

print('\t\tFor grid building\nsample:\nspace=searchspace(li1,li2,li3)\nNote:parameters no more than 5 ')
print(
    '\n\t\tFor ego,kg,maxp\nSample:\nresults=egosearch(X=?,searchspace=?,number=?,regclf=RandomForestRegressor(),'
    'rankway="ego"or"kg"or"maxp"or"No",meanstd=False)')
print(
    'return:\nresult is 2 dimentions array\n1st column = sequence number,\n2nd part = your searchspace,\n3rd part = '
    'mean,std,ego,kg,maxp,sequentially')


def search_space(*arg):
    """

    Parameters
    ----------
    arg: list of np.ndarray
        Examples:
            arg = [
            np.arange(0.1,0.35,0.1),
            np.arange(0.1, 2.1, 0.5),
            np.arange(0,1.3,0.3),
            np.array([0.5,1,1.2,1.3]),]

    Returns
    -------
    np.ndarray

    """
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


class Ego:
    """
    EGO (Efficient global optimization).

    References:
        Jones, D. R., Schonlau, M. & Welch, W. J. Efficient global optimization of expensive black-box functions. J.
        Global Optim. 13, 455–492 (1998)

    Examples:

        searchspace_list = [
            np.arange(0.1,0.35,0.1),
            np.arange(0.1, 1.3, 0.3),
            np.arange(0.1, 2.1, 0.5),
            np.arange(0,1.3,0.3),
            np.arange(0,7.5,1.5),
            np.arange(0,7.5,1.5),
            np.arange(800, 1300, 50),
            np.arange(200, 600, 40),
            np.array([20, 80, 138, 250]),]

        searchspace = search_space(*searchspace_list)

        me = Ego(searchspace, X, y, 500, [svr, gpr], n_jobs=8)

        me.fit()

        Ei = me.CalculateEi()

        result = me.Rank()

    """

    def __init__(self, searchspace, X, y, number, regclf, n_jobs=2):
        """
        Parameters
        ----------
        searchspace: np.ndarray
            Custom or generate by .search_space() function.
        X: np.ndarray
            X data (2D).
        y: np.ndarray
            y data (1D).
        number: int>100
            Repeat number,default is 1000.
        regclf: callable
            sklearn method, with "fit" and "predict".
        n_jobs: int
            Parallelize number.
        """

        self.n_jobs = n_jobs
        check_array(X, ensure_2d=True, force_all_finite=True)
        check_array(y, ensure_2d=True, force_all_finite=True)
        check_array(searchspace, ensure_2d=True, force_all_finite=True)
        assert X.shape[1] == searchspace[1]
        self.searchspace = searchspace
        self.X = X
        self.y = y
        self.regclf = regclf

        self.meanandstd_all = []
        self.predict_y_all = []
        self.number = number

    def fit(self):
        x = self.X
        y = self.y
        njobs = self.n_jobs
        searchspace0 = self.searchspace
        regclf0 = self.regclf

        def fit_parllize(random_state):
            data_train, y_train = sklearn.utils.resample(x, y, n_samples=None, replace=True,
                                                         random_state=random_state)
            regclf0.fit(data_train, y_train)
            predict_data = regclf0.predict(searchspace0)
            predict_data.ravel()

        predict_dataj = parallelize(n_jobs=njobs, func=fit_parllize, iterable=range(self.number))

        return np.array(predict_dataj)

    @staticmethod
    def meanandstd(predict_dataj):
        """calculate meanandstd"""
        mean = np.mean(predict_dataj, axis=1)
        std = np.std(predict_dataj, axis=1)
        data_predict = np.column_stack((mean, std))
        print(data_predict.shape)
        return data_predict

    @staticmethod
    def CalculateEi(y, meanstd0):
        """calculate EI"""
        ego = (meanstd0[:, 0] - max(y)) / (meanstd0[:, 1])
        ei_ego = meanstd0[:, 1] * ego * stats.norm.cdf(ego) + meanstd0[:, 1] * stats.norm.pdf(ego)
        kg = (meanstd0[:, 0] - max(max(meanstd0[:, 0]), max(y))) / (meanstd0[:, 1])
        ei_kg = meanstd0[:, 1] * kg * stats.norm.cdf(kg) + meanstd0[:, 1] * stats.norm.pdf(kg)
        max_P = stats.norm.cdf(ego, loc=meanstd0[:, 0], scale=meanstd0[:, 1])
        ei = np.column_stack((meanstd0, ei_ego, ei_kg, max_P))
        print('ego is done')
        return ei

    def egosearch(self, rankway="ego", meanstd=False):

        y = self.y
        searchspace0 = self.searchspace
        if rankway not in ['ego', 'kg', 'maxp', 'no', 'No']:
            print('Don\'t kidding me,checking rankway=what?\a')
        else:
            if not meanstd:
                predict_data = self.fit()
                meanstd = self.meanandstd(predict_data)
            else:
                pass

            result = self.CalculateEi(y, meanstd)
            bianhao = np.arange(0, len(result))
            result1 = np.column_stack((bianhao, searchspace0, result))
            if rankway == "No" or "no":
                pass
            if rankway == "ego":
                egopaixu = np.argsort(result1[:, -3])
                result1 = result1[egopaixu]
            elif rankway == "kg":
                kgpaixu = np.argsort(result1[:, -2])
                result1 = result1[kgpaixu]
            elif rankway == "maxp":
                max_paixu = np.argsort(result1[:, -1])
                result1 = result1[max_paixu]
            return result1

    def Rank(self, rankway="ego", meanstd=False):
        """rank result,by 'rankway'"""
        return self.egosearch(rankway, meanstd=meanstd)
