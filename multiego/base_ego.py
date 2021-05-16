# -*- coding: utf-8 -*-

# @Time    : 2020/9/8 23:51
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-

"""This is one general method to calculate Efficient global optimization,
There are no restrictions on the type of X and model.

Notes
-----
    The mean and std should calculated by yourself.
"""

import numpy as np
import sklearn.utils
import sklearn
from mgetool.tool import parallelize
from scipy import stats
from sklearn.utils import check_array


class BaseEgo:
    """
    EGO (Efficient global optimization).

    References:
        Jones, D. R., Schonlau, M. & Welch, W. J. Efficient global optimization of expensive black-box functions. J.
        Global Optim. 13, 455–492 (1998)

    Examples:

        me = Ego()

        result = me.Rank(meanstd=meanstd)

    """

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

    def egosearch(self, meanstd, rankway="ego", ):
        """
        Result is 2 dimentions array
        1st column = sequence number,2nd part = your searchspace,3rd part = mean,std,ego,kg,maxp,sequentially.

        Parameters
        ----------
        meanstd:np.ndarray,None

        rankway : str
            ["ego","kg","maxp","No"]
        """
        y = self.y
        searchspace0 = self.searchspace
        if rankway not in ['ego', 'kg', 'maxp', 'no', 'No']:
            print('Don\'t kidding me,checking rankway=what?\a')
        else:
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

    def Rank(self, meanstd, rankway="ego"):
        """
        The same as egosearch method.
        Result is 2 dimentions array.
        1st column = sequence number,2nd part = your searchspace,3rd part = mean,std,ego,kg,maxp,sequentially.

        Parameters
        ----------
        meanstd:np.ndarray

        rankway : str
            ["ego","kg","maxp","No"]
        """
        return self.egosearch(meanstd=meanstd, rankway=rankway)