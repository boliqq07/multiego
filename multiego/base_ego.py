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

from scipy import stats

import pandas as pd

class BaseEgo:
    """
    EGO (Efficient global optimization).

    References:
        Jones, D. R., Schonlau, M. & Welch, W. J.
        Efficient global optimization of expensive black-box functions. J.
        Global Optim. 13, 455–492 (1998)

    Examples:

        >>>me = BaseEgo()
        >>>result = me.rank(y=y, mean_std=mean_std)

    """

    @staticmethod
    def meanandstd(predict_dataj):
        """calculate meanandstd."""
        mean = np.mean(predict_dataj, axis=1)
        std = np.std(predict_dataj, axis=1)
        data_predict = np.column_stack((mean, std))
        print(data_predict.shape)
        return data_predict

    @staticmethod
    def CalculateEi(y, mean_std0):
        """calculate EI."""
        ego = (mean_std0[:, 0] - max(y)) / (mean_std0[:, 1])
        ei_ego = mean_std0[:, 1] * ego * stats.norm.cdf(ego) + mean_std0[:, 1] * stats.norm.pdf(ego)
        kg = (mean_std0[:, 0] - max(max(mean_std0[:, 0]), max(y))) / (mean_std0[:, 1])
        ei_kg = mean_std0[:, 1] * kg * stats.norm.cdf(kg) + mean_std0[:, 1] * stats.norm.pdf(kg)
        max_P = stats.norm.cdf(ego, loc=mean_std0[:, 0], scale=mean_std0[:, 1])
        ei = np.column_stack((mean_std0, ei_ego, ei_kg, max_P))
        print('Ego is done')
        return ei

    def egosearch(self, y, searchspace, mean_std, rankway="ego", return_type="pd", reverse=True):
        """
        Result is 2 dimensions array.
        1st column = sequence number,\n
        2nd part = your search space,\n
        3rd part = mean,std,ego,kg,maxp,sequentially.

        Parameters
        ----------
        y: np.ndarray of shape (n_sample_train, 1)
            train y
        mean_std: np.ndarray of shape (n_sample_pre, n_feature)
            mean_std of n times of prediction on search space.
            First column is mean and second is std.
        rankway : str
            ["ego","kg","maxp","No"]
            resort the result by rankway name.
        searchspace : np.ndarray of shape (n_sample_pre, n_feature)
            search space
            ["ego","kg","maxp","No"]
        return_type: str
            "pd" or "np"
        reverse:bool
            reverse.

        Returns
        ----------
        table:np.ndarray (2d), pd.Dateframe

        """
        if rankway not in ['ego', 'kg', 'maxp', 'no', 'No']:
            print('Don\'t kidding me,checking rankway=what?\a')
        else:
            result = self.CalculateEi(y, mean_std)
            bianhao = np.arange(0, len(result))
            result1 = np.column_stack((bianhao, searchspace, result))
            if rankway == "No" or "no":
                pass
            if rankway == "ego":
                if reverse:
                    egopaixu = np.argsort(result1[:, -3])
                else:
                    egopaixu = np.argsort(-result1[:, -3])
                result1 = result1[egopaixu]
            elif rankway == "kg":
                if reverse:
                    kgpaixu = np.argsort(result1[:, -2])
                else:
                    kgpaixu = np.argsort(-result1[:, -2])
                result1 = result1[kgpaixu]
            elif rankway == "maxp":
                if reverse:
                    max_paixu = np.argsort(result1[:, -1])
                else:
                    max_paixu = np.argsort(-result1[:, -1])
                result1 = result1[max_paixu]
            if return_type != "pd":
                return result1
            else:
                result1 = pd.DataFrame(result1)
                fea = ["feature%d" % i for i in range(searchspace.shape[1])]
                mean_stds = ["mean_std%d" % i for i in range(mean_std.shape[1])]
                name = ["number"] + fea + mean_stds + ['ego', 'kg', 'maxp']
                result1.columns = name
                return result1
