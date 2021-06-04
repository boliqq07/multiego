#!/usr/bin/python
# coding:utf-8
"""
This is one general method to calculate Efficient global optimization with multiple target ,
There are no restrictions on the type of X and model.

Notes
-----
    The mean and std should calculated by yourself.
"""
import warnings

import numpy as np
import pandas as pd
from mgetool.tool import parallelize

warnings.filterwarnings("ignore")


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


class BaseMultiplyEgo:
    """
    EGO (Efficient global optimization).
    References:
        Jones, D. R., Schonlau, M. & Welch, W. J. Efficient global optimization of expensive black-box functions. J.
        Global Optim. 13, 455–492 (1998)
    """

    def __init__(self, n_jobs=2):
        """

        Parameters
        ----------
        n_jobs:int
            parallelize number.
        """

        self.n_jobs = n_jobs
        self.rank = self.egosearch

    @staticmethod
    def meanandstd(predict_dataj):
        mean = np.mean(predict_dataj, axis=0)
        std = np.std(predict_dataj, axis=0)
        data_predict = np.column_stack((mean, std))
        # print(data_predict.shape)
        return data_predict

    def pareto_front_point(self, y, sign=None):
        m = y.shape[0]
        n = y.shape[1]
        if sign is None:
            sign = np.array([1] * n)
        y *= sign
        front_point_index = []
        for i in range(m):
            data_new = y[i, :].reshape(1, -1) - y
            data_max = np.max(data_new, axis=1)
            data_in = np.min(data_max)
            if data_in >= 0:
                front_point_index.append(i)
        front_point = y[front_point_index, :].T
        self.front_point = front_point
        self.front_point_index = front_point_index
        return front_point

    def CalculateL(self, meanandstd_all):
        front_y = self.front_point
        meanstd = np.array(meanandstd_all)
        meanstd = meanstd[:, :, 0].T
        alll = []
        for front_y_i in front_y.T:
            l_i = meanstd - front_y_i
            alll.append(l_i)
        dmin = np.array(alll)

        dmin2 = np.min(np.abs(dmin), axis=0)

        dmin3 = np.min(dmin2, axis=1)

        #        dmin3 = np.sqrt(np.sum(dmin2**2,axis=1))

        dmin3[np.where(dmin3 < 0)[0]] = 0

        self.L = dmin3
        return dmin3

    def CalculateEi(self, y, meanandstd_all=None, predict_y_all=None, sign=None):
        """EI value"""
        self.pareto_front_point(y, sign)
        self.CalculatePi(predict_y_all)
        self.CalculateL(meanandstd_all)
        Ei = self.L * self.Pi
        self.Ei = Ei
        return Ei

    def CalculatePi(self, predict_y_all):
        """PI value"""

        njobs = self.n_jobs
        front_y = self.front_point

        def tile_func(i, front_y0):
            tile = 0
            for front_y_i in front_y0.T:
                big = i - front_y_i
                big_bool = np.max(big, axis=1) < 0
                tile |= big_bool
            return tile

        tile_all = parallelize(n_jobs=njobs, func=tile_func, iterable=predict_y_all, front_y0=front_y)
        pi = np.sum(1 - np.array(tile_all), axis=1) / predict_y_all.shape[1]

        self.Pi = pi

        return pi

    def egosearch(self, y, searchspace, meanandstd_all, predict_y_all, return_type="pd", fraction=1000, sign=None):
        """
        Result is 2 dimensions array.
        1st column = sequence number,\n
        2nd part = your search space,\n
        3rd part = Pi,L,Ei,sequentially.

        Parameters
        ----------
        y: np.ndarray of shape (n_sample_train, n_model)
            true train y.
        searchspace : np.ndarray of shape (n_sample_pre, n_feature)
            search space
        fraction: int
            choice top n_sample/fraction
        return_type:str
            numpy.ndarray or pandas.DataFrame
        meanandstd_all: list of np.ndarray
            n_model meanandstd, Each meanandstd is np.ndarray of shape (n_sample_pre,n_model)
        predict_y_all: np.ndarray of shape (n_sample_pre,n_times,n_model)
            ys.
        sign:np.ndarray of shape (n_model,)
            Each element must be -1 or 1.
            sign to define the max problem or min problem.

        Returns
        ----------
        table:np.ndarray (2d), pd.Dateframe

        """
        bianhao = np.arange(0, searchspace.shape[0])

        self.CalculateEi(y, meanandstd_all, predict_y_all, sign=sign)

        result1 = np.column_stack((bianhao, searchspace, *meanandstd_all, self.Pi, self.L, self.Ei))

        max_paixu = np.argsort(-result1[:, -1])
        if max_paixu.size >= fraction:
            select_number = max_paixu[:int(max_paixu.size / fraction)]
        else:
            print("grid is smaller than fraction, the ``fraction`` is ignored.")
            select_number = max_paixu

        result1 = result1[select_number]

        if return_type == "pd":
            result1 = pd.DataFrame(result1)
            fea = ["feature%d" % i for i in range(searchspace.shape[1])]
            meanstds = ["meanstd%d" % i for i in range(sum([i.shape[1] for i in meanandstd_all]))]
            name = ["number"] + fea + meanstds + ["Pi", "L", "Ei"]
            result1.columns = name
        self.result = result1
        return self.result
