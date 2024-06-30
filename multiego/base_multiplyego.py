#!/usr/bin/python
# coding:utf-8
"""
This is one general method to calculate Efficient global optimization with multiple target ,
There are no restrictions on the type of X and model.

"""
import warnings
from itertools import chain, combinations

import numpy as np
import pandas as pd
from mgetool.tool import batch_parallelize

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

    Keep the all y in same level (MinMaxScaling) !!!

    Attributes
    ------------
    front_point: np.ndarray
        front_point.
    front_point_index: np.ndarray
        front_point index in x.
    Ei: np.ndarray
        Ei
    Pi: np.ndarray
        Pi
    L: np.ndarray
        Pi
    """

    def __init__(self, n_jobs=2, up=True, strategy="min"):
        """

        Parameters
        ----------
        n_jobs:int
            parallelize number.

        up:bool
            The pareto front face is going to be up or down for pareto front point.

        """

        self.n_jobs = n_jobs
        self.strategy = strategy
        self.rank = self.egosearch
        self.up = up

    @staticmethod
    def meanandstd(predict_y):
        mean = np.mean(predict_y, axis=1)
        std = np.std(predict_y, axis=1)
        data_predict = np.column_stack((mean, std))
        # print(data_predict.shape)
        return data_predict

    def get_meanandstd_all(self, predict_ys):
        meanandstd = []
        for i in range(predict_ys.shape[-1]):
            meanandstd_i = self.meanandstd(predict_ys[:, :, i])
            meanandstd.append(meanandstd_i)
        return meanandstd

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
        if self.strategy == "min":
            dmin3 = np.min(dmin2, axis=1)
        elif self.strategy == "mean":
            dmin3 = np.sqrt(np.sum(dmin2**2, axis=1)) # another method.
        else:
            raise NotImplemented("strategy just accept 'min','mean'.")

        dmin3[np.where(dmin3 < 0)[0]] = 0

        self.L = dmin3
        return dmin3

    def CalculateEi(self, y, meanandstd_all=None, predict_y_all=None, sign=None, flexibility=None):
        """EI value"""
        if flexibility is None:
            flexibility = np.array(np.zeros((y.shape[1], 1)))
        else:
            warnings.warn(
                "``Flexibility`` means reduction of y boundary, Please use it if you know what you are doing.")
            flexibility = np.array(flexibility).reshape(-1, 1)

        self.pareto_front_point(y, sign)
        self.CalculatePi(predict_y_all, flexibility=flexibility)
        self.CalculateL(meanandstd_all)
        Ei = self.L * self.Pi
        self.Ei = Ei
        return Ei

    def scrap(self, fp):
        """This function is add the up step to paretofront face.
        For the front, the steps can go down or up, we add up to get tighter requirements.
        """
        # max problem
        if fp.shape[0] == 1:
            raise NotImplemented("For single problem, we dont implement up-pareto front face methods.")
        if fp.shape[0] == 2:
            fp_index = np.argsort(fp[0, :])
            fp = fp[:, fp_index]
            new_fp = np.vstack((fp[0, :][1:], fp[1, :][:-1]))

        else:
            # nps= []
            # for coi in combinations(range(fp.shape[1]),fp.shape[0]):
            #     new_point = fp[:,coi]
            #     nps.append(np.max(new_point,axis=1))
            # nps=np.array(nps)
            # new_fp = self.pareto_front_point(nps)
            raise NotImplemented("For multi problem more than 2, we dont implemente up-pareto front face methods. please set ``up = False``")

        return new_fp

    def CalculatePi(self, predict_y_all, flexibility):
        """PI value"""

        njobs = self.n_jobs

        if self.up:
            front_y = self.scrap(self.front_point)
        else:
            front_y = self.front_point

        front_y -= flexibility

        def tile_func(i):
            tile = 0
            for front_y_i in front_y.T:
                big = i - front_y_i
                big_bool = np.max(big, axis=1) < 0
                tile |= big_bool
            return tile

        tile_all = batch_parallelize(n_jobs=njobs, func=tile_func, iterable=predict_y_all, batch_size=100)
        pi = 1-np.sum(np.array(tile_all), axis=1) / predict_y_all.shape[1]

        self.Pi = pi

        return pi

    def egosearch(self, y, predict_y_all, meanandstd_all=None, searchspace=None, return_type="pd", flexibility=None,
                  fraction=1000, sign=None):
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
            search space, for base_multiego, searchspace is not used and just as one placeholder.
        fraction: int
            choice top n_sample/fraction
        return_type:str
            numpy.ndarray or pandas.DataFrame
        meanandstd_all: list of np.ndarray
            Not required force.
            n_model meanandstd, Each meanandstd is np.ndarray of shape (n_sample_pre,2)
        predict_y_all: np.ndarray of shape (n_sample_pre,n_times,n_model)
            ys.
        sign:np.ndarray of shape (n_model,)
            Each element must be -1 or 1.
            sign to define the max problem or min problem.

        flexibility: List[float]
            Flexibility to calculate PI, the bigger flexibility, the more search space Pi >0. for each y.

        Returns
        ----------
        table:np.ndarray (2d), pd.Dateframe

        """
        if flexibility is None:
            flexibility = np.array(np.zeros((y.shape[1], 1)))
        else:
            warnings.warn(
                "``Flexibility`` means reduction of y boundary, Please use it if you know what you are doing.")
            flexibility = np.array(flexibility).reshape(-1, 1)

        bianhao = np.arange(0, predict_y_all.shape[0]).reshape(-1, 1)

        if searchspace is None:
            searchspace = np.arange(0, predict_y_all.shape[0]).reshape(-1, 1)

        if meanandstd_all is None:
            meanandstd_all = self.get_meanandstd_all(predict_y_all)

        self.CalculateEi(y, meanandstd_all, predict_y_all, sign=sign, flexibility=flexibility)

        assert not np.all(self.Ei <= 1e-10), "All the Ei (and Pi) score is 0, This is invalid calculation. " \
                                             "Please try these methods:\n" \
                                             "1. Improve your model precision, especially near the expected scope for y. " \
                                             "For example, for max proplem, the point near the maximum y should be accurate by model.\n" \
                                             "2. Make sure your search space near the training space.\n" \
                                             "3. If the above methods are still unable to solve, add flexibility to find point by reduction of y boundary. ()\n"

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
