import gc
import warnings
from collections.abc import Iterable

import numpy as np
import sklearn.utils
from mgetool.tool import parallelize
from sklearn.utils import check_array

from multiego.base_multiplyego import BaseMultiplyEgo

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


class MultiplyEgo(BaseMultiplyEgo):
    """
    Multi-target EGO (Efficient global optimization).

    Keep the all y in same level (MinMaxScaling) !!!

    References
    -----------
        Jones, D. R., Schonlau, M. & Welch, W. J. Efficient global optimization of expensive black-box functions. J.
        Global Optim. 13, 455–492 (1998)
    Examples
    -----------

        >>> searchspace_list = [
        ...    np.arange(0.1,0.35,0.1),
        ...    np.arange(0.1, 1.3, 0.3),
        ...    np.arange(0.1, 2.1, 0.5),
        ...    np.arange(0,1.3,0.3),
        ...    np.arange(0,7.5,1.5),
        ...    np.arange(0,7.5,1.5),
        ...    np.arange(800, 1300, 50),
        ...    np.arange(200, 600, 40),
        ...    np.array([20, 80, 138, 250]),]

        >>> searchspace = search_space(*searchspace_list)

        >>> me = MutilplyEgo([svr, gpr], searchspace, X, y, 500, n_jobs=8)

        >>> me.fit()

        >>> select_number = me.rank()
        >>> result = me.result

    Attributes
    ------------
    mean_std_all: np.ndarray
        mean and std.
    predict_y_all: np.ndarray
        predict_y.
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

    def __init__(self, regclf, searchspace, X, y, number=1000, feature_slice=None, n_jobs=2, up=True, strategy= "min"):
        """

        Parameters
        ----------
        searchspace:np.ndarray of shape (n_sample_pre,n_feature)
            custom or generate by .search_space() function.
        X: np.ndarray of shape (n_sample_train, n_feature)
            X data (2D).
        y: np.ndarray of shape (n_sample_train, n_model)
            y data (2D).
        number:int>100
            repeat number,default is 1000.
        regclf: List of callable
            sklearn methods, with "fit" and "predict".
            The same number with the number of target y.
        feature_slice: None,list of tuple.
            The same number with the number of target y,
            Aand each tuple contains the features index for each regclf.
        n_jobs:int
            parallelize number.
        """
        super(MultiplyEgo, self).__init__(n_jobs=n_jobs,up=up,strategy=strategy)
        self.rank = self.egosearch

        check_array(X, ensure_2d=True, force_all_finite=True)
        check_array(y, ensure_2d=False, force_all_finite=True)
        check_array(searchspace, ensure_2d=True, force_all_finite=True)
        assert X.shape[1] == searchspace.shape[1]
        self.searchspace = searchspace
        self.X = X
        self.y = y
        self.sign = []

        assert isinstance(regclf, Iterable)
        assert len(list(regclf)) >= 2
        self.regclf = list(regclf)
        self.dim = len(list(regclf))

        if feature_slice is None:
            feature_slice = tuple([tuple(range(X.shape[1]))] * self.dim)
        assert isinstance(feature_slice, (tuple, list))
        assert isinstance(feature_slice[1], (tuple, list))
        assert self.dim == len(feature_slice) == self.y.shape[1]
        self.feature_slice = feature_slice

        self.Ei = np.zeros_like(searchspace[:, 1])
        self.Pi = np.zeros_like(searchspace[:, 1])
        self.L = np.zeros_like(searchspace[:, 1])
        self.front_point = np.zeros_like(self.y[:, 1])
        self.number = number
        self.center = np.zeros_like(searchspace[:, 1])
        self.rank = self.egosearch


    def _fit(self, x, y, searchspace0, regclf0):
        def fit_parllize(random_state):
            data_train, y_train = sklearn.utils.resample(x, y, n_samples=None, replace=True,
                                                         random_state=random_state)
            regclf0.fit(data_train, y_train)
            predict_data = regclf0.predict(searchspace0)
            predict_data.ravel()
            return predict_data

        predict_dataj = parallelize(n_jobs=self.n_jobs, func=fit_parllize, iterable=range(self.number))

        return np.array(predict_dataj).T

    def fit(self, regclf_number=None):
        """

        Parameters
        ----------
        regclf_number: int,list

            the running sklearn methods, default is all of regclf.

        Returns
        -------
        meanandstd: np.ndarray
             mean and std predicted y value with number(1000).
        """

        if regclf_number is None:
            contain = list(range(self.dim))
        elif isinstance(regclf_number, int):
            contain = [regclf_number]
        elif isinstance(regclf_number, (list, tuple)):
            contain = regclf_number
        else:
            raise TypeError()
        meanandstd = []
        predict_y_all = []
        for i, feature_slicei, yi, regclfi in zip(range(self.dim), self.feature_slice, self.y.T, self.regclf):
            if i in contain:
                predict_y = np.array(self._fit(self.X[:, feature_slicei], yi, self.searchspace[:, feature_slicei],
                                               regclfi))
                predict_y_all.append(predict_y)

                meanandstd_i = self.meanandstd(predict_y)

                del predict_y
                gc.collect()

                meanandstd.append(meanandstd_i)
            else:
                pass

        self.mean_std_all = meanandstd
        self.predict_y_all = np.array(predict_y_all).transpose((1,2,0))
        return self.mean_std_all

    def egosearch(self, searchspace=None, fraction=1000, return_type="pd", predict_y_all=None,
                  meanandstd_all=None, flexibility=None,
                  y=None, sign=None):
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
        y = self.y if y is None else y
        if searchspace is not None:
            self.searchspace = searchspace
        if predict_y_all is not None:
            self.predict_y_all = predict_y_all
        if meanandstd_all is not None:
            self.mean_std_all = meanandstd_all

        if hasattr(self, 'mean_std_all') and hasattr(self, 'predict_y_all'):
            pass
        elif hasattr(self, 'predict_y_all'):
            self.mean_std_all = self.get_meanandstd_all(self.predict_y_all)
        else:
            self.fit()

        return super().egosearch(y, self.predict_y_all, searchspace= self.searchspace,meanandstd_all= self.mean_std_all,
                                 return_type="pd", fraction=fraction, flexibility=flexibility, sign=sign)

# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #!/usr/bin/python
# # # coding:utf-8
# #
# #
# # """
# # sample
# # """
# # import gc
# # import warnings
# # from collections.abc import Iterable
# #
# # import numpy as np
# # import pandas as pd
# # import sklearn.utils
# # from mgetool.tool import parallelize
# # from sklearn.utils import check_array
# #
# # warnings.filterwarnings("ignore")
# #
# #
# # def search_space(*arg):
# #     """
# #
# #     Parameters
# #     ----------
# #     arg: list of np.ndarray
# #         Examples:
# #             arg = [
# #             np.arange(0.1,0.35,0.1),
# #             np.arange(0.1, 2.1, 0.5),
# #             np.arange(0,1.3,0.3),
# #             np.array([0.5,1,1.2,1.3]),]
# #
# #     Returns
# #     -------
# #     np.ndarray
# #
# #     """
# #     meshes = np.meshgrid(*arg)
# #     meshes = [_.ravel() for _ in meshes]
# #     meshes = np.array(meshes).T
# #     return meshes
# #
# #
# # class MultiplyEgo:
# #     """
# #     EGO (Efficient global optimization).
# #     References:
# #         Jones, D. R., Schonlau, M. & Welch, W. J. Efficient global optimization of expensive black-box functions. J.
# #         Global Optim. 13, 455–492 (1998)
# #     Examples:
# #
# #         searchspace_list = [
# #             np.arange(0.1,0.35,0.1),
# #             np.arange(0.1, 1.3, 0.3),
# #             np.arange(0.1, 2.1, 0.5),
# #             np.arange(0,1.3,0.3),
# #             np.arange(0,7.5,1.5),
# #             np.arange(0,7.5,1.5),
# #             np.arange(800, 1300, 50),
# #             np.arange(200, 600, 40),
# #             np.array([20, 80, 138, 250]),]
# #
# #         searchspace = search_space(*searchspace_list)
# #
# #         me = MutilplyEgo([svr, gpr], searchspace, X, y, 500, n_jobs=8)
# #
# #         me.fit()
# #
# #         select_number = me.Rank()
# #
# #         result = self.result
# #
# #     """
# #
# #     def __init__(self, regclf, searchspace, X, y, number=1000, feature_slice=None, n_jobs=2):
# #         """
# #
# #         Parameters
# #         ----------
# #         searchspace:np.ndarray
# #             custom or generate by .search_space() function.
# #         X: np.ndarray
# #             X data (2D).
# #         y: np.ndarray
# #             y data (2D).
# #         number:int>100
# #             repeat number,default is 1000.
# #         regclf: list of callable
# #             sklearn methods, with "fit" and "predict".
# #             The same number with the number of target y.
# #         feature_slice: None,list of tuple.
# #             The same number with the number of target y,
# #             and the tuple contains the features for regclf.
# #         n_jobs:int
# #             parallelize number.
# #         """
# #
# #         self.n_jobs = n_jobs
# #         check_array(X, ensure_2d=True, force_all_finite=True)
# #         check_array(y, ensure_2d=False, force_all_finite=True)
# #         check_array(searchspace, ensure_2d=True, force_all_finite=True)
# #         assert X.shape[1] == searchspace.shape[1]
# #         self.searchspace = searchspace
# #         self.X = X
# #         self.y = y
# #         self.sign = []
# #
# #         assert isinstance(regclf, Iterable)
# #         assert len(list(regclf)) >= 2
# #         self.regclf = list(regclf)
# #         self.dim = len(list(regclf))
# #
# #         if feature_slice is None:
# #             feature_slice = tuple([tuple(range(X.shape[1]))] * self.dim)
# #         assert isinstance(feature_slice, (tuple, list))
# #         assert isinstance(feature_slice[1], (tuple, list))
# #         assert self.dim == len(feature_slice) == self.y.shape[1]
# #         self.feature_slice = feature_slice
# #
# #         self.Ei = np.zeros_like(searchspace[:, 1])
# #         self.Pi = np.zeros_like(searchspace[:, 1])
# #         self.L = np.zeros_like(searchspace[:, 1])
# #         self.front_point = np.zeros_like(self.y[:, 1])
# #         self.number = number
# #         self.center = np.zeros_like(searchspace[:, 1])
# #
# #     def _fit(self, x, y, searchspace0, regclf0):
# #         def fit_parllize(random_state):
# #             data_train, y_train = sklearn.utils.resample(x, y, n_samples=None, replace=True,
# #                                                          random_state=random_state)
# #             regclf0.fit(data_train, y_train)
# #             predict_data = regclf0.predict(searchspace0)
# #             predict_data.ravel()
# #             return predict_data
# #
# #         njobs = self.n_jobs
# #         number = self.number
# #
# #         predict_y = parallelize(n_jobs=njobs, func=fit_parllize, iterable=range(number))
# #
# #         return np.array(predict_y)
# #
# #     @staticmethod
# #     def _mean_and_std(predict_y):
# #         mean = np.mean(predict_y, axis=0)
# #         std = np.std(predict_y, axis=0)
# #         data_predict = np.column_stack((mean, std))
# #         # print(data_predict.shape)
# #         return data_predict
# #
# #     def fit(self, regclf_number=None):
# #         """
# #
# #         Parameters
# #         ----------
# #         regclf_number: int,list
# #
# #             the running sklearn methods, default is all of regclf.
# #
# #         Returns
# #         -------
# #         meanandstd: np.ndarray
# #              mean and std predicted y value with number(1000).
# #         """
# #
# #         if regclf_number is None:
# #             contain = list(range(self.dim))
# #         elif isinstance(regclf_number, int):
# #             contain = [regclf_number]
# #         elif isinstance(regclf_number, (list, tuple)):
# #             contain = regclf_number
# #         else:
# #             raise TypeError()
# #         meanandstd = []
# #         predict_y_all = []
# #         for i, feature_slicei, yi, regclfi in zip(range(self.dim), self.feature_slice, self.y.T, self.regclf):
# #             if i in contain:
# #                 predict_y = np.array(self._fit(self.X[:, feature_slicei], yi, self.searchspace[:, feature_slicei],
# #                                                regclfi))
# #                 predict_y_all.append(predict_y)
# #
# #                 meanandstd_i = self._mean_and_std(predict_y)
# #
# #                 del predict_y
# #                 gc.collect()
# #
# #                 meanandstd.append(meanandstd_i)
# #             else:
# #                 pass
# #
# #         self.meanandstd_all = meanandstd
# #         self.predict_y_all = np.array(predict_y_all).T
# #         return self.meanandstd_all
# #
# #     def pareto_front_point(self):
# #         sign = self.sign
# #         y = self.y
# #         m = y.shape[0]
# #         n = y.shape[1]
# #         if not sign:
# #             sign = np.array([1] * n)
# #         y *= sign
# #         front_point = []
# #         for i in range(m):
# #             data_new = y[i, :].reshape(1, -1) - y
# #             data_max = np.max(data_new, axis=1)
# #             data_in = np.min(data_max)
# #             if data_in >= 0:
# #                 front_point.append(i)
# #         self.front_point_index = front_point
# #         self.front_point = self.y[front_point, :].T
# #         return front_point
# #
# #     # def CalculateEI2(spath):
# #     #     predict_y_all = spath.predict_y_all
# #     #     front_y = spath.pareto_front_point()
# #     #     front_y = spath.y[front_y, :].T
# #     #     meanstd = spath.meanandstd_all
# #     #     pi_all = 1
# #     #     center = []
# #     #     for y_i, meanstd_i in zip(front_y, meanstd):
# #     #         std_ = meanstd_i[:, 1]
# #     #         mean_ = meanstd_i[:, 0]
# #     #         y_max = max(y_i)
# #     #         upper_bound = (mean_ - y_max) / std_
# #     #         numerator = [integrate.quad(lambda x: 1 / np.sqrt(2 * np.pi) * x * np.exp(-0.5 * x ** 2),
# #     #                                     a=upper_bound_i, b=np.inf,
# #     #                                     full_output=0)[0] for upper_bound_i in upper_bound]
# #     #         pi_i_denominator = stats.norm.cdf(upper_bound)
# #     #         center_i = numerator / pi_i_denominator
# #     #         pi_all *= pi_i_denominator
# #     #         center.append(center_i)
# #     #     center = np.array(center)
# #     #     spath.Pi = pi_all
# #     #     spath.center = center
# #     #     L = np.array([[spatial.distance.euclidean(y_j, j) for j in center.T] for y_j in front_y.T])
# #     #     L_min = np.min(L, axis=0)
# #     #     spath.L = L_min
# #
# #     def CalculateL(self, meanandstd_all=None):
# #         front_y = self.pareto_front_point()
# #         front_y = self.y[front_y, :].T
# #         if meanandstd_all is not None:
# #             self.meanandstd_all = meanandstd_all
# #         meanstd = np.array(self.meanandstd_all)
# #         meanstd = meanstd[:, :, 0].T
# #         alll = []
# #         for front_y_i in front_y.T:
# #             l_i = meanstd - front_y_i
# #             alll.append(l_i)
# #         dmin = np.array(alll)
# #
# #         dmin2 = np.min(np.abs(dmin), axis=0)
# #
# #         dmin3 = np.min(dmin2, axis=1)
# #
# #         #        dmin3 = np.sqrt(np.sum(dmin2**2,axis=1))
# #
# #         dmin3[np.where(dmin3 < 0)[0]] = 0
# #
# #         self.L = dmin3
# #         return dmin3
# #
# #     def CalculateEi(self, meanandstd_all=None, predict_y_all=None):
# #         """EI value"""
# #         self.CalculatePi(predict_y_all)
# #         self.CalculateL(meanandstd_all)
# #         Ei = self.L * self.Pi
# #         self.Ei = Ei
# #         return Ei
# #
# #     def CalculatePi(self, predict_y_all=None):
# #         """PI value"""
# #         if predict_y_all is not None:
# #             self.predict_y_all = predict_y_all
# #
# #         njobs = self.n_jobs
# #         front_y = self.pareto_front_point()
# #         front_y = self.y[front_y, :].T
# #
# #         def tile_func(i, front_y0):
# #             tile = 0
# #             for front_y_i in front_y0.T:
# #                 big = i - front_y_i
# #                 big_bool = np.max(big, axis=1) < 0
# #                 tile |= big_bool
# #             return tile
# #
# #         tile_all = parallelize(n_jobs=njobs, func=tile_func, iterable=self.predict_y_all, front_y0=front_y)
# #         pi = np.sum(1 - np.array(tile_all), axis=1) / self.number
# #         self.Pi = pi
# #
# #         return pi
# #
# #     def Rank(self, fraction=1000, return_type="pd", predict_y_all=None, meanandstd_all=None):
# #         """
# #
# #         Parameters
# #         ----------
# #         fraction: int
# #             choice top n_sample/fraction
# #         return_type:str
# #
# #             numpy.ndarray or pandas.DataFrame
# #
# #         meanandstd_all:np.ndarray, None
# #
# #         predict_y_all：np.ndarray
# #
# #         Returns
# #         -------
# #         rank
# #         """
# #         if predict_y_all is not None:
# #             self.predict_y_all = predict_y_all
# #         if meanandstd_all is not None:
# #             self.meanandstd_all = meanandstd_all
# #         bianhao = np.arange(0, self.searchspace.shape[0])
# #         if hasattr(self, 'meanandstd_all') and hasattr(self, 'predict_y_all'):
# #             self.CalculateEi(self.meanandstd_all, self.predict_y_all)
# #         else:
# #             self.fit()
# #             self.CalculateEi(self.meanandstd_all, self.predict_y_all)
# #
# #         del self.predict_y_all
# #         gc.collect()
# #
# #         result1 = np.column_stack((bianhao, self.searchspace, *self.meanandstd_all, self.Pi, self.L, self.Ei))
# #         max_paixu = np.argsort(-result1[:, -1])
# #         select_number = max_paixu[:int(max_paixu.size / fraction)]
# #         result1 = result1[select_number]
# #
# #         if return_type == "pd":
# #             result1 = pd.DataFrame(result1)
# #             fea = ["feature%d" % i for i in range(self.searchspace.shape[1])]
# #             meanstds = ["meanstd%d" % i for i in range(sum([i.shape[1] for i in self.meanandstd_all]))]
# #             name = ["number"] + fea + meanstds + ["Pi", "L", "Ei"]
# #             result1.columns = name
# #         self.result = result1
# #         return self.result
# #
