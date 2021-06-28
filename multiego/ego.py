# -*- coding: utf-8 -*-

# @Time    : 2020/9/8 23:51
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
import sklearn.utils
from mgetool.tool import parallelize
from sklearn.utils import check_array

from multiego.base_ego import BaseEgo


def search_space(*arg):
    """
    Generate grid.

    Note
    ------
        Parameters should be no more than 6

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
    result: np.ndarray
    """
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


class Ego(BaseEgo):
    """
    EGO (Efficient global optimization).

    References
    -----------
        Jones, D. R., Schonlau, M. & Welch, W. J. Efficient global optimization of expensive black-box functions. J.
        Global Optim. 13, 455–492 (1998)

    Examples
    ----------

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
        >>> me = Ego(searchspace, X, y, 500, SVR(), n_jobs=8)
        >>> result = me.rank()

    Notes
    ------
        Result is 2 dimentions array
        1st column = sequence number,
        2nd part = your searchspace,
        3rd part = 'mean,std,ego,kg,maxp,sequentially'

    """

    def __init__(self, regclf=None, searchspace=None, X=None, y=None, number=500, n_jobs=2):
        """

        Parameters
        ----------
        searchspace: np.ndarray of shape (n_sample_pre, n_feature)
            searchspace with the same ``n_feature`` with X,
            Custom or generate by .search_space() function.
        X: np.ndarray of shape (n_sample_train, n_feature)
            X data (2D).
        y: np.ndarray of shape (n_sample_train, 1)
            y data (1D).
        number: int>100
            Repeat number, default is 500.
        regclf: sklearn.estimator
            sklearn module, with "fit" and "predict".
        n_jobs: int
            Parallelize number.
        """
        super(Ego, self).__init__()

        self.n_jobs = n_jobs
        self.searchspace = searchspace
        self.X = X
        self.y = y
        self.regclf = regclf
        self.meanandstd_all = []
        self.predict_y_all = []
        self.number = number
        self.mean_std = None
        self.rank=self.egosearch

    def fit(self, searchspace=None, X=None, y=None, *args):
        """

        Parameters
        ----------
        searchspace: np.ndarray of shape (n_sample_pre, n_feature)
            searchspace with the same ``n_feature`` with X,
            Custom or generate by .search_space() function.
        X: np.ndarray of shape (n_sample_train, n_feature)
            X data (2D).
        y: np.ndarray of shape (n_sample_train, 1)
            y data (1D).

        """
        assert hasattr(self.regclf, "fit")
        assert hasattr(self.regclf, "predict")

        self.searchspace = self.searchspace if searchspace is None else searchspace
        self.X = self.X if X is None else X
        self.y = self.y if y is None else y
        searchspace = self.searchspace
        X = self.X
        y = self.y

        njobs = self.n_jobs
        regclf0 = self.regclf
        assert searchspace is not None and X is not None and y is not None, "searchspace, X, y should be np.array"
        check_array(X, ensure_2d=True, force_all_finite=True)
        check_array(y, ensure_2d=False, force_all_finite=True)
        check_array(searchspace, ensure_2d=True, force_all_finite=True)
        assert X.shape[1] == searchspace.shape[1]

        def fit_parllize(random_state):
            data_train, y_train = sklearn.utils.resample(X, y, n_samples=None, replace=True,
                                                         random_state=random_state)
            regclf0.fit(data_train, y_train)
            predict_data = regclf0.predict(searchspace)
            predict_data.ravel()
            return predict_data

        predict_y = parallelize(n_jobs=njobs, func=fit_parllize, iterable=range(self.number))
        predict_y = np.array(predict_y).T

        self.predict_y = predict_y

        self.meanandstd()

    def meanandstd(self, predict_y=None):
        """calculate meanandstd"""
        if predict_y is not None:
            self.predict_y = predict_y
        if not hasattr(self, "predict_y"):
            raise NotImplemented("Please fit first")
        if self.predict_y is None:
            raise NotImplemented("Please fit first")

        mean_std = super().meanandstd(self.predict_y)

        self.mean_std = mean_std
        return self.mean_std

    def egosearch(self, searchspace=None, mean_std=None, rankway="ego", return_type="pd", flexibility=0, y=None,
                  fraction=1000):
        """
        Result is 2 dimentions array
        1st column = sequence number,2nd part = your searchspace,3rd part = mean,std,ego,kg,maxp,sequentially.

        Parameters
        ----------
        y: np.ndarray of shape (n_sample_true, 1)
            y data (1D).
        searchspace : np.ndarray of shape (n_sample_pre, n_feature)
            search space
            ["ego","kg","maxp","No"]
        return_type:str
            numpy.ndarray or pandas.DataFrame
        mean_std: np.ndarray of shape (n_sample_pre, n_feature)
            mean_std of n times of prediction on search space.
            First column is mean and second is std.
        rankway : str
            ["ego","kg","maxp","No"]
        fraction:int
            choice top n_sample/fraction.
        flexibility:float
            Flexibility to calculate EI, the bigger flexibility, the more search space Ei >0.
        """

        y = self.y if y is None else y
        searchspace = self.searchspace if searchspace is None else searchspace
        mean_std = self.mean_std if mean_std is None else mean_std

        if mean_std is None:
            self.fit()
            mean_std = self.mean_std

        return super().egosearch(y=y, searchspace=searchspace, mean_std=mean_std, rankway=rankway,
                                 return_type=return_type, flexibility=flexibility, fraction=fraction)


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    import numpy as np
    from sklearn.svm import SVR

    #####model1#####
    model = SVR()
    ###

    #####model2#####
    # parameters = {'C': [0.1, 1, 10]}
    # model = GridSearchCV(SVR(), parameters)
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

    me = Ego(model, searchspace, X, y, 100, n_jobs=6)
    me.fit()
    re = me.egosearch()
