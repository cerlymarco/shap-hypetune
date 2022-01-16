import random
import numpy as np
from itertools import product

from shap import TreeExplainer


def _check_boosting(model):
    """Check if the estimator is a LGBModel or XGBModel.

    Returns
    -------
    Model type in string format.
    """

    estimator_type = str(type(model)).lower()

    boost_type = ('LGB' if 'lightgbm' in estimator_type else '') + \
                 ('XGB' if 'xgboost' in estimator_type else '')

    if len(boost_type) != 3:
        raise ValueError("Pass a LGBModel or XGBModel.")

    return boost_type


def _shap_importances(model, X):
    """Extract feature importances from fitted boosting models
    using TreeExplainer from shap.

    Returns
    -------
    array of feature importances.
    """

    explainer = TreeExplainer(
        model, feature_perturbation="tree_path_dependent")
    coefs = explainer.shap_values(X)

    if isinstance(coefs, list):
        coefs = list(map(lambda x: np.abs(x).mean(0), coefs))
        coefs = np.sum(coefs, axis=0)
    else:
        coefs = np.abs(coefs).mean(0)

    return coefs


def _feature_importances(model):
    """Extract feature importances from fitted boosting models.

    Returns
    -------
    array of feature importances.
    """

    if hasattr(model, 'coef_'):  ## booster='gblinear' (xgb)
        coefs = np.square(model.coef_).sum(axis=0)
    else:
        coefs = model.feature_importances_

    return coefs


def _get_categorical_support(n_features, fit_params):
    """Obtain boolean mask for categorical features"""

    cat_support = np.zeros(n_features, dtype=np.bool)
    cat_ids = []

    msg = "When manually setting categarical features, " \
          "pass a 1D array-like of categorical columns indices " \
          "(specified as integers)."

    if 'categorical_feature' in fit_params:  # LGB
        cat_ids = fit_params['categorical_feature']
        if len(np.shape(cat_ids)) != 1:
            raise ValueError(msg)
        if not all([isinstance(c, int) for c in cat_ids]):
            raise ValueError(msg)

    cat_support[cat_ids] = True

    return cat_support


def _set_categorical_indexes(support, cat_support, _fit_params,
                             duplicate=False):
    """Map categorical features in each data repartition"""

    if cat_support.any():

        n_features = support.sum()
        support_id = np.zeros_like(support, dtype='int32')
        support_id[support] = np.arange(n_features, dtype='int32')
        cat_feat = support_id[np.where(support & cat_support)[0]]
        # empty if support and cat_support are not alligned

        if duplicate:  # is Boruta
            cat_feat = cat_feat.tolist() + (n_features + cat_feat).tolist()
        else:
            cat_feat = cat_feat.tolist()

        _fit_params['categorical_feature'] = cat_feat

    return _fit_params


def _check_param(values):
    """Check the parameter boundaries passed in dict values.

    Returns
    -------
    list of checked parameters.
    """

    if isinstance(values, (list, tuple, np.ndarray)):
        return list(set(values))
    elif 'scipy' in str(type(values)).lower():
        return values
    elif 'hyperopt' in str(type(values)).lower():
        return values
    else:
        return [values]


class ParameterSampler(object):
    """Generator on parameters sampled from given distributions.
    If all parameters are presented as a list, sampling without replacement is
    performed. If at least one parameter is given as a scipy distribution,
    sampling with replacement is used. If all parameters are given as hyperopt
    distributions Tree of Parzen Estimators searching from hyperopt is computed.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for random sampling (such as those from scipy.stats.distributions)
        or be hyperopt distributions for bayesian searching.
        If a list is given, it is sampled uniformly.

    n_iter : integer, default=None
        Number of parameter configurations that are produced.

    random_state : int, default=None
        Pass an int for reproducible output across multiple
        function calls.

    Returns
    -------
    param_combi : list of dicts or dict of hyperopt distributions
        Parameter combinations.

    searching_type : str
        The searching algorithm used.
    """

    def __init__(self, param_distributions, n_iter=None, random_state=None):

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def sample(self):
        """Generator parameter combinations from given distributions."""

        param_distributions = self.param_distributions.copy()

        is_grid = all(isinstance(p, list)
                      for p in param_distributions.values())
        is_random = all(isinstance(p, list) or 'scipy' in str(type(p)).lower()
                        for p in param_distributions.values())
        is_hyperopt = all('hyperopt' in str(type(p)).lower()
                          or (len(p) < 2 if isinstance(p, list) else False)
                          for p in param_distributions.values())

        if is_grid:
            param_combi = list(product(*param_distributions.values()))
            param_combi = [
                dict(zip(param_distributions.keys(), combi))
                for combi in param_combi
            ]
            return param_combi, 'grid'

        elif is_random:
            if self.n_iter is None:
                raise ValueError(
                    "n_iter must be an integer >0 when scipy parameter "
                    "distributions are provided. Get None."
                )

            seed = (random.randint(1, 100) if self.random_state is None
                    else self.random_state + 1)
            random.seed(seed)

            param_combi = []
            k = self.n_iter
            for i in range(self.n_iter):
                dist = param_distributions.copy()
                combi = []
                for j, v in enumerate(dist.values()):
                    if 'scipy' in str(type(v)).lower():
                        combi.append(v.rvs(random_state=seed * (k + j)))
                    else:
                        combi.append(v[random.randint(0, len(v) - 1)])
                    k += i + j
                param_combi.append(
                    dict(zip(param_distributions.keys(), combi))
                )
            np.random.mtrand._rand

            return param_combi, 'random'

        elif is_hyperopt:
            if self.n_iter is None:
                raise ValueError(
                    "n_iter must be an integer >0 when hyperopt "
                    "search spaces are provided. Get None."
                )
            param_distributions = {
                k: p[0] if isinstance(p, list) else p
                for k, p in param_distributions.items()
            }

            return param_distributions, 'hyperopt'

        else:
            raise ValueError(
                "Parameters not recognized. "
                "Pass lists, scipy distributions (also in conjunction "
                "with lists), or hyperopt search spaces."
            )