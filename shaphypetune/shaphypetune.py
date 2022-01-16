from sklearn.base import clone

from ._classes import _BoostSearch, _Boruta, _RFA, _RFE


class BoostSearch(_BoostSearch):
    """Hyperparamater searching and optimization on a given validation set
    for LGBModel or XGBModel. 

    Pass a LGBModel or XGBModel, and a dictionary with the parameter boundaries 
    for grid, random or bayesian search. 
    To operate random search pass distributions in the param_grid with rvs 
    method for sampling (such as those from scipy.stats.distributions). 
    To operate bayesian search pass hyperopt distributions.   
    The specification of n_iter or sampling_seed is effective only with random
    or hyperopt searches.
    The best parameter combination is the one which obtain the better score
    (as returned by eval_metric) on the provided eval_set.

    If all parameters are presented as a list/floats/integers, grid-search 
    is performed. If at least one parameter is given as a distribution (such as 
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. Bayesian search is effective only when all the 
    parameters to tune are in form of hyperopt distributions. 
    It is highly recommended to use continuous distributions for continuous 
    parameters.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator of LGBModel or XGBModel type.

    param_grid : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. 

    greater_is_better : bool, default=False
        Whether the quantity to monitor is a score function, 
        meaning high is good, or a loss function, meaning low is good.

    n_iter : int, default=None
        Effective only for random or hyperopt search.
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.

    sampling_seed : int, default=None
        Effective only for random or hyperopt search.
        The seed used to sample from the hyperparameter distributions.

    n_jobs : int, default=None
        Effective only with grid and random search.
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=1
        Verbosity mode. <=0 silent all; >0 print trial logs with the 
        connected score.

    Attributes
    ----------
    estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave the best score on the eval_set.

    best_params_ : dict
        Parameter setting that gave the best results on the eval_set.

    trials_ : list
        A list of dicts. The dicts are all the parameter combinations tried 
        and derived from the param_grid.

    best_score_ : float
        The best score achieved by all the possible combination created.

    scores_ : list
        The scores achieved on the eval_set by all the models tried.

    best_iter_ : int
        The boosting iterations achieved by the best parameters combination.

    iterations_ : list
        The boosting iterations of all the models tried.

    boost_type_ : str
        The type of the boosting estimator (LGB or XGB).
    """

    def __init__(self,
                 estimator, *,
                 param_grid,
                 greater_is_better=False,
                 n_iter=None,
                 sampling_seed=None,
                 verbose=1,
                 n_jobs=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.greater_is_better = greater_is_better
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _build_model(self, params):
        """Private method to build model."""

        model = clone(self.estimator)
        model.set_params(**params)

        return model


class BoostBoruta(_BoostSearch, _Boruta):
    """Simultaneous features selection with Boruta algorithm and hyperparamater
    searching on a given validation set for LGBModel or XGBModel.

    Pass a LGBModel or XGBModel to compute features selection with Boruta
    algorithm. The best features are used to train a new gradient boosting
    instance. When a eval_set is provided, shadow features are build also on it.

    If param_grid is a dictionary with parameter boundaries, a hyperparameter
    tuning is computed simultaneously. The parameter combinations are scored on
    the provided eval_set.
    To operate random search pass distributions in the param_grid with rvs
    method for sampling (such as those from scipy.stats.distributions).
    To operate bayesian search pass hyperopt distributions.
    The specification of n_iter or sampling_seed is effective only with random
    or hyperopt searches.
    The best parameter combination is the one which obtain the better score
    (as returned by eval_metric) on the provided eval_set.

    If all parameters are presented as a list/floats/integers, grid-search
    is performed. If at least one parameter is given as a distribution (such as
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. Bayesian search is effective only when all the
    parameters to tune are in form of hyperopt distributions.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator of LGBModel or XGBModel type.

    perc : int, default=100
        Threshold for comparison between shadow and real features.
        The lower perc is the more false positives will be picked as relevant
        but also the less relevant features will be left out.
        100 correspond to the max.

    alpha : float, default=0.05
        Level at which the corrected p-values will get rejected in the
        correction steps.

    max_iter : int, default=100
        The number of maximum Boruta iterations to perform.

    early_stopping_boruta_rounds : int, default=None
        The maximum amount of iterations without confirming a tentative
        feature. Use early stopping to terminate the selection process
        before reaching `max_iter` iterations if the algorithm cannot
        confirm a tentative feature after N iterations.
        None means no early stopping search.

    importance_type : str, default='feature_importances'
         Which importance measure to use. It can be 'feature_importances'
         (the default feature importance of the gradient boosting estimator)
         or 'shap_importances'.

    train_importance : bool, default=True
        Effective only when importance_type='shap_importances'.
        Where to compute the shap feature importance: on train (True)
        or on eval_set (False).

    param_grid : dict, default=None
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try.
        None means no hyperparameters search.

    greater_is_better : bool, default=False
        Effective only when hyperparameters searching.
        Whether the quantity to monitor is a score function,
        meaning high is good, or a loss function, meaning low is good.

    n_iter : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random or hyperopt seraches.
        Number of parameter settings that are sampled.
        n_iter trades off runtime vs quality of the solution.

    sampling_seed : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random or hyperopt serach.
        The seed used to sample from the hyperparameter distributions.

    n_jobs : int, default=None
        Effective only when hyperparameters searching without hyperopt.
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=1
        Verbosity mode. <=0 silent all; ==1 print trial logs (when
        hyperparameters searching); >1 print feature selection logs plus
        trial logs (when hyperparameters searching).

    Attributes
    ----------
    estimator_ : estimator
        The fitted estimator with the select features and the optimal
        parameter combination (when hyperparameters searching).

    n_features_ : int
        The number of selected features (from the best param config
        when hyperparameters searching).

    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature (from the best param config
        when hyperparameters searching). Selected features are assigned
        rank 1 (2: tentative upper bound, 3: tentative lower bound, 4:
        rejected).

    support_ : ndarray of shape (n_features,)
        The mask of selected features (from the best param config
        when hyperparameters searching).

    importance_history_ : ndarray of shape (n_features, n_iters)
        The importance values for each feature across all iterations.

    best_params_ : dict
        Available only when hyperparameters searching.
        Parameter setting that gave the best results on the eval_set.

    trials_ : list
        Available only when hyperparameters searching.
        A list of dicts. The dicts are all the parameter combinations tried
        and derived from the param_grid.

    best_score_ : float
        Available only when hyperparameters searching.
        The best score achieved by all the possible combination created.

    scores_ : list
        Available only when hyperparameters searching.
        The scores achived on the eval_set by all the models tried.

    best_iter_ : int
        Available only when hyperparameters searching.
        The boosting iterations achieved by the best parameters combination.

    iterations_ : list
        Available only when hyperparameters searching.
        The boosting iterations of all the models tried.

    boost_type_ : str
        The type of the boosting estimator (LGB or XGB).

    Notes
    -----
    The code for the Boruta algorithm is inspired and improved from:
    https://github.com/scikit-learn-contrib/boruta_py
    """

    def __init__(self,
                 estimator, *,
                 perc=100,
                 alpha=0.05,
                 max_iter=100,
                 early_stopping_boruta_rounds=None,
                 param_grid=None,
                 greater_is_better=False,
                 importance_type='feature_importances',
                 train_importance=True,
                 n_iter=None,
                 sampling_seed=None,
                 verbose=1,
                 n_jobs=None):

        self.estimator = estimator
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping_boruta_rounds = early_stopping_boruta_rounds
        self.param_grid = param_grid
        self.greater_is_better = greater_is_better
        self.importance_type = importance_type
        self.train_importance = train_importance
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _build_model(self, params=None):
        """Private method to build model."""

        estimator = clone(self.estimator)

        if params is None:
            model = _Boruta(
                estimator=estimator,
                perc=self.perc,
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping_boruta_rounds=self.early_stopping_boruta_rounds,
                importance_type=self.importance_type,
                train_importance=self.train_importance,
                verbose=self.verbose
            )

        else:
            estimator.set_params(**params)
            model = _Boruta(
                estimator=estimator,
                perc=self.perc,
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping_boruta_rounds=self.early_stopping_boruta_rounds,
                importance_type=self.importance_type,
                train_importance=self.train_importance,
                verbose=self.verbose
            )

        return model


class BoostRFE(_BoostSearch, _RFE):
    """Simultaneous features selection with RFE and hyperparamater searching
    on a given validation set for LGBModel or XGBModel.

    Pass a LGBModel or XGBModel to compute features selection with RFE.
    The gradient boosting instance with the best features is selected.
    When a eval_set is provided, the best gradient boosting and the best
    features are obtained evaluating the score with eval_metric.
    Otherwise, the best combination is obtained looking only at feature
    importance.

    If param_grid is a dictionary with parameter boundaries, a hyperparameter
    tuning is computed simultaneously. The parameter combinations are scored on
    the provided eval_set.
    To operate random search pass distributions in the param_grid with rvs
    method for sampling (such as those from scipy.stats.distributions).
    To operate bayesian search pass hyperopt distributions.
    The specification of n_iter or sampling_seed is effective only with random
    or hyperopt searches.
    The best parameter combination is the one which obtain the better score
    (as returned by eval_metric) on the provided eval_set.

    If all parameters are presented as a list/floats/integers, grid-search
    is performed. If at least one parameter is given as a distribution (such as
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. Bayesian search is effective only when all the
    parameters to tune are in form of hyperopt distributions.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator of LGBModel or XGBModel type.

    step : int or float, default=1
        If greater than or equal to 1, then `step` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than `step` features in
        order to reach `min_features_to_select`.

    min_features_to_select : int, default=None
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and `min_features_to_select` isn't divisible by
        `step`. The default value for min_features_to_select is set to 1 when a
        eval_set is provided, otherwise it always corresponds to n_features // 2.

    importance_type : str, default='feature_importances'
         Which importance measure to use. It can be 'feature_importances'
         (the default feature importance of the gradient boosting estimator)
         or 'shap_importances'.

    train_importance : bool, default=True
        Effective only when importance_type='shap_importances'.
        Where to compute the shap feature importance: on train (True)
        or on eval_set (False).

    param_grid : dict, default=None
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try.
        None means no hyperparameters search.

    greater_is_better : bool, default=False
        Effective only when hyperparameters searching.
        Whether the quantity to monitor is a score function,
        meaning high is good, or a loss function, meaning low is good.

    n_iter : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random or hyperopt serach.
        Number of parameter settings that are sampled.
        n_iter trades off runtime vs quality of the solution.

    sampling_seed : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random or hyperopt serach.
        The seed used to sample from the hyperparameter distributions.

    n_jobs : int, default=None
        Effective only when hyperparameters searching without hyperopt.
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=1
        Verbosity mode. <=0 silent all; ==1 print trial logs (when
        hyperparameters searching); >1 print feature selection logs plus
        trial logs (when hyperparameters searching).

    Attributes
    ----------
    estimator_ : estimator
        The fitted estimator with the select features and the optimal
        parameter combination (when hyperparameters searching).

    n_features_ : int
        The number of selected features (from the best param config
        when hyperparameters searching).

    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature (from the best param config
        when hyperparameters searching). Selected  features are assigned
        rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features (from the best param config
        when hyperparameters searching).

    score_history_ : list
        Available only when a eval_set is provided.
        Scores obtained reducing the features (from the best param config
        when hyperparameters searching).

    best_params_ : dict
        Available only when hyperparameters searching.
        Parameter setting that gave the best results on the eval_set.

    trials_ : list
        Available only when hyperparameters searching.
        A list of dicts. The dicts are all the parameter combinations tried
        and derived from the param_grid.

    best_score_ : float
        Available only when hyperparameters searching.
        The best score achieved by all the possible combination created.

    scores_ : list
        Available only when hyperparameters searching.
        The scores achieved on the eval_set by all the models tried.

    best_iter_ : int
        Available only when hyperparameters searching.
        The boosting iterations achieved by the best parameters combination.

    iterations_ : list
        Available only when hyperparameters searching.
        The boosting iterations of all the models tried.

    boost_type_ : str
        The type of the boosting estimator (LGB or XGB).
    """

    def __init__(self,
                 estimator, *,
                 min_features_to_select=None,
                 step=1,
                 param_grid=None,
                 greater_is_better=False,
                 importance_type='feature_importances',
                 train_importance=True,
                 n_iter=None,
                 sampling_seed=None,
                 verbose=1,
                 n_jobs=None):

        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.param_grid = param_grid
        self.greater_is_better = greater_is_better
        self.importance_type = importance_type
        self.train_importance = train_importance
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _build_model(self, params=None):
        """Private method to build model."""

        estimator = clone(self.estimator)

        if params is None:
            model = _RFE(
                estimator=estimator,
                min_features_to_select=self.min_features_to_select,
                step=self.step,
                greater_is_better=self.greater_is_better,
                importance_type=self.importance_type,
                train_importance=self.train_importance,
                verbose=self.verbose
            )

        else:
            estimator.set_params(**params)
            model = _RFE(
                estimator=estimator,
                min_features_to_select=self.min_features_to_select,
                step=self.step,
                greater_is_better=self.greater_is_better,
                importance_type=self.importance_type,
                train_importance=self.train_importance,
                verbose=self.verbose
            )

        return model


class BoostRFA(_BoostSearch, _RFA):
    """Simultaneous features selection with RFA and hyperparamater searching
    on a given validation set for LGBModel or XGBModel.

    Pass a LGBModel or XGBModel to compute features selection with RFA.
    The gradient boosting instance with the best features is selected.
    When a eval_set is provided, the best gradient boosting and the best
    features are obtained evaluating the score with eval_metric.
    Otherwise, the best combination is obtained looking only at feature
    importance.

    If param_grid is a dictionary with parameter boundaries, a hyperparameter
    tuning is computed simultaneously. The parameter combinations are scored on
    the provided eval_set.
    To operate random search pass distributions in the param_grid with rvs
    method for sampling (such as those from scipy.stats.distributions).
    To operate bayesian search pass hyperopt distributions.
    The specification of n_iter or sampling_seed is effective only with random
    or hyperopt searches.
    The best parameter combination is the one which obtain the better score
    (as returned by eval_metric) on the provided eval_set.

    If all parameters are presented as a list/floats/integers, grid-search
    is performed. If at least one parameter is given as a distribution (such as
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. Bayesian search is effective only when all the
    parameters to tune are in form of hyperopt distributions.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator of LGBModel or XGBModel type.

    step : int or float, default=1
        If greater than or equal to 1, then `step` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than `step` features in
        order to reach `min_features_to_select`.

    min_features_to_select : int, default=None
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and `min_features_to_select` isn't divisible by
        `step`. The default value for min_features_to_select is set to 1 when a
        eval_set is provided, otherwise it always corresponds to n_features // 2.

    importance_type : str, default='feature_importances'
         Which importance measure to use. It can be 'feature_importances'
         (the default feature importance of the gradient boosting estimator)
         or 'shap_importances'.

    train_importance : bool, default=True
        Effective only when importance_type='shap_importances'.
        Where to compute the shap feature importance: on train (True)
        or on eval_set (False).

    param_grid : dict, default=None
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try.
        None means no hyperparameters search.

    greater_is_better : bool, default=False
        Effective only when hyperparameters searching.
        Whether the quantity to monitor is a score function,
        meaning high is good, or a loss function, meaning low is good.

    n_iter : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random or hyperopt serach.
        Number of parameter settings that are sampled.
        n_iter trades off runtime vs quality of the solution.

    sampling_seed : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random or hyperopt serach.
        The seed used to sample from the hyperparameter distributions.

    n_jobs : int, default=None
        Effective only when hyperparameters searching without hyperopt.
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=1
        Verbosity mode. <=0 silent all; ==1 print trial logs (when
        hyperparameters searching); >1 print feature selection logs plus
        trial logs (when hyperparameters searching).

    Attributes
    ----------
    estimator_ : estimator
        The fitted estimator with the select features and the optimal
        parameter combination (when hyperparameters searching).

    n_features_ : int
        The number of selected features (from the best param config
        when hyperparameters searching).

    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature (from the best param config
        when hyperparameters searching). Selected  features are assigned
        rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features (from the best param config
        when hyperparameters searching).

    score_history_ : list
        Available only when a eval_set is provided.
        Scores obtained reducing the features (from the best param config
        when hyperparameters searching).

    best_params_ : dict
        Available only when hyperparameters searching.
        Parameter setting that gave the best results on the eval_set.

    trials_ : list
        Available only when hyperparameters searching.
        A list of dicts. The dicts are all the parameter combinations tried
        and derived from the param_grid.

    best_score_ : float
        Available only when hyperparameters searching.
        The best score achieved by all the possible combination created.

    scores_ : list
        Available only when hyperparameters searching.
        The scores achieved on the eval_set by all the models tried.

    best_iter_ : int
        Available only when hyperparameters searching.
        The boosting iterations achieved by the best parameters combination.

    iterations_ : list
        Available only when hyperparameters searching.
        The boosting iterations of all the models tried.

    boost_type_ : str
        The type of the boosting estimator (LGB or XGB).

    Notes
    -----
    The code for the RFA algorithm is inspired and improved from:
    https://github.com/heberleh/recursive-feature-addition
    """

    def __init__(self,
                 estimator, *,
                 min_features_to_select=None,
                 step=1,
                 param_grid=None,
                 greater_is_better=False,
                 importance_type='feature_importances',
                 train_importance=True,
                 n_iter=None,
                 sampling_seed=None,
                 verbose=1,
                 n_jobs=None):

        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.param_grid = param_grid
        self.greater_is_better = greater_is_better
        self.importance_type = importance_type
        self.train_importance = train_importance
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _build_model(self, params=None):
        """Private method to build model."""

        estimator = clone(self.estimator)

        if params is None:
            model = _RFA(
                estimator=estimator,
                min_features_to_select=self.min_features_to_select,
                step=self.step,
                greater_is_better=self.greater_is_better,
                importance_type=self.importance_type,
                train_importance=self.train_importance,
                verbose=self.verbose
            )

        else:
            estimator.set_params(**params)
            model = _RFA(
                estimator=estimator,
                min_features_to_select=self.min_features_to_select,
                step=self.step,
                greater_is_better=self.greater_is_better,
                importance_type=self.importance_type,
                train_importance=self.train_importance,
                verbose=self.verbose
            )

        return model