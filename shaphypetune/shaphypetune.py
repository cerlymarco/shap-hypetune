import warnings
import numpy as np
import scipy as sp
from copy import deepcopy

from .utils import ParameterSampler, _check_param, _check_boosting
from .utils import _set_categorical_indexes, _get_categorical_support
from .utils import _feature_importances, _shap_importances


class _BoostSearch:
    """Base class for BoostSearch meta-estimator.
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    
    def __init__(self,
                 estimator,
                 param_grid,
                 greater_is_better = False,
                 n_iter = None,
                 sampling_seed = None,
                 verbose = 1):
        
        self.estimator = estimator
        self.param_grid = param_grid
        self.greater_is_better = greater_is_better
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        
    def __repr__(self):
        return "<shaphypetune.{}>".format(self.__class__.__name__)

    def __str__(self):
        return "<shaphypetune.{}>".format(self.__class__.__name__)
    
    def _validate_params(self, fit_params):
        """Private method to validate fitting parameters."""
        
        self.trials_ = []
        self.scores_ = []
        self.iterations_ = []
        
        self.boost_type_ = _check_boosting(self.estimator)

        if not isinstance(self.param_grid, dict):
            raise ValueError("Pass param_grid in dict format.")
        self._param_grid = self.param_grid.copy()
            
        for p_k, p_v in self._param_grid.items():
            self._param_grid[p_k] = _check_param(p_v)
                    
        if 'eval_set' not in fit_params:
            raise ValueError(
                "When tuning parameters, at least "
                "a evaluation set is required.")
                    
        self._eval_score = np.max if self.greater_is_better else np.min
        start_score = -np.inf if self.greater_is_better else np.inf
        self.best_score_ = start_score 
        
        rs = ParameterSampler(n_iter = self.n_iter, 
                              param_distributions = self._param_grid,
                              random_state = self.sampling_seed)
        self._param_combi = rs.sample()
        
        if self.verbose > 0:
            print("\n{} trials detected for {}\n".format(
                len(self._param_combi), tuple(self._param_grid.keys())))
        else:
            fit_params['verbose'] = 0
            
        return fit_params
        
    def _store_results(self, model, trial, param):
        """Private method to evaluate a single trial and store results."""
        
        if self.boost_type_ == 'XGB':
            # w/ eval_set and w/ early_stopping_rounds
            if hasattr(model, 'best_score'):
                iteration = model.best_iteration
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(model.evals_result_.keys())[-1]
                eval_metric = list(model.evals_result_[valid_id])[-1]
                iteration = len(model.evals_result_[valid_id][eval_metric])
        else:
            # w/ eval_set and w/ early_stopping_rounds
            if model.best_iteration_ is not None:
                iteration = model.best_iteration_
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(model.evals_result_.keys())[-1]
                eval_metric = list(model.evals_result_[valid_id])[-1]
                iteration = len(model.evals_result_[valid_id][eval_metric]) 
            
        if self.boost_type_ == 'XGB':
            # w/ eval_set and w/ early_stopping_rounds
            if hasattr(model, 'best_score'):
                score = model.best_score
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(model.evals_result_.keys())[-1]
                eval_metric = list(model.evals_result_[valid_id])[-1]
                score = model.evals_result_[valid_id][eval_metric][-1]            
        else:         
            valid_id = list(model.best_score_.keys())[-1]
            eval_metric = list(model.best_score_[valid_id])[-1]
            score = model.best_score_[valid_id][eval_metric]

        evaluate = self._eval_score([self.best_score_, score])

        if self.best_score_ != evaluate:
            self.best_params_ = param
            self.best_iter_ = iteration
            self.estimator_ = model

        self.best_score_ = evaluate
        self.trials_.append(param)
        self.scores_.append(score)
        self.iterations_.append(iteration)

        if self.verbose > 0:
            msg = "trial: {} ### iterations: {} ### eval_score: {}".format(
                str(trial+1).zfill(4), str(iteration).zfill(5), 
                round(score, 5))
            print(msg)
    

class BoostSearch(_BoostSearch):
    """Hyperparamater searching and optimization on a given validation set
    for LGBModel or XGBModel. 
    
    Pass a LGBModel or XGBModel, and a dictionary with the parameter boundaries 
    for grid or random search. To operate a random search pass distributions 
    in the param_grid with rvs method for sampling (such as those from 
    scipy.stats.distributions) specifing n_iter or sampling_seed.
    The best parameter combination is the one which obtain the better score
    (as returned by eval_metric) on the provided eval_set.
    
    If all parameters are presented as a list/floats/integers, grid-search 
    is performed. If at least one parameter is given as a distribution (such as 
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. It is highly recommended to use continuous 
    distributions for continuous parameters.
    
    Parameters
    ----------
    estimator : object
        A supervised learning estimator of LGBModel or XGBModel type.
    
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        
    greater_is_better : bool, default=False
        Whether the quantity to monitor is a score function, 
        meaning high is good, or a loss function, meaning low is good.
        
    n_iter : int, default=None
        Effective only for random serach.
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.
        
    sampling_seed : int, default=None
        Effective only for random serach.
        The seed used to sample from the hyperparameter distributions.
        
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
        The scores achived on the eval_set by all the models tried.
        
    best_iter_ : int
        The boosting iterations achieved by the best parameters combination.
    
    iterations_ : list
        The boosting iterations of all the models tried.
        
    boost_type_ : str
        The type of the boosting estimator (LGB or XGB).
    """
    
    def __init__(self,
                 estimator,
                 param_grid,
                 greater_is_better = False,
                 n_iter = None,
                 sampling_seed = None,
                 verbose = 1):
        
        self.estimator = estimator
        self.param_grid = param_grid
        self.greater_is_better = greater_is_better
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.verbose = verbose
        
    def fit(self, X, y, **fit_params):
        """Performs a search for best parameters configuration 
        creating all the possible trials and evaluating on the 
        validation set provided.
        
        It takes the same arguments available in the estimator fit.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. 
            
        y : array-like of shape (n_samples,)
            Target values. 
            
        **fit_params : Additional fitting arguments.
        
        Returns
        -------
        self : object
        """
        
        fit_params = self._validate_params(fit_params)
        
        for trial,param in enumerate(self._param_combi):
            
            param = dict(zip(self._param_grid.keys(), param))
            model = deepcopy(self.estimator)
            model.set_params(**param)
                                    
            model.fit(X = X, y = y, **fit_params)
            
            self._store_results(model, trial, param)
            
        return self
    
    def predict(self, X, method='predict', **predargs):
        """Predict X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples. 
            
        method : str, default='predict'
            The method to be invoked by estimator.
            
        **predargs : Additional predict arguments.
        
        Returns
        -------
        pred : ndarray of shape (n_samples,)
            The predicted values.
        """

        if not hasattr(self, 'estimator_'):
            raise AttributeError("Not fitted instance.")
        
        func = getattr(self.estimator_, method)
        return func(X, **predargs)
            
    def score(self, X, y, sample_weight=None):
        """Return the score on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) 
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Accuracy for classification, R2 for regression.
        """

        if not hasattr(self, 'estimator_'):
            raise AttributeError("Not fitted instance.")
                
        return self.estimator_.score(X, y, sample_weight=sample_weight)
    
    
class _Boruta:
    """Base class for BoostBoruta meta-estimator.
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    
    Notes
    -----
    The code for the Boruta algorithm is inspired and improved from:
    https://github.com/scikit-learn-contrib/boruta_py
    """
    
    def __init__(self, 
                 estimator, 
                 perc = 100, 
                 alpha = 0.05,
                 max_iter = 100, 
                 early_stopping_boruta_rounds = None,
                 importance_type = 'feature_importances',
                 train_importance = True,
                 verbose = 0):
        
        self.estimator = estimator
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping_boruta_rounds = early_stopping_boruta_rounds
        self.importance_type = importance_type
        self.train_importance = train_importance
        self.verbose = verbose
        
    def _create_X(self, X, feat_id_real):
        """Private method to add shadow features to the original ones. """
        
        if isinstance(X, np.ndarray):
            X_real = X[:, feat_id_real].copy()
            X_sha = X_real.copy()
            X_sha = np.apply_along_axis(self._random_state.permutation, 0, X_sha)
            
            X = np.hstack((X_real, X_sha))
                        
        elif hasattr(X, 'iloc'):
            X_real = X.iloc[:, feat_id_real].copy()
            X_sha = X_real.copy()
            X_sha = X_sha.apply(self._random_state.permutation)
            X_sha = X_sha.astype(X_real.dtypes)           
            
            X = X_real.join(X_sha, rsuffix='_SHA')
            
        else:
            raise ValueError("Data type not understood.")
        
        return X       
        
    def _check_fit_params(self, fit_params, feat_id_real=None):
        """Private method to validate and check fit_params."""
        
        _fit_params = deepcopy(fit_params)
        
        _fit_params = _set_categorical_indexes(
            self.support_, self._cat_support, _fit_params, duplicate=True)
        
        if feat_id_real is None:  # final model fit
            if 'eval_set' in fit_params:
                _fit_params['eval_set'] = list(map(lambda x: (
                    self.transform(x[0]), x[1]
                ), _fit_params['eval_set']))
        else:
            if 'eval_set' in fit_params:  # iterative model fit
                _fit_params['eval_set'] = list(map(lambda x: (
                    self._create_X(x[0], feat_id_real), x[1]
                ), _fit_params['eval_set']))
                
        if 'feature_name' in _fit_params:  # LGB
            _fit_params['feature_name'] = 'auto'

        if 'feature_weights' in _fit_params:  # XGB
            warnings.warn(
                "feature_weights is not supported when selecting features. "
                "It's automatically set to None.")
            _fit_params['feature_weights'] = None
                
        return _fit_params
        
    def _do_tests(self, dec_reg, hit_reg, iter_id):
        """Private method to operate Bonferroni corrections on the feature
        selections."""
        
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, iter_id, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, iter_id, .5).flatten()

        # Bonferroni correction with the total n_features in each iteration
        to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
        to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        
        return dec_reg
        
    def _fit(self, X, y, **fit_params):
        """Private method to fit the Boruta algorithm and automatically tune 
        the number of selected features."""
        
        self.boost_type_ = _check_boosting(self.estimator)
        
        if self.max_iter < 1:
            raise ValueError('max_iter should be an integer >0.')
        
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('alpha should be between 0 and 1.')
            
        if self.early_stopping_boruta_rounds is None:
            es_boruta_rounds = self.max_iter
        else:
            if self.early_stopping_boruta_rounds < 1:
                raise ValueError(
                    'early_stopping_boruta_rounds should be an integer >0.')
            es_boruta_rounds = self.early_stopping_boruta_rounds
            
        importances = ['feature_importances', 'shap_importances']
        if self.importance_type not in importances:
            raise ValueError(
                "importance_type must be one of {}. Get '{}'".format(
                    importances, self.importance_type))
        
        if self.importance_type == 'shap_importances':
            if not self.train_importance and not 'eval_set' in fit_params:
                raise ValueError(
                    "When train_importance is set to False, using "
                    "shap_importances, pass at least a eval_set.")
            eval_importance = not self.train_importance and 'eval_set' in fit_params
        
        shapes = np.shape(X)
        if len(shapes) != 2:
            raise ValueError("X must be 2D.")
        n_features = shapes[1]
        
        # create mask for user-defined categorical features
        self._cat_support = _get_categorical_support(n_features, fit_params)

        # holds the decision about each feature:
        # default (0); accepted (1); rejected (-1)
        dec_reg = np.zeros(n_features, dtype=np.int)
        dec_history = np.zeros((self.max_iter, n_features), dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_features, dtype=np.int)
        # record the history of the iterations
        imp_history = np.zeros(n_features, dtype=np.float)
        sha_max_history = []

        for i in range(self.max_iter):
            if (dec_reg != 0).all():
                if self.verbose > 1:
                    print("All Features analyzed. Boruta stop!")
                break
                
            if self.verbose > 1:
                print('Iterantion: {} / {}'.format(i+1, self.max_iter))
            
            self._random_state = np.random.RandomState(i+1000)
            estimator = deepcopy(self.estimator)
            estimator.set_params(random_state=i+1000)

            # add shadow attributes, shuffle and train estimator
            self.support_ = dec_reg >= 0
            feat_id_real = np.where(self.support_)[0]
            n_real = feat_id_real.shape[0]
            _fit_params = self._check_fit_params(fit_params, feat_id_real)
            _X = self._create_X(X, feat_id_real)
            estimator.fit(_X, y, **_fit_params)

            # get coefs
            if self.importance_type == 'feature_importances':
                coefs = _feature_importances(estimator)
            else:
                if eval_importance:
                    coefs = _shap_importances(
                        estimator, _fit_params['eval_set'][-1][0])
                else:
                    coefs = _shap_importances(estimator, _X) 

            # separate importances of real and shadow features
            imp_sha = coefs[n_real:]
            imp_real = np.zeros(n_features) *np.nan
            imp_real[feat_id_real] = coefs[:n_real]

            # get the threshold of shadow importances used for rejection
            imp_sha_max = np.percentile(imp_sha, self.perc)
            
            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, imp_real))

            # register which feature is more imp than the max of shadows            
            hit_reg[np.where(imp_real[~np.isnan(imp_real)] > imp_sha_max)[0]] += 1

            # check if a feature is doing better than expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, i+1)
            dec_history[i] = dec_reg
            
            es_id = i-es_boruta_rounds
            if es_id >= 0:
                if np.equal(dec_history[es_id:(i+1)], dec_reg).all():
                    if self.verbose > 0:
                        print("Boruta early stopping at iteration {}".format(i+1))
                    break
                    
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        
        self.support_ = np.zeros(n_features, dtype=np.bool)
        self.ranking_ = np.ones(n_features, dtype=np.int) *4
        self.n_features_ = confirmed.shape[0]
        self.importance_history_ = imp_history[1:]
                
        if tentative.shape[0] > 0:
            tentative_median = np.nanmedian(imp_history[1:, tentative], axis=0)
            tentative_low = tentative[
                np.where(tentative_median <= np.median(sha_max_history))[0]]
            tentative_up = np.setdiff1d(tentative, tentative_low)

            self.ranking_[tentative_low] = 3
            if tentative_up.shape[0] > 0:
                self.ranking_[tentative_up] = 2
            
        if confirmed.shape[0] > 0:
            self.support_[confirmed] = True
            self.ranking_[confirmed] = 1
            
        if (~self.support_).all():
            raise AttributeError(
                "Boruta didn't select any feature. Try to increase max_iter or "
                "increase (if not None) early_stopping_boruta_rounds or "
                "decrese perc.")
            
        self.estimator_ = deepcopy(self.estimator)
        _fit_params = self._check_fit_params(fit_params)
        self.estimator_.fit(self.transform(X), y, **_fit_params)
        
        return self
        
    def transform(self, X):
        """Reduces the input X to the features selected by Boruta.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
 
        Returns
        -------
        X : array-like of shape (n_samples, n_features_)
            The input samples with only the selected features by Boruta.
        """
        
        if not hasattr(self, 'estimator_'):
            raise AttributeError("Not fitted instance.")
        
        shapes = np.shape(X)
        if len(shapes) != 2:
            raise ValueError("X must be 2D.")
                
        if shapes[1] != self.support_.shape[0]:
            raise ValueError(
                "Expected {} features, received {}.".format(
                    self.support_.shape[0], shapes[1]))
        
        if isinstance(X, np.ndarray):
            return X[:, self.support_]
        elif hasattr(X, 'loc'):
            return X.loc[:, self.support_]
        else:
            raise ValueError("Data type not understood.")
            
            
class BoostBoruta(_BoostSearch, _Boruta):
    """Simultaneous features selection with Boruta algorithm and hyperparamater 
    searching on a given validation set for LGBModel or XGBModel. 
    
    Pass a LGBModel or XGBModel to compute features selection with Boruta 
    algorithm. The best features are used to train a new gradient boosting
    instance. When a eval_set is provided, shadow features are build also on it.
    
    If param_grid is a dictionary with parameter boundaries, a hyperparameter 
    tuning is computed simultaneusly. The parameter combinations are scored on
    the provided eval_set. To operate a random search pass distributions in the 
    param_grid with rvs method for sampling (such as those from 
    scipy.stats.distributions) specifing n_iter or sampling_seed. The best 
    parameter combination is the one which obtain the better score (as returned 
    by eval_metric) on the provided eval_set.
    
    If all parameters are presented as a list/floats/integers, grid-search 
    is performed. If at least one parameter is given as a distribution (such as 
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. It is highly recommended to use continuous 
    distributions for continuous parameters.
    
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
         (the default feature importances of the gradient boosting estimator)
         or 'shap_importances'.
    
    train_importance : bool, default=True
        Effective only when importance_type='shap_importances'.
        Where to compute the shap feature importances: on train (True)
        or on eval_set (False). 
        
    param_grid : dict, default=None
        Effective only when hyperparameters searching.
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        None means no hyperparameters search.
        
    greater_is_better : bool, default=False
        Effective only when hyperparameters searching.
        Whether the quantity to monitor is a score function, 
        meaning high is good, or a loss function, meaning low is good.
        
    n_iter : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random serach.
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.
        
    sampling_seed : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random serach.
        The seed used to sample from the hyperparameter distributions.
        
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
                 estimator, 
                 perc = 100, 
                 alpha = 0.05,
                 max_iter = 100, 
                 early_stopping_boruta_rounds = None,
                 param_grid = None,
                 greater_is_better = False,
                 importance_type = 'feature_importances',
                 train_importance = True,
                 n_iter = None,
                 sampling_seed = None,
                 verbose = 1):
        
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
        
    def fit(self, X, y, **fit_params):
        """Fit the Boruta algorithm to automate the selection of the best 
        features and the best parameters configuration (if provided).
        
        It takes the same arguments available in the estimator fit.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. 
            
        y : array-like of shape (n_samples,)
            Target values. 
            
        **fit_params : Additional fitting arguments.
        
        Returns
        -------
        self : object
        """
                
        if self.param_grid is None:
            
            boruta = _Boruta(
                 estimator = self.estimator, 
                 perc = self.perc, 
                 alpha = self.alpha,
                 max_iter = self.max_iter, 
                 early_stopping_boruta_rounds = self.early_stopping_boruta_rounds,
                 importance_type = self.importance_type,
                 train_importance = self.train_importance,
                 verbose = self.verbose)
            boruta._fit(X, y, **fit_params)
            
            self.boost_type_ = boruta.boost_type_
            self.estimator_ =  boruta.estimator_
            self.n_features_ = boruta.n_features_
            self.support_ = boruta.support_
            self.ranking_ = boruta.ranking_
            self.importance_history_ = boruta.importance_history_
            
        else:
            
            fit_params = self._validate_params(fit_params)

            for trial,param in enumerate(self._param_combi):

                param = dict(zip(self._param_grid.keys(), param))
                model = deepcopy(self.estimator)
                model.set_params(**param)
                
                boruta = _Boruta(
                     estimator = model, 
                     perc = self.perc, 
                     alpha = self.alpha,
                     max_iter = self.max_iter, 
                     early_stopping_boruta_rounds = self.early_stopping_boruta_rounds,
                     importance_type = self.importance_type,
                     train_importance = self.train_importance,
                     verbose = self.verbose)
                boruta._fit(X, y, **fit_params)

                self._store_results(boruta.estimator_, trial, param)          
                
                if self.best_params_ == param:
                    self.n_features_ = boruta.n_features_
                    self.support_ = boruta.support_
                    self.ranking_ = boruta.ranking_ 
                    self.importance_history_ = boruta.importance_history_
                                        
        return self
    
    def predict(self, X, method='predict', **predargs):
        """Predict X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples. 
        method : str, default='predict'
            The method to be invoked by estimator.
            
        **predargs : Additional predict arguments.
        
        Returns
        -------
        pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        
        X = self.transform(X)
        func = getattr(self.estimator_, method)
        return func(X, **predargs)
            
    def score(self, X, y, sample_weight=None):
        """Return the score on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) 
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Accuracy for classification, R2 for regression.
        """
        
        X = self.transform(X)
        
        return self.estimator_.score(X, y, sample_weight=sample_weight)
    
    
class _RFE:
    """Base class for BoostRFE meta-estimator.
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    
    def __init__(self,
                 estimator,
                 min_features_to_select = None,
                 step = 1,
                 greater_is_better = False,
                 importance_type = 'feature_importances',
                 train_importance = True,
                 verbose = 0):
        
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select 
        self.step = step
        self.greater_is_better = greater_is_better
        self.importance_type = importance_type
        self.train_importance = train_importance
        self.verbose = verbose
    
    def _check_fit_params(self, fit_params):
        """Private method to validate and check fit_params."""
        
        _fit_params = deepcopy(fit_params)

        _fit_params = _set_categorical_indexes(
            self.support_, self._cat_support, _fit_params)
        
        if 'eval_set' in fit_params:
            _fit_params['eval_set'] = list(map(lambda x: (
                self.transform(x[0]), x[1]
            ), _fit_params['eval_set']))
            
        if 'feature_name' in _fit_params:  # LGB
            _fit_params['feature_name'] = 'auto'

        if 'feature_weights' in _fit_params:  # XGB
            warnings.warn(
                "feature_weights is not supported when selecting features. "
                "It's automatically set to None.")
            _fit_params['feature_weights'] = None
            
        return _fit_params
    
    def _step_score(self, estimator):
        """Return the score for a fit on eval_set."""
        
        if self.boost_type_ == 'LGB':
            valid_id = list(estimator.best_score_.keys())[-1]
            eval_metric = list(estimator.best_score_[valid_id])[-1]
            score = estimator.best_score_[valid_id][eval_metric]
        else:
            # w/ eval_set and w/ early_stopping_rounds
            if hasattr(estimator, 'best_score'):
                score = estimator.best_score
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(estimator.evals_result_.keys())[-1]
                eval_metric = list(estimator.evals_result_[valid_id])[-1]
                score = estimator.evals_result_[valid_id][eval_metric][-1]
                
        return score
        
    def _fit(self, X, y, **fit_params):
        """Private method to fit the RFE algorithm and automatically tune 
        the number of selected features."""
        
        self.boost_type_ = _check_boosting(self.estimator)
        
        importances = ['feature_importances', 'shap_importances']
        if self.importance_type not in importances:
            raise ValueError(
                "importance_type must be one of {}. Get '{}'".format(
                    importances, self.importance_type))
            
        # scoring controls the calculation of self.score_history_
        # scoring is used automatically when 'eval_set' is in fit_params
        scoring = 'eval_set' in fit_params
        if self.importance_type == 'shap_importances':
            if not self.train_importance and not scoring:
                raise ValueError(
                    "When train_importance is set to False, using "
                    "shap_importances, pass at least a eval_set.")
            eval_importance = not self.train_importance and scoring
        
        shapes = np.shape(X)
        if len(shapes) != 2:
            raise ValueError("X must be 2D.")
        n_features = shapes[1]
        
        # create mask for user-defined categorical features
        self._cat_support = _get_categorical_support(n_features, fit_params)
        
        if self.min_features_to_select is None:
            if scoring:
                min_features_to_select = 1
            else:
                min_features_to_select = n_features // 2
        else:
            min_features_to_select = self.min_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0.")

        self.support_ = np.ones(n_features, dtype=np.bool)
        self.ranking_ = np.ones(n_features, dtype=np.int)

        if scoring:
            self.score_history_ = []
            eval_score = np.max if self.greater_is_better else np.min
            best_score = -np.inf if self.greater_is_better else np.inf
        
        while np.sum(self.support_) > min_features_to_select:
            # remaining features
            features = np.arange(n_features)[self.support_]
            _fit_params = self._check_fit_params(fit_params)
            estimator = deepcopy(self.estimator)
            if self.verbose > 1:
                print("Fitting estimator with {} features".format(
                    self.support_.sum()))
            
            estimator.fit(self.transform(X), y, **_fit_params)

            # get coefs
            if self.importance_type == 'feature_importances':
                coefs = _feature_importances(estimator)
            else:
                if eval_importance:
                    coefs = _shap_importances(
                        estimator, _fit_params['eval_set'][-1][0])
                else:
                    coefs = _shap_importances(
                        estimator, self.transform(X)) 
            ranks = np.argsort(coefs)

            # eliminate the worse features
            threshold = min(step, np.sum(self.support_) - min_features_to_select)

            # compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if scoring:
                score = self._step_score(estimator)
                self.score_history_.append(score)
                if best_score != eval_score([score, best_score]):
                    best_score = score
                    best_support = self.support_.copy()
                    best_ranking = self.ranking_.copy()
                    best_estimator = estimator
                
            self.support_[features[ranks][:threshold]] = False
            self.ranking_[np.logical_not(self.support_)] += 1

        # set final attributes
        _fit_params = self._check_fit_params(fit_params)
        self.estimator_ = deepcopy(self.estimator)
        self.estimator_.fit(self.transform(X), y, **_fit_params)

        # compute step score when only min_features_to_select features left
        if scoring:
            score = self._step_score(estimator)
            self.score_history_.append(score)
            if best_score == eval_score([score, best_score]):
                self.support_ = best_support
                self.ranking_ = best_ranking
                self.estimator_ = best_estimator
        self.n_features_ = self.support_.sum()
        
        return self
        
    def transform(self, X):
        """Reduces the input X to the features selected with RFE.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
 
        Returns
        -------
        X : array-like of shape (n_samples, n_features_)
            The input samples with only the selected features by Boruta.
        """
        
        if not hasattr(self, 'support_'):
            raise AttributeError("Not fitted instance.")
        
        shapes = np.shape(X)
        if len(shapes) != 2:
            raise ValueError("X must be 2D.")
                
        if shapes[1] != self.support_.shape[0]:
            raise ValueError(
                "Expected {} features, received {}.".format(
                    self.support_.shape[0], shapes[1]))
        
        if isinstance(X, np.ndarray):
            return X[:, self.support_]
        elif hasattr(X, 'loc'):
            return X.loc[:, self.support_]
        elif sp.sparse.issparse(X):
            return X[:, self.support_]
        else:
            raise ValueError("Data type not understood.")
            
            
class BoostRFE(_BoostSearch, _RFE):
    """Simultaneous features selection with RFE and hyperparamater searching 
    on a given validation set for LGBModel or XGBModel. 
    
    Pass a LGBModel or XGBModel to compute features selection with RFE. 
    The gradient boosting instance with the best features is selected. 
    When a eval_set is provided, the best gradient boosting and the best 
    features are obtained evaluating the score with eval_metric. 
    Otherwise, the best combination is obtained looking only at feature 
    importances.
    
    If param_grid is a dictionary with parameter boundaries, a hyperparameter 
    tuning is computed simultaneusly. The parameter combinations are scored on
    the provided eval_set. To operate a random search pass distributions in the 
    param_grid with rvs method for sampling (such as those from 
    scipy.stats.distributions) specifing n_iter or sampling_seed. The best 
    parameter combination is the one which obtain the better score (as returned 
    by eval_metric) on the provided eval_set.
    
    If all parameters are presented as a list/floats/integers, grid-search 
    is performed. If at least one parameter is given as a distribution (such as 
    those from scipy.stats.distributions), random-search is performed computing
    sampling with replacement. It is highly recommended to use continuous 
    distributions for continuous parameters.
    
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
         (the default feature importances of the gradient boosting estimator)
         or 'shap_importances'.
    
    train_importance : bool, default=True
        Effective only when importance_type='shap_importances'.
        Where to compute the shap feature importances: on train (True)
        or on eval_set (False). 
        
    param_grid : dict, default=None
        Effective only when hyperparameters searching.
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        None means no hyperparameters search.
        
    greater_is_better : bool, default=False
        Effective only when hyperparameters searching.
        Whether the quantity to monitor is a score function, 
        meaning high is good, or a loss function, meaning low is good.
        
    n_iter : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random serach.
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.
        
    sampling_seed : int, default=None
        Effective only when hyperparameters searching.
        Effective only for random serach.
        The seed used to sample from the hyperparameter distributions.
        
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
        The scores achived on the eval_set by all the models tried.
        
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
                 estimator,
                 min_features_to_select = None,
                 step = 1,
                 param_grid = None,
                 greater_is_better = False,
                 importance_type = 'feature_importances',
                 train_importance = True,
                 n_iter = None,
                 sampling_seed = None,
                 verbose = 1):
        
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
        
    def fit(self, X, y, **fit_params):
        """Fit a RFE to automate the selection of the best features 
        and the best parameters configuration (if provided).
        
        It takes the same arguments available in the estimator fit.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. 
            
        y : array-like of shape (n_samples,)
            Target values. 
            
        **fit_params : Additional fitting arguments.
        
        Returns
        -------
        self : object
        """
                
        if self.param_grid is None:
            
            rfe = _RFE(
                estimator = self.estimator,
                min_features_to_select = self.min_features_to_select, 
                step = self.step,
                greater_is_better = self.greater_is_better,
                importance_type = self.importance_type,
                train_importance = self.train_importance,
                verbose = self.verbose)
            rfe._fit(X, y, **fit_params)
            
            self.boost_type_= rfe.boost_type_
            self.estimator_ =  rfe.estimator_
            self.n_features_ = rfe.n_features_
            self.support_ = rfe.support_
            self.ranking_ = rfe.ranking_
            if hasattr(rfe, 'score_history_'):
                self.score_history_ = rfe.score_history_
            
        else:
            
            fit_params = self._validate_params(fit_params)

            for trial,param in enumerate(self._param_combi):

                param = dict(zip(self._param_grid.keys(), param))
                model = deepcopy(self.estimator)
                model.set_params(**param)
                
                rfe = _RFE(
                    estimator = model,
                    min_features_to_select = self.min_features_to_select, 
                    step = self.step,
                    greater_is_better = self.greater_is_better,
                    importance_type = self.importance_type,
                    train_importance = self.train_importance,
                    verbose = self.verbose)
                rfe._fit(X, y, **fit_params)

                self._store_results(rfe.estimator_, trial, param)          
                
                if self.best_params_ == param:
                    self.n_features_ = rfe.n_features_
                    self.support_ = rfe.support_
                    self.ranking_ = rfe.ranking_ 
                    self.score_history_ = rfe.score_history_
                                        
        return self
    
    def predict(self, X, method='predict', **predargs):
        """Predict X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples. 
        method : str, default='predict'
            The method to be invoked by estimator.
            
        **predargs : Additional predict arguments.
        
        Returns
        -------
        pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        
        X = self.transform(X)
        func = getattr(self.estimator_, method)
        return func(X, **predargs)
            
    def score(self, X, y, sample_weight=None):
        """Return the score on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) 
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Accuracy for classification, R2 for regression.
        """
        
        X = self.transform(X)
        
        return self.estimator_.score(X, y, sample_weight=sample_weight)