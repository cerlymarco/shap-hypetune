import io
import contextlib
import warnings
import numpy as np
import scipy as sp
from copy import deepcopy

from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import if_delegate_has_method

from joblib import Parallel, delayed
from hyperopt import fmin, tpe

from .utils import ParameterSampler, _check_param, _check_boosting
from .utils import _set_categorical_indexes, _get_categorical_support
from .utils import _feature_importances, _shap_importances


class _BoostSearch(BaseEstimator):
    """Base class for BoostSearch meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self):
        pass

    def _validate_param_grid(self, fit_params):
        """Private method to validate fitting parameters."""

        if not isinstance(self.param_grid, dict):
            raise ValueError("Pass param_grid in dict format.")
        self._param_grid = self.param_grid.copy()

        for p_k, p_v in self._param_grid.items():
            self._param_grid[p_k] = _check_param(p_v)

        if 'eval_set' not in fit_params:
            raise ValueError(
                "When tuning parameters, at least "
                "a evaluation set is required.")

        self._eval_score = np.argmax if self.greater_is_better else np.argmin
        self._score_sign = -1 if self.greater_is_better else 1

        rs = ParameterSampler(
            n_iter=self.n_iter,
            param_distributions=self._param_grid,
            random_state=self.sampling_seed
        )
        self._param_combi, self._tuning_type = rs.sample()
        self._trial_id = 1

        if self.verbose > 0:
            n_trials = self.n_iter if self._tuning_type is 'hyperopt' \
                else len(self._param_combi)
            print("\n{} trials detected for {}\n".format(
                n_trials, tuple(self.param_grid.keys())))

    def _fit(self, X, y, fit_params, params=None):
        """Private method to fit a single boosting model and extract results."""

        model = self._build_model(params)
        if isinstance(model, _BoostSelector):
            model.fit(X=X, y=y, **fit_params)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X=X, y=y, **fit_params)

        results = {'params': params, 'status': 'ok'}

        if isinstance(model, _BoostSelector):
            results['booster'] = model.estimator_
            results['model'] = model
        else:
            results['booster'] = model
            results['model'] = None

        if 'eval_set' not in fit_params:
            return results

        if self.boost_type_ == 'XGB':
            # w/ eval_set and w/ early_stopping_rounds
            if hasattr(results['booster'], 'best_score'):
                results['iterations'] = results['booster'].best_iteration
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(results['booster'].evals_result_.keys())[-1]
                eval_metric = list(results['booster'].evals_result_[valid_id])[-1]
                results['iterations'] = \
                    len(results['booster'].evals_result_[valid_id][eval_metric])
        else:
            # w/ eval_set and w/ early_stopping_rounds
            if results['booster'].best_iteration_ is not None:
                results['iterations'] = results['booster'].best_iteration_
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(results['booster'].evals_result_.keys())[-1]
                eval_metric = list(results['booster'].evals_result_[valid_id])[-1]
                results['iterations'] = \
                    len(results['booster'].evals_result_[valid_id][eval_metric])

        if self.boost_type_ == 'XGB':
            # w/ eval_set and w/ early_stopping_rounds
            if hasattr(results['booster'], 'best_score'):
                results['loss'] = results['booster'].best_score
            # w/ eval_set and w/o early_stopping_rounds
            else:
                valid_id = list(results['booster'].evals_result_.keys())[-1]
                eval_metric = list(results['booster'].evals_result_[valid_id])[-1]
                results['loss'] = \
                    results['booster'].evals_result_[valid_id][eval_metric][-1]
        else:
            valid_id = list(results['booster'].best_score_.keys())[-1]
            eval_metric = list(results['booster'].best_score_[valid_id])[-1]
            results['loss'] = results['booster'].best_score_[valid_id][eval_metric]

        if params is not None:
            if self.verbose > 0:
                msg = "trial: {} ### iterations: {} ### eval_score: {}".format(
                    str(self._trial_id).zfill(4),
                    str(results['iterations']).zfill(5),
                    round(results['loss'], 5)
                )
                print(msg)

            self._trial_id += 1
            results['loss'] *= self._score_sign

        return results

    def fit(self, X, y, trials=None, **fit_params):
        """Fit the provided boosting algorithm while searching the best subset
        of features (according to the selected strategy) and choosing the best
        parameters configuration (if provided).

        It takes the same arguments available in the estimator fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            Target values.

        trials : hyperopt.Trials() object, default=None
            A hyperopt trials object, used to store intermediate results for all
            optimization runs. Effective (and required) only when hyperopt
            parameter searching is computed.

        **fit_params : Additional fitting arguments.

        Returns
        -------
        self : object
        """

        self.boost_type_ = _check_boosting(self.estimator)

        if self.param_grid is None:
            results = self._fit(X, y, fit_params)

            for v in vars(results['model']):
                if v.endswith("_") and not v.startswith("__"):
                    setattr(self, str(v), getattr(results['model'], str(v)))

        else:
            self._validate_param_grid(fit_params)

            if self._tuning_type == 'hyperopt':
                if trials is None:
                    raise ValueError(
                        "trials must be not None when using hyperopt."
                    )

                search = fmin(
                    fn=lambda p: self._fit(
                        params=p, X=X, y=y, fit_params=fit_params
                    ),
                    space=self._param_combi, algo=tpe.suggest,
                    max_evals=self.n_iter, trials=trials,
                    rstate=np.random.RandomState(self.sampling_seed),
                    show_progressbar=False, verbose=0
                )
                all_results = trials.results

            else:
                all_results = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose * int(bool(self.n_jobs))
                )(delayed(self._fit)(X, y, fit_params, params)
                  for params in self._param_combi)

            # extract results from parallel loops
            self.trials_, self.iterations_, self.scores_, models = [], [], [], []
            for job_res in all_results:
                self.trials_.append(job_res['params'])
                self.iterations_.append(job_res['iterations'])
                self.scores_.append(self._score_sign * job_res['loss'])
                if isinstance(job_res['model'], _BoostSelector):
                    models.append(job_res['model'])
                else:
                    models.append(job_res['booster'])

            # get the best
            id_best = self._eval_score(self.scores_)
            self.best_params_ = self.trials_[id_best]
            self.best_iter_ = self.iterations_[id_best]
            self.best_score_ = self.scores_[id_best]
            self.estimator_ = models[id_best]

            for v in vars(models[id_best]):
                if v.endswith("_") and not v.startswith("__"):
                    setattr(self, str(v), getattr(models[id_best], str(v)))

        return self

    def predict(self, X, **predict_params):
        """Predict X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        **predict_params : Additional predict arguments.

        Returns
        -------
        pred : ndarray of shape (n_samples,)
            The predicted values.
        """

        check_is_fitted(self)

        if hasattr(self, 'transform'):
            X = self.transform(X)

        return self.estimator_.predict(X, **predict_params)

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X, **predict_params):
        """Predict X probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        **predict_params : Additional predict arguments.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes)
            The predicted values.
        """

        check_is_fitted(self)

        if hasattr(self, 'transform'):
            X = self.transform(X)

        return self.estimator_.predict_proba(X, **predict_params)

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

        check_is_fitted(self)

        if hasattr(self, 'transform'):
            X = self.transform(X)

        return self.estimator_.score(X, y, sample_weight=sample_weight)


class _BoostSelector(BaseEstimator, TransformerMixin):
    """Base class for feature selection meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self):
        pass

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

        check_is_fitted(self)

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


class _Boruta(_BoostSelector):
    """Base class for BoostBoruta meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.

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
                 importance_type='feature_importances',
                 train_importance=True,
                 verbose=0):

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
        estimator = clone(self.estimator)
        # add here possible estimator checks in each iteration

        _fit_params = _set_categorical_indexes(
            self.support_, self._cat_support, _fit_params, duplicate=True)

        if feat_id_real is None:  # final model fit
            if 'eval_set' in _fit_params:
                _fit_params['eval_set'] = list(map(lambda x: (
                    self.transform(x[0]), x[1]
                ), _fit_params['eval_set']))
        else:
            if 'eval_set' in _fit_params:  # iterative model fit
                _fit_params['eval_set'] = list(map(lambda x: (
                    self._create_X(x[0], feat_id_real), x[1]
                ), _fit_params['eval_set']))

        if 'feature_name' in _fit_params:  # LGB
            _fit_params['feature_name'] = 'auto'

        if 'feature_weights' in _fit_params:  # XGB  import warnings
            warnings.warn(
                "feature_weights is not supported when selecting features. "
                "It's automatically set to None.")
            _fit_params['feature_weights'] = None

        return _fit_params, estimator

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

    def fit(self, X, y, **fit_params):
        """Fit the Boruta algorithm to automatically tune
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
                print('Iteration: {} / {}'.format(i + 1, self.max_iter))

            self._random_state = np.random.RandomState(i + 1000)

            # add shadow attributes, shuffle and train estimator
            self.support_ = dec_reg >= 0
            feat_id_real = np.where(self.support_)[0]
            n_real = feat_id_real.shape[0]
            _fit_params, estimator = self._check_fit_params(fit_params, feat_id_real)
            estimator.set_params(random_state=i + 1000)
            _X = self._create_X(X, feat_id_real)
            with contextlib.redirect_stdout(io.StringIO()):
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
            imp_real = np.zeros(n_features) * np.nan
            imp_real[feat_id_real] = coefs[:n_real]

            # get the threshold of shadow importances used for rejection
            imp_sha_max = np.percentile(imp_sha, self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, imp_real))

            # register which feature is more imp than the max of shadows
            hit_reg[np.where(imp_real[~np.isnan(imp_real)] > imp_sha_max)[0]] += 1

            # check if a feature is doing better than expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, i + 1)
            dec_history[i] = dec_reg

            es_id = i - es_boruta_rounds
            if es_id >= 0:
                if np.equal(dec_history[es_id:(i + 1)], dec_reg).all():
                    if self.verbose > 0:
                        print("Boruta early stopping at iteration {}".format(i + 1))
                    break

        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]

        self.support_ = np.zeros(n_features, dtype=np.bool)
        self.ranking_ = np.ones(n_features, dtype=np.int) * 4
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
            raise RuntimeError(
                "Boruta didn't select any feature. Try to increase max_iter or "
                "increase (if not None) early_stopping_boruta_rounds or "
                "decrese perc.")

        _fit_params, self.estimator_ = self._check_fit_params(fit_params)
        with contextlib.redirect_stdout(io.StringIO()):
            self.estimator_.fit(self.transform(X), y, **_fit_params)

        return self


class _RFE(_BoostSelector):
    """Base class for BoostRFE meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 estimator, *,
                 min_features_to_select=None,
                 step=1,
                 greater_is_better=False,
                 importance_type='feature_importances',
                 train_importance=True,
                 verbose=0):

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
        estimator = clone(self.estimator)
        # add here possible estimator checks in each iteration

        _fit_params = _set_categorical_indexes(
            self.support_, self._cat_support, _fit_params)

        if 'eval_set' in _fit_params:
            _fit_params['eval_set'] = list(map(lambda x: (
                self.transform(x[0]), x[1]
            ), _fit_params['eval_set']))

        if 'feature_name' in _fit_params:  # LGB
            _fit_params['feature_name'] = 'auto'

        if 'feature_weights' in _fit_params:  # XGB  import warnings
            warnings.warn(
                "feature_weights is not supported when selecting features. "
                "It's automatically set to None.")
            _fit_params['feature_weights'] = None

        return _fit_params, estimator

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

    def fit(self, X, y, **fit_params):
        """Fit the RFE algorithm to automatically tune
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
            _fit_params, estimator = self._check_fit_params(fit_params)

            if self.verbose > 1:
                print("Fitting estimator with {} features".format(
                    self.support_.sum()))
            with contextlib.redirect_stdout(io.StringIO()):
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
        _fit_params, self.estimator_ = self._check_fit_params(fit_params)
        if self.verbose > 1:
            print("Fitting estimator with {} features".format(self.support_.sum()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.estimator_.fit(self.transform(X), y, **_fit_params)

        # compute step score when only min_features_to_select features left
        if scoring:
            score = self._step_score(self.estimator_)
            self.score_history_.append(score)
            if best_score == eval_score([score, best_score]):
                self.support_ = best_support
                self.ranking_ = best_ranking
                self.estimator_ = best_estimator
        self.n_features_ = self.support_.sum()

        return self


class _RFA(_BoostSelector):
    """Base class for BoostRFA meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 estimator, *,
                 min_features_to_select=None,
                 step=1,
                 greater_is_better=False,
                 importance_type='feature_importances',
                 train_importance=True,
                 verbose=0):

        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.greater_is_better = greater_is_better
        self.importance_type = importance_type
        self.train_importance = train_importance
        self.verbose = verbose

    def _check_fit_params(self, fit_params, inverse=False):
        """Private method to validate and check fit_params."""

        _fit_params = deepcopy(fit_params)
        estimator = clone(self.estimator)
        # add here possible estimator checks in each iteration

        _fit_params = _set_categorical_indexes(
            self.support_, self._cat_support, _fit_params)

        if 'eval_set' in _fit_params:
            _fit_params['eval_set'] = list(map(lambda x: (
                self._transform(x[0], inverse), x[1]
            ), _fit_params['eval_set']))

        if 'feature_name' in _fit_params:  # LGB
            _fit_params['feature_name'] = 'auto'

        if 'feature_weights' in _fit_params:  # XGB  import warnings
            warnings.warn(
                "feature_weights is not supported when selecting features. "
                "It's automatically set to None.")
            _fit_params['feature_weights'] = None

        return _fit_params, estimator

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

    def fit(self, X, y, **fit_params):
        """Fit the RFA algorithm to automatically tune
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
            if scoring:
                min_features_to_select = self.min_features_to_select
            else:
                min_features_to_select = n_features - self.min_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0.")

        self.support_ = np.zeros(n_features, dtype=np.bool)
        self._support = np.ones(n_features, dtype=np.bool)
        self.ranking_ = np.ones(n_features, dtype=np.int)
        self._ranking = np.ones(n_features, dtype=np.int)
        if scoring:
            self.score_history_ = []
            eval_score = np.max if self.greater_is_better else np.min
            best_score = -np.inf if self.greater_is_better else np.inf

        while np.sum(self._support) > min_features_to_select:
            # remaining features
            features = np.arange(n_features)[self._support]

            # scoring the previous added features
            if scoring and np.sum(self.support_) > 0:
                _fit_params, estimator = self._check_fit_params(fit_params)
                with contextlib.redirect_stdout(io.StringIO()):
                    estimator.fit(self._transform(X, inverse=False), y, **_fit_params)
                score = self._step_score(estimator)
                self.score_history_.append(score)
                if best_score != eval_score([score, best_score]):
                    best_score = score
                    best_support = self.support_.copy()
                    best_ranking = self.ranking_.copy()
                    best_estimator = estimator

            # evaluate the remaining features
            _fit_params, _estimator = self._check_fit_params(fit_params, inverse=True)
            if self.verbose > 1:
                print("Fitting estimator with {} features".format(self._support.sum()))
            with contextlib.redirect_stdout(io.StringIO()):
                _estimator.fit(self._transform(X, inverse=True), y, **_fit_params)
                if self._support.sum() == n_features:
                    all_features_estimator = _estimator

            # get coefs
            if self.importance_type == 'feature_importances':
                coefs = _feature_importances(_estimator)
            else:
                if eval_importance:
                    coefs = _shap_importances(
                        _estimator, _fit_params['eval_set'][-1][0])
                else:
                    coefs = _shap_importances(
                        _estimator, self._transform(X, inverse=True))
            ranks = np.argsort(-coefs)  # the rank is inverted

            # add the best features
            threshold = min(step, np.sum(self._support) - min_features_to_select)

            # remaining features to test
            self._support[features[ranks][:threshold]] = False
            self._ranking[np.logical_not(self._support)] += 1
            # features tested
            self.support_[features[ranks][:threshold]] = True
            self.ranking_[np.logical_not(self.support_)] += 1

        # set final attributes
        _fit_params, self.estimator_ = self._check_fit_params(fit_params)
        if self.verbose > 1:
            print("Fitting estimator with {} features".format(self._support.sum()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.estimator_.fit(self._transform(X, inverse=False), y, **_fit_params)

        # compute step score when only min_features_to_select features left
        if scoring:
            score = self._step_score(self.estimator_)
            self.score_history_.append(score)
            if best_score == eval_score([score, best_score]):
                self.support_ = best_support
                self.ranking_ = best_ranking
                self.estimator_ = best_estimator

            if len(set(self.score_history_)) == 1:
                self.support_ = np.ones(n_features, dtype=np.bool)
                self.ranking_ = np.ones(n_features, dtype=np.int)
                self.estimator_ = all_features_estimator
        self.n_features_ = self.support_.sum()

        return self

    def _transform(self, X, inverse=False):
        """Private method to reduce the input X to the features selected."""

        shapes = np.shape(X)
        if len(shapes) != 2:
            raise ValueError("X must be 2D.")

        if shapes[1] != self.support_.shape[0]:
            raise ValueError(
                "Expected {} features, received {}.".format(
                    self.support_.shape[0], shapes[1]))

        if inverse:
            if isinstance(X, np.ndarray):
                return X[:, self._support]
            elif hasattr(X, 'loc'):
                return X.loc[:, self._support]
            elif sp.sparse.issparse(X):
                return X[:, self._support]
            else:
                raise ValueError("Data type not understood.")
        else:
            if isinstance(X, np.ndarray):
                return X[:, self.support_]
            elif hasattr(X, 'loc'):
                return X.loc[:, self.support_]
            elif sp.sparse.issparse(X):
                return X[:, self.support_]
            else:
                raise ValueError("Data type not understood.")

    def transform(self, X):
        """Reduces the input X to the features selected with RFA.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        X : array-like of shape (n_samples, n_features_)
            The input samples with only the selected features by Boruta.
        """

        check_is_fitted(self)

        return self._transform(X, inverse=False)
