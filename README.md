# shap-hypetune
A python package for simultaneous Hyperparameters Tuning and Features Selection for Gradient Boosting Models.

![shap-hypetune diagram](https://raw.githubusercontent.com/cerlymarco/shap-hypetune/master/imgs/shap-hypetune-diagram.png#center)

## Overview
Hyperparameters tuning and features selection are two common steps in every machine learning pipeline. Most of the time they are computed separately and independently. This may result in suboptimal performances and in a more time expensive process.

shap-hypetune aims to combine hyperparameters tuning and features selection in a single pipeline optimizing the optimal number of features while searching for the optimal parameters configuration. Hyperparameters Tuning or Features Selection can also be carried out as standalone operations.

**shap-hypetune main features:**

- designed for gradient boosting models, as LGBModel or XGBModel;
- developed to be integrable with the scikit-learn ecosystem;
- effective in both classification or regression tasks;
- customizable training process, supporting early-stopping and all the other fitting options available in the standard algorithms api;
- ranking feature selection algorithms: Recursive Feature Elimination (RFE); Recursive Feature Addition (RFA); or Boruta;
- classical boosting based feature importances or SHAP feature importances (the later can be computed also on the eval_set);
- apply grid-search, random-search, or bayesian-search (from hyperopt);
- parallelized computations with joblib.

## Installation
```shell
pip install --upgrade shap-hypetune
```
lightgbm, xgboost are not needed requirements. The module depends only on NumPy, shap, scikit-learn and hyperopt. Python 3.6 or above is supported.

## Media
- [SHAP for Feature Selection and HyperParameter Tuning](https://towardsdatascience.com/shap-for-feature-selection-and-hyperparameter-tuning-a330ec0ea104)
- [Boruta and SHAP for better Feature Selection](https://towardsdatascience.com/boruta-and-shap-for-better-feature-selection-20ea97595f4a)
- [Recursive Feature Selection: Addition or Elimination?](https://towardsdatascience.com/recursive-feature-selection-addition-or-elimination-755e5d86a791)
- [Boruta SHAP for Temporal Feature Selection](https://towardsdatascience.com/boruta-shap-for-temporal-feature-selection-96a7840c7713)

## Usage
```python
from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta
```
#### Hyperparameters Tuning
```python
BoostSearch(
    estimator,                              # LGBModel or XGBModel
    param_grid=None,                        # parameters to be optimized
    greater_is_better=False,                # minimize or maximize the monitored score
    n_iter=None,                            # number of sampled parameter configurations
    sampling_seed=None,                     # the seed used for parameter sampling
    verbose=1,                              # verbosity mode
    n_jobs=None                             # number of jobs to run in parallel
)
```
#### Feature Selection (RFE)
```python
BoostRFE(  
    estimator,                              # LGBModel or XGBModel
    min_features_to_select=None,            # the minimum number of features to be selected  
    step=1,                                 # number of features to remove at each iteration  
    param_grid=None,                        # parameters to be optimized  
    greater_is_better=False,                # minimize or maximize the monitored score  
    importance_type='feature_importances',  # which importance measure to use: default or shap  
    train_importance=True,                  # where to compute the shap feature importance  
    n_iter=None,                            # number of sampled parameter configurations  
    sampling_seed=None,                     # the seed used for parameter sampling  
    verbose=1,                              # verbosity mode  
    n_jobs=None                             # number of jobs to run in parallel  
)  
```
#### Feature Selection (BORUTA)
```python
BoostBoruta(
    estimator,                              # LGBModel or XGBModel
    perc=100,                               # threshold used to compare shadow and real features
    alpha=0.05,                             # p-value levels for feature rejection
    max_iter=100,                           # maximum Boruta iterations to perform
    early_stopping_boruta_rounds=None,      # maximum iterations without confirming a feature
    param_grid=None,                        # parameters to be optimized
    greater_is_better=False,                # minimize or maximize the monitored score
    importance_type='feature_importances',  # which importance measure to use: default or shap
    train_importance=True,                  # where to compute the shap feature importance
    n_iter=None,                            # number of sampled parameter configurations
    sampling_seed=None,                     # the seed used for parameter sampling
    verbose=1,                              # verbosity mode
    n_jobs=None                             # number of jobs to run in parallel
)
```
#### Feature Selection (RFA)
```python
BoostRFA(
    estimator,                              # LGBModel or XGBModel
    min_features_to_select=None,            # the minimum number of features to be selected
    step=1,                                 # number of features to remove at each iteration
    param_grid=None,                        # parameters to be optimized
    greater_is_better=False,                # minimize or maximize the monitored score
    importance_type='feature_importances',  # which importance measure to use: default or shap
    train_importance=True,                  # where to compute the shap feature importance
    n_iter=None,                            # number of sampled parameter configurations
    sampling_seed=None,                     # the seed used for parameter sampling
    verbose=1,                              # verbosity mode
    n_jobs=None                             # number of jobs to run in parallel
)
```

Full examples in the [notebooks folder](https://github.com/cerlymarco/shap-hypetune/tree/main/notebooks).
