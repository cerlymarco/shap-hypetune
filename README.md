# shap-hypetune
A python package for simultaneous Hyperparameters Tuning and Features Selection for Gradient Boosting Models.

## Overview
Hyperparameters tuning and features selection are two common steps in every machine learning pipeline. Most of the time they are computed separately and independently. This may result in suboptimal performances and in a more time expensive process.

shap-hypetune aims to combine hyperparameters tuning and features selection in a single pipeline optimizing the optimal number of features while searching for the optimal parameters configuration. Hyperparameters Tuning or Features Selection can also be carried out as standalone operations.

**shap-hypetune main features:**

- designed for gradient boosting models, as LGBModel or XGBModel;
- effective in both classification or regression tasks;
- customizable training process, supporting early-stopping and all the other fitting options available in the standard algorithms api;
- ranking feature selection algorithms: Recursive Feature Elimination (RFE) or Boruta;
- classical boosting based feature importances or SHAP feature importances (the later can be computed also on the eval_set);
- apply grid-search or random-search.

## Installation
```shell
pip install --upgrade shap-hypetune
```
lightgbm, xgboost are not needed requirements. The module depends only on NumPy and shap. Python 3.6 or above is supported.

## Media
- [SHAP for Feature Selection and HyperParameter Tuning](https://towardsdatascience.com/shap-for-feature-selection-and-hyperparameter-tuning-a330ec0ea104)
- [Boruta and SHAP for better Feature Selection](https://towardsdatascience.com/boruta-and-shap-for-better-feature-selection-20ea97595f4a)

## Usage
```python
from scipy import stats
from lightgbm import LGBMClassifier
from shaphypetune import BoostSearch, BoostRFE, BoostBoruta
```

#### Only Hyperparameters Tuning
- GRID-SEARCH
```python
param_grid = {'n_estimators': 150,
    	      'learning_rate': [0.2, 0.1],
              'num_leaves': [25, 30, 35],
    	      'max_depth': [10, 12]}

model = BoostSearch(LGBMClassifier(), param_grid=param_grid)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
- RANDOM-SEARCH
```python
param_dist = {'n_estimators': 150,
    	      'learning_rate': stats.uniform(0.09, 0.25),
    	      'num_leaves': stats.randint(20,40),
    	      'max_depth': [10, 12]}

model = BoostSearch(LGBMClassifier(), param_grid=param_dist, n_iter=10, sampling_seed=0)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
#### Only Features Selection
- RFE
```python
model = BoostRFE(LGBMClassifier(),
                 min_features_to_select=1, step=1)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
- Boruta
```python
model = BoostBoruta(LGBMClassifier(),
                    max_iter=100, perc=100)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
#### Only Features Selection with SHAP
- RFE with SHAP
```python
model = BoostRFE(LGBMClassifier(), 
                 min_features_to_select=1, step=1,
                 importance_type='shap_importances', train_importance=False)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
- Boruta with SHAP
```python
model = BoostBoruta(LGBMClassifier(),
                    max_iter=100, perc=100,
                    importance_type='shap_importances', train_importance=False)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
#### Hyperparameters Tuning + Features Selection
- RANDOM-SEARCH + RFE
```python
param_dist = {'n_estimators': 150,
    	      'learning_rate': stats.uniform(0.09, 0.25),
    	      'num_leaves': stats.randint(20,40),
    	      'max_depth': [10, 12]}

model = BoostRFE(LGBMClassifier(), param_grid=param_dist, n_iter=10, sampling_seed=0,
                 min_features_to_select=1, step=1)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
- RANDOM-SEARCH + Boruta
```python
param_dist = {'n_estimators': 150,
    	      'learning_rate': stats.uniform(0.09, 0.25),
    	      'num_leaves': stats.randint(20,40),
    	      'max_depth': [10, 12]}

model = BoostBoruta(LGBMClassifier(), param_grid=param_dist, n_iter=10, sampling_seed=0,
                    max_iter=100, perc=100)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
#### Hyperparameters Tuning + Features Selection with SHAP
- GRID-SEARCH + RFE with SHAP
```python
param_grid = {'n_estimators': 150,
    	      'learning_rate': [0.2, 0.1],
              'num_leaves': [25, 30, 35],
    	      'max_depth': [10, 12]}

model = BoostRFE(LGBMClassifier(), param_grid=param_grid, 
                 min_features_to_select=1, step=1,
                 importance_type='shap_importances', train_importance=False)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```
- GRID-SEARCH + Boruta with SHAP
```python
param_grid = {'n_estimators': 150,
    	      'learning_rate': [0.2, 0.1],
              'num_leaves': [25, 30, 35],
    	      'max_depth': [10, 12]}

model = BoostBoruta(LGBMClassifier(), param_grid=param_grid,
                    max_iter=100, perc=100,
                    importance_type='shap_importances', train_importance=False)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=6, verbose=0)
```

All the examples are reproducible in regression contexts and with XGBModel.

More examples in the [notebooks folder](https://github.com/cerlymarco/shap-hypetune/tree/main/notebooks).

**All the available estimators are fully integrable with sklearn (see [here](https://github.com/cerlymarco/shap-hypetune/blob/main/notebooks/sklearn-wrapper.ipynb)).**
