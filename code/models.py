#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier, callback
from sklearn.metrics import f1_score
import optuna

# Runs in the backgound:
import optuna.integration # pip install optuna-integration[xgboost]

from variables import *
from plots import plot_optimization, plot_eval_metrics

def print_callback(study, trial):
    '''Print some useful information regarding the hyperparameter optimization'''
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    print('-----------------------------------------------------------------------')

def random_forest(weights):
    """
    Returns a Random Forest classifier model configured with the optimal parameters.

    This function initializes a RandomForestClassifier using pre-defined optimal 
    parameters stored in the BEST_PARAMS_RANDOM_FOREST variable. These parameters 
    are expected to be set prior to calling this function.

    Returns:
    RandomForestClassifier: A Random Forest model with optimal settings.
    """
    model = RandomForestClassifier(
        n_estimators=BEST_PARAMS_RANDOM_FOREST['n_estimators'],
        max_depth=BEST_PARAMS_RANDOM_FOREST['max_depth'],
        max_features=BEST_PARAMS_RANDOM_FOREST['max_features'],
        min_samples_split=BEST_PARAMS_RANDOM_FOREST['min_samples_split'],
        min_samples_leaf=BEST_PARAMS_RANDOM_FOREST['min_samples_leaf'],
        class_weight=weights,
        random_state=RANDOM_STATE,
        bootstrap=BEST_PARAMS_RANDOM_FOREST['bootstrap'],
        max_samples=BEST_PARAMS_RANDOM_FOREST['max_samples']
    )
    return model

def isolation_forest():
    """
    Returns an Isolation Forest model configured with the optimal parameters.

    This function initializes an IsolationForest using pre-defined optimal 
    parameters stored in the BEST_PARAMS_ISOLATION_FOREST variable. These parameters 
    are expected to be set prior to calling this function.

    Returns:
    IsolationForest: An Isolation Forest model with optimal settings.
    """
    model = IsolationForest(
        n_estimators=BEST_PARAMS_ISOLATION_FOREST['n_estimators'],
        max_samples=BEST_PARAMS_ISOLATION_FOREST['max_samples'],
        contamination=BEST_PARAMS_ISOLATION_FOREST['contamination'],
        random_state=RANDOM_STATE
    )
    return model

def hist_gradient_boosting(weights):
    """
    Returns a HistGradientBoostingClassifier model configured with the optimal parameters.

    This function initializes a HistGradientBoostingClassifier using pre-defined 
    optimal parameters stored in the BEST_PARAMS_HIST_GRADIENT_BOOSTING variable. 
    These parameters are expected to be set prior to calling this function.

    Returns:
    HistGradientBoostingClassifier: A HistGradientBoosting model with optimal settings.
    """
    model = HistGradientBoostingClassifier(
        learning_rate=BEST_PARAMS_HIST_GRADIENT_BOOSTING['learning_rate'],
        max_iter=BEST_PARAMS_HIST_GRADIENT_BOOSTING['max_iter'],
        max_depth=BEST_PARAMS_HIST_GRADIENT_BOOSTING['max_depth'],
        class_weight=weights,
        random_state=RANDOM_STATE,
        early_stopping=BEST_PARAMS_HIST_GRADIENT_BOOSTING['early_stopping'],
        n_iter_no_change=BEST_PARAMS_HIST_GRADIENT_BOOSTING['n_iter_no_change'],
        validation_fraction=BEST_PARAMS_HIST_GRADIENT_BOOSTING['validation_fraction']
    )
    return model

def xgboost(y_train):
    """
    Returns an XGBoost classifier model configured with the optimal parameters.

    This function initializes an XGBClassifier using pre-defined optimal 
    parameters stored in the BEST_PARAMS_XGBOOST variable. These parameters 
    are expected to be set prior to calling this function.

    Returns:
    XGBClassifier: An XGBoost model with optimal settings.
    """
    cb = callback.EarlyStopping(rounds=5,
            min_delta=1e-4,
            save_best=True,
            maximize=False,
            data_name="validation_1",
            metric_name="mlogloss")
    model = XGBClassifier(
        n_estimators=BEST_PARAMS_XGBOOST['n_estimators'],
        max_depth=BEST_PARAMS_XGBOOST['max_depth'],
        learning_rate=BEST_PARAMS_XGBOOST['learning_rate'],
        subsample=BEST_PARAMS_XGBOOST['subsample'],
        colsample_bytree=BEST_PARAMS_XGBOOST['colsample_bytree'],
        gamma=BEST_PARAMS_XGBOOST['gamma'],
        min_child_weight=BEST_PARAMS_XGBOOST['min_child_weight'],
        max_delta_step=BEST_PARAMS_XGBOOST['max_delta_step'],
        random_state=RANDOM_STATE,
        objective=BEST_PARAMS_XGBOOST['objective'],
        eval_metric=BEST_PARAMS_XGBOOST['eval_metric'],
        num_class=len(set(y_train)), 
        callbacks=[cb], 
        reg_alpha=BEST_PARAMS_XGBOOST['reg_alpha'],
        reg_lambda=BEST_PARAMS_XGBOOST['reg_lambda']
    )
    return model

def optimize_random_forest(X_train, y_train, X_eval, y_eval, weights):
    """
    Optimizes a RandomForest classifier using Optuna for hyperparameter tuning.

    Parameters:
    X_train (DataFrame): The training set feature matrix.
    y_train (Series): The training set target vector.
    X_eval (DataFrame): The evaluation set feature matrix.
    y_eval (Series): The evaluation set target vector.

    Returns:
    RandomForestClassifier: The best RandomForest model after optimization.
    """

    if LOAD_MODEL:
        print(f'Loading Random Forest model from: {MODEL_PATH}')
        return joblib.load(MODEL_PATH)
    
    if OPTIMAL_PARAMS:
        print(f'Training RandomForest model using the specified optimal parameters: {BEST_PARAMS_RANDOM_FOREST}')
        model = random_forest(weights)
        X_combined = pd.concat([X_train, X_eval])
        y_combined = pd.concat([y_train, y_eval])
        model.fit(X_combined, y_combined)
        return model
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 600)
        max_depth = trial.suggest_int('max_depth', 5, 35)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        max_samples = trial.suggest_float('max_samples', 0.5, 1.0) if bootstrap else None
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=RANDOM_STATE,
            class_weight=weights,
            max_samples=max_samples,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)
        f1_macro = f1_score(y_eval, y_pred, average='macro')
        return f1_macro
    
    # Create an Optuna study and optimize
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=30, callbacks=[print_callback], gc_after_trial=True)
    
    joblib.dump(study, STUDIES)
    print("Best Random Forest parameters found: ", study.best_params)
    
    plot_optimization(study, ('n_estimators', 'max_depth'))
    
    best_params = study.best_params
    best_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=weights,
        **best_params
    )
    X_combined = pd.concat([X_train, X_eval])
    y_combined = pd.concat([y_train, y_eval])
    
    best_model.fit(X_combined, y_combined)
    return best_model

def optimize_isolation_forest(X_train, y_train):
    """
    Optimizes an Isolation Forest model for anomaly detection using grid search with cross-validation.

    The function first checks if a pre-trained model should be loaded based on the 
    LOAD_MODEL flag. If this flag is set to True, it will load a model from the 
    specified MODEL_PATH. If the OPTIMAL_PARAMS flag is True, it will use pre-defined 
    optimal parameters for training. Otherwise, it will perform grid search to identify 
    the best hyperparameters.

    Parameters:
    X_train (DataFrame): The training set feature matrix.
    y_train (Series): The training set target vector (required for compatibility).

    Returns:
    IsolationForest: The best Isolation Forest model after optimization.
    """
    if LOAD_MODEL:
        print(F'Loading Iolation Forest model from: {MODEL_PATH}')
        return joblib.load(MODEL_PATH)
    
    if OPTIMAL_PARAMS:
        print(f'Training Isolation Forest model using the specified optimal parameters: {BEST_PARAMS_ISOLATION_FOREST}')
        model = isolation_forest()
        return model.fit(X_train)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_samples': [0.8, 1.0],
        'contamination': [0.001, 0.002, 0.005],
        'random_state': [RANDOM_STATE]
    }
    iso_forest = IsolationForest()
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(iso_forest, param_grid, cv=stratified_cv, scoring='f1_weighted', verbose=1, n_jobs=-1, pre_dispatch='2*n_jobs')
    # Have to give the labels as input even though IsolationForest uses no labels and the input is optional as otherwise there is an error. No idea why, should not impact performance though.
    grid_search.fit(X_train, y_train)
    
    print("Best IsolationForest parameters found: ", grid_search.best_params_)
    
    return grid_search.best_estimator_

def optimize_hist_gradient_boosting(X_train, y_train, weights):
    """
    Optimizes a HistGradientBoosting classifier using grid search with cross-validation.

    The function first checks if a pre-trained model should be loaded based on the 
    LOAD_MODEL flag. If this flag is set to True, it will load a model from the 
    specified MODEL_PATH. If the OPTIMAL_PARAMS flag is True, it will use pre-defined 
    optimal parameters for training. Otherwise, it will perform grid search to identify 
    the best hyperparameters.

    Parameters:
    X_train (DataFrame): The training set feature matrix.
    y_train (Series): The training set target vector (0 for no offset, 1 for offset).
    weights (dict): Class weights to handle imbalanced classes.

    Returns:
    HistGradientBoostingClassifier: The best HistGradientBoosting model after optimization.
    """
    if LOAD_MODEL:
        print(F'Loading Hist Gradient Boosting model from: {MODEL_PATH}')
        return joblib.load(MODEL_PATH)
    
    if OPTIMAL_PARAMS:
        print(f'Training Hist Gradient Boosting model using the specified optimal parameters: {BEST_PARAMS_HIST_GRADIENT_BOOSTING}')
        model = hist_gradient_boosting(weights)
        return OneVsRestClassifier(model).fit(X_train, y_train)
    
    param_grid = {
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__max_iter': [100, 200, 300],
        'estimator__max_depth': [5, 10, 15, 30],
        'estimator__class_weight': [weights],
        'estimator__random_state': [RANDOM_STATE]
    }
        
    hgb = OneVsRestClassifier(HistGradientBoostingClassifier(early_stopping=True,
        n_iter_no_change=7,
        validation_fraction=0.1))
    
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=hgb,
        param_grid=param_grid,
        cv=stratified_cv,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1,
        pre_dispatch='2*n_jobs'
    )
    grid_search.fit(X_train, y_train)
    
    print("Best HistGradientBoosting parameters found: ", grid_search.best_params_)
    
    return grid_search.best_estimator_
    
def optimize_xgboost(X_train, y_train, X_eval, y_eval):
    """
    Optimizes an XGBoost classifier using Optuna for hyperparameter tuning.

    The function first checks if a pre-trained model should be loaded based on the 
    LOAD_MODEL flag. If this flag is set to True, it will load a model from the 
    specified MODEL_PATH. If the OPTIMAL_PARAMS flag is True, it will use pre-defined 
    optimal parameters for training. Otherwise, it will perform Optuna optimization to 
    identify the best hyperparameters.

    Parameters:
    X_train (DataFrame): The training set feature matrix.
    y_train (Series): The training set target vector.
    X_eval (DataFrame): The evaluation set feature matrix.
    y_eval (Series): The evaluation set target vector.

    Returns:
    XGBClassifier: The best XGBoost model after optimization.
    """

    if LOAD_MODEL:
        print(f'Loading XGBoost model from: {MODEL_PATH}')
        return joblib.load(MODEL_PATH)
    
    if OPTIMAL_PARAMS:
        print(f'Training XGBoost model using the specified optimal parameters: {BEST_PARAMS_XGBOOST}')
        model = xgboost(y_train)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_eval, y_eval)], verbose=True)
        eval_results = model.evals_result()
        plot_eval_metrics(eval_results, 'eval_scores')
        return model
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 800)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma = trial.suggest_float('gamma', 0, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 15)
        max_delta_step = trial.suggest_int('max_delta_step', 0, 10)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.5, 3.0)
        
        cb = callback.EarlyStopping(rounds=10,
            min_delta=1e-4,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name="mlogloss")
        
        pruning = optuna.integration.XGBoostPruningCallback(trial, f"validation_0-mlogloss")
        
        xgb = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=len(set(y_train)),
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            callbacks=[pruning, cb],
            random_state=RANDOM_STATE,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda
        )
        
        xgb.fit(X_train, y_train,
                eval_set=[(X_eval, y_eval)],
                verbose=False)

        # Evaluate with macro F1 score to balance all classes equally
        y_pred = xgb.predict(X_eval)
        f1_macro = f1_score(y_eval, y_pred, average='macro')
        return f1_macro
    
    # Create an Optuna study and optimize
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=30, callbacks=[print_callback], gc_after_trial=True)
    
    joblib.dump(study, STUDIES)
    print("Best XGBoost parameters found: ", study.best_params)
    
    plot_optimization(study, ('learning_rate', 'max_depth'))
    
    cb = callback.EarlyStopping(rounds=10,
            min_delta=1e-4,
            save_best=True,
            maximize=False,
            data_name="validation_1",
            metric_name="mlogloss")
    
    # Train the best model on the full training set
    best_params = study.best_params
    best_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=len(set(y_train)),
        random_state=RANDOM_STATE,
        callbacks=[cb],
        **best_params
    )
    
    best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_eval, y_eval)], verbose=True)
    eval_results = best_model.evals_result()
    plot_eval_metrics(eval_results, 'eval_scores')
    return best_model