#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier, callback
from sklearn.metrics import f1_score
import optuna
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Input, GlobalAveragePooling1D, Conv1D, BatchNormalization, ReLU
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from datetime import datetime
from tensorflow.keras.optimizers import Adam



# Runs in the backgound:
import optuna.integration # pip install optuna-integration[xgboost]

from variables_tcn import *
from plots import plot_optimization, plot_eval_metrics

def print_callback(study, trial):
    """
    A callback function to print useful information regarding the hyperparameter optimization progress. 
    It outputs the current trial value and parameters, as well as the best value and parameters found so far in the study.

    Parameters:
    study (optuna.Study): The optimization study object containing the best trial.
    trial (optuna.Trial): The current trial object being evaluated in the optimization.

    Returns:
    None
    """
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    print('-----------------------------------------------------------------------')

def random_forest(weights):
    """
    Returns a Random Forest classifier model configured with the optimal parameters.

    This function initializes a RandomForestClassifier using pre-defined optimal 
    parameters stored in the BEST_PARAMS_RANDOM_FOREST variable. These parameters 
    are expected to be set prior to calling this function. It also accepts class weights 
    as a parameter to handle imbalanced classes.

    Parameters:
    weights (dict): The class weights to be applied to the model. 

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
    These parameters are expected to be set prior to calling this function. It also accepts class weights 
    as a parameter to handle imbalanced classes.

    Parameters:
    weights (dict): The class weights to be applied to the model. 

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
    
    Parameters:
    y_train (pd.Series): The training target variable containing the class labels.
                         The number of unique classes in y_train is used to set the 
                         num_class parameter of the XGBoost model.

    Returns:
    XGBClassifier: An XGBoost model with optimal settings.
    """
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
        num_class=len(set(y_train)), 
        reg_alpha=BEST_PARAMS_XGBOOST['reg_alpha'],
        reg_lambda=BEST_PARAMS_XGBOOST['reg_lambda']
    )
    return model

def optimize_random_forest(X_train, y_train, X_eval, y_eval, weights):
    """
    Optimizes and returns a Random Forest model using hyperparameter optimization (Optuna).

    This function optimizes the hyperparameters for a Random Forest model using 
    Optuna. The function checks whether pre-trained models or optimal parameters 
    should be used or if hyperparameter optimization should be performed. The 
    optimization is based on the F1-score (macro average). If a previously 
    saved study exists, it will be loaded; otherwise, a new study will be created 
    and optimized over 50 trials. Additionally, optimization plots are generated. 

    Parameters:
    X_train (pd.DataFrame): The training features used to train the model.
    y_train (pd.Series): The training target variable (labels).
    X_eval (pd.DataFrame): The evaluation features used to evaluate the model during optimization.
    y_eval (pd.Series): The evaluation target variable (labels).
    weights (dict): The class weights to handle imbalanced classes.

    Returns:
    RandomForestClassifier: The best-trained Random Forest model with the optimal parameters.
    """

    if LOAD_MODEL:
        print(f'Loading Random Forest model from: {MODEL_PATH}')
        return joblib.load(MODEL_PATH)
    
    if OPTIMAL_PARAMS:
        print(f'Training RandomForest model using the specified optimal parameters: {BEST_PARAMS_RANDOM_FOREST}')
        model = random_forest(weights)
        model.fit(X_train, y_train)
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
    
    if LOAD_STUDY:
        print('Loading previous study')
        study = joblib.load(STUDIES)
    else: 
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=50, callbacks=[print_callback], gc_after_trial=True)
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
    X_train (pd.DataFrame): The training set feature matrix.
    y_train (pd.Series): The training set target vector (required for compatibility).

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
    Optimizes a HistGradientBoosting classifier using grid search with cross-validation and 
    implementing OneVsRestClassifier. 

    The function first checks if a pre-trained model should be loaded based on the 
    LOAD_MODEL flag. If this flag is set to True, it will load a model from the 
    specified MODEL_PATH. If the OPTIMAL_PARAMS flag is True, it will use pre-defined 
    optimal parameters for training. Otherwise, it will perform grid search to identify 
    the best hyperparameters.

    Parameters:
    X_train (pd.DataFrame): The training set feature matrix.
    y_train (pd.Series): The training set target vector.
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
    
def optimize_xgboost(X_train, y_train, X_eval, y_eval, weights):
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
    
    """
    Optimizes an XGBoost classifier using Optuna for hyperparameter tuning.

    This function first checks if a pre-trained model should be loaded based on the 
    LOAD_MODEL flag. If this flag is set to True, it will load a model from the 
    specified MODEL_PATH. If the OPTIMAL_PARAMS flag is True, it will use pre-defined 
    optimal parameters for training. Otherwise, it will perform Optuna optimization 
    to identify the best hyperparameters. The optimization is based on the macro F1-score.
    The optimization includes early stopping and pruning, followed by a visualization of the process. 

    Parameters:
    X_train (pd.DataFrame): The training set feature matrix.
    y_train (pd.Series): The training set target vector.
    X_eval (pd.DataFrame): The evaluation set feature matrix.
    y_eval (pd.Series): The evaluation set target vector.
    weights (dict): The class weights to handle imbalanced classes.

    Returns:
    XGBClassifier: The best XGBoost model after optimization.
    """

    if LOAD_MODEL:
        print(f'Loading XGBoost model from: {MODEL_PATH}')
        return joblib.load(MODEL_PATH)
    
    if OPTIMAL_PARAMS:
        print(f'Training XGBoost model using the specified optimal parameters: {BEST_PARAMS_XGBOOST}')
        model = xgboost(y_train)
        model.fit(X_train, y_train, sample_weight=np.array([weights[label] for label in y_train]), verbose=True)
        return model
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 150, 750)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.4, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
        gamma = trial.suggest_float('gamma', 0.0, 10)
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
        
        xgb.fit(X_train, y_train, sample_weight=np.array([weights[label] for label in y_train]), 
                eval_set=[(X_eval, y_eval)],
                verbose=False)

        y_pred = xgb.predict(X_eval)
        f1_macro = f1_score(y_eval, y_pred, average='macro' )
        return f1_macro
    
    if LOAD_STUDY:
        print('Loading previous study')
        study = joblib.load(STUDIES)
    else:
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=50, callbacks=[print_callback], gc_after_trial=True)
        joblib.dump(study, STUDIES)
    
    print("Best XGBoost parameters found: ", study.best_params)
    
    plot_optimization(study, ('learning_rate', 'max_depth'))
    
    cb = callback.EarlyStopping(rounds=10,
            min_delta=1e-4,
            save_best=True,
            maximize=False,
            data_name="validation_1",
            metric_name="mlogloss")
    
    best_params = study.best_params
    best_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=len(set(y_train)),
        random_state=RANDOM_STATE,
        callbacks=[cb],
        **best_params
    )
    
    best_model.fit(X_train, y_train, sample_weight=np.array([weights[label] for label in y_train]), eval_set=[(X_train, y_train), (X_eval, y_eval)], verbose=True)
    eval_results = best_model.evals_result()
    plot_eval_metrics(eval_results, 'eval_scores')
    return best_model

@tf.keras.utils.register_keras_serializable(package="Custom")
def distance_penalized_mse(y_true, y_pred):
    # Get the true and predicted offset positions by finding the index of the max value in each sequence
    true_position = tf.argmax(y_true, axis=-1)
    pred_position = tf.argmax(y_pred, axis=-1)
    # Calculate the absolute difference between true and predicted positions
    distances = tf.abs(tf.cast(true_position - pred_position, tf.float32))
    # Apply a penalty proportional to the distance (increase MSE for farther predictions)
    penalties = distances + 1  # Adding 1 to avoid multiplying by zero
    # Compute the mean squared error and apply the penalty
    mse = tf.square(y_true - y_pred)
    penalized_mse = mse * tf.expand_dims(penalties, axis=-1)  # Expand dimensions for broadcasting
    
    return tf.reduce_mean(penalized_mse)

@tf.keras.utils.register_keras_serializable(package="Custom")
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        cross_entropy = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = K.pow((1 - p_t), gamma)
        alpha_weight_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_weight_factor * modulating_factor * cross_entropy
    return loss

def build_localization_model_tcn(input_shape, depth=3, filters=32, kernel_size=3, dilation_rate=1):
    input_layer = Input(shape=input_shape)
    x = input_layer

    # Temporal Convolutional Layers
    for i in range(depth):  # Three layers, can adjust
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        dilation_rate *= 2  # Increase dilation rate exponentially

    # Global pooling to reduce to single vector
    x = GlobalAveragePooling1D()(x)

    # Output layer for localization
    localization_output = Dense(CHUNK_SIZE, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=localization_output)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=distance_penalized_mse, metrics=['accuracy'])
    return model

def build_binary_model_tcn(input_shape, depth=5, filters=64, kernel_size=5, dilation_rate=1):
    input_layer = Input(shape=input_shape)
    x = input_layer

    for i in range(depth):  # Increase the number of layers
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        dilation_rate *= 2

    x = GlobalAveragePooling1D()(x)
    binary_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=binary_output)
    optimizer = Adam(learning_rate=1e-4)
    # Use the focal_loss function by calling it to create the actual loss function
    model.compile(optimizer=optimizer, loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

    return model


def optimize_tcn_binary(X_train, y_bin_train, X_eval, y_bin_eval):
    """
    Optimize or load a TCN model for binary classification with the option to use predefined parameters.

    Args:
        X_train (np.ndarray): Training input data.
        y_bin_train (np.ndarray): Binary training labels.
        X_eval (np.ndarray): Evaluation input data for parameter selection.
        y_bin_eval (np.ndarray): Binary evaluation labels for parameter selection.

    Returns:
        Model: Trained TCN Binary model.
    """


    if LOAD_BIN_MODEL:
        print(f"Loading Binary TCN model from: {MODEL_BIN_PATH}")
        model = load_model(MODEL_BIN_PATH)
        return model

    if OPTIMAL_BIN_PARAMS:
        print(f"Training Binary TCN model using the specified optimal parameters: {BEST_PARAMS_TCN_BINARY}")
        reduce_lr_binary = ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=5, min_lr=1e-6)
        early_stopping_binary = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        params = BEST_PARAMS_TCN_BINARY
        model = build_binary_model_tcn(
            input_shape=(CHUNK_SIZE,X_train.shape[2]),
            depth=params["depth"],
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            dilation_rate=params["dilation_rate"],
        )
        model.fit(X_train, y_bin_train, epochs=EPOCHS, batch_size=params["batch_size"], validation_split=0.2, callbacks=[early_stopping_binary, reduce_lr_binary], verbose=2)
        if MODEL_BIN_PATH:
            model.save(MODEL_BIN_PATH)
            print(f"Binary classification model saved to {MODEL_BIN_PATH}")
        return model

    # If not optimal params, perform a grid search for hyperparameter optimization
    print("Performing parameter search for Binary TCN...")
    param_grid = {
        "depth": [3,5],
        "filters": [32,64,128],
        "kernel_size": [3,5],
        #Fine tune dilution: Dilution 2 only good if Dilution 1 already performs well
        "dilation_rate": [1], 
        "batch_size": [32,64], 
    }
    best_model = None
    best_params = {}
    best_macro_f1 = -1
    current_time = datetime.now()
    print("Starting search at:", current_time)

    # Simple manual grid search (TensorFlow models are not compatible with sklearn's GridSearchCV)
    for depth in param_grid["depth"]:
        for filters in param_grid["filters"]:
            for kernel_size in param_grid["kernel_size"]:
                for dilation_rate in param_grid["dilation_rate"]:
                    for batch_size in param_grid["batch_size"]:
                        print(f"Training TCN with depth={depth}, filters={filters}, kernel_size={kernel_size}, dilation_rate={dilation_rate}, batch size={batch_size}")
                        model = build_binary_model_tcn(
                            input_shape=(CHUNK_SIZE,X_train.shape[2]),
                            depth=depth,
                            filters=filters,
                            kernel_size=kernel_size,
                            dilation_rate=dilation_rate,
                        )
                        reduce_lr_binary = ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=5, min_lr=1e-6)
                        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                        model.fit(
                            X_train, y_bin_train,
                            validation_split=0.2,
                            epochs=20, batch_size=batch_size,
                            callbacks=[early_stopping, reduce_lr_binary],
                            verbose=0
                        )

                        # Evaluate macro F1-score on the evaluation set
                        y_eval_pred = (model.predict(X_eval, verbose=0) > 0.5).astype(int)
                        eval_macro_f1 = f1_score(y_bin_eval, y_eval_pred, average="macro")
                        current_time = datetime.now()
                        print(f"Macro F1 on Eval Set: {eval_macro_f1:.4f}")
                        print("Found at:", current_time)
                        print(" ")

                        # Track the best model and parameters
                        if eval_macro_f1 > best_macro_f1:
                            best_macro_f1 = eval_macro_f1
                            best_model = model
                            best_params = {
                                "depth": depth,
                                "filters": filters,
                                "kernel_size": kernel_size,
                                "dilation_rate": dilation_rate,
                                "batch_size" : batch_size
                            }

    print(f"Best Binary TCN parameters found: {best_params}, Macro F1: {best_macro_f1:.4f}")

    # Save the best model if needed
    if MODEL_BIN_PATH and best_model:
        best_model.save(MODEL_BIN_PATH)
        print(f"Best model saved to {MODEL_BIN_PATH}")

    return best_model


def optimize_tcn_loc(X_train, y_loc_train, X_eval, y_loc_eval, y_bin_eval):
    """
    Optimize or load a TCN model for binary classification with the option to use predefined parameters.

    Args:
        X_train (np.ndarray): Training input data.
        y_loc_train (np.ndarray): Localization training labels.
        X_eval (np.ndarray): Evaluation input data for parameter selection.
        y_loc_eval (np.ndarray): Localization evaluation labels for parameter selection.

    Returns:
        Model: Trained TCN Localization model.
    """


    if LOAD_LOC_MODEL:
        print(f"Loading TCN Localization model from: {MODEL_LOC_PATH}")
        model = load_model(MODEL_LOC_PATH, custom_objects={"distance_penalized_mse": distance_penalized_mse,"build_localization_model_tcn": build_localization_model_tcn})
        return model

    if OPTIMAL_LOC_PARAMS:
        print(f"Training TCN Localization model using the specified optimal parameters: {BEST_PARAMS_TCN_LOCALIZATION}")
        reduce_lr_loc = ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=5, min_lr=1e-6)
        early_stopping_loc = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        params = BEST_PARAMS_TCN_LOCALIZATION
        model = build_localization_model_tcn(
            input_shape=(CHUNK_SIZE,X_train.shape[2]),
            depth=params["depth"],
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            dilation_rate=params["dilation_rate"],
        )
        model.fit(X_train, y_loc_train, epochs=EPOCHS, batch_size=params["batch_size"], validation_split=0.2, callbacks=[early_stopping_loc, reduce_lr_loc], verbose=2)
        if MODEL_LOC_PATH:
            model.save(MODEL_LOC_PATH)
            print(f"Localization classification model saved to {MODEL_LOC_PATH}")
        return model

    # If not optimal params, perform a grid search for hyperparameter optimization
    print("Performing parameter search for Localization TCN...")
    param_grid = {
        "depth": [5,6,7],
        "filters": [128],
        "kernel_size": [5,7],
        "dilation_rate": [1], 
        "batch_size": [64], 
    }
    best_model = None
    best_params = {}
    best_macro_f1 = -1
    

    # Simple manual grid search (TensorFlow models are not compatible with sklearn's GridSearchCV)
    for filters in param_grid["filters"]:
        for kernel_size in param_grid["kernel_size"]:
            for dilation_rate in param_grid["dilation_rate"]:
                for depth in param_grid["depth"]:
                    for batch_size in param_grid["batch_size"]:
                        print(f"Training TCN with filters={filters}, kernel_size={kernel_size}, dilation_rate={dilation_rate}")
                        model = build_localization_model_tcn(
                            input_shape=(CHUNK_SIZE,X_train.shape[2]),
                            depth=depth,
                            filters=filters,
                            kernel_size=kernel_size,
                            dilation_rate=dilation_rate,
                        )
                        reduce_lr_binary = ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=5, min_lr=1e-6)
                        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                        model.fit(
                            X_train, y_loc_train,
                            validation_split=0.2,
                            epochs=20, batch_size=batch_size,
                            callbacks=[early_stopping, reduce_lr_binary],
                            verbose=0
                        )

                        # Filter samples with offsets for localization training
                        offset_indices = np.where(y_bin_eval == 1)[0]
                        X_eval_localization = X_eval[offset_indices]
                        y_eval_localization = y_loc_eval[offset_indices]
                        y_eval_localization = np.argmax(y_eval_localization, axis=1)

                        # Evaluate macro F1-score on the evaluation set
                        eval_pred = model.predict(X_eval_localization, verbose=0)
                        eval_pred = np.argmax(eval_pred, axis=1)


                        #### np.argmax von eval pred??

                        eval_macro_f1 = f1_score(y_eval_localization, eval_pred, average="macro")

                        current_time = datetime.now()
                        print(f"Macro F1 on Eval Set: {eval_macro_f1:.4f}")
                        print("Found at:", current_time)

                        # Track the best model and parameters
                        if eval_macro_f1 > best_macro_f1:
                            best_macro_f1 = eval_macro_f1
                            best_model = model
                            best_params = {
                                "depth": depth,
                                "filters": filters,
                                "kernel_size": kernel_size,
                                "dilation_rate": dilation_rate,
                                "batch_size" : batch_size
                            }
                    

    print(f"Best TCN Localization parameters found: {best_params}, Macro F1: {best_macro_f1:.4f}")

    # Save the best model if needed
    if MODEL_LOC_PATH and best_model:
        best_model.save(MODEL_LOC_PATH)
        print(f"Best model saved to {MODEL_LOC_PATH}")

    return best_model
