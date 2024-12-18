#!/usr/bin/env python
# coding: utf-8


# Path to the data
DATA_PATH = '/cluster/home/nteutschm/eqdetection/data/'
RANDOM_STATE = 81
# Specify the model to use. Options: 'IsolationForest', 'HistGradientBoosting', 'RandomForest', 'XGBoost'
MODEL_TYPE = 'XGBoost'

# If True, train the model on North American stations and test on New Zealand stations
# Requires ['latitude', 'cos_longitude', 'sin_longitude'] in USED_COLS for regional filtering
TRAIN_NA = True

# Whether to exclude the location from the training process if TRAIN_NA is True. Keep in mind that the location is still needed in USED_COLS to do the regional filtering
EXCLUDE_LOC = True

if TRAIN_NA:
    na='_na'
else:
    na=''

# Various paths to store the output information
MODEL_PATH = f'/cluster/scratch/nteutschm/eqdetection/models/{MODEL_TYPE}{na}.pkl'
PREDICTIONS_PATH = f'/cluster/scratch/nteutschm/eqdetection/predictions/{MODEL_TYPE}{na}.csv'
TEST_LABELS_PATH = f'/cluster/scratch/nteutschm/eqdetection/test_labels/{MODEL_TYPE}{na}.csv'
FEATURES_PATH = f'/cluster/scratch/nteutschm/eqdetection/features/{MODEL_TYPE}{na}'
STATION_NAMES = f'/cluster/scratch/nteutschm/eqdetection/stations/{MODEL_TYPE}{na}.csv'
PLOTS_PATH = f'/cluster/scratch/nteutschm/eqdetection/plots/{MODEL_TYPE}{na}'
CLEANED_DATA_PATH = f'/cluster/scratch/nteutschm/eqdetection/data/dfs{na}.pkl'
STUDIES = f'/cluster/scratch/nteutschm/eqdetection/studies/{MODEL_TYPE}{na}.pkl'
GEOMETRIES_PATH = f"/cluster/home/nteutschm/eqdetection/geometries{na}.json"

# File to store log output
LOG_FILE = f'/cluster/home/nteutschm/eqdetection/logs/{MODEL_TYPE}_logs{na}.txt'

# Set to True to load a pre-trained model from MODEL_PATH and skip training
LOAD_MODEL = False

# Columns to include when creating data chunks. Some fields (e.g., 'offset', 'decay') may not be necessary as label data already contains the relevant information
# Options: ['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU', 'latitude', 'cos_longitude', 'sin_longitude', 'height', 'offset_value', 'offset_error', 'decay_value', 'decay_error', 'decay_tau', 'decay_type']
# Recommended order if including location data: ['latitude', 'cos_longitude', 'sin_longitude', 'height'] (especially if TRAIN_NA = True)
USED_COLS = ['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU', 'latitude', 'cos_longitude', 'sin_longitude']

# Use pre-defined optimal parameters for training. If False, parameters will be tuned
OPTIMAL_PARAMS = True

# Set to True to load an existing Optuna study instead of optimizing again
LOAD_STUDY = False

# Oversampling: Adjusts the minority class proportion relative to the majority class, specified as (lower_bound, upper_bound)
OVERSAMPLING_PERCENTAGES = (0.3, 0.5)

# Number of days to include before the first and after the last earthquake (lower_bound, upper_bound)
NBR_OF_DAYS = (182, 366)

# Size of each data chunk
CHUNK_SIZE = 21

# Optimal parameters for each model type (used if OPTIMAL_PARAMS = True):

BEST_PARAMS_RANDOM_FOREST = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'bootstrap': True,
    'max_samples': 0.8
}

BEST_PARAMS_ISOLATION_FOREST = {
    'n_estimators': 300,
    'max_samples': 0.8,
    'contamination': 0.001
}

BEST_PARAMS_HIST_GRADIENT_BOOSTING = {
    'learning_rate': 0.2,
    'max_iter': 300,
    'max_depth': 15,
    'early_stopping': True,
    'n_iter_no_change': 7,
    'validation_fraction': 0.1
}

BEST_PARAMS_XGBOOST = {'n_estimators': 883, 
                       'max_depth': 42, 
                       'learning_rate': 0.005734244485307966, 
                       'subsample': 0.4022912715125908, 
                       'colsample_bytree': 0.5215518393729897, 
                       'gamma': 3.380498010359085, 
                       'min_child_weight': 16, 
                       'max_delta_step': 9, 
                       'reg_alpha': 0.707763339472735, 
                       'reg_lambda': 2.959082698191286,
                       'objective':'multi:softprob',
                       'eval_metric':'mlogloss'}