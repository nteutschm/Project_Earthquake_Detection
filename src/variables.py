#!/usr/bin/env python
# coding: utf-8

DATA_PATH = '/cluster/home/nteutschm/eqdetection/data/'
RANDOM_STATE = 81
# Available: IsolationForest HistGradientBoosting RandomForest XGBoost
MODEL_TYPE = 'XGBoost'

MODEL_PATH = f'/cluster/scratch/nteutschm/eqdetection/models/{MODEL_TYPE}.pkl'
PREDICTIONS_PATH = f'/cluster/scratch/nteutschm/eqdetection/predictions/{MODEL_TYPE}.csv'
TEST_LABELS_PATH = f'/cluster/scratch/nteutschm/eqdetection/test_labels/{MODEL_TYPE}.csv'
FEATURES_PATH = f'/cluster/scratch/nteutschm/eqdetection/features/{MODEL_TYPE}'
STATION_NAMES = f'/cluster/scratch/nteutschm/eqdetection/stations/{MODEL_TYPE}.csv'
PLOTS_PATH = f'/cluster/scratch/nteutschm/eqdetection/plots/{MODEL_TYPE}'
CLEANED_DATA_PATH = f'/cluster/scratch/nteutschm/eqdetection/data/dfs.pkl'
STUDIES = f'/cluster/scratch/nteutschm/eqdetection/studies/{MODEL_TYPE}.pkl'
GEOMETRIES_PATH = "/cluster/home/nteutschm/eqdetection/geometries.json"

# Stores all print information
LOG_FILE = f'/cluster/home/nteutschm/eqdetection/logs/{MODEL_TYPE}_logs.txt'

LOAD_MODEL = True # If already trained model is saved under MODEL_PATH, it can be loaded if set to True to skip the entire training process

#columns to include in creating the chunks, (offset and decay not really necessary, as crucial information already present in labels)
#available: ['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU', 'latitude', 'cos_longitude', 'sin_longitude', 'height', 'offset_value', 'offset_error', 'decay_value', 'decay_error', 'decay_tau', 'decay_type']
USED_COLS = ['N', 'E', 'U', 'latitude', 'cos_longitude', 'sin_longitude']

OPTIMAL_PARAMS = True # If optimal parametrs should be used to train the model, or the parameters should be tuned (set to False)

# How much percentage (expressed between 0 and 1) of the majority class the minority classes should be (lower end, upper end)
OVERSAMPLING_PERCENTAGES = (0.3, 0.5)

# How many days before the first and after the last earthquake should be selected (lower end, upper end)
NBR_OF_DAYS = (182, 366)

# How big each chunk should be
CHUNK_SIZE = 21

# If OPTIMAL_PARAMS is True, these parameters are used for the training process:
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

BEST_PARAMS_XGBOOST = {'n_estimators': 262, 
                       'max_depth': 23, 
                       'learning_rate': 0.028080076902022896, 
                       'subsample': 0.887455634212151, 
                       'colsample_bytree': 0.5404440391799717, 
                       'gamma': 7.819897209633202, 
                       'min_child_weight': 1, 
                       'max_delta_step': 6, 
                       'reg_alpha': 0.5938051232258645, 
                       'reg_lambda': 1.4497126942471206,
                       'objective':'multi:softprob',
                       'eval_metric':'mlogloss'}