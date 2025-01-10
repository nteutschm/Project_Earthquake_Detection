#!/usr/bin/env python
# coding: utf-8

# Path to the data
DATA_PATH = '/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Data/'
RANDOM_STATE = 81
EPOCHS =  120
# Specify the model to use. Options: 'IsolationForest', 'HistGradientBoosting', 'RandomForest', 'XGBoost'
MODEL_TYPE = 'TCN_60d'
MODEL_TYPE1 = 'Optimize_bin'
MODEL_TYPE2 = 'Optimize_loc2'

# Various paths to store the output information
MODEL_BIN_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/models_bin/{MODEL_TYPE1}.keras'
MODEL_LOC_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/models_loc/{MODEL_TYPE2}.keras'
PREDICTIONS_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/predictions/{MODEL_TYPE}.csv'
TEST_LABELS_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/test_labels/{MODEL_TYPE}.csv'
FEATURES_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/features/{MODEL_TYPE}'
STATION_NAMES = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/stations/{MODEL_TYPE}.csv'
PLOTS_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/plots/{MODEL_TYPE}'
TWO_STEPS_PLOTS_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/Plots/{MODEL_TYPE}'
CLEANED_DATA_PATH = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/Data/dfs.pkl'
STUDIES = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/studies/{MODEL_TYPE}.pkl'
GEOMETRIES_PATH = "/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/geometries.json"

# File to store log output
LOG_FILE = f'/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/Storage/New_Storage/{MODEL_TYPE}_logs.txt'

# Set to True to load a pre-trained model from MODEL_PATH and skip training
LOAD_BIN_MODEL = True
LOAD_LOC_MODEL = True

# Columns to include when creating data chunks. Some fields (e.g., 'offset', 'decay') may not be necessary as label data already contains the relevant information
# Options: ['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU', 'latitude', 'cos_longitude', 'sin_longitude', 'height', 'offset_value', 'offset_error', 'decay_value', 'decay_error', 'decay_tau', 'decay_type']
# Recommended order if including location data: ['latitude', 'cos_longitude', 'sin_longitude', 'height'] (especially if TRAIN_NA = True)
USED_COLS = ['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU', 'latitude', 'cos_longitude', 'sin_longitude']

# Use pre-defined optimal parameters for training. If False, parameters will be tuned
OPTIMAL_BIN_PARAMS = True
OPTIMAL_LOC_PARAMS = True

# Oversampling: Adjusts the minority class proportion relative to the majority class, specified as (lower_bound, upper_bound)
OVERSAMPLING_PERCENTAGES = True

# Number of days to include before the first and after the last earthquake (lower_bound, upper_bound)
NBR_OF_DAYS = (182, 366)

# Size of each data chunk
CHUNK_SIZE = 60

# If True, train the model on North American stations and test on New Zealand stations
# Requires ['latitude', 'cos_longitude', 'sin_longitude'] in USED_COLS for regional filtering
TRAIN_NA = False

# Best parameters for the binary tcn
BEST_PARAMS_TCN_BINARY = {'depth': 5, 'filters': 128, 'kernel_size': 3, 'dilation_rate': 1, 'batch_size': 32}

# Best parameters for the localization tcn
BEST_PARAMS_TCN_LOCALIZATION = {'depth':5, 'filters': 64, 'kernel_size': 5, 'dilation_rate': 1, 'batch_size':64}


