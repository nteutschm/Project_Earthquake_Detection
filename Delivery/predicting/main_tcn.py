#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import sys
from sklearn.utils import resample
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping

pd.options.mode.chained_assignment = None

# Modified the path to ensure all files are correctly located
sys.path.append('/Users/merlinalfredsson/Notebooks/GNSS_Displacement/Github/Project_Earthquake_Detection/')

if 'variables' in sys.modules:
    del sys.modules['variables']  # Remove the existing cache for variables

from variables_tcn import *
from preprocess import *
from models import *
from evaluation import *
from plots import *
from two_step_testing import *

def prepare_data(X, y, start_index, stations, geometries):
    """
    Prepares the training, evaluation, and test datasets for a machine learning model by splitting the data 
    based on geographical locations or station allocation and applying feature scaling and oversampling to 
    handle imbalanced classes.

    Parameters:
    - X (DataFrame): Feature matrix containing input data (chunked GNSS data).
    - y (list): List of target labels.
    - start_index (list): List of start indices corresponding to each chunk.
    - stations (list): List of station identifiers for each chunk.
    - geometries (list): List of geographical data corresponding to each chunk.

    Returns:
    - X_train (np.ndarray): Feature matrix for the training dataset.
    - X_eval (np.ndarray): Feature matrix for the evaluation dataset.
    - X_test (np.ndarray): Feature matrix for the test dataset.
    - y_train (np.ndarray): Binary target labels for the training dataset.
    - y_eval (np.ndarray): Binary target labels for the evaluation dataset.
    - y_test (np.ndarray): Binary target labels for the test dataset.
    - y_loc_train (np.ndarray): Localization target labels for the training dataset.
    - y_loc_eval (np.ndarray): Localization target labels for the evaluation dataset.
    - y_loc_test (np.ndarray): Localization target labels for the test dataset.
    - test_start_index (list): List of start indices for each chunk in the test dataset.
    - train_stations, eval_stations, test_stations (list): List of stations in the training, evaluation, and test datasets.
    - geometries_train, geometries_eval, geometries_test (list): Geographical data of each chunk for the respective datasets.

    Process:
    1. If the `TRAIN_NA` flag is set, the function splits the dataset into training and testing sets based on 
    geographical boundaries. North America data is used for training, and New Zealand data is used for testing. 
    The datasets are filtered based on latitude and longitude conditions. 
    2. If `TRAIN_NA` is not enabled, the data is split into training, evaluation, and test sets by randomly 
    allocating stations into these sets.
    3. The function applies MinMax scaling to the features for each station separately, except for the location and height, 
    which are scaled across all stations to prevent constant features.
    4. It oversamples the minority classes in the training data to ensure a more balanced distribution of labels.
    6. Optionally combines training and evaluation datasets if specific optimization flags are set.
    """

    if TRAIN_NA:
        # Extract latitude and longitude information
        latitude = X[540]  # Replace with the correct index for latitude
        cos_longitude = X[541]  # Replace with the correct index for cos_longitude
        sin_longitude = X[542]  # Replace with the correct index for sin_longitude
        longitude = np.degrees(np.arctan2(sin_longitude, cos_longitude))

        # Define geographic masks
        north_america_mask = (latitude > 0) & (longitude >= -180) & (longitude <= -100)
        new_zealand_mask = (latitude >= -50) & (latitude <= -30) & (longitude >= 165) & (longitude <= 180)

        # Keeping only the displacents by slicing 
        X = X.iloc[:, :CHUNK_SIZE*3]
        # MinMax Scaling of the features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        num_features = int(X.shape[1]/CHUNK_SIZE) 
        # Reshaping to the right input shape for the TCN
        X = X.reshape(-1, CHUNK_SIZE, num_features, order='F')
        y_bin = []
        y_loc = []
    
        for value in y:
            # y_bin: 1 if value is not 0, else 0
            y_bin.append(1 if value != 0 else 0)
            # y_loc: Array of 21 zeros, with a 1 at the position specified by y[i] (if not 0)
            loc_array = np.zeros(CHUNK_SIZE)
            if value != 0:
                loc_array[value] = 1
            y_loc.append(loc_array)
        y_loc = np.array(y_loc)
        y_bin = np.array(y_bin)

        # Apply masks to stations
        train_mask = north_america_mask
        test_mask = new_zealand_mask
        eval_mask = ~train_mask & ~test_mask  # Use remaining stations as evaluation set

        # Filter data based on masks
        X_train, y_bin_train, y_loc_train, geometries_train = (
            X[train_mask],
            [y_bin[i] for i in range(len(y_bin)) if train_mask[i]],
            [y_loc[i] for i in range(len(y_loc)) if train_mask[i]],
            [geometries[i] for i in range(len(geometries)) if train_mask[i]]
        )
        X_eval, y_bin_eval, y_loc_eval, geometries_eval = (
            X[eval_mask],
            [y_bin[i] for i in range(len(y_bin)) if eval_mask[i]],
            [y_loc[i] for i in range(len(y_loc)) if eval_mask[i]],
            [geometries[i] for i in range(len(geometries)) if eval_mask[i]]
        )
        X_test, y_bin_test, y_loc_test, test_start_index, geometries_test = (
            X[test_mask],
            [y_bin[i] for i in range(len(y_bin)) if test_mask[i]],
            [y_loc[i] for i in range(len(y_loc)) if test_mask[i]],
            [start_index[i] for i in range(len(start_index)) if test_mask[i]],
            [geometries[i] for i in range(len(geometries)) if test_mask[i]]
        )

        # Extract station IDs for reference
        train_stations = [stations[i] for i in range(len(stations)) if train_mask[i]]
        eval_stations = [stations[i] for i in range(len(stations)) if eval_mask[i]]
        test_stations = [stations[i] for i in range(len(stations)) if test_mask[i]]

    else:
        # Keeping only the displacents by slicing 
        X = X.iloc[:, :CHUNK_SIZE*3]
        # MinMax Scaling of the features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        num_features = int(X.shape[1]/CHUNK_SIZE)
        # Reshaping to the right input shape for the TCN 
        X = X.reshape(-1, CHUNK_SIZE, num_features, order='F')
        y_bin = []
        y_loc = []
    
        for value in y:
            # y_bin: 1 if value is not 0, else 0
            y_bin.append(1 if value != 0 else 0)
            
            # y_loc: Array of 21 zeros, with a 1 at the position specified by y[i] (if not 0)
            loc_array = np.zeros(CHUNK_SIZE)
            if value != 0:
                loc_array[value] = 1
            y_loc.append(loc_array)
        y_loc = np.array(y_loc)
        y_bin = np.array(y_bin)

        rng = np.random.default_rng(seed=RANDOM_STATE)
        
        # Splitting the stations randomly
        unique_stations = sorted(list(set(stations))) 
        unique_stations = rng.permutation(unique_stations) 
        n_train = int(0.7 * len(unique_stations))
        n_eval = int(0.15 * len(unique_stations))
        
        train_stations = unique_stations[:n_train]
        eval_stations = unique_stations[n_train:n_train + n_eval]
        test_stations_unique = unique_stations[n_train + n_eval:]

        train_mask = [s in train_stations for s in stations]
        eval_mask = [s in eval_stations for s in stations]
        test_mask = [s in test_stations_unique for s in stations]

        # Filter data based on masks
        X_train, y_bin_train, y_loc_train, geometries_train = (
            X[train_mask],
            [y_bin[i] for i in range(len(y_bin)) if train_mask[i]],
            [y_loc[i] for i in range(len(y_loc)) if train_mask[i]],
            [geometries[i] for i in range(len(geometries)) if train_mask[i]]
        )
        X_eval, y_bin_eval, y_loc_eval, geometries_eval = (
            X[eval_mask],
            [y_bin[i] for i in range(len(y_bin)) if eval_mask[i]],
            [y_loc[i] for i in range(len(y_loc)) if eval_mask[i]],
            [geometries[i] for i in range(len(geometries)) if eval_mask[i]]
        )
        X_test, y_bin_test, y_loc_test, test_start_index, geometries_test = (
            X[test_mask],
            [y_bin[i] for i in range(len(y_bin)) if test_mask[i]],
            [y_loc[i] for i in range(len(y_loc)) if test_mask[i]],
            [start_index[i] for i in range(len(start_index)) if test_mask[i]],
            [geometries[i] for i in range(len(geometries)) if test_mask[i]]
        )
        
        train_stations = [stations[i] for i in range(len(stations)) if train_mask[i]]
        eval_stations = [stations[i] for i in range(len(stations)) if eval_mask[i]]
        test_stations = [stations[i] for i in range(len(stations)) if test_mask[i]]
        
    if OPTIMAL_BIN_PARAMS and OPTIMAL_LOC_PARAMS:
        # Combine train and eval for ndarrays
        print("combining training and evaluation data")
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_bin_train = np.concatenate((y_bin_train, y_bin_eval), axis=0)
        y_loc_train = np.concatenate((y_loc_train, y_loc_eval), axis=0)
        # Append lists directly for non-array objects
        train_stations += eval_stations
        geometries_train += geometries_eval
        # Clear the eval variables
        X_eval, y_bin_eval, y_loc_eval, geometries_eval, eval_stations = np.array([]), [], [], [], []

    
    # Oversample the minority classes
    if OVERSAMPLING_PERCENTAGES is True:
        X_y_combined = [(X_train[i], y_bin_train[i], y_loc_train[i]) for i in range(len(y_bin_train))]

        class_0 = [(x, binary, loc) for x, binary, loc in X_y_combined if binary == 0]
        class_1 = [(x, binary, loc) for x, binary, loc in X_y_combined if binary == 1]
        print("class 0: ", len(class_0))
        print("class 1: ", len(class_1))

        if len(class_0) > len(class_1):
            class_1_resampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=RANDOM_STATE)
            balanced_data = class_0 + class_1_resampled
            print("class 1 resampled: ", len(class_1_resampled))
        else:
            class_0_resampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=RANDOM_STATE)
            balanced_data = class_1 + class_0_resampled
        
        X_train, y_bin_train, y_loc_train = zip(*balanced_data)

    X_train, y_bin_train, y_loc_train = np.array(X_train), np.array(y_bin_train), np.array(y_loc_train)
    X_eval, y_bin_eval, y_loc_eval = np.array(X_eval), np.array(y_bin_eval), np.array(y_loc_eval)
    X_test, y_bin_test, y_loc_test = np.array(X_test), np.array(y_bin_test), np.array(y_loc_test)
    
    return X_train, X_eval, X_test, y_bin_train, y_bin_eval, y_bin_test, y_loc_train, y_loc_eval, y_loc_test, test_start_index, test_stations, (geometries_train, geometries_eval, geometries_test)

def train_model(X, y, start_index, stations, cleaned_dfs, geometries):
    """
    Trains a two-step machine learning model consisting of a binary classification model and a localization model 
    to predict offsets and their specific locations in GNSS time series data.

    Parameters:
    - X (np.ndarray): Feature matrix containing the input data for the models.
    - y (np.ndarray): Target labels (binary classification and localization) for the models.
    - start_index (list): List of start indices for each data chunk in the test dataset.
    - stations (list): List of station identifiers for each data chunk.
    - cleaned_dfs (list): Cleaned GNSS datasets for evaluation and visualization purposes.
    - geometries (list): List of geographical information for each data chunk.

    Process:
    1. **Data Preparation**: 
    - Prepares the dataset by splitting it into training, evaluation, and test sets using `prepare_data`.
    - Filters samples with offsets (binary label = 1) for the localization model.
    
    2. **Binary Classification Model**:
    - Builds and trains a Temporal Convolutional Network (TCN) binary classification model using `optimize_tcn_binary`.
    - Predicts binary classification labels for the test set.
    - Converts predicted probabilities into binary labels using a threshold (default: 0.5).

    3. **Localization Model**:
    - Filters the training set for samples with offsets to create the localization training dataset.
    - Builds and trains a TCN localization model using `optimize_tcn_loc`.
    - Predicts localization labels (offset positions) for the test set.

    4. **Test Predictions**:
    - Creates a mask to filter predictions from the localization model for samples predicted as having offsets by the binary model.
    - Combines binary and localization predictions to create a final test prediction dataset.

    5. **Evaluation and Saving Results**:
    - Converts test predictions and labels into `pandas.Series` for saving to disk.
    - Saves the trained binary and localization models and test predictions, labels, and station names to specified file paths.
    - Evaluates the two-step model's performance on the test set using `test_two_step`.
    - Performs additional evaluation using different tolerance windows to allow for flexibility in offset position predictions.

    6. **Model and Results Export**:
    - Saves both trained models to disk for future inference or analysis.
    - Saves predictions, labels, and station information for further analysis.

    Returns:
    - model_binary: Trained binary classification model.
    - geometries: Geographical information for each data chunk (unchanged from input).

    """

    
    # Prepare the data by reshaping, resampling and reshaping
    X_train, X_eval, X_test, y_bin_train, y_bin_eval, y_bin_test, y_loc_train, y_loc_eval, y_loc_test,  test_start_index, test_stations, geometries = prepare_data(X, y, start_index, stations, geometries)

    # Build and train localization model
    print("Training binary classification model...")
    model_binary = optimize_tcn_binary(X_train, y_bin_train, X_eval, y_bin_eval)

    # Predict Binary
    y_pred_probs = model_binary.predict(X_test,verbose=0)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # Filter samples with offsets for localization training
    #offset_indices = np.where(y_bin_train == 1)[0]
    offset_indices = np.where(np.argmax(y_loc_train, axis=1) != 0)[0]
    X_loc_train = X_train[offset_indices]
    y_loc_train_filtered = y_loc_train[offset_indices]

    # Build and train localization model
    print("Training localization model...")
    model_localization = optimize_tcn_loc(X_loc_train, y_loc_train_filtered, X_eval, y_loc_eval, y_bin_eval)
    
    # Predict Localization
    y_pred_localization = model_localization.predict(X_test,verbose=0)

    # Create a mask for binary predictions that are 1
    mask = y_pred_binary == 1
    mask = mask.flatten()
    test_predictions = np.zeros_like(y_pred_binary, dtype=int)
    test_predictions = test_predictions.flatten()
    test_predictions[mask] = np.argmax(y_pred_localization[mask], axis=1)

    # Create a mask for test lables that include offsets
    mask = y_bin_test == 1
    mask = mask.flatten()
    y_test = np.zeros_like(y_bin_test, dtype=int)
    y_test = y_test.flatten()
    y_test[mask] = np.argmax(y_loc_test[mask], axis=1)

    # Convert multiclass test predictions 
    test_predictions = pd.Series(test_predictions, index=test_start_index)
    test_labels = pd.Series(y_test, index=test_start_index)
    test_stations = pd.Series(test_stations, index=test_start_index)
    
    # Save the models
    joblib.dump(model_binary, MODEL_BIN_PATH)
    joblib.dump(model_localization, MODEL_LOC_PATH)
    
    for data, path in zip([test_predictions, test_labels, test_stations], [PREDICTIONS_PATH, TEST_LABELS_PATH, STATION_NAMES]):
        data.to_csv(path, index=True)

    # Advanced Testing, designed to encorperate the sequential approach
    test_two_step(X_test, y_bin_test, y_loc_test, test_stations)
    
    # Further evaluations for the multiclass predictions
    for window in [None, 1]:
        test_predictions = evaluate(test_predictions, test_labels, model=model_binary, X_test=X_test, tolerance_window=window, cleaned_dfs=cleaned_dfs, stations=test_stations, start_indices=test_start_index)
    
    return model_binary, geometries

def main():
    """
    Main function to execute the data processing and model training pipeline.

    This function executes the following steps:
    1. Reads all data files from the specified data directory and stores them in a list.
    2. Cleans the DataFrames using the clean_dataframes function, applying the specified thresholds for missing values and minimal offsets.
    3. Extracts features and labels from the cleaned DataFrames, with an option to interpolate missing data based on the selected model type.
    4. Saves the extracted features and target labels to CSV files for further use.
    5. Trains the specified model using the extracted features and target labels, and outputs a report of the model's performance.
    6. Optionally visualizes station geometries and applies simulations for testing.

    Returns:
    None
    """
    log = open(LOG_FILE, 'w', buffering=1)
    sys.stdout = log
    print("Starting")

    print("Reading")
    dfs = []
    dir = Path(DATA_PATH)
    for file_path in dir.iterdir():
        if file_path.is_file():
            dfs.append(read_file(file_path.name))

    print("Cleaning")
    cleaned_dfs, simulated_dfs = clean_dataframes(dfs, missing_value_threshold=0, limited_period=True, minimal_offset=5)
    
    df_dict = {f'df_{i}': df for i, df in enumerate(cleaned_dfs)}
    # Add the .name attribute
    for i, df in enumerate(cleaned_dfs):
        df.name = f"df_{i}"  # Assign a name
    pd.to_pickle(df_dict, CLEANED_DATA_PATH)

    # ALTERNATIVELY: Load cleaned data if it has been saved previously
    loaded_df_dict = pd.read_pickle(CLEANED_DATA_PATH)


    for key, df in loaded_df_dict.items():
        df.name = key
    print("Loaded data!")
    cleaned_dfs = [df for df in loaded_df_dict.values()]
    
    
    # HistGradientBoosting designed to deal with None data -> No interpolation needed
    interpolate = False if MODEL_TYPE == 'HistGradientBoosting' else True
    print("Extracting features")
    X, y, start_index, stations, geometries = extract_features(cleaned_dfs, interpolate=interpolate)
    
    X.to_csv(f'{FEATURES_PATH}_features.csv', index=True)
    pd.Series(y).to_csv(f'{FEATURES_PATH}_target.csv', index=False, header=False)
    
    print("Training")
    _, geometries = train_model(X, y, start_index, stations, cleaned_dfs=cleaned_dfs, geometries=geometries)
    
    plot_station_geometries(geometries)
    
    sys.stdout = sys.__stdout__
    log.close()

if __name__=='__main__':
    main()
    