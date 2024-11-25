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
from variables_tcn import *
from preprocess import *
from models import *
from evaluation import *
from plots import *

def prepare_data(X, y, start_index, stations, geometries):
    """
    Prepares the training, evaluation, and test datasets for a machine learning model by splitting the data 
    based on geographical locations or station allocation and applying feature scaling and oversampling to 
    handle imbalanced classes.

    Parameters:
    - X_Full (DataFrame): Feature matrix containing input data (chunked GNSS data).
    - y (list): List of target labels.
    - start_index (list): List of start indices corresponding to each chunk.
    - stations (list): List of station identifiers for each chunk.
    - geometries (list): List of geographical data corresponding to each chunk.

    Returns:
    - X_train (DataFrame): Feature matrix for the training dataset.
    - X_eval (DataFrame): Feature matrix for the evaluation dataset.
    - X_test (DataFrame): Feature matrix for the test dataset.
    - y_train (Series): Target labels for the training dataset.
    - y_eval (Series): Target labels for the evaluation dataset.
    - y_test (Series): Target labels for the test dataset.
    - class_weights (dict): Dictionary containing class weights for imbalanced data.
    - test_start_index (list): List of start indices for each chunk in the test dataset.
    - test_stations (list): List of stations of each chunk in the test dataset.
    - tuple of geometries_train, geometries_eval, geometries_test (list): Geographical data of each chunk for the respective datasets.

    Process:
    1. If the `TRAIN_NA` flag is set, the function splits the dataset into training and testing sets based on 
    geographical boundaries. North America data is used for training, and New Zealand data is used for testing. 
    The datasets are filtered based on latitude and longitude conditions. 
    2. If `TRAIN_NA` is not enabled, the data is split into training, evaluation, and test sets by randomly 
    allocating stations into these sets.
    3. The function applies MinMax scaling to the features for each station separately, except for the location and height, 
    which is scaled across all stations to prevent constant features.
    4. It oversamples the minority classes in the training data to ensure a more balanced distribution of labels by 
    generating synthetic samples using the SMOTE technique.
    5. Class weights are computed to handle the remaining imbalance in the training set.
    """
    X = X.iloc[:, :63]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    num_features = int(X.shape[1]/CHUNK_SIZE) 
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
    
    if OPTIMAL_PARAMS:
        # Combine train and eval for ndarrays
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_bin_train = np.concatenate((y_bin_train, y_bin_eval), axis=0)
        y_loc_train = np.concatenate((y_loc_train, y_loc_eval), axis=0)
        # Append lists directly for non-array objects
        train_stations += eval_stations
        geometries_train += geometries_eval
        # Clear the eval variables
        X_eval, y_bin_eval, y_loc_eval, geometries_eval, eval_stations = np.array([]), [], [], [], []

    
    # Oversample the minority classes
    # Handle balancing if labels are provided
    if OVERSAMPLING_PERCENTAGES is True:
        X_y_combined = [(X_train[i], y_bin_train[i], y_loc_train[i]) for i in range(len(y_bin_train))]

        class_0 = [(x, binary, loc) for x, binary, loc in X_y_combined if binary == 0]
        class_1 = [(x, binary, loc) for x, binary, loc in X_y_combined if binary == 1]

        if len(class_0) > len(class_1):
            class_1_resampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=RANDOM_STATE)
            balanced_data = class_0 + class_1_resampled
        else:
            class_0_resampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=RANDOM_STATE)
            balanced_data = class_1 + class_0_resampled

        X_train, y_bin_train, y_loc_train = zip(*balanced_data)

    X_train, y_bin_train, y_loc_train = np.array(X_train), np.array(y_bin_train), np.array(y_loc_train)
    X_eval, y_bin_eval, y_loc_eval = np.array(X_eval), np.array(y_bin_eval), np.array(y_loc_eval)
    X_test, y_bin_test, y_loc_test = np.array(X_test), np.array(y_bin_test), np.array(y_loc_test)
    
    return X_train, X_eval, X_test, y_bin_train, y_bin_eval, y_bin_test, y_loc_train, y_loc_eval, y_loc_test, test_start_index, test_stations, (geometries_train, geometries_eval, geometries_test)

def train_model(X, y, start_index, stations, model_type, cleaned_dfs, geometries):
    """
    Trains a machine learning model on the provided data and evaluates its performance on the test set.

    This function prepares the data, selects a model based on the specified model_type, and performs training and evaluation. 
    It then saves the model and related outputs using the specified paths. 
    The model can be selected from different algorithms, including Isolation Forest, Random Forest, HistGradientBoosting, and XGBoost.

    Parameters:
    - X (pd.DataFrame): The feature data used for training the model.
    - y (pd.Series): The target labels corresponding to the feature data.
    - start_index (list): The list of start indices for each chunk.
    - stations (list): The list of stations associated with each chunk.
    - model_type (str): The type of model to use. Can be one of ['IsolationForest', 'RandomForest', 'HistGradientBoosting', 'XGBoost'].
    - cleaned_dfs (dict): A dictionary containing cleaned dataframes used for evaluation.
    - geometries (list): A list of geometry data corresponding to each chunk.

    Returns:
    - model: The trained machine learning model.
    - geometris: tuple of geometries_train, geometries_eval, geometries_test (list): Geographical data of each chunk for the respective datasets.

    Process:
    1. The function calls prepare_data to split and scale the data into training, evaluation, and test sets.
    2. Based on the chosen `model_type`, it selects the appropriate model and trains it using the training data (X_train, y_train).
    3. The trained model is used to generate predictions on the test data (X_test).
    4. The predictions, actual labels, and station names are saved to CSV files for further analysis.
    5. The evaluate function is called to assess model performance, varying the tolerance window for predictions.
    6. The model is saved for future use, and the function returns the trained model along with the geometries.

    The function supports four model types:
    - 'IsolationForest': A model used for anomaly detection, suitable for identifying outliers in the data.
    - 'RandomForest': A versatile ensemble learning method for classification and regression.
    - 'HistGradientBoosting': A gradient boosting method for classification tasks, optimized for large datasets.
    - 'XGBoost': A highly efficient and scalable gradient boosting framework.
    """
    
    X_train, X_eval, X_test, y_bin_train, y_bin_eval, y_bin_test, y_loc_train, y_loc_eval, y_loc_test,  test_start_index, test_stations, geometries = prepare_data(X, y, start_index, stations, geometries)



    if model_type == 'TCN' and OPTIMAL_PARAMS is True:
        # Build and train localization model
        print("Training binary classification model...")
        reduce_lr_binary = ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=5, min_lr=1e-6)
        early_stopping_binary = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        model_binary = build__binary_model_tcn(input_shape=(CHUNK_SIZE, X_train.shape[2]))
        model_binary.fit(X_train, y_bin_train, epochs=EPOCHS, batch_size=64, validation_split=0.2, callbacks=[early_stopping_binary, reduce_lr_binary], verbose=2)

        # Filter samples with offsets for localization training
        offset_indices = np.where(y_bin_train == 1)[0]
        X_train_localization = X_train[offset_indices]
        y_train_localization_filtered = y_loc_train[offset_indices]

        # Build and train localization model
        print("Training localization model...")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        model_localization = build_localization_model_tcn(input_shape=(CHUNK_SIZE, X_train.shape[2]), time_steps=CHUNK_SIZE)
        model_localization.fit(X_train_localization, y_train_localization_filtered, epochs=EPOCHS, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=2)

        # Predict
        y_pred_probs = model_binary.predict(X_test)
        y_pred_binary = (y_pred_probs > 0.5).astype(int)
        y_pred_localization = model_localization.predict(X_test)

        # Binary Evaluation
        binary_cm = confusion_matrix(y_bin_test, y_pred_binary)
        print("binary reports:")
        print(binary_cm)
        report_binary = classification_report(y_bin_test, y_pred_binary, target_names=['No Offset', 'Offset'])
        print(report_binary)

        # Create a mask for binary predictions that are 1
        mask = y_pred_binary == 1
        mask = mask.flatten()
        test_predictions = np.zeros_like(y_pred_binary, dtype=int)
        test_predictions = test_predictions.flatten()
        test_predictions[mask] = np.argmax(y_pred_localization[mask], axis=1)

        # Create a mask for binary lables that are 1
        mask = y_bin_test == 1
        mask = mask.flatten()
        y_test = np.zeros_like(y_bin_test, dtype=int)
        y_test = y_test.flatten()
        y_test[mask] = np.argmax(y_loc_test[mask], axis=1)

    else:
        raise ValueError('Used Model Type not implemented. Please control spelling!')
    
    test_predictions = pd.Series(test_predictions, index=test_start_index)
    test_labels = pd.Series(y_test, index=test_start_index)
    test_stations = pd.Series(test_stations, index=test_start_index)
    
    joblib.dump(model_binary, MODEL_BIN_PATH)
    joblib.dump(model_binary, MODEL_LOC_PATH)
    
    for data, path in zip([test_predictions, test_labels, test_stations], [PREDICTIONS_PATH, TEST_LABELS_PATH, STATION_NAMES]):
        data.to_csv(path, index=True)
    
    for window in [None, 1]:
        print("/n/n/n")
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
    """
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

    """

    loaded_df_dict = pd.read_pickle(CLEANED_DATA_PATH)
    for key, df in loaded_df_dict.items():
        df.name = key
    print("loaded data!")
    cleaned_dfs = [df for df in loaded_df_dict.values()]
    
    
    # HistGradientBoosting designed to deal with None data -> No interpolation needed
    interpolate = False if MODEL_TYPE == 'HistGradientBoosting' else True
    print("extracting features")
    X, y, start_index, stations, geometries = extract_features(cleaned_dfs, interpolate=interpolate)
    
    X.to_csv(f'{FEATURES_PATH}_features.csv', index=True)
    pd.Series(y).to_csv(f'{FEATURES_PATH}_target.csv', index=False, header=False)
    
    print("training")
    model, geometries = train_model(X, y, start_index, stations, model_type=MODEL_TYPE, cleaned_dfs=cleaned_dfs, geometries=geometries)
    
    plot_station_geometries(geometries)
    
    sys.stdout = sys.__stdout__
    log.close()

if __name__=='__main__':
    main()