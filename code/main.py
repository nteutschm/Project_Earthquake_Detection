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

pd.options.mode.chained_assignment = None

# Make sure that it finds all additional files correctly
sys.path.append('/cluster/home/nteutschm/eqdetection/')
from variables import *
from preprocess import *
from models import *
from evaluation import *
from plots import *

def prepare_data(X, y, start_index, stations, geometries):
    """
    Prepares training, evaluation, and testing data by splitting based on stations,
    so that no station appears in more than one set.

    Parameters:
    X (DataFrame): The feature matrix (chunked GNSS data).
    y (list or Series): The target labels (multiclass).
    start_index (list): Start index for each sample in `X`.
    stations (list): Station names associated with each sample.
    random_state (int): Random seed for reproducibility.
    oversampling_percentages (tuple): Range of percentages to oversample minority classes.

    Returns:
    X_train, X_eval, X_test, y_train, y_eval, y_test, class_weights, test_start_index, test_stations
    """
    unique_stations = sorted(list(set(stations)))
    rng = np.random.default_rng(seed=RANDOM_STATE) 
    unique_stations = rng.permutation(unique_stations) 
    n_train = int(0.7 * len(unique_stations))
    n_eval = int(0.15 * len(unique_stations))
    
    train_stations = unique_stations[:n_train]
    eval_stations = unique_stations[n_train:n_train + n_eval]
    test_stations_unique = unique_stations[n_train + n_eval:]

    # Create masks based on station allocations
    train_mask = [s in train_stations for s in stations]
    eval_mask = [s in eval_stations for s in stations]
    test_mask = [s in test_stations_unique for s in stations]

    # Filter data based on masks
    X_train, y_train, _, geometries_train = (
        X[train_mask],
        [y[i] for i in range(len(y)) if train_mask[i]],
        [start_index[i] for i in range(len(start_index)) if train_mask[i]],
        [geometries[i] for i in range(len(geometries)) if train_mask[i]]
    )
    X_eval, y_eval, _, geometries_eval = (
        X[eval_mask],
        [y[i] for i in range(len(y)) if eval_mask[i]],
        [start_index[i] for i in range(len(start_index)) if eval_mask[i]],
        [geometries[i] for i in range(len(geometries)) if eval_mask[i]]
    )
    X_test, y_test, test_start_index, geometries_test = (
        X[test_mask],
        [y[i] for i in range(len(y)) if test_mask[i]],
        [start_index[i] for i in range(len(start_index)) if test_mask[i]],
        [geometries[i] for i in range(len(geometries)) if test_mask[i]]
    )
    
    def scale_features_per_station(X, station_list):
        unique_stations = set(station_list)
        scaled_data = pd.DataFrame(index=X.index, columns=X.columns)
        geo_columns = []

        expected_geo_columns = ['latitude', 'sin_longitude', 'cos_longitude', 'height']

        if any(col in USED_COLS for col in expected_geo_columns):
            num_geo_cols = sum(col in USED_COLS for col in expected_geo_columns)
            geo_columns = X.columns[-num_geo_cols:]

        if not isinstance(geo_columns, list):
            scaler = MinMaxScaler()
            X[geo_columns] = scaler.fit_transform(X[geo_columns])

        for station in unique_stations:
            station_indices = [i for i, s in enumerate(station_list) if s == station]
            X_station = X.iloc[station_indices]

            columns_to_scale = [col for col in X_station.columns if col not in geo_columns]

            if columns_to_scale:
                scaler = MinMaxScaler()
                X_station_scaled = X_station[columns_to_scale]
                X_station_scaled = scaler.fit_transform(X_station_scaled)
                X_station[columns_to_scale] = X_station_scaled

            scaled_data.iloc[station_indices] = X_station

        return scaled_data.astype(np.float32)
    
    train_stations = [stations[i] for i in range(len(stations)) if train_mask[i]]
    eval_stations = [stations[i] for i in range(len(stations)) if eval_mask[i]]
    test_stations = [stations[i] for i in range(len(stations)) if test_mask[i]]

    X_train = scale_features_per_station(X_train, train_stations)
    X_eval = scale_features_per_station(X_eval, eval_stations)
    X_test = scale_features_per_station(X_test, test_stations)
    
    # Oversample the minority classes to a random percentage between 10% and 40% of the majority class
    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)

    sampling_strategy = {}
    for cls in class_counts:
        if cls != majority_class:
            random_percentage = rng.uniform(OVERSAMPLING_PERCENTAGES[0], OVERSAMPLING_PERCENTAGES[1])
            target_count = int(class_counts[majority_class] * random_percentage)
            sampling_strategy[cls] = target_count
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Compute class weights to handle imbalanced dataset
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {c: w for c, w in zip(classes, class_weights)}
    
    return pd.DataFrame(X_train), pd.DataFrame(X_eval), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_eval), pd.Series(y_test), class_weights, test_start_index, test_stations, (geometries_train, geometries_eval, geometries_test)

def train_model(X, y, start_index, stations, model_type, cleaned_dfs, geometries):
    """
    Trains a model based on the specified type using both North, East, and Up component data.

    Parameters:
    X (DataFrame): Feature set, including target values.
    model_type (str): The type of model to train ('IsolationForest' or 'HistGradientBoosting').

    Returns:
    model: Trained model.
    test_predictions: Predictions on the test set.
    report: Classification report for the test set.
    """
    
    X_train, X_eval, X_test, y_train, y_eval, y_test, weights, test_start_index, test_stations, geometries = prepare_data(X, y, start_index, stations, geometries)

    if model_type == 'IsolationForest':
        model = optimize_isolation_forest(X_train, y_train)
        test_predictions = model.predict(X_test)
        
    elif model_type == 'RandomForest':
        model = optimize_random_forest(X_train, y_train, X_eval, y_eval, weights)
        test_predictions = model.predict(X_test)
    
    elif model_type == 'HistGradientBoosting':
        model = optimize_hist_gradient_boosting(X_train, y_train, weights)
        test_predictions = model.predict(X_test)
        
    elif model_type == 'XGBoost':
        model = optimize_xgboost(X_train, y_train, X_eval, y_eval)
        test_predictions = model.predict(X_test)
        
    else:
        raise ValueError('Used Model Type not implemented. Please control spelling!')
    
    test_predictions = pd.Series(test_predictions, index=test_start_index)
    test_labels = pd.Series(y_test.values, index=test_start_index)
    test_stations = pd.Series(test_stations, index=test_start_index)
    
    joblib.dump(model, MODEL_PATH)
    for data, path in zip([test_predictions, test_labels, test_stations], [PREDICTIONS_PATH, TEST_LABELS_PATH, STATION_NAMES]):
        data.to_csv(path, index=True)
    for window in [None, 1]:
        test_predictions = evaluate(test_predictions, test_labels, model=model, X_test=X_test, tolerance_window=window, cleaned_dfs=cleaned_dfs, stations=test_stations, start_indices=test_start_index)
    return model, geometries


def main():
    """
    Main function to execute the data processing and model training pipeline.

    This function orchestrates the following steps:
    1. Reads all data files from the specified data directory and stores them in a list.
    2. Cleans the DataFrames using the clean_dataframes function, applying specified thresholds 
       for missing values and minimal offsets.
    3. Extracts features and labels from the cleaned DataFrames, with an option to interpolate 
       missing data based on the selected model type.
    4. Saves the extracted features and target labels to CSV files for further use.
    5. Trains the specified model using the extracted features and target labels, and outputs 
       a report of the model's performance.

    Returns:
    None
    """
    log = open(LOG_FILE, 'w', buffering=1)
    sys.stdout = log
    dfs = []
    dir = Path(DATA_PATH)
    for file_path in dir.iterdir():
        if file_path.is_file():
            dfs.append(read_file(file_path.name))

    cleaned_dfs, simulated_dfs = clean_dataframes(dfs, missing_value_threshold=0, limited_period=True, minimal_offset=5)
    
    df_dict = {f'df_{i}': df for i, df in enumerate(cleaned_dfs)}
    pd.to_pickle(df_dict, CLEANED_DATA_PATH)
    simulated_dfs = generate_synthetic_offsets(simulated_dfs)
    
    # HistGradientBoosting designed to deal with None data -> No interpolation needed
    interpolate = False if MODEL_TYPE == 'HistGradientBoosting' else True
    X, y, start_index, stations, geometries = extract_features(cleaned_dfs, interpolate=interpolate)
    
    X.to_csv(f'{FEATURES_PATH}_features.csv', index=True)
    pd.Series(y).to_csv(f'{FEATURES_PATH}_target.csv', index=False, header=False)
    
    model, geometries = train_model(X, y, start_index, stations, model_type=MODEL_TYPE, cleaned_dfs=cleaned_dfs, geometries=geometries)
    
    X_test, y_test, start_index_test, stations_test, geometries_sim = extract_features(simulated_dfs, interpolate=interpolate)
    
    plot_station_geometries(geometries, geometries_sim)
    
    test_predictions = model.predict(X_test)
    test_predictions = pd.Series(test_predictions, index=start_index_test)
    y_test = pd.Series(y_test, index=start_index_test)
    stations_test = pd.Series(stations_test, index=start_index_test)
    
    # For now excluded due to the bad accurcay of the simulations
    #for window in [None, 1]:
    #    test_predictions = evaluate(test_predictions, y_test, model=model, X_test=X_test, tolerance_window=window, cleaned_dfs=simulated_dfs, stations=stations_test, start_indices=start_index_test, simulated=True)
    
    sys.stdout = sys.__stdout__
    log.close()

if __name__=='__main__':
    main()