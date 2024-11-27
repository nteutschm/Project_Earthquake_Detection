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

# Modified the path to ensure all files are correctly located
sys.path.append('/cluster/home/nteutschm/eqdetection/')
from variables import *
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
    - X (DataFrame): Feature matrix containing input data (chunked GNSS data).
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
    
    rng = np.random.default_rng(seed=RANDOM_STATE)
    impl_cols = USED_COLS.copy()
    if TRAIN_NA:
        print('The model will be trained using stations in North America and then tested on stations in New Zealand')
        assert 'latitude' in impl_cols and 'cos_longitude' in impl_cols and 'sin_longitude' in impl_cols, 'Not all location features are in the data, check that latitude, cos_longitude and sin_longitude are in USED_COLS'
        expected_geo_columns = ['latitude', 'sin_longitude', 'cos_longitude', 'height']
        num_geo_cols = sum(col in impl_cols for col in expected_geo_columns)
        tot_location = X.columns[-num_geo_cols:]
        if 'height' in impl_cols:
            location = tot_location[:-1]
        else:
            location = tot_location
        
        latitude = X[location[0]]
        cos_longitude = X[location[1]]
        sin_longitude = X[location[2]]
        longitude = np.degrees(np.arctan2(sin_longitude, cos_longitude))
        
        north_america_mask = (latitude > 0) & (longitude >= -180) & (longitude <= -100)
        new_zealand_mask = (latitude >= -50) & (latitude <= -30) & (longitude >= 165) & (longitude <= 180)
        
        if EXCLUDE_LOC:
            X = X.drop(columns=tot_location)
            impl_cols = [col for col in impl_cols if col not in expected_geo_columns]
        
        train_mask = north_america_mask

        nz_stations = [stations[i] for i in range(len(stations)) if new_zealand_mask[i]]
        unique_nz_stations = sorted(list(set(nz_stations)))
        unique_nz_stations = rng.permutation(unique_nz_stations)

        n_eval = len(unique_nz_stations) // 2
        eval_stations = unique_nz_stations[:n_eval]
        test_stations = unique_nz_stations[n_eval:]

        eval_mask = [stations[i] in eval_stations for i in range(len(stations))]
        test_mask = [stations[i] in test_stations for i in range(len(stations))]
        
        X_train = X[train_mask]
        y_train = [y[i] for i in range(len(y)) if train_mask[i]]
        geometries_train = [geometries[i] for i in range(len(geometries)) if train_mask[i]]
        
        X_test = X[test_mask]
        y_test = [y[i] for i in range(len(y)) if test_mask[i]]
        geometries_test = [geometries[i] for i in range(len(geometries)) if test_mask[i]]
        
        X_eval = X[eval_mask]
        y_eval = [y[i] for i in range(len(y)) if eval_mask[i]]
        geometries_eval = [geometries[i] for i in range(len(geometries)) if eval_mask[i]]
        
        test_start_index = [start_index[i] for i in range(len(start_index)) if test_mask[i]]
        train_stations = [stations[i] for i in range(len(stations)) if train_mask[i]]
        test_stations = [stations[i] for i in range(len(stations)) if test_mask[i]]
        eval_stations = [stations[i] for i in range(len(stations)) if eval_mask[i]]
        
        if OPTIMAL_PARAMS:
            # Delete evaluation sets, as the algorithm should not be trained using stations from new zealand, 
            # but the evaluation should also not be applied on stations it already has seen during optimization
            X_eval, y_eval, geometries_eval, eval_stations = pd.DataFrame(), [], [], []
    else: 
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
        X_train, y_train, geometries_train = (
            X[train_mask],
            [y[i] for i in range(len(y)) if train_mask[i]],
            [geometries[i] for i in range(len(geometries)) if train_mask[i]]
        )
        X_eval, y_eval, geometries_eval = (
            X[eval_mask],
            [y[i] for i in range(len(y)) if eval_mask[i]],
            [geometries[i] for i in range(len(geometries)) if eval_mask[i]]
        )
        X_test, y_test, test_start_index, geometries_test = (
            X[test_mask],
            [y[i] for i in range(len(y)) if test_mask[i]],
            [start_index[i] for i in range(len(start_index)) if test_mask[i]],
            [geometries[i] for i in range(len(geometries)) if test_mask[i]]
        )
        
        train_stations = [stations[i] for i in range(len(stations)) if train_mask[i]]
        eval_stations = [stations[i] for i in range(len(stations)) if eval_mask[i]]
        test_stations = [stations[i] for i in range(len(stations)) if test_mask[i]]
        
        if OPTIMAL_PARAMS:
            # Combine train and eval, as no evaluation set needed anymore if we are not tuning the hyperparameters
            X_train = pd.concat([X_train, X_eval], axis=0)
            y_train = pd.concat([pd.Series(y_train), pd.Series(y_eval)], axis=0)
            train_stations += eval_stations
            geometries_train += geometries_eval
            X_eval, y_eval, geometries_eval, eval_stations = pd.DataFrame(), [], [], []
            
    def scale_features_per_station(X, station_list):
        if X.empty:
            return X
        unique_stations = set(station_list)
        scaled_data = pd.DataFrame(index=X.index, columns=X.columns)
        geo_columns = []

        expected_geo_columns = ['latitude', 'sin_longitude', 'cos_longitude', 'height']

        if any(col in impl_cols for col in expected_geo_columns):
            num_geo_cols = sum(col in impl_cols for col in expected_geo_columns)
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

    X_train = scale_features_per_station(X_train, train_stations)
    X_eval = scale_features_per_station(X_eval, eval_stations)
    X_test = scale_features_per_station(X_test, test_stations)
    
    # Oversample the minority classes
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

    # Compute class weights to handle remaining imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {c: w for c, w in zip(classes, class_weights)}
    
    return pd.DataFrame(X_train), pd.DataFrame(X_eval), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_eval), pd.Series(y_test), class_weights, test_start_index, test_stations, (geometries_train, geometries_eval, geometries_test), impl_cols

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
    
    X_train, X_eval, X_test, y_train, y_eval, y_test, weights, test_start_index, test_stations, geometries, impl_cols = prepare_data(X, y, start_index, stations, geometries)

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
        model = optimize_xgboost(X_train, y_train, X_eval, y_eval, weights)
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
        test_predictions = evaluate(test_predictions, test_labels, model=model, X_test=X_test, tolerance_window=window, cleaned_dfs=cleaned_dfs, stations=test_stations, start_indices=test_start_index, impl_cols=impl_cols)
    
    return model, geometries

def apply_simulations(simulated_dfs, interpolate, model):
    """
    Applies a trained machine learning model to simulated datasets and evaluates its performance.

    This function generates synthetic offsets for the given datasets, extracts relevant features, 
    makes predictions using a trained model, and evaluates the model's performance. 
    It also handles the evaluation of predictions with different tolerance windows.

    Parameters:
    - simulated_dfs (list): A list of dataframes, which are used to generate synthetic offsets.
    - interpolate (bool): A flag indicating whether interpolation should be applied during feature extraction.
    - model (sklearn): A trained machine learning model to be applied to the test data.
    
    Returns:
    None
    """
        
    simulated_dfs = generate_synthetic_offsets(simulated_dfs)
    
    X_test, y_test, start_index_test, stations_test, _ = extract_features(simulated_dfs, interpolate=interpolate)
    
    test_predictions = model.predict(X_test)
    
    test_predictions = pd.Series(test_predictions, index=start_index_test)
    y_test = pd.Series(y_test, index=start_index_test)
    stations_test = pd.Series(stations_test, index=start_index_test)
    
    for window in [None, 1]:
        test_predictions = evaluate(test_predictions, y_test, model=model, X_test=X_test, tolerance_window=window, cleaned_dfs=simulated_dfs, stations=stations_test, start_indices=start_index_test, simulated=True)

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
    dfs = []
    dir = Path(DATA_PATH)
    for file_path in dir.iterdir():
        if file_path.is_file():
            dfs.append(read_file(file_path.name))

    cleaned_dfs, simulated_dfs = clean_dataframes(dfs, missing_value_threshold=0, limited_period=True, minimal_offset=5)
    
    df_dict = {f'df_{i}': df for i, df in enumerate(cleaned_dfs)}
    pd.to_pickle(df_dict, CLEANED_DATA_PATH)
    
    # HistGradientBoosting designed to deal with None data -> No interpolation needed
    interpolate = False if MODEL_TYPE == 'HistGradientBoosting' else True
    X, y, start_index, stations, geometries = extract_features(cleaned_dfs, interpolate=interpolate)
    
    X.to_csv(f'{FEATURES_PATH}_features.csv', index=True)
    pd.Series(y).to_csv(f'{FEATURES_PATH}_target.csv', index=False, header=False)
    
    model, geometries = train_model(X, y, start_index, stations, model_type=MODEL_TYPE, cleaned_dfs=cleaned_dfs, geometries=geometries)
    
    plot_station_geometries(geometries)
    
    # Simulations excluded due to their bad accuracy
    # apply_simulations(simulated_dfs, interpolate, model)
    
    sys.stdout = sys.__stdout__
    log.close()

if __name__=='__main__':
    main()