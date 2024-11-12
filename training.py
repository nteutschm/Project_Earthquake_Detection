#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np
import re
from shapely.geometry import Point
from pathlib import Path
from collections import Counter
import joblib
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, f1_score
from xgboost import XGBClassifier, callback
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import contextily as ctx
import seaborn as sns
import sys
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_contour

# Runs in the backgound:
import optuna.integration # pip install optuna-integration[xgboost]

pd.options.mode.chained_assignment = None


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

# Stores all print information
LOG_FILE = f'/cluster/home/nteutschm/eqdetection/logs/{MODEL_TYPE}_logs.txt'

LOAD_MODEL = False # If already trained model is saved under MODEL_PATH, it can be loaded if set to True to skip the entire training process

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


def get_offsets(header_lines):
    """
    Extracts offset and postseismic decay information from the header lines of a GNSS file.

    The function captures both coseismic and non-coseismic offsets, along with postseismic decays, for 
    north (N), east (E), and up (U) components. It parses lines starting with '#' and collects the relevant 
    values into a structured dictionary categorized by the component.

    Parameters:
    - header_lines (list of str): Lines from the file that contain metadata and comments starting with '#'.

    Returns:
    - components (dict): A dictionary with keys 'n', 'e', and 'u' representing the north, east, and up components.
      Each component holds a dictionary with:
        - 'offsets': A list of dictionaries containing offset information (value, error, date, coseismic flag).
        - 'ps_decays': A list of dictionaries containing postseismic decay information (value, error, tau, date, type).
    """
    
    # Capture important information from the header
    offset_pattern = re.compile(r"#\s*(\*?)\s*offset\s+\d+:?\s+([-\d.]+)\s+\+/\-\s+([-\d.]+)\s+mm.*?\((\d{4}-\d{2}-\d{2}).*?\)")
    ps_decay_pattern = re.compile(r'#!?\s*ps decay\s+\d:\s*(-?\d+\.\d+)\s+\+/-\s+(\d+\.\d+)\s+mm\s+\((\d{4}-\d{2}-\d{2})\s+\[(\d{4}\.\d+)\]\);\s*tau:\s*(\d+)\s+days')
    component_pattern = re.compile(r"#\s+([neu])\s+component")

    components = {'n': {'offsets': [], 'ps_decays': []}, 'e': {'offsets': [], 'ps_decays': []}, 'u': {'offsets': [], 'ps_decays': []}}
    current_component = None

    for line in header_lines:
        comp_match = component_pattern.match(line)
        if comp_match:
            current_component = comp_match.group(1)
            continue

        # Check for offset
        offset_match = offset_pattern.match(line)
        if offset_match and current_component:
            coseismic = bool(offset_match.group(1))  # True if * present, meaning coseismic
            offset_value = float(offset_match.group(2))
            offset_error = float(offset_match.group(3))
            offset_date = offset_match.group(4)
            components[current_component]['offsets'].append({
                'value': offset_value,
                'error': offset_error,
                'date': offset_date,
                'coseismic': coseismic
            })

        # Check for postseismic decay
        ps_decay_match = ps_decay_pattern.match(line)
        if ps_decay_match and current_component:
            decay_value = float(ps_decay_match.group(1))
            decay_error = float(ps_decay_match.group(2))
            decay_date = ps_decay_match.group(3)
            tau = int(ps_decay_match.group(5))
            # Determine decay type based on the presence of '!'
            decay_type = 'logarithmic' if '!' in line else 'exponential'
            components[current_component]['ps_decays'].append({
                'value': decay_value,
                'error': decay_error,
                'tau': tau,
                'date': decay_date,
                'type': decay_type
            })

    return components

def read_file(filename):
    """
    Reads a GNSS file, extracting both header and data information into a pandas DataFrame.

    The function processes the header to extract metadata (e.g., station coordinates, height, offsets, decays) 
    and processes the data section to extract time-series GNSS measurements. It combines these into a DataFrame 
    with attributes containing additional metadata.

    Parameters:
    - filename (str): The path to the file containing GNSS data.

    Returns:
    - data (pandas.DataFrame): A DataFrame containing the time-series GNSS data (N, E, U components, sigmas, correlations),
      indexed by date. The DataFrame has additional attributes storing station geometry (latitude, longitude), height, 
      and offset/decay information.
    """
    
    with open(DATA_PATH+filename, 'r') as file:
        lines = file.readlines()

    header_lines = [line for line in lines if line.startswith('#')]
    if header_lines:
        column_names = re.split(r'\s{2,}', header_lines[-1].lstrip('#').strip())
    else:
        column_names = []
        
    data_lines = []
    for line in lines:
        if not line.startswith('#'):
            parts = line.strip().split()
            # Check if the number of parts matches the expected number of columns
            if len(parts) < len(column_names):
                # Add None for missing values
                parts.extend([None] * (len(column_names) - len(parts)))
            data_lines.append(parts)

    data = pd.DataFrame(data_lines)
    data.columns = column_names
    
    # Extracts latitude, longitude and height
    pattern = r'Latitude\(DD\)\s*:\s*(-?\d+\.\d+)|East Longitude\(DD\)\s*:\s*(-?\d+\.\d+)|Height\s*\(M\)\s*:\s*(-?\d+\.\d+)'
    matches = re.findall(pattern, ' '.join(header_lines))
    geom = Point(float(matches[1][1]), float(matches[0][0]))
    
    offsets = get_offsets(header_lines)

    data['Date'] = pd.to_datetime(data['Yr'].astype(str) + data['DayOfYr'].astype(str), format='%Y%j')
    data.set_index('Date', inplace=True)
    data.drop(['Dec Yr', 'Yr', 'DayOfYr', 'Chi-Squared'], axis=1, inplace=True)
    cols = ['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU']
    data[cols] = data[cols].astype(float)
    
    data.name = filename.replace("RawTrend.neu", "")
    data.attrs['geometry'] = geom
    data.attrs['height'] = float(matches[2][2])
    data.attrs['offsets'] = offsets
    
    return data

def add_missing_dates(df):
    """
    This function takes a DataFrame with a datetime index and reindexes it to include
    all dates in the range from the minimum to the maximum date present in the index.
    Missing dates are filled with NaN values, ensuring that the DataFrame retains its 
    original structure while providing a complete date range.

    Parameters:
    df (DataFrame): The input DataFrame with a datetime index that may contain missing dates.

    Returns:
    DataFrame: A new DataFrame with a complete date range as its index, with NaN values 
    for any missing dates.
    """
    df.index = pd.to_datetime(df.index)
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_full = df.reindex(full_date_range)
    df_full.name = df.name
    return df_full

def clean_dataframes(dfs, missing_value_threshold=None, limited_period=False, minimal_offset=0):
    """
    Cleans the dataframes by:
    1. Removing dataframes without any coseismic offsets in any of the 3 components (n, e, u).
    2. Removing non-coseismic offsets from all components.
    3. Optionally removing dataframes with excessive missing values in all 3 components.
    4. Optionally limiting data to a random period around coseismic offsets. The start and end of the 
       period are randomly determined between 100 and 365 days before the first coseismic offset and 
       after the last coseismic offset, ensuring reproducibility using the specified random state.
    5. Optionally selecting only coseismic offsets with absolute values greater than 'minimal_offset'.

    Parameters:
    dfs (list): List of dataframes with GNSS data.
    missing_value_threshold (float, optional): Percentage (0 to 1) of allowed missing values.
                                               If exceeded, the dataframe is removed.
    limited_period (bool, optional): Whether to limit the data to a random period around the coseismic 
                                     offsets. Uses random state for reproducibility.
    minimal_offset (float, optional): Minimum offset magnitude to consider for coseismic offsets.

    Returns:
    list: Cleaned list of dataframes.
    """
    
    if limited_period:
        rng = np.random.default_rng(seed=RANDOM_STATE)

    cleaned_dfs = []
    components = ['N', 'E', 'U']
    components_offsets = ['n', 'e', 'u']
    simulated_dfs = []

    for df in dfs:
        
        name = df.name
        
        has_coseismic = False
        df = add_missing_dates(df)

        # Determine the range of coseismic offsets
        first_coseismic_date = None
        last_coseismic_date = None
        
        for comp in components_offsets:
            filtered_offsets = []
            for offset in df.attrs['offsets'][comp]['offsets']:
                if offset['coseismic'] and abs(offset['value']) >= minimal_offset:
                    has_coseismic = True
                    filtered_offsets.append(offset)
                    offset_date = pd.to_datetime(offset['date'])
                    if first_coseismic_date is None or offset_date < first_coseismic_date:
                        first_coseismic_date = offset_date
                    if last_coseismic_date is None or offset_date > last_coseismic_date:
                        last_coseismic_date = offset_date
            # Update offsets to retain only coseismic
            df.attrs['offsets'][comp]['offsets'] = filtered_offsets

        # Skip dataframe if no coseismic offsets in any component
        if not has_coseismic:
            if missing_value_threshold is not None:
                total_values = sum(df[comp].size for comp in components)
                missing_values = sum(df[comp].isna().sum() for comp in components)

                missing_percentage = missing_values / total_values
                if missing_percentage > missing_value_threshold:
                    continue  # Skip the dataframe if missing values exceed the threshold
            df.name = name
            simulated_dfs.append(df)
            continue

        # Trim data to include the range around the coseismic offsets if days_included is provided
        # Data can not be interpolated if there are None values at the edges -> Prevent this
        if first_coseismic_date and limited_period:
            start_date = first_coseismic_date - pd.Timedelta(days=rng.integers(NBR_OF_DAYS[0], NBR_OF_DAYS[1]))
            end_date = last_coseismic_date + pd.Timedelta(days=rng.integers(NBR_OF_DAYS[0], NBR_OF_DAYS[1]))
            start_date = max(start_date, df.index[0])
            end_date = min(end_date, df.index[-1])

            while start_date <= first_coseismic_date - pd.Timedelta(days=NBR_OF_DAYS[0]):
                if pd.isna(df.loc[start_date.strftime('%Y-%m-%d')]).any():
                    start_date = start_date + pd.Timedelta(days=1) 
                else:
                    break
            else:
                continue

            while end_date >= last_coseismic_date + pd.Timedelta(days=NBR_OF_DAYS[0]):
                if pd.isna(df.loc[end_date.strftime('%Y-%m-%d')]).any():
                    end_date = end_date - pd.Timedelta(days=1)
                else:
                    break
            else:
                continue
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Check missing values for all components combined, if threshold is provided
        if missing_value_threshold is not None:
            total_values = sum(df[comp].size for comp in components)
            missing_values = sum(df[comp].isna().sum() for comp in components)

            missing_percentage = missing_values / total_values
            if missing_percentage > missing_value_threshold:
                continue  # Skip the dataframe if missing values exceed the threshold
        df.name = name
        cleaned_dfs.append(df)

    return cleaned_dfs, simulated_dfs

def extract_features(dfs, interpolate=True, chunk_size=CHUNK_SIZE):
    """
    Extracts relevant features from a list of dataframes, including displacement values, 
    errors, offsets, decay information, station locations, and heights.

    Parameters:
    dfs (list): List of dataframes with GNSS data.
    interpolate (bool): Whether to interpolate missing values or retain `None`.
    chunk_size (int): Number of consecutive days to combine into one sample (row).

    Returns:
    Tuple (DataFrame, list): Combined dataframe with extracted features, and the target vector.
    """
    feature_matrix = []
    target_vector = []
    time_index = []
    station_names = []
    components_offsets = ['n', 'e', 'u'] 
    geometries = []
    non_chunked_columns = ['latitude', 'cos_longitude', 'sin_longitude', 'height']
    
    cols = USED_COLS
    for df in dfs:
        # Extract basic features (displacement, errors, correlations)
        features = df[['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU']].copy()
        
        # Only necessary if missing_value_thershold was bigger than 0 in the clean_dataframes function
        if interpolate:
            features.interpolate(method='time', inplace=True)

        # Get station location and height information
        location = df.attrs.get('geometry')
        latitude, longitude = location.y, location.x
        height = df.attrs.get('height')

        # Extract offsets and decay information for each component
        for comp in components_offsets:
            series_names = ['offset_value', 'offset_error', 'decay_value', 'decay_error', 'decay_tau']
            series_dict = {name: pd.Series(0.0, dtype='float64', index=df.index) for name in series_names}
            series_dict['decay_type'] = pd.Series(0, dtype='int64', index=df.index)

            for offset in df.attrs['offsets'][comp]['offsets']:
                series_dict['offset_value'].loc[offset['date']] = offset['value']
                series_dict['offset_error'].loc[offset['date']] = offset['error']

            for decay in df.attrs['offsets'][comp]['ps_decays']:
                series_dict['decay_value'].loc[decay['date']] = decay['value']
                series_dict['decay_error'].loc[decay['date']] = decay['error']
                series_dict['decay_tau'].loc[decay['date']] = decay['tau']
                series_dict['decay_type'].loc[decay['date']] = 1 if decay['type'] == 'logarithmic' else 2

            # Add series to features
            for name, series in series_dict.items():
                features[f'{comp}_{name}'] = series
        
        # Encode the longitude using cosine and sine to take into account that 0° and 360° refer to the same point
        longitude_radians = np.radians(longitude)
        features['sin_longitude'] = np.sin(longitude_radians)
        features['cos_longitude'] = np.cos(longitude_radians)

        # Add other station metadata (location and height)
        features['latitude'] = latitude
        features['height'] = height

        # Create the feature matrix with chunking -> chunks are only created for the columns that were specified earlier in the cols variable
        for i in range(len(features) - chunk_size + 1):
            # Create a chunk of size `chunk_size` for each feature
            feature_row = []

            for col in cols:
                if col in non_chunked_columns:
                    # Add the value of the column directly without chunking
                    feature_row.append(features[col].values[i])  # Just take the value at position i
                else:
                    # For columns that should be chunked, take the chunk
                    feature_row.extend(features[col].values[i:i + chunk_size])
            feature_matrix.append(feature_row)
            
            time_index.append(features.index[i])
            station_names.append(df.name)
            geometries.append(df.attrs.get('geometry'))
            
            offset_values_chunk = features[['n_offset_value', 'e_offset_value', 'u_offset_value']].iloc[i:i + chunk_size]
            
            if MODEL_TYPE=='IsolationForest':
                # For Isolation Forest, there are not multiple classes:
                # Determine the target value for this chunk: 1 if earthquake happened in chunk, 0 otherwise
                if (offset_values_chunk != 0).any().any():
                    target_vector.append(1)
                else:
                    target_vector.append(0)
            
            else:    
                # Otherwise actually save the index of when the earthquake occured as well (using the mean over each row to detect the biggest earthquake in the chunk):
                max_mean_index = offset_values_chunk.abs().mean(axis=1).idxmax()
                target_vector.append(offset_values_chunk.index.get_loc(max_mean_index))

    return pd.DataFrame(np.array(feature_matrix)), target_vector, time_index, station_names, geometries

def generate_synthetic_offsets(dfs, num_series=50, offset_range=(-1, 1), exclusion_range=(-0.5, 0.5), random_state=RANDOM_STATE):
    """
    Generates synthetic test data by adding random offsets with random decay and noise to selected dataframes.
    
    Parameters:
    dfs (list): List of dataframes to process.
    num_series (int, optional): Number of dataframes to modify by adding synthetic offsets.
    offset_range (tuple): Full range (min, max) for the offsets to be generated.
    exclusion_range (tuple): Range (min, max) to exclude from one of the offsets.
    decay_rate_range (tuple): Range (min, max) for decay rate.
    noise_level (float): Standard deviation of Gaussian noise to add to the offset.
    random_state (int, optional): Seed for random number generator for reproducibility.
    
    Returns:
    list: List of dataframes with added synthetic offsets.
    """
    
    rng = np.random.default_rng(seed=random_state)
    synthetic_dfs = []
    # Helper function to generate an offset within exclusion range
    def generate_within_exclusion():
        return rng.uniform(*offset_range)
    
    # Helper function to generate an offset outside the exclusion range
    def generate_outside_exclusion():
        while True:
            offset = rng.uniform(*offset_range)
            if offset < exclusion_range[0] or offset > exclusion_range[1]:
                return offset
            
    selected_dfs = rng.choice(np.asarray(dfs, dtype='object'), size=num_series, replace=False)
    for df in selected_dfs:
        random_date = rng.choice(df.index)
        name = df.name
        # Randomly choose one component to have an offset outside the exclusion range
        components = ['n', 'e', 'u']
        comps = ['N', 'E', 'U']
        outlier_component = rng.choice(components)
        
        synthetic_offsets = {}
        for comp in components:
            if comp == outlier_component:
                synthetic_offsets[comp] = generate_outside_exclusion()
            else:
                synthetic_offsets[comp] = generate_within_exclusion()
        
        # Apply offsets to the displacement columns starting from the random date
        for comp, cm in zip(components, comps):
            offset_value = synthetic_offsets[comp]
            start_date = random_date - pd.Timedelta(days=rng.integers(NBR_OF_DAYS[0], NBR_OF_DAYS[1]))
            end_date = random_date + pd.Timedelta(days=rng.integers(NBR_OF_DAYS[0], NBR_OF_DAYS[1]))
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            df.loc[random_date:, cm] += offset_value
            
            if 'offsets' not in df.attrs:
                df.attrs['offsets'] = {c: {'offsets': [], 'ps_decays': []} for c in components}
            
            df.attrs['offsets'][comp]['offsets'].append({
                'date': random_date,
                'value': offset_value,
                'error': 0.0,
                'coseismic': True
            })
            df.attrs['offsets'][comp]['ps_decays'] = [{'date': random_date, 'value': 0, 'error': 0, 'tau': 0, 'type': ''}]
        
        df.name = name
        synthetic_dfs.append(df)
    return synthetic_dfs

def save_csv(data, path):
    """
    Saves the test predictions to a CSV file.

    This function takes a DataFrame containing the predictions made on the test dataset
    and saves it to a specified path defined by the PREDICTIONS_PATH variable.

    Parameters:
    test_predictions (DataFrame): The DataFrame containing the predictions to be saved.

    Returns:
    None
    """
    data.to_csv(path, index=True)
    
def compute_weights(train_labels):
    """
    Computes class weights for handling imbalanced classes in the training dataset.

    This function calculates the weights for each class in the training labels to address 
    class imbalance. The weights are computed using a balanced scheme, which assigns a 
    larger weight to underrepresented classes and a smaller weight to overrepresented ones.

    Parameters:
    train_labels (array-like): The target labels for the training dataset.

    Returns:
    dict: A dictionary mapping each class to its corresponding weight.
    """
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    return {c: w for c, w in zip(classes, class_weights)}

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

        if 'latitude' in USED_COLS:
            geo_columns = X.columns[-3:]
        else:
            geo_columns = []

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
    class_weights = compute_weights(y_train)
    
    return pd.DataFrame(X_train), pd.DataFrame(X_eval), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_eval), pd.Series(y_test), class_weights, test_start_index, test_stations, (geometries_train, geometries_eval, geometries_test)

def print_callback(study, trial):
    '''Print some useful information regarding the hyperparameter optimization'''
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    print('-----------------------------------------------------------------------')

def plot_eval_metrics(eval_results, name):
    epochs = len(eval_results['validation_0']['mlogloss'])
    x_axis = range(epochs)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis, eval_results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, eval_results['validation_1']['mlogloss'], label='Eval')
    ax.set_title('XGBoost Log Loss Over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log Loss')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()

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
    
    ax = plot_optimization_history(study, target_name='Validation loss')
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Validation loss (Macro F1 Score)', fontsize=20)
    ax.set_title('Optimization history', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'optimization_history.png')
    
    ax = plot_param_importances(study)
    ax.set_xlabel('Importance for validation loss', fontsize=20)
    ax.set_ylabel('Hyperparameter', fontsize=20)
    ax.set_title('Hyperparameter Importances', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'parameter_importances.png')
    
    ax = plot_contour(study, params=['n_estimators', 'max_depth'], target_name='Validation loss')
    ax.set_xlabel('Estimators', fontsize=20)
    ax.set_ylabel('Max depth', fontsize=20)
    ax.set_title('Contour plot', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'contour.png')
    
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
        
        pruning = optuna.integration.XGBoostPruningCallback(trial, "validation_0-mlogloss")
        
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
    
    ax = plot_optimization_history(study, target_name='Validation loss')
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Validation loss (Macro F1 Score)', fontsize=20)
    ax.set_title('Optimization history', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'optimization_history.png')
    
    ax = plot_param_importances(study)
    ax.set_xlabel('Importance for validation loss', fontsize=20                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               )
    ax.set_ylabel('Hyperparameter', fontsize=20)
    ax.set_title('Hyperparameter Importances', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'parameter_importances.png')
    
    ax = plot_contour(study, params=['learning_rate', 'max_depth'], target_name='Validation loss')
    ax.set_xlabel('Learning rate', fontsize=20)
    ax.set_ylabel('Max depth', fontsize=20)
    ax.set_title('Contour plot', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'contour.png')
    
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

def plot_histogram(series, xlabel, ylabel, title, name):
    plt.figure(figsize=(12, 10))
    plt.bar(series.index, series.values, color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_heatmap(heatmap_data, mask, name):
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        cmap='viridis', 
        mask=mask,       
        cbar_kws={'label': 'Log(Number of Misclassifications)'},
        linewidths=0.5,
        linecolor='lightgrey'
    )
    tick_step = 6  
    plt.xticks(ticks=np.arange(0, len(heatmap_data.columns), tick_step), 
            labels=[date.strftime('%Y-%m') for date in heatmap_data.columns[::tick_step]], 
            rotation=45)

    plt.title('Logarithmic Heatmap of Missed Predictions Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Days Missed By')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_cumulative_metrics(cumulative_metrics_df, name):
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_metrics_df['Date'], cumulative_metrics_df['Smoothed Cumulative Accuracy'], label='Cumulative Accuracy', color='blue')
    plt.plot(cumulative_metrics_df['Date'], cumulative_metrics_df['Smoothed Cumulative Precision'], label='Cumulative Precision', color='orange')
    plt.plot(cumulative_metrics_df['Date'], cumulative_metrics_df['Smoothed Cumulative Recall'], label='Cumulative Recall', color='green')
    plt.plot(cumulative_metrics_df['Date'], cumulative_metrics_df['Smoothed Cumulative F1'], label='Cumulative F1 Score', color='purple')

    plt.title('Cumulative Metrics Over Time (Smoothed over 180 days)')
    plt.xlabel('Date')
    plt.ylabel('Metric Value')

    plt.xticks(ticks=cumulative_metrics_df['Date'][::10000],
            labels=cumulative_metrics_df['Date'][::10000].dt.strftime('%Y-%m'), rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def group_confusion_matrix(true_labels, predicted_labels, n_classes=21):
    groups = {'No': [0], 'Early': range(1, 7), 'Middle': range(7, 14), 'Late': range(14, n_classes)}
    
    def map_to_group(label):
        for group, indices in groups.items():
            if label in indices:
                return group
        return None

    true_groups = np.array([map_to_group(label) for label in true_labels])
    pred_groups = np.array([map_to_group(label) for label in predicted_labels])
    
    unique_groups = list(groups.keys())
    
    group_conf_matrix = confusion_matrix(true_groups, pred_groups, labels=unique_groups)
    
    return group_conf_matrix

def plot_statistics(report, name):
    # Extract precision, recall, and f1-score for each class
    precision = []
    recall = []
    f1_score = []
    class_labels = []

    for i in range(21):
        class_key = str(i)
        
        if class_key in report:
            metrics = report[class_key]
            if isinstance(metrics, dict):
                precision.append(metrics.get('precision', 0))
                recall.append(metrics.get('recall', 0))
                f1_score.append(metrics.get('f1-score', 0))
                class_labels.append(class_key)

    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    r1 = np.arange(len(precision))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, precision, color='b', width=bar_width, edgecolor='grey', label='Precision')
    plt.bar(r2, recall, color='g', width=bar_width, edgecolor='grey', label='Recall')
    plt.bar(r3, f1_score, color='r', width=bar_width, edgecolor='grey', label='F1-Score')

    plt.xlabel('Classes', fontweight='bold')
    plt.ylabel('Scores', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(class_labels))], class_labels)

    plt.title('Precision, Recall, and F1 Scores per Class')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()

def plot_neu_data_with_labels_predictions(df, idx, name, zoom_window=10):
    """
    Plots the N, E, and U columns from a DataFrame with datetime index on the x-axis in separate subplots,
    including 'labels_sum' and 'preds_sum' as bar plots in each subplot on a secondary y-axis,
    with zoomed-in views around dates where 'labels_sum' > 0.

    Parameters:
    df (pd.DataFrame): The input DataFrame with columns 'N', 'E', 'U', 'labels_sum', 'preds_sum', and a datetime index.
    idx (int): Index or identifier for the plot filename.
    name (str): Name identifier for the plot filename.
    zoom_window (int): Number of days before and after the event to zoom in on.
    """
    
    df.index = pd.to_datetime(df.index)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Displacement and Earthquakes Over Time for Station: {df.name}', fontsize=16)

    # Plot for 'N'
    axs[0].plot(df.index, df['N'], color='blue', label='N')
    ax2_n = axs[0].twinx()  # Create a secondary y-axis
    ax2_n.bar(df.index, df['labels_sum'], width=1, color='gray', alpha=0.3, label='True Labels')
    ax2_n.bar(df.index, df['preds_sum'], width=1, color='orange', alpha=0.3, label='Predictions')
    axs[0].set_ylabel('N Displacement')
    ax2_n.set_ylabel('Counts')
    axs[0].legend(loc='upper left')
    ax2_n.legend(loc='upper right')

    # Plot for 'E'
    axs[1].plot(df.index, df['E'], color='blue', label='E')
    ax2_e = axs[1].twinx()  # Create a secondary y-axis
    ax2_e.bar(df.index, df['labels_sum'], width=1, color='gray', alpha=0.3)
    ax2_e.bar(df.index, df['preds_sum'], width=1, color='orange', alpha=0.3)
    axs[1].set_ylabel('E Displacement')
    ax2_e.set_ylabel('Counts')

    # Plot for 'U'
    axs[2].plot(df.index, df['U'], color='blue', label='U')
    ax2_u = axs[2].twinx()  # Create a secondary y-axis
    ax2_u.bar(df.index, df['labels_sum'], width=1, color='gray', alpha=0.3)
    ax2_u.bar(df.index, df['preds_sum'], width=1, color='orange', alpha=0.3)
    axs[2].set_ylabel('U Displacement')
    ax2_u.set_ylabel('Counts')

    axs[2].set_xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}{idx}.png')
    plt.close()

    # Zoomed-In Plot(s)
    zoom_dates = df[df['labels_sum'] > 0].index

    for i, date in enumerate(zoom_dates):
        start_date = date - pd.Timedelta(days=zoom_window)
        end_date = date + pd.Timedelta(days=zoom_window)
        zoom_df = df.loc[start_date:end_date]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Zoomed Displacement and Earthquakes for Station {df.name} around {date.date()}', fontsize=16)

        # Zoom for 'N'
        axs[0].plot(zoom_df.index, zoom_df['N'], color='blue', label='N')
        ax2_zoom_n = axs[0].twinx()  # Create a secondary y-axis
        ax2_zoom_n.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='gray', alpha=0.3, label='True Labels')
        ax2_zoom_n.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='orange', alpha=0.3, label='Predictions')
        axs[0].set_ylabel('N Displacement')
        ax2_zoom_n.set_ylabel('Counts')
        axs[0].legend(loc='upper left')
        ax2_zoom_n.legend(loc='upper right')

        # Zoom for 'E'
        axs[1].plot(zoom_df.index, zoom_df['E'], color='blue', label='E')
        ax2_zoom_e = axs[1].twinx()  # Create a secondary y-axis
        ax2_zoom_e.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='gray', alpha=0.3)
        ax2_zoom_e.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='orange', alpha=0.3)
        axs[1].set_ylabel('E Displacement')
        ax2_zoom_e.set_ylabel('Counts')

        # Zoom for 'U'
        axs[2].plot(zoom_df.index, zoom_df['U'], color='blue', label='U')
        ax2_zoom_u = axs[2].twinx()  # Create a secondary y-axis
        ax2_zoom_u.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='gray', alpha=0.3)
        ax2_zoom_u.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='orange', alpha=0.3)
        axs[2].set_ylabel('U Displacement')
        ax2_zoom_u.set_ylabel('Counts')

        axs[2].set_xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_PATH}_{name}_zoom_{idx}_{i}.png')
        plt.close()
    
def dechunk_labels_predictions(dfs, chunk_size=CHUNK_SIZE):
    time_series = []
    
    for df in dfs:
        df.index = pd.to_datetime(df.index)
        start_date = df.index.min()
        end_date = df.index.max() + pd.Timedelta(days=chunk_size)
        continuous_index = pd.date_range(start=start_date, end=end_date)

        time_series_df = pd.DataFrame(index=continuous_index, columns=['labels_sum', 'preds_sum'], dtype='int64')
        time_series_df.fillna(0, inplace=True)

        for _, row in df.iterrows():
            if row['labels'] > 0:
                label_date = row.name + pd.Timedelta(days=row['labels'] + 1)
                time_series_df.loc[label_date, 'labels_sum'] += 1
            if row['preds'] > 0:
                prediction_date = row.name + pd.Timedelta(days=row['preds'] + 1)
                time_series_df.loc[prediction_date, 'preds_sum'] += 1

        prediction_dates = time_series_df[time_series_df['preds_sum'] > 0].index
        mean_prediction_days = []

        i = 0
        while i < len(prediction_dates):
            cluster = [prediction_dates[i]]
            j = i + 1

            while j < len(prediction_dates) and (prediction_dates[j] - cluster[-1]).days < chunk_size:
                cluster.append(prediction_dates[j])
                j += 1
            
            timestamps = list(map(pd.Timestamp.timestamp, cluster))
            mean_timestamp = sum(timestamps) / len(cluster)
            mean_day = pd.to_datetime(mean_timestamp, unit='s').normalize()
            mean_prediction_days.append(mean_day)
            i = j

        time_series_df['mean_day_pred'] = 0
        time_series_df.loc[mean_prediction_days, 'mean_day_pred'] = 1

        time_series_df.name = df.attrs['station']
        time_series.append(time_series_df)

    return time_series


def combined_df(cleaned_dfs, dfs):
    """
    Combines cleaned DataFrames (with labels and predictions) with additional DataFrames
    based on matching station names and adds mean prediction days for earthquake days.

    Parameters:
    cleaned_dfs (list of pd.DataFrame): List of DataFrames containing labels and predictions.
    dfs (list of pd.DataFrame): List of additional DataFrames with station information.

    Returns:
    list of pd.DataFrame: A list of combined DataFrames for each station.
    """
    total_dfs = []

    cleaned_dict = {df.name: df for df in cleaned_dfs}

    for additional_df in dfs:
        station_name = additional_df.name
        if station_name in cleaned_dict:
            cleaned_df = cleaned_dict[station_name]
            cleaned_df.index = pd.to_datetime(cleaned_df.index)
            additional_df.index = pd.to_datetime(additional_df.index)

            filtered_cleaned_df = cleaned_df.loc[cleaned_df.index.intersection(additional_df.index)]
            combined_df = pd.concat([filtered_cleaned_df, additional_df.loc[filtered_cleaned_df.index]], axis=1)
            combined_df.name = station_name
            total_dfs.append(combined_df)

    return total_dfs

def plot_full_and_zoomed_displacement(df, name, idx, zoom_window=10):
    """
    Combines full plot and zoomed-in plots for N, E, and U displacements with earthquake labels and mean predictions as vertical lines.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'N', 'E', 'U', 'labels_sum', and 'mean_day_pred'.
    station_name (str): Name of the GPS station for the title.
    zoom_window (int): Number of days before and after each event to include in the zoomed plot.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    station_name = df.name
    fig.suptitle(f"Station: {station_name} - Earthquake Labels and Predicted Dates (Full View)", fontsize=16)

    for i, component in enumerate(['N', 'E', 'U']):
        axs[i].plot(df.index, df[component], label=f"{component} Displacement", color='gray')
        for label_date in df[df['labels_sum'] > 0].index:
            axs[i].axvline(label_date, color='blue', linestyle='-', label="Actual Earthquake Label" if i == 0 else "")
        
        for pred_date in df[df['mean_day_pred'] > 0].index:
            axs[i].axvline(pred_date, color='red', linestyle='--', label="Predicted Earthquake (Mean)" if i == 0 else "")

        axs[i].set_ylabel(f"{component} Displacement")
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    axs[-1].set_xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}_{idx}.png')
    plt.close()
    
    zoom_dates = df[df['labels_sum'] > 0].index

    for date in zoom_dates:
        start_date = date - pd.Timedelta(days=zoom_window)
        end_date = date + pd.Timedelta(days=zoom_window)
        zoom_df = df.loc[start_date:end_date]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Zoomed View for Station: {station_name} around {date.date()}', fontsize=16)

        for i, component in enumerate(['N', 'E', 'U']):
            axs[i].plot(zoom_df.index, zoom_df[component], color='gray', label=f"{component} Displacement")
            for label_date in zoom_df[zoom_df['labels_sum'] > 0].index:
                axs[i].axvline(label_date, color='blue', linestyle='-', label="Actual Earthquake Label" if i == 0 else "")
            
            for pred_date in zoom_df[zoom_df['mean_day_pred'] > 0].index:
                axs[i].axvline(pred_date, color='red', linestyle='--', label="Predicted Earthquake (Mean)" if i == 0 else "")

            axs[i].set_ylabel(f"{component} Displacement")
            axs[i].legend(loc='upper right')
            axs[i].grid(True)

        axs[-1].set_xlabel("Date")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_PATH}_{name}_zoom_{idx}.png')
        plt.close()

def calculate_prediction_statistics(dfs, name, chunk_size=CHUNK_SIZE):
    """
    Calculates and prints statistics on prediction accuracy.
    
    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames, one per station, containing 'mean_day_pred' and 'labels_sum'.
    chunk_size (int): Tolerance window (in days) to match predictions with test labels.
    """
    total_labels = 0
    predicted_within_window = 0
    predicted_exact = 0
    undetected = 0

    for df in dfs:
        test_label_dates = df[df['labels_sum'] > 0].index
        prediction_dates = df[df['mean_day_pred'] > 0].index

        total_labels += len(test_label_dates)

        for label_date in test_label_dates:
            if label_date in prediction_dates:
                predicted_exact += 1
            else:
                within_window = any(abs((label_date - pred_date).days) <= chunk_size for pred_date in prediction_dates)
                if within_window:
                    predicted_within_window += 1
                else:
                    undetected += 1

    print("\n=== Mean Prediction Statistics ===")
    print(f"Total earthquakes (test labels): {total_labels}")

    # Calculate key percentages for predictions
    predicted_within_percentage = (predicted_within_window / total_labels) * 100 if total_labels > 0 else 0
    predicted_exact_percentage = (predicted_exact / total_labels) * 100 if total_labels > 0 else 0
    undetected_percentage = (undetected / total_labels) * 100 if total_labels > 0 else 0

    print(f"Predictions within {chunk_size} days:")
    print(f"  Count: {predicted_within_window}")
    print(f"  Percentage: {predicted_within_percentage:.2f}% of total earthquakes")

    print(f"Exact date matches:")
    print(f"  Count: {predicted_exact}")
    print(f"  Percentage: {predicted_exact_percentage:.2f}% of total earthquakes")

    print(f"Undetected earthquakes:")
    print(f"  Count: {undetected}")
    print(f"  Percentage: {undetected_percentage:.2f}% of total earthquakes")

    print("\n==================================\n")
    
    labels = ['Exact Matches', f'Predicted within {chunk_size} Days', 'Undetected']
    values = [predicted_exact_percentage, predicted_within_percentage, undetected_percentage]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=colors, edgecolor='black')
    plt.ylabel('Percentage')
    plt.title("Prediction Accuracy Breakdown")
    plt.ylim(0, 100)
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def calculate_mean_metrics(time_series_data):
    all_labels = []
    all_predictions = []

    for df in time_series_data:
        labels_binary = (df['labels_sum'] > 0).astype(int)
        preds_binary = df['mean_day_pred'].astype(int)

        all_labels.extend(labels_binary)
        all_predictions.extend(preds_binary)

    report = classification_report(all_labels, all_predictions, target_names=['No Earthquake', 'Earthquake'])
    print("Classification Report for mean predictions:\n", report)
    

def evaluation(test_predictions, test_labels, cleaned_dfs, stations, start_indices, chunk_size=CHUNK_SIZE, X_test=None, model=None, tolerance_window=None, simulated=False):
    
    # To avoid overwriting the original predictions when applying a tolerance window
    original_test_predictions = test_predictions.copy()
    tolerance_str = f"_tolerance_{tolerance_window}" if tolerance_window is not None else "_default"
    tolerance_str = f"{tolerance_str}_sim" if simulated else f"{tolerance_str}"
    
    if not simulated and tolerance_window is None:
        print(f'Evaluation of performance for model: {MODEL_TYPE} using columns: {USED_COLS} \n')
    if simulated and tolerance_window is None:
        print('Using simulated data by atrtificially adding random offsets')
    
    if MODEL_TYPE == 'IsolationForest':
        test_predictions = test_predictions.map({-1: 1, 1: 0})
        report = classification_report(test_labels, test_predictions, target_names=['No Coseismic Event', 'Coseismic Event'])
        
        print(f'Evaluation of performance for model (trained solely binary): {MODEL_TYPE} \n')
        print(f"Binary Classification Report: \n {report}")
        
        conf_matrix = confusion_matrix(test_labels, test_predictions)
        print(f"Confusion Matrix: \n{conf_matrix}")
        
        return original_test_predictions
    
    if tolerance_window is not None:
        print(f'Tolerance window is active, classifications that were missed by {tolerance_window} day(s) count as correctly classified for earthquakes (class index > 0).')
        test_labels_np = test_labels.to_numpy()
        test_predictions_np = test_predictions.to_numpy()

        for i in range(len(test_labels_np)):
            # Check if the actual label is an earthquake (index > 0) and it was misclassified
            if test_labels_np[i] > 0 and test_predictions_np[i] != test_labels_np[i]:
                if abs(test_predictions_np[i] - test_labels_np[i]) <= tolerance_window:
                    test_predictions_np[i] = test_labels_np[i]

        test_predictions = pd.Series(test_predictions_np, index=test_labels.index)

    report = classification_report(test_labels, test_predictions)
    print(f'Multi-Class Classification Report: \n {report} \n')
    plot_statistics(classification_report(test_labels, test_predictions, output_dict=True), f'statistics{tolerance_str}')
    
    conf_matrix = group_confusion_matrix(test_labels, test_predictions)
    print(f"Grouped Confusion Matrix by date of earthquake (No, Early, Middle, Late): \n{conf_matrix} \n")
    if isinstance(test_labels, (list, np.ndarray)):
        test_labels = pd.Series(test_labels, index=start_indices)
    if isinstance(test_predictions, (list, np.ndarray)):
        test_predictions = pd.Series(test_predictions, index=start_indices)
    plot_histogram(test_predictions.value_counts().subtract(test_labels.value_counts(), fill_value=0), 'Chunk Index', 'Difference (Predicted - True)', 'Difference between predicted and actual earthquakes at chunk indices', f'hist_predtest{tolerance_str}')
    
    false_positive_rate = np.sum((test_predictions == 1) & (test_labels == 0)) / np.sum(test_labels == 0)
    false_negative_rate = np.sum((test_predictions == 0) & (test_labels == 1)) / np.sum(test_labels == 1)

    print(f"False Positive Rate: {false_positive_rate}")
    print(f"False Negative Rate: {false_negative_rate}")
    
    num_chunks = len(test_labels) // chunk_size
    chunk_indices = np.array_split(np.arange(len(test_labels)), num_chunks)

    chunk_labels = []
    chunk_predictions = []

    for chunk in chunk_indices:
        chunk_label = 1 if np.any(test_labels.iloc[chunk].to_numpy() != 0) else 0
        chunk_pred = 1 if np.any(test_predictions.iloc[chunk].to_numpy() != 0) else 0
        
        chunk_labels.append(chunk_label)
        chunk_predictions.append(chunk_pred)

    chunk_labels = np.array(chunk_labels)
    chunk_predictions = np.array(chunk_predictions)

    chunk_report = classification_report(chunk_labels, chunk_predictions)
    print(f"Chunk-Based Binary Classification Report: {chunk_report} \n")

    time_index = test_labels.index.to_numpy()
    test_labels = test_labels.to_numpy()
    test_predictions = test_predictions.to_numpy()

    incorrect_indices = np.where(test_labels != test_predictions)[0]
    index_miss_differences = np.abs(test_labels[incorrect_indices] - test_predictions[incorrect_indices])

    result_df = pd.DataFrame({
        'True Label': test_labels[incorrect_indices],
        'Predicted Label': test_predictions[incorrect_indices],
        'Missed By': index_miss_differences,
        'Index': incorrect_indices,
        'Date': time_index[incorrect_indices]
    })
    
    plot_histogram(result_df['Missed By'].value_counts(), xlabel='Number of Days Missed By', ylabel='Count', title='Histogram of Misclassifications by Number of Days Missed By', name=f'hist_nbrdays{tolerance_str}')
    
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    result_df['Month'] = result_df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
    result_df['Missed By Rounded'] = result_df['Missed By'].round()
    heatmap_data = result_df.pivot_table(index='Missed By Rounded', columns='Month', aggfunc='size', fill_value=0)
    mask = heatmap_data == 0

    plot_heatmap(np.log1p(heatmap_data), mask, f'heatmap{tolerance_str}')
    
    cumulative_metrics_df = pd.DataFrame({
    'Date': time_index, 
    'Labels': test_labels,
    'Predictions': test_predictions
})

    cumulative_metrics_df['Cumulative Correct'] = (cumulative_metrics_df['Labels'] == cumulative_metrics_df['Predictions']).cumsum()
    cumulative_metrics_df['Cumulative Accuracy'] = cumulative_metrics_df['Cumulative Correct'] / np.arange(1, len(cumulative_metrics_df) + 1)

    cumulative_precisions = []
    cumulative_recalls = []
    cumulative_f1_scores = []

    for i in range(1, len(cumulative_metrics_df) + 1):
        current_labels = cumulative_metrics_df['Labels'][:i]
        current_predictions = cumulative_metrics_df['Predictions'][:i]

        precision, recall, f1, _ = precision_recall_fscore_support(current_labels, current_predictions, average='macro', zero_division=0)

        cumulative_precisions.append(precision)
        cumulative_recalls.append(recall)
        cumulative_f1_scores.append(f1)

    cumulative_metrics_df['Cumulative Precision'] = cumulative_precisions
    cumulative_metrics_df['Cumulative Recall'] = cumulative_recalls
    cumulative_metrics_df['Cumulative F1 Score'] = cumulative_f1_scores

    window_size = 180
    cumulative_metrics_df['Smoothed Cumulative Accuracy'] = cumulative_metrics_df['Cumulative Accuracy'].rolling(window=window_size, min_periods=1).mean()
    cumulative_metrics_df['Smoothed Cumulative Precision'] = cumulative_metrics_df['Cumulative Precision'].rolling(window=window_size, min_periods=1).mean()
    cumulative_metrics_df['Smoothed Cumulative Recall'] = cumulative_metrics_df['Cumulative Recall'].rolling(window=window_size, min_periods=1).mean()
    cumulative_metrics_df['Smoothed Cumulative F1'] = cumulative_metrics_df['Cumulative F1 Score'].rolling(window=window_size, min_periods=1).mean()

    cumulative_metrics_df['Date'] = pd.to_datetime(cumulative_metrics_df['Date'])
    cumulative_metrics_df = cumulative_metrics_df.sort_values('Date')
    
    plot_cumulative_metrics(cumulative_metrics_df, f'cumulative_metrics{tolerance_str}')
    
    if X_test is not None and model is not None:
        try:
            probs = model.predict_proba(X_test)
            auc_score = roc_auc_score(test_labels, probs, multi_class='ovr')
            print(f"AUC Score: {auc_score}")
        except AttributeError:
            print("AUC score calculation skipped, as the model does not support probabilities.")

    if model is not None and hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        num_features = len(USED_COLS)
        averaged_importances = np.mean(feature_importances.reshape(-1, num_features), axis=0)
        print(f"Averaged Feature Importances for used columns {USED_COLS}: \n{averaged_importances}")
        
    time_series = pd.concat([pd.Series(stations.values, index=start_indices, name='stations'), pd.Series(test_labels, index=start_indices, name='labels'), pd.Series(test_predictions, index=start_indices, name='preds')], axis=1)
    time_series = [group for _, group in time_series.groupby('stations')]
    for i, df in enumerate(time_series):
        df.attrs['station'] = df['stations'].iloc[0]
        df.drop(columns=['stations'], inplace=True)
        df.sort_index(inplace=True)
    dfs = dechunk_labels_predictions(time_series)
    print('Forming mean predictions over all chunks')
    total_df = combined_df(cleaned_dfs, dfs)
    calculate_prediction_statistics(total_df, f'mean_preds_stats{tolerance_str}')
    calculate_mean_metrics(total_df)
    rng = np.random.default_rng(seed=RANDOM_STATE)
    for idx in range(10):
        plot_neu_data_with_labels_predictions(total_df[rng.integers(0, len(total_df))], idx=idx, name=f'comp_preds{tolerance_str}')
        plot_full_and_zoomed_displacement(total_df[rng.integers(0, len(total_df))], idx=idx, name=f'mean_preds{tolerance_str}')
        
    return original_test_predictions


def extract_unique_geometries(geometries):
    """
    Extract unique geometries from a list of Point geometries.
    
    Parameters:
    geometries (list): List of geometry Points.
    
    Returns:
    GeoSeries: A GeoSeries of unique geometries.
    """
    return gpd.GeoSeries(list(set(geometries)), crs="EPSG:4326")

def plot_station_geometries(geometries, geometries_sim):
    """
    Plots station locations with different colors for train, eval, test, and simulated stations on a basemap.

    Parameters:
    geometries (tuple): A tuple of lists containing geometries for (train, eval, test) stations.
    geometries_sim (list): A list containing geometries for simulated stations.
    """
    # Unpack train, eval, and test station geometries and remove duplicates
    train_stations = extract_unique_geometries(geometries[0])
    eval_stations = extract_unique_geometries(geometries[1])
    test_stations = extract_unique_geometries(geometries[2])
    sim_stations = extract_unique_geometries(geometries_sim)

    gdf = gpd.GeoDataFrame({
        'geometry': pd.concat([train_stations, eval_stations, test_stations, sim_stations]),
        'type': (['Train'] * len(train_stations) +
                 ['Eval'] * len(eval_stations) +
                 ['Test'] * len(test_stations) +
                 ['Simulated'] * len(sim_stations))
    }, crs="EPSG:4326")

    # Transform coordinates to match the basemap's CRS (Web Mercator)
    gdf.to_crs(epsg=3857, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    for station_type, color, marker in zip(['Train', 'Eval', 'Test', 'Simulated'],
                                           ['blue', 'green', 'red', 'orange'],
                                           ['o', 's', '^', 'x']):
        gdf[gdf['type'] == station_type].plot(
            ax=ax, color=color, marker=marker, markersize=50, label=f'{station_type} Stations', alpha=0.7
        )
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Station Locations by Usage Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_stations.png')
    plt.close()

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
    save_csv(test_predictions, PREDICTIONS_PATH)
    save_csv(test_labels, TEST_LABELS_PATH)
    save_csv(test_stations, STATION_NAMES)
    for window in [None, 1]:
        test_predictions = evaluation(test_predictions, test_labels, model=model, X_test=X_test, tolerance_window=window, cleaned_dfs=cleaned_dfs, stations=test_stations, start_indices=test_start_index)
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

    cleaned_dfs, simulated_dfs = clean_dataframes(dfs, missing_value_threshold=0, limited_period=True, minimal_offset=10)
    
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
    
    # Not possible on cluster due to network restrictions
    # plot_station_geometries(geometries, geometries_sim)
    
    test_predictions = model.predict(X_test)
    test_predictions = pd.Series(test_predictions, index=start_index_test)
    y_test = pd.Series(y_test, index=start_index_test)
    stations_test = pd.Series(stations_test, index=start_index_test)
    for window in [None, 1]:
        test_predictions = evaluation(test_predictions, y_test, model=model, X_test=X_test, tolerance_window=window, cleaned_dfs=simulated_dfs, stations=stations_test, start_indices=start_index_test, simulated=True)
    
    sys.stdout = sys.__stdout__
    log.close()

if __name__=='__main__':
    main()