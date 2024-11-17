import pandas as pd
import numpy as np
import re
from shapely.geometry import Point
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D, Conv1D, BatchNormalization, ReLU
from sklearn.utils import resample
import matplotlib.pyplot as plt
import random
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from tensorflow.keras import backend as K
from datetime import datetime, timedelta

EPOCHS_BIN = 120
EPOCHS_LOC = 120
TIME_STEPS = 21
RANDOM_STATE = 10
MIN_OFFSET = 5

DATA_PATH = 'Data/'
MODEL_BINARY_SAVE_PATH = "Models/inclStationId_bin.keras"
MODEL_LOC_SAVE_PATH = "Models/inclStationId_loc.keras"


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


def clean_dataframes(dfs, missing_value_threshold=None, days_included=None, minimal_offset=0, randomize_stations=True, random_state=42):
    """
    Cleans the dataframes by:
    1. Removing dataframes without any coseismic offsets in any of the 3 components (n, e, u).
    2. Removing non-coseismic offsets from all components.
    3. Optionally removing dataframes with excessive missing values in all 3 components.
    4. Optionally keeping only data within a specified range of days before the first coseismic offset
       and after the last coseismic offset, if 'days_included' is provided.
    5. Optionally selecting only coseismic offsets with absolute values greater than 'minimal_offset'.

    Parameters:
    dfs (list): List of dataframes with GNSS data.
    missing_value_threshold (float, optional): Percentage (0 to 1) of allowed missing values.
                                               If exceeded, the dataframe is removed.
    days_included (int, optional): Number of days before and after coseismic offsets to keep.

    Returns:
    list: Cleaned list of dataframes.
    """

    cleaned_dfs = []
    station_ids = list(range(len(dfs)))
    components = ['N', 'E', 'U']
    components_offsets = ['n', 'e', 'u']

    if randomize_stations is True:
        random.seed(random_state) 
        random.shuffle(station_ids)
        dfs_shuffled = [dfs[i] for i in station_ids] 

    for org_df in dfs_shuffled:
        
        has_coseismic = False
        df = add_missing_dates(org_df)

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

        # Skip dataframe if no coseismic offsets in any component, possibly change to add simulated data
        if not has_coseismic:
            continue

        # Trim data to include the range around the coseismic offsets if days_included is provided
        if first_coseismic_date and days_included is not None:
            start_date = first_coseismic_date - pd.Timedelta(days=days_included)
            end_date = last_coseismic_date + pd.Timedelta(days=days_included)
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Check missing values for all components combined, if threshold is provided
        if missing_value_threshold is not None:
            total_values = sum(df[comp].size for comp in components)
            missing_values = sum(df[comp].isna().sum() for comp in components)

            missing_percentage = missing_values / total_values
            if missing_percentage > missing_value_threshold:
                continue  # Skip the dataframe if missing values exceed the threshold

        cleaned_dfs.append(df)

    return cleaned_dfs, station_ids

def extract_features(dfs, station_ids, interpolate=True, chunk_size=21):
    """
    Extracts relevant features from a list of dataframes, including displacement values, 
    errors, offsets, decay information, station locations, and heights.

    Parameters:
    dfs (list): List of dataframes with GNSS data.
    interpolate (bool): Whether to interpolate missing values or retain `None`.
    chunk_size (int): Number of consecutive days to combine into one sample (row).

    Returns:
    Tuple (DataFrame, list, list): Combined dataframe with extracted features, the binary target vector,
                                   and the localization vector indicating the offset day.
    """
    feature_matrix = []
    target_vector = []
    localization_vector = []
    station_id_vector = []
    components_offsets = ['n', 'e', 'u'] 
    cols = ['N', 'E', 'U']

    for idx, df in enumerate(dfs):
        # First step extract all features
        features = df[['N', 'E', 'U', 'N sig', 'E sig', 'U sig', 'CorrNE', 'CorrNU', 'CorrEU']].copy()

        # Interpolate if needed
        if interpolate:
            features.interpolate(method='time', inplace=True)

        location = df.attrs.get('geometry')
        latitude, longitude = location.y, location.x
        height = df.attrs.get('height')

        for comp in components_offsets:
            series_names = ['offset_value', 'offset_error', 'decay_value', 'decay_error', 'decay_tau', 'decay_type']
            series_dict = {name: pd.Series(0.0 if interpolate else None, dtype='float64', index=df.index) for name in series_names}
            series_dict['decay_type'] = pd.Series(0, dtype='int64', index=df.index)

            for offset in df.attrs['offsets'][comp]['offsets']:
                series_dict['offset_value'].loc[offset['date']] = offset['value']
                series_dict['offset_error'].loc[offset['date']] = offset['error']

            for decay in df.attrs['offsets'][comp]['ps_decays']:
                series_dict['decay_value'].loc[decay['date']] = decay['value']
                series_dict['decay_error'].loc[decay['date']] = decay['error']
                series_dict['decay_tau'].loc[decay['date']] = decay['tau']
                series_dict['decay_type'].loc[decay['date']] = 1 if decay['type'] == 'logarithmic' else 2

            for name, series in series_dict.items():
                features[f'{comp}_{name}'] = series

        # Maybe use radians in future
        features['latitude'] = latitude
        features['longitude'] = longitude
        features['height'] = height
        station_id = station_ids[idx]

        for i in range(len(features) - chunk_size + 1):
            feature_row = np.hstack([features[col].values[i:i + chunk_size] for col in cols])
            feature_matrix.append(feature_row)

            # Determine the target values:
            offset_values_chunk = features[['n_offset_value', 'e_offset_value', 'u_offset_value']].iloc[i:i + chunk_size]
            if (offset_values_chunk != 0).any().any():
                localization_label = np.zeros(chunk_size)
                target_vector.append(1)  # Chunk with offset
                max_offset_idx = offset_values_chunk.abs().values.argmax(axis=0)
                max_offset_day = max_offset_idx.max()  # Get the max index across components

                # Ensure y_localization is a scalar value indicating the day of the max offset
                if max_offset_day < chunk_size:
                    localization_label[max_offset_day] = 1
                localization_vector.append(localization_label)
            else:
                target_vector.append(0)  # No offset
                localization_vector.append(np.zeros(chunk_size))  # Indicating no offset day

            station_id_vector.append(station_id)

    return pd.DataFrame(feature_matrix), target_vector, localization_vector, station_id_vector




def prepare_lstm_data(X, y_binary=None, y_localization=None, time_steps=21, balance=True, random_state=42):
    """
    Prepares the time series data for LSTM by reshaping each sample to (21, 3) 
    representing time steps and features (N, E, U), handling cases where labels 
    may not be provided (e.g., for predictions).

    Parameters:
    X (DataFrame): The input features, with each row containing 63 elements 
                   (21 timesteps per N, E, U).
    y_binary (Series or None): The binary target labels.
    y_localization (Series or None): The localization target labels.
    time_steps (int): Number of time steps to include in each sequence.
    balance (bool): Whether to balance the classes if labels are provided.

    Returns:
    Tuple: Prepared input features, and optionally binary and localization labels.
    """
    num_features = 3  # Assuming features are N, E, U

    # Reshape X to (num_samples, time_steps, num_features)
    X_lstm = X.values.reshape(-1, time_steps, num_features, order='F')

    # Optionally prepare labels
    y_binary_lstm = np.array(y_binary) if y_binary is not None else None
    y_localization_lstm = np.array(y_localization) if y_localization is not None else None

    # Handle balancing if labels are provided
    if balance and y_binary is not None:
        X_y_combined = [(X_lstm[i], y_binary_lstm[i], y_localization_lstm[i]) for i in range(len(y_binary_lstm))]

        class_0 = [(x, binary, loc) for x, binary, loc in X_y_combined if binary == 0]
        class_1 = [(x, binary, loc) for x, binary, loc in X_y_combined if binary == 1]

        if len(class_0) > len(class_1):
            class_1_resampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=random_state)
            balanced_data = class_0 + class_1_resampled
        else:
            class_0_resampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=random_state)
            balanced_data = class_1 + class_0_resampled

        X_lstm, y_binary_lstm, y_localization_lstm = zip(*balanced_data)
        X_lstm, y_binary_lstm, y_localization_lstm = np.array(X_lstm), np.array(y_binary_lstm), np.array(y_localization_lstm)

    return X_lstm, y_binary_lstm, y_localization_lstm



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

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        cross_entropy = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = K.pow((1 - p_t), gamma)
        alpha_weight_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_weight_factor * modulating_factor * cross_entropy
    return loss

def build_binary_model_tcn(input_shape, filters=32, kernel_size=3, dilation_rate=1):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for i in range(3):
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)  # Dropout added
        dilation_rate *= 2
    x = GlobalAveragePooling1D()(x)
    binary_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=binary_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_localization_model_tcn(input_shape, time_steps=21, filters=32, kernel_size=3, dilation_rate=1):
    print("Building tcn model")
    input_layer = Input(shape=input_shape)
    x = input_layer

    # Temporal Convolutional Layers
    for i in range(3):  # Three layers, can adjust
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        dilation_rate *= 2  # Increase dilation rate exponentially

    # Global pooling to reduce to single vector
    x = GlobalAveragePooling1D()(x)

    # Output layer for localization
    localization_output = Dense(time_steps, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=localization_output)
    model.compile(optimizer='adam', loss=distance_penalized_mse, metrics=['accuracy'])
    return model

def build_improved_binary_model_tcn(input_shape, filters=64, kernel_size=5, dilation_rate=1):
    input_layer = Input(shape=input_shape)
    x = input_layer

    for i in range(5):  # Increase the number of layers
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        dilation_rate *= 2

    x = GlobalAveragePooling1D()(x)
    binary_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=binary_output)
    # Use the focal_loss function by calling it to create the actual loss function
    model.compile(optimizer='adam', loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def split(X, y_binary, y_localization, station_id_vector, time_steps=21, random_state=42):
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    num_series = len(X_scaled)  # Approximate number of chunks
    print(f"Full data shape length: {num_series}")

    # Calculate the initial split index at 70% of the data
    approx_split_index = int(0.7 * num_series)

    # Find the first occurrence where the station ID changes after the 70% split index
    current_station_id = station_id_vector[approx_split_index]
    for idx in range(approx_split_index, len(station_id_vector)):
        if station_id_vector[idx] != current_station_id:
            split_index = idx
            break

    print(f"Split at station ID: {station_id_vector[split_index]}, index: {split_index}")


    # Prepare the data
    X_train_raw, X_test_raw = X_scaled[:split_index], X_scaled[split_index:]
    y_train_binary_raw, y_test_binary_raw = y_binary[:split_index], y_binary[split_index:]
    y_train_localization_raw, y_test_localization_raw = y_localization[:split_index], y_localization[split_index:]
    station_id_train, station_id_test = station_id_vector[:split_index], station_id_vector[split_index:]
    print(len(X_train_raw))
    print(len(X_test_raw))
    
    # Prepare the training data (with resampling)
    X_train, y_train_binary, y_train_localization = prepare_lstm_data(pd.DataFrame(X_train_raw), y_train_binary_raw, y_train_localization_raw, time_steps, balance=True, random_state=random_state)
    training_data = [X_train, y_train_binary, y_train_localization, station_id_train]

    # Prepare the test data (without resampling)
    X_test, y_test_binary, y_test_localization = prepare_lstm_data(pd.DataFrame(X_test_raw), y_test_binary_raw, y_test_localization_raw, time_steps, balance=False)
    testing_data = [X_test, y_test_binary, y_test_localization, station_id_test]

    return training_data, testing_data

        
def train(training_data_array, time_steps=21, epochs_binary=80, epochs_localization=80, model_save_path_binary=None, model_save_path_localization=None, plot_losses=False):
    """
    Train and evaluate the LSTM model for chunk classification and localization.
    Optionally save the model parameters for future use.
    
    Returns model performance and confusion matrices.
    """

    X_train, y_train_binary, y_train_localization = training_data_array[0], training_data_array[1], training_data_array[2]

    # Compute class weights
    class_weights_dict = {0: 1.0, 1: 1.0}
    
    # Build and train binary classification model
    print("Training binary classification model...")
    binary_history = LossHistory() 
    reduce_lr_binary = ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=5, min_lr=1e-6)
    early_stopping_binary = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model_binary = build_improved_binary_model_tcn(input_shape=(time_steps, X_train.shape[2]))
    model_binary.fit(X_train, y_train_binary, epochs=epochs_binary, batch_size=64, validation_split=0.2, callbacks=[early_stopping_binary, reduce_lr_binary, binary_history],class_weight=class_weights_dict)
    
    # Filter samples with offsets for localization training
    offset_indices = np.where(y_train_binary == 1)[0]
    X_train_localization = X_train[offset_indices]
    y_train_localization_filtered = y_train_localization[offset_indices]
    
    # Build and train localization model
    print("Training localization model...")
    localization_history = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model_localization = build_localization_model_tcn(input_shape=(time_steps, X_train.shape[2]), time_steps=time_steps)
    model_localization.fit(X_train_localization, y_train_localization_filtered, epochs=epochs_localization, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr, localization_history])

    
    binary_train_losses = np.array(binary_history.losses)
    binary_val_losses = np.array(binary_history.val_losses)
    localization_train_losses = np.array(localization_history.losses)
    localization_val_losses = np.array(localization_history.val_losses)
    plt.figure(figsize=(12, 5))
    # Binary Losses
    plt.subplot(1, 2, 1)
    plt.plot(binary_train_losses, label='Training Loss')
    plt.plot(binary_val_losses, label='Validation Loss')
    plt.title('Binary Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()   
    # Localization Losses
    plt.subplot(1, 2, 2)
    plt.plot(localization_train_losses, label='Training Loss')
    plt.plot(localization_val_losses, label='Validation Loss')
    plt.title('Localization Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Storage/Loss.png')
    if plot_losses is True:
        plt.show()
    else:
        plt.close()

    # Save the models if paths are provided
    if model_save_path_binary:
        model_binary.save(model_save_path_binary)
        print(f"Binary classification model saved to {model_save_path_binary}")
    if model_save_path_localization:
        model_localization.save(model_save_path_localization)
        print(f"Localization model saved to {model_save_path_localization}")

    return model_binary, model_localization


def unchunk_and_plot(X_test, y_pred_probs, y_bin_labels, station_ids, window_size=21):
    station_data = {}

    # Step 1: Create N, E, U series for each station
    for i, (x, y_prob, y_label, station_id) in enumerate(zip(X_test, y_pred_probs, y_bin_labels, station_ids)):
        if station_id not in station_data:
            station_data[station_id] = {
                'N': list(x[:, 0]),  # Start with the first 21 coordinates for N
                'E': list(x[:, 1]),  # Start with the first 21 coordinates for E
                'U': list(x[:, 2]),  # Start with the first 21 coordinates for U
                'binary_preds': [],   # Store the binary prediction values
                'binary_label': []   # Store the binary test labels
            }

        else:
            # Append the last coordinate from the current window to each series
            station_data[station_id]['N'].append(x[-1, 0])
            station_data[station_id]['E'].append(x[-1, 1])
            station_data[station_id]['U'].append(x[-1, 2])

        # Store the prediction for this window
        station_data[station_id]['binary_preds'].append(y_prob)
        station_data[station_id]['binary_label'].append(y_label)

    # Step 2: Create the binary matrix for each station
    summed_predictions = {}
    summed_labels = {}
    for station_id, data in station_data.items():
        num_days = len(data['N'])
        num_windows = len(data['binary_preds'])

        # Initialize a binary matrix with zeros
        binary_matrix = np.zeros((num_windows, num_days))
        label_matrix = np.zeros((num_windows, num_days))

        # Step 3: Assign predicted binary values to the matrix
        for window_index, y_pred in enumerate(data['binary_preds']):
            start_index = window_index
            end_index = start_index + window_size
            # Assign y_pred to the appropriate slice in the row
            binary_matrix[window_index, start_index:end_index] = y_pred

        for window_index, y_label in enumerate(data['binary_label']):
            start_index = window_index
            end_index = start_index + window_size
            # Assign y_pred to the appropriate slice in the row
            label_matrix[window_index, start_index:end_index] = y_label

        # Sum the binary matrix along the columns to get the final summed prediction per day
        summed_predictions[station_id] = np.sum(binary_matrix, axis=0)
        summed_labels[station_id] = np.sum(label_matrix, axis=0)


        selected_stations = random.sample(list(station_data.keys()), 8)
    
    for station_id in selected_stations:
        data = station_data[station_id]
        days = range(len(data['N']))  # X-axis (days)
        summed_pred = summed_predictions[station_id]  # Summed binary predictions
        summed_lab = summed_labels[station_id]
        true_offset_pos = np.where(summed_lab > 20.5)[0]
        
        # Adjust the figure size to better fit the screen
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f"Station {station_id}", fontsize=16)

        # Plot N, E, U series with corresponding subplots
        series_labels = ['N', 'E', 'U']
        for i, label in enumerate(series_labels):
            ax = axes[i]
            ax.plot(days, data[label], label=f'{label} Series', color='blue')
            ax.set_xlabel('Days')
            ax.set_ylabel(f'{label} Coordinate')
            if len(true_offset_pos) > 0:
                for pos in true_offset_pos:
                    ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, label='True offset position')

            # Secondary y-axis for summed predictions
            ax2 = ax.twinx()
            ax2.bar(days, summed_pred, alpha=0.4, color='orange', label='Summed Predictions')
            ax2.set_ylabel('Summed Predictions')

            # Legends and grid
            ax.legend(loc='upper left', fontsize=10)
            ax2.legend(loc='upper right', fontsize=10)
            ax.grid(True)

        # Use tight_layout with padding to avoid cutting labels
        plt.tight_layout(pad=2, rect=[0, 0, 1, 0.95])
        plt.show()

   
    return



def test(test_data_array, model_binary_path, model_localization_path, time_steps=21, random_state=42, enhanced_evaluation=False):
    """
    Train and evaluate the LSTM model for chunk classification and localization.
    Optionally save the model parameters for future use.
    
    Returns model performance and confusion matrices.
    """

    X_test, y_test_binary, y_test_localization , test_stations_ids = test_data_array[0], test_data_array[1], test_data_array[2], test_data_array[3]
    print(X_test.shape)
    print(y_test_binary.shape)

    model_binary = load_model(model_binary_path, custom_objects={'focal_loss': focal_loss})
    model_localization = load_model(model_localization_path, custom_objects={'distance_penalized_mse': distance_penalized_mse})

    # Ensure predictions are binary
    y_pred_probs = model_binary.predict(X_test)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    # Calculate binary confusion matrix and classification report
    binary_cm = confusion_matrix(y_test_binary, y_pred_binary)
    TN, FP, FN, TP = binary_cm.ravel()
    report_binary = classification_report(y_test_binary, y_pred_binary, target_names=['No Offset', 'Offset'])
    
    # Initialize localization confusion matrix
    localization_cm = {'TP': 0, 'TP*': 0, 'FP': FP, 'FN': FN, 'TN': TN}
    difference_days = {}
    
    y_pred_localization = model_localization.predict(X_test)
    for i in range(len(X_test)):
        if y_test_binary[i] == 1 and y_pred_binary[i] == 1:  # TP or TP*
            predicted_day = np.argmax(y_pred_localization[i])
            true_day = np.argmax(y_test_localization[i])
        
            if predicted_day == true_day:
                localization_cm['TP'] += 1
                difference_days['0'] = difference_days.get('0', 0) + 1
            else:
                localization_cm['TP*'] += 1
                difference = abs(predicted_day - true_day)
                difference_days[str(difference)] = difference_days.get(str(difference), 0) + 1

    
    
    # Print and verify results
    print("Binary Classification Report:\n", report_binary)
    print("Binary Confusion Matrix:\n", binary_cm)
    print("Localization Confusion Matrix:\n", localization_cm)
    print("Difference Days:", difference_days)

    if enhanced_evaluation is True:

        # Histogram of Resiudals
        sorted_data = dict(sorted(difference_days.items(), key=lambda item: int(item[0])))
        # Extract keys and values for plotting
        labels = list(sorted_data.keys())
        values = list(sorted_data.values())
        # Create the histogram
        x = range(len(labels))  # the x locations for the groups
        width = 0.35  # width of the bars

        plt.figure(figsize=(12, 6))
        plt.bar([pos - width for pos in x], values, width=width, color='skyblue', label='TCN')
        plt.xlabel('Residuals')
        plt.ylabel('Counts')
        plt.title('Histogram of Localisation Residuals for binary TP')
        plt.xticks(x, labels, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Storage/Hist.png")
        plt.show()


        #Accuracy per Position
        days = range(time_steps)  # X-axis for the day positions
        # Set up empty array
        binary_true_counts = np.zeros(time_steps)
        binary_false_counts = np.zeros(time_steps)
        localization_true_counts = np.zeros(time_steps)
        localization_false_counts = np.zeros(time_steps)

        # Calculate counts for binary and localization accuracy per position
        for i in range(0,len(y_test_binary)):
            # Get the true position of the offset (using argmax on the localization label)
            true_position = np.argmax(y_test_localization[i])
            pred_position = np.argmax(y_pred_localization[i])
            
            # Check if binary prediction is correct
            if y_pred_binary[i] == 1 and y_test_binary[i] == 1:
                binary_true_counts[true_position] += 1
            elif y_pred_binary[i] == 0 and y_test_binary[i] == 1:
                binary_false_counts[true_position] += 1
            
            # Check if localization prediction matches the true position
            if abs(pred_position-  true_position) < 1.5 and y_test_binary[i] == 1:
                localization_true_counts[true_position] += 1
            elif abs(pred_position-  true_position) > 1.5 and y_test_binary[i] == 1:
                localization_false_counts[true_position] += 1

        # Calculate percentages
        print(binary_true_counts, binary_false_counts)
        binary_true_percentage = binary_true_counts / (binary_true_counts + binary_false_counts) * 100
        binary_false_percentage = binary_false_counts / (binary_true_counts + binary_false_counts) * 100
        localization_true_percentage = localization_true_counts / (localization_true_counts + localization_false_counts) * 100
        localization_false_percentage = localization_false_counts / (localization_true_counts + localization_false_counts) * 100

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Binary and Localization Prediction Accuracy by Offset Position  (only for samples including offsets)")
        # Binary Prediction Plot
        ax1.bar(days, binary_true_percentage, color="green", label="True")
        ax1.bar(days, binary_false_percentage, bottom=binary_true_percentage, color="red", label="False")
        ax1.set_xlabel("Offset Position (Days)")
        ax1.set_ylabel("Percentage")
        ax1.set_title("Binary Prediction Accuracy by Offset Position")
        ax1.legend()
        # Localization Prediction Plot
        ax2.bar(days, localization_true_percentage, color="green", label="True")
        ax2.bar(days, localization_false_percentage, bottom=localization_true_percentage, color="red", label="False")
        ax2.set_xlabel("Offset Position (Days)")
        ax2.set_ylabel("Percentage")
        ax2.set_title("Localization Prediction Accuracy by Offset Position")
        ax2.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("Storage/Accuracy.png")
        plt.show()

        # Example usage (assuming you already have X_test, y_test_binary, and station_ids):
        unchunk_and_plot(X_test, y_pred_probs, y_test_binary, test_stations_ids)        

    return binary_cm, localization_cm, difference_days



###########
# RUN THE CODE
#########
print("reading data...")
dfs = []
dir = Path(DATA_PATH)
for file_path in dir.iterdir():
    if file_path.is_file():
        dfs.append(read_file(file_path.name))
        
print("cleaning now...")
cleaned_dfs, station_ids = clean_dataframes(dfs, missing_value_threshold=0, days_included=100, minimal_offset=MIN_OFFSET, randomize_stations=True, random_state=RANDOM_STATE)
print("feature extraction...")
features, target, localization, station_id_vector = extract_features(cleaned_dfs, station_ids, interpolate=True, chunk_size=TIME_STEPS)
print("splitting data...")
training_data, testing_data = split(features, target, localization, station_id_vector, time_steps=TIME_STEPS, random_state=RANDOM_STATE)
print("training...")
#model_binary, model_localization = train(training_data, time_steps=TIME_STEPS, epochs_binary=EPOCHS_BIN, epochs_localization=EPOCHS_LOC, model_save_path_binary=MODEL_BINARY_SAVE_PATH, model_save_path_localization=MODEL_LOC_SAVE_PATH, plot_losses=True)
print("testing...")
binary_cm, localization_cm, difference_days = test(testing_data, MODEL_BINARY_SAVE_PATH, MODEL_LOC_SAVE_PATH, time_steps=TIME_STEPS, random_state=RANDOM_STATE, enhanced_evaluation=True)