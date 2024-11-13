#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
from shapely.geometry import Point

from variables import *


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
            feature_row = []
            for col in cols:
                if col not in non_chunked_columns:
                    # For columns that should be chunked, take the chunk
                    feature_row.extend(features[col].values[i:i + chunk_size])

            # Then, add non-chunked columns at the end of the row, if they are in the data -> needed in this order for later step
            for col in non_chunked_columns:
                if col in cols:
                    # Add the value of the non-chunked column directly without chunking
                    feature_row.append(features[col].values[i])  # Just take the value at position i

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