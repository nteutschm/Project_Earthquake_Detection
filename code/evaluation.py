#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from scipy.stats import binom

from variables import *
from plots import *

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
    
    return group_conf_matrix, unique_groups, groups
    
def dechunk_labels_predictions(dfs, chunk_size=CHUNK_SIZE):
    time_series = []
    alpha = 0.05
    threshold_percentage = binom.ppf(1 - alpha, chunk_size, 0.5) / chunk_size
    threshold_predictions = int(threshold_percentage * chunk_size)
    
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
            total_predictions = time_series_df.loc[prediction_dates[i], 'preds_sum']
            j = i + 1

            while j < len(prediction_dates) and (prediction_dates[j] - cluster[-1]).days < chunk_size:
                cluster.append(prediction_dates[j])
                total_predictions += time_series_df.loc[prediction_dates[j], 'preds_sum']
                j += 1

            # Check if cluster total predictions are statistically significant
            if total_predictions >= threshold_predictions:
                # Calculate mean timestamp weighted by the number of predictions on each day
                timestamps = [
                    pd.Timestamp.timestamp(date) * time_series_df.loc[date, 'preds_sum'] 
                    for date in cluster
                ]
                mean_timestamp = sum(timestamps) / total_predictions
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

def calculate_prediction_statistics(dfs, chunk_size=CHUNK_SIZE):
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
    total_predictions = 0
    false_positives = 0

    for df in dfs:
        test_label_dates = df[df['labels_sum'] > 0].index
        prediction_dates = df[df['mean_day_pred'] > 0].index
        total_predictions+=len(prediction_dates)
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
                    
        for pred_date in prediction_dates:
            if not any(abs((pred_date - label_date).days) <= chunk_size for label_date in test_label_dates):
                false_positives += 1

    print("\n=== Mean Prediction Statistics ===")
    print(f"Total earthquakes (test labels): {total_labels}")

    # Calculate key percentages for predictions
    predicted_within_percentage = (predicted_within_window / total_labels) * 100 if total_labels > 0 else 0
    predicted_exact_percentage = (predicted_exact / total_labels) * 100 if total_labels > 0 else 0
    undetected_percentage = (undetected / total_labels) * 100 if total_labels > 0 else 0
    false_positive_percentage = (false_positives / total_predictions) * 100 if total_predictions > 0 else 0

    print(f"Predictions within {chunk_size} days:")
    print(f"  Count: {predicted_within_window}")
    print(f"  Percentage: {predicted_within_percentage:.2f}% of total earthquakes")

    print(f"Exact date matches:")
    print(f"  Count: {predicted_exact}")
    print(f"  Percentage: {predicted_exact_percentage:.2f}% of total earthquakes")

    print(f"Undetected earthquakes:")
    print(f"  Count: {undetected}")
    print(f"  Percentage: {undetected_percentage:.2f}% of total earthquakes")
    
    print(f"False positives (predicted but no actual earthquake):")
    print(f"  Count: {false_positives}")
    print(f"  Percentage: {false_positive_percentage:.2f}% of total predictions")

    print("\n==================================\n")
    
    return predicted_exact_percentage, predicted_within_percentage, undetected_percentage, false_positive_percentage

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
    

def evaluate(test_predictions, test_labels, cleaned_dfs, stations, start_indices, chunk_size=CHUNK_SIZE, X_test=None, model=None, tolerance_window=None, simulated=False):
    
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
    
    conf_matrix, unique_groups, groups = group_confusion_matrix(test_labels, test_predictions, n_classes=CHUNK_SIZE)
    plot_grouped_confusion_matrix(conf_matrix, unique_groups, groups, f'grouped_conf{tolerance_str}')
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
    stats = calculate_prediction_statistics(total_df)
    plot_mean_preds_statistics(stats, f'mean_preds_stats{tolerance_str}')
    calculate_mean_metrics(total_df)
    rng = np.random.default_rng(seed=RANDOM_STATE)
    for idx in range(10):
        plot_neu_displacements(total_df[rng.integers(0, len(total_df))], idx=idx, name=f'comp_preds{tolerance_str}')
        plot_neu_mean_displacements(total_df[rng.integers(0, len(total_df))], idx=idx, name=f'mean_preds{tolerance_str}')
        
    return original_test_predictions