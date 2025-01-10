#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import f1_score
from keras.models import  load_model
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy
from scipy.signal import find_peaks
import random
from models import *
from plots import plot_mean_preds_statistics
import matplotlib.pyplot as plt



def unchunk_and_plot(X_test, y_pred_probs, y_pred_loc, y_bin_labels, station_ids, window_size=CHUNK_SIZE):
    """
    Visualizes full time series data and TCN predictions by unchunking sequences and plotting results.

    Parameters:
    ----------
    X_test : The test dataset containing time series data, shaped as (n_samples, window_size, n_features).
    y_pred_probs : Predicted probabilities for binary classification, indicating the likelihood of an offset in each chunk.
    y_pred_loc : Predicted localization of offsets within chunks, output as integers representing positions in the window.
    y_bin_labels : Ground-truth binary labels for each chunk, indicating whether an offset exists in the chunk.
    station_ids : Station identifiers corresponding to each chunk, used for associating chunks with specific time series.
    window_size : The size of each window (default is CHUNK_SIZE).

    Returns:
    -------
    None
        Prints key mean prediction values and visualizes them
        Displays time series plots showing actual vs. predicted offsets, overlaying binary predictions and localization.
    """
        
    print("\n Unchunking \n")
    station_data = {}

    # Create N, E, U series for each station
    for i, (x, y_prob, y_loc, y_label, station_id) in enumerate(zip(X_test, y_pred_probs, y_pred_loc, y_bin_labels, station_ids)):
        if station_id not in station_data:
            station_data[station_id] = {
                'N': list(x[:, 0]),  # Start with the first 21 coordinates for N
                'E': list(x[:, 1]),  # Start with the first 21 coordinates for E
                'U': list(x[:, 2]),  # Start with the first 21 coordinates for U
                'binary_preds': [],   # Store the binary prediction values
                'binary_label': [],   # Store the binary test labels
                'loc_preds': [],   # Store the localization prediction values
                'loc_entropy': [],   # Store the localization entropy values
            }

        else:
            # Append the last coordinate from the current window to each series
            station_data[station_id]['N'].append(x[-1, 0])
            station_data[station_id]['E'].append(x[-1, 1])
            station_data[station_id]['U'].append(x[-1, 2])

        # Store the prediction for this window
        station_data[station_id]['binary_preds'].append(y_prob)
        station_data[station_id]['binary_label'].append(y_label)
        station_data[station_id]['loc_preds'].append(y_loc)
        station_data[station_id]['loc_entropy'].append(1 - (entropy(y_loc)/3.0445224377234217))


    # Create the binary matrix for each station
    summed_predictions = {}
    summed_labels = {}
    summed_localizations = {}
    summed_entropy = {}

    for station_id, data in station_data.items():
        num_days = len(data['N'])
        num_windows = len(data['binary_preds'])

        # Initialize a binary matrix with zeros
        binary_matrix = np.zeros((num_windows, num_days))
        label_matrix = np.zeros((num_windows, num_days))
        loc_matrix = np.zeros((num_windows, num_days))
        entropy_matrix = np.zeros((num_windows, num_days))

        # Assign predicted binary values to the matrix
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

        for window_index, y_loc in enumerate(data['loc_preds']):
            start_index = window_index
            end_index = start_index + window_size
            # Assign y_pred to the appropriate slice in the row
            loc_matrix[window_index, start_index:end_index] = y_loc

        for window_index, y_entropy in enumerate(data['loc_entropy']):
            start_index = window_index
            end_index = start_index + window_size
            # Assign y_pred to the appropriate slice in the row
            entropy_matrix[window_index, start_index:end_index] = y_entropy

        # Sum the binary matrix along the columns to get the final summed prediction per day
        summed_labels[station_id] = np.sum(label_matrix, axis=0)

        summed_predictions[station_id] = np.sum(binary_matrix, axis=0)
        summed_localizations[station_id] = np.sum(loc_matrix, axis=0)
        summed_entropy[station_id] = np.sum(entropy_matrix, axis=0)

    global_max_prediction = max([np.max(arr) for arr in summed_predictions.values()])
    global_max_localization = max([np.max(arr) for arr in summed_localizations.values()])
    global_max_entropy = max([np.max(arr) for arr in summed_entropy.values()])

    # Normalize all arrays using the global maximums
    def normalize_list(arr, global_max):
        if global_max == 0:
            return arr  # Avoid division by zero; if all elements are 0, return the original array.
        return arr / global_max

    summed_total = {}

    for station_id in summed_predictions.keys():
        # Normalize using the global maximums
        normalized_predictions = normalize_list(summed_predictions[station_id], global_max_prediction)
        normalized_localizations = normalize_list(summed_localizations[station_id], global_max_localization)
        normalized_entropy = normalize_list(summed_entropy[station_id], global_max_entropy)

        # Sum the normalized arrays
        summed_total[station_id] = (
            normalized_predictions +
            normalized_localizations +
            normalized_entropy
        )


    # Metrics storage for each window size
    metrics = {
        '0_day': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        '1_day': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        '21_day': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    }

    # Iterate through each station
    for station_id in summed_labels.keys():
        station_length = len(summed_labels[station_id])  # Length of the time series

        true_offset_pos, _ = find_peaks(summed_labels[station_id], height=CHUNK_SIZE/2, distance=CHUNK_SIZE)

        
        # Find predicted positions: local maximas above threshold (1.5) with 21-day window
        threshold = 1.15
        peak_indices, _ = find_peaks(summed_total[station_id], height=threshold, distance=CHUNK_SIZE)

        for window_name, window_size in [('0_day', 0), ('1_day', 1), ('21_day', 21)]:
            TP = 0  # True Positives
            FP = 0  # False Positives
            FN = 0  # False Negatives
            TN = 0  # True Negatives

            detected_offsets = []  # Track detected true offsets

            # Count TP and FN
            for true_pos in true_offset_pos:
                if any(abs(pred - true_pos) <= window_size for pred in peak_indices):
                    TP += 1
                    detected_offsets.append(true_pos)
                else:
                    FN += 1

            # Count FP
            for pred in peak_indices:
                if not any(abs(pred - true_pos) <= window_size for true_pos in true_offset_pos):
                    FP += 1

            # Count TN
            TN = station_length - (TP+FP+FN)

            # Store metrics
            metrics[window_name]['TP'] += TP
            metrics[window_name]['FP'] += FP
            metrics[window_name]['TN'] += TN
            metrics[window_name]['FN'] += FN

    # Print the metrics
    for window_name, data in metrics.items():
        print(f"Metrics for {window_name} window:")
        print(f"  True Positives (TP): {data['TP']}")
        print(f"  False Positives (FP): {data['FP']}")
        print(f"  True Negatives (TN): {data['TN']}")
        print(f"  False Negatives (FN): {data['FN']}")
        precision = data['TP'] / (data['TP'] + data['FP']) if (data['TP'] + data['FP']) > 0 else 0
        recall = data['TP'] / (data['TP'] + data['FN']) if (data['TP'] + data['FN']) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1: {f1:.2f}\n")

    classification_reports = {}
    for window_name, data in metrics.items():
        TP = data['TP']
        FP = data['FP']
        TN = data['TN']
        FN = data['FN']
        
        # Calculate the required metrics
        y_true = [1] * (TP + FN) + [0] * (TN + FP)  # True labels
        y_pred = [1] * TP + [0] * FN + [0] * TN + [1] * FP  # Predicted labels
        
        # Create the classification report
        report = classification_report(y_true, y_pred, target_names=["No Offset", "Offset"], zero_division=0)
        classification_reports[window_name] = report

    # Print the classification reports
    for window_name, report in classification_reports.items():
        print(f"Classification Report for {window_name} window:")
        print(report)
        print("\n")

   
    total = metrics['1_day']['TP'] + metrics['1_day']['FN']
    exact = 100 / total * metrics['1_day']['TP']
    window = 100 / total * (metrics['21_day']['TP']-metrics['1_day']['TP'])
    undetected =  100 / total * metrics['21_day']['FN']
    false_positive = 100 / total * metrics['21_day']['FP']
    values = [exact, window, undetected, false_positive]
    # Plot the mean statistics
    plot_mean_preds_statistics(values, "mean_adjusted_bars")

    # plot random stations
    selected_stations = random.sample(list(station_data.keys()), 5)
    for station_id in selected_stations:
        data = station_data[station_id]
        days = range(len(data['N']))  # X-axis (days)
        summed_pred = summed_total[station_id]  # Summed binary predictions
        true_offset_pos, _ = find_peaks(summed_labels[station_id], height=30, distance=CHUNK_SIZE)
        pred_pos, _ = find_peaks(summed_total[station_id], height=1.15, distance=CHUNK_SIZE)
        
        # Adjust the figure size to better fit the screen
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f"Station {station_id}", fontsize=16)

        # Plot N, E, U series with corresponding subplots
        series_labels = ['N', 'E', 'U']
        for i, label in enumerate(series_labels):
            ax = axes[i]
            ax.plot(days, data[label], label=f'{label}-Coordinate Time Series',color='blue')
            ax.set_xlabel('Days')
            ax.set_ylabel(f'{label} Coordinate')

            # Secondary y-axis for summed predictions
            ax2 = ax.twinx()
            ax2.bar(days, summed_pred, alpha=1, color='orange', zorder=1, label= 'Summed Predictions')
            legend2 = ax2.legend(loc='upper right', fontsize=10)
            legend2.get_frame().set_alpha(1.0)  # Set the legend box to be fully opaque
            legend2.get_frame().set_facecolor('white')  # Optional: Set background color of the legend box
            legend2.get_frame().set_edgecolor('black')  # Optional: Set border color of the legend box
            ax2.set_ylabel('Aggregated Score')

            ax.set_zorder(2) 
            ax2.set_zorder(2) 
            threshold_line = ax2.axhline(y=1.15, color='black', linestyle='--', zorder=2, linewidth=1, label='Threshold')

            # Update the secondary axis legend to include the horizontal line
            handles, labels = ax2.get_legend_handles_labels()  # Get current handles and labels from ax2
            handles.append(threshold_line)  # Add the horizontal line handle
            #labels.append('Threshold')  # Add the corresponding label

            # Create a new legend for ax2
            legend2 = ax2.legend(handles, labels, loc='upper right', fontsize=10)
            legend2.get_frame().set_alpha(1.0)  # Set the legend box to be fully opaque
            legend2.get_frame().set_facecolor('white')  # Optional: Set background color of the legend box
            legend2.get_frame().set_edgecolor('black')  # Optional: Set border color of the legend box


            if len(true_offset_pos) > 0:
                for pos in true_offset_pos:
                    ax.axvline(x=pos, color='black', linestyle='-', zorder=3,linewidth=2.5, label='True offset position')

            legend = ax.legend(loc='upper left', fontsize=10)
            legend.get_frame().set_alpha(1.0)  # Set the legend box to be fully opaque
            legend.get_frame().set_facecolor('white')  # Optional: Set background color of the legend box
            legend.get_frame().set_edgecolor('black')  # Optional: Set border color of the legend box

            ax.grid(True)

        # Use tight_layout with padding to avoid cutting labels
        plt.tight_layout(pad=2, rect=[0, 0, 1, 0.95])
        plt.savefig(f'{TWO_STEPS_PLOTS_PATH}_example_{str(station_id)}.png', dpi=300)
        plt.close()

    return




def test_two_step(X_test, y_test_binary, y_test_localization, test_stations, enhanced_evaluation=True):
    """
    Train and evaluate the TCN model for chunk classification and localization.

    X_test : The test dataset containing time series data, shaped as (n_samples, chunk_size, n_features).
    y_test_binary : Ground-truth binary labels for each chunk, indicating the presence of an offset (1 for offset, 0 for no offset).
    y_test_localization :  Ground-truth localization labels for chunks with offsets, shaped as (n_samples, chunk_size).
    test_stations : Station identifiers corresponding to each chunk, used for associating predictions with specific time series.
    enhanced_evaluation : bool, optional
        If True, performs additional analyses including:
        - Histogram of localization residuals (difference between predicted and true offset positions).
        - Binary and localization accuracy analysis for each offset position.
        - Visualizations of selected time series data with predictions overlaid.
        Default is True.

    Returns:
    -------
    None
        Prints key prediction values and visualizes them, also calls for analysis of mean predictions by unrolling the chunks
    """

    model_binary = load_model(MODEL_BIN_PATH, custom_objects={'focal_loss': focal_loss})

    model_localization = load_model(MODEL_LOC_PATH, custom_objects={'distance_penalized_mse': distance_penalized_mse})

    # Ensure predictions are binary
    y_pred_probs = model_binary.predict(X_test, verbose=0)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    # Calculate binary confusion matrix and classification report
    binary_cm = confusion_matrix(y_test_binary, y_pred_binary)
    TN, FP, FN, TP = binary_cm.ravel()
    report_binary = classification_report(y_test_binary, y_pred_binary, target_names=['No Offset', 'Offset'])
    
    # Initialize localization confusion matrix
    localization_cm = {'TP': 0, 'TP*': 0, 'FP': FP, 'FN': FN, 'TN': TN}
    difference_days = {}

    y_pred_localization = model_localization.predict(X_test, verbose=0)
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
        plt.savefig(f'{TWO_STEPS_PLOTS_PATH}_Hist.png', dpi=300)
        plt.close()


        #Accuracy per Position
        days = range(CHUNK_SIZE)  # X-axis for the day positions
        # Set up empty array
        binary_true_counts = np.zeros(CHUNK_SIZE)
        binary_false_counts = np.zeros(CHUNK_SIZE)
        localization_true_counts = np.zeros(CHUNK_SIZE)
        localization_false_counts = np.zeros(CHUNK_SIZE)

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
        ax2.set_title("Localization Prediction Accuracy by Offset Position (1 day tolerance)")
        ax2.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{TWO_STEPS_PLOTS_PATH}_Accuracy.png', dpi=300)
        plt.close()

        # Unroll the chunks for mean evaluation metrices and visualizations
        unchunk_and_plot(X_test, y_pred_probs, y_pred_localization, y_test_binary, test_stations)        

    return