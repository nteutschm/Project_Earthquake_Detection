#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import geopandas as gpd
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_contour

from variables import *

# Define standard font sizes and other plotting parameters for consistency
legend_fontsize = 16
label_fontsize = 20
title_fontsize = 22
tick_fontsize = 14
dpi_setting = 300
plot_size = (12, 8)
subplot_size = (14, 12)

def plot_eval_metrics(eval_results, name):
    """
    The function takes in evaluation results and generates a line plot comparing the training and evaluation log loss 
    across epochs, before saving the plot as a PNG file using the specified PLOTS_PATH variable and name. 

    Parameters:
    eval_results (dict): A dictionary containing evaluation metrics, specifically the log loss for both the training 
                         ('validation_0') and evaluation ('validation_1') datasets.
    name (str): The name used for saving the PNG file using the PLOTS_PATH variable.

    Returns:
    None
    """
    
    epochs = len(eval_results['validation_0']['mlogloss'])
    x_axis = range(epochs)
    
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(x_axis, eval_results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, eval_results['validation_1']['mlogloss'], label='Eval')
    ax.set_title(f'{MODEL_TYPE} Log Loss Over Epochs', fontsize=title_fontsize, pad=20)
    ax.set_xlabel('Epochs', fontsize=label_fontsize, labelpad=15)
    ax.set_ylabel('Log Loss', fontsize=label_fontsize, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()
    
def plot_optimization(study, parameters):
    """
    Generates and saves Optuna optimization plots to visualize the optimization process and parameter importances. 
    This function creates three plots: an optimization history plot, a hyperparameter importance plot, 
    and a contour plot for selected parameters. All plots are saved as PNG files using the specified PLOTS_PATH variable.

    Parameters:
    study (optuna.study.Study): An Optuna study object containing the results of the hyperparameter optimization.
    parameters (list): A list of two parameter names (strings) to be used in the contour plot.

    Returns:
    None
    """
    
    plt.figure(figsize=subplot_size)
    ax = plot_optimization_history(study, target_name='Validation loss')
    ax.set_xlabel('Trial', fontsize=label_fontsize)
    ax.set_ylabel('Validation loss (Macro F1 Score)', fontsize=label_fontsize)
    ax.set_title(f'Optimization history for {MODEL_TYPE}', fontsize=title_fontsize, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + 'optimization_history.png', dpi=dpi_setting)
    
    plt.figure(figsize=subplot_size)
    ax = plot_param_importances(study)
    ax.set_xlabel('Importance for validation loss', fontsize=label_fontsize)
    ax.set_ylabel('Hyperparameter', fontsize=label_fontsize)
    ax.set_title(f'Hyperparameter Importances for {MODEL_TYPE}', fontsize=title_fontsize, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + 'parameter_importances.png', dpi=dpi_setting)
    
    plt.figure(figsize=subplot_size)
    ax = plot_contour(study, params=[parameters[0], parameters[1]], target_name='Validation loss')
    ax.set_xlabel(' '.join([word.capitalize() for word in parameters[0].replace('_', ' ').split()]), fontsize=label_fontsize)
    ax.set_ylabel(' '.join([word.capitalize() for word in parameters[1].replace('_', ' ').split()]), fontsize=label_fontsize)
    ax.set_title(f'Contour plot for {MODEL_TYPE}', fontsize=title_fontsize, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + 'contour.png', dpi=dpi_setting)
    
def plot_histogram(series, xlabel, ylabel, title, name):
    """
    Generates and saves a histogram plot based on the provided data. The function customizes the appearance 
    of the plot with specified labels and title, and saves it as a PNG file using the given name and the PLOTS_PATH variable.

    Parameters:
    series (pd.Series): A pandas Series containing the data to be plotted.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    title (str): The title of the plot.
    name (str): The name used for saving the PNG file using the PLOTS_PATH variable.

    Returns:
    None
    """
    
    plt.figure(figsize=plot_size)
    plt.bar(series.index, series.values, color='#4c9fdb', alpha=0.85)
    plt.xlabel(xlabel, fontsize=label_fontsize, labelpad=15)
    plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=15)
    plt.title(title, fontsize=title_fontsize, pad=20)
    plt.xticks(fontsize=tick_fontsize, rotation=45, ha='right')
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()
    
def plot_heatmap(heatmap_data, mask, name):
    """
    Generates and saves a heatmap visualization of misclassified data. 
    The data is scaled logarithmically and categorized by the index (number of days missed) over time. 
    The resulting plot is saved as a PNG file with the specified name using the PLOTS_PATH variable.

    Parameters:
    heatmap_data (pd.DataFrame): A pandas DataFrame containing the heatmap data.
    mask (pd.DataFrame): A boolean mask to hide specific areas of the heatmap (True values will be masked).
    name (str): The name used for saving the PNG file using the PLOTS_PATH variable.

    Returns:
    None
    """
    
    plt.figure(figsize=plot_size)
    ax = sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        mask=mask,
        cbar_kws={'label': 'Log(Number of Misclassifications)'},
        linewidths=0.5,
        linecolor='lightgrey'
    )
    ax.figure.axes[-1].yaxis.label.set_size(label_fontsize)
    
    tick_step = 6
    plt.xticks(
        ticks=np.arange(0, len(heatmap_data.columns), tick_step),
        labels=[date.strftime('%Y-%m') for date in heatmap_data.columns[::tick_step]],
        fontsize=tick_fontsize,
        rotation=45,
        ha='right'
    )
    plt.yticks(fontsize=tick_fontsize)
    plt.title('Logarithmic Heatmap of Missed Predictions Over Time', fontsize=title_fontsize, pad=20)
    plt.xlabel('Year-Month', fontsize=label_fontsize, labelpad=15)
    plt.ylabel('Days Missed By', fontsize=label_fontsize, labelpad=15)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_grouped_confusion_matrix(group_conf_matrix, unique_groups, groups, name):
    """
    Generates and saves a normalized grouped confusion matrix as a heatmap. The matrix visualizes the proportions of 
    predictions for each true group across all predicted groups. 
    The resulting visualization is saved as a PNG file using the specified name and PLOTS_PATH.

    Parameters:
    group_conf_matrix (ndarray): A 2D array representing the confusion matrix of the grouped data.
    unique_groups (list): A list of unique group labels used as axis labels in the heatmap.
    groups (dict): A dictionary mapping group names ('No', 'Early', 'Middle', 'Late') to their respective value ranges 
                   for annotation purposes.
    name (str): The name used for saving the PNG file using the PLOTS_PATH variable.

    Returns:
    None
    
    The labels are categorized into four groups based on the groups dictionary:
        - 'No': Labels corresponding to index 0.
        - 'Early': Labels with indices from 1 to 6.
        - 'Middle': Labels with indices from 7 to 13.
        - 'Late': Labels with indices from 14 to the number of classes (default CHUNK_SIZE).
    """
    
    plt.figure(figsize=plot_size)
    
    # Normalize by rows to show proportions
    row_sums = group_conf_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = group_conf_matrix / row_sums.clip(min=1)
    
    ax = sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt='.2%',
        cmap='PuBuGn',
        xticklabels=unique_groups,
        yticklabels=unique_groups,
        cbar_kws={'label': 'Proportion'}
    )
    ax.figure.axes[-1].yaxis.label.set_size(label_fontsize)

    plt.xlabel('Predicted Group', fontsize=label_fontsize)
    plt.ylabel('True Group', fontsize=label_fontsize)
    plt.title('Normalized Grouped Confusion Matrix', fontsize=title_fontsize, pad=20)
    
    # Add group descriptions below the plot
    group_descriptions = [
        f"No: {groups['No']}",
        f"Early: Days {min(groups['Early'])} - {max(groups['Early'])}",
        f"Middle: Days {min(groups['Middle'])} - {max(groups['Middle'])}",
        f"Late: Days {min(groups['Late'])} - {max(groups['Late'])}"
    ]
    description_text = '\n'.join(group_descriptions)
    plt.figtext(0.5, -0.15, description_text, ha='center', fontsize=tick_fontsize, wrap=True)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()

def plot_statistics(report, name):
    """
    Generates and saves a bar plot visualizing precision, recall, and F1 scores for each class extracted from the 
    classification report. The plot includes separate bars for each metric grouped by class. 
    The resulting visualization is saved as a PNG file using the specified name and PLOTS_PATH.

    Parameters:
    report (dict): A dictionary containing evaluation metrics for each class, with keys as class labels and values 
                   as dictionaries of the precision, recall, and F1-score metrics.
    name (str): The name used for saving the PNG file using the PLOTS_PATH variable.

    Returns:
    None
    """

    precision = []
    recall = []
    f1_score = []
    class_labels = []

    for class_idx in range(21):
        class_key = str(class_idx)
        if class_key in report:
            metrics = report[class_key]
            if isinstance(metrics, dict):
                precision.append(metrics.get('precision', 0))
                recall.append(metrics.get('recall', 0))
                f1_score.append(metrics.get('f1-score', 0))
                class_labels.append(class_key)

    plt.figure(figsize=plot_size)
    bar_width = 0.25
    positions = np.arange(len(precision))

    plt.bar(positions - bar_width, precision, width=bar_width, color='#1f77b4', edgecolor='grey', label='Precision')
    plt.bar(positions, recall, width=bar_width, color='#2ca02c', edgecolor='grey', label='Recall')
    plt.bar(positions + bar_width, f1_score, width=bar_width, color='#d62728', edgecolor='grey', label='F1-Score')

    plt.xlabel('Classes', fontsize=label_fontsize, labelpad=15)
    plt.ylabel('Scores', fontsize=label_fontsize, labelpad=15)
    plt.xticks(positions, class_labels, rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title('Precision, Recall, and F1 Scores per Class', fontsize=title_fontsize, pad=20)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()

def plot_neu_displacements(df, idx, name, zoom_window=10):
    """
    The function creates a main plot with displacements in North (N), East (E), and Up (U) directions 
    along with true and predicted labels visualized as bar overlays. For each earthquake, additional zoomed-in plots 
    are created to highlight displacement and label details within a specified window.
    The resulting visualization is saved as a PNG file using the specified name, inex and PLOTS_PATH.

    Parameters:
    df (pd.DataFrame): DataFrame containing displacement data and labels. The index should be datetime, and it must 
                       include columns 'N', 'E', 'U', 'labels_sum', and 'preds_sum'.
    idx (int): Index used for saving the plots, as multiple stations are selected.
    name (str): Name used for saving the plots.
    zoom_window (int): Number of days around each earthquake for the zoomed-in view (default: 10).

    Returns:
    None
    """
    
    df.index = pd.to_datetime(df.index)
    fig, axs = plt.subplots(3, 1, figsize=subplot_size, sharex=True)
    fig.suptitle(f'Displacement and Earthquakes Over Time for Station: {df.name}', fontsize=title_fontsize)
    
    earthquake_label_added = False
    pred_label_added = False
    
    displacement_components = ['N', 'E', 'U']
    for i, component in enumerate(displacement_components):
        axs[i].plot(df.index, df[component], color='gray', linewidth=2)
        
        # Create twin axis for the counts
        ax2 = axs[i].twinx()
        ax2.bar(df.index, df['labels_sum'], width=2, color='blue', alpha=0.3, label='True Labels'if i == 0 and not earthquake_label_added else "")
        earthquake_label_added = True
        ax2.bar(df.index, df['preds_sum'], width=2, color='red', alpha=0.3, label='Predictions' if i == 0 and not pred_label_added else "")
        pred_label_added = True
        axs[i].set_ylabel(f'{component} Displacement', fontsize=label_fontsize)
        ax2.set_ylabel('Counts', fontsize=label_fontsize)

        axs[i].tick_params(axis='y', which='major', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', which='major', labelsize=tick_fontsize)
        if i == 0:
            ax2.legend(loc='best', fontsize=legend_fontsize)

    axs[-1].set_xlabel('Date', fontsize=label_fontsize)
    axs[-1].tick_params(axis='x', which='major', labelsize=tick_fontsize, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}{idx}.png', dpi=dpi_setting)
    plt.close()

    # Zoomed-In Plot(s)
    zoom_dates = df[df['labels_sum'] > 0].index

    for j, date in enumerate(zoom_dates):
        start_date = date - pd.Timedelta(days=zoom_window)
        end_date = date + pd.Timedelta(days=zoom_window)
        zoom_df = df.loc[start_date:end_date]

        fig, axs = plt.subplots(3, 1, figsize=subplot_size, sharex=True)
        fig.suptitle(f'Zoomed Displacement and Earthquakes for Station {df.name} around {date.date()}', fontsize=title_fontsize)
        
        earthquake_label_added = False
        pred_label_added = False

        for i, (ax, component) in enumerate(zip(axs, ['N', 'E', 'U'])):
            ax.plot(zoom_df.index, zoom_df[component], color='gray', linewidth=2)
            ax2_zoom = ax.twinx()
            label_true = 'True Labels' if not earthquake_label_added else None
            label_pred = 'Predictions' if not pred_label_added else None
            ax2_zoom.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='blue', alpha=0.3, label=label_true)
            ax2_zoom.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='red', alpha=0.3, label=label_pred)
            
            earthquake_label_added = True
            pred_label_added = True
            ax.set_ylabel(f'{component} Displacement', fontsize=label_fontsize)
            ax2_zoom.set_ylabel('Counts', fontsize=label_fontsize)
            ax.tick_params(axis='y', which='major', labelsize=tick_fontsize)
            ax2_zoom.tick_params(axis='y', which='major', labelsize=tick_fontsize)
            if i == 0:
                ax2_zoom.legend(loc='best', fontsize=legend_fontsize)

        axs[-1].set_xlabel("Date", fontsize=label_fontsize)
        axs[-1].tick_params(axis='x', which='major', labelsize=tick_fontsize, rotation=45)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_PATH}_{name}_zoom_{idx}_{j}.png', dpi=dpi_setting)
        plt.close()

def plot_neu_mean_displacements(df, name, idx, zoom_window=10):
    """
    The function creates a main plot with displacements in North (N), East (E), and Up (U) directions 
    along with true and mean predicted labels visualized as vertical lines. For each earthquake, additional zoomed-in plots 
    are created to highlight displacement and label details within a specified time window.
    The resulting visualizations are saved as PNG files using the specified name, index, and PLOTS_PATH.

    Parameters:
    df (pd.DataFrame): DataFrame containing displacement data and labels. The index should be datetime, and it must 
                       include columns 'N', 'E', 'U', 'labels_sum', and 'mean_day_pred'.
    name (str): Name used for saving the plots.
    idx (int): Index used for saving the plots, as multiple stations are processed.
    zoom_window (int): Number of days around each earthquake for the zoomed-in view (default: 10).

    Returns:
    None
    """
    
    fig, axs = plt.subplots(3, 1, figsize=subplot_size, sharex=True)
    fig.suptitle(f'Displacement and Mean Earthquakes Over Time for Station: {df.name}', fontsize=title_fontsize)
    
    earthquake_label_added = False
    pred_label_added = False

    for i, component in enumerate(['N', 'E', 'U']):
        axs[i].plot(df.index, df[component], color='gray', linewidth=2)
        for label_date in df[df['labels_sum'] > 0].index:
            axs[i].axvline(label_date, color='blue', linestyle='-', linewidth=2, label="Actual Earthquake Label" if i == 0 and not earthquake_label_added else "")
            earthquake_label_added = True
        for pred_date in df[df['mean_day_pred'] > 0].index:
            axs[i].axvline(pred_date, color='red', linestyle='--', linewidth=2, label="Predicted Earthquake (Mean)" if i == 0 and not pred_label_added else "")
            pred_label_added = True
        
        axs[i].set_ylabel(f"{component} Displacement", fontsize=label_fontsize)
        if i == 0:
            axs[i].legend(loc='best', fontsize=legend_fontsize, frameon=False)
        axs[i].grid(True, linestyle='--', alpha=0.5)
        axs[i].tick_params(axis='y', which='major', labelsize=tick_fontsize)

    axs[-1].set_xlabel("Date", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}_{idx}.png', dpi=dpi_setting)
    plt.close()

    # Zoomed-in plot(s)
    zoom_dates = df[df['labels_sum'] > 0].index

    for date in zoom_dates:
        start_date = date - pd.Timedelta(days=zoom_window)
        end_date = date + pd.Timedelta(days=zoom_window)
        zoom_df = df.loc[start_date:end_date]

        fig, axs = plt.subplots(3, 1, figsize=subplot_size, sharex=True)
        fig.suptitle(f'Zoomed Displacement and Mean Earthquakes for Station {df.name} around {date.date()}', fontsize=title_fontsize)
        
        earthquake_label_added = False
        pred_label_added = False

        for i, component in enumerate(['N', 'E', 'U']):
            axs[i].plot(zoom_df.index, zoom_df[component], color='gray', linewidth=2)
            for label_date in zoom_df[zoom_df['labels_sum'] > 0].index:
                axs[i].axvline(label_date, color='blue', linestyle='-', linewidth=2, label="Actual Earthquake Label" if i == 0 and not earthquake_label_added else "")
                earthquake_label_added = True
            for pred_date in zoom_df[zoom_df['mean_day_pred'] > 0].index:
                axs[i].axvline(pred_date, color='red', linestyle='--', linewidth=2, label="Predicted Earthquake (Mean)" if i == 0 and not pred_label_added else "")
                pred_label_added = True
            
            axs[i].set_ylabel(f"{component} Displacement", fontsize=label_fontsize)
            if i == 0:
                axs[i].legend(loc='best', fontsize=legend_fontsize, frameon=False)
            axs[i].grid(True, linestyle='--', alpha=0.5)
            axs[i].tick_params(axis='y', which='major', labelsize=tick_fontsize)

        axs[-1].set_xlabel("Date", fontsize=label_fontsize)
        plt.xticks(rotation=45, fontsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_PATH}_{name}_zoom_{idx}.png', dpi=dpi_setting)
        plt.close()
        
def plot_mean_preds_statistics(values, name, chunk_size=CHUNK_SIZE):
    """
    This function generates a bar chart to visualize the prediction accuracy statistics. The bars represent 
    different categories of prediction performance, including exact matches, predictions within a tolerance window, 
    undetected earthquakes, and false positives. The chart is saved as a PNG file using the specified name and PLOTS_PATH variable.

    Parameters:
    values (tuple): A tuple containing the percentages of each category. The order should match 
                    the labels: ['Exact Matches', 'Predicted within chunk_size Days', 'Undetected', 'False Positives'].
    name (str): Name used for saving the plot.
    chunk_size (int): The window (in days) within which a predicted label is considered acceptable (default: CHUNK_SIZE).

    Returns:
    None
    """
    
    labels = ['Exact Matches', f'Predicted within {chunk_size} Days', 'Undetected', 'False Positives']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    
    plt.figure(figsize=plot_size)
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.ylabel('Percentage', fontsize=label_fontsize, labelpad=15)
    plt.title("Prediction Accuracy Breakdown", fontsize=title_fontsize, pad=20)
    plt.ylim(0, 100)
    
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=legend_fontsize, color='black')
    
    plt.xticks(fontsize=tick_fontsize, rotation=45, ha='center')
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting, bbox_inches='tight')
    plt.close()
    
def extract_unique_geometries(geometries):
    """
    Extract unique geometries from a list of Point geometries.
    
    Parameters:
    geometries (list): List of geometry Points.
    
    Returns:
    GeoSeries: A GeoSeries of unique geometries.
    """
    return gpd.GeoSeries(list(set(geometries)), crs="EPSG:4326")
    
def plot_station_geometries(geometries):    
    """
    This function plots the locations of train, evaluation, and test stations on a map, with each station type 
    represented by different colors and markers. If working on a cluster, the geometries need to be saved in a 
    GeoJSON file due to network restrictions. Otherwise, the stations are visualized on a basemap from OpenStreetMap.
    The resulting plot is saved as a PNG file.

    Parameters:
    geometries (tuple): A tuple of lists containing geometries for train, eval, and test stations.

    Returns:
    None
    """
    
    # Unpack train, eval, and test station geometries and remove duplicates
    train_stations = extract_unique_geometries(geometries[0])
    eval_stations = extract_unique_geometries(geometries[1])
    test_stations = extract_unique_geometries(geometries[2])

    gdf = gpd.GeoDataFrame({
        'geometry': pd.concat([train_stations, eval_stations, test_stations]),
        'type': (['Train'] * len(train_stations) +
                 ['Eval'] * len(eval_stations) +
                 ['Test'] * len(test_stations))
    }, crs="EPSG:4326")
    
    # Remaining code not possible on cluster due to network restrictions
    gdf.to_file(GEOMETRIES_PATH, driver="GeoJSON")
    return

    # Transform coordinates to match the basemap's CRS (Web Mercator)
    gdf.to_crs(epsg=3857, inplace=True)
    
    fig, ax = plt.subplots(figsize=subplot_size)
    station_types = ['Train', 'Eval', 'Test']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    markers = ['o', 's', '^']
    
    for station_type, color, marker in zip(station_types, colors, markers):
        gdf[gdf['type'] == station_type].plot(
            ax=ax, color=color, marker=marker, markersize=80,
            label=f'{station_type} Stations', alpha=0.8
        )
    
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.xlabel("Longitude", fontsize=label_fontsize, labelpad=15)
    plt.ylabel("Latitude", fontsize=label_fontsize, labelpad=15)
    plt.title("Station Locations by Usage Type", fontsize=title_fontsize, pad=20)
    
    plt.legend(fontsize=legend_fontsize, loc='best', frameon=True, title="Station Types", title_fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.savefig(f'{PLOTS_PATH}_stations.png', dpi=dpi_setting, bbox_inches='tight')
    plt.close()
    



