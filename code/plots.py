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

# Define consistent font sizes for legends and labels
legend_fontsize = 16
label_fontsize = 20
title_fontsize = 22
tick_fontsize = 14
dpi_setting = 300
plot_size = (12, 8)
subplot_size = (14, 12)

def plot_eval_metrics(eval_results, name):
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
    plt.figure(figsize=plot_size)
    ax = plot_optimization_history(study, target_name='Validation loss')
    ax.set_xlabel('Trial', fontsize=label_fontsize)
    ax.set_ylabel('Validation loss (Macro F1 Score)', fontsize=label_fontsize)
    ax.set_title(f'Optimization history for {MODEL_TYPE}', fontsize=title_fontsize, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + 'optimization_history.png', dpi=dpi_setting)
    
    plt.figure(figsize=plot_size)
    ax = plot_param_importances(study)
    ax.set_xlabel('Importance for validation loss', fontsize=label_fontsize)
    ax.set_ylabel('Hyperparameter', fontsize=label_fontsize)
    ax.set_title(f'Hyperparameter Importances for {MODEL_TYPE}', fontsize=title_fontsize, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + 'parameter_importances.png', dpi=dpi_setting)
    
    plt.figure(figsize=plot_size)
    ax = plot_contour(study, params=[parameters[0], parameters[1]], target_name='Validation loss')
    ax.set_xlabel(' '.join([word.capitalize() for word in parameters[0].replace('_', ' ').split()]), fontsize=label_fontsize)
    ax.set_ylabel(' '.join([word.capitalize() for word in parameters[1].replace('_', ' ').split()]), fontsize=label_fontsize)
    ax.set_title(f'Contour plot for {MODEL_TYPE}', fontsize=title_fontsize, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + 'contour.png', dpi=dpi_setting)
    
def plot_histogram(series, xlabel, ylabel, title, name):
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
    plt.figure(figsize=plot_size)
    sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        mask=mask,
        cbar_kws={'label': 'Log(Number of Misclassifications)'},
        linewidths=0.5,
        linecolor='lightgrey'
    )
    
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
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()
    
def plot_heatmap(heatmap_data, mask, name):
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
    plt.title('Logarithmic Heatmap of Missed Predictions Over Time', fontsize=20, pad=20)
    plt.xlabel('Year-Month', fontsize=label_fontsize, labelpad=15)
    plt.ylabel('Days Missed By', fontsize=label_fontsize, labelpad=15)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_grouped_confusion_matrix(group_conf_matrix, unique_groups, groups, name):
    plt.figure(figsize=plot_size)
    
    # Normalize by rows to show proportions
    row_sums = group_conf_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = group_conf_matrix / row_sums.clip(min=1)  # Avoid division by zero
    
    # Plot the heatmap with percentages
    ax = sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt='.2%',  # Display values as percentages
        cmap='PuBuGn',
        xticklabels=unique_groups,
        yticklabels=unique_groups,
        cbar_kws={'label': 'Proportion'}
    )
    ax.figure.axes[-1].yaxis.label.set_size(label_fontsize)

    # Add labels and title
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

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()

def plot_statistics(report, name):
    precision = []
    recall = []
    f1_score = []
    class_labels = []

    # Extract metrics from report dictionary
    for class_idx in range(21):
        class_key = str(class_idx)
        if class_key in report:
            metrics = report[class_key]
            if isinstance(metrics, dict):
                precision.append(metrics.get('precision', 0))
                recall.append(metrics.get('recall', 0))
                f1_score.append(metrics.get('f1-score', 0))
                class_labels.append(class_key)

    # Define bar width and positions
    plt.figure(figsize=plot_size)
    bar_width = 0.25
    positions = np.arange(len(precision))

    # Plot bars for precision, recall, and f1 score
    plt.bar(positions - bar_width, precision, width=bar_width, color='#1f77b4', edgecolor='grey', label='Precision')
    plt.bar(positions, recall, width=bar_width, color='#2ca02c', edgecolor='grey', label='Recall')
    plt.bar(positions + bar_width, f1_score, width=bar_width, color='#d62728', edgecolor='grey', label='F1-Score')

    plt.xlabel('Classes', fontsize=label_fontsize, labelpad=15)
    plt.ylabel('Scores', fontsize=label_fontsize, labelpad=15)
    plt.xticks(positions, class_labels, rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title('Precision, Recall, and F1 Scores per Class', fontsize=title_fontsize, pad=20)
    
    # Grid and legend
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=dpi_setting)
    plt.close()

def plot_neu_displacements(df, idx, name, zoom_window=10):
    """
    Plots the N, E, and U columns from a DataFrame with datetime index on the x-axis in separate subplots,
    including 'labels_sum' and 'preds_sum' as bar plots in each subplot on a secondary y-axis,
    with zoomed-in views around dates where 'labels_sum' > 0.
    """
    
    df.index = pd.to_datetime(df.index)
    
    # Create the main plot with three subplots (N, E, U)
    fig, axs = plt.subplots(3, 1, figsize=subplot_size, sharex=True)
    fig.suptitle(f'Displacement and Earthquakes Over Time for Station: {df.name}', fontsize=title_fontsize)
    
    earthquake_label_added = False
    pred_label_added = False
    
    displacement_components = ['N', 'E', 'U']
    for i, component in enumerate(displacement_components):
        # Plot displacement
        axs[i].plot(df.index, df[component], color='gray', linewidth=2)
        
        # Create twin axis for the counts
        ax2 = axs[i].twinx()
        ax2.bar(df.index, df['labels_sum'], width=2, color='blue', alpha=0.3, label='True Labels'if i == 0 and not earthquake_label_added else "")
        earthquake_label_added = True
        ax2.bar(df.index, df['preds_sum'], width=2, color='red', alpha=0.3, label='Predictions' if i == 0 and not pred_label_added else "")
        pred_label_added = True
        # Set labels for y-axes
        axs[i].set_ylabel(f'{component} Displacement', fontsize=label_fontsize)
        ax2.set_ylabel('Counts', fontsize=label_fontsize)

        # Set tick parameters
        axs[i].tick_params(axis='y', which='major', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', which='major', labelsize=tick_fontsize)

        # Set legend for the twin axis
        if i == 0:
            ax2.legend(loc='best', fontsize=legend_fontsize)

    # X-axis label
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
            
            # Add 'True Labels' and 'Predictions' in the first subplot only
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
    Combines full plot and zoomed-in plots for N, E, and U displacements with earthquake labels and mean predictions as vertical lines.
    """
    
    # Full plot with displacement and earthquake labels
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
    labels = ['Exact Matches', f'Predicted within {chunk_size} Days', 'Undetected', 'False Positives']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    
    plt.figure(figsize=plot_size)
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.ylabel('Percentage', fontsize=label_fontsize, labelpad=15)
    plt.title("Prediction Accuracy Breakdown", fontsize=title_fontsize, pad=20)
    plt.ylim(0, 100)
    
    # Adding percentage labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=legend_fontsize, color='black')
    
    plt.xticks(fontsize=tick_fontsize, rotation=45, ha='center')
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot to specified path
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
    
    # Remaining code not possible on cluster due to network restrictions
    gdf.to_file(GEOMETRIES_PATH, driver="GeoJSON")
    return

    # Transform coordinates to match the basemap's CRS (Web Mercator)
    gdf.to_crs(epsg=3857, inplace=True)
    
    fig, ax = plt.subplots(figsize=plot_size)
    station_types = ['Train', 'Eval', 'Test', 'Simulated']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    markers = ['o', 's', '^', 'x']
    
    for station_type, color, marker in zip(station_types, colors, markers):
        gdf[gdf['type'] == station_type].plot(
            ax=ax, color=color, marker=marker, markersize=80,
            label=f'{station_type} Stations', alpha=0.8
        )
    
    # Add basemap from OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Labeling and styling
    plt.xlabel("Longitude", fontsize=label_fontsize, labelpad=15)
    plt.ylabel("Latitude", fontsize=label_fontsize, labelpad=15)
    plt.title("Station Locations by Usage Type", fontsize=title_fontsize, pad=20)
    
    # Legend settings
    plt.legend(fontsize=legend_fontsize, loc='best', frameon=True, title="Station Types", title_fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    
    # Save plot
    plt.savefig(f'{PLOTS_PATH}_stations.png', dpi=dpi_setting, bbox_inches='tight')
    plt.close()
    



