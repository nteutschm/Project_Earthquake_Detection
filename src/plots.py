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

def plot_eval_metrics(eval_results, name):
    epochs = len(eval_results['validation_0']['mlogloss'])
    x_axis = range(epochs)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis, eval_results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, eval_results['validation_1']['mlogloss'], label='Eval')
    ax.set_title(f'{MODEL_TYPE} Log Loss Over Epochs', fontsize=24, pad=20)
    ax.set_xlabel('Epochs', fontsize=20, labelpad=15)
    ax.set_ylabel('Log Loss', fontsize=20, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_optimization(study, parameters):
    plt.figure(figsize=(12, 8))
    ax = plot_optimization_history(study, target_name='Validation loss')
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Validation loss (Macro F1 Score)', fontsize=20)
    ax.set_title(f'Optimization history for {MODEL_TYPE}', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'optimization_history.png')
    
    plt.figure(figsize=(12, 8))
    ax = plot_param_importances(study)
    ax.set_xlabel('Importance for validation loss', fontsize=20)
    ax.set_ylabel('Hyperparameter', fontsize=20)
    ax.set_title(f'Hyperparameter Importances for {MODEL_TYPE}', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'parameter_importances.png')
    
    plt.figure(figsize=(12, 8))
    ax = plot_contour(study, params=[parameters[0], parameters[1]], target_name='Validation loss')
    ax.set_xlabel(' '.join([word.capitalize() for word in parameters[0].replace('_', ' ').split()]), fontsize=20)
    ax.set_ylabel(' '.join([word.capitalize() for word in parameters[1].replace('_', ' ').split()]), fontsize=20)
    ax.set_title(f'Contour plot for {MODEL_TYPE}', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig(PLOTS_PATH+'contour.png')
    
def plot_histogram(series, xlabel, ylabel, title, name):
    plt.figure(figsize=(12, 10))
    plt.bar(series.index, series.values, color='#4c9fdb')
    plt.xlabel(xlabel, fontsize=20, labelpad=15)
    plt.ylabel(ylabel, fontsize=20, labelpad=15)
    plt.title(title, fontsize=24, pad=20)
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_heatmap(heatmap_data, mask, name):
    plt.figure(figsize=(16, 8))
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
        fontsize=14,
        rotation=45,
        ha='right'
    )
    plt.yticks(fontsize=14)
    plt.title('Logarithmic Heatmap of Missed Predictions Over Time', fontsize=20, pad=20)
    plt.xlabel('Year-Month', fontsize=16, labelpad=15)
    plt.ylabel('Days Missed By', fontsize=16, labelpad=15)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_cumulative_metrics(cumulative_metrics_df, name):
    plt.figure(figsize=(14, 8))
    plt.plot(
        cumulative_metrics_df['Date'], 
        cumulative_metrics_df['Smoothed Cumulative Accuracy'], 
        label='Cumulative Accuracy', 
        color='cornflowerblue', linewidth=2
    )
    plt.plot(
        cumulative_metrics_df['Date'], 
        cumulative_metrics_df['Smoothed Cumulative Precision'], 
        label='Cumulative Precision', 
        color='darkorange', linewidth=2
    )
    plt.plot(
        cumulative_metrics_df['Date'], 
        cumulative_metrics_df['Smoothed Cumulative Recall'], 
        label='Cumulative Recall', 
        color='mediumseagreen', linewidth=2
    )
    plt.plot(
        cumulative_metrics_df['Date'], 
        cumulative_metrics_df['Smoothed Cumulative F1'], 
        label='Cumulative F1 Score', 
        color='mediumpurple', linewidth=2
    )
    plt.title('Cumulative Metrics Over Time (Smoothed over 180 days)', fontsize=20, pad=20)
    plt.xlabel('Date', fontsize=16, labelpad=15)
    plt.ylabel('Metric Value', fontsize=16, labelpad=15)
    
    plt.xticks(
        ticks=cumulative_metrics_df['Date'][::10000],
        labels=cumulative_metrics_df['Date'][::10000].dt.strftime('%Y-%m'),
        fontsize=12,
        rotation=45,
        ha='right'
    )
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
    plt.legend(fontsize=14, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_grouped_confusion_matrix(group_conf_matrix, unique_groups, groups, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        group_conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=unique_groups, 
        yticklabels=unique_groups,
        cbar_kws={'label': 'Counts'}
    )

    plt.xlabel('Predicted Group', fontsize=14)
    plt.ylabel('True Group', fontsize=14)
    plt.title('Grouped Confusion Matrix', fontsize=16, pad=20)
    group_descriptions = [
        f"No: {groups['No']}",
        f"Early: Days {min(groups['Early'])} - {max(groups['Early'])}",
        f"Middle: Days {min(groups['Middle'])} - {max(groups['Middle'])}",
        f"Late: Days {min(groups['Late'])} - {max(groups['Late'])}"
    ]
    description_text = '\n'.join(group_descriptions)
    
    plt.figtext(0.5, -0.15, description_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
    plt.close()
    
def plot_statistics(report, name):
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

    plt.figure(figsize=(12, 10))
    bar_width = 0.25
    positions = np.arange(len(precision))

    plt.bar(positions - bar_width, precision, width=bar_width, color='#1f77b4', edgecolor='grey', label='Precision')
    plt.bar(positions, recall, width=bar_width, color='#2ca02c', edgecolor='grey', label='Recall')
    plt.bar(positions + bar_width, f1_score, width=bar_width, color='#d62728', edgecolor='grey', label='F1-Score')

    plt.xlabel('Classes', fontsize=20, labelpad=15, fontweight='bold')
    plt.ylabel('Scores', fontsize=20, labelpad=15, fontweight='bold')
    plt.xticks(positions, class_labels, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Precision, Recall, and F1 Scores per Class', fontsize=24, pad=20, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}.png', dpi=300)
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
    
    # Create the main plot with three subplots (N, E, U)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Displacement and Earthquakes Over Time for Station: {df.name}', fontsize=24, fontweight='bold')

    # Plot for 'N'
    axs[0].plot(df.index, df['N'], color='blue', label='N', linewidth=2)
    ax2_n = axs[0].twinx()  # Create a secondary y-axis
    ax2_n.bar(df.index, df['labels_sum'], width=1, color='gray', alpha=0.3, label='True Labels')
    ax2_n.bar(df.index, df['preds_sum'], width=1, color='orange', alpha=0.3, label='Predictions')
    axs[0].set_ylabel('N Displacement', fontsize=18, fontweight='bold')
    ax2_n.set_ylabel('Counts', fontsize=18, fontweight='bold')
    axs[0].legend(loc='upper left', fontsize=16)
    ax2_n.legend(loc='upper right', fontsize=16)

    # Plot for 'E'
    axs[1].plot(df.index, df['E'], color='blue', label='E', linewidth=2)
    ax2_e = axs[1].twinx()  # Create a secondary y-axis
    ax2_e.bar(df.index, df['labels_sum'], width=1, color='gray', alpha=0.3)
    ax2_e.bar(df.index, df['preds_sum'], width=1, color='orange', alpha=0.3)
    axs[1].set_ylabel('E Displacement', fontsize=18, fontweight='bold')
    ax2_e.set_ylabel('Counts', fontsize=18, fontweight='bold')

    # Plot for 'U'
    axs[2].plot(df.index, df['U'], color='blue', label='U', linewidth=2)
    ax2_u = axs[2].twinx()  # Create a secondary y-axis
    ax2_u.bar(df.index, df['labels_sum'], width=1, color='gray', alpha=0.3)
    ax2_u.bar(df.index, df['preds_sum'], width=1, color='orange', alpha=0.3)
    axs[2].set_ylabel('U Displacement', fontsize=18, fontweight='bold')
    ax2_u.set_ylabel('Counts', fontsize=18, fontweight='bold')

    # X-axis label
    axs[2].set_xlabel('Date', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_{name}{idx}.png', dpi=300)
    plt.close()

    # Zoomed-In Plot(s)
    zoom_dates = df[df['labels_sum'] > 0].index

    for i, date in enumerate(zoom_dates):
        start_date = date - pd.Timedelta(days=zoom_window)
        end_date = date + pd.Timedelta(days=zoom_window)
        zoom_df = df.loc[start_date:end_date]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Zoomed Displacement and Earthquakes for Station {df.name} around {date.date()}', fontsize=24, fontweight='bold')

        # Zoom for 'N'
        axs[0].plot(zoom_df.index, zoom_df['N'], color='blue', label='N', linewidth=2)
        ax2_zoom_n = axs[0].twinx()  # Create a secondary y-axis
        ax2_zoom_n.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='gray', alpha=0.3, label='True Labels')
        ax2_zoom_n.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='orange', alpha=0.3, label='Predictions')
        axs[0].set_ylabel('N Displacement', fontsize=18, fontweight='bold')
        ax2_zoom_n.set_ylabel('Counts', fontsize=18, fontweight='bold')
        axs[0].legend(loc='upper left', fontsize=16)
        ax2_zoom_n.legend(loc='upper right', fontsize=16)

        # Zoom for 'E'
        axs[1].plot(zoom_df.index, zoom_df['E'], color='blue', label='E', linewidth=2)
        ax2_zoom_e = axs[1].twinx()  # Create a secondary y-axis
        ax2_zoom_e.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='gray', alpha=0.3)
        ax2_zoom_e.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='orange', alpha=0.3)
        axs[1].set_ylabel('E Displacement', fontsize=18, fontweight='bold')
        ax2_zoom_e.set_ylabel('Counts', fontsize=18, fontweight='bold')

        # Zoom for 'U'
        axs[2].plot(zoom_df.index, zoom_df['U'], color='blue', label='U', linewidth=2)
        ax2_zoom_u = axs[2].twinx()  # Create a secondary y-axis
        ax2_zoom_u.bar(zoom_df.index, zoom_df['labels_sum'], width=1, color='gray', alpha=0.3)
        ax2_zoom_u.bar(zoom_df.index, zoom_df['preds_sum'], width=1, color='orange', alpha=0.3)
        axs[2].set_ylabel('U Displacement', fontsize=18, fontweight='bold')
        ax2_zoom_u.set_ylabel('Counts', fontsize=18, fontweight='bold')

        axs[2].set_xlabel('Date', fontsize=18, fontweight='bold')
        plt.xticks(rotation=45, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_PATH}_{name}_zoom_{idx}_{i}.png', dpi=300)
        plt.close()
        
def plot_full_and_zoomed_displacement(df, name, idx, zoom_window=10):
    """
    Combines full plot and zoomed-in plots for N, E, and U displacements with earthquake labels and mean predictions as vertical lines.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'N', 'E', 'U', 'labels_sum', and 'mean_day_pred'.
    name (str): Name of the GPS station for the title.
    zoom_window (int): Number of days before and after each event to include in the zoomed plot.
    """
    
    # Full plot with displacement and earthquake labels
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    station_name = df.name
    fig.suptitle(f"Station: {station_name} - Earthquake Labels and Predicted Dates (Full View)", fontsize=22)

    for i, component in enumerate(['N', 'E', 'U']):
        # Plot displacement data
        axs[i].plot(df.index, df[component], label=f"{component} Displacement", color='gray', linewidth=2)
        
        # Plot earthquake labels (true events) as vertical blue lines
        for label_date in df[df['labels_sum'] > 0].index:
            axs[i].axvline(label_date, color='blue', linestyle='-', linewidth=2, label="Actual Earthquake Label" if i == 0 else "")
        
        # Plot predicted earthquake dates (mean) as vertical red dashed lines
        for pred_date in df[df['mean_day_pred'] > 0].index:
            axs[i].axvline(pred_date, color='red', linestyle='--', linewidth=2, label="Predicted Earthquake (Mean)" if i == 0 else "")
        
        axs[i].set_ylabel(f"{component} Displacement", fontsize=18, labelpad=15)
        axs[i].legend(loc='upper left', fontsize=14, frameon=False)
        axs[i].grid(True, linestyle='--', alpha=0.5)

    axs[-1].set_xlabel("Date", fontsize=18, labelpad=15)
    plt.xticks(rotation=45, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{PLOTS_PATH}_{name}_{idx}.png')
    plt.close()

    # Zoomed-in plot(s)
    zoom_dates = df[df['labels_sum'] > 0].index

    for date in zoom_dates:
        start_date = date - pd.Timedelta(days=zoom_window)
        end_date = date + pd.Timedelta(days=zoom_window)
        zoom_df = df.loc[start_date:end_date]

        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'Zoomed View for Station: {station_name} around {date.date()}', fontsize=22, pad=20)

        for i, component in enumerate(['N', 'E', 'U']):
            # Plot displacement data
            axs[i].plot(zoom_df.index, zoom_df[component], color='gray', label=f"{component} Displacement", linewidth=2)
            
            # Plot earthquake labels (true events) as vertical blue lines
            for label_date in zoom_df[zoom_df['labels_sum'] > 0].index:
                axs[i].axvline(label_date, color='blue', linestyle='-', linewidth=2, label="Actual Earthquake Label" if i == 0 else "")
            
            # Plot predicted earthquake dates (mean) as vertical red dashed lines
            for pred_date in zoom_df[zoom_df['mean_day_pred'] > 0].index:
                axs[i].axvline(pred_date, color='red', linestyle='--', linewidth=2, label="Predicted Earthquake (Mean)" if i == 0 else "")
            
            axs[i].set_ylabel(f"{component} Displacement", fontsize=18, labelpad=15)
            axs[i].legend(loc='upper left', fontsize=14, frameon=False)
            axs[i].grid(True, linestyle='--', alpha=0.5)

        axs[-1].set_xlabel("Date", fontsize=18, labelpad=15)
        plt.xticks(rotation=45, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f'{PLOTS_PATH}_{name}_zoom_{idx}.png')
        plt.close()
        
def plot_mean_preds_statistics(stats, name, chunk_size=CHUNK_SIZE):
    labels = ['Exact Matches', f'Predicted within {chunk_size} Days', 'Undetected', 'False Positives']
    values = [stats[0], stats[1], stats[2], stats[3]]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    plt.figure(figsize=(10, 8))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Percentage', fontsize=20, labelpad=15)
    plt.title("Prediction Accuracy Breakdown", fontsize=24, pad=20)
    plt.ylim(0, 100)
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=16, color='black')
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{PLOTS_PATH}_{name}.png')
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
    
    fig, ax = plt.subplots(figsize=(14, 12))
    for station_type, color, marker in zip(['Train', 'Eval', 'Test', 'Simulated'],
                                           ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'],
                                           ['o', 's', '^', 'x']):
        gdf[gdf['type'] == station_type].plot(
            ax=ax, color=color, marker=marker, markersize=80, label=f'{station_type} Stations', alpha=0.8
        )
    
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.xlabel("Longitude", fontsize=18, labelpad=15)
    plt.ylabel("Latitude", fontsize=18, labelpad=15)
    plt.title("Station Locations by Usage Type", fontsize=22, pad=20)
    plt.legend(fontsize=16, loc='upper left', frameon=True, title="Station Types", title_fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}_stations.png')
    plt.close()
    



