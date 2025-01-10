# Project Earthquake Detection

This repository contains the workflow and analysis for Project Earthquake Detection. The project code and data are organized into folders for ease of use and clarity.

## Folder Structure

### `Delivery/analysis`
This folder contains the initial scripts for downloading the data and conducting a preliminary analysis. These scripts are independent of the main workflow and serve as the foundation for the subsequent steps.

### `Delivery/predicting`
The primary workflow is located here, organized into multiple files for better structure and maintainability:  
- **`main.py`**: Entry point for running the XGBoost model.  
- **`main_tcn.py`**: Entry point for running the TCN model.  
- **`variables.py`**: Contains globally defined variables for easy modification, including path variables for all models except TCN.  
- **`variables_tcn.py`**: Holds specific variables for the TCN model due to its sequential approach.  
- **Supporting Files**: All scripts required for model training and prediction. Ensure all files are placed in the same directory.

If any files cannot be located, adjust the `sys.path` at the beginning of the `main.py` or `main_tcn.py` files accordingly.

## Running the Models

### XGBoost
To run the XGBoost model:  
1. Execute `main.py`.  
2. Note: Joblib 1.4.2 may be required to load old XGBoost models.

### TCN
To run the TCN model:  
1. Update the variable imports in the supporting files to use `variables_tcn.py` instead of `variables.py`.  
2. Execute `main_tcn.py`.

## Variable Files

- **`variables.py`**: Key variables for models except TCN.  
- **`variables_tcn.py`**: Key variables for TCN-specific runs.  
Both files include detailed descriptions of their variables for quick reference.

