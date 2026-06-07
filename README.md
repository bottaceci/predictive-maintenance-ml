# Predictive Maintenance ML System

## Goal
The goal of this project is to build a small machine learning system that predicts whether a machine is likely to fail based on sensor data such as vibration, temperature, and wear.

The project is designed as a learning project focused on both machine learning and software engineering. The implementation will use a clean Python package structure, object-oriented components, reproducible experiments, and clear evaluation metrics.

## Data
We are going to use data from the **[AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)**. This dataset was chosen from the following reasons:
- It is explicitly a predictive maintenance dataset.
- It is small: 10,000 rows, so it is easy to inspect and debug.
- It has a binary target: Machine failure.
- It has industrial-style features: air temperature, process temperature, rotational speed, torque, and tool wear.
- It has no missing values.

The dataset variables are:
- UID
- Product ID
- Features
    - Type
    - Air temperature [K]
    - Process temperature [K]
    - Rotational speed [rpm]
    - Torque [Nm]
    - Tool wear [min]
- Targets
    - Machine failure
    - TWF
    - HDF
    - PWF
    - OSF
    - RNF

## Problem Type
This is a **binary classification** problem. 
The model receives time-windowed sensor measurements and predicts whether the machine is operating normally or approaching a failure condition.

The target labels are:
- `0`: normal operation
- `1`: failure or pre-failure condition

## Planned Approach
The planned approach is to build the project incrementally, by writing classes that will do the following:
1. Load raw sensor data.
2. Generate rolling-window features from the sensor signals.
3. Train baseline classification models such as Logistic Regression, Random Forest, and Gradient Boosting.
4. Evaluate the models using appropriate classification metrics, especially precision, recall, F1-score, and confusion matrix.
5. Compare model performance and document the results.

As a later extension, I plan to implement selected signal-processing feature calculations in Fortran, such as RMS, peak-to-peak amplitude, kurtosis, and moving averages, then expose them to Python using F2PY.

A neural network model may also be considered as a future extension after the classical machine learning pipeline is complete.


## Project Structure
The current project structure is:
```
.
├── README.md
├── data
│   ├── processed
│   └── raw
├── notebooks
├── requirements.txt
├── src
│   └── predictive_maintenance
│       ├── __init__.py
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── experiment.py
│       ├── feature_engineering.py
│       └── model.py
└── tests
```

## Current Status
The project has been initialized. The current focus is on defining the project structure and implementing the first version of the data loading and feature engineering pipeline.
