# Irrigation System Machine Learning Project

This repository contains a machine learning project focused on predicting irrigation needs for multiple land parcels based on sensor data. The solution leverages Python with popular libraries like pandas, scikit-learn, matplotlib, and seaborn to provide a comprehensive workflow from data ingestion to model deployment readiness.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results and Visualizations](#results-and-visualizations)
- [Model Deployment](#model-deployment)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The core of this project is a machine learning model designed to determine the optimal irrigation status for three different land parcels (`parcel_0`, `parcel_1`, `parcel_2`) using readings from 20 environmental sensors (`sensor_0` to `sensor_19`). The goal is to automate and optimize irrigation decisions based on real-time sensor data.

## Dataset

The project uses the `irrigation_machine.csv` dataset.
- **`irrigation_machine.csv`**: Contains sensor readings and the corresponding irrigation status for each parcel.
  - Features: `sensor_0` to `sensor_19` (numerical values)
  - Targets: `parcel_0`, `parcel_1`, `parcel_2` (categorical/binary indicating irrigation status)

## Project Structure

- `Irrigation_System.ipynb`: The main Jupyter Notebook containing the complete workflow.
- `irrigation_machine.csv`: The dataset used for training and evaluation.
- `multi_output_irrigation_model.joblib`: The trained machine learning model.
- `min_max_scaler.joblib`: The fitted MinMaxScaler used for data preprocessing.

## Key Features

- **Data Loading and Preprocessing**: Efficient loading of CSV data, handling of extraneous columns, and robust feature scaling using `MinMaxScaler`.
- **Comprehensive Data Visualization**:
    - Bar graphs visualizing the distribution of irrigation statuses for each parcel.
    - Histograms/KDE plots illustrating the distributions of individual sensor readings.
    - A detailed correlation heatmap showing relationships between all sensor features and parcel statuses.
- **Machine Learning Model**: Implementation of a `MultiOutputClassifier` with `RandomForestClassifier` as the base estimator to predict irrigation for multiple parcels simultaneously.
- **Model Evaluation**: Provides `classification_report` for each parcel output, detailing precision, recall, and f1-score.
- **Feature Importance Analysis**: Visualizations indicating the most important sensor features for predicting the irrigation status of each individual parcel.
- **Model Persistence**: Saves the trained `MultiOutputClassifier` and the `MinMaxScaler` using `joblib` for easy re-use and deployment.

## Results and Visualizations

The `Irrigation_System.ipynb` notebook generates various plots, including:
- Distributions of `parcel_0`, `parcel_1`, `parcel_2` states.
- Histograms showing the distribution of `sensor_0` to `sensor_19`.
- A correlation heatmap of all features and targets.
- Bar plots depicting the importance of each sensor feature for predicting `parcel_0`, `parcel_1`, and `parcel_2`.

These visualizations provide critical insights into the data characteristics and the model's predictive drivers.

## Model Deployment

The trained model (`multi_output_irrigation_model.joblib`) and the scaler (`min_max_scaler.joblib`) are saved, allowing for straightforward loading and prediction on new data without retraining:

```python
import joblib
import pandas as pd

# Load the model and scaler
loaded_model = joblib.load('multi_output_irrigation_model.joblib')
loaded_scaler = joblib.load('min_max_scaler.joblib')

# Example new data (replace with actual new sensor readings)
new_sensor_data = pd.DataFrame([
    # Example row: 20 sensor readings
    [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
], columns=[f'sensor_{i}' for i in range(20)])

# Scale the new data
new_sensor_data_scaled = loaded_scaler.transform(new_sensor_data)

# Make predictions
predictions = loaded_model.predict(new_sensor_data_scaled)
print(f"Predicted irrigation status for parcels: {predictions}")
