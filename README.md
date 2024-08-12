# Telco Customer Churn Prediction

## Overview
This project implements a machine learning pipeline to predict customer churn for a telecommunications company. It uses an ensemble of models, including Neural Networks, XGBoost, LightGBM, and CatBoost, to achieve high accuracy and robust predictions.

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Pipeline](#pipeline)
6. [Models](#models)
7. [Evaluation](#evaluation)
8. [Checkpointing](#checkpointing)

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- tensorflow
- xgboost
- lightgbm
- catboost
- joblib

## Installation
1. Clone this repository:https://github.com/AffanShaikhsurab/ChurnPrediction/
2. cd telco-churn-prediction
   
## Install the required packages:
1.pip install -r requirements.txt

## Run the main script:
1.python main.py

## Data
The script expects a CSV file named `telco_customer_churn.csv` in the `./` directory. Ensure this file is present before running the script.

## Pipeline
1. Data loading and preprocessing
2. Feature selection
3. Model training (Neural Network, XGBoost, LightGBM, CatBoost)
4. Model evaluation

## Models
- Neural Network: Custom architecture with Dense layers, BatchNormalization, and Dropout
- XGBoost: Default parameters with early stopping
- LightGBM: Default parameters with early stopping
- CatBoost: Default parameters with early stopping

## Evaluation
Models are evaluated using the following metrics:
- Accuracy
- AUC (Area Under the ROC Curve)
- Average Precision

## Checkpointing
The script uses checkpointing to save intermediate results, allowing for faster rerun and continuation from previous states. Checkpoints are saved in the `./checkpoints` directory.

## Contributing
Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License
[MIT License](LICENSE)
