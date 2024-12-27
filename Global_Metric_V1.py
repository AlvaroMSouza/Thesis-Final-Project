#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:43:52 2023

@author: alvarosouza
"""

import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score, specificity_score, sensitivity_score

# Assuming you have a DataFrame called 'data' with columns: 'GT_Class', 'Pred_Class', 'Classifier'
data = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\expanded_final_matrix_v2.csv").fillna(0)

knn_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\knn_predictions_v2.csv", delimiter=',', skiprows=1)  
rf_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\rf_predictions_v2.csv", delimiter=',', skiprows=1)
xgb_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\xgb_predictions_v2.csv", delimiter=',', skiprows=1)
svm_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\svm_predictions_v2.csv", delimiter=',', skiprows=1)
nn_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\nn_predictions.csv", delimiter=',', skiprows=1)


# Create a list of classifier names and their corresponding predictions
classifiers = ['KNN', 'RF', 'XGBoost', 'SVM', 'Neural Network']
classifier_predictions = [knn_predictions, rf_predictions, xgb_predictions, svm_predictions, nn_predictions]

# Create lists to store metric values
classifier_list = []
metric_name_list = []
metric_value_list = []

# Iterate over each classifier and compute the metrics
for classifier, predictions in zip(classifiers, classifier_predictions):
    
    # Filter the data DataFrame for the specific classifier
    classifier_data = data[data['Classifier'] == classifier]
    # True ground truth values from the dataset
    y_true = classifier_data['GT_Class']
    # Predicted values by the classifier
    y_pred = predictions

    # Calculate classification metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall_sensitivity = sensitivity_score(y_true, y_pred, average='weighted')
    specificity = specificity_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    g_mean = geometric_mean_score(y_true, y_pred, average='weighted')
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Store metrics in lists
    classifier_list.extend([classifier] * 9)  # Number of metrics per classifier
    metric_name_list.extend([ 'Overall_Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall_Sensitivity', 'Specificity', 'F1-Score', 'G-Mean', 'Cohen Kappa', 'MCC'])
    metric_value_list.extend([overall_accuracy, balanced_accuracy, precision, recall_sensitivity, specificity, f1, g_mean, cohen_kappa, mcc])

# Create a DataFrame from the lists
metrics_df = pd.DataFrame({'Classifier': classifier_list, 'Metric Name': metric_name_list, 'Metric Value': metric_value_list})

print(metrics_df)

# Save the DataFrame to a CSV file
metrics_df.to_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\global_metricsV1_v2.csv", index=False)
