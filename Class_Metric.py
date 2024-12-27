#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:12:27 2023

@author: alvarosouza
"""

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, average_precision_score, roc_curve

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


data = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\expanded_final_matrix_v2.csv").fillna(0)


# Create a list of classifier names
classifiers = ['KNN', 'RF', 'XGBoost', 'SVM', 'Neural Network']

# Define a list of class labels
class_labels = [0, 1, 2, 3, 4] 



# Initialize a dictionary to store confusion matrices for each classifier
confusion_matrices = {}

# Iterate over each classifier
for classifier in classifiers:
    # Filter the data for the current classifier
    classifier_data = data[data['Classifier'] == classifier]

    # Initialize a confusion matrix for the current classifier
    confusion_matrix_classifier = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # Iterate over the data for the current classifier
    for _, row in classifier_data.iterrows():
        true_class = int(row['GT_Class'])
        predicted_class = int(row['Pred_Class'])
        # Update the confusion matrix
        confusion_matrix_classifier[true_class, predicted_class] += 1

    # Store the confusion matrix in the dictionary
    confusion_matrices[classifier] = confusion_matrix_classifier



# Initialize a list to store DataFrames for each class
class_metrics = []



knn_predict_prob_df = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Metrics\knn_predict_prob_v2.csv")
rf_predict_prob_df = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Metrics\rf_predict_prob_v2.csv")
xgb_predict_prob_df = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Metrics\xgb_predict_prob_v2.csv")
svm_predict_prob_df = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Metrics\svm_predict_prob_v2.csv")
nn_predict_prob_df = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Metrics\nn_predict_prob_v2.csv")


classifier_predictions = [knn_predict_prob_df, rf_predict_prob_df, xgb_predict_prob_df, svm_predict_prob_df, nn_predict_prob_df]

# Iterate over each classifier
for classifier, class_pred in zip(classifiers, classifier_predictions):
    
    confusion_matrix = confusion_matrices[classifier]
    
    # Iterate over each class
    for class_label in class_labels:
            
        # Filter data for the specific class
        class_data = classifier_data[classifier_data['GT_Class'] == class_label]

        # True ground truth values for the class
        y_true = class_data['GT_Class']
        # Predicted values by the classifier for the class
        y_pred = class_data['Pred_Class']
        
        
        accuracy = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for the class
        TP = confusion_matrix[class_label, class_label]
        FP = np.sum(confusion_matrix[:, class_label]) - TP
        FN = np.sum(confusion_matrix[class_label, :]) - TP
        TN = np.sum(confusion_matrix) - (TP + FP + FN)

        # Compute metrics for this class
        UA = TP / (TP + FN)  # User Accuracy
        PA = TP / (TP + FP)  # Producer Accuracy
        TPR = TP / (TP + FN)  # Recall, Sensitivity
        TNR = TN / (TN + FP)  # Specificity or true negative rate
        PPV = TP / (TP + FP)  # Precision or positive predictive value
        NPV = TN / (TN + FN)  # Negative predictive value
        FPR = FP / (FP + TN)  # False Positive Rate
        FNR = FN / (TP + FN)  # False Negative Rate
        FM_aux = PPV * TPR
        FM = np.sqrt(FM_aux)
        mcc_aux = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        MCC = (TP * TN - FP * FN) / mcc_aux
        quantity_disagreement = np.sum(confusion_matrix[class_label, :]) - np.sum(confusion_matrix[:, class_label])
        allocation_disagreement = np.sum(confusion_matrix) - np.sum(np.diag(confusion_matrix))
        auc_roc = class_pred[(class_pred['Class'] == class_label) & (class_pred['Metric Name'] == 'AUC-ROC Score')]['Metric Value'].iloc[0]
        pr = class_pred[(class_pred['Class'] == class_label) & (class_pred['Metric Name'] == 'PR AUC Score')]['Metric Value'].iloc[0]
        

        # Append the metrics to the list
        class_metrics.extend([
            [classifier, class_label, 'Balanced Accuracy', accuracy],
            [classifier, class_label, 'User Accuracy', UA],
            [classifier, class_label, 'Producer Accuracy', PA],
            [classifier, class_label, 'Precision (PPV)', PPV],
            [classifier, class_label, 'Recall (Sensitivity)', TPR],
            [classifier, class_label, 'Specificity (TNR)', TNR],
            [classifier, class_label, 'F1-Score', f1],
            [classifier, class_label, 'Negative Predictive Value (NPV)', NPV],
            [classifier, class_label, 'False Positive Rate (FPR)', FPR],
            [classifier, class_label, 'False Negative Rate (FNR)', FNR],
            [classifier, class_label, 'Fowlkes Mallows Index (FM)', FM],
            [classifier, class_label, 'Matthews Correlation Coefficient (MCC)', MCC],
            [classifier, class_label, 'Quantity Disagreement', quantity_disagreement],
            [classifier, class_label, 'Allocation Disagreement', allocation_disagreement],
            [classifier, class_label, 'AUC-ROC Score', auc_roc],
            [classifier, class_label, 'PR AUC Score', pr]
        ])

# Create a DataFrame from the class-level metrics
class_dataframes = pd.DataFrame(class_metrics, columns=['Classifier', 'Class', 'Metric Name', 'Metric Value'])
 
# Now 'results_df' contains the DataFrame with metrics for all classifiers and classes
print(class_dataframes)

# Save the DataFrame to a CSV file
class_dataframes.to_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\all_metric_class_level.csv", index=False)