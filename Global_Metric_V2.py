import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score, specificity_score, sensitivity_score

data = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\expanded_final_matrix_v2.csv").fillna(0)

knn_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\knn_predictions_v2.csv", delimiter=',', skiprows=1)  
rf_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\rf_predictions_v2.csv", delimiter=',', skiprows=1)
xgb_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\xgb_predictions_v2.csv", delimiter=',', skiprows=1)
svm_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\svm_predictions_v2.csv", delimiter=',', skiprows=1)
nn_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\nn_predictions.csv", delimiter=',', skiprows=1)



# Create a list of classifier names and their corresponding predictions
classifiers = ['KNN', 'RF', 'XGBoost', 'SVM', 'Neural Network']
classifier_predictions = [knn_predictions, rf_predictions, xgb_predictions, svm_predictions, nn_predictions]

# Create an empty list to store the results
results = []

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
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall_sensitivity = sensitivity_score(y_true, y_pred, average='weighted')
    specificity = specificity_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    g_mean = geometric_mean_score(y_true, y_pred, average='weighted')
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Append the metrics to the results list as a dictionary
    results.append({
        'Classifier': classifier,
        'Overall_Accuracy': overall_accuracy,
        'Balanced_Accuracy': accuracy,
        'Precision': precision,
        'Recall_Sensitivity': recall_sensitivity,
        'Specificity': specificity,
        'F1-Score': f1,
        'G-Mean': g_mean,
        'Cohen Kappa': cohen_kappa,
        'MCC': mcc
    })

# Create a DataFrame from the list of results
result_df = pd.DataFrame(results)

print(result_df)

# Save the DataFrame to a CSV file
result_df.to_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\global_metricsV2.csv", index=False)




