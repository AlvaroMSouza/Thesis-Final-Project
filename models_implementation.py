
import numpy as np
import pandas as pd

# Import necessary files
data = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\Almada_DT\hp_s1s2t2_17_18.csv").fillna(0)

x_test = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Dataset Split\x_test_v2.csv").fillna(0)
y_test = pd.read_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Dataset Split\y_test_v2.csv").fillna(0)

indices = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Dataset Split\indices_v2.csv", delimiter=',', skiprows=1)
knn_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\knn_predictions_v2.csv", delimiter=',', skiprows=1)  
rf_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\rf_predictions_v2.csv", delimiter=',', skiprows=1)
xgb_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\xgb_predictions_v2.csv", delimiter=',', skiprows=1)
svm_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\svm_predictions_v2.csv", delimiter=',', skiprows=1)
nn_predictions = np.loadtxt(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\Predictions\nn_predictions.csv", delimiter=',', skiprows=1)

# Create an empty list to store the new rows for the expanded table
expanded_rows = []

classifiers = ['KNN', 'RF', 'XGBoost', 'SVM', 'Neural Network']

# Iterate over each pixel in the original matrix
for i, index in enumerate(indices):
    x = data.loc[index, 'x']
    y = data.loc[index, 'y']
    gt_class = y_test.iloc[i].item()
    gt_subclass = data.loc[index, 'class']
    pred_classes = [knn_predictions[i], rf_predictions[i], xgb_predictions[i], svm_predictions[i], nn_predictions[i]]

    # Iterate over each classifier and its corresponding prediction
    for pred_class, classifier in zip(pred_classes, classifiers):
        # Determine the error classification (0 if prediction matches ground truth, 1 otherwise)
        error_classification = 0 if pred_class == gt_class else 1

        # Create a new row with the updated column values
        expanded_rows.append([i, y, x, gt_class, gt_subclass, pred_class, classifier, error_classification])

# Convert the expanded rows list to a DataFrame
expanded_matrix_df = pd.DataFrame(expanded_rows, columns=['ID', 'Latitude', 'Longitude', 'GT_Class', 'GT_SubClass', 'Pred_Class', 'Classifier', 'Error Classification'])

# Save the expanded DataFrame to a new CSV file
expanded_matrix_df.to_csv(r"C:\Users\alvar\OneDrive\Ambiente de Trabalho\FCT\5 Ano\Thesis\New Models Almada Dataset\expanded_final_matrix_v2.csv", index=False)

# Print the expanded matrix DataFrame
print(expanded_matrix_df)
