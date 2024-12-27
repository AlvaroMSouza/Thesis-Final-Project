import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors


# Import necessary files
data_path = "/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/hp_s1s2t2_17_18.csv"
data = pd.read_csv(data_path)
X_test = pd.read_csv("/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/dataset_split_v2/x_test_v2.csv")
y_test = pd.read_csv("/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/dataset_split_v2/y_test_v2.csv")

knn_predictions = np.loadtxt("/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/models_predictions_v2/knn_predictions_v2.csv", delimiter=',', skiprows=1)  
rf_predictions = np.loadtxt("/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/models_predictions_v2/rf_predictions_v2.csv", delimiter=',', skiprows=1)
xgb_predictions = np.loadtxt("/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/models_predictions_v2/xgb_predictions_v2.csv", delimiter=',', skiprows=1)
#matrix_df = pd.read_csv('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/expanded_final_matrix_v2.csv')

class_labels = np.unique(y_test)
models = ['KNN', 'Random Forest', 'XGBoost']
model_predictions = [knn_predictions, rf_predictions, xgb_predictions]
class_names = ['0', '1', '2', '3', '4']
true_labels = y_test['cos_class'].values.flatten()
feature_names = X_test.columns

knn_color_map = mcolors.LinearSegmentedColormap.from_list('knn_color_map', ['#cfe2f3', '#3182bd'])
rf_color_map = mcolors.LinearSegmentedColormap.from_list('rf_color_map', ['#fddbc7', '#e6550d'])
xgb_color_map = mcolors.LinearSegmentedColormap.from_list('xgb_color_map', ['#c7e9c0', '#31a354'])
color_maps = [knn_color_map, rf_color_map, xgb_color_map]
#matrix_df = pd.read_csv('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Almada_DT/final_matrix.csv')


# Create Confusion matrices & Missclassification - DONE RIGHT!

colors = [cmap(0.8) for cmap in color_maps]  # Get the lighter colors from the colormaps

# Create Confusion matrices
confusion_matrices = []
normalized_cm_final = []

for predictions in model_predictions:
    cm = confusion_matrix(true_labels, predictions.astype(int))
    confusion_matrices.append(cm)

# Plot confusion matrices and bar charts
num_models = len(models)
fig, axes = plt.subplots(2, num_models, figsize=(15, 10))

# Plot confusion matrices
# Plot confusion matrices
for i, ax in enumerate(axes[0]):
    cm = confusion_matrices[i]

    # Normalize each row of the confusion matrix
    normalized_cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    normalized_cm_final.append(normalized_cm)

    cmap = color_maps[i]  # Get the corresponding color map for the model
    sns.heatmap(normalized_cm, annot=True, cmap=cmap, fmt='.2f', ax=ax, vmin=0, vmax=1)

    ax.set_title(f'Confusion Matrix - {models[i]}')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')


# Plot bar charts
for i, ax in enumerate(axes[1]):
    misclassifications = np.where(model_predictions[i] != true_labels)[0]
    class_counts = np.bincount(true_labels[misclassifications])

    ax.bar(range(len(class_counts)), class_counts, color=colors[i])
    ax.set_xlabel('Class')
    ax.set_ylabel('Misclassifications')
    ax.set_title(f'{models[i]} Misclassifications')
    ax.set_xticks(range(len(class_counts)))

    for j, count in enumerate(class_counts):
        ax.text(j, count, str(count), ha='center', va='bottom')

plt.tight_layout()
#plt.savefig('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Plots/Confusion_Matrix_Missclassification.png')
plt.show()

"""

# Subclass Study for True Class x

threshold = 0.02
class_to_subclass_mapping = {
    0: [14, 15, 16, 17, 18, 20],
    1: [11, 12, 13],
    2: [6, 7, 8, 10, 21],
    3: [0, 1, 2, 3, 5, 9],
    4: [4]
}

for i, (model, predictions) in enumerate(zip(models, model_predictions)):
    cm = confusion_matrix(y_test, predictions.astype(int))
    normalized_cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    rows, cols = np.where((normalized_cm >= threshold) & (np.eye(normalized_cm.shape[0]) != 1))

    if len(rows) == 0:
        continue  # Skip if no misclassifications above threshold

    for j, true_label in enumerate(np.unique(rows)):
        col = cols[np.where(rows == true_label)][0]
        subclasses = class_to_subclass_mapping[true_label]
        subclasses = subclasses[::-1]
        new_matrix = np.zeros((len(subclasses), normalized_cm.shape[1]))

        for col in range(normalized_cm.shape[1]):
            for k, subclass in enumerate(subclasses):
                count = np.sum((matrix_df['Ground Truth'] == true_label) & (matrix_df['KNN Prediction'] == col) & (matrix_df['Sub-Class Prediction'] == subclass))
                new_matrix[k, col] = count
        
        subclass_mapping = {subclass_id: str(subclass_id) for subclass_id in subclasses}

        plt.figure(figsize=(10, 6))
        sns.heatmap(new_matrix, annot=True, cmap=color_maps[i], fmt='g', cbar_kws={'label': 'Count'})
        row_labels = [f'{subclass: <4}' for subclass in subclasses]
        plt.yticks(np.arange(len(subclasses)) + 0.5, row_labels, va='center')

        for k in range(len(subclasses)):
            for l in range(normalized_cm.shape[1]):
                if new_matrix[k, l] > 0:
                    count = int(new_matrix[k, l])
                    count_color = 'white' if count > 0.5 * new_matrix.max() else 'black'
                    plt.text(l + 0.5, k + 0.5, count, ha='center', va='center', color=count_color)

        plt.xlabel('Class')
        plt.ylabel('Subclass')
        plt.title(f'Subclass Study for True Class {true_label} - {model} Algorithm')
        plt.savefig('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Plots/subclass_study_true_class_{}_{}.png'.format(true_label, model))
        plt.show()



"""


"""
# Distributed error - DONE TENHO QUE MUDAR ISTO

# Calculate the error for each model
errors = []
for predictions in model_predictions:
    error = predictions - true_labels
    errors.append(error)


# Plot the error distribution using KDE plot
plt.figure(figsize=(12, 8))  # Adjust the figure size as desired
for i, error in enumerate(errors):
    sns.kdeplot(error, label=models[i], bw_method='scott', linestyle='-', linewidth=2)
    sns.rugplot(error, color=sns.color_palette()[i], alpha=0.2)

plt.xlabel('Error')
plt.ylabel('Density')
plt.title('Error Distribution (KDE Plot)')
plt.legend(loc='upper right')  # Adjust the legend position if needed
plt.grid(True)  # Add grid lines
plt.savefig('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Plots/Distributed_error.png')
plt.show()





# Class-wise Accuracy

def calculate_class_accuracies(predictions, labels, num_classes):
    accuracies = []

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        class_predictions = predictions[class_indices]
        class_labels = labels[class_indices]
        accuracy = np.mean(class_predictions == class_labels)
        accuracies.append(accuracy)

    return accuracies


# Calculate class-wise accuracies for each model
class_accuracies = []
for predictions in model_predictions:
    accuracies = calculate_class_accuracies(predictions, true_labels, num_classes=5)
    class_accuracies.append(accuracies)

# Plot Class-wise Accuracy with Confidence Intervals
plt.figure(figsize=(12, 8))

# Define color palette for classes
color_palette = plt.get_cmap('tab10')

# Plot accuracy lines with confidence intervals
for i, accuracies in enumerate(class_accuracies):
    model_name = models[i]
    model_color = color_palette(i)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    plt.plot(range(len(accuracies)), accuracies, marker='o', color=model_color, label=model_name)
    plt.fill_between(range(len(accuracies)), accuracies - std_accuracy, accuracies + std_accuracy, alpha=0.2, color=model_color)

# Customize plot
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy')
plt.xticks(range(5))
plt.legend()
plt.grid(True)
plt.savefig('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Plots/Classwise_accuracy.png')
plt.show()


# Calculate Subclass-wise accuracies for each model

# Calculate the misclassifications for each subclass
subclass_misclassifications = {}
for model, predictions in zip(models, model_predictions):
    misclassifications = np.where(predictions != true_labels)[0]
    subclass_labels = matrix_df.iloc[misclassifications]['Specific Class']
    subclass_counts = subclass_labels.value_counts()
    subclass_misclassifications[model] = subclass_counts

# Plot the misclassifications for each subclass
plt.figure(figsize=(12, 6))
for model, misclassifications in subclass_misclassifications.items():
    cmap = color_maps[models.index(model)]
    colors = [cmap(i) for i in np.linspace(0, 1, len(misclassifications))]
    plt.bar(misclassifications.index, misclassifications.values, color=colors, label=model)

plt.xlabel('Subclass')
plt.ylabel('Misclassifications')
plt.title('Misclassifications for each Subclass')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/alvarosouza/Desktop/FCT/5ºAno/Dissertação/Plots/SubClasswise_accuracy.png')
plt.show()



"""


# FORGET THIS: Feature Importance - CORRER esta parte no AZURE demora no meu computador
"""
def calculate_feature_importance(predictions, y_test):

    # Get the number of features
    num_features = 24177

    # Initialize the importance scores array
    importance_scores = np.zeros(num_features)

    # Calculate the baseline accuracy
    baseline_accuracy = accuracy_score(y_test, predictions)

    # Iterate over each feature
    for feature in range(num_features):
        # Shuffle the values of the current feature
        shuffled_predictions = np.random.permutation(predictions)

        # Calculate the accuracy with the shuffled feature
        shuffled_accuracy = accuracy_score(y_test, shuffled_predictions)

        # Compute the feature importance as the difference between baseline accuracy and shuffled accuracy
        feature_importance = baseline_accuracy - shuffled_accuracy

        # Store the feature importance score
        importance_scores[feature] = feature_importance

    return importance_scores



model_importances = []

# Calculate feature importance for each model
for predictions in model_predictions:
    importances = calculate_feature_importance(predictions, y_test)
    model_importances.append(importances)

# Sort indices based on feature importance
sorted_indices = [np.argsort(importances)[::-1] for importances in model_importances]

# Select top N features to plot
top_n = 10

# Plot feature importance for each model
colors = ['b', 'g', 'r']
for i, indices in enumerate(sorted_indices):
    model_name = models[i]
    importance_scores = model_importances[i]

    plt.figure(figsize=(8, 6))
    plt.bar(range(top_n), importance_scores[indices][:top_n], color=colors[i], align='center')
    plt.xticks(range(top_n), feature_names[indices][:top_n], rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plt.show()

"""




    
    