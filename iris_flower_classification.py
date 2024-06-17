# -*- coding: utf-8 -*-
"""IRIS FLOWER CLASSIFICATION"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 2: Explore the data
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 samples:\n", pd.DataFrame(X, columns=iris.feature_names).head())

# Step 3: Preprocess the data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 6: Make predictions
sample_data = [[5.1, 3.5, 1.4, 0.2],  # Expected: setosa
               [6.7, 3.0, 5.2, 2.3]]  # Expected: virginica

predictions = clf.predict(sample_data)
predicted_species = [iris.target_names[p] for p in predictions]
print("Predictions for sample data:", predicted_species)
