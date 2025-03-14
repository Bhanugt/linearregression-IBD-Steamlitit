# -*- coding: utf-8 -*-
"""linear regression IBD Streamlit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a6bs2RiBFUwvwPsWq8TKIZDJK2xk6p7_
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

#  Load the actual dataset (Replace with your file path)
df = pd.read_csv("/content/sample_data/synthetic_ibd_dataset.csv")  # Change filename if needed

#  Encode categorical variables (if needed)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoders for later use

# ✅ Define Features (X) and Target (y)
target_col = "IBD_Type"  # Change if your dataset has a different target column
X = df.drop(columns=[target_col])
y = df[target_col]

#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#  Make Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

#  Evaluate Model Performance
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

train_precision = precision_score(y_train, y_pred_train, average='weighted')
test_precision = precision_score(y_test, y_pred_test, average='weighted')

train_recall = recall_score(y_train, y_pred_train, average='weighted')
test_recall = recall_score(y_test, y_pred_test, average='weighted')

train_f1 = f1_score(y_train, y_pred_train, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')

#  Print Metrics
print(" Model Performance Metrics:")
print(f" Train Accuracy: {train_accuracy:.4f}")
print(f" Test Accuracy: {test_accuracy:.4f}")
print(f" Train Precision: {train_precision:.4f}")
print(f" Test Precision: {test_precision:.4f}")
print(f" Train Recall: {train_recall:.4f}")
print(f" Test Recall: {test_recall:.4f}")
print(f" Train F1 Score: {train_f1:.4f}")
print(f" Test F1 Score: {test_f1:.4f}")

print("\n Classification Report on Test Data:")
print(classification_report(y_test, y_pred_test))

#  Save Model, Scaler, and Label Encoders
joblib.dump(model, "logistic_regression.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n Model, Scaler, and Encoders Saved Successfully!")

from google.colab import files

#  Download the trained model
files.download("linear_regression.pkl")

#  Download the scaler
files.download("scaler.pkl")

# Download the label encoders (if categorical variables were encoded)
files.download("label_encoders.pkl")

import joblib

#  Load the saved Linear Regression model
model = joblib.load("linear_regression.pkl")

print(" Model loaded successfully!")
print("Model details:", model)