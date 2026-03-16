from data_preprocessing import loan_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import joblib

# Load data
df = pd.read_csv("C:\Code\loan-default-early-warning\data\cs-training.csv")

# Preprocess
df = loan_preprocessing(df)

# Separate features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

print("Original Class Distribution:")
print(y.value_counts())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Resampled Class Distribution:")
print(pd.Series(y_train_res).value_counts())

# Feature scaling
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train_res, y_train_res)

# Save trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/logistic_regression_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
print("Model saved successfully.")