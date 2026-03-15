from data_preprocessing import loan_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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

# Train model
rf = RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_split=4,min_samples_leaf=2,random_state=42,n_jobs=-1)

rf.fit(X_train_res, y_train_res)

# Save trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(rf, "../models/random_forest_model.pkl")

print("Model saved successfully.")
