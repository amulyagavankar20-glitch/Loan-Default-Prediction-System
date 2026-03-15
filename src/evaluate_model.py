from data_preprocessing import loan_preprocessing
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load model
model = joblib.load("../models/random_forest_model.pkl")

# Load data
df = pd.read_csv("C:\Code\loan-default-early-warning\data\cs-training.csv")

# Preprocess
df = loan_preprocessing(df)

# Separate features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

import matplotlib.pyplot as plt
import seaborn as sns

importance = model.feature_importances_

sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.show()