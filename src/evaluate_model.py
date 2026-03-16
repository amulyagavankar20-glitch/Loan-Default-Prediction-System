from data_preprocessing import loan_preprocessing
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns


# Load models
rf_model = joblib.load("../models/random_forest_model.pkl")
lr_model = joblib.load("../models/logistic_regression_model.pkl")

# Load scaler for logistic regression
scaler = joblib.load("../models/scaler.pkl")


# Load data
df = pd.read_csv("../data/cs-training.csv")

# Preprocess
df = loan_preprocessing(df)

# Separate features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data for logistic regression
X_test_scaled = scaler.transform(X_test)

# RANDOM FOREST EVALUATION
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_report = classification_report(y_test, rf_pred, output_dict=True)
rf_cm = confusion_matrix(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

# LOGISTIC REGRESSION EVALUATION
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_report = classification_report(y_test, lr_pred, output_dict=True)
lr_cm = confusion_matrix(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

# PRINT METRICS
print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_pred))

print("\nLogistic Regression Classification Report")
print(classification_report(y_test, lr_pred))


print("\nRandom Forest ROC-AUC:", rf_auc)
print("Logistic Regression ROC-AUC:", lr_auc)

# COMPARISON PLOT (F1 SCORE)
rf_f1 = rf_report['1']['f1-score']
lr_f1 = lr_report['1']['f1-score']

models = ["Random Forest", "Logistic Regression"]
f1_scores = [rf_f1, lr_f1]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=f1_scores)
plt.title("Model Comparison (F1 Score)")
plt.ylabel("F1 Score")
plt.show()

# ROC-AUC COMPARISON
auc_scores = [rf_auc, lr_auc]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=auc_scores)
plt.title("Model Comparison (ROC-AUC)")
plt.ylabel("ROC-AUC")
plt.show()


# CONFUSION MATRIX PLOTS
fig, axes = plt.subplots(1,2, figsize=(10,4))

sns.heatmap(rf_cm, annot=True, fmt="d", ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")

sns.heatmap(lr_cm, annot=True, fmt="d", ax=axes[1])
axes[1].set_title("Logistic Regression Confusion Matrix")

plt.show()


# FEATURE IMPORTANCE (RF)
importance = rf_model.feature_importances_

plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=X.columns)
plt.title("Random Forest Feature Importance")
plt.show()