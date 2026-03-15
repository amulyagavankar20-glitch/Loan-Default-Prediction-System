# Project Workflow and Implementation

## 1. Data Collection

The dataset used in this project is the **Give Me Some Credit** dataset, obtained from Kaggle.

This dataset contains financial and behavioral information about borrowers and is commonly used for credit risk modeling. The objective is to predict whether a borrower will experience serious delinquency within two years.

### Target Variable

**SeriousDlqin2yrs**

- `0` → No serious delinquency
- `1` → Serious delinquency (default risk)

### Example Features

- **age** – age of the borrower
- **MonthlyIncome** – borrower's monthly income
- **DebtRatio** – ratio of monthly debt payments to income
- **RevolvingUtilizationOfUnsecuredLines** – credit utilization ratio
- **NumberOfOpenCreditLinesAndLoans** – total open credit accounts
- **NumberOfDependents** – number of dependents supported by the borrower

### Data Split

- **Training data** → used to train machine learning models
- **Testing data** → used to evaluate model performance

---

## 2. Project File Structure

The project is organized into multiple folders to maintain a clean and modular machine learning pipeline.

### Example Structure

```
loan-default-early-warning/
│
├── data/
│   ├── cs-training.csv
│   └── cs-test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_data_exploration.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   └── random_forest_model.pkl
│
└── README.md
```

Each file in this structure serves a specific purpose.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset before building machine learning models. EDA is conducted using Jupyter notebooks.

### Notebook 1: `notebooks/01_data_exploration.ipynb`

This notebook analyzes the training dataset.

**Steps performed:**

- Dataset inspection (`df.info()`)
- Statistical summary (`df.describe()`)
- Missing value analysis
- Feature distribution visualization
- Outlier detection using boxplots
- Feature correlation analysis using heatmaps
- Pairwise feature relationships using pairplots

### Notebook 2: `notebooks/02_data_exploration.ipynb`

This notebook performs similar analysis on the test dataset to ensure that its structure and distribution are consistent with the training dataset.

---

## 4. Libraries Used in the Project

Several Python libraries are used for data processing, visualization, and machine learning.

### Pandas

Library used for data manipulation and analysis.

**Used for:**

- Reading datasets
- Handling missing values
- Filtering and transforming data

**Example:**

```python
import pandas as pd
df = pd.read_csv("data/cs-training.csv")
```

### NumPy

Used for numerical computations and array operations.

**Example:**

```python
import numpy as np
```

NumPy is often used internally by machine learning libraries.

### Matplotlib

Used for basic data visualization.

**Example plots:**

- Histograms
- Scatter plots
- Line charts

**Example:**

```python
import matplotlib.pyplot as plt
```

### Seaborn

Built on top of Matplotlib and used for advanced statistical visualization.

**Used for:**

- Correlation heatmaps
- Boxplots
- Pairplots
- Countplots

**Example:**

```python
import seaborn as sns
```

### Scikit-learn

The primary machine learning library used in the project.

**Modules used include:**

- `sklearn.model_selection`
- `sklearn.linear_model`
- `sklearn.ensemble`
- `sklearn.metrics`

**These modules help with:**

- Train-test splitting
- Building models
- Evaluating predictions

### Imbalanced-Learn

Used to handle class imbalance problems.

**The project uses:**

- **SMOTE** (Synthetic Minority Over-sampling Technique) – which generates synthetic examples of the minority class

---

## 5. Data Preprocessing

Data preprocessing is handled in the file: `src/data_preprocessing.py`

This file contains the function: `loan_preprocessing()`

This function performs all data cleaning steps before model training.

### Step 1 — Handling Missing Values

The dataset contains missing values in:

- `MonthlyIncome`
- `NumberOfDependents`

These values are filled using **median imputation** because the dataset contains large outliers and skewed distributions. Median is less affected by extreme values compared to the mean.

### Step 2 — Fixing Invalid Delinquency Values

Some delinquency columns contain unrealistic values such as:

- `96`
- `98`

These represent the number of late payments in the past two years and are considered data errors. These values are replaced with the median of the column.

**Affected columns include:**

- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfTimes90DaysLate`

### Step 3 — Outlier Treatment

Financial datasets often contain extreme outliers.

Outliers were detected in:

- `MonthlyIncome`
- `DebtRatio`
- `RevolvingUtilizationOfUnsecuredLines`

These were treated using **percentile capping**.

The values are limited between:

- 1st percentile
- 99th percentile

This reduces the impact of extreme values without removing important observations.

### Step 4 — Removing Invalid Data

Rows where `age = 0` are removed because they represent invalid records.

---

## 6. Train-Test Split

The dataset is divided into training and testing subsets using the Scikit-learn function: `train_test_split()`

**Typical split:**

- 80% training data
- 20% testing data

This ensures that the model is evaluated on unseen data.

---

## 7. Handling Class Imbalance

The dataset is highly imbalanced.

**Typical distribution:**

- Non-Default → ~93%
- Default → ~7%

This imbalance can cause models to favor the majority class.

To solve this problem, **SMOTE** is applied to the training data only.

SMOTE creates synthetic examples of the minority class by interpolating between existing minority samples. This results in a more balanced dataset and improves the model's ability to detect default cases.

---

## 8. Model Training

Model training is performed in: `src/train_model.py`

Two machine learning algorithms are used.

### Logistic Regression

A statistical classification model used for binary classification problems. It estimates the probability that a borrower will default.

**Advantages:**

- Simple
- Interpretable
- Fast to train

### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees.

**Advantages:**

- Handles nonlinear relationships
- Robust to noise
- Good performance on tabular datasets

The model is trained on the balanced dataset produced by SMOTE.

---

## 9. Model Evaluation

Model evaluation is performed in: `src/evaluate_model.py`

The trained model is loaded and evaluated on the test dataset.

**Several metrics are used:**

### Accuracy

Measures the percentage of correct predictions.

### Precision

Measures how many predicted defaults are actually correct.

### Recall

Measures how many actual defaults were correctly detected.

### F1 Score

The harmonic mean of precision and recall. This metric is important when dealing with imbalanced datasets.

### Confusion Matrix

Shows the number of:

- True Positives
- True Negatives
- False Positives
- False Negatives

This helps analyze model performance in more detail.

---

## Final Workflow Summary

1. Collect dataset from Kaggle (Give Me Some Credit)
2. Perform EDA on train and test datasets using notebooks
3. Handle missing values using median imputation
4. Clean invalid delinquency values
5. Treat extreme outliers using percentile capping
6. Remove invalid records (age = 0)
7. Split dataset into training and testing sets
8. Handle class imbalance using SMOTE on training data
9. Train Logistic Regression and Random Forest models
10. Evaluate models using classification metrics
