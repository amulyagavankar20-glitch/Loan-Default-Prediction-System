import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the csv using pandas
df = pd.read_csv("C:\Code\loan-default-early-warning\data\cs-training.csv")

def loan_preprocessing(df):
    # 1 Fill missing values
    df['MonthlyIncome'] = np.log1p(df['MonthlyIncome'].fillna(df['MonthlyIncome'].median()))
    df['NumberOfDependents'] = np.log1p(df['NumberOfDependents'].fillna(df['NumberOfDependents'].median()))
    # 2 Fix delinquency errors
    delinq_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]

    for col in delinq_cols:
        df.loc[df[col] > 90, col] = df[col].median()

    # 3 Cap outliers
    outlier_cols = [
        'RevolvingUtilizationOfUnsecuredLines',
        'DebtRatio',
        'MonthlyIncome'
    ]

    for col in outlier_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)


    # 4 Remove invalid age
    df = df[df['age'] > 0]

    return df