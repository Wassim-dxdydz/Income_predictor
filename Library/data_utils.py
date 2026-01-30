"""
This module provides utility functions for loading, inspecting,
and preparing socio-economic data for analysis and modeling.
"""

import pandas as pd

def load_and_clean_data(csv_path):
    """
    Load data from the CSV file, replace missing values marked as '?', and clean the dataset.

    Args:
        csv_path (str): Path to the adult.csv file.

    Returns:
        DataFrame: Cleaned dataset.
    """
    expected_columns = [
        "age", "workclass", "fnlwgt", "education", "educational-num",
        "marital-status", "occupation", "relationship", "race", "gender",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df = pd.read_csv(csv_path)

    if not all(col in df.columns for col in expected_columns):
        raise ValueError("CSV does not contain the required columns.")

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().replace('?', 'Unknown')

    return df

def inspect_data(df):
    """
    Print basic info about the dataset: shape, missing values, and basic stats.

    Args:
        df (DataFrame): The dataset to inspect.
    """
    print("\n🧾 Dataset Info:")
    print(df.info())

    print("\n📏 Shape:", df.shape)

    missing_count = df.isna().sum().sum()
    print("\n❓ Missing (NaN) values:", missing_count)

    question_marks = (df == "?").sum().sum()
    print("❓ Entries with '?':", question_marks)

    print("\n📊 Basic Description:")
    print(df.describe())

def split_features_target(df):
    """
    Split the dataset into features (features_df) and target (target_series).

    Returns:
        features_df (DataFrame): Input features
        target_series (Series): Binary target variable
    """
    features_df = df.drop("income", axis=1)
    target_series = df["income"].apply(lambda x: 1 if str(x).strip() == ">50K" else 0)
    return features_df, target_series
