"""
This module defines utility functions for building, training, evaluating,
and saving machine learning models for income classification.
"""

import os
from datetime import datetime

import joblib
from tabulate import tabulate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def build_model_pipeline(numerical_cols, categorical_cols):
    """
    Create a full preprocessing + classification pipeline using Random Forest.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return model

def train_model(model, x_train, y_train):
    """
    Train the model on the provided training data.
    """
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model and print performance metrics.
    """
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("\n✅ Accuracy Score:", acc)
    print("\n📊 Classification Report:\n", report)
    return acc, report

def save_model(model, path="income_model.pkl"):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, path)
    print(f"\n💾 Model saved as '{path}'")

def load_model(path="income_model.pkl"):
    """
    Load a model from a file.
    """
    return joblib.load(path)

def save_report(df, filename, base_dir, title=None, description=None):
    """
    Save a classification or metrics report from a DataFrame to a text file with optional metadata.

    Args:
        df (pd.DataFrame): The data to include in the report.
        filename (str): The name of the output file.
        base_dir (str): Base directory to store the report under Output/reports.
        title (str, optional): Title for the report.
        description (str, optional): Additional description to include.
    """
    output_dir = os.path.join(base_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"{title}\nGenerated on: {now}\n"
    if description:
        header += f"\n{description}\n"

    table = tabulate(df, headers='keys', tablefmt='github', showindex=False)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + table)

    print(f"✅ Report saved to {file_path}")
