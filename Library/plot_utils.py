"""
This module contains functions to generate various plots
for demographic and income-related features using seaborn and matplotlib.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import pandas as pd


sns.set(style="whitegrid")

def _graphics_output_dir(base_dir):
    """Create and return the path to the graphics output directory."""
    path = os.path.join(base_dir, 'Graphics')
    os.makedirs(path, exist_ok=True)
    return path

def plot_age_distribution(df, base_dir):
    """Generate and save a histogram showing the age distribution."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['age'], bins=30, kde=False)
    plt.title('Age Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.close()
    print("✅ age_distribution.png saved")

def plot_hours_per_week_distribution(df, base_dir):
    """Plot and save the distribution of weekly working hours."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['hours-per-week'], bins=30, kde=False)
    plt.title('Hours Per Week Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hours_per_week_distribution.png'))
    plt.close()
    print("✅ hours_per_week_distribution.png saved")

def plot_education_level_pie(df, base_dir):
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    edu_counts = df['education'].value_counts()
    plt.pie(edu_counts, labels=edu_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Education Level Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'education_level_pie.png'))
    plt.close()
    print("✅ education_level_pie.png saved")

def plot_workclass_distribution(df, base_dir):
    """Generate a bar chart of workclass distribution."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    sns.countplot(y='workclass', data=df, order=df['workclass'].value_counts().index)
    plt.title('Workclass Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'workclass_distribution.png'))
    plt.close()
    print("✅ workclass_distribution.png saved")

def plot_income_distribution(df, base_dir):
    """Plot the distribution of income categories."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(6, 6))
    sns.countplot(x='income', data=df)
    plt.title('Income Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_distribution.png'))
    plt.close()
    print("✅ income_distribution.png saved")

def plot_age_vs_income(df, base_dir):
    """Create a box plot comparing age across income groups."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='income', y='age', data=df)
    plt.title('Age vs. Income')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_vs_income.png'))
    plt.close()
    print("✅ age_vs_income.png saved")

def plot_marital_status_distribution(df, base_dir):
    """Plot the distribution of marital status categories."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    sns.countplot(y='marital-status', data=df, order=df['marital-status'].value_counts().index)
    plt.title('Marital Status Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marital_status_distribution.png'))
    plt.close()
    print("✅ marital_status_distribution.png saved")

def plot_occupation_treemap(df, base_dir):
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(10, 6))
    occ_counts = df['occupation'].value_counts()
    squarify.plot(sizes=occ_counts.values, label=occ_counts.index, alpha=0.8)
    plt.title('Occupation Treemap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'occupation_treemap.png'))
    plt.close()
    print("✅ occupation_treemap.png saved")

def plot_race_pie(df, base_dir):
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(6, 6))
    race_counts = df['race'].value_counts()
    plt.pie(race_counts, labels=race_counts.index, autopct='%1.1f%%')
    plt.title('Race Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'race_pie.png'))
    plt.close()
    print("✅ race_pie.png saved")

def plot_gender_distribution(df, base_dir):
    """Create a bar chart comparing gender counts."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(6, 6))
    sns.countplot(x='gender', data=df)
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    plt.close()
    print("✅ gender_distribution.png saved")

def plot_education_vs_income(df, base_dir):
    """Compare education levels with income groups using a grouped bar chart."""
    output_dir = _graphics_output_dir(base_dir)
    plt.figure(figsize=(8, 6))
    sns.countplot(y='education', hue='income', data=df, order=df['education'].value_counts().index)
    plt.title('Education Level vs. Income')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'education_vs_income.png'))
    plt.close()
    print("✅ education_vs_income.png saved")

def plot_occupation_vs_income_heatmap(df, base_dir):
    output_dir = _graphics_output_dir(base_dir)
    cross_tab = pd.crosstab(df['occupation'], df['income'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Occupation vs Income Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'occupation_vs_income_heatmap.png'))
    plt.close()
    print("✅ occupation_vs_income_heatmap.png saved")

def plot_correlation_heatmap(df, base_dir):
    """Generate and save a heatmap of correlations among selected numeric features."""
    output_dir = _graphics_output_dir(base_dir)
    df['income_numerical'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    selected_columns = ['age', 'educational-num',
                        'capital-gain', 'capital-loss',
                        'hours-per-week', 'income_numerical']
    correlation_matrix = df[selected_columns].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Selected Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("✅ correlation_heatmap.png saved")
