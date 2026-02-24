import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations directory if it doesn't exist
os.makedirs('../visualizations', exist_ok=True)

def load_data():
    return pd.read_csv('../data/raw/loan_default_data.csv')

def plot_class_distribution(df):
    """
    Insight: Checks for class imbalance. If there are way more '0's (No Default) 
    than '1's (Default), the model might become biased.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='default.payment.next.month', data=df, palette='Set2')
    plt.title('Class Distribution (0: No Default, 1: Default)')
    plt.xlabel('Default Payment Next Month')
    plt.ylabel('Count')
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('../visualizations/class_distribution.png')
    plt.close()
    print("Saved class_distribution.png")

def plot_correlation_heatmap(df):
    """
    Insight: Shows which features are highly correlated. 
    High correlation with target is good. High correlation between two features (e.g., PAY_AMT1 and PAY_AMT2) 
    indicates multicollinearity, meaning one might be redundant.
    """
    plt.figure(figsize=(20, 16))
    # Calculate correlation matrix
    corr = df.corr()
    # Plot heatmap
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../visualizations/correlation_heatmap.png')
    plt.close()
    print("Saved correlation_heatmap.png")

def plot_boxplots(df):
    """
    Insight: Boxplots show the median, quartiles, and outliers. 
    Useful for identifying if defaulters have significantly different distributions 
    in numerical features like Limit Balance or Age.
    """
    features = ['LIMIT_BAL', 'AGE']
    
    plt.figure(figsize=(14, 6))
    for i, feature in enumerate(features, 1):
        plt.subplot(1, 2, i)
        sns.boxplot(x='default.payment.next.month', y=feature, data=df, palette='Set3')
        plt.title(f'Boxplot of {feature} by Default Status')
        plt.xlabel('Default (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('../visualizations/boxplots.png')
    plt.close()
    print("Saved boxplots.png")

def plot_default_rate_by_limit_bal(df):
    """
    Insight: LIMIT_BAL is a proxy for loan amount/income limit. 
    We bin the balances to see if people with lower or higher limits default more.
    """
    # Create bins for Limit Balance
    df['LIMIT_BAL_BINS'] = pd.qcut(df['LIMIT_BAL'], q=5)
    
    plt.figure(figsize=(10, 6))
    rate_df = df.groupby('LIMIT_BAL_BINS')['default.payment.next.month'].mean().reset_index()
    sns.barplot(x='LIMIT_BAL_BINS', y='default.payment.next.month', data=rate_df, palette='viridis')
    
    plt.title('Default Rate by Limit Balance (Loan Amount)')
    plt.xlabel('Limit Balance Bins')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../visualizations/default_rate_by_limit_bal.png')
    plt.close()
    print("Saved default_rate_by_limit_bal.png")
    # Clean up the temp column
    df.drop('LIMIT_BAL_BINS', axis=1, inplace=True)

if __name__ == "__main__":
    print("Running EDA Script...\n")
    df = load_data()
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_default_rate_by_limit_bal(df)
    print("\nEDA Visualizations successfully generated in 'visualizations/' folder.")
