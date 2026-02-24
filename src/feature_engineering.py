import pandas as pd
import numpy as np

def load_processed_data(filepath):
    """Loads the preprocessed dataset."""
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Initial shape: {df.shape}")
    return df

def feature_engineering(df):
    """
    Creates domain-specific features based on the UCI Credit Card Dataset to 
    give machine learning models more predictive signals.
    """
    print("\nStarting Feature Engineering...")
    
    # 1. Credit Utilization Ratio (Average Bill / Limit Balance)
    # High utilization often indicates financial distress
    bill_columns = [col for col in df.columns if col.startswith('BILL_AMT')]
    # Calculate average bill amount across the 6 months
    df['AVG_BILL_AMT'] = df[bill_columns].mean(axis=1)
    
    # We must be careful about division by zero conceptually, though LIMIT_BAL was scaled.
    # To calculate utilization conceptually correctly, it's better to calculate it BEFORE scaling.
    # However, since the user already requested normalization in Phase 2, we will approximate 
    # interaction features using the scaled data, or create an interaction metric based on recent behavior.
    
    # Let's create an "Average Payment to Bill Ratio"
    pay_amt_columns = [col for col in df.columns if col.startswith('PAY_AMT')]
    df['AVG_PAY_AMT'] = df[pay_amt_columns].mean(axis=1)
    
    # If a person pays very little compared to their bill, it's a red flag.
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-6
    df['PAY_TO_BILL_RATIO'] = df['AVG_PAY_AMT'] / (df['AVG_BILL_AMT'].replace(0, epsilon).abs())
    
    # 2. Payment Delay Count
    # PAY_0 to PAY_6 are integer codes representing delay. Positive values mean delayed by X months.
    # We will engineer a feature that counts how many times the user was delayed by 2 or more months.
    # Note: PAY_0 is actually named 'PAY_0' in the raw dataset, representing repayment status in September.
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # Create a feature tracking the total number of severe delays
    df['TOTAL_SEVERE_DELAYS'] = 0
    for col in pay_status_cols:
        if col in df.columns:
            # Add 1 if the payment was delayed by 2 or more months
            df['TOTAL_SEVERE_DELAYS'] += (df[col] >= 2).astype(int)
            
    # 3. Overall Payment Trend (Is their financial situation getting worse?)
    # If PAY_0 (recent) > PAY_6 (past), their delays are increasing.
    if 'PAY_0' in df.columns and 'PAY_6' in df.columns:
        df['DELAY_TREND_WORSENING'] = (df['PAY_0'] > df['PAY_6']).astype(int)

    # 4. Age Groups
    # Since Age was normalized, we will bin the normalized age. 
    # For a StandardScaler, 0 is the mean (around 35 years old).
    # < -1 is roughly < 25. > 1 is roughly > 45.
    df['IS_YOUNG'] = (df['AGE'] < -1.0).astype(int)
    df['IS_SENIOR'] = (df['AGE'] > 1.0).astype(int)

    print(f"Feature Engineering Complete. Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    input_filepath = "../data/processed/processed_loan_data.csv"
    output_filepath = "../data/processed/engineered_loan_data.csv"
    
    df = load_processed_data(input_filepath)
    df = feature_engineering(df)
    
    df.to_csv(output_filepath, index=False)
    print(f"\nEngineered dataset saved to {output_filepath}")
