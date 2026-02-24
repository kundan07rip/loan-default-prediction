import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads the dataset and prints basic info."""
    df = pd.read_csv(filepath)
    print("--- DATASET SHAPE ---")
    print(df.shape)
    print("\n--- DATASET INFO ---")
    df.info()
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    return df

def handle_missing_values(df):
    """Handles any missing values in the dataset."""
    # The UCI Credit Card dataset usually doesn't have missing values.
    # But as a best practice, we fill numerical NA with median and drop rest.
    # Check if there are missing values
    if df.isnull().sum().sum() > 0:
        print("\nHandling missing values...")
        # Since it's mostly numeric, we will fill with median
        df = df.fillna(df.median())
        print("Missing values handled.")
    else:
        print("\nNo missing values found.")
    return df

def encode_categorical(df):
    """Encodes categorical features. 
    In the UCI dataset, SEX, EDUCATION, MARRIAGE are already integers 
    but act as categories. We could one-hot encode them, but for tree-based 
    models and simple logistic regression, we can explicitly convert or one-hot encode.
    We will use One-Hot Encoding to be rigorous.
    """
    print("\nEncoding categorical features...")
    # These columns are categorical but represented as integers
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    # One-Hot Encoding using pd.get_dummies
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Dataset shape after encoding: {df.shape}")
    return df

def normalize_numerical(df, target_col='default.payment.next.month'):
    """Normalizes numerical columns using StandardScaler."""
    print("\nNormalizing numerical features...")
    scaler = StandardScaler()
    
    # Columns to normalize: LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6
    # We will exclude ID, Target, and encoded categorical columns
    cols_to_exclude = ['ID', target_col] + [col for col in df.columns if col.startswith(('SEX_', 'EDUCATION_', 'MARRIAGE_', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'))]
    
    # Actually, PAY_X are repayment statuses (categorical-ish). We'll leave them as is.
    numerical_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    if len(numerical_cols) > 0:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
    print("Normalization complete.")
    return df

if __name__ == "__main__":
    filepath = "../data/raw/loan_default_data.csv"
    print("Executing Data Preprocessing Pipeline...\n")
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = normalize_numerical(df)
    
    # Save the processed data
    processed_path = "../data/processed/processed_loan_data.csv"
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to {processed_path}")
