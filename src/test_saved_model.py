import joblib
import pandas as pd
import numpy as np

def test_model():
    """Loads the saved models and tests predictions."""
    model_path = "../models/best_model.pkl"
    features_path = "../models/feature_names.pkl"
    
    print("Loading saved artifacts...")
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    print(f"Model loaded successfully: {type(model).__name__}")
    
    # Create a dummy row of data using the exact feature names expected
    # We will just fill it with zeros, creating a "perfectly average/zero" customer
    dummy_data = {feature: [0.0] for feature in feature_names}
    df_fake = pd.DataFrame(dummy_data)
    
    print("\nRunning inference on fake data shape:", df_fake.shape)
    
    prediction = model.predict(df_fake)
    probability = model.predict_proba(df_fake)[0][1]
    
    print(f"Prediction Output: {'Default' if prediction[0] == 1 else 'No Default'}")
    print(f"Probability of Default: {probability:.4f}")

if __name__ == "__main__":
    test_model()
