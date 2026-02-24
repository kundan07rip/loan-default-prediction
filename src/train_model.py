import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# Import our custom evaluation metrics
from evaluate_model import evaluate_predictions, plot_roc_curves, plot_feature_importance

def load_and_split_data(filepath):
    print("Loading engineered data...")
    df = pd.read_csv(filepath)
    
    # Target feature
    target_col = 'default.payment.next.month'
    
    # Ensure ID is dropped if it exists
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 80-20 Train-Test split
    # stratify=y ensures the 78:22 class imbalance ratio is maintained in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns

def train_and_evaluate():
    filepath = "../data/processed/engineered_loan_data.csv"
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(filepath)
    
    # Class weights for imbalanced data. 
    # The default class is ~22%, so we assign higher weight to the minority class (1).
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    print("\n--- Initializing Models ---")
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    }
    
    trained_models = {}
    evaluation_results = []
    
    print("\n--- Training and Cross-Validating Models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Perform 5-Fold Cross Validation on the training set using ROC-AUC as the primary metric
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"{name} 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Fit the model on the full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on Test Set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred, y_prob, model_name=name)
        evaluation_results.append(metrics)
        
    print("\n--- FINAL TEST SET RESULTS ---")
    results_df = pd.DataFrame(evaluation_results)
    print(results_df.to_string(index=False))
    
    # Plot ROC Curves for all models
    plot_roc_curves(trained_models, X_test, y_test)
    
    # Plot Feature Importance for Random Forest and XGBoost
    if "Random Forest" in trained_models:
        plot_feature_importance(trained_models["Random Forest"], feature_names, "Random Forest", "../visualizations/feature_importance_rf.png")
    if "XGBoost" in trained_models:
        plot_feature_importance(trained_models["XGBoost"], feature_names, "XGBoost", "../visualizations/feature_importance_xgb.png")
    
    # Identify the best model based on ROC-AUC
    # The evaluation_results contains metrics dictionaries
    best_model_name = max(evaluation_results, key=lambda x: x['ROC-AUC'])['Model']
    print(f"\nBest Model based on ROC-AUC: {best_model_name}")
    best_model = trained_models[best_model_name]
    
    # Save the best model
    model_path = "../models/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # Also save the feature names used for training, we need them for the Streamlit App
    features_path = "../models/feature_names.pkl"
    joblib.dump(list(feature_names), features_path)
    print(f"Saved feature names to {features_path}")

    return trained_models, results_df, X_train.columns

if __name__ == "__main__":
    trained_models, results_df, feature_names = train_and_evaluate()
