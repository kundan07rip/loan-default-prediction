import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

def evaluate_predictions(y_true, y_pred, y_prob, model_name="Model"):
    """
    Calculates and returns standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    }
    return metrics

def plot_roc_curves(models_dict, X_test, y_test, output_path='../visualizations/roc_curves.png'):
    """
    Takes a dictionary of trained models, calculates their ROC curves, 
    and plots them on a single graph.
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # For linear models like SVM if predict_proba is not available, though Logistic Regression has it.
            y_prob = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\nROC curves saved to {output_path}")

def plot_feature_importance(model, feature_names, model_name, output_path):
    """
    Plots the top 10 most important features for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1][:15] # Top 15
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(15), importances[indices], align="center", color='teal')
        plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, 15])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Feature importance saved to {output_path}")
    else:
        print(f"Skipping feature importance for {model_name} (Not applicable)")
