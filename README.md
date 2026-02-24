# 🏦 Loan Default Prediction System

## 📌 Problem Statement
Financial institutions encounter significant losses when borrowers fail to repay their loans (default).
The objective of this project is to build an end-to-end Machine Learning Classification system capable of accurately predicting whether an applicant will default on a credit loan in the upcoming month based on their demographic profiles, historical billing data, and payment behavioral trends.

## 🛠️ Tech Stack & MLOps Tools
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Logistic Regression, Random Forest, Cross-Validation, Metrics)
- **Advanced Modeling**: XGBoost (Extreme Gradient Boosting)
- **Data Visualization**: Matplotlib, Seaborn
- **Deployment & UI**: Streamlit
- **Serialization**: Joblib

## 📁 Project Structure
Following strict MLOps principles, the repository separates logic from data:
```
loan-default-prediction/
│
├── data/
│   ├── raw/                  # Original raw datasets (Read-only)
│   └── processed/            # Intermediary engineered datasets
│
├── notebooks/                # Jupyter Notebooks for scratchpad EDA
│
├── src/                      # Production Python pipelines
│   ├── data_preprocessing.py # Missing value handling, One-Hot Encoding, Scaling
│   ├── eda.py                # Visual analytical script
│   ├── feature_engineering.py# Creation of proxy utilization & trend features
│   ├── train_model.py        # Model Training, CV, and evaluation
│   ├── evaluate_model.py     # Custom scoring and ROC extraction
│   └── test_saved_model.py   # Sanity check for serialization
│
├── visualizations/           # Auto-generated PNGs (Correlation, Class Distrib, Feature Importances)
├── models/                   # Serialized XGBoost (.pkl files)
├── app.py                    # Streamlit Deployment Web App
├── Interview.txt             # Exhaustive Q&A Concept Guide
└── README.md                 # You are here
```

## 📊 Model Comparison & Results
We trained three algorithms utilizing a strict 80-20 Train-Test split layered with 5-Fold Cross Validation. To counter the 78:22 Class Imbalance inherently present in the UCI Default Dataset, we applied explicit algorithmic Class Weighting.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6543   | 0.3547    | 0.6277 | 0.4533   | 0.6970  |
| Random Forest       | 0.8143   | 0.6186    | 0.4072 | 0.4911   | 0.7788  |
| **XGBoost (WINNER)**| **0.8038** | **0.5516** | **0.5282** | **0.5396** | **0.7816** |

**Conclusion**: While Random Forest achieved slightly higher raw accuracy, **XGBoost** dominated in the pivotal ROC-AUC metric (0.7816), indicating superior capability in distinguishing true defaults from non-defaults across all probability thresholds. Furthermore, XGBoost achieved a significantly higher F1-Score, striking a better equilibrium between Precision and Recall.

### 🖼️ ROC Curve
*(Found in `/visualizations/roc_curves.png`)*
The ROC Curve illustrates XGBoost consistently outperforming the baseline Random Guess (0.5 AUC) and Logistic Regression by a wide margin.

## 🚀 How to Run the Project Locally

**1. Clone the repository & Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. (Optional) Run pipeline completely from scratch:**
```bash
cd src
python data_preprocessing.py
python feature_engineering.py
python eda.py
python train_model.py
```

**3. Launch the Streamlit Web Application:**
```bash
streamlit run app.py
```
*The Streamlit App maps 6 intuitive user inputs (Age, Income, Credit Score, Debt Ratio, etc.) directly into the complex 30-feature vector expected by the backend XGBoost model.*

## 📈 Future Improvements
- **SMOTE & ADASYN**: Explore synthetic oversampling techniques instead of algorithmic class weighting to potentially boost Recall.
- **Hyperparameter Tuning**: Run `GridSearchCV` or `Optuna` over XGBoost's `max_depth`, `learning_rate`, and `gamma` arguments to squeeze out additional AUC points.
- **Cloud Deployment**: Containerize `app.py` using Docker and deploy to AWS Elastic Beanstalk or Render.
