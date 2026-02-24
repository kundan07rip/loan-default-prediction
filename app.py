import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model_and_features():
    """Load model and feature names, cached so it doesn't reload on every UI interaction."""
    model = joblib.load("models/best_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

def map_user_input_to_model(age, income, loan_amount, credit_score, emp_years, debt_ratio, feature_names):
    """
    Since the backend model was trained on 30 highly specific, standardized features 
    from the UCI dataset, we must map the user's intuitive inputs to approximate those features.
    """
    # Initialize a dictionary with zeros for all required features
    input_data = {feat: [0.0] for feat in feature_names}
    
    # 1. MAP AGE (Model expects standardized age. Mean ~35, STD ~9.2)
    # If the model wasn't standardized, we'd just pass age. Here we approximate:
    std_age = (age - 35.48) / 9.21
    if 'AGE' in input_data:
        input_data['AGE'] = [std_age]
        
    # Age grouping features we engineered
    if 'IS_YOUNG' in input_data:
        input_data['IS_YOUNG'] = [1 if age < 25 else 0]
    if 'IS_SENIOR' in input_data:
        input_data['IS_SENIOR'] = [1 if age > 55 else 0]
        
    # 2. MAP LOAN AMOUNT & INCOME -> LIMIT_BAL proxies
    # UCI dataset used LIMIT_BAL (Mean ~167k, STD ~129k)
    std_limit_bal = (loan_amount - 167484) / 129747
    if 'LIMIT_BAL' in input_data:
        input_data['LIMIT_BAL'] = [std_limit_bal]
        
    # 3. MAP DEBT RATIO -> PAY_TO_BILL_RATIO
    # A high debt ratio implies high credit utilization.
    if 'PAY_TO_BILL_RATIO' in input_data:
        input_data['PAY_TO_BILL_RATIO'] = [debt_ratio]
        
    # 4. MAP CREDIT SCORE -> TOTAL_SEVERE_DELAYS & PAYMENT STATUS
    # Credit score is a proxy for past repayment behavior.
    # Suppose < 600 means bad history (delays), > 700 means good history.
    severe_delays = 0
    recent_delay_status = 0 # 0 means paid on time
    if credit_score < 600:
        severe_delays = 3
        recent_delay_status = 2 # 2 months delayed recently
    elif credit_score < 680:
        severe_delays = 1
        recent_delay_status = 1
    
    if 'TOTAL_SEVERE_DELAYS' in input_data:
        input_data['TOTAL_SEVERE_DELAYS'] = [severe_delays]
        
    if 'PAY_0' in input_data:
        input_data['PAY_0'] = [recent_delay_status]
        
    # 5. MAPPING EMPLOYMENT YEARS
    # Employment years gives stability. If employed > 5 years, less likely to have worsening trend.
    if 'DELAY_TREND_WORSENING' in input_data:
        input_data['DELAY_TREND_WORSENING'] = [1 if emp_years < 2 and credit_score < 650 else 0]

    return pd.DataFrame(input_data)

def main():
    st.title("🏦 Loan Default Prediction System")
    st.markdown("""
    Welcome to the **Loan Default Predictor**. 
    Enter the applicant's details below to assess their risk of defaulting on a loan next month.
    """)
    
    st.divider()
    
    # Load model
    try:
        model, feature_names = load_model_and_features()
    except Exception as e:
        st.error("Error loading model. Please ensure Phase 8 was completed successfully.")
        return
        
    # Layout using columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=10000, value=65000, step=5000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, value=20000, step=1000)
        
    with col2:
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
        emp_years = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        debt_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

    st.divider()
    
    # Prediction Button
    if st.button("Predict Default Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing applicant profile..."):
            # Map UI inputs to model inputs
            df_input = map_user_input_to_model(age, income, loan_amount, credit_score, emp_years, debt_ratio, feature_names)
            
            # Predict
            probability = model.predict_proba(df_input)[0][1]
            
            # Display Results
            st.markdown("### Risk Assessment")
            
            if probability >= 0.8:
                st.error(f"🚨 **HIGH RISK** (Probability of Default: {probability:.1%})")
                st.write("This applicant has a very high mathematical probability of defaulting. Recommendation: **Reject or Require Co-signer/Collateral**.")
            elif probability >= 0.5:
                st.warning(f"⚠️ **MEDIUM RISK** (Probability of Default: {probability:.1%})")
                st.write("This applicant shows concerning financial indicators. Recommendation: **Manual Underwriting Review**.")
            else:
                st.success(f"✅ **LOW RISK** (Probability of Default: {probability:.1%})")
                st.write("This applicant displays healthy financial indicators. Recommendation: **Approve**.")
                
            # Show a simple progress bar scaled to 100%
            st.progress(float(probability))

if __name__ == "__main__":
    main()
