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

def load_custom_css():
    st.markdown("""
    <style>
        /* Import premium Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Global Typography and Background */
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* App Background */
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgb(20, 20, 22) 0%, rgb(10, 10, 12) 90%);
        }

        /* Animations */
        @keyframes fadeInSlideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.2); }
            50% { box-shadow: 0 0 25px rgba(255, 215, 0, 0.6); }
            100% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.2); }
        }

        /* Apply animation to main container */
        .block-container {
            animation: fadeInSlideUp 0.8s ease-out forwards;
        }

        /* Styling Inputs with Glassmorphism */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="slider"] > div {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.3s ease;
        }
        
        div[data-baseweb="input"] > div:hover, 
        div[data-baseweb="slider"] > div:hover {
            border: 1px solid rgba(255, 215, 0, 0.5) !important;
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
        }

        /* Primary Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
            color: #1a1a1a !important;
            font-weight: 700;
            border-radius: 30px;
            border: none;
            padding: 15px 30px;
            transition: all 0.4s ease;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.5);
            animation: pulseGlow 1.5s infinite;
        }
        
        /* Hide Default Streamlit Elements (Header, Footer, Menu) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Headers styling */
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        h1 {
            background: -webkit-linear-gradient(45deg, #ffffff, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


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
    st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")
    load_custom_css()
    
    st.markdown("<h1>🏦 Loan Default Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #a0a0a0; font-size: 1.1rem; margin-bottom: 2rem;'>
    Welcome to the <b>Loan Default Predictor</b>. 
    Enter the applicant's details below to assess their risk of defaulting on a loan next month.
    </p>
    """, unsafe_allow_html=True)
    
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

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Button
    if st.button("Predict Default Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing applicant profile..."):
            import time
            time.sleep(1) # Add a small artificial delay to show off the spinner/animations
            
            # Map UI inputs to model inputs
            df_input = map_user_input_to_model(age, income, loan_amount, credit_score, emp_years, debt_ratio, feature_names)
            
            # Predict
            probability = model.predict_proba(df_input)[0][1]
            
            # Display Results
            st.markdown("<h3 style='margin-top: 2rem;'>Risk Assessment</h3>", unsafe_allow_html=True)
            
            # Custom Animated Result Cards
            if probability >= 0.8:
                st.markdown(f"""
                <div style="background: rgba(255, 50, 50, 0.1); border: 1px solid #ff3232; border-radius: 12px; padding: 20px; text-align: center; animation: pulseGlow 2s infinite; box-shadow: 0 0 15px rgba(255,50,50,0.5);">
                    <h2 style="color: #ff3232; margin:0;">🚨 HIGH RISK</h2>
                    <h1 style="color: #ffffff; font-size: 3rem; margin: 10px 0;">{probability:.1%}</h1>
                    <p style="color: #dddddd; font-size: 1.1rem; margin:0;">This applicant has a very high mathematical probability of defaulting.</p>
                    <p style="color: #ffaa00; font-weight: bold; margin-top: 10px;">Recommendation: Reject or Require Co-signer/Collateral.</p>
                </div>
                """, unsafe_allow_html=True)
            elif probability >= 0.5:
                st.markdown(f"""
                <div style="background: rgba(255, 215, 0, 0.1); border: 1px solid #ffd700; border-radius: 12px; padding: 20px; text-align: center;">
                    <h2 style="color: #ffd700; margin:0;">⚠️ MEDIUM RISK</h2>
                    <h1 style="color: #ffffff; font-size: 3rem; margin: 10px 0;">{probability:.1%}</h1>
                    <p style="color: #dddddd; font-size: 1.1rem; margin:0;">This applicant shows concerning financial indicators.</p>
                    <p style="color: #ffaa00; font-weight: bold; margin-top: 10px;">Recommendation: Manual Underwriting Review.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(50, 255, 100, 0.1); border: 1px solid #32ff64; border-radius: 12px; padding: 20px; text-align: center;">
                    <h2 style="color: #32ff64; margin:0;">✅ LOW RISK</h2>
                    <h1 style="color: #ffffff; font-size: 3rem; margin: 10px 0;">{probability:.1%}</h1>
                    <p style="color: #dddddd; font-size: 1.1rem; margin:0;">This applicant displays healthy financial indicators.</p>
                    <p style="color: #32ff64; font-weight: bold; margin-top: 10px;">Recommendation: Approve.</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
