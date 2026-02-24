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
        /* Import premium Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap');
        
        /* Global Typography */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
            color: #e0e0e0;
        }
        
        h1, h2, h3, .st-emotion-cache-10trblm {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        /* Animated Gradient App Background */
        .stApp {
            background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Advanced Animations */
        @keyframes fadeInSlideUp {
            0% { opacity: 0; transform: translateY(40px) scale(0.95); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        @keyframes glowingBorder {
            0% { border-color: rgba(255, 215, 0, 0.2); box-shadow: 0 0 10px rgba(255, 215, 0, 0.1); }
            50% { border-color: rgba(255, 215, 0, 0.8); box-shadow: 0 0 25px rgba(255, 215, 0, 0.6), inset 0 0 10px rgba(255, 215, 0, 0.2); }
            100% { border-color: rgba(255, 215, 0, 0.2); box-shadow: 0 0 10px rgba(255, 215, 0, 0.1); }
        }

        @keyframes floatEffect {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        @keyframes textShine {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }

        /* Main container glassmorphism and animation */
        .block-container {
            animation: fadeInSlideUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            background: rgba(20, 20, 25, 0.6);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 3rem !important;
            margin-top: 2rem !important;
            margin-bottom: 2rem !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        /* Styling Inputs with Glassmorphism and Focus Effects */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div,
        input[type="number"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            color: white !important;
        }
        
        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="select"] > div:hover {
            border: 1px solid rgba(255, 215, 0, 0.4) !important;
            background: rgba(255, 255, 255, 0.08) !important;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        div[data-baseweb="input"] > div:focus-within {
            animation: glowingBorder 2s infinite;
            background: rgba(255, 255, 255, 0.1) !important;
        }

        /* Sliders */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #ffd700, #ff8c00);
        }

        /* Primary Button Styling - Hyper Animated */
        div.stButton > button:first-child {
            background: linear-gradient(45deg, #ffd700, #ff8c00, #ffd700);
            background-size: 200% auto;
            color: #1a1a1a !important;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 800;
            font-size: 1.1rem;
            border-radius: 50px;
            border: none;
            padding: 1.2rem 3rem;
            transition: all 0.5s cubic-bezier(0.25, 1, 0.5, 1);
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.3);
            text-transform: uppercase;
            letter-spacing: 2px;
            position: relative;
            overflow: hidden;
            animation: textShine 3s linear infinite;
        }
        
        div.stButton > button:first-child::before {
            content: '';
            position: absolute;
            top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: all 0.5s ease;
        }
        
        div.stButton > button:first-child:hover::before {
            left: 100%;
            transition: all 0.5s ease;
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-5px) scale(1.03);
            box-shadow: 0 15px 35px rgba(255, 215, 0, 0.5), 0 0 20px rgba(255, 140, 0, 0.4);
        }
        
        /* Hide Default Streamlit Elements (Header, Footer, Menu) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Headers styling */
        h1 {
            background: linear-gradient(to right, #ffffff 20%, #ffd700 40%, #ffd700 60%, #ffffff 80%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: textShine 4s linear infinite;
            margin-bottom: 0.5rem;
            font-size: 3.5rem !important;
            text-align: center;
            letter-spacing: -1px;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #b0b0b0;
            font-size: 1.2rem;
            opacity: 0;
            animation: fadeInSlideUp 1s ease 0.5s forwards;
            margin-bottom: 2rem;
            line-height: 1.6;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Animated dividers */
        hr {
            border: 0;
            height: 1px;
            background-image: linear-gradient(to right, rgba(255,215,0,0), rgba(255,215,0,0.5), rgba(255,215,0,0));
            margin: 2rem 0;
        }
        
        /* Labels inside inputs */
        .stNumberInput label, .stSlider label {
            color: #ffd700 !important;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 0.85rem;
            opacity: 0.9;
        }
        
        /* Results Cards styling */
        .result-card {
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: floatEffect 6s ease-in-out infinite;
        }
        
        .result-card:hover {
            transform: scale(1.03) translateY(-10px) !important;
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
    
    st.markdown("<h1>🏦 Loan Default Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p class='subtitle'>
    Welcome to the <b>Loan Default Predictor</b>.<br>
    Enter the applicant's details below to accurately assess their risk of defaulting on a loan next month.
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
                <div class="result-card" style="background: rgba(255, 50, 50, 0.15); border: 2px solid #ff3232; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 0 25px rgba(255,50,50,0.4), inset 0 0 15px rgba(255,50,50,0.2);">
                    <h2 style="color: #ff3232; margin:0; font-family: 'Space Grotesk', sans-serif;">🚨 HIGH RISK</h2>
                    <h1 style="color: #ffffff; font-size: 4.5rem; margin: 15px 0; background: none; -webkit-text-fill-color: white;">{probability:.1%}</h1>
                    <p style="color: #dddddd; font-size: 1.2rem; margin:0;">This applicant has a very high mathematical probability of defaulting.</p>
                    <div style="background: rgba(255, 50, 50, 0.2); border-radius: 8px; padding: 10px; margin-top: 20px;">
                        <p style="color: #ffaa00; font-weight: bold; margin: 0;">RECOMMENDATION: Reject or Require Co-signer/Collateral.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif probability >= 0.5:
                st.markdown(f"""
                <div class="result-card" style="background: rgba(255, 215, 0, 0.15); border: 2px solid #ffd700; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 0 25px rgba(255,215,0,0.4), inset 0 0 15px rgba(255,215,0,0.2);">
                    <h2 style="color: #ffd700; margin:0; font-family: 'Space Grotesk', sans-serif;">⚠️ MEDIUM RISK</h2>
                    <h1 style="color: #ffffff; font-size: 4.5rem; margin: 15px 0; background: none; -webkit-text-fill-color: white;">{probability:.1%}</h1>
                    <p style="color: #dddddd; font-size: 1.2rem; margin:0;">This applicant shows concerning financial indicators.</p>
                    <div style="background: rgba(255, 215, 0, 0.2); border-radius: 8px; padding: 10px; margin-top: 20px;">
                        <p style="color: #ffd700; font-weight: bold; margin: 0;">RECOMMENDATION: Manual Underwriting Review.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="background: rgba(50, 255, 100, 0.15); border: 2px solid #32ff64; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 0 25px rgba(50,255,100,0.4), inset 0 0 15px rgba(50,255,100,0.2);">
                    <h2 style="color: #32ff64; margin:0; font-family: 'Space Grotesk', sans-serif;">✅ LOW RISK</h2>
                    <h1 style="color: #ffffff; font-size: 4.5rem; margin: 15px 0; background: none; -webkit-text-fill-color: white;">{probability:.1%}</h1>
                    <p style="color: #dddddd; font-size: 1.2rem; margin:0;">This applicant displays healthy financial indicators.</p>
                    <div style="background: rgba(50, 255, 100, 0.2); border-radius: 8px; padding: 10px; margin-top: 20px;">
                        <p style="color: #32ff64; font-weight: bold; margin: 0;">RECOMMENDATION: Approve.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
