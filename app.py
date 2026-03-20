import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap

# App Configuration
st.set_page_config(page_title="CardioSafe for Diabetics", page_icon="🫀", layout="wide")

# ==========================================
# 1. Load Pretrained Models & Preprocessors
# ==========================================
@st.cache_resource
def load_assets():
    """Loads the model, imputer, and scaler using joblib and caches them for fast UI reload."""
    model = joblib.load("xgb_cvd_model.joblib")
    scaler = joblib.load("scaler.joblib")
    imputer = joblib.load("imputer.joblib")
    return model, scaler, imputer

try:
    model, scaler, imputer = load_assets()
except Exception as e:
    st.error(f"Error loading models. Please ensure you ran 'cvd_prediction_diabetics.py' first to generate the .joblib files.\nDetails: {e}")
    st.stop()

# ==========================================
# 2. Interface Design (Sidebar Inputs)
# ==========================================
st.sidebar.header("Patient Data Input 🩺")
st.sidebar.markdown("Enter the patient's vitals to calculate their 10-Year Cardiovascular Disease Risk. *(For Diabetic patients only)*")

# Use number_input and sliders based on reasonable medical ranges
male_input = st.sidebar.selectbox("Biological Sex", ["Female", "Male"])
male = 1 if male_input == "Male" else 0

age = st.sidebar.slider("Age (years)", min_value=30, max_value=90, value=50)

education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4], index=0)

currentSmoker_input = st.sidebar.checkbox("Current Smoker")
currentSmoker = 1 if currentSmoker_input else 0

cigsPerDay = st.sidebar.number_input("Cigarettes per Day", min_value=0, max_value=60, value=0)

bpMeds_input = st.sidebar.checkbox("Currently on Blood Pressure Medication")
BPMeds = 1 if bpMeds_input else 0

stroke_input = st.sidebar.checkbox("History of Stroke")
prevalentStroke = 1 if stroke_input else 0

hyp_input = st.sidebar.checkbox("History of Hypertension")
prevalentHyp = 1 if hyp_input else 0

totChol = st.sidebar.slider("Total Cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
sysBP = st.sidebar.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
diaBP = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=150, value=80)
bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
heartRate = st.sidebar.slider("Heart Rate (BPM)", min_value=40, max_value=150, value=75)
glucose = st.sidebar.slider("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120)

# Package the input into a DataFrame matching training column names EXACTLY
input_data = pd.DataFrame([{
    'male': male,
    'age': age,
    'education': education,
    'currentSmoker': currentSmoker,
    'cigsPerDay': cigsPerDay,
    'BPMeds': BPMeds,
    'prevalentStroke': prevalentStroke,
    'prevalentHyp': prevalentHyp,
    'totChol': totChol,
    'sysBP': sysBP,
    'diaBP': diaBP,
    'BMI': bmi,
    'heartRate': heartRate,
    'glucose': glucose
}])

# ==========================================
# 3. Main Data Processing & Prediction
# ==========================================
st.title("🫀 CardioSafe CVD Risk Analyzer")
st.markdown("### High-Precision Model for Diabetic Patients")
st.write("This tool uses a highly-tuned XGBoost model engineered specifically to flag cardiovascular risk with **minimal false positives** (Precision-optimized).")

# Only scale the exact 8 numerical columns originally scaled during training
numeric_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
input_imputed = input_data.copy()

# Ensure we handle potential missing values strictly for numeric_cols using the loaded imputer
# Since the UI enforces valid inputs, the imputer just passes them through, but this retains pipeline consistency.
input_imputed[numeric_cols] = imputer.transform(input_data[numeric_cols])

input_scaled = input_imputed.copy()
input_scaled[numeric_cols] = scaler.transform(input_imputed[numeric_cols])

# Generate Probability using predict_proba (Index 1 is high risk)
prediction_probability = model.predict_proba(input_scaled)[0, 1]

# High Precision Logic: Apply Custom 0.85 Threshold for >88% Precision
# (As tested previously, the higher the threshold, the lower the false positive rate.)
CUSTOM_THRESHOLD = 0.75 
is_high_risk = prediction_probability >= CUSTOM_THRESHOLD

st.divider()

# ==========================================
# 4. Display Results & Visuals
# ==========================================
st.subheader("Results")

col1, col2 = st.columns([1, 2])

with col1:
    if is_high_risk:
        st.error(f"⚠️ **WARNING: HIGH RISK**\nThe model is highly confident the patient is at risk for CVD.")
    else:
        st.success(f"✅ **LOW RISK (Success)**\nThe probability is below our stringent safety threshold.")
        st.info(f"*(Below {CUSTOM_THRESHOLD*100:.0f}% Threshold)*")

with col2:
    st.markdown("**Probability Meter:**")
    # Convert probability to a 0-100 integer for the progress bar
    progress_val = int(prediction_probability * 100)
    st.progress(progress_val)
    st.write(f"The model predicts a **{prediction_probability*100:.1f}%** probability that this patient will develop Cardiovascular Disease in 10 years.")


st.divider()

# ==========================================
# 5. Explainability (The "Why")
# ==========================================
st.subheader("Model Explainability (SHAP Logic)")
st.write("The chart below explains *exactly* which factors contributed to this specific patient's risk score. Red pushes the risk higher, Blue pushes it lower.")

# Initialize Explainer
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(input_scaled)

# Since we predict 1 patient, shap_vals is a 2D array: [1, num_features]
current_shap_values = shap_vals[0]

# Display Force Plot directly in Streamlit using streamlit_shap
st_shap(shap.force_plot(explainer.expected_value, current_shap_values, input_data.iloc[0, :]), height=150)

# Display Top Risk Factors Directly
def display_risk_factors(shap_values, feature_names, input_df):
    """
    Simply displays the top factors mathematically affecting the current patient.
    """
    contributions = pd.Series(shap_values, index=feature_names)
    sorted_contributions = contributions.sort_values(ascending=False)
    
    st.markdown("### Primary Factors Affecting Your Score:")
    
    col_risk, col_protect = st.columns(2)
    
    with col_risk:
        st.error("**Factors Increasing Risk:**")
        # Get top 3 factors pushing the score up
        risk_drivers = sorted_contributions[sorted_contributions > 0].head(3)
        if not risk_drivers.empty:
            for feature, shap_val in risk_drivers.items():
                patient_val = input_df[feature].values[0]
                st.markdown(f"- **{feature}** (Patient Value: {patient_val})")
        else:
            st.markdown("- None significant.")

    with col_protect:
        st.success("**Factors Lowering Risk:**")
        # Get top 3 factors pushing the score down
        protectors = sorted_contributions[sorted_contributions < 0].tail(3).sort_values(ascending=True)
        if not protectors.empty:
            for feature, shap_val in protectors.items():
                patient_val = input_df[feature].values[0]
                st.markdown(f"- **{feature}** (Patient Value: {patient_val})")
        else:
            st.markdown("- None significant.")

# Display the factors
display_risk_factors(current_shap_values, input_data.columns, input_data)
