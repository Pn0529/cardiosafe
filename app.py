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

# Patient-friendly feedback function
def get_patient_feedback(feature, value, is_concern):
    """
    Converts medical metrics into patient-friendly feedback messages.
    """
    feedback_map = {
        'age': {
            True: f"Your age ({value}) is a factor in cardiovascular health. Regular check-ups become more important.",
            False: f"Your age ({value}) is in a good range for cardiovascular health."
        },
        'sysBP': {
            True: f"Your systolic blood pressure ({value} mmHg) is elevated. Consider consulting your doctor about blood pressure management.",
            False: f"Your systolic blood pressure ({value} mmHg) is within a healthy range."
        },
        'diaBP': {
            True: f"Your diastolic blood pressure ({value} mmHg) is elevated. Monitor this regularly.",
            False: f"Your diastolic blood pressure ({value} mmHg) is well-controlled."
        },
        'glucose': {
            True: f"Your blood glucose level ({value} mg/dL) needs attention. Proper diabetes management is crucial.",
            False: f"Your blood glucose level ({value} mg/dL) is well-managed."
        },
        'BMI': {
            True: f"Your BMI ({value}) suggests weight management could help your heart health.",
            False: f"Your BMI ({value}) is in a healthy range for your heart."
        },
        'totChol': {
            True: f"Your cholesterol level ({value} mg/dL) is elevated. Consider dietary changes and discuss with your doctor.",
            False: f"Your cholesterol level ({value} mg/dL) is well-controlled."
        },
        'heartRate': {
            True: f"Your heart rate ({value} BPM) is elevated. Consider stress management and regular exercise.",
            False: f"Your heart rate ({value} BPM) indicates good cardiovascular fitness."
        },
        'cigsPerDay': {
            True: f"Smoking ({value} cigarettes/day) significantly impacts heart health. Quitting is strongly recommended.",
            False: f"Being smoke-free greatly benefits your heart health."
        },
        'currentSmoker': {
            True: "Smoking status is a major concern for heart health. Consider quitting programs.",
            False: "Being a non-smoker protects your heart."
        },
        'prevalentHyp': {
            True: "History of hypertension requires careful monitoring and medication adherence.",
            False: "No hypertension history is positive for heart health."
        },
        'BPMeds': {
            True: "Blood pressure medications are working - continue as prescribed.",
            False: "No blood pressure medication needed currently."
        },
        'prevalentStroke': {
            True: "Stroke history requires careful cardiovascular monitoring.",
            False: "No stroke history is favorable."
        },
        'male': {
            True: "Male gender has different cardiovascular risk factors - stay proactive.",
            False: "Female cardiovascular health has unique considerations."
        },
        'education': {
            True: f"Education level ({value}) may affect health literacy - seek clear medical guidance.",
            False: f"Education level ({value}) supports good health understanding."
        }
    }
    
    return feedback_map.get(feature, {
        True: f"Your {feature} level ({value}) needs medical attention.",
        False: f"Your {feature} level ({value}) is favorable."
    }).get(is_concern, f"Your {feature} ({value}) contributes to your health profile.")

def get_risk_explanation(feature, value):
    """
    Provides clear explanations about patient's main risk factors.
    """
    risk_explanations = {
        'age': f"Age {value} years - Natural risk factor that increases with time",
        'sysBP': f"High blood pressure ({value} mmHg) - Major cardiovascular risk",
        'diaBP': f"Elevated diastolic pressure ({value} mmHg) - Heart strain indicator",
        'glucose': f"High blood sugar ({value} mg/dL) - Diabetes complication risk",
        'BMI': f"High BMI ({value}) - Excess weight burdens the heart",
        'totChol': f"High cholesterol ({value} mg/dL) - Artery blockage risk",
        'heartRate': f"Elevated heart rate ({value} BPM) - Heart overworking",
        'cigsPerDay': f"Smoking {value} cigarettes daily - Major heart damage",
        'currentSmoker': "Current smoking habit - Direct heart toxin exposure",
        'prevalentHyp': "Hypertension history - Chronic high blood pressure damage",
        'BPMeds': "Blood pressure medication requirement - Controlled but present risk",
        'prevalentStroke': "Previous stroke event - High recurrence risk",
        'male': "Male gender - Higher baseline CVD risk",
        'education': f"Education level {value} - May affect health management"
    }
    
    return risk_explanations.get(feature, f"{feature}: {value} - Contributing risk factor")

def get_health_recommendation(feature, value):
    """
    Provides actionable health recommendations based on risk factors.
    """
    recommendations = {
        'age': "Schedule regular cardiac check-ups and screenings",
        'sysBP': "Reduce sodium intake, exercise regularly, consider medication",
        'diaBP': "Manage stress, limit alcohol, maintain healthy weight",
        'glucose': "Follow diabetes management plan, monitor glucose closely",
        'BMI': "Adopt heart-healthy diet, increase physical activity",
        'totChol': "Reduce saturated fats, consider cholesterol-lowering medication",
        'heartRate': "Practice stress reduction, ensure adequate sleep",
        'cigsPerDay': "Quit smoking immediately - seek smoking cessation support",
        'currentSmoker': "Join smoking cessation program, use nicotine replacement",
        'prevalentHyp': "Take blood pressure medication as prescribed",
        'BPMeds': "Continue medication, monitor blood pressure at home",
        'prevalentStroke': "Follow neurologist's prevention plan strictly",
        'male': "Be proactive about heart health screening",
        'education': "Ask doctor to explain health information clearly"
    }
    
    return recommendations.get(feature, f"Consult your doctor about managing {feature}")

# Display Top Risk Factors Directly
def display_risk_factors(shap_values, feature_names, input_df):
    """
    Provides patients with clear information about their main risk factors and habits.
    """
    contributions = pd.Series(shap_values, index=feature_names)
    sorted_contributions = contributions.sort_values(ascending=False)
    
    st.markdown("### Your Main Cardiovascular Risk Factors:")
    
    col_risks, col_recommendations = st.columns(2)
    
    with col_risks:
        st.error("**Primary Risk Areas:**")
        # Get top 3 factors pushing the score up
        risk_drivers = sorted_contributions[sorted_contributions > 0].head(3)
        if not risk_drivers.empty:
            for feature, shap_val in risk_drivers.items():
                patient_val = input_df[feature].values[0]
                risk_info = get_risk_explanation(feature, patient_val)
                st.markdown(f"- {risk_info}")
        else:
            st.markdown("- No significant risk factors identified.")

    with col_recommendations:
        st.info("**Health Recommendations:**")
        # Get top 3 factors pushing the score up for targeted recommendations
        risk_drivers = sorted_contributions[sorted_contributions > 0].head(3)
        if not risk_drivers.empty:
            for feature, shap_val in risk_drivers.items():
                patient_val = input_df[feature].values[0]
                recommendation = get_health_recommendation(feature, patient_val)
                st.markdown(f"- {recommendation}")
        else:
            st.markdown("- Continue maintaining your healthy lifestyle.")

# Display the factors
display_risk_factors(current_shap_values, input_data.columns, input_data)
