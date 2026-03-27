# Mini Project Report: CardioSafe - CVD Prediction App for Diabetics

## 1. Project Title
**CardioSafe**: High-Precision Cardiovascular Disease (CVD) Risk Analyzer for Diabetic Patients.

## 2. Introduction & Objective
Cardiovascular Disease (CVD) is a leading cause of mortality globally, and diabetic patients are at a significantly higher risk. The objective of the **CardioSafe** project is to develop a machine learning-based web application that predicts a patient's 10-year risk of developing Cardiovascular Disease. 

The primary focus of this application is **high precision**. In medical diagnostics, minimizing false positives is often critical to avoid unnecessary stress and treatments. Thus, the model is finely tuned to ensure a high confidence level before labeling a patient as "High Risk." Furthermore, to build trust with medical professionals, the prediction is paired with **SHAP (SHapley Additive exPlanations)** to provide transparent, individualized reasoning for every prediction.

## 3. Dataset
The project utilizes the renowned **Framingham Heart Study dataset**. 
- **Filtering**: To serve its specific objective, the dataset is strictly filtered to include **only diabetic patients**. 
- **Features**: Patient metrics used for prediction include Age, Biological Sex, Education, Smoking Habits (Cigarettes per Day), Blood Pressure Medication, History of Stroke/Hypertension, Total Cholesterol, Systolic/Diastolic Blood Pressure, BMI, Heart Rate, and Blood Glucose levels.
- **Target Variable**: `TenYearCHD` (10-year risk of Coronary Heart Disease).

## 4. Technology Stack
- **Programming Language**: Python 3.8+
- **Web Framework**: Streamlit (for building the interactive User Interface)
- **Machine Learning**: Scikit-Learn (Preprocessing, Evaluation), XGBoost (XGBClassifier for predictive modeling)
- **Model Explainability**: SHAP (TreeExplainer and Force Plots)
- **Data Manipulation**: Pandas, NumPy
- **Serialization**: Joblib (for saving and loading the trained model and preprocessors)

## 5. System Architecture
The project is divided into two primary scripts:
1. **Model Training & Pipeline (`cvd_prediction_diabetics.py`)**: Responsible for data loading, preprocessing, model training, precision-tuning, SHAP evaluation, and saving the artifacts (`.joblib` files).
2. **Web Application (`app.py`)**: The Streamlit frontend that loads the saved model pipeline, collects user input via a sidebar, processes the input, and dynamically displays the prediction result and SHAP explanations.

## 6. Machine Learning Methodology
### A. Preprocessing
- **Handling Missing Data**: Missing numerical values are imputed using the **Median** (`SimpleImputer(strategy='median')`), providing robustness against outliers.
- **Feature Scaling**: Numerical features are scaled to a standard normal distribution using **StandardScaler** to ensure all features contribute proportionately to the model.

### B. Modeling
- The core algorithm is the **XGBoost Classifier**, chosen for its high performance, capability of handling complex non-linear relationships, and tree-based architecture which is highly compatible with SHAP explainers.

### C. Precision Tuning
- Instead of using the default 0.5 probability threshold, the model dynamically computes an optimal threshold during training to target a **Precision score of >88%**. In the Streamlit app, a strict custom threshold (e.g., 0.75) is utilized to ensure the model issues a "High Risk" warning only when highly confident, drastically reducing false positives.

## 7. Model Explainability (SHAP)
To avoid the "black-box" nature of traditional ML models, CardioSafe integrates **SHAP**:
- **Global Explainability**: Analyzes overall feature importance across the entire dataset during the training phase.
- **Local Explainability**: For every individual patient analyzed in the app, SHAP calculates exactly how much each specific vitals reading (e.g., high systolic BP or low BMI) pushed the risk score higher or lower. This is visualized directly in the Streamlit UI using customized Force plots and factor breakdowns.

## 8. User Interface (Streamlit App)
The frontend provides a seamless experience:
- **Sidebar Data Input**: Clinicians enter patient vitals using sliders and dropdowns restricted to realistic medical ranges.
- **Real-Time Analysis**: Upon input, data is routed through the loaded scaler and imputer, then inferred by the XGBoost model.
- **Results Dashboard**: Displays a comprehensive Probability Meter and a clear "High Risk" or "Low Risk" badge based on the stringent threshold.
- **Explainability Chart**: A dynamic SHAP Force chart is rendered inside the app, detailing the **Primary Factors Increasing Risk** and **Factors Lowering Risk** to aid clinical decision-making.

## 9. Conclusion
 CardioSafe successfully bridges the gap between advanced predictive machine learning and clinical applicability. By focusing exclusively on diabetic patients, enforcing strict precision thresholds to minimize false alarms, and providing crystal-clear SHAP explainability for every single prediction, CardioSafe serves as a transparent and reliable assistive tool for proactive cardiovascular healthcare.
