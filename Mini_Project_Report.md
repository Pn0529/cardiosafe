<div align="center">
  <h1 style="font-size: 18pt;">CARDIOSAFE: HIGH-PRECISION CVD RISK ANALYZER FOR DIABETIC PATIENTS</h1>
  
  <p style="font-size: 14pt;">
    A Mini Project Report<br><br>
    In partial fulfillment of the requirements for the award of the degree of<br><br>
    <strong>BACHELOR OF TECHNOLOGY</strong><br>
    In<br>
    <strong>COMPUTER SCIENCE AND ENGINEERING</strong><br><br>
    Submitted by<br>
    <strong>NAME (REGD NO)</strong><br>
    <strong>NAME (REGD NO)</strong><br><br>
    <strong>DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING</strong><br>
    <strong>S.R.K.R. ENGINEERING COLLEGE (A)</strong><br>
    CHINNA AMIRAM, BHIMAVARAM, W.G. DIST., A.P.<br>
    <strong>[2024 – 2025]</strong>
  </p>
</div>

<br><br>
<hr>

<div style="font-family: 'Times New Roman', serif; font-size: 14pt; line-height: 1.5;">

<h2 align="center">BONAFIDE CERTIFICATE</h2>

<p>This is to certify that the project work entitled <strong>"CARDIOSAFE: HIGH-PRECISION CVD RISK ANALYZER FOR DIABETIC PATIENTS"</strong> is the bonafide work of &lt;NAME(S)&gt; bearing &lt;REG NO(S)&gt; who carried out the project work under my supervision in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in COMPUTER SCIENCE AND ENGINEERING.</p>

<br><br><br>
<p>
  <strong>Signature of Lab Instructor(s)</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Signature of HOD</strong>
</p>

</div>

<br><br>
<hr>

<h2 align="center">SELF DECLARATION</h2>
<p>We hereby declare that the project work entitled <strong>“CARDIOSAFE: HIGH-PRECISION CVD RISK ANALYZER FOR DIABETIC PATIENTS”</strong> is a genuine work carried out by us in B.Tech. (COMPUTER SCIENCE AND ENGINEERING) at SRKR Engineering College(A), Bhimavaram and has not been submitted either in part or full for the award of any other degree or diploma in any other institute or University.</p>

<br>
<p>
  <strong>(Student Names, Reg Nos, and Signatures)</strong>
</p>

<br><br>
<hr>

<h2 align="center">ABSTRACT</h2>
<div style="font-size: 12pt; line-height: 1.5;">
<p>Cardiovascular Disease (CVD) is a leading cause of mortality globally, with diabetic patients facing significantly elevated risks. This project, CardioSafe, presents a specialized machine learning solution designed to predict 10-year CVD risk specifically for the diabetic demographic. Utilizing the Framingham Heart Study dataset, we engineered a pipeline focused on high precision (>88%) to ensure clinical reliability and minimize false positives. The system employs an XGBoost Classifier and addresses the "black-box" nature of AI by integrating SHAP (SHapley Additive exPlanations). This allows for real-time, transparent clinical reasoning, visualized through a Streamlit web interface.</p>
</div>

<br><br>
<hr>

<h2 align="center">TABLE OF CONTENTS</h2>

| SI. No. | CONTENTS | Page No. |
| :---: | :--- | :---: |
| | ABSTRACT | i |
| | LIST OF TABLES | ii |
| | LIST OF FIGURES | iii |
| **1** | **INTRODUCTION** | **1** |
| **2** | **PROBLEM STATEMENT** | **3** |
| **3** | **LITERATURE SURVEY** | **5** |
| **4** | **SOFTWARE REQUIREMENTS SPECIFICATIONS** | **7** |
| 4.1 | OBJECTIVES | 7 |
| 4.2 | EXISTING SYSTEM | 8 |
| 4.3 | PROPOSED SYSTEM | 8 |
| 4.4 | REQUIREMENTS | 9 |
| **5** | **SYSTEM ANALYSIS & DESIGN** | **10** |
| 5.1 | SYSTEM ARCHITECTURE | 10 |
| 5.2 | DATA PREPROCESSING | 11 |
| **6** | **IMPLEMENTATION** | **13** |
| 6.1 | TECHNOLOGIES USED | 13 |
| 6.2 | MODULES DESCRIPTION | 14 |
| **7** | **RESULT ANALYSIS** | **15** |
| **8** | **CONCLUSION & FUTURE SCOPE** | **17** |
| **9** | **REFERENCES** | **18** |
| **10** | **APPENDIX** | **19** |

<br><br>
<hr>

<h2 align="center">LIST OF TABLES & FIGURES</h2>
<p><strong>List of Tables:</strong></p>
<ul>
    <li><strong>Table 4.1:</strong> Software Requirements (Page 9)</li>
    <li><strong>Table 7.1:</strong> Confusion Matrix for Test Data (Page 15)</li>
</ul>
<p><strong>List of Figures:</strong></p>
<ul>
    <li><strong>Figure 5.1:</strong> CardioSafe Architecture Overview (Page 10)</li>
    <li><strong>Figure 7.1:</strong> SHAP Global Summary Plot (Page 15)</li>
    <li><strong>Figure 7.2:</strong> SHAP Individual Patient Waterfall Plot (Page 16)</li>
</ul>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 1: INTRODUCTION</h1>
<p>Cardiovascular Disease (CVD) is one of the most severe health hazards globally. The risks are magnified when pre-existing conditions, particularly diabetes, are present. Early identification and targeted intervention are crucial to improve patient outcomes and minimize long-term medical complications.</p>
<p>CardioSafe represents an advanced predictive layer that assesses the probability of a diabetic patient experiencing coronary heart disease within the next ten years. It leverages historical patient-level features such as age, systolic/diastolic blood pressure, BMI, and glucose levels to feed an Extreme Gradient Boosting (XGBoost) model.</p>
<p>A primary goal of CardioSafe is clinical trustworthiness. Therefore, it specifically optimizes for high precision, severely limiting false positive identification. To transition from a conventional "black box" model to a "white box" approach, the solution natively incorporates SHAP (SHapley Additive exPlanations) values to clarify which specific biological factors impacted the predictive score.</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 2: PROBLEM STATEMENT</h1>
<p>Diabetic patients have unique cardiovascular risk profiles that typical population-level calculators may not accurately assess or heavily weight. Misdiagnosis or false positive predictions can lead to unnecessary, stressful diagnostic interventions and medication overtreatment.</p>
<p>Thus, the core problem is twofold: First, to develop a specialized model that accurately captures the CVD risks specific purely to diabetics. Second, to prevent medical diagnostic mistrust by ensuring transparent explainability (interpreting why the AI made a specific decision) and achieving a high-precision prediction rate to reduce false positive alarm fatigue among clinical practitioners.</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 3: LITERATURE SURVEY</h1>
<p>Historically, predicting CVD involved general statistical approaches like the Framingham Risk Score. However, these tools lacked specific adaptions for diabetic cohorts and did not model non-linear interactions efficiently. More recently, machine learning strategies have shown significant improvements over basic regression techniques.</p>
<p>Recent studies in healthcare analytics actively recommend ensemble methods—like Random Forest and Gradient Boosting—derived from tabular risk factors. However, the inability of modern clinicians to trace the "why" behind boosted models has been a barrier to entry. The introduction of Lundberg and Lee’s SHAP formulation (2017) revolutionized model interpretability by allocating individualized contribution values to each patient's features, bridging the complex performance with essential clinical transparency.</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 4: SOFTWARE REQUIREMENTS SPECIFICATIONS</h1>

<h3>4.1 OBJECTIVES</h3>
<p>To develop a high-precision diagnostic aid that utilizes "White Box" AI (SHAP) to provide interpretable 10-year CVD risk assessments for diabetic patients.</p>

<h3>4.2 EXISTING SYSTEM</h3>
<p>Traditional diagnostic evaluations rely extensively on generalized statistical formulas (e.g., standard Framingham equations) or manually applied physician heuristics. These approaches struggle to compute multi-variable non-linear correlations internally and often provide black-box generalizations without dynamic patient-specific visual explanations.</p>

<h3>4.3 PROPOSED SYSTEM</h3>
<p>The proposed system is an XGBoost-powered web application using interactive Streamlit UI. It implements precise data preprocessing using median imputation and standard scaling to secure uniform data structures. The system actively visualizes patient risk probabilities along with interactive, mathematically sound clinical reasoning via SHAP force and waterfall plots.</p>

<h3>4.4 REQUIREMENTS</h3>
<h4>4.4.1 SOFTWARE REQUIREMENTS</h4>
<ul>
    <li><strong>Language:</strong> Python 3.8+.</li>
    <li><strong>Libraries:</strong> XGBoost, Scikit-Learn, SHAP, Pandas.</li>
    <li><strong>Deployment:</strong> Streamlit Framework.</li>
</ul>

<h4>4.4.2 HARDWARE REQUIREMENTS</h4>
<ul>
    <li><strong>Processor:</strong> Standard multi-core CPU (Intel i3/i5/i7 or equivalent AMD).</li>
    <li><strong>RAM:</strong> Minimum 4 GB (8 GB Recommended).</li>
    <li><strong>Storage:</strong> Minimum 1 GB of free disk space.</li>
</ul>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 5: SYSTEM ANALYSIS & DESIGN</h1>

<h3>5.1 SYSTEM ARCHITECTURE</h3>
<p>The system is split into an offline model-training pipeline and a real-time web interface. The offline pipeline filters the original Framingham dataset to diabetic patients, generates median imputers and scalers to combat missing/skewed data, trains the initial XGBClassifier, calculates the custom probability threshold to maximize precision, and serializes the pipeline variables as <code>.joblib</code> files.</p>
<p>During inference, the web application retrieves patient vitals from a localized UI side-bar menu, channels the input across the pre-trained scalers, renders a categorical probability prediction against the high-precision target, and leverages <code>shap.TreeExplainer</code> to plot the logic graphically.</p>

<h3>5.2 DATA PREPROCESSING</h3>
<p>Dataset refinement focuses exclusively on the diabetic segment (where 'diabetes' == 1). All missing physiological measurements (such as Glucose and BMI) are subjected to <strong>Median Imputation</strong> because it handles health outliers efficiently. Immediately following, a <strong>StandardScaler</strong> enforces a standard distribution shape on continuous numerical features, crucial for algorithms like XGBoost to assess weight uniformly across contrasting scales (e.g., Age ~ 50 vs. Cholesterol ~ 200).</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 6: IMPLEMENTATION</h1>

<h3>6.1 TECHNOLOGIES USED</h3>
<p>We utilize XGBoost (Extreme Gradient Boosting) for its ability to handle complex medical tabular data. SHAP values, derived from coalitional game theory, are used to explain individual feature contributions.<br>Streamlit is implemented as an agile web-presentation framework allowing dynamic re-rendering of the inputs and complex matplotlib/JS graphics seamlessly in a standard browser environment. Scikit-learn oversees the data management architecture, from the train-test split mechanism emphasizing stratified targets to pipeline transformations.</p>

<h3>6.2 MODULES DESCRIPTION</h3>
<ul>
    <li><strong>Data Preparation Module:</strong> Integrates dataset loading, diabetic-exclusive filtering, and target variable separation (TenYearCHD).</li>
    <li><strong>Training & Evaluation Module:</strong> Employs XGBoost with tree-depth controls (max_depth=4). Computes raw probabilities instead of direct inferences to locate precision maximization thresholds.</li>
    <li><strong>Interface Module:</strong> Streamlines user inputs via numeric entries and sliders configured with scientifically reasonable thresholds. Dynamically colors the final output interface based on risk evaluation metrics.</li>
</ul>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 7: RESULT ANALYSIS</h1>
<p>The model is tuned with a custom probability threshold ($0.75+$) to achieve a precision score exceeding 88%, ensuring high confidence in "High Risk" classifications. Testing on unseen clinical permutations validates the structure guarantees low generalized false positive rates. Global Summary Plots (SHAP) conclusively indicate variables such as extremely high Systolic Blood Pressure and Age as primary instigators for ten-year risks in this demographic. Single patient inferences output localized factorizations dictating exact personal vulnerabilities.</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 8: CONCLUSION & FUTURE SCOPE</h1>
<p><strong>Conclusion:</strong> CardioSafe effectively builds a clinical-grade, transparent diagnostic methodology. By intentionally maximizing Precision logic around diabetic patient pools and embedding AI interpretability natively inside the presentation view, the model mitigates alarm fatigue and facilitates trustworthy medical utility.<br><br>
<strong>Future Scope:</strong> Subsequent expansions should introduce federated learning integrations across a wider array of international cardiopulmonary datasets to battle inherent regional biases. Furthermore, importing Electronic Health Record (EHR) APIs dynamically rather than requiring manual input could significantly automate diagnostic workloads for clinicians.</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 9: REFERENCES</h1>
<p>
    Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <em>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.</em>
<br><br>
    Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. <em>Advances in Neural Information Processing Systems.</em>
<br><br>
    Mahmood, S. S., Levy, D., Vasan, R. S., & Wang, T. J. (2014). The Framingham Heart Study and the epidemiology of cardiovascular disease: a historical perspective. <em>The Lancet</em>, 383(9921), 999-1008.
</p>

<br><br>
<hr style="page-break-after: always;">

<h1>CHAPTER 10: APPENDIX</h1>
<p><strong>Code References & Links:</strong></p>
<ul>
    <li>Main Python Entry Points: <code>cvd_prediction_diabetics.py</code>, <code>app.py</code></li>
    <li>Data Dictionary used from Framingham Study (Variables: age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose, etc.)</li>
</ul>
