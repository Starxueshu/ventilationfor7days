# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("An artificial intelligence model to predict prolonged dependence on mechanical ventilation among patients with critical orthopaedic trauma: a large observational cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Spinefracture = st.sidebar.selectbox("Spine fracture", ("No", "Yes"))
Lowlimbfracture = st.sidebar.selectbox("Low limb fracture", ("No", "Yes"))
Obesity = st.sidebar.selectbox("Obesity", ("No", "Yes"))
Vasopressors = st.sidebar.selectbox("Vasopressors", ("No", "Yes"))
Glucose = st.sidebar.slider("Glucose (mg/dL)", 100, 260)
pO2 = st.sidebar.slider("pO2 (mmHg)", 45, 360)
pCO2 = st.sidebar.slider("pCO2 (mmHg)", 20, 160)
pH = st.sidebar.slider("pH (unit)", 7.00, 7.60)
Heartrate = st.sidebar.slider("Heart rate (BPM)", 50, 125)
Resprate = st.sidebar.slider("Respiratory rate (BPM)", 8, 30)
SAPSII = st.sidebar.slider("SAPSII", 10, 50)
SOFA = st.sidebar.slider("SOFA", 0, 12)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round-web.pkl")
    x = pd.DataFrame([[Spinefracture,Lowlimbfracture,Obesity,Vasopressors,Glucose,pO2,pCO2,pH,Heartrate,Resprate,SAPSII,SOFA]],
                     columns=["Spinefracture","Lowlimbfracture","Obesity","Vasopressors","Glucose","pO2","pCO2","pH","Heartrate","Resprate","SAPSII","SOFA"])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Risk of prolonged dependence on mechanical ventilation: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.618:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")
    if prediction < 0.618:
        st.success(f"In low-risk cases, proactive measures may include regular assessment and monitoring to ensure early detection of any signs of respiratory improvement, with a focus on gradually reducing ventilator support and transitioning to less invasive forms of respiratory assistance. Furthermore, physical therapy and mobilization may be initiated to prevent muscle weakness and deconditioning associated with prolonged bed rest.")
    else:
        st.error(f"Conversely, high-risk patients may require more aggressive interventions, including close monitoring and frequent assessments to identify and address any potential complications promptly. Additionally, advanced ventilator management strategies, such as lung-protective ventilation techniques, prone positioning, and consideration of extracorporeal membrane oxygenation (ECMO) in refractory cases, may be warranted. Moreover, close collaboration between multidisciplinary teams, including critical care specialists, respiratory therapists, and physical therapists, is essential to tailor individualized management plans and provide comprehensive support for high-risk patients.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_trainy = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["Spinefracture","Lowlimbfracture","Obesity","Vasopressors","Glucose","pO2","pCO2","pH","Heartrate","Resprate","SAPSII","SOFA"]]
    y_train = y_trainy.Ventilation7days
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    #st.text(shap_value)

    shap.initjs()
    #image = shap.plots.force(shap_value)
    #image = shap.plots.bar(shap_value)

    shap.plots.waterfall(shap_value[0])
    st.pyplot(bbox_inches='tight')
    st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Model information')
st.markdown('The AI application was deployed based on the eXGBM model, which demonstrated superior performance compared to other models. It achieved the highest recall (0.892) and exhibited excellent results in terms of Brier score (0.088), log loss (0.291), and calibration slope (0.999). Additionally, the model ranked second in terms of area under the curve value (0.949, 95%: 0.933-0.961), accuracy (0.871), F1 score (0.873), and discrimination slope (0.647). However, it is important to note that patient treatment should not be solely based on the AI platform, and the model’s predictions should be considered in conjunction with clinical expertise and other relevant factors.')
