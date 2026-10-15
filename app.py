
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "disease_model.joblib"
SYMPTOMS_PATH = BASE_DIR / "models" / "symptoms.json"

st.set_page_config(
    page_title="MediPredict | Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.hero {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(120deg, #111c33, #172554);
    border: 1px solid #2b3c5c;
}
.hero h1 {
    color: #f8fafc;
}
.hero p {
    color: #a8b7d0;
}
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #273449;
    border-radius: 12px;
    padding: 15px;
}
.stButton button {
    background: #38bdf8;
    color: #07111f;
    border-radius: 9px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🩺 MediPredict</h1>
    <p>Disease Prediction Using Machine Learning</p>
    <p>Interactive dark dashboard</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational demonstration only. This synthetic-data model "
    "cannot diagnose disease or replace a healthcare professional."
)

if not MODEL_PATH.exists() or not SYMPTOMS_PATH.exists():
    st.error("Model files are missing. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(SYMPTOMS_PATH, encoding="utf-8") as file:
    symptoms = json.load(file)

st.subheader("Select your symptoms")

selected_symptoms = st.multiselect(
    "Symptoms",
    symptoms,
    format_func=lambda x: x.replace("_", " ").title(),
    placeholder="Search and select symptoms"
)

st.caption("Age is collected for demonstration and is not used by this model.")
age = st.slider("Age", 1, 100, 25)

duration = st.selectbox(
    "Duration of symptoms",
    [
        "Less than 24 hours",
        "1–3 days",
        "4–7 days",
        "More than a week"
    ]
)

if st.button("Analyze Symptoms", use_container_width=True):
    if not selected_symptoms:
        st.info("Please select at least one symptom.")
    else:
        input_data = pd.DataFrame([
            {
                symptom: int(symptom in selected_symptoms)
                for symptom in symptoms
            }
        ])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        ranking = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True
        )[:3]

        st.divider()
        st.subheader("Demo Prediction Result")

        st.markdown(
            f"""
            <div style="
                background:#111827;
                padding:24px;
                border-radius:15px;
                border:1px solid #273449;">
                <p style="color:#9ca3af;">Top model class</p>
                <h2 style="color:#38bdf8;">{prediction}</h2>
                <p style="color:#9ca3af;">
                    This is an educational model output,
                    not a confirmed diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Demo class": [name for name, _ in ranking],
            "Model score (%)": [
                round(float(score) * 100, 1)
                for _, score in ranking
            ]
        })

        st.subheader("Top 3 model scores")
        st.bar_chart(
            chart_data.set_index("Demo class"),
            color="#38bdf8"
        )

        st.subheader("Selected symptoms")
        st.write(
            ", ".join(
                symptom.replace("_", " ").title()
                for symptom in selected_symptoms
            )
        )

        if (
            "breathlessness" in selected_symptoms
            or "chest_discomfort" in selected_symptoms
        ):
            st.error(
                "Severe breathing difficulty or chest pain can be "
                "an emergency. Seek urgent medical care."
            )

        st.info(
            "Model scores are based on synthetic data and are not "
            "real-world disease probabilities. Consult a qualified "
            "healthcare professional for medical advice."
        )

else:
    st.divider()
    st.subheader("Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Symptom features", len(symptoms))
    col2.metric("Algorithm", "Random Forest")
    col3.metric("Interface", "Dark Mode")

st.divider()
st.caption(
    "MediPredict | Python • Streamlit • Pandas • Scikit-learn"
)
