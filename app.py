from pathlib import Path
import pickle

import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    with SCALER_PATH.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="F",
    layout="centered",
)

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error("Model files could not be loaded. Check the deployment dependencies and saved artifacts.")
    st.exception(exc)
    st.stop()


st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it is legitimate or fraudulent.")
st.divider()

st.subheader("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
with col2:
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0, step=1.0)

st.markdown("**PCA Feature Values (V1 to V28)**")
st.caption("Keep the default 0.0 values unless you have real PCA-transformed inputs.")

v_columns = st.columns(4)
v_values = []
for i in range(1, 29):
    column_index = (i - 1) % 4
    with v_columns[column_index]:
        value = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")
        v_values.append(value)

st.divider()

if st.button("Predict Transaction", use_container_width=True):
    raw_input = np.array([[time] + v_values + [amount]], dtype=float)

    # Scale Time and Amount only.
    raw_input[:, [0, 29]] = scaler.transform(raw_input[:, [0, 29]])

    prediction = int(model.predict(raw_input)[0])
    probabilities = model.predict_proba(raw_input)[0]

    fraud_probability = round(float(probabilities[1]) * 100, 2)
    legitimate_probability = round(float(probabilities[0]) * 100, 2)

    st.divider()
    if prediction == 1:
        st.error("Fraudulent transaction detected")
        st.metric("Fraud Probability", f"{fraud_probability}%")
        st.metric("Legitimate Probability", f"{legitimate_probability}%")
    else:
        st.success("Legitimate transaction")
        st.metric("Legitimate Probability", f"{legitimate_probability}%")
        st.metric("Fraud Probability", f"{fraud_probability}%")
