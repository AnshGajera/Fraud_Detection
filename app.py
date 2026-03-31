# ── app.py — Streamlit Fraud Detection App ───────────────────────────────────

import streamlit as st
import numpy as np
import pickle

# ── load saved model and scaler ──────────────────────────────────────────────
model  = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="centered")

st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it is **Legitimate** or **Fraudulent**.")
st.divider()

# ── input section ─────────────────────────────────────────────────────────────
st.subheader("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
with col2:
    time   = st.number_input("Time (seconds since first txn)", min_value=0.0, value=0.0, step=1.0)

st.markdown("**PCA Feature Values (V1 to V28)**")
st.caption("Default is 0.0 — adjust only if you have actual PCA values")

v_cols  = st.columns(4)
v_vals  = []
for i in range(1, 29):
    col_idx = (i - 1) % 4
    with v_cols[col_idx]:
        val = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")
        v_vals.append(val)

st.divider()

# ── prediction ────────────────────────────────────────────────────────────────
if st.button("🔎 Predict Transaction", use_container_width=True):

    # build input array — order: Time, V1-V28, Amount
    raw_input = np.array([[time] + v_vals + [amount]])

    # scale Time and Amount (columns 0 and 29)
    raw_input[:, [0, 29]] = scaler.transform(raw_input[:, [0, 29]])

    prediction   = model.predict(raw_input)[0]
    probability  = model.predict_proba(raw_input)[0]

    fraud_prob   = round(float(probability[1]) * 100, 2)
    legit_prob   = round(float(probability[0]) * 100, 2)

    st.divider()
    if prediction == 1:
        st.error(f"🚨 FRAUDULENT Transaction Detected")
        st.metric("Fraud Probability",    f"{fraud_prob}%")
        st.metric("Legitimate Probability", f"{legit_prob}%")
    else:
        st.success(f"✅ Legitimate Transaction")
        st.metric("Legitimate Probability", f"{legit_prob}%")
        st.metric("Fraud Probability",      f"{fraud_prob}%")