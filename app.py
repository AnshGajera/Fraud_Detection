from pathlib import Path
import pickle

import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_PROFILE = {feature: 0.0 for feature in FEATURE_ORDER}
DEFAULT_PROFILE["Amount"] = 100.0

LEGITIMATE_SAMPLE = {
    "Time": 0.0,
    "V1": -1.3598071337,
    "V2": -0.0727811733,
    "V3": 2.5363467380,
    "V4": 1.3781552243,
    "V5": -0.3383207699,
    "V6": 0.4623877778,
    "V7": 0.2395985541,
    "V8": 0.0986979013,
    "V9": 0.3637869696,
    "V10": 0.0907941720,
    "V11": -0.5515995333,
    "V12": -0.6178008558,
    "V13": -0.9913898472,
    "V14": -0.3111693537,
    "V15": 1.4681769721,
    "V16": -0.4704005253,
    "V17": 0.2079712419,
    "V18": 0.0257905802,
    "V19": 0.4039929603,
    "V20": 0.2514120982,
    "V21": -0.0183067779,
    "V22": 0.2778375756,
    "V23": -0.1104739102,
    "V24": 0.0669280749,
    "V25": 0.1285393583,
    "V26": -0.1891148439,
    "V27": 0.1335583767,
    "V28": -0.0210530535,
    "Amount": 149.62,
}

FRAUD_SAMPLE = {
    "Time": 406.0,
    "V1": -2.3122265423,
    "V2": 1.9519920106,
    "V3": -1.6098507323,
    "V4": 3.9979055875,
    "V5": -0.5221878647,
    "V6": -1.4265453192,
    "V7": -2.5373873062,
    "V8": 1.3916572483,
    "V9": -2.7700892772,
    "V10": -2.7722721447,
    "V11": 3.2020332071,
    "V12": -2.8999073885,
    "V13": -0.5952218813,
    "V14": -4.2892537824,
    "V15": 0.3897241203,
    "V16": -1.1407471798,
    "V17": -2.8300556745,
    "V18": -0.0168224682,
    "V19": 0.4169557050,
    "V20": 0.1269105591,
    "V21": 0.5172323709,
    "V22": -0.0350493686,
    "V23": -0.4652110762,
    "V24": 0.3201981985,
    "V25": 0.0445191675,
    "V26": 0.1778397983,
    "V27": 0.2611450026,
    "V28": -0.1432758747,
    "Amount": 0.0,
}


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    with SCALER_PATH.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


def initialize_state():
    for feature, value in DEFAULT_PROFILE.items():
        st.session_state.setdefault(feature, value)


def apply_profile(profile):
    for feature, value in profile.items():
        st.session_state[feature] = value


def get_current_input():
    values = [float(st.session_state[feature]) for feature in FEATURE_ORDER]
    return np.array([values], dtype=float)


st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="F",
    layout="centered",
)

initialize_state()

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error("Model files could not be loaded. Check the deployment dependencies and saved artifacts.")
    st.exception(exc)
    st.stop()


st.title("Credit Card Fraud Detection")
st.markdown("Use a sample profile for quick testing or open the advanced section for all PCA inputs.")
st.divider()

st.subheader("Quick Input")

quick_col1, quick_col2, quick_col3 = st.columns(3)
with quick_col1:
    if st.button("Use Legitimate Sample", use_container_width=True):
        apply_profile(LEGITIMATE_SAMPLE)
with quick_col2:
    if st.button("Use Fraud Sample", use_container_width=True):
        apply_profile(FRAUD_SAMPLE)
with quick_col3:
    if st.button("Reset Fields", use_container_width=True):
        apply_profile(DEFAULT_PROFILE)

basic_col1, basic_col2 = st.columns(2)
with basic_col1:
    st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        step=0.01,
        key="Amount",
    )
with basic_col2:
    st.number_input(
        "Time (seconds since first transaction)",
        min_value=0.0,
        step=1.0,
        key="Time",
    )

st.caption("Quick demo: click a sample button and then press Predict Transaction.")

with st.expander("Advanced PCA Inputs (Optional)"):
    st.caption("Edit V1 to V28 only if you have actual PCA-transformed feature values.")
    v_columns = st.columns(4)
    for i in range(1, 29):
        column_index = (i - 1) % 4
        with v_columns[column_index]:
            st.number_input(
                f"V{i}",
                format="%.4f",
                key=f"V{i}",
            )

st.divider()

if st.button("Predict Transaction", use_container_width=True):
    raw_input = get_current_input()
    raw_input[:, [0, 29]] = scaler.transform(raw_input[:, [0, 29]])

    prediction = int(model.predict(raw_input)[0])
    probabilities = model.predict_proba(raw_input)[0]

    fraud_probability = round(float(probabilities[1]) * 100, 2)
    legitimate_probability = round(float(probabilities[0]) * 100, 2)

    st.subheader("Prediction Result")
    result_col1, result_col2 = st.columns(2)

    if prediction == 1:
        st.error("Fraudulent transaction detected")
    else:
        st.success("Legitimate transaction")

    with result_col1:
        st.metric("Fraud Probability", f"{fraud_probability}%")
    with result_col2:
        st.metric("Legitimate Probability", f"{legitimate_probability}%")
