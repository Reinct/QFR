"""
streamlit_app.py — ISR Risk Predictor
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="ISR Risk Predictor", layout="wide")

@st.cache_resource
def load_model():
    import urllib.request
    path = os.path.join(os.path.dirname(__file__), "streamlit_model.joblib")
    if not os.path.exists(path):
        url = "https://github.com/Reinct/QFR/releases/latest/download/streamlit_model.joblib"
        st.info(f"Downloading model (~105 MB)...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            st.error(f"Download failed: {e}. Make sure the Release exists with the model file.")
            st.stop()
    return joblib.load(path)

st.title("ISR Risk Predictor — QFR/RWS")
st.markdown("*In-Stent Restenosis prediction based on QFR and RWS biomechanical parameters*")

data = load_model()
model = data["model"]
means = data["scaler_means"]
stds = data["scaler_stds"]
sel_names = data["sel_names"]
sel_vars = data["sel_vars"]
n_patients = len(data["y_ref"])
n_events = int(data["y_ref"]["retenosis"].sum())

st.sidebar.header("Patient Parameters")
st.sidebar.caption(f"Trained on {n_patients} patients ({n_events} ISR events)")
st.sidebar.caption(f"{len(sel_vars)} features selected by LASSO")

inputs = {}
cols = st.sidebar.columns(2)
for i, v in enumerate(sel_vars):
    with cols[i % 2]:
        if v in means:
            mu = means[v]; sg = max(stds[v], 1e-8)
            inputs[v] = st.number_input(v, value=float(round(mu, 2)),
                                        step=float(max(0.01, round(sg/5, 2))),
                                        format="%.3f", key=v)
        else:
            inputs[v] = st.selectbox(v, [0, 1], key=v)


def build_input():
    x = np.zeros(len(sel_names))
    for j, sn in enumerate(sel_names):
        if sn in inputs:
            val = inputs[sn]
            if sn in means:
                val = (val - means[sn]) / stds[sn]
            x[j] = val
        else:
            for inp_name, inp_val in inputs.items():
                if sn.startswith(inp_name + "_"):
                    parts = sn.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        if int(parts[1]) == int(inp_val):
                            x[j] = 1.0
                    break
    return x


if st.sidebar.button("Predict ISR Risk", type="primary", use_container_width=True):
    x_input = build_input()
    # 用 survival function 计算 2 年事件概率
    surv = model.predict_survival_function(x_input.reshape(1, -1), return_array=True)[0]
    t2 = min(730, model.unique_times_[-1]) if hasattr(model, 'unique_times_') else 730
    k2 = np.argmin(np.abs(model.unique_times_ - t2)) if hasattr(model, 'unique_times_') else len(surv) - 1
    prob = 1.0 - float(surv[k2])
    prob_pct = prob * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("ISR Probability (2yr)", f"{prob_pct:.1f}%")
    level = "Low Risk" if prob_pct < 20 else ("Moderate Risk" if prob_pct < 40 else "High Risk")
    col2.metric("Risk Level", level)
    col3.metric("Predicted at", "2 years")

    st.subheader("Feature Contributions (SHAP)")
    try:
        X_ref = data["X_ref"][:min(50, data["X_ref"].shape[0])]
        explainer = shap.KernelExplainer(model.predict, X_ref)
        sv = explainer.shap_values(x_input.reshape(1, -1), nsamples=200, silent=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(values=sv[0], base_values=float(explainer.expected_value),
                             feature_names=sel_names), show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP unavailable: {e}")

with st.expander("Model Info"):
    st.write(f"**Model:** Gradient Boosting Survival Analysis (GBSA)")
    st.write(f"**Training data:** {n_patients} PCI patients, {n_events} ISR events")
    st.write(f"**Features:** {sel_vars}")
    st.write(f"**Validation:** .632+ Bootstrap (100 iterations)")
