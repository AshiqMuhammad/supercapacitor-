import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Supercapacitor Capacitance Predictor",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background:#f6f8fb; }
.block-container { max-width:1200px; padding-top:2rem; }
.hero {
    background:linear-gradient(135deg,#111827,#334155);
    padding:2.2rem; border-radius:20px; margin-bottom:1.5rem;
    box-shadow:0 10px 30px rgba(15,23,42,.12);
}
.hero-kicker { color:#93c5fd; font-weight:700; letter-spacing:.12em; font-size:.8rem; }
.hero h1 { color:white; margin:.35rem 0; font-size:2.35rem; }
.hero p { color:#dbeafe; font-size:1rem; max-width:850px; }
.card {
    background:white; border:1px solid #e5e7eb; border-radius:16px;
    padding:1.25rem; box-shadow:0 4px 16px rgba(15,23,42,.05);
}
.prediction {
    background:linear-gradient(135deg,#fff,#eff6ff);
    border:1px solid #bfdbfe; border-radius:18px; padding:1.6rem;
    text-align:center; margin-top:1rem;
}
.pred-label { color:#475569; font-weight:650; }
.pred-value { color:#0f172a; font-size:2.6rem; font-weight:800; }
.pred-unit { color:#64748b; }
.info {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px;
    padding:1rem; color:#334155; line-height:1.5;
}
.footer { text-align:center; color:#64748b; font-size:.82rem;
          padding-top:1.5rem; margin-top:2rem; border-top:1px solid #e5e7eb; }
div[data-testid="stMetric"] { background:white; border:1px solid #e5e7eb;
                               padding:1rem; border-radius:14px; }
.stButton > button { border-radius:10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "supercapacitor_capacitance_model.pkl"
    if not path.exists():
        st.error("Model file not found: supercapacitor_capacitance_model.pkl")
        st.stop()
    package = joblib.load(path)
    return package["model"], package["features"]

model, features = load_model()

st.markdown("""
<div class="hero">
<div class="hero-kicker">MACHINE LEARNING • ENERGY STORAGE</div>
<h1>Supercapacitor Capacitance Predictor</h1>
<p>Predict the specific capacitance of a porous carbon electrode from
selected material and electrochemical properties using a trained
Gradient Boosting regression model.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚡ Project")
    st.caption("ML-based prediction of supercapacitor electrode performance")
    st.divider()
    st.markdown("### Model")
    st.write("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Inputs:** 5 material properties")
    st.write("**Output:** Specific capacitance")
    st.divider()
    st.markdown("### Test results")
    st.metric("R²", "0.706")
    st.metric("RMSE", "58.91 F/g")
    st.metric("MAE", "42.15 F/g")

st.markdown("### Enter material properties")
st.markdown("""
<div class="info">Enter the five properties below. The model uses them
to estimate specific capacitance.</div>
""", unsafe_allow_html=True)
st.write("")

c1, c2 = st.columns(2)

with c1:
    surface_area = st.number_input("Specific Surface Area (BET)", min_value=0.0,
                                   value=1000.0, step=10.0)
    pore_diameter = st.number_input("Average Pore Diameter", min_value=0.0,
                                    value=5.0, step=0.1)
    pore_volume = st.number_input("Total Pore Volume", min_value=0.0,
                                 value=0.50, step=0.01)

with c2:
    nitrogen = st.number_input("Nitrogen Content", min_value=0.0,
                               value=2.0, step=0.1)
    voltage_window = st.number_input("Potential Window", min_value=0.0,
                                     value=1.0, step=0.1)
    predict = st.button("⚡ Predict Specific Capacitance",
                        type="primary", use_container_width=True)

if predict:
    data = pd.DataFrame([[
        surface_area, pore_diameter, pore_volume, nitrogen, voltage_window
    ]], columns=features)
    try:
        prediction = model.predict(data)[0]
        st.markdown(f"""
        <div class="prediction">
        <div class="pred-label">Predicted Specific Capacitance</div>
        <div class="pred-value">{prediction:,.2f}</div>
        <div class="pred-unit">F/g</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Prediction generated successfully.")
        with st.expander("View input values"):
            st.dataframe(pd.DataFrame({
                "Feature": ["Specific Surface Area (BET)",
                            "Average Pore Diameter", "Total Pore Volume",
                            "Nitrogen Content", "Potential Window"],
                "Value": [surface_area, pore_diameter, pore_volume,
                          nitrogen, voltage_window]
            }), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
tab1, tab2, tab3 = st.tabs(["📘 About", "🧠 How It Works", "⚠️ Important Note"])

with tab1:
    st.markdown("""
### About the Project
This application demonstrates a machine-learning workflow for predicting
the specific capacitance of porous carbon materials considered for
supercapacitor electrodes.

The deployed Gradient Boosting model uses:
- Specific Surface Area (BET)
- Average Pore Diameter
- Total Pore Volume
- Nitrogen Content
- Potential Window
""")

with tab2:
    st.markdown("""
### Prediction Workflow
**Material Properties → Gradient Boosting ML Model → Specific Capacitance**

The model learns relationships between selected material properties and
experimentally reported capacitance values.
""")

with tab3:
    st.markdown("""
### Important
This is an ML prediction and screening tool. A predicted value does not
replace experimental fabrication, electrochemical testing, or validation.
Predictions are most meaningful for inputs reasonably represented by the
training data.
""")

st.markdown("""
<div class="footer">
Supercapacitor Capacitance Predictor • Machine Learning for Energy Storage
</div>
""", unsafe_allow_html=True)
