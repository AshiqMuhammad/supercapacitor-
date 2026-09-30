import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Supercapacitor Capacitance Predictor",
    page_icon="⚡",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #ddd;
    margin-top: 25px;
}

.result-value {
    font-size: 38px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Supercapacitor Capacitance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Specific Capacitance Prediction'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_package = joblib.load(
        "supercapacitor_capacitance_model.pkl"
    )

    model = model_package["model"]
    features = model_package["features"]

    return model, features


model, features = load_model()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Enter Material Properties")

st.write(
    "Enter the five material/electrochemical properties "
    "used by the trained machine learning model."
)


col1, col2 = st.columns(2)


with col1:

    surface_area = st.number_input(
        "Specific Surface Area (BET)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    pore_diameter = st.number_input(
        "Average Pore Diameter",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    pore_volume = st.number_input(
        "Total Pore Volume",
        min_value=0.0,
        value=0.5,
        step=0.01
    )


with col2:

    nitrogen = st.number_input(
        "Nitrogen Content",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    voltage_window = st.number_input(
        "Potential Window",
        min_value=0.0,
        value=1.0,
        step=0.1
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.write("")

if st.button(
    "Predict Specific Capacitance",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            surface_area,
            pore_diameter,
            pore_volume,
            nitrogen,
            voltage_window
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="result-box">

        <div>Predicted Specific Capacitance</div>

        <div class="result-value">
        {prediction:.2f} F/g
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

with st.expander("About this model"):

    st.write(
        """
        This application uses a trained Gradient Boosting
        regression model to predict the specific capacitance
        of carbon-based electrode materials.

        Model inputs:

        • Specific Surface Area (BET)
        • Average Pore Diameter
        • Total Pore Volume
        • Nitrogen Content
        • Potential Window

        Model output:

        • Predicted Specific Capacitance
        """
    )


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.caption(
    "Research prototype. Predictions should be interpreted "
    "within the range and conditions represented in the "
    "training data and should be experimentally validated."
)