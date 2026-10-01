# ⚡ Supercapacitor Capacitance Predictor

A Streamlit machine-learning application for predicting the **specific capacitance of porous carbon materials for supercapacitor electrodes**.

## 🔬 Project Overview

The project follows:

**Material Properties → Machine Learning Model → Specific Capacitance**

The deployed model is a **Gradient Boosting Regressor**.

## 🧠 Model Inputs

| Dataset feature | Application label |
|---|---|
| `BET_surface_area` | Specific Surface Area (BET) |
| `avg_pore_diameter_caculated` | Average Pore Diameter |
| `total_pore_volume` | Total Pore Volume |
| `element_N` | Nitrogen Content |
| `voltage_span` | Potential Window |

**Target:** `capacitance_value` — displayed as Predicted Specific Capacitance (F/g).

## 📊 Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 43.5084 | 60.0460 | 0.6941 |
| Gradient Boosting | 42.1474 | 58.9057 | 0.7056 |
| XGBoost | 42.5700 | 59.4130 | 0.7005 |

The deployed application uses the Gradient Boosting model from this comparison.

## 📁 Repository Structure

```text
supercapacitor-/
├── app.py
├── supercapacitor_capacitance_model.pkl
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

```bash
git clone https://github.com/AshiqMuhammad/supercapacitor-.git
cd supercapacitor-
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deployment

For Streamlit deployment, keep `app.py`, `requirements.txt`, and
`supercapacitor_capacitance_model.pkl` in the repository root.

The model filename must be exactly:

```text
supercapacitor_capacitance_model.pkl
```

## ⚙️ Model Environment

The model was saved with:

```text
Python:       3.13.15
scikit-learn: 1.6.1
NumPy:        2.1.3
SciPy:        1.16.3
Joblib:       1.6.0
```

These versions are recorded because serialized scikit-learn models can
depend on their Python/package environment.

## 📚 Research Context

The project applies machine learning to a supercapacitor/carbon-electrode
materials-science problem. The implementation uses five available
features from the selected dataset; it should not be described as an
exact reproduction of every feature used in the referenced thesis.

## ⚠️ Limitation

This application is an ML prediction and screening tool. Predictions do
not replace material synthesis, electrode fabrication, electrochemical
testing, characterization, or experimental validation.

## 🛠️ Technologies

Python • Pandas • NumPy • SciPy • Scikit-learn • Joblib • Streamlit

**Model:** Gradient Boosting Regression

---

Built as an academic machine-learning project combining **Machine Learning,
Materials Science, and Energy Storage**.
