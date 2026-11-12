import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(page_title="Algorithmic Reliability & FMEA Terminal", layout="wide")
st.title("Algorithmic Reliability & FMEA Terminal")
st.markdown("Stress-testing AI models under simulated systemic market volatility.")

# 2. Data Loading, Preprocessing & Model Training (Cached in memory)
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("corporateCreditRatingWithFinancialRatios.csv")
        y = df["Binary Rating"]

        cols_to_drop = [
            "Rating Agency", "Corporation", "Rating", "Rating Date",
            "CIK", "SIC Code", "Sector", "Ticker",
            "Binary Rating", "Name", "Symbol", "Rating Agency Name", "Date"
        ]
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        X = X.select_dtypes(include=["number"]).fillna(0)

    except FileNotFoundError:
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(2500, 16), columns=[f"Feature_{i}" for i in range(16)])
        y = np.where(X["Feature_0"] + X["Feature_1"] > 0, 1, 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Tuned Random Forest (single-threaded to stay well within Streamlit Cloud's 1GB RAM)
    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1
    )
    best_rf.fit(X_train, y_train)

    # Pre-calculate degradation trajectory across 0.00 to 3.00 sigma
    curve_data = []
    for n in np.arange(0.0, 3.05, 0.05):
        np.random.seed(42)
        noisy_X = X_test + np.random.normal(0, n, X_test.shape)
        preds = best_rf.predict(noisy_X)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        type1_error = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
        type2_error = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
        acc = accuracy_score(y_test, preds) * 100

        curve_data.append({
            "Noise": round(float(n), 2),
            "Accuracy": acc,
            "Type I Error": type1_error,
            "Type II Error": type2_error
        })

    return X_test, y_test, best_rf, pd.DataFrame(curve_data)

X_test, y_test, rf_model, df_curve = load_and_train()

# 3. Wide Slider Control
st.markdown("### Market Volatility Simulation")
noise_level = st.slider(
    "Inject Gaussian Noise (σ)",
    min_value=0.00,
    max_value=3.00,
    value=0.00,
    step=0.05,
    help="Drag this slider to simulate increasing levels of systemic market volatility."
)

# 4. Evaluate Current Slider Value
np.random.seed(42)
X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
predictions = rf_model.predict(X_test_noisy)
cm_current = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm_current.ravel()
current_type1 = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
current_type2 = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
current_acc = accuracy_score(y_test, predictions) * 100

# 5. Algorithmic Circuit Breaker Logic
st.divider()
if current_type2 > 20.0:
    st.error(
        f"🚨 **CIRCUIT BREAKER TRIGGERED:** Type II False Alarms breached the 20% safety threshold at {noise_level:.2f}σ market volatility."
    )
else:
    st.success(
        f"✅ **SYSTEM STABLE:** Error rates within normal operating limits at {noise_level:.2f}σ."
    )

# 6. Dashboard Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Optimized Model Accuracy", f"{current_acc:.1f}%")
col2.metric("Type I Error (Missed Defaults)", f"{current_type1:.1f}%")
col3.metric("Type II Error (False Alarms)", f"{current_type2:.1f}%")

st.divider()

# 7. Interactive Plotly Visualizations
col_chart, col_matrix = st.columns([2, 1])

with col_chart:
    st.subheader("Failure Rate Trajectory")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_curve["Noise"], y=df_curve["Type II Error"],
        mode="lines", name="Type II (False Alarms)",
        line=dict(color="#e74c3c", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["Noise"], y=df_curve["Type I Error"],
        mode="lines", name="Type I (Missed Defaults)",
        line=dict(color="#2980b9", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["Noise"], y=df_curve["Accuracy"],
        mode="lines", name="Accuracy (%)",
        line=dict(color="#27ae60", width=2, dash="dot")
    ))

    fig.add_hline(
        y=20, line_width=2, line_dash="dash", line_color="gray",
        annotation_text="20% Failure Threshold", annotation_position="bottom right"
    )
    fig.add_vline(
        x=noise_level, line_width=2, line_color="black",
        annotation_text=f"Current: {noise_level:.2f}σ", annotation_position="top right"
    )

    fig.update_layout(
        xaxis_title="Gaussian Noise (Standard Deviations)",
        yaxis_title="Percentage (%)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_matrix:
    st.subheader("Confusion Matrix")

    fig2 = px.imshow(
        cm_current,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Status", y="Actual Status"),
        x=["Distressed (0)", "Safe (1)"],
        y=["Distressed (0)", "Safe (1)"]
    )
    fig2.update_xaxes(side="bottom")
    fig2.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)