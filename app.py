import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import RobustScaler

# 1. Page Configuration
st.set_page_config(page_title="Algorithmic Reliability & FMEA Terminal", layout="wide")
st.title("Algorithmic Reliability & FMEA Terminal")
st.markdown("Stress-testing AI models under simulated systemic market volatility.")

# Force Cache Clearing by renaming the cached function
@st.cache_resource
def load_and_train_synced_v2():
    try:
        df_raw = pd.read_csv("corporateCreditRatingWithFinancialRatios.csv")
        columns_to_drop = [
            'Rating Agency', 'Corporation', 'Rating', 'Rating Date',
            'CIK', 'SIC Code', 'Sector', 'Ticker',
            'Name', 'Symbol', 'Rating Agency Name', 'Date'
        ]
        
        # EXACT PARITY WITH COLAB: Apply dropna() on the entire dataset first
        df = df_raw.drop(columns=[c for c in columns_to_drop if c in df_raw.columns], errors='ignore').dropna()
        y = df['Binary Rating']
        X = df.drop(columns=['Binary Rating']).select_dtypes(include=['number']).fillna(0)

    except FileNotFoundError:
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(2500, 16), columns=[f"Feature_{i}" for i in range(16)])
        y = np.where(X["Feature_0"] + X["Feature_1"] > 0, 1, 0)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    np.random.seed(42)
    aug_X = [X_train]
    aug_y = [y_train]
    for s in [0.15, 0.30, 0.50, 0.75, 1.00]:
        aug_X.append(X_train + np.random.normal(0, s, X_train.shape))
        aug_y.append(y_train)

    X_train_aug = np.vstack(aug_X)
    y_train_aug = np.hstack(aug_y)

    # Use n_jobs=-1 as executed in the Colab screenshot
    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight={0: 2.3, 1: 1.0},
        random_state=42,
        n_jobs=-1
    )
    best_rf.fit(X_train_aug, y_train_aug)

    decision_threshold = 0.61
    
    # Pre-calculate degradation trajectory matching Colab loop exactly
    curve_data = []
    noise_levels = np.arange(0.0, 3.05, 0.05)
    for noise in noise_levels:
        np.random.seed(42)
        noisy_X = X_test + np.random.normal(0, noise, X_test.shape)
        
        probs = best_rf.predict_proba(noisy_X)[:, 1]
        preds = (probs >= decision_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        type1_error = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
        type2_error = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
        acc = accuracy_score(y_test, preds) * 100

        curve_data.append({
            "Noise": round(float(noise), 2),
            "Accuracy": acc,
            "Type I Error": type1_error,
            "Type II Error": type2_error
        })

    return X_test, y_test, best_rf, pd.DataFrame(curve_data), decision_threshold

X_test, y_test, rf_model, df_curve, decision_threshold = load_and_train_synced_v2()

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

# 4. Pull pre-calculated metrics matching the slider value exactly
current_metrics = df_curve[df_curve['Noise'] == round(noise_level, 2)].iloc[0]
current_acc = current_metrics['Accuracy']
current_type1 = current_metrics['Type I Error']
current_type2 = current_metrics['Type II Error']

# Calculate exact confusion matrix for the visual
np.random.seed(42)
X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
probs_current = rf_model.predict_proba(X_test_noisy)[:, 1]
predictions = (probs_current >= decision_threshold).astype(int)
cm_current = confusion_matrix(y_test, predictions)

# 5. DUAL-TRIGGER Algorithmic Circuit Breaker Logic
st.divider()
if current_type1 > 20.0 or current_type2 > 20.0:
    breach_type = []
    if current_type1 > 20.0: breach_type.append("Type I (Missed Defaults)")
    if current_type2 > 20.0: breach_type.append("Type II (False Alarms)")
    breach_text = " and ".join(breach_type)
    
    st.error(
        f"🚨 **CIRCUIT BREAKER TRIGGERED:** {breach_text} breached the 20% safety threshold at {noise_level:.2f}σ market volatility."
    )
else:
    st.success(
        f"✅ **SYSTEM STABLE:** Both error rates are safely within normal operating limits at {noise_level:.2f}σ."
    )

# 6. Dashboard Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Optimized Model Accuracy", f"{current_acc:.2f}%")
col2.metric("Type I Error (Missed Defaults)", f"{current_type1:.1f}%")
col3.metric("Type II Error (False Alarms)", f"{current_type2:.1f}%")

st.divider()

# 7. Interactive Plotly Visualizations
col_chart, col_matrix = st.columns([2, 1])

with col_chart:
    st.subheader("Failure Rate Trajectory")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_curve["Noise"], y=df_curve["Type I Error"],
        mode="lines", name="Type I (Missed Defaults - Fatal)",
        line=dict(color="#2980b9", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["Noise"], y=df_curve["Type II Error"],
        mode="lines", name="Type II (False Alarms)",
        line=dict(color="#e74c3c", width=3)
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
