import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Page Configuration
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")
st.markdown("### Algorithmic Reliability & FMEA Terminal")

# 2. Sidebar Controls
st.sidebar.header("Market Volatility Simulation")
noise_level = st.sidebar.slider("Inject Gaussian Noise (σ)", min_value=0.0, max_value=3.0, value=0.0, step=0.1)

# 3. Data Loading & Model Training (Cached for speed)
@st.cache_data
def load_and_train():
    # Attempt to load actual data, fallback to realistic synthetic data if CSV is missing
    try:
        df = pd.read_csv("corporateCreditRatingWithFinancialRatios.csv")
        
        # Drop the target columns first
        X = df.drop(columns=['Rating', 'Binary Rating'], errors='ignore')
        
        # Force X to only keep numerical columns (drops text like 'Name' or 'Sector')
        X = X.select_dtypes(include=['number'])
        
        # Fill any missing blank cells with 0 to prevent NaN errors
        X = X.fillna(0)
        
        y = df['Binary Rating']
    except FileNotFoundError:
        # Fallback for deployment testing
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(2500, 10), columns=[f"Feature_{i}" for i in range(10)])
        y = np.where(X['Feature_0'] + X['Feature_1'] > 0, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Pre-calculate baseline metrics for the degradation curve
    curve_data = []
    for n in np.arange(0.0, 3.1, 0.1):
        noisy_X = X_test + np.random.normal(0, n, X_test.shape)
        preds = rf.predict(noisy_X)
        cm = confusion_matrix(y_test, preds)
        
        # Calculate Error Rates
        tn, fp, fn, tp = cm.ravel()
        type1_error = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
        type2_error = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
        acc = accuracy_score(y_test, preds) * 100
        
        curve_data.append({"Noise": round(n, 1), "Accuracy": acc, "Type I Error": type1_error, "Type II Error": type2_error})
        
    return X_test, y_test, rf, pd.DataFrame(curve_data)

X_test, y_test, rf_model, df_curve = load_and_train()

# 4. Apply Noise for Current Slider Value
np.random.seed(42) # For reproducible noise
X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
predictions = rf_model.predict(X_test_noisy)
cm_current = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm_current.ravel()
current_type1 = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
current_type2 = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
current_acc = accuracy_score(y_test, predictions) * 100

# 5. Algorithmic Circuit Breaker Logic
if current_type2 > 20.0:
    st.error(f"🚨 CIRCUIT BREAKER TRIGGERED: Type II False Alarms breached 20% safety threshold at {noise_level}σ market volatility.")
else:
    st.success("✅ SYSTEM STABLE: Error rates within normal operating limits.")

# 6. Dashboard Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{current_acc:.1f}%")
col2.metric("Type I Error (Missed Defaults)", f"{current_type1:.1f}%")
col3.metric("Type II Error (False Alarms)", f"{current_type2:.1f}%")

st.divider()

# 7. Visualizations
col_chart, col_matrix = st.columns([2, 1])

with col_chart:
    st.subheader("Reliability Degradation Curve")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot curves
    sns.lineplot(data=df_curve, x="Noise", y="Accuracy", label="Accuracy (%)", color="green", linewidth=2, ax=ax)
    sns.lineplot(data=df_curve, x="Noise", y="Type II Error", label="Type II (False Alarms)", color="red", linewidth=2, ax=ax)
    sns.lineplot(data=df_curve, x="Noise", y="Type I Error", label="Type I (Missed Defaults)", color="blue", linewidth=2, ax=ax)
    
    # Vertical line for current slider position
    ax.axvline(x=noise_level, color='gray', linestyle='--', label=f'Current Noise ({noise_level}σ)')
    
    ax.set_xlabel("Gaussian Noise (Standard Deviations)")
    ax.set_ylabel("Percentage (%)")
    ax.legend()
    st.pyplot(fig)

with col_matrix:
    st.subheader(f"Confusion Matrix ({noise_level}σ)")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_current, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted Status')
    ax2.set_ylabel('Actual Status')
    st.pyplot(fig2)