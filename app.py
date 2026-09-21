import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time
import os

# Page Configuration
st.set_page_config(
    page_title="AI Digital Twin Monitoring Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    .stApp { background: #080b11; color: #e0e6ed; }
    
    @keyframes radar-sweep {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .radar-box {
        position: relative; width: 80px; height: 80px; border-radius: 50%;
        border: 2px solid #00f2fe; background: radial-gradient(circle, rgba(0,242,254,0.1) 0%, rgba(0,0,0,0.8) 70%);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4); margin: 0 auto; overflow: hidden;
    }
    .radar-sweep-line {
        position: absolute; top: 50%; left: 50%; width: 50%; height: 50%;
        background: linear-gradient(45deg, rgba(0,242,254,0.6), transparent);
        transform-origin: top left; animation: radar-sweep 2s linear infinite;
    }
    .glass-card {
        background: rgba(26, 31, 44, 0.65); backdrop-filter: blur(10px);
        border-radius: 12px; padding: 18px; border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    @keyframes pulse-danger {
        0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 23, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }
    }
    .alert-ok {
        background: rgba(13, 59, 46, 0.8); border-left: 5px solid #00e676;
        padding: 12px 20px; border-radius: 8px; color: #00e676; font-weight: bold;
    }
    .alert-danger {
        background: rgba(74, 13, 13, 0.9); border-left: 5px solid #ff1744;
        padding: 12px 20px; border-radius: 8px; color: #ff1744; font-weight: bold;
        animation: pulse-danger 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load ML Model Bundle and Dataset
@st.cache_resource
def load_ml_assets():
    if not os.path.exists("twin_model.pkl") or not os.path.exists("industrial_sensor_dataset.csv"):
        st.error("❌ Model or Dataset file missing! Run `python train_model.py` in cmd first.")
        st.stop()
    models = joblib.load("twin_model.pkl")
    dataset = pd.read_csv("industrial_sensor_dataset.csv")
    return models, dataset

models, full_dataset = load_ml_assets()
rf_classifier = models["classifier"]
iso_forest = models["anomaly_detector"]

# Dashboard Header & Sidebar
st.title("🏭 AI Digital Twin - Industrial Monitoring")
st.caption("Real-Time Dataset Telemetry Feed & Scikit-Learn Model Inference")

st.sidebar.title("🎮 Control Hub")
selected_asset = st.sidebar.selectbox("Asset Target:", ["Gas Turbine #1", "CNC Milling Unit #2", "Hydraulic Pump #3"])
run_monitoring = st.sidebar.toggle("Stream Live Telemetry", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Threshold Alerts")
vibration_threshold = st.sidebar.slider("Vibration Max (mm/s)", 1.0, 10.0, 6.5, 0.1)
temp_threshold = st.sidebar.slider("Temp Max (°C)", 40, 110, 80)
current_threshold = st.sidebar.slider("Current Max (A)", 5.0, 50.0, 32.0, 0.5)

# Initialize Session Data Buffer
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Vibration", "Temperature", "Current", "ML_Fault_Pred", "Anomaly_Score"])

# Helper: 3D Twin Mesh Generator
def build_3d_digital_twin(rotation_angle, is_critical):
    z = np.linspace(0, 10, 25)
    theta = np.linspace(0, 2 * np.pi, 25) + rotation_angle
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = (2 + 0.5 * np.cos(theta_grid * 4)) * np.cos(theta_grid)
    y = (2 + 0.5 * np.cos(theta_grid * 4)) * np.sin(theta_grid)
    color_scheme = 'Reds' if is_critical else 'YlGnBu'
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z_grid, colorscale=color_scheme, showscale=False)])
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), height=300
    )
    return fig

# Helper: Gauge Creator
def create_gauge(value, min_val, max_val, title, threshold, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 13, 'color': "#ffffff"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "#1a1f2c",
            'threshold': {'line': {'color': "#ff1744", 'width': 4}, 'value': threshold}
        }
    ))
    fig.update_layout(height=160, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

# UI Tabs Layout
tab_live, tab_ai, tab_diagnostics = st.tabs(["📡 Live Telemetry Twin", "🤖 AI Model Predictions", "⚙️ Raw Dataset & Logs"])

warning_placeholder = st.empty()

with tab_live:
    col_twin_3d, col_gauges = st.columns([1, 1.2])
    with col_twin_3d:
        twin_3d_p = st.empty()
        radar_p = st.empty()
    with col_gauges:
        col_g1, col_g2, col_g3 = st.columns(3)
        gauge_vib_p = col_g1.empty()
        gauge_temp_p = col_g2.empty()
        gauge_curr_p = col_g3.empty()

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    chart_vib_p = chart_col1.empty()
    chart_temp_p = chart_col2.empty()
    chart_curr_p = chart_col3.empty()

with tab_ai:
    st.subheader("Random Forest & Isolation Forest ML Model Outputs")
    ai_metrics_p = st.empty()
    ai_chart_p = st.empty()

with tab_diagnostics:
    st.subheader("Streamed Telemetry Record History")
    log_table_p = st.empty()

# Streaming Loop Counter
step_count = 0
total_records = len(full_dataset)

# Main Live Streaming Loop (Replays dataset rows sequentially)
while run_monitoring:
    row_idx = step_count % total_records
    step_count += 1
    
    # 1. Fetch current row from dataset
    current_row = full_dataset.iloc[row_idx]
    now_time = pd.Timestamp.now().strftime("%H:%M:%S")
    
    vib = float(current_row["Vibration"])
    temp = float(current_row["Temperature"])
    curr = float(current_row["Current"])

    # 2. Perform ML Model Inference
    features = pd.DataFrame([[vib, temp, curr]], columns=["Vibration", "Temperature", "Current"])
    
    # Random Forest Failure Prediction (0 = Normal, 1 = Fault)
    ml_fault_pred = int(rf_classifier.predict(features)[0])
    
    # Isolation Forest Anomaly Score (-1 to 1 normalized into 0 to 1 risk ratio)
    raw_anomaly_score = iso_forest.score_samples(features)[0]
    anomaly_risk = np.clip(round((0.5 - raw_anomaly_score), 2), 0.0, 1.0)

    # 3. Save to Live Buffer
    new_entry = pd.DataFrame([{
        "Timestamp": now_time,
        "Vibration": vib,
        "Temperature": temp,
        "Current": curr,
        "ML_Fault_Pred": ml_fault_pred,
        "Anomaly_Score": anomaly_risk
    }])
    
    st.session_state.data = pd.concat([st.session_state.data, new_entry]).tail(25)
    df = st.session_state.data

    # Alert Conditions (Combines manual thresholds and ML classification)
    is_threshold_breach = vib > vibration_threshold or temp > temp_threshold or curr > current_threshold
    is_critical = (ml_fault_pred == 1) or is_threshold_breach

    # 4. Display Alert Banner
    if is_critical:
        warning_placeholder.markdown(
            f'<div class="alert-danger">🚨 ML ALERT ON {selected_asset}: Failure/Anomaly Detected! (ML Status: {"FAULT PREDICTED" if ml_fault_pred == 1 else "Normal"}, Vib: {vib}mm/s, Temp: {temp}°C, Curr: {curr}A)</div>',
            unsafe_allow_html=True
        )
    else:
        warning_placeholder.markdown(
            f'<div class="alert-ok">✅ SYSTEM OPTIMAL: {selected_asset} streaming dataset record {row_idx + 1}/{total_records}. All ML scores normal.</div>',
            unsafe_allow_html=True
        )

    # 5. Render 3D Mesh & Radar
    angle = (step_count * 0.3) % (2 * np.pi)
    twin_3d_p.plotly_chart(build_3d_digital_twin(angle, is_critical), use_container_width=True, key=f"3d_{step_count}")
    radar_p.markdown("""
        <div style="text-align: center; margin-bottom: 5px;">
            <div class="radar-box"><div class="radar-sweep-line"></div></div>
            <span style="font-size: 0.7rem; color: #00f2fe;">DATASET IoT STREAM ACTIVE</span>
        </div>
    """, unsafe_allow_html=True)

    # 6. Render Gauges & Time-Series Charts
    gauge_vib_p.plotly_chart(create_gauge(vib, 0, 10, "Vibration (mm/s)", vibration_threshold, "#00f2fe"), use_container_width=True, key=f"gv_{step_count}")
    gauge_temp_p.plotly_chart(create_gauge(temp, 30, 120, "Temperature (°C)", temp_threshold, "#ff9f43"), use_container_width=True, key=f"gt_{step_count}")
    gauge_curr_p.plotly_chart(create_gauge(curr, 0, 50, "Current Draw (A)", current_threshold, "#a855f7"), use_container_width=True, key=f"gc_{step_count}")

    fig_v = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Vibration"], mode="lines+markers", line=dict(color="#00f2fe", width=2)))
    fig_v.update_layout(title="Vibration Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=200, margin=dict(l=10,r=10,t=30,b=10))
    chart_vib_p.plotly_chart(fig_v, use_container_width=True, key=f"cv_{step_count}")

    fig_t = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Temperature"], mode="lines+markers", line=dict(color="#ff9f43", width=2)))
    fig_t.update_layout(title="Thermal Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=200, margin=dict(l=10,r=10,t=30,b=10))
    chart_temp_p.plotly_chart(fig_t, use_container_width=True, key=f"ct_{step_count}")

    fig_c = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Current"], mode="lines+markers", line=dict(color="#a855f7", width=2)))
    fig_c.update_layout(title="Current Draw Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=200, margin=dict(l=10,r=10,t=30,b=10))
    chart_curr_p.plotly_chart(fig_c, use_container_width=True, key=f"cc_{step_count}")

    # 7. Render AI Tab Outputs
    rul_hours = int((1.0 - anomaly_risk) * 1000)
    ai_metrics_p.markdown(f"""
        <div style="display: flex; gap: 20px;">
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">RANDOM FOREST CLASSIFICATION</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: {'#ff1744' if ml_fault_pred == 1 else '#00e676'};">
                    {'🚨 FAULT DETECTED' if ml_fault_pred == 1 else '✅ NORMAL OPERATION'}
                </div>
            </div>
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">ISOLATION FOREST ANOMALY RISK</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #00f2fe;">{anomaly_risk * 100:.0f}%</div>
            </div>
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">PREDICTED RUL</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #a855f7;">{rul_hours} Hours</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig_ai = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Anomaly_Score"], fill='tozeroy', line=dict(color="#00f2fe")))
    fig_ai.update_layout(title="ML Isolation Forest Anomaly Risk Trend", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=220)
    ai_chart_p.plotly_chart(fig_ai, use_container_width=True, key=f"c_ai_{step_count}")

    # 8. Diagnostics Tab
    log_table_p.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

    time.sleep(0.8)
