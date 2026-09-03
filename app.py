import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Page Configuration
st.set_page_config(
    page_title="Industrial AI Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS: Animations, Glassmorphism, Neon Glows
st.markdown("""
<style>
    /* Dark Sci-Fi Theme Background */
    .stApp {
        background: #080b11;
        color: #e0e6ed;
    }
    
    /* Glowing Radar Keyframe Animation */
    @keyframes radar-sweep {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .radar-box {
        position: relative;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 2px solid #00f2fe;
        background: radial-gradient(circle, rgba(0,242,254,0.1) 0%, rgba(0,0,0,0.8) 70%);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        margin: 0 auto;
        overflow: hidden;
    }
    
    .radar-sweep-line {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 50%;
        height: 50%;
        background: linear-gradient(45deg, rgba(0,242,254,0.6), transparent);
        transform-origin: top left;
        animation: radar-sweep 2s linear infinite;
    }
    
    /* Neon Metric Glass Cards */
    .glass-card {
        background: rgba(26, 31, 44, 0.65);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease-in-out;
    }
    
    .glass-card:hover {
        border-color: #00f2fe;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    }
    
    /* Alert Banners with Pulsing Keyframes */
    @keyframes pulse-danger {
        0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 23, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }
    }
    
    .alert-ok {
        background: rgba(13, 59, 46, 0.8);
        border-left: 5px solid #00e676;
        padding: 12px 20px;
        border-radius: 8px;
        color: #00e676;
        font-weight: bold;
    }
    
    .alert-danger {
        background: rgba(74, 13, 13, 0.9);
        border-left: 5px solid #ff1744;
        padding: 12px 20px;
        border-radius: 8px;
        color: #ff1744;
        font-weight: bold;
        animation: pulse-danger 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Dynamic Machine Selection Sidebar
st.title("🏭 AI-Powered Industrial Digital Twin")
st.caption("Real-Time Cyber-Physical Asset Telemetry & Anomaly Forecasting")

st.sidebar.title("🎮 Twin Control Hub")
selected_machine = st.sidebar.selectbox(
    "Select Industrial Asset:",
    ["Gas Turbine #1 (Primary)", "CNC Milling Center #2", "Hydraulic Pump Unit #3"]
)

run_monitoring = st.sidebar.toggle("Stream Telemetry", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Controls")
vibration_threshold = st.sidebar.slider("Vibration Threshold (mm/s)", 1.0, 10.0, 6.5, 0.1)
temp_threshold = st.sidebar.slider("Temperature Limit (°C)", 40, 110, 82)

# Session State Setup
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Vibration", "Temperature", "Anomaly_Score"])

# Helper: Generate 3D Spinning Mesh Twin
def build_3d_digital_twin(rotation_angle, is_critical):
    # Constructing a 3D Cylinder/Rotor representation dynamically
    z = np.linspace(0, 10, 30)
    theta = np.linspace(0, 2 * np.pi, 30) + rotation_angle
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = (2 + 0.5 * np.cos(theta_grid * 4)) * np.cos(theta_grid)
    y = (2 + 0.5 * np.cos(theta_grid * 4)) * np.sin(theta_grid)

    color_scheme = 'Reds' if is_critical else 'YlGnBu'
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z_grid, colorscale=color_scheme, showscale=False)])
    fig.update_layout(
        title="3D Dynamic Physical Twin Model",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=320
    )
    return fig

# Helper: Gauge Creator
def create_gauge(value, min_val, max_val, title, threshold, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': "#ffffff"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "#1a1f2c",
            'threshold': {'line': {'color': "#ff1744", 'width': 4}, 'value': threshold}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

# Define UI Navigation Tabs
tab_live, tab_ai, tab_diagnostics = st.tabs(["📡 Live Telemetry Twin", "🤖 Predictive AI Engine", "⚙️ Machine Diagnostics & Logs"])

# Top Alert Banner & Radar Placeholder
warning_placeholder = st.empty()

with tab_live:
    col_twin_3d, col_gauges = st.columns([1.2, 1])
    
    with col_twin_3d:
        twin_3d_placeholder = st.empty()
        
    with col_gauges:
        col_g1, col_g2 = st.columns(2)
        gauge_vib_p = col_g1.empty()
        gauge_temp_p = col_g2.empty()
        radar_p = st.empty()

    chart_col1, chart_col2 = st.columns(2)
    chart_vib_p = chart_col1.empty()
    chart_temp_p = chart_col2.empty()

with tab_ai:
    st.subheader("AI Anomaly Engine & RUL (Remaining Useful Life) Forecast")
    ai_metrics_p = st.empty()
    ai_chart_p = st.empty()

with tab_diagnostics:
    st.subheader("Asset Health Logs & Parameter History")
    log_table_p = st.empty()

# Animation Counter
step_count = 0

# Stream Loop
while run_monitoring:
    step_count += 1
    angle = (step_count * 0.4) % (2 * np.pi)
    now = pd.Timestamp.now().strftime("%H:%M:%S")

    # 1. Simulate Telemetry
    vib = round(np.random.normal(5.2, 1.3), 2)
    temp = round(np.random.normal(72.0, 5.5), 2)
    anomaly_score = min(round((vib / 10.0) * 0.5 + (temp / 110.0) * 0.5, 2), 1.0)

    # 2. Store History
    new_data = pd.DataFrame([{"Timestamp": now, "Vibration": vib, "Temperature": temp, "Anomaly_Score": anomaly_score}])
    st.session_state.data = pd.concat([st.session_state.data, new_data]).tail(25)
    df = st.session_state.data

    is_critical = vib > vibration_threshold or temp > temp_threshold

    # 3. Dynamic Alert Bar
    if is_critical:
        warning_placeholder.markdown(
            f'<div class="alert-danger">🚨 CRITICAL ALERT: Threshold Breached on {selected_machine}! Vibration: {vib} mm/s | Temp: {temp}°C</div>',
            unsafe_allow_html=True
        )
    else:
        warning_placeholder.markdown(
            f'<div class="alert-ok">✅ SYSTEM OPTIMAL: {selected_machine} operating normally. Radar tracking active.</div>',
            unsafe_allow_html=True
        )

    # 4. Render 3D Model & Radar
    twin_3d_placeholder.plotly_chart(
        build_3d_digital_twin(angle, is_critical), 
        use_container_width=True, 
        key=f"3d_twin_{step_count}"
    )

    radar_p.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <div class="radar-box"><div class="radar-sweep-line"></div></div>
            <span style="font-size: 0.75rem; color: #00f2fe;">LIVE SENSOR RADAR SCANNING</span>
        </div>
    """, unsafe_allow_html=True)

    # 5. Render Gauges & Time-Series
    gauge_vib_p.plotly_chart(create_gauge(vib, 0, 10, "Vibration", vibration_threshold, "#00f2fe"), use_container_width=True, key=f"g_vib_{step_count}")
    gauge_temp_p.plotly_chart(create_gauge(temp, 30, 120, "Temperature", temp_threshold, "#ff9f43"), use_container_width=True, key=f"g_temp_{step_count}")

    # Vibration Chart
    fig_vib = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Vibration"], mode="lines+markers", line=dict(color="#00f2fe", width=3)))
    fig_vib.add_hline(y=vibration_threshold, line_dash="dash", line_color="#ff1744")
    fig_vib.update_layout(title="Real-Time Vibration Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=240, margin=dict(l=20,r=20,t=30,b=20))
    chart_vib_p.plotly_chart(fig_vib, use_container_width=True, key=f"c_vib_{step_count}")

    # Temperature Chart
    fig_temp = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Temperature"], mode="lines+markers", line=dict(color="#ff9f43", width=3)))
    fig_temp.add_hline(y=temp_threshold, line_dash="dash", line_color="#ff1744")
    fig_temp.update_layout(title="Real-Time Thermal Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=240, margin=dict(l=20,r=20,t=30,b=20))
    chart_temp_p.plotly_chart(fig_temp, use_container_width=True, key=f"c_temp_{step_count}")

    # 6. Render Tab 2 (AI Tab)
    rul_hours = int((1 - anomaly_score) * 1200)
    ai_metrics_p.markdown(f"""
        <div style="display: flex; gap: 20px;">
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">ANOMALY CONFIDENCE</div>
                <div style="font-size: 2rem; font-weight: bold; color: {'#ff1744' if anomaly_score > 0.7 else '#00e676'};">{anomaly_score*100:.1f}%</div>
            </div>
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">ESTIMATED REMAINING USEFUL LIFE (RUL)</div>
                <div style="font-size: 2rem; font-weight: bold; color: #00f2fe;">{rul_hours} Hours</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig_ai = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Anomaly_Score"], fill='tozeroy', line=dict(color="#ff1744" if anomaly_score > 0.7 else "#00e676")))
    fig_ai.update_layout(title="Predicted AI Anomaly Risk Curve", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=250)
    ai_chart_p.plotly_chart(fig_ai, use_container_width=True, key=f"c_ai_{step_count}")

    # 7. Render Tab 3 (Logs Tab)
    log_table_p.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

    time.sleep(0.8)
