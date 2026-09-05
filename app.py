import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Industrial AI Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS: Glassmorphism, Neon Glows
st.markdown("""
<style>
    .stApp {
        background: #080b11;
        color: #e0e6ed;
    }
    
    .glass-card {
        background: rgba(26, 31, 44, 0.65);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease-in-out;
    }
    
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

# Main Title & Dynamic Sidebar
st.title("🏭 AI-Powered Industrial Digital Twin")
st.caption("Real-Time Telemetry: Vibration, Temperature & Motor Current Draw")

st.sidebar.title("🎮 Twin Control Hub")
selected_machine = st.sidebar.selectbox(
    "Select Industrial Asset:",
    ["Gas Turbine #1 (Primary)", "CNC Milling Center #2", "Hydraulic Pump Unit #3"]
)

run_monitoring = st.sidebar.toggle("Stream Telemetry", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Safety Limits")
vibration_threshold = st.sidebar.slider("Vibration Limit (mm/s)", 1.0, 10.0, 6.5, 0.1)
temp_threshold = st.sidebar.slider("Temperature Limit (°C)", 40, 110, 82)
current_threshold = st.sidebar.slider("Current Limit (A)", 5.0, 50.0, 32.0, 0.5)

# Session State Setup
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Vibration", "Temperature", "Current", "Anomaly_Score"])

# Helper: Gauge Creator
def create_gauge(value, min_val, max_val, title, threshold, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
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

# Tabs
tab_live, tab_ai, tab_diagnostics = st.tabs(["📡 Live Telemetry Twin", "🤖 Predictive AI Engine", "⚙️ Diagnostics & Logs"])

warning_placeholder = st.empty()

with tab_live:
    col_g1, col_g2, col_g3 = st.columns(3)
    gauge_vib_p = col_g1.empty()
    gauge_temp_p = col_g2.empty()
    gauge_curr_p = col_g3.empty()

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    chart_vib_p = chart_col1.empty()
    chart_temp_p = chart_col2.empty()
    chart_curr_p = chart_col3.empty()

with tab_ai:
    st.subheader("AI Anomaly Engine & RUL Forecast")
    ai_metrics_p = st.empty()
    ai_chart_p = st.empty()

with tab_diagnostics:
    st.subheader("Asset Telemetry Logs")
    log_table_p = st.empty()

step_count = 0

# Main Monitoring Loop
while run_monitoring:
    step_count += 1
    
    # Updated: Explicitly fetches local system clock time
    now = datetime.now().astimezone().strftime("%H:%M:%S")

    # 1. Simulate 3 Sensor Inputs (Vibration, Temp, Current)
    vib = round(np.random.normal(5.2, 1.3), 2)
    temp = round(np.random.normal(72.0, 5.5), 2)
    curr = round(np.random.normal(24.5, 4.0), 2)  # Motor Current in Amps

    # AI Anomaly Score calculation with 3 sensor weights
    anomaly_score = min(round((vib / 10.0) * 0.35 + (temp / 110.0) * 0.35 + (curr / 50.0) * 0.30, 2), 1.0)

    # 2. Append Data
    new_data = pd.DataFrame([{"Timestamp": now, "Vibration": vib, "Temperature": temp, "Current": curr, "Anomaly_Score": anomaly_score}])
    st.session_state.data = pd.concat([st.session_state.data, new_data]).tail(25)
    df = st.session_state.data

    # Check for Breaches
    vib_crit = vib > vibration_threshold
    temp_crit = temp > temp_threshold
    curr_crit = curr > current_threshold
    is_critical = vib_crit or temp_crit or curr_crit

    # 3. Dynamic Alert Banner
    if is_critical:
        breaches = []
        if vib_crit: breaches.append(f"Vibration ({vib} mm/s)")
        if temp_crit: breaches.append(f"Temp ({temp}°C)")
        if curr_crit: breaches.append(f"Current ({curr} A)")
        warning_placeholder.markdown(
            f'<div class="alert-danger">🚨 CRITICAL ALERT on {selected_machine}: Over Limit in {", ".join(breaches)}!</div>',
            unsafe_allow_html=True
        )
    else:
        warning_placeholder.markdown(
            f'<div class="alert-ok">✅ SYSTEM OPTIMAL: {selected_machine} operating normally. All 3 sensor feeds within range.</div>',
            unsafe_allow_html=True
        )

    # 4. Render Gauges (Vibration, Temp, Current)
    gauge_vib_p.plotly_chart(create_gauge(vib, 0, 10, "Vibration (mm/s)", vibration_threshold, "#00f2fe"), use_container_width=True, key=f"g_vib_{step_count}")
    gauge_temp_p.plotly_chart(create_gauge(temp, 30, 120, "Temp (°C)", temp_threshold, "#ff9f43"), use_container_width=True, key=f"g_temp_{step_count}")
    gauge_curr_p.plotly_chart(create_gauge(curr, 0, 50, "Current (A)", current_threshold, "#a855f7"), use_container_width=True, key=f"g_curr_{step_count}")

    # 5. Render 3 Time-Series Charts
    fig_vib = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Vibration"], mode="lines+markers", line=dict(color="#00f2fe", width=2)))
    fig_vib.add_hline(y=vibration_threshold, line_dash="dash", line_color="#ff1744")
    fig_vib.update_layout(title="Vibration Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=220, margin=dict(l=20,r=20,t=30,b=20))
    chart_vib_p.plotly_chart(fig_vib, use_container_width=True, key=f"c_vib_{step_count}")

    fig_temp = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Temperature"], mode="lines+markers", line=dict(color="#ff9f43", width=2)))
    fig_temp.add_hline(y=temp_threshold, line_dash="dash", line_color="#ff1744")
    fig_temp.update_layout(title="Thermal Feed", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=220, margin=dict(l=20,r=20,t=30,b=20))
    chart_temp_p.plotly_chart(fig_temp, use_container_width=True, key=f"c_temp_{step_count}")

    fig_curr = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Current"], mode="lines+markers", line=dict(color="#a855f7", width=2)))
    fig_curr.add_hline(y=current_threshold, line_dash="dash", line_color="#ff1744")
    fig_curr.update_layout(title="Current Draw (Amps)", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=220, margin=dict(l=20,r=20,t=30,b=20))
    chart_curr_p.plotly_chart(fig_curr, use_container_width=True, key=f"c_curr_{step_count}")

    # 6. Render AI Predictive Tab
    rul_hours = int((1 - anomaly_score) * 1200)
    ai_metrics_p.markdown(f"""
        <div style="display: flex; gap: 20px;">
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">ANOMALY RISK SCORE</div>
                <div style="font-size: 2rem; font-weight: bold; color: {'#ff1744' if anomaly_score > 0.7 else '#00e676'};">{anomaly_score*100:.1f}%</div>
            </div>
            <div class="glass-card" style="flex: 1; text-align: center;">
                <div style="color: #8b9bb4; font-size: 0.8rem;">ESTIMATED RUL (REMAINING USEFUL LIFE)</div>
                <div style="font-size: 2rem; font-weight: bold; color: #00f2fe;">{rul_hours} Hours</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig_ai = go.Figure(go.Scatter(x=df["Timestamp"], y=df["Anomaly_Score"], fill='tozeroy', line=dict(color="#ff1744" if anomaly_score > 0.7 else "#00e676")))
    fig_ai.update_layout(title="Multi-Sensor Predictive Anomaly Score", paper_bgcolor="#1a1f2c", plot_bgcolor="#1a1f2c", font=dict(color="white"), height=250)
    ai_chart_p.plotly_chart(fig_ai, use_container_width=True, key=f"c_ai_{step_count}")

    # 7. Diagnostics Tab
    log_table_p.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

    time.sleep(0.8)
