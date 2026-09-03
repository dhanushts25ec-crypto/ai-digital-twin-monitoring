# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:37:52 2026

@author: lenov
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Page Configuration
st.set_page_config(page_title="Industrial Digital Twin", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Industrial Dark Dashboard Styling
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Card Container Styling */
    .metric-card {
        background-color: #1a1f2c;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #2e364f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    
    /* Dynamic Warning Banner Styles */
    .alert-ok {
        background: linear-gradient(90deg, #0d3b2e 0%, #13523f 100%);
        border-left: 6px solid #00e676;
        padding: 15px 20px;
        border-radius: 8px;
        color: #e0f2f1;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .alert-warning {
        background: linear-gradient(90deg, #4d3a00 0%, #664d00 100%);
        border-left: 6px solid #ffab00;
        padding: 15px 20px;
        border-radius: 8px;
        color: #fff8e1;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .alert-danger {
        background: linear-gradient(90deg, #4a0d0d 0%, #6b1212 100%);
        border-left: 6px solid #ff1744;
        padding: 15px 20px;
        border-radius: 8px;
        color: #ffebee;
        font-weight: 700;
        font-size: 1.2rem;
        animation: blinker 1.5s linear infinite;
    }
    
    @keyframes blinker {
        50% { opacity: 0.75; }
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.title("🏭 AI Digital Twin Monitoring Center")
st.caption("Real-Time Predictive Maintenance & Telemetry")

# Sidebar Configuration
st.sidebar.title("⚙️ Control Panel")
run_monitoring = st.sidebar.toggle("Stream Telemetry", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Safety Thresholds")
vibration_threshold = st.sidebar.slider("Max Vibration (mm/s)", 1.0, 10.0, 6.5, 0.1)
temp_threshold = st.sidebar.slider("Max Temp (°C)", 40, 110, 80)

# Session State Initialization
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Vibration", "Temperature", "Anomaly_Score"])

# Layout Structure: Warning Banner Top, Gauge Row, Chart Row
warning_placeholder = st.empty()

col_m1, col_m2, col_m3 = st.columns(3)
card_vib_placeholder = col_m1.empty()
card_temp_placeholder = col_m2.empty()
card_score_placeholder = col_m3.empty()

chart_col1, chart_col2 = st.columns(2)
chart_vib_placeholder = chart_col1.empty()
chart_temp_placeholder = chart_col2.empty()

# Simulated AI Inference
def predict_anomaly(vib, temp):
    score = (vib / 10.0) * 0.5 + (temp / 110.0) * 0.5
    return min(round(score, 2), 1.0)

# Helper Function: Gauge Chart Generator
def create_gauge(value, min_val, max_val, title, threshold, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': "#ffffff"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': color},
            'bgcolor': "#1a1f2c",
            'bordercolor': "#2e364f",
            'threshold': {
                'line': {'color': "#ff1744", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        height=200, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig

# Initialize loop step counter
step_count = 0

# Main Loop
while run_monitoring:
    step_count += 1
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    
    # 1. Simulate Live Readings
    vib = round(np.random.normal(5.0, 1.2), 2)
    temp = round(np.random.normal(70.0, 6.0), 2)
    anomaly_score = predict_anomaly(vib, temp)

    # 2. Update Data Store (Keep last 25 readings)
    new_row = pd.DataFrame([{"Timestamp": now, "Vibration": vib, "Temperature": temp, "Anomaly_Score": anomaly_score}])
    st.session_state.data = pd.concat([st.session_state.data, new_row]).tail(25)
    df = st.session_state.data

    # 3. Dynamic Warning System Assessment
    vib_crit = vib > vibration_threshold
    temp_crit = temp > temp_threshold
    
    if vib_crit and temp_crit:
        warning_placeholder.markdown(
            f'<div class="alert-danger">🚨 CRITICAL SYSTEM ALERT: Simultaneous High Vibration ({vib} mm/s) & High Temperature ({temp}°C) Detected!</div>', 
            unsafe_allow_html=True
        )
    elif vib_crit:
        warning_placeholder.markdown(
            f'<div class="alert-warning">⚠️ VIBRATION WARNING: Current level ({vib} mm/s) exceeds safety limit ({vibration_threshold} mm/s).</div>', 
            unsafe_allow_html=True
        )
    elif temp_crit:
        warning_placeholder.markdown(
            f'<div class="alert-warning">⚠️ TEMPERATURE WARNING: Current level ({temp}°C) exceeds safety limit ({temp_threshold}°C).</div>', 
            unsafe_allow_html=True
        )
    else:
        warning_placeholder.markdown(
            '<div class="alert-ok">✅ SYSTEM HEALTHY: All metrics operating within normal operating bounds.</div>', 
            unsafe_allow_html=True
        )

    # 4. Render Metric Cards & Gauges
    card_vib_placeholder.plotly_chart(
        create_gauge(vib, 0, 10, "Vibration (mm/s)", vibration_threshold, "#00f2fe"), 
        use_container_width=True,
        key=f"gauge_vib_{step_count}"
    )
    card_temp_placeholder.plotly_chart(
        create_gauge(temp, 30, 120, "Temperature (°C)", temp_threshold, "#ff9f43"), 
        use_container_width=True,
        key=f"gauge_temp_{step_count}"
    )
    
    # AI Risk Score Custom Card
    card_score_placeholder.markdown(
        f"""
        <div class="metric-card" style="text-align: center; margin-top: 15px;">
            <div class="metric-label">AI Anomaly Risk Score</div>
            <div class="metric-value" style="color: {'#ff1744' if anomaly_score > 0.75 else '#00e676'};">
                {anomaly_score * 100:.0f}%
            </div>
            <div style="font-size: 0.8rem; color: #8b9bb4;">Predictive Engine Confidence</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # 5. Render Dark Theme Time-Series Charts
    fig_vib = go.Figure()
    fig_vib.add_trace(go.Scatter(x=df["Timestamp"], y=df["Vibration"], mode="lines+markers", line=dict(color="#00f2fe", width=2)))
    fig_vib.add_hline(y=vibration_threshold, line_dash="dash", line_color="#ff1744", annotation_text="Limit")
    fig_vib.update_layout(
        title="Vibration Trend (Live)",
        paper_bgcolor="#1a1f2c",
        plot_bgcolor="#1a1f2c",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2e364f"),
        margin=dict(l=30, r=30, t=40, b=30)
    )
    chart_vib_placeholder.plotly_chart(fig_vib, use_container_width=True, key=f"chart_vib_{step_count}")

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df["Timestamp"], y=df["Temperature"], mode="lines+markers", line=dict(color="#ff9f43", width=2)))
    fig_temp.add_hline(y=temp_threshold, line_dash="dash", line_color="#ff1744", annotation_text="Limit")
    fig_temp.update_layout(
        title="Temperature Trend (Live)",
        paper_bgcolor="#1a1f2c",
        plot_bgcolor="#1a1f2c",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2e364f"),
        margin=dict(l=30, r=30, t=40, b=30)
    )
    chart_temp_placeholder.plotly_chart(fig_temp, use_container_width=True, key=f"chart_temp_{step_count}")

    time.sleep(1)