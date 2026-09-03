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
st.set_page_config(page_title="Digital Twin Dashboard", layout="wide")
st.title("🏭 AI Digital Twin - Industrial Sensor Monitoring")

# Sidebar - Machine Controls & Thresholds
st.sidebar.header("Control Panel")
vibration_threshold = st.sidebar.slider("Vibration Anomaly Limit", 0.0, 10.0, 7.5)
temp_threshold = st.sidebar.slider("Temperature Warning Limit (°C)", 50, 100, 85)

# Initialize Session State Data if empty
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp", "Vibration", "Temperature", "Anomaly_Score"])

# Layout Placeholders for Real-Time Streaming
metric_col1, metric_col2, metric_col3 = st.columns(3)
chart_col1, chart_col2 = st.columns(2)

metric_p1 = metric_col1.empty()
metric_p2 = metric_col2.empty()
metric_p3 = metric_col3.empty()

chart_p1 = chart_col1.empty()
chart_p2 = chart_col2.empty()

# Simulated AI Model (Replace with your group's ML model inference function)
def predict_anomaly(vib, temp):
    # Simulated anomaly risk score based on sensor inputs
    score = (vib / 10.0) * 0.5 + (temp / 100.0) * 0.5
    return min(round(score, 2), 1.0)

# Main Real-Time Event Loop
run_monitoring = st.checkbox("Start Live Sensor Feed", value=True)

while run_monitoring:
    # 1. Simulate Incoming Sensor Data
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    vib = round(np.random.normal(5.0, 1.2), 2)
    temp = round(np.random.normal(70.0, 5.0), 2)
    anomaly_score = predict_anomaly(vib, temp)

    # 2. Append New Point to DataFrame (Keep last 30 data points)
    new_row = pd.DataFrame([{"Timestamp": now, "Vibration": vib, "Temperature": temp, "Anomaly_Score": anomaly_score}])
    st.session_state.data = pd.concat([st.session_state.data, new_row]).tail(30)
    df = st.session_state.data

    # 3. Update Metrics Display
    metric_p1.metric(label="Vibration Level", value=f"{vib} mm/s", delta=round(vib - 5.0, 2))
    metric_p2.metric(label="Temperature", value=f"{temp} °C", delta=round(temp - 70.0, 2))
    
    status = "🚨 Anomaly Detected!" if vib > vibration_threshold or temp > temp_threshold else "✅ Normal"
    metric_p3.metric(label="Machine Status", value=status, delta=f"Risk Score: {anomaly_score}")

    # 4. Render Interactive Real-Time Plots
    fig_vib = go.Figure()
    fig_vib.add_trace(go.Scatter(x=df["Timestamp"], y=df["Vibration"], mode="lines+markers", name="Vibration"))
    fig_vib.add_hline(y=vibration_threshold, line_dash="dash", line_color="red", annotation_text="Limit")
    fig_vib.update_layout(title="Vibration Feed (mm/s)", xaxis_title="Time", yaxis_title="mm/s", margin=dict(l=20, r=20, t=40, b=20))
    chart_p1.plotly_chart(fig_vib, use_container_width=True)

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df["Timestamp"], y=df["Temperature"], mode="lines+markers", name="Temp (°C)", line=dict(color="orange")))
    fig_temp.add_hline(y=temp_threshold, line_dash="dash", line_color="red", annotation_text="Warning Threshold")
    fig_temp.update_layout(title="Temperature Feed (°C)", xaxis_title="Time", yaxis_title="°C", margin=dict(l=20, r=20, t=40, b=20))
    chart_p2.plotly_chart(fig_temp, use_container_width=True)

    # Stream Interval (1 update per second)
    time.sleep(1)