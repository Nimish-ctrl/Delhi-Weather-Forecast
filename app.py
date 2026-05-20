import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading, time, os
from model_utils import predict_next_6_hours, finetune_on_recent, model, fetch_realtime, engineer_features,fetch_forecast_only

st.set_page_config(page_title="Delhi Weather AI", layout="wide")
FINETUNE_FILE = 'last_finetune.txt'

def should_finetune():
    if not os.path.exists(FINETUNE_FILE):
        return True
    with open(FINETUNE_FILE) as f:
        last = datetime.fromisoformat(f.read().strip())
    return datetime.now() - last > timedelta(days=7)

def run_finetune():
    finetune_on_recent(model, hours=720)
    with open(FINETUNE_FILE, 'w') as f:
        f.write(datetime.now().isoformat())

def scheduler_loop():
    while True:
        if should_finetune():
            run_finetune()
        time.sleep(3600)

if 'scheduler_started' not in st.session_state:
    threading.Thread(target=scheduler_loop, daemon=True).start()
    st.session_state['scheduler_started'] = True

st.title("🌤️ Delhi Weather Forecast")
st.caption(f"Last fine-tune: {open(FINETUNE_FILE).read() if os.path.exists(FINETUNE_FILE) else 'Never'}")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🔮 Get 6-Hour Forecast", use_container_width=True):
        with st.spinner("Fetching & predicting..."):
            results = predict_next_6_hours()
            st.session_state['forecast'] = results

    if st.button("🔧 Fine-tune Now", use_container_width=True):
        with st.spinner("Fine-tuning on last 30 days..."):
            run_finetune()
            st.success("Fine-tuning complete!")

    if 'forecast' in st.session_state:
        df = pd.DataFrame(st.session_state['forecast'])

        st.dataframe(df[['datetime', 'temperature', 'apparent', 'rain_prob', 'is_raining']],
                     use_container_width=True)

with col2:
    if 'forecast' in st.session_state:
        df_pred = pd.DataFrame(st.session_state['forecast'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_pred['datetime'], y=df_pred['temperature'],
            name='Temperature', line=dict(color='#ff6b35', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df_pred['datetime'], y=df_pred['apparent'],
            name='Feels Like', line=dict(color='#f7c59f', width=2, dash='dash')
        ))
        fig.update_layout(
            title='6-Hour Temperature Forecast',
            xaxis_title='Time', yaxis_title='°C',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)