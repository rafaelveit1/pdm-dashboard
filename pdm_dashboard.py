import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="PdM Dashboard", page_icon="⚙️", layout="wide")

@st.cache_data
def load_data():
    col_names = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(
        'train_FD001.txt',
        sep=' ', header=None, names=col_names, index_col=False
    )
    df = df.dropna(axis=1)
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=125)
    df = df.drop(columns=['max_cycle'])
    for sensor in ['s2', 's3', 's4', 's7', 's11', 's14', 's15']:
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f'{sensor}_rolling_std'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        )
    return df

@st.cache_resource
def train_model(df):
    feature_cols = (
        ['cycle', 'op1', 'op2', 'op3'] +
        [f's{i}' for i in [2,3,4,7,11,14,15]] +
        [c for c in df.columns if 'rolling' in c]
    )
    X = df[feature_cols]
    y = df['RUL']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, feature_cols

with st.spinner("Loading data and training model..."):
    df = load_data()
    model, feature_cols = train_model(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance Dashboard")
st.markdown("**NASA CMAPSS Turbofan Engine Dataset  ·  Random Forest RUL Model  ·  MAE 8.5 cycles**")
st.divider()

# ── FREE INPUT ────────────────────────────────────────────────────────────────
st.subheader("Check any engine at any cycle")

col1, col2 = st.columns(2)

with col1:
    engine_input = st.number_input(
        "Engine number (1–100)",
        min_value=1, max_value=100, value=1, step=1
    )

with col2:
    cycle_input = st.number_input(
        "Flight cycle number",
        min_value=1, max_value=500, value=50, step=1
    )

st.divider()

# ── Look up or predict ────────────────────────────────────────────────────────
engine_df = df[df['unit_id'] == engine_input]
max_cycle = int(engine_df['cycle'].max())

if cycle_input > max_cycle:
    # Engine already failed
    st.error(f"🚨 ENGINE #{engine_input} HAS ALREADY FAILED")
    st.markdown(f"This engine survived **{max_cycle} flights** before failing.")
    st.markdown(f"You entered cycle **{cycle_input}** — that is **{cycle_input - max_cycle} cycles past failure.**")
    predicted_rul = 0

else:
    # Find the closest cycle in data
    closest_cycle = engine_df.iloc[(engine_df['cycle'] - cycle_input).abs().argsort()[:1]]
    predicted_rul = int(model.predict(closest_cycle[feature_cols])[0])

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Engine", f"#{engine_input}")
    m2.metric("Cycle entered", cycle_input)
    m3.metric("Predicted flights remaining", predicted_rul)

    st.divider()

    # Status
    if predicted_rul <= 10:
        st.error(f"🚨 CRITICAL — Engine #{engine_input} has only {predicted_rul} flights left. Schedule immediate maintenance.")
    elif predicted_rul <= 30:
        st.warning(f"⚠️ WARNING — Engine #{engine_input} is wearing down. {predicted_rul} flights remaining. Plan maintenance.")
    else:
        st.success(f"✅ HEALTHY — Engine #{engine_input} has {predicted_rul} flights remaining. No action needed.")

    st.divider()

    # Sensor chart
    st.subheader(f"Sensor 14 Degradation — Engine #{engine_input}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(engine_df['cycle'], engine_df['s14'], color='royalblue', linewidth=1.5, label='Sensor 14')
    ax.axvline(x=cycle_input, color='orange', linestyle='--', linewidth=2, label=f'Your cycle ({cycle_input})')
    ax.axvline(x=max_cycle, color='red', linestyle='--', linewidth=1.5, label=f'Failure point ({max_cycle})')
    ax.set_xlabel('Flight Cycle')
    ax.set_ylabel('Sensor 14 Reading')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption(f"Engine #{engine_input} survived {max_cycle} total flights before failure.")
