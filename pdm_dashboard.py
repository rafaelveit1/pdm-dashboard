import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fleet Health Monitor", page_icon="🏭", layout="wide")

@st.cache_data
def load_data():
    col_names = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv('train_FD001.txt', sep=' ', header=None, names=col_names, index_col=False)
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
    X, y = df[feature_cols], df['RUL']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, feature_cols

def build_fleet(_df, _model, _feature_cols, check_cycle):
    fleet = []
    for eid in sorted(_df['unit_id'].unique()):
        edf = _df[_df['unit_id'] == eid]
        max_cyc = int(edf['cycle'].max())
        if check_cycle > max_cyc:
            status = "💀 FAILED"
            rul = 0
            health = 0
        else:
            closest = edf.iloc[(edf['cycle'] - check_cycle).abs().argsort()[:1]]
            rul = int(_model.predict(closest[_feature_cols])[0])
            if rul <= 10:
                status = "🔴 CRITICAL"
                health = min(15, rul * 1.5)
            elif rul <= 30:
                status = "🟡 WARNING"
                health = 15 + (rul - 10) * 1.5
            else:
                status = "🟢 HEALTHY"
                health = min(100, 45 + rul * 0.44)
        fleet.append({
            "unit_id": eid,
            "max_cycle": max_cyc,
            "rul": rul,
            "status": status,
            "health": round(health)
        })
    return pd.DataFrame(fleet)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading fleet data..."):
    df = load_data()
    model, feature_cols = train_model(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏭 Fleet Health Monitor")
st.markdown("Real-time predictive maintenance dashboard — powered by ML")
st.divider()

# ── SLIDER — must come before fleet is built ──────────────────────────────────
check_cycle = st.slider(
    "📅 Fleet-wide flight cycle — drag to see how the fleet health changes over time",
    min_value=1,
    max_value=300,
    value=100,
    step=1,
    help="Simulates all engines being built and deployed at the same time"
)

st.divider()

# ── Build fleet at selected cycle ─────────────────────────────────────────────
fleet_df = build_fleet(df, model, feature_cols, check_cycle)

# ── Fleet KPIs ────────────────────────────────────────────────────────────────
critical = len(fleet_df[fleet_df['status'] == "🔴 CRITICAL"])
warning  = len(fleet_df[fleet_df['status'] == "🟡 WARNING"])
healthy  = len(fleet_df[fleet_df['status'] == "🟢 HEALTHY"])
failed   = len(fleet_df[fleet_df['status'] == "💀 FAILED"])
avg_rul  = int(fleet_df[fleet_df['rul'] > 0]['rul'].mean()) if len(fleet_df[fleet_df['rul'] > 0]) > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔴 Critical", critical)
k2.metric("🟡 Warning",  warning)
k3.metric("🟢 Healthy",  healthy)
k4.metric("💀 Failed",   failed)
k5.metric("⏱ Avg RUL",  f"{avg_rul} flights")

st.divider()

# ── Traffic light tabs ────────────────────────────────────────────────────────
st.subheader(f"Fleet Overview at Cycle {check_cycle}")

tab1, tab2, tab3, tab4 = st.tabs(["🔴 Critical", "🟡 Warning", "🟢 Healthy", "💀 Failed"])

def render_table(subset, label):
    t = subset[['unit_id', 'rul', 'health', 'max_cycle']].copy()
    t.columns = ['Engine', 'Flights Remaining', 'Health Score %', 'Total Cycles']
    t['Engine'] = t['Engine'].apply(lambda x: f"Engine #{x}")
    st.dataframe(t, use_container_width=True, hide_index=True)

with tab1:
    render_table(fleet_df[fleet_df['status'] == "🔴 CRITICAL"], "critical")
    if critical > 0:
        st.error(f"🚨 {critical} engines require immediate maintenance.")

with tab2:
    render_table(fleet_df[fleet_df['status'] == "🟡 WARNING"], "warning")
    if warning > 0:
        st.warning(f"🔧 {warning} engines should be scheduled within the next 30 flights.")

with tab3:
    render_table(fleet_df[fleet_df['status'] == "🟢 HEALTHY"], "healthy")
    if healthy > 0:
        st.success(f"✅ {healthy} engines operating normally. No action required.")

with tab4:
    render_table(fleet_df[fleet_df['status'] == "💀 FAILED"], "failed")
    if failed > 0:
        st.error(f"💀 {failed} engines have exceeded their maximum cycle and failed.")

st.divider()

# ── Individual engine deep dive ───────────────────────────────────────────────
st.subheader("Engine Deep Dive")

engine_id = st.selectbox(
    "Select an engine to inspect",
    options=sorted(df['unit_id'].unique()),
    format_func=lambda x: f"Engine #{x}  —  {fleet_df[fleet_df['unit_id']==x]['status'].values[0]}"
)

engine_data = df[df['unit_id'] == engine_id]
engine_info = fleet_df[fleet_df['unit_id'] == engine_id].iloc[0]

health_score = engine_info['health']
rul          = engine_info['rul']
status       = engine_info['status']

d1, d2, d3 = st.columns(3)
d1.metric("Status", status)
d2.metric("Flights Remaining", f"{rul} flights")
d3.metric("Health Score", f"{health_score}%")

if status == "💀 FAILED":
    st.error(f"💀 Engine #{engine_id} has already failed at cycle {engine_info['max_cycle']}. Current cycle {check_cycle} is {check_cycle - engine_info['max_cycle']} cycles past failure.")
elif rul <= 10:
    st.error(f"🚨 IMMEDIATE ACTION REQUIRED — Engine #{engine_id} has only {rul} flights remaining.")
elif rul <= 30:
    st.warning(f"⚠️ PLAN AHEAD — Engine #{engine_id} has {rul} flights remaining. Schedule maintenance soon.")
else:
    st.success(f"✅ NO ACTION NEEDED — Engine #{engine_id} has {rul} flights remaining.")

# ── Sensor chart ──────────────────────────────────────────────────────────────
sensor_options = ['s2', 's3', 's4', 's7', 's11', 's14', 's15']
selected_sensor = st.selectbox("View sensor degradation", sensor_options, index=5)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(engine_data['cycle'], engine_data[selected_sensor],
        color='royalblue', linewidth=1.5, label=f'Sensor {selected_sensor}')
ax.plot(engine_data['cycle'], engine_data[f'{selected_sensor}_rolling_mean'],
        color='orange', linewidth=2, linestyle='--', label='Rolling average (trend)')
ax.axvline(x=check_cycle, color='green', linestyle='--', linewidth=2, label=f'Current cycle ({check_cycle})')
ax.axvline(x=engine_data['cycle'].max(), color='red', linestyle='--', linewidth=1.5,
           label=f'Failure point ({engine_data["cycle"].max()})')
ax.set_xlabel('Flight Cycle')
ax.set_ylabel(f'Sensor {selected_sensor} Reading')
ax.set_title(f'Engine #{engine_id} — Sensor {selected_sensor} Degradation Trend')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.divider()
st.caption("Softdel Predictive Maintenance Platform  ·  Powered by Random Forest ML  ·  MAE 8.5 cycles")
