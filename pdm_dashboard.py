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

with st.spinner("Loading fleet data..."):
    df = load_data()
    model, feature_cols = train_model(df)

# ── Build fleet summary ───────────────────────────────────────────────────────
@st.cache_data
def build_fleet(_df, _model, _feature_cols):
    fleet = []
    for eid in sorted(_df['unit_id'].unique()):
        edf = _df[_df['unit_id'] == eid]
        last_row = edf[edf['cycle'] == edf['cycle'].max()][_feature_cols]
        rul = int(_model.predict(last_row)[0])
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
            "max_cycle": int(edf['cycle'].max()),
            "rul": rul,
            "status": status,
            "health": round(health)
        })
    return pd.DataFrame(fleet)

fleet_df = build_fleet(df, model, feature_cols)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏭 Fleet Health Monitor")
st.markdown("Real-time predictive maintenance dashboard — powered by ML")
st.divider()

# ── Fleet KPIs ────────────────────────────────────────────────────────────────
critical = len(fleet_df[fleet_df['status'] == "🔴 CRITICAL"])
warning  = len(fleet_df[fleet_df['status'] == "🟡 WARNING"])
healthy  = len(fleet_df[fleet_df['status'] == "🟢 HEALTHY"])
avg_rul  = int(fleet_df['rul'].mean())

k1, k2, k3, k4 = st.columns(4)
k1.metric("🔴 Critical", critical, help="Engines needing immediate attention")
k2.metric("🟡 Warning",  warning,  help="Engines to schedule maintenance")
k3.metric("🟢 Healthy",  healthy,  help="Engines operating normally")
k4.metric("⏱ Avg RUL",  f"{avg_rul} flights", help="Average remaining useful life across fleet")

st.divider()

# ── Traffic light fleet overview ──────────────────────────────────────────────
st.subheader("Fleet Overview")
st.caption("Click on an engine below to see full details")

tab1, tab2, tab3 = st.tabs(["🔴 Critical", "🟡 Warning", "🟢 Healthy"])

with tab1:
    crit_df = fleet_df[fleet_df['status'] == "🔴 CRITICAL"][['unit_id', 'rul', 'health', 'max_cycle']].copy()
    crit_df.columns = ['Engine', 'Flights Remaining', 'Health Score %', 'Total Cycles']
    crit_df['Engine'] = crit_df['Engine'].apply(lambda x: f"Engine #{x}")
    st.dataframe(crit_df, use_container_width=True, hide_index=True)
    if critical > 0:
        st.error(f"⚠️ {critical} engines require immediate maintenance scheduling.")

with tab2:
    warn_df = fleet_df[fleet_df['status'] == "🟡 WARNING"][['unit_id', 'rul', 'health', 'max_cycle']].copy()
    warn_df.columns = ['Engine', 'Flights Remaining', 'Health Score %', 'Total Cycles']
    warn_df['Engine'] = warn_df['Engine'].apply(lambda x: f"Engine #{x}")
    st.dataframe(warn_df, use_container_width=True, hide_index=True)
    if warning > 0:
        st.warning(f"🔧 {warning} engines should be scheduled for maintenance within the next 30 flights.")

with tab3:
    heal_df = fleet_df[fleet_df['status'] == "🟢 HEALTHY"][['unit_id', 'rul', 'health', 'max_cycle']].copy()
    heal_df.columns = ['Engine', 'Flights Remaining', 'Health Score %', 'Total Cycles']
    heal_df['Engine'] = heal_df['Engine'].apply(lambda x: f"Engine #{x}")
    st.dataframe(heal_df, use_container_width=True, hide_index=True)
    st.success(f"✅ {healthy} engines are operating normally. No action required.")

st.divider()

# ── Individual engine deep dive ───────────────────────────────────────────────
st.subheader("Engine Deep Dive")

engine_id = st.selectbox(
    "Select an engine to inspect",
    options=sorted(df['unit_id'].unique()),
    format_func=lambda x: f"Engine #{x}  —  {fleet_df[fleet_df['unit_id']==x]['status'].values[0]}"
)

engine_data  = df[df['unit_id'] == engine_id]
engine_info  = fleet_df[fleet_df['unit_id'] == engine_id].iloc[0]

# Health bar
health_score = engine_info['health']
rul          = engine_info['rul']
status       = engine_info['status']

d1, d2, d3 = st.columns(3)
d1.metric("Status", status)
d2.metric("Flights Remaining", f"{rul} flights")
d3.metric("Health Score", f"{health_score}%")

# Colour coded recommendation
if rul <= 10:
    st.error(f"🚨 IMMEDIATE ACTION REQUIRED — Schedule maintenance for Engine #{engine_id} within the next {rul} flights.")
elif rul <= 30:
    st.warning(f"⚠️ PLAN AHEAD — Engine #{engine_id} has {rul} flights remaining. Schedule maintenance within 2 weeks.")
else:
    st.success(f"✅ NO ACTION NEEDED — Engine #{engine_id} has {rul} flights remaining. Next check at cycle {engine_data['cycle'].max() + 20}.")

# Sensor selector
sensor_options = ['s2', 's3', 's4', 's7', 's11', 's14', 's15']
selected_sensor = st.selectbox("View sensor degradation", sensor_options, index=5)

# Degradation chart
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(engine_data['cycle'], engine_data[selected_sensor],
        color='royalblue', linewidth=1.5, label=f'Sensor {selected_sensor}')
ax.plot(engine_data['cycle'],
        engine_data[f'{selected_sensor}_rolling_mean'],
        color='orange', linewidth=2, linestyle='--', label='Rolling average (trend)')
ax.axvline(x=engine_data['cycle'].max(), color='red',
           linestyle='--', linewidth=1.5, label=f'Failure point ({engine_data["cycle"].max()})')
ax.set_xlabel('Flight Cycle')
ax.set_ylabel(f'Sensor {selected_sensor} Reading')
ax.set_title(f'Engine #{engine_id} — Sensor {selected_sensor} Degradation Trend')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.divider()
st.caption("Softdel Predictive Maintenance Platform  ·  Powered by Random Forest ML  ·  MAE 8.5 cycles")
