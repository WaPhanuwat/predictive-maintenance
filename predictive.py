import re
import os
import glob
import warnings
import sys
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# SECTION 0: Configuration & Environment Setup
# ============================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if not DATA_DIR: DATA_DIR = '.'

WINDOW_SEC = 0.05  # Rolling window size in seconds
TRAIN_RATIO = 0.8  # 80% for training, 20% for testing
OUT_DIR = os.path.join(DATA_DIR, 'pm_output_v3')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ISO 10816-3 Boundaries (Velocity RMS mm/s)
ISO_ZONES = {'A': (0.0, 2.3), 'B': (2.3, 4.5), 'C': (4.5, 7.1), 'D': (7.1, 9999.)}
ISO_COLORS = {'A': '#2ecc71', 'B': '#f1c40f', 'C': '#e67e22', 'D': '#e74c3c', 'N/A': '#95a5a6'}
ISO_DESC = {
    'A': 'Good (Condition A)',
    'B': 'Acceptable (Condition B)',
    'C': 'Alert (Condition C)',
    'D': 'Danger (Condition D)'
}
RISK_COLORS = {'1_Normal': '#2ecc71', '2_Warning': '#f39c12', '3_Alert': '#e67e22', '4_Critical': '#e74c3c'}

# ============================================================
# SECTION 1: Advanced Analysis Functions
# ============================================================

def get_dominant_freq(amp_g, sr_hz):
    n = len(amp_g)
    if n < 10: return 0.0
    data = amp_g - np.mean(amp_g)
    yf = rfft(data)
    xf = rfftfreq(n, 1/sr_hz)
    idx = np.argmax(np.abs(yf))
    return float(xf[idx])

def exponential_smoothing(series, alpha=0.05):
    return series.ewm(alpha=alpha, adjust=False).mean()

def compute_velocity_rms(amp_g, sr_hz):
    nyq = sr_hz / 2.0
    f_low, f_high = 10.0, min(1000.0, sr_hz * 0.45)
    if f_low >= nyq:
        accel_ms2 = (amp_g - np.mean(amp_g)) * 9.81
    else:
        b, a_filt = signal.butter(4, [f_low/nyq, f_high/nyq], btype='band')
        accel_ms2 = signal.filtfilt(b, a_filt, (amp_g - np.mean(amp_g)).astype(float)) * 9.81
    dt = 1.0 / sr_hz
    vel_ms = cumulative_trapezoid(accel_ms2, dx=dt, initial=0)
    vel_ms = signal.detrend(vel_ms)
    v_rms = np.sqrt(np.mean(vel_ms**2)) * 1000.0
    return float(v_rms)

def classify_iso_zone(v_rms):
    if np.isnan(v_rms): return 'N/A'
    for zone, (lo, hi) in ISO_ZONES.items():
        if lo <= v_rms < hi: return zone
    return 'D'

# ============================================================
# SECTION 2: Data Parsing & Loading
# ============================================================

MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

def parse_file(path):
    try:
        try:
            with open(path, 'r', encoding='utf-8') as f: lines = f.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='cp1252') as f: lines = f.readlines()
        meas_dt = None
        for l in lines[:8]:
            m = re.search(r'Date/Time:\s+(\d+)-(\w+)-(\d+)\s+(\d+:\d+:\d+)', l)
            if m:
                d, mon_s, y2, t_s = m.groups()
                mon, year = MONTH_MAP.get(mon_s.lower()), 2000 + int(y2)
                if mon: meas_dt = datetime.strptime(f"{year}-{mon:02d}-{int(d):02d} {t_s}", "%Y-%m-%d %H:%M:%S")
                break
        data_points = []
        for l in lines[8:]:
            l_fixed = re.sub(r'(\d)([-+]\d+)(?=[\s\r\n])', r'\1e\2', l)
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', l_fixed)
            if len(nums) < 8: continue
            try:
                pairs = [(float(nums[i]), float(nums[i+1])) for i in range(0, 8, 2)]
                for t, a in pairs:
                    if t >= 0: data_points.append((t, a))
            except: continue
        if not data_points: return None
        data_points.sort(key=lambda x: x[0])
        times, amps = np.array([x[0] for x in data_points]), np.array([x[1] for x in data_points])
        diffs = np.diff(times)
        median_diff = np.median(diffs[diffs > 0]) if any(diffs > 0) else 1.0
        sr_hz = round(1000.0 / median_diff)
        return {'dt': meas_dt, 'sr': sr_hz, 't': times, 'a': amps, 'n': len(amps)}
    except: return None

def get_machine_info(filename):
    base = os.path.basename(filename)
    clean_name = re.sub(r'__[A-Za-z]{3}\d{2}\.txt$', '', base, flags=re.IGNORECASE)
    m = re.match(r'^(A_[A-Za-z0-9\-\s]+)', clean_name)
    mid = m.group(1).strip() if m else clean_name.split('_')[0]
    if 'CH-06' in mid: mid = 'A_CH-06'
    elif 'Cooling' in mid.lower(): mid = 'A_Cooling Pump'
    elif 'Jockey' in mid.lower(): mid = 'A_Jockey pump'
    m_date = re.search(r'([A-Za-z]{3})(\d{2})\.txt$', base, re.IGNORECASE)
    bdate = datetime(2000+int(m_date.group(2)), MONTH_MAP.get(m_date.group(1).lower()), 1) if m_date else None
    return mid, bdate

# ============================================================
# SECTION 3: Processing Pipeline
# ============================================================

print("-" * 60); print("PHASE 1: Data Loading & Analysis"); print("-" * 60)

file_list = sorted(glob.glob(os.path.join(DATA_DIR, 'A_*.txt')))
if not file_list: print("Error: No A_*.txt files found!"); sys.exit(1)

all_dfs, machine_sr = [], {}
for fpath in file_list:
    fname = os.path.basename(fpath)
    mid, bdate = get_machine_info(fname)
    p = parse_file(fpath)
    if not p: continue
    fdt = p['dt'] or bdate or datetime(2024, 1, 1)
    if p['sr']: machine_sr[mid] = p['sr']
    dom_f = get_dominant_freq(p['a'], p['sr'] or 2500)
    df_tmp = pd.DataFrame({
        'Timestamp': pd.to_datetime(fdt) + pd.to_timedelta(p['t'], unit='ms'),
        'Amplitude': p['a'], 'Machine_ID': mid, 'File': fname, 'Dom_Freq': dom_f
    })
    all_dfs.append(df_tmp); print(f"  Loaded: {fname} ({p['n']:,} pts, SR={p['sr']}Hz)")

df = pd.concat(all_dfs, ignore_index=True).sort_values(['Machine_ID', 'Timestamp']).reset_index(drop=True)

print("\n" + "-" * 60); print("PHASE 2: Feature Engineering & ML Inference"); print("-" * 60)

def extract_features(group):
    mid = group.name
    sr = machine_sr.get(mid, 2500)
    w = max(4, int(WINDOW_SEC * sr))
    amp = group['Amplitude'].values
    res = group.copy()
    res['RMS'] = pd.Series(amp).pow(2).rolling(w, min_periods=1).mean().pow(0.5).values
    res['Peak'] = pd.Series(np.abs(amp)).rolling(w, min_periods=1).max().values
    res['Crest_Factor'] = res['Peak'] / (res['RMS'] + 1e-9)
    res['Kurtosis'] = pd.Series(amp).rolling(w, min_periods=4).apply(kurtosis, raw=True).values
    res['Skewness'] = pd.Series(amp).rolling(w, min_periods=4).apply(skew, raw=True).values
    res['Duration_hr'] = (group['Timestamp'] - group['Timestamp'].min()).dt.total_seconds().values / 3600
    res['Window_size'] = w
    return res

df = df.groupby('Machine_ID', group_keys=False).apply(extract_features).dropna(subset=['RMS', 'Kurtosis'])

FEATURE_COLS = ['RMS', 'Peak', 'Crest_Factor', 'Kurtosis', 'Skewness']
results = []
for mid, group in df.groupby('Machine_ID'):
    group = group.copy()
    model_path = os.path.join(MODEL_DIR, f"{mid}_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{mid}_scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        X_all = scaler.transform(group[FEATURE_COLS].fillna(0))
        group['Data_Split'] = 'Inference'
    else:
        split_idx = int(len(group) * TRAIN_RATIO)
        train_data = group.iloc[:split_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[FEATURE_COLS].fillna(0))
        X_all = scaler.transform(group[FEATURE_COLS].fillna(0))
        model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42).fit(X_train)
        joblib.dump(model, model_path); joblib.dump(scaler, scaler_path)
        group['Data_Split'] = ['Train' if i < split_idx else 'Test' for i in range(len(group))]
    
    group['Anomaly_Score'] = model.score_samples(X_all)
    group['Is_Anomaly'] = (model.predict(X_all) == -1).astype(int)
    w = int(group['Window_size'].iloc[0])
    group['Anomaly_Pct'] = group['Is_Anomaly'].rolling(w, min_periods=1).mean() * 100
    results.append(group)
df = pd.concat(results, ignore_index=True)

print("\n" + "-" * 60); print("PHASE 3: Degradation & RUL Prediction"); print("-" * 60)

final_results = []
for mid, group in df.groupby('Machine_ID'):
    group = group.copy()
    s_min, s_max = group['Anomaly_Score'].min(), group['Anomaly_Score'].max()
    group['DI_Raw'] = (1 - (group['Anomaly_Score'] - s_min) / (s_max - s_min)) if s_max != s_min else 0.0
    group['DI'] = exponential_smoothing(group['DI_Raw'], alpha=0.05)
    deg_rate = group['DI'].diff().rolling(100, min_periods=1).mean().clip(lower=1e-6)
    rem = (1.0 - group['DI']).clip(lower=0)
    group['RUL_hr_Raw'] = (rem / deg_rate).clip(upper=10000)
    group['RUL_hr'] = exponential_smoothing(group['RUL_hr_Raw'], alpha=0.05)
    final_results.append(group)

df = pd.concat(final_results, ignore_index=True)
df['Risk_Level'] = df.apply(lambda r: '4_Critical' if r['DI'] >= 0.85 or r['Anomaly_Pct'] >= 30 else ('3_Alert' if r['DI'] >= 0.65 or r['Anomaly_Pct'] >= 15 else ('2_Warning' if r['DI'] >= 0.40 or r['Anomaly_Pct'] >= 5 else '1_Normal')), axis=1)

# ============================================================
# SECTION 4: ISO 10816-3 & Reporting
# ============================================================

iso_recs = []
for mid, group in df.groupby('Machine_ID'):
    sr = machine_sr.get(mid, 2500)
    for fname, fgrp in group.groupby('File'):
        v_rms = compute_velocity_rms(fgrp['Amplitude'].values, sr)
        iso_recs.append({
            'Machine_ID': mid, 'Timestamp': fgrp['Timestamp'].max(), 
            'Velocity_RMS': v_rms, 'ISO_Zone': classify_iso_zone(v_rms),
            'Dom_Freq_ISO': fgrp['Dom_Freq'].iloc[0]
        })
iso_df = pd.DataFrame(iso_recs)

# Visualization: Comprehensive Dashboard
machines = df['Machine_ID'].unique()
fig, axes = plt.subplots(len(machines), 3, figsize=(22, 7*len(machines)))
if len(machines) == 1: axes = axes.reshape(1, -1)

for i, mid in enumerate(machines):
    mdf = df[df['Machine_ID'] == mid]
    idf = iso_df[iso_df['Machine_ID'] == mid]
    axes[i,0].scatter(mdf['Duration_hr'], mdf['RMS'], c=mdf['Risk_Level'].map(RISK_COLORS), s=2, alpha=0.4)
    axes[i,0].set_title(f'{mid}: RMS (G) & Risk Level', fontweight='bold')
    axes[i,1].plot(mdf['Duration_hr'], mdf['DI'], color='#8e44ad', linewidth=1.5, label='DI')
    axes[i,1].set_title(f'{mid}: Degradation Index', fontweight='bold')
    axes[i,2].bar(idf['Timestamp'].dt.strftime('%m-%d'), idf['Velocity_RMS'], color=[ISO_COLORS.get(z, '#95a5a6') for z in idf['ISO_Zone']], alpha=0.6)
    axes[i,2].set_title(f'{mid}: ISO 10816-3 Velocity', fontweight='bold')

plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'dashboard.png'), dpi=200); plt.close()

# Merge ISO Zone into Latest Status
latest = df.sort_values('Timestamp').groupby('Machine_ID').tail(1).copy()
latest_iso = iso_df.sort_values('Timestamp').groupby('Machine_ID').tail(1)
latest = latest.merge(latest_iso[['Machine_ID', 'Velocity_RMS', 'ISO_Zone']], on='Machine_ID', how='left')

# Console Summary Output
print("\n" + "=" * 60)
print("Machine Health Summary (ISO 10816-3)")
print("=" * 60)
for _, row in latest.iterrows():
    zone = row.get('ISO_Zone', 'N/A')
    print(f"Machine: {row['Machine_ID']}")
    print(f"  ISO Zone     : {zone} ({ISO_DESC.get(zone, 'Unknown')})")
    print(f"  Velocity RMS : {row.get('Velocity_RMS', 0.0):.4f} mm/s")
    print(f"  Risk Level   : {row.get('Risk_Level', 'Unknown')}")
    print(f"  RUL (Est.)   : {row.get('RUL_hr', 0.0):.1f} Hours")
    print(f"  Dom. Freq    : {row.get('Dom_Freq', 0.0):.1f} Hz")
    print("-" * 30)

# Export Final Reports
latest.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
print(f"\nResults saved to: {os.path.abspath(OUT_DIR)}")
