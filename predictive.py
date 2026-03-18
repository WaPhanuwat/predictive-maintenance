import re
import os
import glob
import warnings
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kurtosis, skew
from sklearn.svm import OneClassSVM
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# ส่วนที่ 1: การตั้งค่า (Setup & Config)
# ============================================================

# ปรับเปลี่ยน DATA_DIR ตามความเหมาะสม
DATA_DIR = r"C:\Users\Acer\Desktop\predictive maintenance\data"
if not os.path.isdir(DATA_DIR):
    # Fallback สำหรับการทดสอบใน sandbox
    DATA_DIR = "/home/ubuntu/upload"
    if not os.path.isdir(DATA_DIR):
        print(f"Error: ไม่พบไดเรกทอรีข้อมูลที่: {DATA_DIR}")
        # sys.exit(1) # ปิดไว้เพื่อให้โค้ดรันต่อได้ถ้ามีไฟล์ในที่อื่น

OUT_DIR = os.path.join(DATA_DIR, 'pm_output_v6_final')
os.makedirs(OUT_DIR, exist_ok=True)

# ISO 10816-3 Boundaries (Velocity RMS mm/s)
ISO_ZONES  = {'A': (0.0, 2.3), 'B': (2.3, 4.5), 'C': (4.5, 7.1), 'D': (7.1, 9999.)}
ISO_COLORS = {'A': '#2ecc71', 'B': '#f1c40f', 'C': '#e67e22', 'D': '#e74c3c', 'N/A': '#95a5a6'}
ISO_DESC   = {'A': 'Good (A)', 'B': 'Acceptable (B)', 'C': 'Alert (C)', 'D': 'Danger (D)'}

# Health Score weights: 40% Velocity + 25% RMS + 20% Kurtosis + 15% Spectral
W_VELOCITY = 0.40
W_RMS      = 0.25
W_KURTOSIS = 0.20
W_SPECTRAL = 0.15

# RUL Config
RUL_BASELINE_HR  = 8760   # 1 ปี — ปรับตามอายุการใช้งานจริง
HS_CRITICAL      = 0.85   # Health Score ที่ถือว่าถึงจุดวิกฤต
V_DANGER         = 7.1    # Velocity RMS ที่ Zone D boundary

RISK_LEVELS = {1: '1_Normal', 2: '2_Warning', 3: '3_Alert', 4: '4_Critical'}
RISK_COLORS = {
    '1_Normal':   '#2ecc71',
    '2_Warning':  '#f39c12',
    '3_Alert':    '#e67e22',
    '4_Critical': '#e74c3c',
}

MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

# ============================================================
# ส่วนที่ 2: ฟังก์ชันพื้นฐาน (Parsing & Feature Extraction)
# ============================================================

def parse_file(path):
    """อ่านไฟล์ waveform คืน dict ที่มี timestamp, sr, times, amps"""
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
                mon  = MONTH_MAP.get(mon_s.lower(), 1)
                year = 2000 + int(y2)
                meas_dt = datetime(year, mon, int(d),
                                   *[int(x) for x in t_s.split(':')])
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
        times = np.array([x[0] for x in data_points])
        amps  = np.array([x[1] for x in data_points])
        diffs = np.diff(times)
        sr_hz = round(1000.0 / np.median(diffs[diffs > 0]))
        return {'dt': meas_dt, 'sr': sr_hz, 't': times, 'a': amps, 'n': len(amps)}
    except Exception as e:
        print(f"  [parse_file] Error {e}")
        return None

def extract_file_features(times, amps, sr_hz):
    """สกัด Feature ครบถ้วนระดับไฟล์"""
    a = amps - np.mean(amps)
    rms   = float(np.sqrt(np.mean(a**2)))
    peak  = float(np.max(np.abs(a)))
    cf    = peak / (rms + 1e-9)
    kurt  = float(kurtosis(a))
    skew_ = float(skew(a))

    yf  = rfft(a)
    xf  = rfftfreq(len(a), 1.0 / sr_hz)
    mag = np.abs(yf) ** 2
    f_max = min(1000.0, sr_hz * 0.45)

    mask_dom = (xf >= 5.0) & (xf <= f_max)
    dom_freq = float(xf[mask_dom][np.argmax(mag[mask_dom])]) if mask_dom.any() else 0.0

    e_low  = float(np.sum(mag[(xf >= 5)   & (xf < 100)]))
    e_mid  = float(np.sum(mag[(xf >= 100) & (xf < 500)]))
    e_high = float(np.sum(mag[(xf >= 500) & (xf <= f_max)]))
    e_tot  = e_low + e_mid + e_high + 1e-9

    psd_norm     = mag[mask_dom] / (mag[mask_dom].sum() + 1e-9)
    spec_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-9)))

    nyq = sr_hz / 2.0
    f_low_bp, f_high_bp = 10.0, min(1000.0, sr_hz * 0.45)
    if f_low_bp < nyq:
        b, af = signal.butter(4, [f_low_bp / nyq, f_high_bp / nyq], btype='band')
        accel = signal.filtfilt(b, af, a.astype(float)) * 9.81
    else:
        accel = a * 9.81
    vel   = cumulative_trapezoid(accel, dx=1.0 / sr_hz, initial=0)
    vel   = signal.detrend(vel)
    v_rms = float(np.sqrt(np.mean(vel**2)) * 1000.0)

    return {
        'RMS': rms, 'Peak': peak, 'CrestFactor': cf,
        'Kurtosis': kurt, 'Skewness': skew_,
        'DomFreq': dom_freq,
        'E_Low_pct':  e_low  / e_tot,
        'E_Mid_pct':  e_mid  / e_tot,
        'E_High_pct': e_high / e_tot,
        'SpecEntropy': spec_entropy,
        'VelocityRMS': v_rms,
    }

def classify_iso_zone(v_rms):
    if np.isnan(v_rms): return 'N/A'
    for zone, (lo, hi) in ISO_ZONES.items():
        if lo <= v_rms < hi: return zone
    return 'D'

def compute_health_score(v_rms, rms_norm, kurt_norm, spectral_shift_norm):
    hs = (W_VELOCITY * np.clip(v_rms / V_DANGER, 0, 1) +
          W_RMS      * np.clip(rms_norm, 0, 1)          +
          W_KURTOSIS * np.clip(kurt_norm, 0, 1)          +
          W_SPECTRAL * np.clip(spectral_shift_norm, 0, 1))
    return float(np.clip(hs, 0, 1))

def classify_risk(iso_zone, health_score):
    iso_min     = {'A': 1, 'B': 2, 'C': 3, 'D': 4}.get(iso_zone, 1)
    iso_ceiling = {'A': 2, 'B': 2, 'C': 3, 'D': 4}.get(iso_zone, 4)
    ml = 4 if health_score >= 0.85 else (3 if health_score >= 0.65
          else (2 if health_score >= 0.40 else 1))
    return RISK_LEVELS[min(max(ml, iso_min), iso_ceiling)]

# ============================================================
# ส่วนที่ 3: Global One-Class SVM (Anomaly Detection)
# ============================================================

IF_FEATURES = [
    'RMS', 'Peak', 'CrestFactor', 'Kurtosis', 'Skewness',
    'DomFreq', 'E_Low_pct', 'E_Mid_pct', 'E_High_pct',
    'SpecEntropy', 'VelocityRMS',
]

def apply_global_ocsvm(df):
    """เทรนและทำนาย Anomaly ด้วย Global OCSVM แยกตามประเภทเครื่องจักร"""
    results = []
    # สมมติประเภทเครื่องจักรจาก Machine_ID
    df['Machine_Type'] = df['Machine_ID'].apply(lambda x: 'Pump' if 'Pump' in x or 'CH' in x else 'General')
    
    for m_type, grp in df.groupby('Machine_Type'):
        X = grp[IF_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # nu=0.05 หมายถึงคาดว่าจะมี anomaly ประมาณ 5%
        model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
        model.fit(X_scaled)
        
        scores = model.decision_function(X_scaled)
        preds = model.predict(X_scaled)
        
        grp['OCSVM_Score'] = scores
        grp['OCSVM_Is_Anomaly'] = (preds == -1).astype(int)
        
        # Normalize score (0=ปกติ, 1=ผิดปกติ)
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            grp['OCSVM_Anomaly_norm'] = 1.0 - (scores - s_min) / (s_max - s_min)
        else:
            grp['OCSVM_Anomaly_norm'] = 0.0
            
        results.append(grp)
    
    return pd.concat(results).sort_values(['Machine_ID', 'Timestamp']).reset_index(drop=True)

# ============================================================
# ส่วนที่ 4: Bayesian Ridge Regression (Trend & RUL)
# ============================================================

def compute_bayesian_trend(grp):
    """วิเคราะห์แนวโน้มด้วย Bayesian Ridge Regression"""
    n = len(grp)
    t0 = grp['Timestamp'].iloc[0]
    days = np.array([(t - t0).total_seconds() / 86400 for t in grp['Timestamp']]).reshape(-1, 1)
    hs = grp['HealthScore_Final'].values
    
    model = BayesianRidge()
    model.fit(days, hs)
    
    # ทำนายไปข้างหน้า 180 วัน
    days_last = days[-1][0]
    horizons = np.linspace(0, days_last + 180, 100).reshape(-1, 1)
    y_mean, y_std = model.predict(horizons, return_std=True)
    
    # คำนวณ RUL (จุดที่ Mean + 1.96*std แตะ HS_CRITICAL คือจุดที่เร็วที่สุดที่อาจพัง)
    slope = model.coef_[0]
    hs_now = hs[-1]
    
    if slope > 1e-6:
        days_to_crit = (HS_CRITICAL - hs_now) / slope
        rul_hr = max(days_to_crit * 24, 0)
    else:
        days_to_crit = None
        rul_hr = RUL_BASELINE_HR
        
    return {
        'model': model,
        'slope': slope,
        'intercept': model.intercept_,
        'days': days,
        'hs': hs,
        'horizons': horizons,
        'y_mean': y_mean,
        'y_std': y_std,
        'rul_hr': rul_hr,
        'days_to_crit': days_to_crit,
        'n_points': n
    }

# ============================================================
# ส่วนที่ 5: การประมวลผลหลัก (Main Execution)
# ============================================================

print("  Predictive Maintenance v6 Final — Global OCSVM + Bayesian Ridge")
print("=" * 75)

# 1. อ่านไฟล์และสกัด Features
file_list = sorted(glob.glob(os.path.join(DATA_DIR, 'A_*.txt')))
if not file_list:
    print(f"Error: ไม่พบไฟล์ข้อมูลใน {DATA_DIR}")
    sys.exit(1)

records = []
for fpath in file_list:
    fname = os.path.basename(fpath)
    # สกัด Machine_ID (ปรับตามรูปแบบชื่อไฟล์ของคุณ)
    clean = re.sub(r'__?[A-Za-z]{3}\d{2}\.txt$', '', fname, flags=re.IGNORECASE)
    m = re.match(r'^(A_[A-Za-z0-9\-\s]+)', clean)
    mid = m.group(1).strip() if m else clean.split('_')[0]
    
    p = parse_file(fpath)
    if not p: continue
    
    feats = extract_file_features(p['t'], p['a'], p['sr'])
    feats.update({'Machine_ID': mid, 'File': fname, 'Timestamp': p['dt'], 'ISO_Zone': classify_iso_zone(feats['VelocityRMS'])})
    records.append(feats)

df = pd.DataFrame(records).sort_values(['Machine_ID', 'Timestamp']).reset_index(drop=True)

# 2. คำนวณ Health Score เบื้องต้น (Rule-based)
df_list = []
for mid, grp in df.groupby('Machine_ID'):
    grp = grp.copy()
    rms_base = grp['RMS'].iloc[0]
    rms_max = grp['RMS'].max()
    grp['RMS_norm'] = (grp['RMS'] - rms_base) / (rms_max - rms_base + 1e-9)
    grp['Kurt_norm'] = np.clip(np.abs(grp['Kurtosis']) / 10.0, 0.0, 1.0)
    mid_base = grp['E_Mid_pct'].iloc[0]
    grp['SpectralShift_norm'] = np.clip(np.abs(grp['E_Mid_pct'] - mid_base) * 3, 0.0, 1.0)
    
    grp['HealthScore_Rule'] = grp.apply(lambda r: compute_health_score(r['VelocityRMS'], r['RMS_norm'], r['Kurt_norm'], r['SpectralShift_norm']), axis=1)
    df_list.append(grp)
df = pd.concat(df_list)

# 3. ใช้ Global OCSVM
df = apply_global_ocsvm(df)

# 4. รวมผลลัพธ์เป็น HealthScore_Final (80% Rule + 20% OCSVM)
df['HealthScore_Final'] = (0.8 * df['HealthScore_Rule'] + 0.2 * df['OCSVM_Anomaly_norm']).clip(0, 1)
df['Risk_Level_Final'] = df.apply(lambda r: classify_risk(r['ISO_Zone'], r['HealthScore_Final']), axis=1)

# 5. วิเคราะห์แนวโน้ม Bayesian และสร้าง Dashboard
print(f"\nสร้าง Dashboard สำหรับ {df['Machine_ID'].nunique()} เครื่องจักร...")
n_mac = df['Machine_ID'].nunique()
fig = plt.figure(figsize=(20, 6 * n_mac))

for i, (mid, grp) in enumerate(df.groupby('Machine_ID')):
    trend = compute_bayesian_trend(grp)
    
    # กราฟ 1: Health Score Forecast พร้อม Confidence Band
    ax1 = fig.add_subplot(n_mac, 2, 2*i + 1)
    ax1.plot(trend['horizons'], trend['y_mean'], color='#2c3e50', lw=2, label='Predicted HS (Mean)')
    ax1.fill_between(trend['horizons'].flatten(), 
                     trend['y_mean'] - 1.96 * trend['y_std'], 
                     trend['y_mean'] + 1.96 * trend['y_std'], 
                     color='#3498db', alpha=0.2, label='95% Confidence Interval')
    
    # จุดข้อมูลจริง
    ax1.scatter(trend['days'], trend['hs'], color='#e74c3c', s=50, zorder=5, label='Actual Data')
    
    # เส้นเกณฑ์
    ax1.axhline(y=HS_CRITICAL, color='#c0392b', linestyle='--', label='Critical Threshold (0.85)')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'{mid}: Health Score Forecast & Uncertainty', fontweight='bold')
    ax1.set_xlabel('Days from Start')
    ax1.set_ylabel('Health Score')
    ax1.legend(loc='upper left', fontsize=8)
    
    # กราฟ 2: Anomaly Comparison (Global OCSVM)
    ax2 = fig.add_subplot(n_mac, 2, 2*i + 2)
    m_type = grp['Machine_Type'].iloc[0]
    global_data = df[df['Machine_Type'] == m_type]['OCSVM_Score']
    
    ax2.hist(global_data, bins=20, color='#bdc3c7', alpha=0.5, label=f'Global {m_type} Distribution')
    for score in grp['OCSVM_Score']:
        ax2.axvline(x=score, color='#e74c3c', lw=2, linestyle='-', alpha=0.8)
    ax2.axvline(x=grp['OCSVM_Score'].iloc[-1], color='#e74c3c', lw=3, label='Current Machine State')
    
    ax2.set_title(f'{mid}: Global Anomaly Comparison', fontweight='bold')
    ax2.set_xlabel('OCSVM Decision Score (Higher = More Normal)')
    ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'dashboard_v6_final.png'), dpi=150)
plt.close()

# 6. Export ผลลัพธ์
df.to_csv(os.path.join(OUT_DIR, 'predictive_results_v6.csv'), index=False, encoding='utf-8-sig')
print(f"\nเสร็จสิ้น! ผลลัพธ์ถูกบันทึกไว้ที่: {OUT_DIR}")
print(f"  - Dashboard: dashboard_v6_final.png")
print(f"  - Data: predictive_results_v6.csv")