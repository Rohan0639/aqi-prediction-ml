"""
validate_predictions.py
Dynamically validates AQI predictions against actual values in live_aqi_dataset.csv.
Outputs terminal results + two PNG charts saved to validation_results/
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH         = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'
LIVE_DATA_PATH     = 'data/live_aqi_dataset.csv'
OUT_DIR            = 'validation_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def aqi_color(aqi):
    if aqi is None:  return '#9ca3af'
    if aqi <= 50:    return '#22c55e'
    if aqi <= 100:   return '#eab308'
    if aqi <= 150:   return '#f97316'
    if aqi <= 200:   return '#ef4444'
    if aqi <= 300:   return '#a855f7'
    return '#9f1239'

def load_global_payload():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def load_station_payload(station_name):
    fname = station_name.lower().replace(' ', '_') + '.pkl'
    path  = os.path.join(STATION_MODELS_DIR, fname)
    if os.path.exists(path):
        return joblib.load(path)
    return load_global_payload()

# ── Main ──────────────────────────────────────────────────────────────────────
def run_validation():
    print("\n" + "="*60)
    print("  AQI PREDICTION VALIDATION")
    print("="*60)

    if not os.path.exists(LIVE_DATA_PATH):
        print(f"[ERROR] {LIVE_DATA_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(LIVE_DATA_PATH, parse_dates=['Date'])
    df = df.drop_duplicates(subset=['Date', 'Station'], keep='last')
    df = df.sort_values(['Station', 'Date'])

    all_dates = sorted(df['Date'].unique())
    print(f"\nDates in live_aqi_dataset.csv: {[str(d)[:10] for d in all_dates]}")

    if len(all_dates) < 2:
        print("[ERROR] Need at least 2 dates.")
        sys.exit(1)

    target_date = all_dates[-1]
    lag1_date   = all_dates[-2]
    lag2_date   = all_dates[-3] if len(all_dates) >= 3 else all_dates[-2]

    print(f"\n  Lag-2  date : {str(lag2_date)[:10]}")
    print(f"  Lag-1  date : {str(lag1_date)[:10]}")
    print(f"  Target date : {str(target_date)[:10]}  <-- comparing predictions here\n")

    global_payload = load_global_payload()
    if not global_payload:
        print("[ERROR] Global model not found.")
        sys.exit(1)

    features     = global_payload['features']
    name_to_code = {name: code for code, name in global_payload['station_mapping'].items()}

    target_rows = df[df['Date'] == target_date]
    results = []

    for _, actual_row in target_rows.iterrows():
        station = actual_row['Station']
        code    = name_to_code.get(station)
        if code is None:
            print(f"  [SKIP] {station}: not in model mapping")
            continue

        lag1_rows = df[(df['Station'] == station) & (df['Date'] == lag1_date)]
        lag2_rows = df[(df['Station'] == station) & (df['Date'] == lag2_date)]

        lag1_aqi = float(lag1_rows.iloc[0]['AQI']) if not lag1_rows.empty else float(actual_row['AQI'])
        lag2_aqi = float(lag2_rows.iloc[0]['AQI']) if not lag2_rows.empty else lag1_aqi
        rolling3 = (float(actual_row['AQI']) + lag1_aqi + lag2_aqi) / 3

        payload = load_station_payload(station)
        model   = payload['model']

        input_data = {
            'PM2.5':         float(actual_row.get('PM2.5', 0)   or 0),
            'PM10':          float(actual_row.get('PM10', 0)    or 0),
            'NO2':           float(actual_row.get('NO2', 0)     or 0),
            'SO2':           float(actual_row.get('SO2', 0)     or 0),
            'O3':            float(actual_row.get('O3', 0)      or 0),
            'CO':            float(actual_row.get('CO', 0)      or 0),
            'Temperature':   float(actual_row.get('Temperature', 25) or 25),
            'Humidity':      float(actual_row.get('Humidity', 60)    or 60),
            'Wind_Speed':    float(actual_row.get('Wind_Speed', 1)   or 1),
            'Rainfall':      float(actual_row.get('Rainfall', 0)     or 0),
            'Station_Code':  code,
            'AQI_Lag_1':     lag1_aqi,
            'AQI_Lag_2':     lag2_aqi,
            'AQI_Rolling_3': rolling3,
        }

        df_input   = pd.DataFrame([{f: input_data.get(f, 0) for f in features}])
        prediction = float(model.predict(df_input)[0])
        actual_aqi = float(actual_row['AQI'])
        error      = prediction - actual_aqi
        pct_error  = abs(error) / actual_aqi * 100 if actual_aqi != 0 else 0

        results.append({
            'Station':   station,
            'Lag-1':     lag1_aqi,
            'Lag-2':     lag2_aqi,
            'Actual':    actual_aqi,
            'Predicted': round(prediction, 1),
            'Error':     round(error, 1),
            'AbsError':  round(abs(error), 1),
            'PctError':  round(pct_error, 1),
        })

        model_src = 'Station-specific' if 'station_name' in payload else 'Global Fallback'
        status = "[OK]  " if abs(error) <= 20 else ("[WARN]" if abs(error) <= 40 else "[FAIL]")
        print(f"  {status}  {station:<30}  Actual={actual_aqi:>6.1f}  "
              f"Predicted={prediction:>6.1f}  Error={error:>+7.1f}  [{model_src}]")

    if not results:
        print("\n[ERROR] No predictions generated.")
        sys.exit(1)

    res_df = pd.DataFrame(results)
    mae  = res_df['AbsError'].mean()
    rmse = float((res_df['Error'] ** 2).mean() ** 0.5)
    mape = res_df['PctError'].mean()

    print(f"\n" + "-"*60)
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mape:.1f}%")
    print("-"*60 + "\n")

    # ── Chart 1: Bar — Actual vs Predicted ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    labels   = [s.replace(' SPCB', '') for s in res_df['Station']]
    x        = np.arange(len(labels))
    bw       = 0.35

    ax1 = axes[0]
    bars_a = ax1.bar(x - bw/2, res_df['Actual'],    bw, color=[aqi_color(v) for v in res_df['Actual']],
                     alpha=0.9, edgecolor='white', linewidth=0.5, label='Actual AQI')
    bars_p = ax1.bar(x + bw/2, res_df['Predicted'], bw, color=[aqi_color(v) for v in res_df['Predicted']],
                     alpha=0.55, edgecolor='white', linewidth=0.5, hatch='//', label='Predicted AQI')

    for b in bars_a:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f'{b.get_height():.0f}',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
    for b in bars_p:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f'{b.get_height():.0f}',
                 ha='center', va='bottom', color='#93c5fd', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', color='#cbd5e1', fontsize=9)
    ax1.set_ylabel('AQI Value', color='#94a3b8', fontsize=11)
    ax1.set_title(f'Actual vs Predicted AQI  ({str(target_date)[:10]})',
                  color='white', fontsize=13, fontweight='bold', pad=12)
    ax1.tick_params(colors='#94a3b8')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=9)

    # ── Chart 1: Error bars (right panel) ────────────────────────────────────
    ax2 = axes[1]
    err_colors = ['#22c55e' if abs(e) <= 20 else '#f97316' if abs(e) <= 40 else '#ef4444'
                  for e in res_df['Error']]
    bars_e = ax2.bar(x, res_df['Error'], color=err_colors, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    ax2.axhline(0, color='#60a5fa', linewidth=1.2, linestyle='--', alpha=0.7)

    for b, err in zip(bars_e, res_df['Error']):
        va = 'bottom' if err >= 0 else 'top'
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + (1.5 if err >= 0 else -1.5),
                 f'{err:+.1f}', ha='center', va=va, color='white', fontsize=8.5, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', color='#cbd5e1', fontsize=9)
    ax2.set_ylabel('Error  (Predicted - Actual)', color='#94a3b8', fontsize=11)
    ax2.set_title(f'Prediction Error per Station\nMAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.1f}%',
                  color='white', fontsize=13, fontweight='bold', pad=12)
    ax2.tick_params(colors='#94a3b8')
    ax2.legend(handles=[
        mpatches.Patch(color='#22c55e', label='<=20 (Good)'),
        mpatches.Patch(color='#f97316', label='21-40 (Acceptable)'),
        mpatches.Patch(color='#ef4444', label='>40 (Poor)'),
    ], facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=9)

    fig.suptitle('AQI Prediction Validation Report', color='white',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, 'validation_chart.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    print(f"[Saved] {p1}")

    # ── Chart 2: Scatter — Actual vs Predicted ────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(8, 7))
    fig2.patch.set_facecolor('#0f172a')
    ax3.set_facecolor('#1e293b')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#334155')

    ax3.scatter(res_df['Actual'], res_df['Predicted'],
                c=[aqi_color(v) for v in res_df['Actual']],
                s=180, edgecolors='white', linewidths=1.2, zorder=5)

    lo = min(res_df['Actual'].min(), res_df['Predicted'].min()) - 10
    hi = max(res_df['Actual'].max(), res_df['Predicted'].max()) + 10
    ax3.plot([lo, hi], [lo, hi], color='#60a5fa', linewidth=1.5,
             linestyle='--', label='Perfect Prediction', alpha=0.8)
    ax3.fill_between([lo, hi], [lo-20, hi-20], [lo+20, hi+20],
                     alpha=0.1, color='#22c55e', label='+-20 Tolerance')

    for _, row in res_df.iterrows():
        ax3.annotate(row['Station'].replace(' SPCB', ''),
                     (row['Actual'], row['Predicted']),
                     textcoords='offset points', xytext=(8, 4),
                     color='#cbd5e1', fontsize=8.5)

    ax3.set_xlabel('Actual AQI', color='#94a3b8', fontsize=12)
    ax3.set_ylabel('Predicted AQI', color='#94a3b8', fontsize=12)
    ax3.set_title(f'Actual vs Predicted  --  Scatter Plot\n'
                  f'Target: {str(target_date)[:10]}  |  MAE={mae:.1f}  RMSE={rmse:.1f}',
                  color='white', fontsize=13, fontweight='bold')
    ax3.tick_params(colors='#94a3b8')
    ax3.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=9)
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)

    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, 'scatter_chart.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    print(f"[Saved] {p2}")

    print(f"\n[DONE] Results saved to: {os.path.abspath(OUT_DIR)}/\n")
    print(res_df[['Station', 'Actual', 'Predicted', 'Error', 'PctError']].to_string(index=False))

if __name__ == '__main__':
    run_validation()
