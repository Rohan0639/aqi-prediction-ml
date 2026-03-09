import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts'
import './App.css'

const API = 'http://localhost:5000/api'

// ── Colour map ────────────────────────────────────────────────
const AQI_COLOR = {
  Good: '#3fb950', Satisfactory: '#58a6ff',
  Moderate: '#d29922', Poor: '#ff7b72',
  'Very Poor': '#f85149', Severe: '#da3633',
}
const MODEL_COLOR = {
  'Random Forest': '#3fb950',
  'XGBoost': '#58a6ff',
  'LightGBM': '#d29922',
  'Ensemble': '#a371f7',
}
const aqiColor = (cat) => AQI_COLOR[cat] ?? '#8b949e'
const aqiClass = (cat) => 'aqi-' + (cat ?? '').replace(' ', '_')

// ── useFetch ──────────────────────────────────────────────────
function useFetch(url) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const fetch_ = useCallback(() =>
    fetch(url).then(r => r.json()).then(setData).catch(e => setError(e.message))
    , [url])
  useEffect(() => { fetch_() }, [fetch_])
  return { data, error }
}

// ── Custom Tooltip ────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#161b22', border: '1px solid #30363d',
      borderRadius: 8, padding: '10px 14px', fontSize: '0.82rem'
    }}>
      <p style={{ color: '#8b949e', marginBottom: 6 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</strong>
        </p>
      ))}
    </div>
  )
}

// ── MetricCard ────────────────────────────────────────────────
function MetricCard({ label, value, sub, color = 'blue', badge, badgeCat, highlight }) {
  return (
    <div className={`metric-card ${color}`} style={highlight ? {
      boxShadow: '0 0 0 2px #a371f7, 0 4px 24px rgba(163,113,247,0.2)'
    } : {}}>
      {highlight && <div style={{
        position: 'absolute', top: 10, right: 12,
        fontSize: '0.68rem', color: '#a371f7', fontWeight: 700, letterSpacing: '0.06em'
      }}>★ BEST</div>}
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={highlight ? { color: '#a371f7' } : {}}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
      {badge && <div className={`metric-badge ${aqiClass(badgeCat ?? badge)}`}>{badge}</div>}
    </div>
  )
}

// ── MetricsRow — all 4 models ─────────────────────────────────
function MetricsRow() {
  const { data } = useFetch(`${API}/metrics`)
  if (!data) return null

  const get = (name) => data.find(d => d.model === name) ?? {}
  const rf = get('Random Forest')
  const xgb = get('XGBoost')
  const lgb = get('LightGBM')
  const ens = get('Ensemble')

  return (
    <>
      {/* Row 1 – RMSE */}
      <div style={{ marginBottom: 10 }}>
        <div className="card-title" style={{ marginBottom: 10 }}>📉 RMSE — lower is better</div>
        <div className="grid-4">
          <MetricCard label="Random Forest RMSE" value={rf.rmse} color="green" sub="Root Mean Squared Error" />
          <MetricCard label="XGBoost RMSE" value={xgb.rmse} color="blue" sub="Root Mean Squared Error" />
          <MetricCard label="LightGBM RMSE" value={lgb.rmse} color="yellow" sub="Root Mean Squared Error" />
          <MetricCard label="Ensemble RMSE" value={ens.rmse} color="purple" sub="RF×0.4 + LGB×0.4 + XGB×0.2" highlight />
        </div>
      </div>
      {/* Row 2 – R² */}
      <div style={{ marginBottom: 24 }}>
        <div className="card-title" style={{ marginBottom: 10 }}>📈 R² Score — higher is better</div>
        <div className="grid-4">
          <MetricCard label="Random Forest R²" value={rf.r2} color="green" sub="Coefficient of Determination" />
          <MetricCard label="XGBoost R²" value={xgb.r2} color="blue" sub="Coefficient of Determination" />
          <MetricCard label="LightGBM R²" value={lgb.r2} color="yellow" sub="Coefficient of Determination" />
          <MetricCard label="Ensemble R²" value={ens.r2} color="purple" sub="RF×0.4 + LGB×0.4 + XGB×0.2" highlight />
        </div>
      </div>
    </>
  )
}

// ── ForecastPanel — 4 models ──────────────────────────────────
function ForecastPanel() {
  const { data } = useFetch(`${API}/forecast`)
  if (!data) return (
    <div className="card">
      <div className="card-title">🌅 Tomorrow's Forecast</div>
      <p style={{ color: 'var(--muted)' }}>Loading…</p>
    </div>
  )

  const models = [
    { label: 'Random Forest', pred: data.rf_pred, cat: data.rf_category, color: '#3fb950' },
    { label: 'XGBoost', pred: data.xgboost_pred, cat: data.xgboost_category, color: '#58a6ff' },
    { label: 'LightGBM', pred: data.lgboost_pred, cat: data.lgboost_category, color: '#d29922' },
    { label: 'Ensemble ★', pred: data.ensemble_pred, cat: data.ensemble_category, color: '#a371f7' },
  ]

  return (
    <div className="card">
      <div className="card-title">🌅 Tomorrow's AQI Forecast — All Models</div>
      <div style={{ marginBottom: 16 }}>
        <span style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>Last known observation: </span>
        <strong>{data.current_station}</strong>
        <span style={{ color: 'var(--muted)' }}> ({data.current_date}) — AQI </span>
        <strong style={{ color: aqiColor(data.current_category) }}>{data.current_aqi}</strong>
        <span className={`metric-badge ${aqiClass(data.current_category)}`} style={{ marginLeft: 8 }}>
          {data.current_category}
        </span>
      </div>
      <div className="forecast-row">
        {models.map(m => (
          <div key={m.label} className="forecast-model"
            style={{
              borderTop: `2px solid ${m.color}`,
              boxShadow: m.label.includes('★') ? `0 0 12px ${m.color}33` : ''
            }}>
            <div className="model-name">{m.label}</div>
            <div className="big-aqi" style={{ color: aqiColor(m.cat) }}>{m.pred}</div>
            <span className={`metric-badge ${aqiClass(m.cat)}`}>{m.cat}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── PredictionChart — 4 lines ─────────────────────────────────
function PredictionChart() {
  const { data } = useFetch(`${API}/predictions`)
  if (!data) return <p style={{ textAlign: 'center', padding: 40, color: 'var(--muted)' }}>Loading…</p>

  const step = Math.max(1, Math.floor(data.length / 200))
  const sample = data.filter((_, i) => i % step === 0)

  return (
    <ResponsiveContainer width="100%" height={330}>
      <LineChart data={sample} margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 10 }}
          interval={Math.floor(sample.length / 8)} />
        <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: '0.82rem' }} />
        <Line type="monotone" dataKey="actual" stroke="#e6edf3" dot={false} strokeWidth={2} name="Actual AQI" />
        <Line type="monotone" dataKey="random_forest" stroke="#3fb950" dot={false} strokeWidth={1.2} name="Random Forest" strokeDasharray="4 2" />
        <Line type="monotone" dataKey="xgboost" stroke="#58a6ff" dot={false} strokeWidth={1.2} name="XGBoost" strokeDasharray="4 2" />
        <Line type="monotone" dataKey="lightgbm" stroke="#d29922" dot={false} strokeWidth={1.2} name="LightGBM" strokeDasharray="4 2" />
        <Line type="monotone" dataKey="ensemble" stroke="#a371f7" dot={false} strokeWidth={2} name="Ensemble ★" />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── ScatterPlot ───────────────────────────────────────────────
function ScatterPlot() {
  const { data } = useFetch(`${API}/predictions`)
  if (!data) return null

  const step = Math.max(1, Math.floor(data.length / 300))
  const ensPts = data.filter((_, i) => i % step === 0).map(d => ({ x: d.actual, y: d.ensemble }))
  const rfPts = data.filter((_, i) => i % step === 0).map(d => ({ x: d.actual, y: d.random_forest }))

  const allVals = data.map(d => d.actual)
  const domMin = Math.floor(Math.min(...allVals) / 50) * 50
  const domMax = Math.ceil(Math.max(...allVals) / 50) * 50
  const perfect = [{ x: domMin, y: domMin }, { x: domMax, y: domMax }]

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis type="number" dataKey="x" name="Actual"
          tick={{ fill: '#8b949e', fontSize: 11 }} domain={[domMin, domMax]}
          label={{ value: 'Actual AQI', fill: '#8b949e', position: 'insideBottom', dy: 14, fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="Predicted"
          tick={{ fill: '#8b949e', fontSize: 11 }} domain={[domMin, domMax]}
          label={{ value: 'Predicted AQI', fill: '#8b949e', angle: -90, position: 'insideLeft', dx: -4, fontSize: 11 }} />
        <Tooltip content={({ active, payload }) => {
          if (!active || !payload?.length) return null
          return <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: '8px 12px', fontSize: '0.8rem' }}>
            <p style={{ color: '#8b949e' }}>Actual: <strong>{payload[0]?.value?.toFixed(1)}</strong></p>
            <p style={{ color: '#8b949e' }}>Predicted: <strong>{payload[1]?.value?.toFixed(1)}</strong></p>
          </div>
        }} />
        <Legend wrapperStyle={{ fontSize: '0.82rem' }} />
        <Scatter name="Ensemble ★" data={ensPts} fill="#a371f7" fillOpacity={0.5} r={2} />
        <Scatter name="Random Forest" data={rfPts} fill="#3fb950" fillOpacity={0.4} r={2} />
        <Line type="linear" data={perfect} dataKey="y" stroke="#e6edf3"
          strokeDasharray="6 3" dot={false} name="Perfect fit" strokeWidth={1.5} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// ── TrendChart ────────────────────────────────────────────────
function TrendChart() {
  const { data } = useFetch(`${API}/trend`)
  if (!data) return <p style={{ textAlign: 'center', padding: 40, color: 'var(--muted)' }}>Loading…</p>
  const sample = data.filter((_, i) => i % 7 === 0)
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={sample} margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
        <defs>
          <linearGradient id="aqiGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#58a6ff" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 10 }} interval={52} />
        <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={[0, 'auto']} />
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: '0.83rem' }} />
        <Area type="monotone" dataKey="mean_aqi" stroke="#58a6ff" fill="url(#aqiGrad)"
          strokeWidth={1.2} dot={false} name="Daily Mean AQI" />
        <Line type="monotone" dataKey="ma30" stroke="#f85149" strokeWidth={2}
          dot={false} name="30-day MA" />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── MonthlyChart ──────────────────────────────────────────────
function MonthlyChart() {
  const { data } = useFetch(`${API}/monthly_aqi`)
  if (!data) return null
  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="month_name" tick={{ fill: '#8b949e', fontSize: 11 }} />
        <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="avg" name="Avg AQI" radius={[4, 4, 0, 0]} maxBarSize={40}>
          {data.map((e, i) => (
            <Cell key={i} fill={
              e.avg <= 50 ? '#3fb950' : e.avg <= 100 ? '#58a6ff' :
                e.avg <= 200 ? '#d29922' : e.avg <= 300 ? '#ff7b72' : '#f85149'
            } />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── DistributionChart ─────────────────────────────────────────
function DistributionChart() {
  const { data } = useFetch(`${API}/aqi_distribution`)
  if (!data) return null
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="category" tick={{ fill: '#8b949e', fontSize: 11 }} />
        <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="count" name="Days" radius={[4, 4, 0, 0]} maxBarSize={50}>
          {data.map((e, i) => <Cell key={i} fill={aqiColor(e.category)} fillOpacity={0.85} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── ModelComparisonBar — all 4 ────────────────────────────────
function ModelComparisonBar() {
  const { data } = useFetch(`${API}/metrics`)
  if (!data) return null

  const rmseData = data.map(d => ({ model: d.model.replace('Random Forest', 'RF').replace('LightGBM', 'LGB').replace('XGBoost', 'XGB'), rmse: d.rmse, mae: d.mae }))
  const r2Data = data.map(d => ({ model: d.model.replace('Random Forest', 'RF').replace('LightGBM', 'LGB').replace('XGBoost', 'XGB'), r2: d.r2 }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <div>
        <div style={{ fontSize: '0.82rem', color: 'var(--muted)', marginBottom: 10 }}>RMSE Comparison (lower = better)</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={rmseData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="model" tick={{ fill: '#8b949e', fontSize: 10 }} />
            <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="rmse" name="RMSE" radius={[4, 4, 0, 0]} maxBarSize={44}>
              {rmseData.map((_, i) => (
                <Cell key={i} fill={['#3fb950', '#58a6ff', '#d29922', '#a371f7'][i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <div style={{ fontSize: '0.82rem', color: 'var(--muted)', marginBottom: 10 }}>R² Score (higher = better)</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={r2Data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="model" tick={{ fill: '#8b949e', fontSize: 10 }} />
            <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={[0, 1]} tickFormatter={v => v.toFixed(2)} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="r2" name="R²" radius={[4, 4, 0, 0]} maxBarSize={44}>
              {r2Data.map((_, i) => (
                <Cell key={i} fill={['#3fb950', '#58a6ff', '#d29922', '#a371f7'][i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ── FeatureImportance ─────────────────────────────────────────
function FeatureImportance() {
  const { data } = useFetch(`${API}/feature_importance`)
  if (!data) return null
  const max = data[0]?.importance ?? 1
  return (
    <div>
      {data.map((d, i) => (
        <div key={i} className="imp-row">
          <div className="imp-label" title={d.feature}>{d.feature}</div>
          <div className="imp-bar-bg">
            <div className="imp-bar-fill" style={{ width: `${(d.importance / max) * 100}%` }} />
          </div>
          <div className="imp-val">{(d.importance * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  )
}

// ── StationTable ──────────────────────────────────────────────
function StationTable() {
  const { data } = useFetch(`${API}/station_stats`)
  if (!data) return <p style={{ color: 'var(--muted)' }}>Loading…</p>
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="station-table">
        <thead>
          <tr><th>Station</th><th>Mean AQI</th><th>Median</th><th>Max</th><th>Min</th><th>Records</th></tr>
        </thead>
        <tbody>
          {data.map((s, i) => (
            <tr key={i}>
              <td><strong>{s.Station}</strong></td>
              <td><span className={`metric-badge ${aqiClass(
                s.mean <= 50 ? 'Good' : s.mean <= 100 ? 'Satisfactory' :
                  s.mean <= 200 ? 'Moderate' : s.mean <= 300 ? 'Poor' : 'Very Poor'
              )}`}>{s.mean}</span></td>
              <td>{s.median}</td>
              <td style={{ color: '#ff7b72' }}>{s.max}</td>
              <td style={{ color: '#3fb950' }}>{s.min}</td>
              <td style={{ color: 'var(--muted)' }}>{s.count?.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Header ────────────────────────────────────────────────────
function Header() {
  const { data } = useFetch(`${API}/forecast`)
  const { data: metrics } = useFetch(`${API}/metrics`)
  const cat = data?.current_category ?? '---'
  const ensR2 = metrics?.find(d => d.model === 'Ensemble')?.r2
  const time = new Date().toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short',
    year: 'numeric', hour: '2-digit', minute: '2-digit'
  })

  return (
    <div className="header">
      <div className="header-left">
        <h1>🌫 AQI Prediction Dashboard</h1>
        <p>Hyderabad Air Quality Intelligence · {time} IST
          {ensR2 && <span style={{ color: '#a371f7', marginLeft: 12, fontWeight: 600 }}>
            ★ Ensemble R² = {ensR2}
          </span>}
        </p>
      </div>
      <div>
        <span className={`header-badge ${aqiClass(cat)}`}>● Current: {cat}</span>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('overview')
  const { data: health, error } = useFetch(`${API}/health`)

  if (error) return (
    <div className="loading-screen">
      <div style={{ fontSize: '2rem' }}>⚠️</div>
      <div style={{ fontWeight: 600 }}>Cannot reach backend</div>
      <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
        Run: <code>python dashboard/backend/api.py</code>
      </div>
    </div>
  )
  if (!health) return (
    <div className="loading-screen">
      <div className="spinner" />
      <div>Connecting to AQI backend…</div>
    </div>
  )

  return (
    <div className="dashboard">
      <Header />

      <div className="tabs">
        {[['overview', '📊 Overview'], ['predictions', '🎯 Predictions'],
        ['trend', '📈 Trends'], ['models', '🤖 Models'], ['stations', '🏭 Stations']
        ].map(([id, label]) => (
          <button key={id} className={`tab-btn ${tab === id ? 'active' : ''}`}
            onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && <>
        <MetricsRow />
        <div className="full-width"><ForecastPanel /></div>
        <div className="grid-2">
          <div className="card">
            <div className="card-title">📊 AQI Category Distribution (10 years)</div>
            <DistributionChart />
          </div>
          <div className="card">
            <div className="card-title">📅 Monthly Average AQI</div>
            <MonthlyChart />
          </div>
        </div>
      </>}

      {/* ── PREDICTIONS ── */}
      {tab === 'predictions' && <>
        <div className="card full-width">
          <div className="card-title">🎯 Actual vs Predicted AQI — All 4 Models (Test Set)</div>
          <PredictionChart />
        </div>
        <div className="card full-width">
          <div className="card-title">⬡ Scatter Plot: Ensemble vs Actual</div>
          <ScatterPlot />
        </div>
      </>}

      {/* ── TRENDS ── */}
      {tab === 'trend' && <>
        <div className="card full-width">
          <div className="card-title">📈 10-Year AQI Trend — Hyderabad</div>
          <TrendChart />
        </div>
        <div className="grid-2">
          <div className="card">
            <div className="card-title">📅 Monthly Seasonal Pattern</div>
            <MonthlyChart />
          </div>
          <div className="card">
            <div className="card-title">📊 AQI Category Distribution</div>
            <DistributionChart />
          </div>
        </div>
      </>}

      {/* ── MODELS ── */}
      {tab === 'models' && <>
        <MetricsRow />
        <div className="card full-width" style={{ marginBottom: 24 }}>
          <div className="card-title">📊 Model Performance Comparison — RMSE & R²</div>
          <ModelComparisonBar />
        </div>
        <div className="card full-width">
          <div className="card-title">🌟 Top 20 Feature Importances (Avg across RF + XGB + LGB)</div>
          <FeatureImportance />
        </div>
      </>}

      {/* ── STATIONS ── */}
      {tab === 'stations' && <>
        <div className="card full-width">
          <div className="card-title">🏭 Station-wise AQI Statistics</div>
          <StationTable />
        </div>
      </>}
    </div>
  )
}
