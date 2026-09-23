import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, CalendarRange, CheckCircle2, Download, Gauge, LoaderCircle, RefreshCw, Thermometer, Wind } from "lucide-react";
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./styles.css";

const API = "/api";

async function api(path, options) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Ошибка API: ${response.status}`);
  }
  return response.json();
}

function formatHour(value) {
  return new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function App() {
  const [turbine, setTurbine] = useState(1);
  const [issueDate, setIssueDate] = useState("2026-01-31");
  const [horizon, setHorizon] = useState(48);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [batchMessage, setBatchMessage] = useState("");
  const [exportUrl, setExportUrl] = useState("");

  const loadHistory = async () => {
    try { setHistory(await api("/forecasts?limit=8")); } catch { setHistory([]); }
  };

  useEffect(() => {
    loadHistory();
    api("/models/metrics").then(setMetrics).catch(() => setMetrics(null));
  }, []);

  const runForecast = async () => {
    setLoading(true); setError(""); setBatchMessage("");
    try {
      const data = await api("/forecasts", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ turbine_id: turbine, issue_date: issueDate, horizon_hours: horizon }),
      });
      setResult(data);
      setExportUrl(`${API}/forecasts/${data.run_id}/export.csv`);
      await loadHistory();
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  };

  const runFebruary = async () => {
    setLoading(true); setError(""); setBatchMessage("");
    try {
      const data = await api(`/forecasts/february?turbine_id=${turbine}`, { method: "POST" });
      setResult({ ...data, issue_date: "01-28.02.2026", horizon_hours: data.forecast.length, quality_warnings: [] });
      setExportUrl(`${API}/forecasts/february/export.csv?turbine_id=${turbine}`);
      setBatchMessage(`Сформировано ${data.generated_runs} ежедневных прогнозов и ${data.forecast.length} почасовых значений.`);
      await loadHistory();
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  };

  const openHistory = async (runId) => {
    setError(""); setBatchMessage("");
    try {
      const data = await api(`/forecasts/${runId}`);
      setResult(data);
      setExportUrl(`${API}/forecasts/${runId}/export.csv`);
    } catch (requestError) { setError(requestError.message); }
  };

  const chartData = useMemo(() => result?.forecast.map((point) => ({
    ...point,
    label: formatHour(point.timestamp),
    powerPercent: Math.round(point.normalized_power * 1000) / 10,
    confidenceRange: [Math.round(point.confidence_low * 1000) / 10, Math.round(point.confidence_high * 1000) / 10],
  })) || [], [result]);

  const peak = result ? Math.max(...result.forecast.map((point) => point.normalized_power)) : 0;
  const averageWind = result ? result.forecast.reduce((sum, point) => sum + point.wind_speed_100m_ms, 0) / result.forecast.length : 0;
  const metricKey = horizon === 48 ? "day_2" : "day_1";
  const selectedMetric = metrics?.[`turbine_${turbine}`]?.[metricKey];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><Wind size={22} /> WindFlow AI</div>
        <div className="system-status"><span /> Модели готовы</div>
      </header>

      <div className="page">
        <section className="page-heading">
          <div><p>Диспетчерская ВЭС</p><h1>Почасовой прогноз выработки</h1></div>
          <button className="secondary" onClick={loadHistory} title="Обновить историю"><RefreshCw size={17} /> Обновить</button>
        </section>

        <section className="layout">
          <aside className="sidebar">
            <h2>Параметры расчёта</h2>
            <label>Турбина<select value={turbine} onChange={(event) => setTurbine(Number(event.target.value))}><option value={1}>Турбина 1</option><option value={2}>Турбина 2</option></select></label>
            <label>Дата выпуска<input type="date" value={issueDate} onChange={(event) => setIssueDate(event.target.value)} /></label>
            <label>Горизонт<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}><option value={24}>24 часа</option><option value={48}>48 часов</option></select></label>
            <button onClick={runForecast} disabled={loading}>{loading ? <LoaderCircle className="spin" size={18} /> : <Activity size={18} />} Рассчитать прогноз</button>
            <button className="secondary full" onClick={runFebruary} disabled={loading}><CalendarRange size={18} /> Весь февраль 2026</button>

            <div className="model-quality">
              <span>Проверка на январе 2026</span>
              <strong>{selectedMetric ? `${(selectedMetric.mae * 100).toFixed(1)}% MAE` : "нет данных"}</strong>
              {selectedMetric && <small>R² {selectedMetric.r2.toFixed(2)} · {selectedMetric.model_profile}</small>}
            </div>

            <div className="history">
              <h3>Последние запуски</h3>
              {history.length === 0 && <p className="muted">История пока пуста</p>}
              {history.map((item) => <button className="history-row" key={item.run_id} onClick={() => openHistory(item.run_id)}><span>Т{item.turbine_id} · {item.horizon_hours} ч</span><time>{item.issue_date}</time></button>)}
            </div>
          </aside>

          <div className="content">
            {error && <div className="alert error">{error}</div>}
            {batchMessage && <div className="alert success">{batchMessage}</div>}
            {!result && <section className="empty-state"><Wind size={42} /><h2>Выберите параметры и запустите расчёт</h2><p>Агент загрузит архивный прогноз погоды и рассчитает мощность каждой турбины.</p></section>}

            {result && <>
              <section className="metrics">
                <article><Gauge /><div><span>Ожидаемая энергия</span><strong>{result.expected_normalized_energy.toFixed(2)}</strong><small>норм. турбино-часов</small></div></article>
                <article><Activity /><div><span>Пиковая мощность</span><strong>{(peak * 100).toFixed(1)}%</strong><small>от номинала</small></div></article>
                <article><Wind /><div><span>Средний ветер</span><strong>{averageWind.toFixed(1)}</strong><small>м/с на высоте 100 м</small></div></article>
                <article><Thermometer /><div><span>Прогнозных точек</span><strong>{result.forecast.length}</strong><small>почасовых значений</small></div></article>
              </section>

              <section className="chart-section">
                <div className="section-heading"><div><h2>Прогноз мощности</h2><p>Турбина {result.turbine_id}, выпуск {result.issue_date}</p></div><a className="download" href={exportUrl}><Download size={17} /> CSV</a></div>
                <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={chartData} margin={{ top: 10, right: 8, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="#e5eaec" vertical={false} /><XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={32} /><YAxis yAxisId="power" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} /><YAxis yAxisId="wind" orientation="right" tick={{ fontSize: 11 }} unit=" м/с" /><Tooltip /><Legend />
                  <Area yAxisId="power" type="monotone" dataKey="confidenceRange" stroke="none" fill="#dce9e6" name="Доверительный диапазон" /><Line yAxisId="power" type="monotone" dataKey="powerPercent" stroke="#087f6c" strokeWidth={2.5} dot={false} name="Мощность, %" /><Line yAxisId="wind" type="monotone" dataKey="wind_speed_100m_ms" stroke="#3378a6" strokeWidth={1.7} dot={false} name="Ветер, м/с" />
                </ComposedChart></ResponsiveContainer></div>
              </section>

              <section className="agent-section">
                <div className="section-heading"><div><h2>Журнал агента</h2><p>Автоматический цикл расчёта</p></div></div>
                <div className="agent-steps">{(result.steps.length ? result.steps : [{ id: "stored", title: "Результат загружен из истории", status: "completed" }]).map((step, index) => <div className="agent-step" key={step.id}><CheckCircle2 size={18} /><b>{index + 1}</b><span>{step.title}</span><em>Готово</em></div>)}</div>
              </section>

              <section className="table-section">
                <div className="section-heading"><div><h2>Почасовые значения</h2><p>{result.forecast.length} прогнозных точек</p></div></div>
                <div className="table-scroll"><table><thead><tr><th>Время</th><th>Мощность</th><th>Диапазон</th><th>Ветер 100 м</th><th>Температура</th></tr></thead><tbody>{result.forecast.map((point) => <tr key={point.timestamp}><td>{formatHour(point.timestamp)}</td><td><strong>{(point.normalized_power * 100).toFixed(1)}%</strong></td><td>{(point.confidence_low * 100).toFixed(0)}–{(point.confidence_high * 100).toFixed(0)}%</td><td>{point.wind_speed_100m_ms.toFixed(1)} м/с</td><td>{point.temperature_c.toFixed(1)} °C</td></tr>)}</tbody></table></div>
              </section>
            </>}
          </div>
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
