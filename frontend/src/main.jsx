import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, CalendarRange, CheckCircle2, Circle, CircleX, Cloud, Database, Download, Gauge, LoaderCircle, RefreshCw, Thermometer, Wind } from "lucide-react";
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./styles.css";
import QualityReport from "./QualityReport";

const API = "/api";

async function api(path, options) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg).join("; ") : payload.detail;
    throw new Error(detail || `Ошибка API: ${response.status}`);
  }
  return response.json();
}

function formatHour(value) {
  return new Intl.DateTimeFormat("ru-RU", { timeZone: "Asia/Almaty", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(value.includes("+") || value.endsWith("Z") ? value : `${value}+05:00`));
}

function App() {
  const [turbine, setTurbine] = useState(1);
  const [issueDate, setIssueDate] = useState("2026-01-29");
  const [horizon, setHorizon] = useState(48);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [batchMessage, setBatchMessage] = useState("");
  const [exportUrl, setExportUrl] = useState("");
  const [dataMode, setDataMode] = useState("auto");
  const [job, setJob] = useState(null);
  const [health, setHealth] = useState(null);
  const [tab, setTab] = useState("forecast");

  const loadHistory = async () => {
    try {
      const [jobs, forecasts] = await Promise.all([
        api("/jobs?limit=8"),
        api("/forecasts?limit=20"),
      ]);
      const records = jobs.map((item) => {
        const request = item.request || {};
        const forecast = item.kind === "forecast" && item.status === "completed"
          ? forecasts.find((candidate) => (
              candidate.turbine_id === request.turbine_id
              && candidate.issue_date === request.issue_date
              && candidate.horizon_hours === request.horizon_hours
            ))
          : null;
        return { ...item, run_id: forecast?.run_id || null };
      });
      setHistory(records.length ? records : forecasts.map((item) => ({
        ...item,
        kind: "forecast",
        status: "completed",
        request: item,
      })));
    } catch {
      setHistory([]);
    }
  };

  const refreshDashboard = async () => {
    await loadHistory();
    api("/health").then(setHealth).catch(() => setHealth({ models_ready: false }));
    api("/models/metrics").then(setMetrics).catch(() => setMetrics(null));
  };

  useEffect(() => {
    loadHistory();
    api("/models/metrics").then(setMetrics).catch(() => setMetrics(null));
    api("/health").then(setHealth).catch(() => setHealth({ models_ready: false }));
  }, []);

  const runCalculation = async (february = false) => {
    setLoading(true); setError(""); setBatchMessage(""); setTab("forecast"); setJob(null);
    setResult(null); setExportUrl("");
    try {
      let current = await api(february ? "/jobs/february" : "/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ turbine_id: turbine, issue_date: issueDate, horizon_hours: horizon, data_mode: dataMode }),
      });
      setJob(current);
      const deadline = Date.now() + (february ? 20 * 60_000 : 3 * 60_000);
      while (current.status === "pending" || current.status === "running") {
        if (Date.now() > deadline) throw new Error("Расчёт выполняется дольше ожидаемого. Проверьте журнал сервера или откройте задачу позже.");
        await new Promise((resolve) => setTimeout(resolve, 450));
        current = await api(`/jobs/${current.job_id}`);
        setJob(current);
      }
      if (current.status === "failed") throw new Error(current.error || "Расчёт не завершён");
      const data = current.result;
      setResult(february ? { ...data, issue_date: "01–28.02.2026", horizon_hours: 672 } : data);
      setExportUrl(february ? `${API}/forecasts/february/export.csv?turbine_id=${data.turbine_id}` : `${API}/forecasts/${data.run_id}/export.csv`);
      if (february) setBatchMessage(`Сформировано ${data.generated_runs} ежедневных прогнозов и ${data.forecast.length} уникальных часов.`);
      await loadHistory();
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  };

  const openHistory = async (runId) => {
    setError(""); setBatchMessage(""); setJob(null); setTab("forecast"); setLoading(true);
    try {
      const data = await api(`/forecasts/${runId}`);
      setResult(data);
      setExportUrl(`${API}/forecasts/${runId}/export.csv`);
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
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
  const shownSteps = job?.steps || result?.steps || [];
  const sourceNames = { open_meteo: "Open-Meteo · получено по сети", cache: "Сохранённый архив погоды", bundled_archive: "Архив погоды из комплекта демо", provided: "Подготовленный архив", mixed: "Сеть и сохранённый архив", unknown: "Источник не записан" };
  const completedSteps = shownSteps.filter((step) => step.status === "completed").length;
  const jobProgress = job?.total > 1 ? Math.round((job.progress / job.total) * 100) : shownSteps.length ? Math.round((completedSteps / shownSteps.length) * 100) : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><Wind size={22} /> WindFlow AI</div>
        <div className={`system-status ${health?.models_ready ? "" : "unavailable"}`}><span />{health === null ? "Проверяем сервер…" : health.models_ready ? "Модели готовы" : "Проверьте подключение к серверу"}</div>
      </header>

      <div className="page">
        <section className="page-heading">
          <div><p>Диспетчерская ВЭС</p><h1>Почасовой прогноз выработки</h1></div>
          <button className="secondary" onClick={refreshDashboard} title="Обновить состояние"><RefreshCw size={17} /> Обновить</button>
        </section>

        <section className="layout">
          <aside className="sidebar">
            <h2>Параметры расчёта</h2>
            <label>Турбина<select disabled={loading} value={turbine} onChange={(event) => setTurbine(Number(event.target.value))}><option value={1}>Турбина 1</option><option value={2}>Турбина 2</option></select></label>
            <label>Дата выпуска<input disabled={loading} type="date" min="2025-12-31" value={issueDate} onChange={(event) => setIssueDate(event.target.value)} /></label>
            <label>Горизонт<select disabled={loading} value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}><option value={24}>24 часа</option><option value={48}>48 часов</option></select></label>
            <label>Погода<select disabled={loading} value={dataMode} onChange={(event) => setDataMode(event.target.value)}><option value="auto">Архив или загрузка из сети</option><option value="offline">Только сохранённый архив</option><option value="live">Загрузить из Open-Meteo</option></select></label>
            <p className="sidebar-note">Демо без интернета: выпуск 29.01.2026, прогноз на 30–31 января. Время в UTC+5.</p>
            <button onClick={() => runCalculation(false)} disabled={loading || !issueDate}>{loading ? <LoaderCircle className="spin" size={18} /> : <Activity size={18} />} Рассчитать прогноз</button>
            <button className="secondary full" onClick={() => runCalculation(true)} disabled={loading}><CalendarRange size={18} /> Февраль · 28 × 24 часа</button>

            <div className="model-quality">
              <span>Проверка на январе 2026</span>
              <strong>{selectedMetric ? `MAE ${selectedMetric.mae.toFixed(3)}` : "нет данных"}</strong>
              {selectedMetric && <small>R² {selectedMetric.r2.toFixed(3)} · {horizon === 48 ? "второй" : "первый"} день</small>}
              <button className="text-button" onClick={() => setTab("validation")}>Факт и прогноз →</button>
            </div>

            <div className="history">
              <h3>Задания агента</h3>
              {history.length === 0 && <p className="muted">История пока пуста</p>}
              {history.map((item) => {
                const request = item.request || item;
                const statusName = { completed: "Готово", failed: "Ошибка", running: "Выполняется", pending: "В очереди" }[item.status] || item.status;
                return <div className={`history-item ${item.status}`} key={item.job_id || item.run_id}>
                  <button disabled={loading || !item.run_id} className="history-row" onClick={() => item.run_id && openHistory(item.run_id)} title={item.error || statusName}>
                    <span>{item.kind === "february" ? "Февраль" : `Т${request.turbine_id} · ${request.horizon_hours} ч`}</span>
                    <time>{request.issue_date}</time>
                  </button>
                  <small>{statusName}{item.error ? ` · ${item.error}` : ""}</small>
                </div>;
              })}
            </div>
          </aside>

          <div className="content">
            <nav className="tabs" aria-label="Разделы"><button className={tab === "forecast" ? "" : "secondary"} onClick={() => setTab("forecast")}>Расчёт прогноза</button><button className={tab === "validation" ? "" : "secondary"} onClick={() => setTab("validation")}>Проверка качества</button></nav>
            {error && <div className="alert error">{error}</div>}
            {batchMessage && <div className="alert success">{batchMessage}</div>}
            {tab === "validation" && <QualityReport turbine={turbine} lead={horizon === 48 ? 2 : 1} metric={selectedMetric} />}
            {tab === "forecast" && <>
            {(job || shownSteps.length > 0) && <section className="agent-section" aria-live="polite">
              <div className="section-heading"><div><h2>Журнал агента</h2><p>{job?.message || "Сохранённый журнал выполнения"}{job?.total > 1 ? ` · ${job.progress}/${job.total} дней` : ""}</p></div><div className="job-status">{loading && <LoaderCircle className="spin" size={18} />}<strong>{jobProgress}%</strong></div></div>
              <div className="progress-track" aria-label={`Прогресс ${jobProgress}%`}><span style={{ width: `${jobProgress}%` }} /></div>
              <div className="agent-steps">{shownSteps.map((step, index) => <div className={`agent-step ${step.status}`} key={step.id}>
                {step.status === "completed" ? <CheckCircle2 size={18} /> : step.status === "running" ? <LoaderCircle className="spin" size={18} /> : step.status === "failed" ? <CircleX size={18} /> : <Circle size={18} />}
                <b>{index + 1}</b><span>{step.title}</span><em>{step.duration_ms != null ? `${(step.duration_ms / 1000).toFixed(2)} с` : ""}</em><small>{step.detail}</small>
              </div>)}</div>
            </section>}
            {!result && !loading && <section className="empty-state"><Wind size={42} /><h2>Прогноз для двух ветротурбин</h2><p>Запустите расчёт на выбранную дату. Демо на 29 января работает на сохранённой погоде; модель рассчитает результат заново.</p></section>}

            {result && <>
              <div className="result-context"><span><Cloud size={15} />{sourceNames[result.data_source] || result.data_source}</span><span><Database size={15} />{result.model_version} · UTC+5</span></div>
              {result.quality_warnings?.map((warning) => <div className="alert warning" key={warning}>{warning}</div>)}
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
                  <Area yAxisId="power" type="monotone" dataKey="confidenceRange" stroke="none" fill="#dce9e6" name="Оценочный диапазон ошибки" /><Line yAxisId="power" type="monotone" dataKey="powerPercent" stroke="#087f6c" strokeWidth={2.5} dot={false} name="Мощность, %" /><Line yAxisId="wind" type="monotone" dataKey="wind_speed_100m_ms" stroke="#3378a6" strokeWidth={1.7} dot={false} name="Ветер, м/с" />
                </ComposedChart></ResponsiveContainer></div>
                <p className="explanation">Диапазон рассчитан по прошлым ошибкам модели и не гарантирует покрытие будущих значений. Мощность нормализована: 100% соответствует номиналу турбины.</p>
                {result.timing_note && <details><summary>Время выпуска и источник погоды</summary><p className="explanation">{result.timing_note}</p></details>}
              </section>

              <section className="table-section">
                <div className="section-heading"><div><h2>Почасовые значения</h2><p>{result.forecast.length} прогнозных точек</p></div></div>
                <div className="table-scroll"><table><thead><tr><th>Время</th><th>Мощность</th><th>Диапазон</th><th>Ветер 100 м</th><th>Температура</th></tr></thead><tbody>{result.forecast.map((point) => <tr key={point.timestamp}><td>{formatHour(point.timestamp)}</td><td><strong>{(point.normalized_power * 100).toFixed(1)}%</strong></td><td>{(point.confidence_low * 100).toFixed(0)}–{(point.confidence_high * 100).toFixed(0)}%</td><td>{point.wind_speed_100m_ms.toFixed(1)} м/с</td><td>{point.temperature_c.toFixed(1)} °C</td></tr>)}</tbody></table></div>
              </section>
            </>}
            </>}
          </div>
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
