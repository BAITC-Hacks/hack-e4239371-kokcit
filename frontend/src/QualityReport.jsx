import React, { useEffect, useState } from "react";
import { BarChart3, CheckCircle2, Gauge, Target } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function QualityReport({ turbine, lead, metric }) {
  const [points, setPoints] = useState([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setPoints([]); setError("");
    fetch(`/api/models/validation?turbine_id=${turbine}&lead_days=${lead}`, { signal: controller.signal })
      .then(async (response) => { if (!response.ok) throw new Error("Отчёт проверки пока недоступен"); return response.json(); })
      .then((data) => setPoints(data.points.map((point) => ({ ...point, label: point.timestamp.slice(5, 16).replace("T", " ") }))))
      .catch((error) => { if (error.name !== "AbortError") setError(error.message); });
    return () => controller.abort();
  }, [turbine, lead]);
  if (!metric) return <div className="empty-state compact"><p>Загружается отчёт проверки модели…</p></div>;
  const previous = metric.previous_version;
  const windBaseline = metric.baselines?.wind_curve;
  const improvement = metric.improvement_vs_baseline?.wind_curve;
  const coverage = metric.interval?.coverage;
  const rows = [
    ["WindFlow · текущая модель", metric],
    ...(previous ? [["Предыдущая версия", previous]] : []),
    ...(metric.baselines ? [["Простая кривая мощности", metric.baselines.wind_curve], ["Средняя мощность", metric.baselines.mean]] : []),
  ];
  return <>
    <section className="quality-kpis">
      <article><Gauge /><div><span>MAE модели</span><strong>{metric.mae.toFixed(3)}</strong><small>шкала мощности 0–1</small></div></article>
      <article><BarChart3 /><div><span>Улучшение к baseline</span><strong>{improvement ? `${improvement.mae_reduction_percent.toFixed(1)}%` : "—"}</strong><small>снижение MAE</small></div></article>
      <article><Target /><div><span>R² на январе</span><strong>{metric.r2.toFixed(3)}</strong><small>объяснённая вариативность</small></div></article>
      <article><CheckCircle2 /><div><span>Покрытие диапазона</span><strong>{coverage != null ? `${(coverage * 100).toFixed(1)}%` : "—"}</strong><small>на {metric.samples} часах</small></div></article>
    </section>
    <section className="chart-section">
      <div className="section-heading"><div><h2>Проверка на январе 2026</h2><p>Турбина {turbine} · {lead === 1 ? "первый" : "второй"} день прогноза · {metric.samples} часов</p></div><span className="badge">R² {metric.r2.toFixed(3)}</span></div>
      <p className="explanation">Конфигурации внутри семейств моделей подбирались на декабре. Январь использован как последовательный development-validation период для сравнения версий; в признаки не входят будущая мощность и измеренный будущий ветер.</p>
      {error && <p className="alert error">{error}</p>}
      <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={points}>
        <CartesianGrid stroke="#e5eaec" vertical={false} /><XAxis dataKey="label" minTickGap={48} tick={{ fontSize: 11 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 11 }} /><Tooltip /><Legend />
        <Line dataKey="actual" stroke="#182f43" dot={false} strokeWidth={1.5} name="Факт" />
        <Line dataKey="prediction" stroke="#087f6c" dot={false} strokeWidth={1.6} name="Прогноз модели" />
        <Line dataKey="baseline" stroke="#b56a36" dot={false} strokeWidth={1.2} strokeDasharray="5 4" name="Кривая мощности" />
      </LineChart></ResponsiveContainer></div>
      <div className="table-scroll"><table><thead><tr><th>Метод</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr></thead><tbody>{rows.map(([label, values]) => <tr key={label}><td>{label}</td><td>{values.mae.toFixed(3)}</td><td>{values.rmse.toFixed(3)}</td><td>{values.r2.toFixed(3)}</td></tr>)}</tbody></table></div>
      <p className="explanation">MAE — средняя абсолютная ошибка мощности в шкале 0–1. R² — качество объяснения изменений мощности, не процент точности.{windBaseline && ` Модель снижает MAE относительно простой кривой мощности на ${improvement.mae_reduction_percent.toFixed(1)}%.`}</p>
    </section>
    <section className="method-band"><div><h2>Входные данные</h2><p>Архивный прогноз ветра и температуры, изменение погоды внутри прогнозируемого дня и календарные признаки. Будущие показания датчиков турбины в расчёт не входят.</p></div><div><h2>Диапазон ошибки</h2><p>Радиус получен по квантилю прошлых ошибок. Январское покрытие составляет {coverage != null ? `${(coverage * 100).toFixed(1)}%` : "—"}; для февраля оно рассчитывается после получения фактической мощности.</p></div><div><h2>Контроль качества</h2><p>Февральская production-модель обучена на доступной истории по 31 января. После поступления февральских измерений тот же отчёт используется для контроля качества и дрейфа.</p></div></section>
  </>;
}
