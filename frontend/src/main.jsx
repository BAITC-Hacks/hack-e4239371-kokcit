import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Wind } from "lucide-react";
import "./styles.css";

const steps = [
  "Получить архивный прогноз погоды",
  "Проверить входные данные",
  "Подготовить признаки",
  "Рассчитать выработку",
  "Проверить результат",
  "Сохранить прогноз",
];

function App() {
  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><Wind size={22} /> WindFlow AI</div>
        <span className="status"><span /> Система готова</span>
      </header>

      <section className="intro">
        <p>Самрук-Казына · прогноз ВЭС</p>
        <h1>Прогноз выработки на 24–48 часов</h1>
        <p className="lead">Агент самостоятельно получает архивную погоду, запускает модель и проверяет результат.</p>
      </section>

      <section className="workspace">
        <div className="controls">
          <label>Турбина<select><option>Турбина 1</option><option>Турбина 2</option></select></label>
          <label>Дата прогноза<input type="date" defaultValue="2026-01-31" /></label>
          <label>Горизонт<select><option>24 часа</option><option>48 часов</option></select></label>
          <button><Activity size={18} /> Запустить расчёт</button>
        </div>

        <div className="panel">
          <div className="panel-title"><h2>Цикл агента</h2><span>6 этапов</span></div>
          <ol>{steps.map((step, index) => <li key={step}><b>{index + 1}</b><span>{step}</span><em>Ожидает</em></li>)}</ol>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
