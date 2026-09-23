"""Bounded background job registry for one local API process."""

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Lock
from uuid import uuid4

from app.agent import run_february, run_forecast
from app.schemas import ForecastJob, ForecastRequest
from app.services.storage import get_job_audit, save_job_audit
from app.services.weather import WeatherUnavailable

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="windflow")
_lock = Lock()
_jobs: dict[str, ForecastJob] = {}
logger = logging.getLogger(__name__)


def get_job(job_id: str) -> ForecastJob | None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            return job.model_copy(deep=True)
    return get_job_audit(job_id)


def submit(request: ForecastRequest, february: bool = False) -> ForecastJob:
    if february:
        request = request.model_copy(update={"issue_date": date(2026, 1, 31), "horizon_hours": 24})
    job = ForecastJob(
        job_id=str(uuid4()),
        total=28 if february else 1,
        kind="february" if february else "forecast",
        request=request,
    )
    with _lock:
        if sum(value.status in ("pending", "running") for value in _jobs.values()) >= 4:
            raise ValueError("Уже выполняются расчёты. Дождитесь их завершения.")
        if len(_jobs) >= 50:
            completed = next(
                (key for key, value in _jobs.items() if value.status in ("completed", "failed")),
                None,
            )
            if completed:
                del _jobs[completed]
        _jobs[job.job_id] = job

    def update(steps):
        with _lock:
            job.steps = [step.model_copy(deep=True) for step in steps]
            active = next((step for step in steps if step.status == "running"), None)
            if active:
                job.message = active.title

    def progress(current, total):
        with _lock:
            job.progress, job.total = current, total
            job.message = f"Готово {current} из {total} дней"

    def work():
        terminal_status = "failed"
        with _lock:
            job.status = "running"
        try:
            result = (
                run_february(request.turbine_id, request.data_mode, update, progress)
                if february
                else run_forecast(request, on_update=update)
            )
            with _lock:
                job.result, job.progress = result, job.total
                job.message = "Расчёт завершён"
            terminal_status = "completed"
        except (ValueError, FileNotFoundError, WeatherUnavailable) as error:
            with _lock:
                job.error, job.message = str(error), "Расчёт остановлен"
        except Exception:
            logger.exception("Forecast job %s failed", job.job_id)
            with _lock:
                job.error = "Не удалось завершить расчёт. Проверьте журнал сервера."
                job.message = "Ошибка расчёта"
        finally:
            with _lock:
                snapshot = job.model_copy(deep=True)
                snapshot.status = terminal_status
            try:
                save_job_audit(snapshot)
            except (OSError, sqlite3.Error):
                logger.exception("Could not persist job audit %s", job.job_id)
            with _lock:
                job.status = terminal_status

    _pool.submit(work)
    return get_job(job.job_id)
