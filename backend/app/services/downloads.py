"""Readable export names shared by PDF, Excel and CSV downloads."""

from datetime import date
from urllib.parse import quote


def report_filename(
    turbine_id: int,
    extension: str,
    *,
    start_date: date | None = None,
    horizon_hours: int | None = None,
    ascii_only: bool = False,
) -> str:
    """Without a start date, name the fixed February 2026 monthly report."""
    if extension not in {"pdf", "xlsx", "csv"} or turbine_id not in (1, 2):
        raise ValueError("Некорректный формат отчёта или номер турбины")
    if start_date is not None and horizon_hours not in (24, 48):
        raise ValueError("Для почасового отчёта требуется горизонт 24 или 48 часов")
    if ascii_only:
        period = (
            f"Forecast from {start_date.isoformat()} - {horizon_hours}h"
            if start_date is not None
            else "February 2026"
        )
        return f"WindFlow - Turbine {turbine_id} - {period}.{extension}"
    period = (
        f"Прогноз с {start_date:%d.%m.%Y} — {horizon_hours} ч"
        if start_date is not None
        else "Февраль 2026"
    )
    return f"WindFlow — Турбина {turbine_id} — {period}.{extension}"


def report_content_disposition(
    turbine_id: int,
    extension: str,
    *,
    start_date: date | None = None,
    horizon_hours: int | None = None,
) -> str:
    options = {"start_date": start_date, "horizon_hours": horizon_hours}
    name = report_filename(turbine_id, extension, **options)
    fallback = report_filename(turbine_id, extension, ascii_only=True, **options)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name, safe='')}"
