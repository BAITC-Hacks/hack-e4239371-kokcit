"""Download reports only from complete, persisted forecast results."""

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.services.downloads import report_content_disposition
from app.services.reporting import PDF_TYPE, XLSX_TYPE, Report, export_pdf, export_xlsx
from app.services.storage import get_february_forecasts, get_forecast

router = APIRouter(prefix="/api/forecasts", tags=["Reports"])
logger = logging.getLogger(__name__)


def _single_report(run_id: str) -> Report:
    result = get_forecast(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Прогноз не найден")
    try:
        return Report([result])
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _month_report(turbine_id: int) -> Report:
    try:
        results = get_february_forecasts(turbine_id)
        if not results:
            raise HTTPException(status_code=404, detail="Февральский прогноз не найден")
        return Report(results, monthly=True)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _download(report: Report, extension: str) -> Response:
    try:
        content = export_pdf(report) if extension == "pdf" else export_xlsx(report)
    except ImportError as error:
        logger.error("Reporting dependency unavailable: %s", type(error).__name__)
        raise HTTPException(
            status_code=503, detail="Экспорт временно недоступен на сервере"
        ) from error
    first = report.runs[0]
    disposition = report_content_disposition(
        first.turbine_id,
        extension,
        start_date=None if report.monthly else first.forecast[0].timestamp.date(),
        horizon_hours=None if report.monthly else first.horizon_hours,
    )
    return Response(
        content=content,
        media_type=PDF_TYPE if extension == "pdf" else XLSX_TYPE,
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@router.get("/february/report.pdf")
def february_pdf(turbine_id: int = Query(ge=1, le=2)):
    return _download(_month_report(turbine_id), "pdf")


@router.get("/february/export.xlsx")
def february_xlsx(turbine_id: int = Query(ge=1, le=2)):
    return _download(_month_report(turbine_id), "xlsx")


@router.get("/{run_id}/report.pdf")
def forecast_pdf(run_id: str):
    return _download(_single_report(run_id), "pdf")


@router.get("/{run_id}/export.xlsx")
def forecast_xlsx(run_id: str):
    return _download(_single_report(run_id), "xlsx")
