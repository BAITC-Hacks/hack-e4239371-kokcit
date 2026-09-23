import csv
import io
import json
import sqlite3
from datetime import UTC, datetime

from app.config import settings
from app.schemas import ForecastAccepted, ForecastPoint, ForecastSummary


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS forecast_runs (
                run_id TEXT PRIMARY KEY,
                turbine_id INTEGER NOT NULL,
                issue_date TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                expected_normalized_energy REAL NOT NULL,
                quality_warnings TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS forecast_points (
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                normalized_power REAL NOT NULL,
                confidence_low REAL NOT NULL,
                confidence_high REAL NOT NULL,
                wind_speed_100m_ms REAL NOT NULL,
                temperature_c REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES forecast_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_forecast_points_run
            ON forecast_points(run_id);
            """
        )


def save_forecast(result: ForecastAccepted) -> None:
    initialize_database()
    created_at = datetime.now(UTC).isoformat()
    with connect() as connection:
        previous_runs = connection.execute(
            """
            SELECT run_id FROM forecast_runs
            WHERE turbine_id = ? AND issue_date = ? AND horizon_hours = ?
            """,
            (result.turbine_id, result.issue_date.isoformat(), result.horizon_hours),
        ).fetchall()
        for previous_run in previous_runs:
            connection.execute(
                "DELETE FROM forecast_points WHERE run_id = ?",
                (previous_run["run_id"],),
            )
        connection.execute(
            """
            DELETE FROM forecast_runs
            WHERE turbine_id = ? AND issue_date = ? AND horizon_hours = ?
            """,
            (result.turbine_id, result.issue_date.isoformat(), result.horizon_hours),
        )
        connection.execute(
            """
            INSERT INTO forecast_runs VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.run_id,
                result.turbine_id,
                result.issue_date.isoformat(),
                result.horizon_hours,
                result.expected_normalized_energy,
                json.dumps(result.quality_warnings, ensure_ascii=False),
                created_at,
            ),
        )
        connection.executemany(
            """
            INSERT INTO forecast_points VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result.run_id,
                    point.timestamp.isoformat(),
                    point.normalized_power,
                    point.confidence_low,
                    point.confidence_high,
                    point.wind_speed_100m_ms,
                    point.temperature_c,
                )
                for point in result.forecast
            ],
        )


def list_forecasts(limit: int = 20) -> list[ForecastSummary]:
    initialize_database()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT run_id, turbine_id, issue_date, horizon_hours,
                   expected_normalized_energy, created_at
            FROM forecast_runs ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [ForecastSummary(**dict(row)) for row in rows]


def get_forecast(run_id: str) -> ForecastAccepted | None:
    initialize_database()
    with connect() as connection:
        run = connection.execute(
            "SELECT * FROM forecast_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            return None
        rows = connection.execute(
            "SELECT * FROM forecast_points WHERE run_id = ? ORDER BY timestamp", (run_id,)
        ).fetchall()

    points = []
    for row in rows:
        values = dict(row)
        values.pop("run_id")
        points.append(ForecastPoint(**values))
    return ForecastAccepted(
        run_id=run["run_id"],
        turbine_id=run["turbine_id"],
        issue_date=run["issue_date"],
        horizon_hours=run["horizon_hours"],
        expected_normalized_energy=run["expected_normalized_energy"],
        quality_warnings=json.loads(run["quality_warnings"]),
        steps=[],
        forecast=points,
    )


def forecast_to_csv(result: ForecastAccepted) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "timestamp",
            "turbine_id",
            "normalized_power",
            "confidence_low",
            "confidence_high",
            "wind_speed_100m_ms",
            "temperature_c",
        ]
    )
    for point in result.forecast:
        writer.writerow(
            [
                point.timestamp.isoformat(),
                result.turbine_id,
                round(point.normalized_power, 6),
                round(point.confidence_low, 6),
                round(point.confidence_high, 6),
                round(point.wind_speed_100m_ms, 3),
                round(point.temperature_c, 2),
            ]
        )
    return output.getvalue()


def february_forecast_to_csv(turbine_id: int) -> str | None:
    initialize_database()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT r.issue_date, p.timestamp, r.turbine_id, p.normalized_power,
                   p.confidence_low, p.confidence_high, p.wind_speed_100m_ms,
                   p.temperature_c
            FROM forecast_runs AS r
            JOIN forecast_points AS p ON p.run_id = r.run_id
            WHERE r.turbine_id = ?
              AND r.horizon_hours = 24
              AND r.issue_date BETWEEN '2026-01-31' AND '2026-02-27'
            ORDER BY p.timestamp
            """,
            (turbine_id,),
        ).fetchall()
    if not rows:
        return None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "issue_date",
            "timestamp",
            "turbine_id",
            "normalized_power",
            "confidence_low",
            "confidence_high",
            "wind_speed_100m_ms",
            "temperature_c",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["issue_date"],
                row["timestamp"],
                row["turbine_id"],
                round(row["normalized_power"], 6),
                round(row["confidence_low"], 6),
                round(row["confidence_high"], 6),
                round(row["wind_speed_100m_ms"], 3),
                round(row["temperature_c"], 2),
            ]
        )
    return output.getvalue()
