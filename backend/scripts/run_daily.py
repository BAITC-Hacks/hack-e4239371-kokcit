"""Run both turbines once, or every day at 23:59:59 in UTC+5."""

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta

from app.agent import run_forecast
from app.catalog import LOCAL_TIMEZONE
from app.schemas import ForecastRequest

logger = logging.getLogger(__name__)


def run_once(issue_date, mode="auto"):
    failed = False
    for turbine in (1, 2):
        try:
            result = run_forecast(
                ForecastRequest(
                    turbine_id=turbine, issue_date=issue_date, horizon_hours=48, data_mode=mode
                )
            )
            print(
                json.dumps(
                    {
                        "turbine": turbine,
                        "issue_date": str(issue_date),
                        "run_id": result.run_id,
                        "hours": len(result.forecast),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            logger.exception("Daily forecast failed for turbine %s", turbine)
            failed = True
            print(
                json.dumps(
                    {"turbine": turbine, "issue_date": str(issue_date), "error": str(error)},
                    ensure_ascii=True,
                ),
                flush=True,
            )
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-date", type=date.fromisoformat)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.loop:
        issue = args.issue_date or (datetime.now(LOCAL_TIMEZONE).date() - timedelta(days=1))
        raise SystemExit(0 if run_once(issue, "offline" if args.offline else "auto") else 1)
    if args.issue_date or args.offline:
        parser.error("--loop cannot be combined with --issue-date or --offline")
    while True:
        now = datetime.now(LOCAL_TIMEZONE)
        next_run = now.replace(hour=23, minute=59, second=59, microsecond=0)
        if now >= next_run:
            next_run += timedelta(days=1)
        print(f"Next daily run: {next_run.isoformat()}", flush=True)
        while datetime.now(LOCAL_TIMEZONE) < next_run:
            time.sleep(
                min(30, max(0.01, (next_run - datetime.now(LOCAL_TIMEZONE)).total_seconds()))
            )
        run_once(next_run.date())


if __name__ == "__main__":
    main()
