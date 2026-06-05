"""Alerting via a Slack Incoming Webhook URL held in a Dagster+ env var.

The webhook URL is read at runtime from SLACK_WEBHOOK_URL (set in Dagster+ prod
scope, never committed). The sensor no-ops gracefully if it is unset so local
and CI loads don't fail.
"""

import os

import dagster as dg
import requests

SLACK_WEBHOOK_ENV = "SLACK_WEBHOOK_URL"


def build_failure_text(job_name: str, run_id: str, error: str) -> str:
    return (
        ":rotating_light: *airpulse run failed*\n"
        f"*Job:* {job_name}\n"
        f"*Run ID:* `{run_id}`\n"
        f"*Error:* {error}"
    )


@dg.run_failure_sensor(
    default_status=dg.DefaultSensorStatus.RUNNING,
    description="POST a Slack alert via incoming webhook whenever a run fails "
    "(includes blocking data-quality / drift check failures).",
)
def slack_on_run_failure(context: dg.RunFailureSensorContext) -> None:
    webhook_url = os.getenv(SLACK_WEBHOOK_ENV)
    if not webhook_url:
        context.log.warning(
            f"{SLACK_WEBHOOK_ENV} not set; skipping Slack alert."
        )
        return

    run = context.dagster_run
    error = getattr(context.failure_event, "message", None) or "Run failed"
    text = build_failure_text(run.job_name, run.run_id, error)

    resp = requests.post(webhook_url, json={"text": text}, timeout=15)
    resp.raise_for_status()
    context.log.info("Posted run-failure alert to Slack.")
