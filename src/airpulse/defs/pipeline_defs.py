import dagster as dg

from airpulse.defs.postgres import postgres_resource


@dg.definitions
def defs() -> dg.Definitions:
    daily_job = dg.define_asset_job(
        name="daily_air_quality_pipeline",
        selection=[
            "raw_air_quality",
            "cleaned_air_quality",
            "model_predictions",
            "model_metrics",
        ],
    )

    hourly_schedule = dg.ScheduleDefinition(
        job=daily_job,
        cron_schedule="0 * * * *",  # hourly, matching EPA API update frequency
    )

    return dg.Definitions(
        jobs=[daily_job],
        schedules=[hourly_schedule],
        resources={"postgres": postgres_resource},
    )
