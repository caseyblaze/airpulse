import dagster as dg

from airpulse.defs.postgres import postgres_resource


@dg.definitions
def defs() -> dg.Definitions:
    hourly_job = dg.define_asset_job(
        name="hourly_air_quality_pipeline",
        selection=[
            "raw_air_quality",
            "cleaned_air_quality",
            "air_quality_history",
            "model_predictions",
            "model_metrics",
        ],
    )

    hourly_schedule = dg.ScheduleDefinition(
        job=hourly_job,
        cron_schedule="0 * * * *",  # hourly, matching EPA API update frequency
    )

    return dg.Definitions(
        jobs=[hourly_job],
        schedules=[hourly_schedule],
        resources={"postgres": postgres_resource},
    )
