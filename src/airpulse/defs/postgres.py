import os

import dagster as dg
from sqlalchemy import Engine, create_engine


class PostgresResource(dg.ConfigurableResource):
    host: str
    port: int
    db: str
    user: str
    password: str

    def get_engine(self) -> Engine:
        url = (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )
        return create_engine(url)


postgres_resource = PostgresResource(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    db=os.getenv("POSTGRES_DB", "airquality"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", ""),
)
