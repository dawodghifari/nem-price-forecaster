"""Settings, read from the environment with .env as a fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    region: str
    spike_threshold: float
    user_agent: str
    pg_user: str
    pg_password: str
    pg_host: str
    pg_port: int
    pg_database: str

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def load_settings() -> Settings:
    data_dir = Path(os.getenv("NEMFC_DATA_DIR", REPO_ROOT / "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    return Settings(
        data_dir=data_dir,
        region=os.getenv("NEMFC_REGION", "NSW1"),
        spike_threshold=float(os.getenv("NEMFC_SPIKE_THRESHOLD", "300")),
        user_agent=os.getenv("NEMFC_USER_AGENT", "nem-price-forecaster/0.1"),
        pg_user=os.getenv("POSTGRES_USER", "nemfc"),
        pg_password=os.getenv("POSTGRES_PASSWORD", "nemfc"),
        pg_host=os.getenv("POSTGRES_HOST", "localhost"),
        pg_port=int(os.getenv("POSTGRES_PORT", "5432")),
        pg_database=os.getenv("POSTGRES_DB", "nemfc"),
    )
