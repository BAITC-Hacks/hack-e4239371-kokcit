from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR.parent / ".env", extra="ignore")

    app_name: str = "WindFlow AI"
    open_meteo_base_url: str = "https://previous-runs-api.open-meteo.com/v1/forecast"
    model_dir: Path = BACKEND_DIR / "models"
    database_path: Path = BACKEND_DIR / "windflow.sqlite3"
    weather_cache_dir: Path = BACKEND_DIR / "data_cache" / "requests"
    demo_dir: Path = BACKEND_DIR / "demo"
    weather_timeout_seconds: float = Field(default=15, gt=0, le=120)
    weather_attempts: int = Field(default=3, ge=1, le=5)
    frontend_dir: Path = BACKEND_DIR.parent / "frontend" / "dist"


settings = Settings()
