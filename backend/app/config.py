from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    app_name: str = "WindFlow AI"
    open_meteo_base_url: str = "https://previous-runs-api.open-meteo.com/v1/forecast"
    model_dir: Path = Path("models")
    database_path: Path = Path("windflow.sqlite3")


settings = Settings()
