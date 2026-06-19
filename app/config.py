from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str
    supabase_secret_key: str

    openai_api_key: str

    allowed_origins: str = ""

    max_video_duration_seconds: int = 600
    max_requests_per_minute: int = 20
    environment: str = "production"
    instagram_cookies_path: str | None = None

    class Config:
        env_file = ".env"
        extra = "forbid"

    @property
    def allowed_origins_list(self) -> list[str]:
        if not self.allowed_origins:
            return []
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()