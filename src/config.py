"""Application configuration sourced from environment variables."""

from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openaq_api_key: str | None = None
    request_timeout_seconds: int = 20

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        return cls(openaq_api_key=os.getenv("OPENAQ_API_KEY"))
