from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.getprojectroot import define_project_root_path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(define_project_root_path() / ".env"))
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    DOCUMENTS_DIR: Path = Field(default=define_project_root_path() / "data/input_docs")
    LLM_PROVIDER: str | None = Field(default=None)
    LLM_API_KEY: str | None = Field(default=None)
