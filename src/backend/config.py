from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.getprojectroot import define_project_root_path
from dataclasses import dataclass, field

class Settings(BaseSettings):
    root = define_project_root_path()
    model_config = SettingsConfigDict(env_file= root / ".env")
    QDRANT_HOST: str            = Field(default="localhost")
    QDRANT_PORT: int            = Field(default=6333)
    DOCUMENTS_IN_DIR: Path      = Field(default= root  / "data/input_docs")
    DOCUMENTS_OUT_DIR: Path     = Field(default= root  / "data/output_docs")
    DOC_MANIFEST_OUT_DIR: Path    = Field(defualt= root / "data/manifest_doc")
    CHUNKS_OUT_DIR: Path        = Field( root  / "data/chunks")
    LLM_PROVIDER: str | None    = Field(default=None)
    LLM_API_KEY: str | None     = Field(default=None)
    
    def model_post_init(self, context) -> None:
        for name in type(self).model_fields:
            if name.endswith("_DIR"):
                value = getattr(self,name)
                if isinstance(value,Path):
                    value.mkdir(parents=True, exist_ok=True)
        manifest_path = self.DOC_MANIFEST_OUT_DIR / "manifest.json"
        if not Path(manifest_path).exists():
            manifest_path.touch()
            

settings = Settings()

@dataclass(frozen=True, slots=True)
class Chunking_Constants:
    MAX_TOKENS: int = 512
    CEILING: int = MAX_TOKENS + 128
    TEXT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    THRESHOLD_SMALL_CHUNKS: int = 100