import pytest
from backend.config import Settings
from pathlib import Path

def test_settings_env_llm(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER", value="Google")
        m.setenv(name="LLM_API_KEY", value="PASSWORD")
        settings = Settings()
        assert settings.LLM_PROVIDER == "Google"
        assert settings.LLM_API_KEY =="PASSWORD"

def test_settings_env_llm_not_found_wrong_env(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER1", value="Google")
        m.setenv(name="LLM_API_KEY2", value="PASSWORD")
        settings = Settings()
        assert settings.LLM_PROVIDER == None
        assert settings.LLM_API_KEY ==None

def test_settings_env_llm_not_found() -> None:
    settings = Settings()
    assert settings.LLM_PROVIDER == None
    assert settings.LLM_API_KEY ==None

def test_settings_env_qdrant_default() -> None:
    settings = Settings()
    assert settings.QDRANT_HOST == "localhost"
    assert settings.QDRANT_PORT ==6333
    
def test_documents_dir_path() -> None:
    settings = Settings()
    assert isinstance(settings.DOCUMENTS_DIR, Path)
    
