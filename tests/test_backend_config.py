from pathlib import Path
import pytest
from backend.config import Settings


@pytest.fixture
def clean_settings():
    def _make(**overrides):
        return Settings(_env_file=None, **overrides)
    return _make

def test_settings_env_llm(monkeypatch,clean_settings) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER", value="Google")
        m.setenv(name="LLM_API_KEY", value="PASSWORD")
        settings = clean_settings()
        assert settings.LLM_PROVIDER == "Google"
        assert settings.LLM_API_KEY == "PASSWORD"


def test_settings_env_llm_not_found_wrong_env(monkeypatch,clean_settings) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER1", value="Google")
        m.setenv(name="LLM_API_KEY2", value="PASSWORD")
        settings = clean_settings()
        assert settings.LLM_PROVIDER == None
        assert settings.LLM_API_KEY == None


def test_settings_env_llm_not_found(monkeypatch,clean_settings) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER2", value="Goo")
        m.setenv(name="LLM_API_KEY3", value="PW")
        settings = clean_settings()
        assert settings.LLM_PROVIDER == None
        assert settings.LLM_API_KEY == None


def test_settings_env_qdrant_default(monkeypatch,clean_settings) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER2", value="Goo")
        m.setenv(name="LLM_API_KEY3", value="PW")
        settings = clean_settings()
        assert settings.QDRANT_HOST == "localhost"
        assert settings.QDRANT_PORT == 6333


def test_documents_dir_path(monkeypatch,clean_settings) -> None:
    with monkeypatch.context() as m:
        m.setenv(name="LLM_PROVIDER2", value="Goo")
        m.setenv(name="LLM_API_KEY3", value="PW")
        settings = clean_settings()
        assert isinstance(settings.DOCUMENTS_IN_DIR, Path)
        assert isinstance(settings.DOCUMENTS_OUT_DIR, Path)
        assert isinstance(settings.CHUNKS_OUT_DIR, Path)
