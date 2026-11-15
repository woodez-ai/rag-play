"""Privacy guard for the LlamaIndex rebuild: nothing may fall back to OpenAI (its default)."""

import pytest

pytest.importorskip("llama_index.core")


def test_settings_are_local_ollama():
    import ragplay_llamaindex  # noqa: F401  (sets Settings on import)
    from llama_index.core import Settings
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    assert isinstance(Settings.llm, Ollama) and Settings.llm.base_url.startswith("http://localhost")
    assert isinstance(Settings.embed_model, OllamaEmbedding)
    assert Settings.embed_model.base_url.startswith("http://localhost")


def test_vector_store_url_keeps_password_unmasked_in_memory(monkeypatch):
    """SQLAlchemy 2 masks passwords in str(URL); PGVectorStore str()s it -> auth failure."""
    from ragplay_llamaindex import app

    u = app._url("postgresql+psycopg2", "DB_USER", "DB_PASSWORD")
    assert "***" in str(u)  # the trap
    assert "***" not in u.render_as_string(hide_password=False)
