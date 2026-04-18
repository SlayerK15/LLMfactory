from collection_system.adapters.llm.cerebras_adapter import CerebrasAdapter
from collection_system.adapters.llm.failover_adapter import FailoverLLMAdapter
from collection_system.adapters.llm.groq_adapter import GroqAdapter
from collection_system.adapters.llm.ollama_adapter import OllamaAdapter
from collection_system.bootstrap import build_llm
from collection_system.infra.config import Settings


def _settings(**kwargs: object) -> Settings:
    defaults = {
        "GROQ_API_KEY": "groq-test",
        "CEREBRAS_API_KEY": "cerebras-test",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def test_build_llm_defaults_to_groq_plus_cerebras_fallback() -> None:
    llm = build_llm(_settings())

    assert isinstance(llm, FailoverLLMAdapter)
    assert isinstance(llm._primary, GroqAdapter)
    assert isinstance(llm._fallback, CerebrasAdapter)


def test_build_llm_explicit_cerebras_provider() -> None:
    llm = build_llm(_settings(llm_provider="cerebras"))

    assert isinstance(llm, CerebrasAdapter)


def test_build_llm_uses_configured_fallback_provider() -> None:
    llm = build_llm(_settings(llm_fallback_provider="ollama"))

    assert isinstance(llm, FailoverLLMAdapter)
    assert isinstance(llm._primary, GroqAdapter)
    assert isinstance(llm._fallback, OllamaAdapter)
