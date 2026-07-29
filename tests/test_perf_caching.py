"""Regression checks for the inference-latency fixes.

Guards the three things that made /chat take ~23s:
1. Ollama keep_alive must be set (a 70B cold load costs 17-25s).
2. Retrievers/chains/vectorstore must be cached (else BM25 reindexes per request).
3. BM25 must honour its _documents_loaded flag.
"""

import asyncio
from unittest.mock import MagicMock

from src.api.dependencies import (
    _build_retriever,
    _build_vectorstore,
    create_retriever_for_method,
    get_chat_chain_for_method,
    get_settings,
    invalidate_retriever_caches,
)
from src.api.models import RetrievalMethod
from src.retrieval.hybrid import HybridRAGRetriever


def test_ollama_keep_alive_configured() -> None:
    settings = get_settings()
    assert settings.llm.ollama_keep_alive, "LLM would cold-load on every idle request"
    assert settings.embedding.ollama_keep_alive


def test_llm_factory_passes_keep_alive() -> None:
    from src.core.llm import LLMFactory

    llm = LLMFactory.create_ollama(model="llama3.2:3b", keep_alive="7m")
    assert llm.keep_alive == "7m"


def test_dependencies_are_cached() -> None:
    invalidate_retriever_caches()
    _build_vectorstore.cache_clear()
    settings = get_settings()
    vs = _build_vectorstore()

    a = create_retriever_for_method(RetrievalMethod.HYBRID, settings, vs)
    b = create_retriever_for_method(RetrievalMethod.HYBRID, settings, vs)
    assert a is b, "retriever rebuilt per request -> full BM25 reindex per query"

    c = create_retriever_for_method(RetrievalMethod.SEMANTIC, settings, vs)
    assert c is not a, "different methods must not share a retriever"

    assert get_chat_chain_for_method(
        RetrievalMethod.HYBRID, settings, vs
    ) is get_chat_chain_for_method(RetrievalMethod.HYBRID, settings, vs)

    invalidate_retriever_caches()
    assert create_retriever_for_method(RetrievalMethod.HYBRID, settings, vs) is not a


def test_bm25_loads_corpus_once() -> None:
    vs = MagicMock()
    vs.collection.get.return_value = {
        "documents": ["rektor unnes adalah martono", "kampus sekaran gunungpati"],
        "metadatas": [{"source": "a.md"}, {"source": "a.md"}],
    }
    retriever = HybridRAGRetriever(vectorstore=vs, k=2)

    async def run() -> None:
        await retriever._ensure_bm25_loaded()
        await retriever._ensure_bm25_loaded()
        await retriever._ensure_bm25_loaded()

    asyncio.run(run())
    assert vs.collection.get.call_count == 1, "corpus re-read on every query"

    retriever.invalidate_bm25_cache()
    asyncio.run(retriever._ensure_bm25_loaded())
    assert vs.collection.get.call_count == 2


if __name__ == "__main__":
    test_ollama_keep_alive_configured()
    test_llm_factory_passes_keep_alive()
    test_dependencies_are_cached()
    test_bm25_loads_corpus_once()
    print("all perf-regression checks passed")
