from __future__ import annotations

from dataclasses import dataclass

import pytest

from ra.retrieval.chroma_cache import ChromaCache
from ra.retrieval.unified import Paper


class _FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        for idx, paper_id in enumerate(ids):
            self.records[paper_id] = {
                "document": documents[idx],
                "metadata": metadatas[idx],
                "embedding": embeddings[idx],
            }

    def get(self, *, ids: list[str], include: list[str] | None = None) -> dict[str, object]:
        _ = include
        return {"ids": [paper_id for paper_id in ids if paper_id in self.records]}

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, object]:
        _ = query_embeddings
        _ = include
        ids = list(self.records.keys())[:n_results]
        metadatas = [self.records[paper_id]["metadata"] for paper_id in ids]
        documents = [self.records[paper_id]["document"] for paper_id in ids]
        distances = [0.05 for _ in ids]
        return {
            "ids": [ids],
            "metadatas": [metadatas],
            "documents": [documents],
            "distances": [distances],
        }


class _FakeClient:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(self, name: str) -> _FakeCollection:  # noqa: ARG002
        return self.collection


@dataclass
class _FakeChromaModule:
    client: _FakeClient

    def PersistentClient(self, path: str) -> _FakeClient:  # noqa: N802, ARG002
        return self.client



def _paper() -> Paper:
    return Paper(
        id="p1",
        title="Attention Is All You Need",
        abstract="Transformer architecture for sequence modeling.",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS",
        citation_count=123,
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
        source="semantic_scholar",
    )



def test_chroma_cache_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import ra.retrieval.chroma_cache as chroma_cache_module

    fake_client = _FakeClient()
    monkeypatch.setattr(
        chroma_cache_module,
        "chromadb",
        _FakeChromaModule(client=fake_client),
    )

    cache = ChromaCache(persist_directory=tmp_path / "chroma")

    assert cache.cache_paper(_paper()) is True
    assert cache.has_paper("p1") is True

    matches = cache.search_cached("transformer architecture", limit=5)
    assert len(matches) == 1
    assert matches[0]["id"] == "p1"
    assert matches[0]["title"] == "Attention Is All You Need"
    assert matches[0]["authors"] == ["Ashish Vaswani", "Noam Shazeer"]



def test_chroma_cache_gracefully_disables_when_chroma_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import ra.retrieval.chroma_cache as chroma_cache_module

    monkeypatch.setattr(chroma_cache_module, "chromadb", None)

    cache = ChromaCache(persist_directory=tmp_path / "chroma")

    assert cache.available is False
    assert cache.cache_paper(_paper()) is False
    assert cache.has_paper("p1") is False
    assert cache.search_cached("transformers", limit=3) == []
