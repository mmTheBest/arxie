from __future__ import annotations

import json

import httpx

from paperbase.search.query_builder import build_search_query
from paperbase.search.runtime import ElasticsearchSearchBackend


def test_elasticsearch_backend_creates_missing_index_on_404() -> None:
    seen_requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((request.method, request.url.path))
        if request.method == "HEAD":
            return httpx.Response(404)
        if request.method == "PUT":
            return httpx.Response(200, json={"acknowledged": True})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = ElasticsearchSearchBackend(base_url="http://search.local", client=client)

    backend.ensure_index("paperbase-papers", {"mappings": {"properties": {}}})

    assert seen_requests == [
        ("HEAD", "/paperbase-papers"),
        ("PUT", "/paperbase-papers"),
    ]


def test_elasticsearch_backend_posts_top_level_hybrid_search_body() -> None:
    captured_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {"_source": {"paper_id": "paper-1", "title": "scLong"}},
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = ElasticsearchSearchBackend(base_url="http://search.local", client=client)

    query = build_search_query(
        query_text="gene regulation",
        filters={"collection_ids": ["collection-1"]},
        embedding_vector=[0.1, 0.2, 0.3],
        k=5,
    )
    results = backend.search("paperbase-papers", query, 5)

    assert results == [{"paper_id": "paper-1", "title": "scLong"}]
    assert captured_payloads == [
        {
            "size": 5,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "gene regulation",
                                "fields": [
                                    "title^3",
                                    "abstract^2",
                                    "section_title^2",
                                    "text",
                                    "figure_label^2",
                                    "table_label^2",
                                    "caption",
                                    "authors",
                                    "tags",
                                ],
                            }
                        }
                    ],
                    "filter": [{"terms": {"collection_ids": ["collection-1"]}}],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": [0.1, 0.2, 0.3],
                "k": 5,
                "num_candidates": 20,
            },
        }
    ]
