"""Query builders for Paperbase search/read models."""

from __future__ import annotations

from collections.abc import Sequence


def build_search_query(
    *,
    query_text: str | None = None,
    filters: dict[str, object] | None = None,
    embedding_vector: Sequence[float] | None = None,
    k: int = 10,
) -> dict[str, object]:
    bool_query: dict[str, list[dict[str, object]]] = {"must": [], "filter": []}

    if query_text:
        bool_query["must"].append(
            {
                "multi_match": {
                    "query": query_text,
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
        )

    filters = filters or {}
    year_gte = filters.get("year_gte")
    if year_gte is not None:
        bool_query["filter"].append({"range": {"publication_year": {"gte": year_gte}}})

    year_lte = filters.get("year_lte")
    if year_lte is not None:
        bool_query["filter"].append({"range": {"publication_year": {"lte": year_lte}}})

    collection_ids = filters.get("collection_ids")
    if collection_ids:
        bool_query["filter"].append({"terms": {"collection_ids": list(collection_ids)}})

    venue = filters.get("venue")
    if venue:
        bool_query["filter"].append({"terms": {"venue.keyword": list(venue)}})

    authors = filters.get("authors")
    if authors:
        bool_query["filter"].append({"terms": {"authors.keyword": list(authors)}})

    tags = filters.get("tags")
    if tags:
        bool_query["filter"].append({"terms": {"tags.keyword": list(tags)}})

    datasets = filters.get("datasets")
    if datasets:
        bool_query["filter"].append({"terms": {"datasets.keyword": list(datasets)}})

    methods = filters.get("methods")
    if methods:
        bool_query["filter"].append({"terms": {"methods.keyword": list(methods)}})

    metrics = filters.get("metrics")
    if metrics:
        bool_query["filter"].append({"terms": {"metrics.keyword": list(metrics)}})

    extraction_state = filters.get("extraction_state")
    if extraction_state:
        bool_query["filter"].append({"term": {"extraction_state": extraction_state}})

    request_body: dict[str, object] = {"query": {"bool": bool_query}}
    if embedding_vector is not None:
        request_body["knn"] = {
            "field": "embedding",
            "query_vector": list(embedding_vector),
            "k": k,
            "num_candidates": max(k * 4, 20),
        }

    return request_body
