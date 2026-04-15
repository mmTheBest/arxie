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
                        "text",
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

    venue = filters.get("venue")
    if venue:
        bool_query["filter"].append({"terms": {"venue.keyword": list(venue)}})

    tags = filters.get("tags")
    if tags:
        bool_query["filter"].append({"terms": {"tags.keyword": list(tags)}})

    query: dict[str, object] = {"bool": bool_query}
    if embedding_vector is not None:
        query["knn"] = {
            "field": "embedding",
            "query_vector": list(embedding_vector),
            "k": k,
            "num_candidates": max(k * 4, 20),
        }

    return query
