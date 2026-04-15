"""Elasticsearch-style index templates for Paperbase read models."""

from __future__ import annotations


def _dense_vector_property() -> dict[str, object]:
    return {
        "type": "dense_vector",
        "dims": 1536,
        "index": True,
        "similarity": "cosine",
    }


def paper_index_template() -> dict[str, object]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "abstract": {"type": "text"},
                "publication_year": {"type": "integer"},
                "venue": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "authors": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "embedding": _dense_vector_property(),
            }
        },
    }


def chunk_index_template() -> dict[str, object]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "section_title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "text": {"type": "text"},
                "embedding": _dense_vector_property(),
            }
        },
    }


def figure_index_template() -> dict[str, object]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "figure_id": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "figure_label": {"type": "keyword"},
                "caption": {"type": "text"},
                "embedding": _dense_vector_property(),
            }
        },
    }
