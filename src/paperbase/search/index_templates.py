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
                "provider": {"type": "keyword"},
                "external_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "abstract": {"type": "text"},
                "publication_year": {"type": "integer"},
                "venue": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "doi": {"type": "keyword"},
                "arxiv_id": {"type": "keyword"},
                "authors": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "tags": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "datasets": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "methods": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "metrics": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "collection_ids": {"type": "keyword"},
                "project_id": {"type": "keyword"},
                "extraction_state": {"type": "keyword"},
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
                "collection_ids": {"type": "keyword"},
                "project_id": {"type": "keyword"},
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
                "collection_ids": {"type": "keyword"},
                "project_id": {"type": "keyword"},
                "embedding": _dense_vector_property(),
            }
        },
    }


def table_index_template() -> dict[str, object]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "table_id": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "table_label": {"type": "keyword"},
                "caption": {"type": "text"},
                "structured_payload": {"type": "object", "enabled": False},
                "collection_ids": {"type": "keyword"},
                "project_id": {"type": "keyword"},
                "embedding": _dense_vector_property(),
            }
        },
    }


def structured_entity_index_template() -> dict[str, object]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "entity_id": {"type": "keyword"},
                "entity_type": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "normalized_name": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "display_name": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "metadata": {"type": "object", "enabled": False},
                "metadata_text": {"type": "text"},
                "collection_ids": {"type": "keyword"},
                "project_id": {"type": "keyword"},
                "embedding": _dense_vector_property(),
            }
        },
    }


def result_row_index_template() -> dict[str, object]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "result_row_id": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "dataset_id": {"type": "keyword"},
                "dataset": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "method_id": {"type": "keyword"},
                "method": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "metric_id": {"type": "keyword"},
                "metric": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "split_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "value_numeric": {"type": "float"},
                "value_text": {"type": "text"},
                "comparator_text": {"type": "text"},
                "notes": {"type": "text"},
                "collection_ids": {"type": "keyword"},
                "project_id": {"type": "keyword"},
                "embedding": _dense_vector_property(),
            }
        },
    }
