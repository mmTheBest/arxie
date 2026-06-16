"""Runtime search backend integration and reindex orchestration for Paperbase."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import (
    Chunk,
    CollectionPaper,
    Dataset,
    ExtractionRun,
    Figure,
    Method,
    Metric,
    Paper,
    ResultRow,
    Section,
    TableArtifact,
)
from paperbase.db.repositories import PaperRepository
from paperbase.search.embeddings import EmbeddingProvider
from paperbase.search.index_names import search_index_name, search_index_prefix
from paperbase.search.index_templates import (
    chunk_index_template,
    figure_index_template,
    paper_index_template,
    result_row_index_template,
    structured_entity_index_template,
    table_index_template,
)
from paperbase.search.indexer import (
    build_chunk_document,
    build_figure_document,
    build_paper_document,
    build_result_row_document,
    build_structured_entity_document,
    build_table_document,
)


class SearchBackend(Protocol):
    def ensure_index(self, index_name: str, template: dict[str, object]) -> None: ...

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None: ...

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]: ...


class ElasticsearchSearchBackend:
    """Small Elasticsearch/OpenSearch-compatible backend using plain HTTP."""

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)

    def ensure_index(self, index_name: str, template: dict[str, object]) -> None:
        head_response = self.client.head(f"{self.base_url}/{index_name}")
        if head_response.status_code == 200:
            return
        if head_response.status_code not in {404}:
            head_response.raise_for_status()
        response = self.client.put(f"{self.base_url}/{index_name}", json=template)
        response.raise_for_status()

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None:
        if not documents:
            return

        lines: list[str] = []
        for document in documents:
            lines.append(json.dumps({"index": {"_index": index_name}}))
            lines.append(json.dumps(document))
        payload = "\n".join(lines) + "\n"
        response = self.client.post(
            f"{self.base_url}/_bulk",
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise RuntimeError(f"Bulk indexing returned errors for index {index_name}.")

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]:
        payload = dict(query)
        payload.setdefault("size", size)
        response = self.client.post(
            f"{self.base_url}/{index_name}/_search",
            json=payload,
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]


class PaperbaseSearchReindexer:
    """Rebuild Paperbase read-model documents from the canonical relational store."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        backend: SearchBackend,
        index_prefix: str | None = None,
        project_id: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.backend = backend
        self.embedding_provider = embedding_provider
        self.project_id = project_id
        resolved_index_prefix = index_prefix or search_index_prefix(project_id)
        self.paper_index_name = search_index_name("papers", index_prefix=resolved_index_prefix)
        self.chunk_index_name = search_index_name("chunks", index_prefix=resolved_index_prefix)
        self.figure_index_name = search_index_name("figures", index_prefix=resolved_index_prefix)
        self.table_index_name = search_index_name("tables", index_prefix=resolved_index_prefix)
        self.structured_entity_index_name = search_index_name(
            "structured-entities",
            index_prefix=resolved_index_prefix,
        )
        self.result_row_index_name = search_index_name(
            "result-rows",
            index_prefix=resolved_index_prefix,
        )

    def reindex_all(self) -> dict[str, int]:
        paper_documents = self._build_paper_documents()
        chunk_documents = self._build_chunk_documents()
        figure_documents = self._build_figure_documents()
        table_documents = self._build_table_documents()
        structured_entity_documents = self._build_structured_entity_documents()
        result_row_documents = self._build_result_row_documents()

        self.backend.ensure_index(self.paper_index_name, paper_index_template())
        self.backend.ensure_index(self.chunk_index_name, chunk_index_template())
        self.backend.ensure_index(self.figure_index_name, figure_index_template())
        self.backend.ensure_index(self.table_index_name, table_index_template())
        self.backend.ensure_index(
            self.structured_entity_index_name,
            structured_entity_index_template(),
        )
        self.backend.ensure_index(
            self.result_row_index_name,
            result_row_index_template(),
        )

        self.backend.bulk_index(self.paper_index_name, paper_documents)
        self.backend.bulk_index(self.chunk_index_name, chunk_documents)
        self.backend.bulk_index(self.figure_index_name, figure_documents)
        self.backend.bulk_index(self.table_index_name, table_documents)
        self.backend.bulk_index(
            self.structured_entity_index_name,
            structured_entity_documents,
        )
        self.backend.bulk_index(
            self.result_row_index_name,
            result_row_documents,
        )

        return {
            "papers": len(paper_documents),
            "chunks": len(chunk_documents),
            "figures": len(figure_documents),
            "tables": len(table_documents),
            "structured_entities": len(structured_entity_documents),
            "result_rows": len(result_row_documents),
        }

    def _build_paper_documents(self) -> list[dict[str, object]]:
        with self.session_factory() as session:
            papers = session.execute(select(Paper).order_by(Paper.created_at.asc())).scalars().all()
            repository = PaperRepository(session)
            paper_ids = [paper.id for paper in papers]
            authors_by_paper_id = repository.list_author_names_by_paper_ids(paper_ids)
            tags_by_paper_id = repository.list_tags_by_paper_ids(paper_ids)
            datasets_by_paper_id = self._load_named_entities_by_paper_id(session, Dataset, paper_ids)
            methods_by_paper_id = self._load_named_entities_by_paper_id(session, Method, paper_ids)
            metrics_by_paper_id = self._load_named_entities_by_paper_id(session, Metric, paper_ids)
            collection_ids_by_paper_id = self._load_collection_ids_by_paper_id(session, paper_ids)
            extracted_paper_ids = set(
                session.execute(
                    select(ExtractionRun.paper_id).where(ExtractionRun.status == "completed")
                ).scalars()
            )

        return [
            build_paper_document(
                paper_id=paper.id,
                title=paper.canonical_title,
                abstract=paper.abstract,
                year=paper.publication_year,
                venue=paper.venue,
                provider=paper.provider,
                external_id=paper.external_id,
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                authors=authors_by_paper_id.get(paper.id, []),
                tags=tags_by_paper_id.get(paper.id, []),
                datasets=datasets_by_paper_id.get(paper.id, []),
                methods=methods_by_paper_id.get(paper.id, []),
                metrics=metrics_by_paper_id.get(paper.id, []),
                collection_ids=collection_ids_by_paper_id.get(paper.id, []),
                project_id=self.project_id,
                extraction_state="extracted" if paper.id in extracted_paper_ids else "unextracted",
                embedding_vector=self._embed_text(
                    " ".join(
                        filter(
                            None,
                            [
                                paper.canonical_title,
                                paper.abstract or "",
                                " ".join(authors_by_paper_id.get(paper.id, [])),
                                " ".join(tags_by_paper_id.get(paper.id, [])),
                                " ".join(datasets_by_paper_id.get(paper.id, [])),
                                " ".join(methods_by_paper_id.get(paper.id, [])),
                                " ".join(metrics_by_paper_id.get(paper.id, [])),
                            ],
                        )
                    )
                ),
            )
            for paper in papers
        ]

    def _build_chunk_documents(self) -> list[dict[str, object]]:
        with self.session_factory() as session:
            rows = session.execute(
                select(Chunk, Paper, Section)
                .join(Paper, Paper.id == Chunk.paper_id)
                .outerjoin(Section, Section.id == Chunk.section_id)
                .order_by(Chunk.created_at.asc())
            ).all()
            paper_ids = [paper.id for _, paper, _ in rows]
            collection_ids_by_paper_id = self._load_collection_ids_by_paper_id(session, paper_ids)

        return [
            build_chunk_document(
                chunk_id=chunk.id,
                paper_id=paper.id,
                title=paper.canonical_title,
                section_title=section.title if section is not None else None,
                text=chunk.text,
                collection_ids=collection_ids_by_paper_id.get(paper.id, []),
                project_id=self.project_id,
                embedding_vector=self._embed_text(
                    " ".join(
                        filter(
                            None,
                            [
                                paper.canonical_title,
                                section.title if section is not None else "",
                                chunk.text,
                            ],
                        )
                    )
                ),
            )
            for chunk, paper, section in rows
        ]

    def _build_figure_documents(self) -> list[dict[str, object]]:
        with self.session_factory() as session:
            rows = session.execute(
                select(Figure, Paper)
                .join(Paper, Paper.id == Figure.paper_id)
                .order_by(Figure.created_at.asc())
            ).all()
            paper_ids = [paper.id for _, paper in rows]
            collection_ids_by_paper_id = self._load_collection_ids_by_paper_id(session, paper_ids)

        return [
            build_figure_document(
                figure_id=figure.id,
                paper_id=paper.id,
                title=paper.canonical_title,
                figure_label=figure.figure_label,
                caption=figure.caption,
                collection_ids=collection_ids_by_paper_id.get(paper.id, []),
                project_id=self.project_id,
                embedding_vector=self._embed_text(
                    " ".join(filter(None, [paper.canonical_title, figure.figure_label, figure.caption]))
                ),
            )
            for figure, paper in rows
        ]

    def _build_table_documents(self) -> list[dict[str, object]]:
        with self.session_factory() as session:
            rows = session.execute(
                select(TableArtifact, Paper)
                .join(Paper, Paper.id == TableArtifact.paper_id)
                .order_by(TableArtifact.created_at.asc())
            ).all()
            paper_ids = [paper.id for _, paper in rows]
            collection_ids_by_paper_id = self._load_collection_ids_by_paper_id(session, paper_ids)

        return [
            build_table_document(
                table_id=table.id,
                paper_id=paper.id,
                title=paper.canonical_title,
                table_label=table.table_label,
                caption=table.caption,
                structured_payload=dict(table.structured_payload_json or {}),
                collection_ids=collection_ids_by_paper_id.get(paper.id, []),
                project_id=self.project_id,
                embedding_vector=self._embed_text(
                    " ".join(
                        filter(
                            None,
                            [
                                paper.canonical_title,
                                table.table_label,
                                table.caption,
                                json.dumps(table.structured_payload_json or {}, sort_keys=True),
                            ],
                        )
                    )
                ),
            )
            for table, paper in rows
        ]

    def _build_structured_entity_documents(self) -> list[dict[str, object]]:
        entity_rows: list[tuple[str, Dataset | Method | Metric, Paper]] = []
        with self.session_factory() as session:
            for entity_type, model in (
                ("dataset", Dataset),
                ("method", Method),
                ("metric", Metric),
            ):
                rows = session.execute(
                    select(model, Paper)
                    .join(Paper, Paper.id == model.paper_id)
                    .order_by(model.created_at.asc(), model.id.asc())
                ).all()
                entity_rows.extend(
                    (entity_type, entity, paper)
                    for entity, paper in rows
                )
            paper_ids = [paper.id for _, _, paper in entity_rows]
            collection_ids_by_paper_id = self._load_collection_ids_by_paper_id(
                session,
                paper_ids,
            )

        return [
            build_structured_entity_document(
                entity_id=entity.id,
                entity_type=entity_type,
                paper_id=paper.id,
                title=paper.canonical_title,
                normalized_name=entity.normalized_name,
                display_name=entity.display_name,
                metadata=dict(entity.metadata_json or {}),
                collection_ids=collection_ids_by_paper_id.get(paper.id, []),
                project_id=self.project_id,
                embedding_vector=self._embed_text(
                    " ".join(
                        filter(
                            None,
                            [
                                paper.canonical_title,
                                entity_type,
                                entity.normalized_name,
                                entity.display_name,
                                json.dumps(
                                    entity.metadata_json or {},
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            ],
                        )
                    )
                ),
            )
            for entity_type, entity, paper in entity_rows
        ]

    def _build_result_row_documents(self) -> list[dict[str, object]]:
        with self.session_factory() as session:
            rows = session.execute(
                select(ResultRow, Paper, Dataset, Method, Metric)
                .join(Paper, Paper.id == ResultRow.paper_id)
                .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
                .outerjoin(Method, Method.id == ResultRow.method_id)
                .outerjoin(Metric, Metric.id == ResultRow.metric_id)
                .order_by(ResultRow.created_at.asc(), ResultRow.id.asc())
            ).all()
            paper_ids = [paper.id for _, paper, *_entities in rows]
            collection_ids_by_paper_id = self._load_collection_ids_by_paper_id(
                session,
                paper_ids,
            )

        return [
            build_result_row_document(
                result_row_id=result.id,
                paper_id=paper.id,
                title=paper.canonical_title,
                dataset_id=result.dataset_id,
                dataset=dataset.display_name if dataset is not None else None,
                method_id=result.method_id,
                method=method.display_name if method is not None else None,
                metric_id=result.metric_id,
                metric=metric.display_name if metric is not None else None,
                split_name=result.split_name,
                value_numeric=result.value_numeric,
                value_text=result.value_text,
                comparator_text=result.comparator_text,
                notes=result.notes,
                collection_ids=collection_ids_by_paper_id.get(paper.id, []),
                project_id=self.project_id,
                embedding_vector=self._embed_text(
                    " ".join(
                        filter(
                            None,
                            [
                                paper.canonical_title,
                                dataset.display_name if dataset is not None else "",
                                method.display_name if method is not None else "",
                                metric.display_name if metric is not None else "",
                                result.split_name or "",
                                result.value_text or "",
                                result.comparator_text or "",
                                result.notes or "",
                            ],
                        )
                    )
                ),
            )
            for result, paper, dataset, method, metric in rows
        ]

    def _embed_text(self, text: str) -> list[float] | None:
        if self.embedding_provider is None:
            return None
        return self.embedding_provider.embed(text)

    @staticmethod
    def _load_collection_ids_by_paper_id(
        session: Session,
        paper_ids: Sequence[str],
    ) -> dict[str, list[str]]:
        if not paper_ids:
            return {}

        rows = session.execute(
            select(CollectionPaper.paper_id, CollectionPaper.collection_id)
            .where(CollectionPaper.paper_id.in_(paper_ids))
            .order_by(CollectionPaper.paper_id.asc(), CollectionPaper.created_at.asc())
        ).all()
        grouped: dict[str, list[str]] = {paper_id: [] for paper_id in paper_ids}
        for paper_id, collection_id in rows:
            grouped.setdefault(paper_id, []).append(collection_id)
        return grouped

    @staticmethod
    def _load_named_entities_by_paper_id(
        session: Session,
        model: type[Dataset] | type[Method] | type[Metric],
        paper_ids: Sequence[str],
    ) -> dict[str, list[str]]:
        if not paper_ids:
            return {}

        rows = session.execute(
            select(model.paper_id, model.display_name)
            .where(model.paper_id.in_(paper_ids))
            .order_by(model.paper_id.asc(), model.display_name.asc())
        ).all()
        grouped: dict[str, list[str]] = {paper_id: [] for paper_id in paper_ids}
        for paper_id, display_name in rows:
            grouped.setdefault(paper_id, []).append(display_name)
        return grouped
