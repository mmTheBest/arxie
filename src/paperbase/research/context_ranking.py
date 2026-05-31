"""Deterministic task-aware ranking for Paperbase research context items."""

from __future__ import annotations

import re
from typing import Any

from paperbase.research.task_descriptors import ResearchTaskDescriptor, task_descriptor_for

_CONTEXT_ROLE_SORT_PRIORITY = {
    "selected": 0,
    "pinned_context": 1,
    "backend_retrieved": 2,
}


def rank_context_papers(
    task_type: str,
    papers: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Rank paper summaries for a task and attach selection metadata."""

    descriptor = task_descriptor_for(task_type)
    ranked = [
        _annotate_item(
            item=paper,
            descriptor=descriptor,
            features=_paper_features(paper),
            intelligence_layer="source_library",
            default_role="collection_default",
        )
        for paper in papers
    ]
    return _rank_annotated_items(ranked, limit=limit)


def rank_context_sources(
    task_type: str,
    sources: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Rank explicit Study sources and attach selection metadata."""

    descriptor = task_descriptor_for(task_type)
    ranked = [
        _annotate_item(
            item=source,
            descriptor=descriptor,
            features=_source_features(source),
            intelligence_layer="source_library",
            default_role="user_draft" if _is_draft_source(source) else "study_source",
        )
        for source in sources
    ]
    return _rank_annotated_items(ranked, limit=limit)


def _rank_annotated_items(
    items: list[dict[str, Any]],
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        enumerate(items),
        key=lambda item: (
            _context_role_sort_priority(item[1]),
            -float(item[1]["selection_score"]),
            item[0],
        ),
    )
    ordered = [item for _index, item in ranked]
    if limit is None or len(ordered) <= limit:
        return ordered

    selected = ordered[:limit]
    selected_ids = {_item_identity(item) for item in selected}
    ordered_index = {_item_identity(item): index for index, item in enumerate(ordered)}
    for protected in ordered[limit:]:
        protected_id = _item_identity(protected)
        if protected_id in selected_ids or not _is_protected_context_item(protected):
            continue
        replacement_index = _last_unprotected_index(selected)
        if replacement_index is None:
            continue
        selected_ids.remove(_item_identity(selected[replacement_index]))
        selected[replacement_index] = protected
        selected_ids.add(protected_id)
    return sorted(selected, key=lambda item: ordered_index[_item_identity(item)])


def _context_role_sort_priority(item: dict[str, Any]) -> int:
    role = item.get("context_role")
    return _CONTEXT_ROLE_SORT_PRIORITY.get(role, 3) if isinstance(role, str) else 3


def _annotate_item(
    *,
    item: dict[str, Any],
    descriptor: ResearchTaskDescriptor,
    features: dict[str, Any],
    intelligence_layer: str,
    default_role: str,
) -> dict[str, Any]:
    annotated = dict(item)
    context_role = str(annotated.get("context_role") or default_role)
    features["selected"] = context_role == "selected"
    features["pinned"] = context_role == "pinned_context"
    features["draft_source"] = context_role == "user_draft" or bool(features.get("draft_source"))
    score = _selection_score(descriptor, features)
    annotated["context_role"] = context_role
    annotated["context_reason"] = _context_reason(
        role=context_role,
        descriptor=descriptor,
        features=features,
        fallback=str(annotated.get("context_reason") or ""),
    )
    annotated["selection_score"] = score
    annotated["selection_features"] = features
    annotated["intelligence_layer"] = intelligence_layer
    return annotated


def _selection_score(
    descriptor: ResearchTaskDescriptor,
    features: dict[str, Any],
) -> float:
    score = 0.0
    for feature, weight in descriptor.context_feature_weights.items():
        score += weight * _numeric_feature_value(features.get(feature))
    return round(score, 4)


def _paper_features(paper: dict[str, Any]) -> dict[str, Any]:
    text_blob = _item_text(paper)
    sections = _list_value(paper.get("sections"))
    design_elements = _list_value(paper.get("research_design_elements"))
    methods = _list_value(paper.get("methods"))
    datasets = _list_value(paper.get("datasets"))
    metrics = _list_value(paper.get("metrics"))
    results = _list_value(paper.get("results"))
    limitations = _list_value(paper.get("limitations"))
    findings = _list_value(paper.get("findings"))
    evidence_references = _list_value(paper.get("evidence_references"))
    retrieved_chunks = _list_value(paper.get("retrieved_chunks"))
    return {
        "sections": len(sections),
        "retrieved_chunks": len(retrieved_chunks),
        "methods": len(methods),
        "datasets": len(datasets),
        "metrics": len(metrics),
        "results": len(results),
        "limitations": len(limitations),
        "findings": len(findings),
        "evidence_references": len(evidence_references),
        "ablations": _ablation_count(design_elements),
        "baselines": _pattern_count(text_blob, ("baseline", "control")),
        "benchmark_signal": _has_any(text_blob, ("benchmark", "evaluation", "leaderboard")),
        "comparison_sections": _comparison_section_count(sections),
        "contradictions": _pattern_count(text_blob, ("contradict", "conflict", "inconsistent")),
        "source_claims": 0,
        "validation_warnings": len(_list_value(paper.get("validation_warnings"))),
        "themes": _theme_signal_count(text_blob),
        "broad_evidence": _broad_evidence_count(
            [methods, datasets, metrics, results, limitations, findings]
        ),
        "direct_evidence": _direct_evidence_count(
            sections=sections,
            retrieved_chunks=retrieved_chunks,
            results=results,
            evidence_references=evidence_references,
        ),
        "draft_source": False,
    }


def _source_features(source: dict[str, Any]) -> dict[str, Any]:
    text_blob = _item_text(source)
    evidence_references = _list_value(source.get("evidence_references"))
    return {
        "sections": 0,
        "methods": 0,
        "datasets": 0,
        "metrics": 0,
        "results": 0,
        "limitations": _pattern_count(text_blob, ("limitation", "weakness", "risk")),
        "findings": 0,
        "evidence_references": len(evidence_references),
        "ablations": _pattern_count(text_blob, ("ablation",)),
        "baselines": _pattern_count(text_blob, ("baseline", "control")),
        "benchmark_signal": _has_any(text_blob, ("benchmark", "evaluation", "leaderboard")),
        "comparison_sections": _pattern_count(text_blob, ("compare", "comparison")),
        "contradictions": _pattern_count(text_blob, ("contradict", "conflict", "inconsistent")),
        "source_claims": 1 if text_blob else 0,
        "validation_warnings": 1 if source.get("error_message") else 0,
        "themes": _theme_signal_count(text_blob),
        "broad_evidence": 1 if text_blob else 0,
        "direct_evidence": 1 if text_blob or evidence_references else 0,
        "draft_source": _is_draft_source(source),
    }


def _context_reason(
    *,
    role: str,
    descriptor: ResearchTaskDescriptor,
    features: dict[str, Any],
    fallback: str,
) -> str:
    role_reason = {
        "selected": "The user selected this item for the research thread.",
        "pinned_context": "The item is pinned in the active Study.",
        "backend_retrieved": "The paper was retrieved by backend-assisted context search.",
        "collection_default": "The item was included from the active collection scope.",
        "study_source": "The item is an explicit source in the active Study.",
        "user_draft": "The item is a user draft or work source in the active Study.",
    }.get(role, fallback or "The item was included as task context.")
    signals = _top_feature_labels(descriptor, features)
    if not signals:
        return role_reason
    return (
        f"{role_reason} Prioritized for {descriptor.task_type} because it has {', '.join(signals)}."
    )


def _top_feature_labels(
    descriptor: ResearchTaskDescriptor,
    features: dict[str, Any],
    *,
    limit: int = 4,
) -> list[str]:
    weighted_features = [
        (feature, weight, _numeric_feature_value(features.get(feature)))
        for feature, weight in descriptor.context_feature_weights.items()
    ]
    ranked = sorted(
        (
            (feature, weight * value)
            for feature, weight, value in weighted_features
            if value > 0 and feature not in {"selected", "pinned"}
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [_feature_label(feature) for feature, _score in ranked[:limit]]


def _feature_label(feature: str) -> str:
    return feature.replace("_", " ")


def _numeric_feature_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list | tuple | set | dict):
        return float(len(value))
    return 0.0


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "abstract", "summary", "content", "source_type"):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
    for section in _list_value(item.get("sections")):
        if isinstance(section, dict):
            parts.extend(
                str(section.get(key) or "") for key in ("title", "text") if section.get(key)
            )
    for chunk in _list_value(item.get("retrieved_chunks")):
        if isinstance(chunk, dict):
            parts.extend(
                str(chunk.get(key) or "") for key in ("section_title", "text") if chunk.get(key)
            )
    return " ".join(parts).lower()


def _ablation_count(design_elements: list[Any]) -> int:
    count = 0
    for element in design_elements:
        if isinstance(element, dict):
            text = " ".join(
                str(element.get(key) or "") for key in ("element_type", "title", "description")
            )
        else:
            text = str(element)
        if "ablation" in text.lower():
            count += 1
    return count


def _comparison_section_count(sections: list[Any]) -> int:
    count = 0
    for section in sections:
        if isinstance(section, dict):
            text = " ".join(str(section.get(key) or "") for key in ("title", "text"))
        else:
            text = str(section)
        if _has_any(text.lower(), ("comparison", "compare", "baseline", "benchmark", "evaluation")):
            count += 1
    return count


def _pattern_count(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(pattern)}\w*\b", text)) for pattern in patterns)


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _theme_signal_count(text: str) -> int:
    return _pattern_count(text, ("theme", "trend", "cluster", "direction", "gap"))


def _broad_evidence_count(groups: list[list[Any]]) -> int:
    return sum(1 for group in groups if group)


def _direct_evidence_count(
    *,
    sections: list[Any],
    retrieved_chunks: list[Any],
    results: list[Any],
    evidence_references: list[Any],
) -> int:
    chunk_count = len(retrieved_chunks)
    return min(6, len(sections) + chunk_count + len(results) + len(evidence_references))


def _is_draft_source(source: dict[str, Any]) -> bool:
    source_type = str(source.get("source_type") or "").lower()
    title = str(source.get("title") or "").lower()
    return "draft" in source_type or "draft" in title


def _item_identity(item: dict[str, Any]) -> str:
    for key in ("paper_id", "source_id", "memory_record_id", "graph_node_id", "graph_edge_id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return str(id(item))


def _is_protected_context_item(item: dict[str, Any]) -> bool:
    return item.get("context_role") in {"selected", "pinned_context", "user_draft"}


def _last_unprotected_index(items: list[dict[str, Any]]) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        if not _is_protected_context_item(items[index]):
            return index
    return None
