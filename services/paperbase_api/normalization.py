"""Normalization helpers for Paperbase API presentation and filtering."""

from __future__ import annotations

import re

_METRIC_NAME_ALIASES = {
    "adjusted rand index": "ARI",
    "ari": "ARI",
    "area under the precision recall curve": "AUPRC",
    "area under the receiver operating characteristic curve": "AUROC",
    "aupr": "AUPRC",
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "jsd": "JSD",
    "mean squared error": "MSE",
    "mse": "MSE",
    "pcc": "PCC",
    "pearson correlation": "PCC",
    "pearson correlation coefficient": "PCC",
    "pearson correlation coefficient pcc": "PCC",
}


def normalize_summary_key(value: str) -> str:
    collapsed = re.sub(r"[^0-9A-Za-z]+", " ", value)
    return re.sub(r"\s+", " ", collapsed).strip().casefold()


def canonicalize_metric_display_name(value: str) -> str:
    display_name = value.strip()
    return _METRIC_NAME_ALIASES.get(normalize_summary_key(display_name), display_name)
