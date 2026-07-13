"""
TSE_Eval — a unified, easy-to-use evaluation toolkit for
Target Speech Extraction (TSE).

Public API:
    from tse_eval import evaluate_csv, compute_row_metrics
    from tse_eval.metrics import si_sdr, si_sdri, pesq_wb, stoi_metric, dnsmos
"""

from .metrics import (
    METRIC_COLUMNS,
    compute_row_metrics,
    si_sdr,
    si_sdri,
    pesq_wb,
    stoi_metric,
    dnsmos,
)
from .evaluate import evaluate_csv, summarize, resolve_columns

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "METRIC_COLUMNS",
    "compute_row_metrics",
    "si_sdr",
    "si_sdri",
    "pesq_wb",
    "stoi_metric",
    "dnsmos",
    "evaluate_csv",
    "summarize",
    "resolve_columns",
]
