"""
TSE_Eval — a unified, easy-to-use evaluation toolkit for
Target Speech Extraction (TSE).

Public API:
    from tse_eval import evaluate_csv, compute_row_metrics
    from tse_eval.metrics import si_sdr, si_sdri, pesq_wb, stoi_metric, dnsmos
"""

from .metrics import (
    DEFAULT_METRICS,
    METRIC_COLUMNS,
    METRIC_SR_POLICY,
    compute_row_metrics,
    si_sdr,
    si_sdr_family,
    si_sdri,
    micro_wer,
    pesq_wb,
    stoi_metric,
    dnsmos,
    spk_sim,
    wer,
)
from .evaluate import (evaluate_csv, summarize, resolve_columns,
                       add_derived_axes)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_METRICS",
    "METRIC_COLUMNS",
    "METRIC_SR_POLICY",
    "compute_row_metrics",
    "si_sdr",
    "si_sdr_family",
    "si_sdri",
    "micro_wer",
    "pesq_wb",
    "stoi_metric",
    "dnsmos",
    "spk_sim",
    "wer",
    "evaluate_csv",
    "summarize",
    "resolve_columns",
    "add_derived_axes",
]
