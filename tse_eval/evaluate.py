"""
tse_eval.evaluate — CSV-in / CSV-out evaluation pipeline.

Flow:
    input CSV  ──►  detect columns  ──►  per-row metric compute  ──►  per-row CSV
                                                              └────►  overlap summary CSV

The input CSV is expected to have, per row: an id, a path to the extracted
(estimated) target speech, a path to the ground-truth target speech, and a path
to the mixture.  Column names are auto-detected (and overridable).  An optional
overlap-ratio column, if present, is passed through and used to group the summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .audio import load_wav
from .metrics import METRIC_COLUMNS, compute_row_metrics

# Auto-detection candidates (case-insensitive), first match wins.
_CANDIDATES = {
    "id":  ["file_id", "id", "utt_id", "utterance_id", "name", "filename"],
    "est": ["estimate", "extracted", "est", "pred", "prediction",
            "enhanced", "output", "est_path", "estimate_path"],
    "ref": ["reference", "target", "gt", "ground_truth", "clean", "ref",
            "ref_path", "target_path"],
    "mix": ["mixture", "mixed", "mix", "mixture_path", "mixed_path", "noisy"],
    "ovr": ["overlap", "ovr", "overlap_ratio", "ovr_ratio", "overlap_pct",
            "overlap_percent"],
}


@dataclass
class ColumnMap:
    """Resolved input-CSV column names."""
    id: Optional[str]
    est: str
    ref: str
    mix: str
    ovr: Optional[str]


def _auto_detect(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first CSV column whose lower-cased name matches a candidate."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def resolve_columns(
    columns: Sequence[str],
    id_col: Optional[str] = None,
    est_col: Optional[str] = None,
    ref_col: Optional[str] = None,
    mix_col: Optional[str] = None,
    ovr_col: Optional[str] = None,
) -> ColumnMap:
    """Resolve required/optional columns from explicit overrides or auto-detection.

    Explicit arguments win; otherwise names are auto-detected from ``columns``.
    Raises ``ValueError`` if a required column (est / ref / mix) cannot be found.
    """
    def pick(explicit: Optional[str], key: str, required: bool) -> Optional[str]:
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(
                    f"Column '{explicit}' (for {key}) not in CSV. "
                    f"Available: {list(columns)}")
            return explicit
        found = _auto_detect(columns, _CANDIDATES[key])
        if found is None and required:
            raise ValueError(
                f"Could not auto-detect the '{key}' column. Tried "
                f"{_CANDIDATES[key]}. Pass --{key}-col explicitly. "
                f"Available columns: {list(columns)}")
        return found

    return ColumnMap(
        id=pick(id_col, "id", required=False),
        est=pick(est_col, "est", required=True),
        ref=pick(ref_col, "ref", required=True),
        mix=pick(mix_col, "mix", required=True),
        ovr=pick(ovr_col, "ovr", required=False),
    )


def _evaluate_row(row: pd.Series, cols: ColumnMap, target_sr: int,
                  metrics: Optional[set]) -> Dict[str, float]:
    """Load the three wavs for one row and compute its metrics.

    On any I/O or decode error, returns all-``nan`` metrics plus an ``error``
    string so the pipeline never aborts on a single bad row.
    """
    result: Dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}
    result["error"] = ""
    try:
        est = load_wav(str(row[cols.est]), target_sr)
        ref = load_wav(str(row[cols.ref]), target_sr)
        mix = load_wav(str(row[cols.mix]), target_sr)
        result.update(compute_row_metrics(est, ref, mix, target_sr, metrics))
    except Exception as exc:                                # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def evaluate_csv(
    input_csv: str,
    target_sr: int = 16_000,
    metrics: Optional[set] = None,
    id_col: Optional[str] = None,
    est_col: Optional[str] = None,
    ref_col: Optional[str] = None,
    mix_col: Optional[str] = None,
    ovr_col: Optional[str] = None,
    progress: bool = True,
) -> "tuple[pd.DataFrame, pd.DataFrame, ColumnMap]":
    """Evaluate every row of ``input_csv``.

    Returns:
        (per_row_df, summary_df, column_map)

        * ``per_row_df`` — all original input columns, in order, followed by the
          metric columns and an ``error`` column.
        * ``summary_df`` — mean of each metric per overlap group (if an overlap
          column exists) plus an ``ALL`` row; otherwise a single ``ALL`` row.
    """
    df = pd.read_csv(input_csv)
    if len(df) == 0:
        raise ValueError(f"Input CSV '{input_csv}' has no rows.")
    cols = resolve_columns(df.columns, id_col, est_col, ref_col, mix_col, ovr_col)

    rows_iter = df.iterrows()
    if progress:
        try:
            from tqdm import tqdm
            rows_iter = tqdm(rows_iter, total=len(df), desc="Evaluating", unit="row")
        except Exception:
            pass

    metric_records: List[Dict[str, float]] = [
        _evaluate_row(row, cols, target_sr, metrics) for _, row in rows_iter
    ]
    metrics_df = pd.DataFrame(metric_records)

    # Per-row output: original columns passed through + metrics appended.
    per_row = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    summary = summarize(per_row, cols.ovr)
    return per_row, summary, cols


def summarize(per_row: pd.DataFrame, ovr_col: Optional[str]) -> pd.DataFrame:
    """Build the overlap-stratified summary (means + counts).

    Groups by ``ovr_col`` (treated as a literal categorical key so values like
    ``0%``, ``0L``, ``0S`` or ``0-40%`` all work) and appends an ``ALL`` row.
    If ``ovr_col`` is ``None`` the summary is a single ``ALL`` row.
    """
    present = [m for m in METRIC_COLUMNS if m in per_row.columns]

    def safe_nanmean(values: np.ndarray) -> float:
        """Mean ignoring NaN; returns NaN for an all-NaN/empty slice (no warning)."""
        vals = values[~np.isnan(values)]
        return float(vals.mean()) if vals.size else float("nan")

    def group_stats(sub: pd.DataFrame, label: str) -> Dict[str, object]:
        rec: Dict[str, object] = {"group": label, "n": int(len(sub))}
        for m in present:
            rec[m] = safe_nanmean(sub[m].to_numpy(dtype=float))
        return rec

    records: List[Dict[str, object]] = []
    if ovr_col is not None and ovr_col in per_row.columns:
        for key, sub in per_row.groupby(per_row[ovr_col].astype(str), sort=True):
            records.append(group_stats(sub, str(key)))
    records.append(group_stats(per_row, "ALL"))

    summary = pd.DataFrame.from_records(records)
    if ovr_col is not None and ovr_col in per_row.columns:
        summary = summary.rename(columns={"group": ovr_col})
    return summary
