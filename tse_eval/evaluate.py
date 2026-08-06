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

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .audio import load_wav
from .metrics import (DEFAULT_METRICS, METRIC_COLUMNS, _DNSMOS_COLUMNS,
                      compute_row_metrics, micro_wer)
from .ort_setup import (DEFAULT_INTRA_OP_THREADS, configure_onnxruntime,
                        provenance as ort_provenance)

# Auto-detection candidates (case-insensitive), first match wins.
#
# New names are appended, never inserted, so a CSV that already resolved to a
# particular column keeps resolving to it.
#
# ``pred_path`` matters most: all three sibling manifests (TPEX / LLM-TSE /
# StyleTSE) name the extracted audio that way, and its absence made est
# auto-detection fail on every one of them.
_CANDIDATES = {
    "id":  ["file_id", "id", "utt_id", "utterance_id", "name", "filename"],
    "est": ["estimate", "extracted", "est", "pred", "prediction",
            "enhanced", "output", "est_path", "estimate_path",
            "pred_path", "prediction_path", "extracted_path", "enhanced_path"],
    "ref": ["reference", "target", "gt", "ground_truth", "clean", "ref",
            "ref_path", "target_path"],
    "mix": ["mixture", "mixed", "mix", "mixture_path", "mixed_path", "noisy"],
    "ovr": ["overlap", "ovr", "overlap_ratio", "ovr_ratio", "overlap_pct",
            "overlap_percent"],
    # Interference (competing speaker). TPEX calls it ``infer_path``; LLM-TSE and
    # StyleTSE call it ``interference_path`` — same meaning, different name.
    # Detected and passed through only; no metric consumes it yet (SDR/SIR/SAR
    # are deliberately not implemented — see tse_eval.metrics docstring).
    "itf": ["interference_path", "infer_path", "interference", "interferer_path"],
    # Ground-truth transcript for WER. Absent from StyleTSE manifests.
    "txt": ["target_sentence", "text", "transcript", "reference_text"],
}


@dataclass
class ColumnMap:
    """Resolved input-CSV column names."""
    id: Optional[str]
    est: str
    ref: str
    mix: str
    ovr: Optional[str]
    itf: Optional[str] = None      # interference / competing-speaker wav (unused yet)
    txt: Optional[str] = None      # ground-truth transcript, for WER


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
    txt_col: Optional[str] = None,
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
        itf=pick(None, "itf", required=False),
        txt=pick(txt_col, "txt", required=False),
    )


def _evaluate_row(row: pd.Series, cols: ColumnMap, target_sr: int,
                  metrics: Optional[set],
                  si_sdr_backend: str = "native") -> Dict[str, float]:
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
        text = None
        if cols.txt is not None:
            raw = row.get(cols.txt)
            text = None if raw is None or pd.isna(raw) else str(raw)
        result.update(compute_row_metrics(
            est, ref, mix, target_sr, metrics,
            reference_text=text, si_sdr_backend=si_sdr_backend))
    except Exception as exc:                                # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ─────────────────────────────────────────────────────────────────────────
# Stratification axes
# ─────────────────────────────────────────────────────────────────────────
# A derived axis is computed from manifest columns rather than read directly.
# ``same_gender`` is the standard TSE difficulty axis reviewers ask for.
DERIVED_AXES = {"same_gender": ("target_gender", "infer_gender")}

# Above this many distinct values a numeric axis is treated as continuous and
# bucketed into quartiles. ``overlap_ratio`` has 6 levels, so it stays discrete.
_MAX_DISCRETE_LEVELS = 10


def add_derived_axes(df: pd.DataFrame) -> "list[str]":
    """Add derived stratification columns in place; return the ones created.

    ``same_gender`` = ``target_gender == infer_gender`` (case/space-insensitive),
    labelled ``"same"`` / ``"diff"`` so it reads well in the summary table.
    """
    created: "list[str]" = []
    for axis, (a, b) in DERIVED_AXES.items():
        if axis in df.columns or a not in df.columns or b not in df.columns:
            continue
        left = df[a].astype(str).str.strip().str.lower()
        right = df[b].astype(str).str.strip().str.lower()
        df[axis] = np.where(left == right, "same", "diff")
        created.append(axis)
    return created


def _axis_keys(series: pd.Series) -> pd.Series:
    """Turn an axis column into group labels.

    Discrete values are used verbatim (as strings). A numeric column with many
    distinct values is bucketed into quartiles labelled ``Q1..Q4`` with their
    ranges, so continuous metadata (``snr_db``, ``lufs_diff``, …) is usable
    without the caller pre-binning it.
    """
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > _MAX_DISCRETE_LEVELS:
        try:
            binned = pd.qcut(series, 4, duplicates="drop")
            return binned.apply(
                lambda iv: "nan" if pd.isna(iv)
                else f"Q{binned.cat.categories.get_loc(iv) + 1} "
                     f"({iv.left:.3g}, {iv.right:.3g}]")
        except Exception:                                   # noqa: BLE001
            pass                                            # fall through
    return series.astype(str)


def evaluate_csv(
    input_csv: str,
    target_sr: int = 24_000,
    metrics: Optional[set] = None,
    id_col: Optional[str] = None,
    est_col: Optional[str] = None,
    ref_col: Optional[str] = None,
    mix_col: Optional[str] = None,
    ovr_col: Optional[str] = None,
    txt_col: Optional[str] = None,
    progress: bool = True,
    source_csv: Optional[str] = None,
    group_by: "str | Sequence[str] | None" = None,
    model_name: Optional[str] = None,
    si_sdr_backend: str = "native",
    dnsmos_threads: int = DEFAULT_INTRA_OP_THREADS,
    dnsmos_providers: "str | Sequence[str] | None" = None,
) -> "tuple[pd.DataFrame, pd.DataFrame, ColumnMap, Dict[str, object]]":
    """Evaluate every row of ``input_csv``.

    Args:
        input_csv: Path to the manifest CSV.
        target_sr: Working sample rate (default 24000 = PORTE-v3 native).
            SI-SDR/SI-SDRi/STOI/ESTOI are computed here; PESQ and DNSMOS are
            resampled to 16 kHz internally. See ``tse_eval.metrics``.

        source_csv: Optional PORTE-v3 source CSV, left-joined on the id column.
            Supplies metadata the manifest lacks (notably ``target_sentence``
            for WER on StyleTSE, plus continuous axes like ``snr_db``).
        group_by: Stratification axis or axes for the summary.
        model_name: Label written into the summary and the sidecar.
        si_sdr_backend: ``"native"`` (default) or ``"asteroid"``.

    Returns:
        (per_row_df, summary_df, column_map, info)

        * ``per_row_df`` — all original input columns, in order, followed by the
          metric columns and an ``error`` column.
        * ``summary_df`` — long-format, one row per (axis, group) plus ``ALL``.
        * ``info`` — provenance details for the sidecar (join, axes, columns).
    """
    df = pd.read_csv(input_csv)
    if len(df) == 0:
        raise ValueError(f"Input CSV '{input_csv}' has no rows.")

    info: Dict[str, object] = {"n_rows": int(len(df)), "source_join": None}

    # Resolve the id column first — the join key.
    pre = resolve_columns(df.columns, id_col, est_col, ref_col, mix_col,
                          ovr_col, txt_col)
    if source_csv:
        df, join_info = _join_source(df, source_csv, pre.id, pre)
        info["source_join"] = join_info

    # Re-resolve after the join: it can introduce columns (e.g. target_sentence).
    cols = resolve_columns(df.columns, id_col, est_col, ref_col, mix_col,
                           ovr_col, txt_col)
    info["columns"] = {k: getattr(cols, k) for k in
                       ("id", "est", "ref", "mix", "ovr", "itf", "txt")}
    info["derived_axes"] = add_derived_axes(df)

    # DNSMOS runs through onnxruntime, and speechmos creates its sessions with no
    # options at all — bound the thread pool and pick providers before the first
    # session exists. Only when a DNSMOS column was actually requested.
    want = set(DEFAULT_METRICS) if metrics is None else set(metrics)
    if want & _DNSMOS_COLUMNS:
        configure_onnxruntime(threads=dnsmos_threads, providers=dnsmos_providers)

    rows_iter = df.iterrows()
    if progress:
        try:
            from tqdm import tqdm
            rows_iter = tqdm(rows_iter, total=len(df), desc="Evaluating", unit="row")
        except Exception:
            pass

    metric_records: List[Dict[str, float]] = [
        _evaluate_row(row, cols, target_sr, metrics, si_sdr_backend)
        for _, row in rows_iter
    ]
    metrics_df = pd.DataFrame(metric_records)

    # Read after the rows ran: the ONNX session is built lazily on first use, so
    # the *actual* providers are only known now. This is provenance, not config —
    # a CUDA request can legitimately end up on CPU.
    info["dnsmos"] = ort_provenance()

    # Per-row output: original columns passed through + metrics appended.
    per_row = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    # Default axis: the detected overlap column, preserving the old behaviour.
    axes = group_by if group_by is not None else (
        [cols.ovr] if cols.ovr is not None else [])
    if isinstance(axes, str):
        axes = [a.strip() for a in axes.split(",") if a.strip()]
    requested = list(axes)
    axes = [a for a in requested if a in per_row.columns]
    info["group_by"] = axes
    info["group_by_missing"] = [a for a in requested if a not in per_row.columns]

    summary = summarize(per_row, axes, model_name=model_name)
    return per_row, summary, cols, info


def _join_source(df: pd.DataFrame, source_csv: str, id_col: Optional[str],
                 cols: ColumnMap) -> "tuple[pd.DataFrame, Dict[str, object]]":
    """Left-join the PORTE-v3 source CSV onto the manifest on the id column.

    Conflict policy (deliberate, and recorded in the sidecar):

    * **the three audio paths stay from the manifest** — they point at what was
      actually scored, and the source CSV stores them relative to ``base_dir``;
    * **every other overlapping column is taken from the source**, which is the
      metadata's origin and therefore authoritative.
    """
    if id_col is None:
        raise ValueError(
            "--source-csv needs an id column to join on, but none was detected. "
            "Pass --id-col explicitly.")
    src = pd.read_csv(source_csv)
    if id_col not in src.columns:
        raise ValueError(
            f"--source-csv '{source_csv}' has no '{id_col}' column to join on. "
            f"Available: {list(src.columns)[:10]}…")

    keep_from_manifest = {c for c in (cols.est, cols.ref, cols.mix, cols.itf) if c}
    overlapping = (set(df.columns) & set(src.columns)) - {id_col}
    from_source = sorted(overlapping - keep_from_manifest)
    from_manifest = sorted(overlapping & keep_from_manifest)

    # Drop the manifest's copy of the columns the source wins, then join.
    merged = df.drop(columns=from_source).merge(
        src.drop(columns=[c for c in from_manifest if c in src.columns]),
        on=id_col, how="left", suffixes=("", "_src"))

    matched = int(merged[id_col].isin(src[id_col]).sum())
    return merged, {
        "source_csv": os.path.abspath(source_csv),
        "join_key": id_col,
        "source_rows": int(len(src)),
        "matched_rows": matched,
        "unmatched_rows": int(len(df) - matched),
        "columns_from_source": from_source,
        "columns_kept_from_manifest": from_manifest,
        "columns_added": sorted(set(merged.columns) - set(df.columns)),
    }


def _safe_nanmean(values: np.ndarray) -> float:
    """Mean ignoring NaN; NaN for an all-NaN/empty slice (and no warning)."""
    vals = values[~np.isnan(values)]
    return float(vals.mean()) if vals.size else float("nan")


def _aggregate_metric(metric: str, sub: pd.DataFrame) -> float:
    """Aggregate one metric over one group.

    Everything is a NaN-safe mean except ``wer``, which must be **corpus
    micro-WER** (total edits / total reference words). Averaging per-row WER
    over-weights short utterances and yields a different number, so the paper
    table uses the micro figure — see :func:`tse_eval.metrics.micro_wer`.
    """
    if metric == "wer" and {"wer_edits", "wer_words"} <= set(sub.columns):
        return micro_wer(sub["wer_edits"].to_numpy(dtype=float),
                         sub["wer_words"].to_numpy(dtype=float))
    return _safe_nanmean(sub[metric].to_numpy(dtype=float))


def summarize(per_row: pd.DataFrame, group_by: "str | Sequence[str] | None" = None,
              model_name: Optional[str] = None) -> pd.DataFrame:
    """Build a long-format, multi-axis summary.

    One row per (axis, group) with an ``ALL`` group per axis, so several
    projects' summaries can simply be concatenated into one comparison table.

    Columns: ``model_name, axis, group, n, <metrics…>``.

    Args:
        per_row: The per-row results frame.
        group_by: Axis column name, or several. ``None``/empty produces a single
            ``axis="ALL"`` row over the whole set. Axes missing from the frame
            are skipped (a warning is the caller's job).
        model_name: Value for the ``model_name`` column (e.g. ``tpex``).

    Returns:
        Long-format summary DataFrame.
    """
    present = [m for m in METRIC_COLUMNS if m in per_row.columns]
    if group_by is None:
        axes: List[str] = []
    elif isinstance(group_by, str):
        axes = [group_by]
    else:
        axes = list(group_by)
    axes = [a for a in axes if a in per_row.columns]

    def record(axis: str, label: str, sub: pd.DataFrame) -> Dict[str, object]:
        rec: Dict[str, object] = {
            "model_name": model_name if model_name is not None else "",
            "axis": axis, "group": label, "n": int(len(sub)),
        }
        for m in present:
            rec[m] = _aggregate_metric(m, sub)
        return rec

    records: List[Dict[str, object]] = []
    for axis in axes:
        keys = _axis_keys(per_row[axis])
        # observed=True: quartile keys are categorical, and empty bins must not
        # become phantom groups.
        for key, sub in per_row.groupby(keys, sort=True, observed=True):
            records.append(record(axis, str(key), sub))
        records.append(record(axis, "ALL", per_row))
    if not axes:
        records.append(record("ALL", "ALL", per_row))

    return pd.DataFrame.from_records(records)
