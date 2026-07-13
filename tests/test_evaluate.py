"""
Tests for tse_eval.evaluate (CSV-in/CSV-out pipeline) and the light
tse_eval.cli entrypoint.

All wavs are tiny synthetic wavs written under ``tmp_path`` (see
tests/conftest.py); no real data, GPU, or network access is required.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tse_eval.cli import main as cli_main
from tse_eval.evaluate import evaluate_csv, resolve_columns, summarize
from tse_eval.metrics import METRIC_COLUMNS


# ─────────────────────────────────────────────────────────────────────────
# resolve_columns
# ─────────────────────────────────────────────────────────────────────────
def test_resolve_columns_auto_detects_standard_names():
    columns = ["file_id", "estimate", "reference", "mixture", "overlap"]
    cm = resolve_columns(columns)
    assert cm.id == "file_id"
    assert cm.est == "estimate"
    assert cm.ref == "reference"
    assert cm.mix == "mixture"
    assert cm.ovr == "overlap"


def test_resolve_columns_auto_detects_alternative_names():
    columns = ["utt_id", "est", "target", "mix"]
    cm = resolve_columns(columns)
    assert cm.id == "utt_id"
    assert cm.est == "est"
    assert cm.ref == "target"
    assert cm.mix == "mix"
    assert cm.ovr is None


def test_resolve_columns_explicit_override_wins():
    columns = ["file_id", "estimate", "reference", "mixture", "custom_pred"]
    cm = resolve_columns(columns, est_col="custom_pred")
    assert cm.est == "custom_pred"
    # ref/mix remain auto-detected
    assert cm.ref == "reference"
    assert cm.mix == "mixture"


def test_resolve_columns_missing_required_raises():
    columns = ["file_id", "reference", "mixture"]  # no est-like column present
    with pytest.raises(ValueError):
        resolve_columns(columns)


def test_resolve_columns_bad_explicit_override_raises():
    columns = ["file_id", "estimate", "reference", "mixture"]
    with pytest.raises(ValueError):
        resolve_columns(columns, mix_col="does_not_exist")


# ─────────────────────────────────────────────────────────────────────────
# evaluate_csv — end to end
# ─────────────────────────────────────────────────────────────────────────
def test_evaluate_csv_end_to_end_with_overlap(tmp_path, wav_triple, write_input_csv):
    """Passthrough columns first, metrics + error appended, grouped summary."""
    overlaps = ["40%", "40%", "100%"]
    rows = []
    for i, ovr in enumerate(overlaps):
        est_p, ref_p, mix_p = wav_triple(idx=i)
        rows.append({
            "file_id": f"utt{i}",
            "estimate": est_p,
            "reference": ref_p,
            "mixture": mix_p,
            "overlap": ovr,
        })
    csv_path = write_input_csv(rows)

    per_row, summary, cols = evaluate_csv(csv_path, target_sr=16_000, progress=False)

    expected_cols = ["file_id", "estimate", "reference", "mixture", "overlap"] + METRIC_COLUMNS + ["error"]
    assert list(per_row.columns) == expected_cols
    assert len(per_row) == 3
    assert (per_row["error"] == "").all()
    assert per_row["si_sdr"].apply(math.isfinite).all()

    # Two distinct overlap groups (40%, 100%) plus the ALL row.
    assert cols.ovr == "overlap"
    assert set(summary["overlap"]) == {"40%", "100%", "ALL"}
    assert len(summary) == 3
    n_by_group = dict(zip(summary["overlap"], summary["n"]))
    assert n_by_group["40%"] == 2
    assert n_by_group["100%"] == 1
    assert n_by_group["ALL"] == 3


def test_evaluate_csv_without_overlap_column_single_all_row(tmp_path, wav_triple, write_input_csv):
    rows = []
    for i in range(2):
        est_p, ref_p, mix_p = wav_triple(idx=i)
        rows.append({"file_id": f"utt{i}", "estimate": est_p, "reference": ref_p, "mixture": mix_p})
    csv_path = write_input_csv(rows)

    per_row, summary, cols = evaluate_csv(csv_path, progress=False)

    assert cols.ovr is None
    assert len(per_row) == 2
    assert len(summary) == 1
    assert summary.iloc[0]["group"] == "ALL"
    assert int(summary.iloc[0]["n"]) == 2


def test_evaluate_csv_missing_wav_sets_error_without_raising(tmp_path, wav_triple, write_input_csv):
    est_p, ref_p, mix_p = wav_triple(idx=0)
    missing_path = str(tmp_path / "does_not_exist.wav")
    rows = [
        {"file_id": "good", "estimate": est_p, "reference": ref_p, "mixture": mix_p},
        {"file_id": "bad", "estimate": missing_path, "reference": ref_p, "mixture": mix_p},
    ]
    csv_path = write_input_csv(rows)

    # Must not raise even though one row's wav is missing.
    per_row, summary, cols = evaluate_csv(csv_path, progress=False)

    assert len(per_row) == 2
    bad_row = per_row[per_row["file_id"] == "bad"].iloc[0]
    assert bad_row["error"] != ""
    assert all(math.isnan(bad_row[m]) for m in METRIC_COLUMNS)

    good_row = per_row[per_row["file_id"] == "good"].iloc[0]
    assert good_row["error"] == ""
    assert math.isfinite(good_row["si_sdr"])


def test_evaluate_csv_metrics_subset_propagates(tmp_path, wav_triple, write_input_csv):
    est_p, ref_p, mix_p = wav_triple(idx=0)
    rows = [{"file_id": "u0", "estimate": est_p, "reference": ref_p, "mixture": mix_p}]
    csv_path = write_input_csv(rows)

    per_row, _summary, _cols = evaluate_csv(csv_path, metrics={"si_sdr"}, progress=False)

    assert math.isfinite(per_row.loc[0, "si_sdr"])
    for key in METRIC_COLUMNS:
        if key != "si_sdr":
            assert math.isnan(per_row.loc[0, key])


def test_evaluate_csv_empty_raises():
    # An input csv with a header but no rows should raise, not silently pass.
    import csv as _csv
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["file_id", "estimate", "reference", "mixture"])
        with pytest.raises(ValueError):
            evaluate_csv(path, progress=False)
    finally:
        os.remove(path)


# ─────────────────────────────────────────────────────────────────────────
# summarize — direct unit tests (no audio I/O)
# ─────────────────────────────────────────────────────────────────────────
def test_summarize_ignores_nan_and_counts_correctly():
    df = pd.DataFrame({
        "overlap": ["a", "a", "b"],
        "si_sdr": [10.0, np.nan, 5.0],
        "si_sdri": [1.0, 2.0, np.nan],
    })
    summary = summarize(df, "overlap")

    assert len(summary) == 3  # groups "a", "b" + "ALL"
    row_a = summary[summary["overlap"] == "a"].iloc[0]
    assert int(row_a["n"]) == 2
    assert row_a["si_sdr"] == pytest.approx(10.0)  # nanmean([10, nan]) == 10

    row_b = summary[summary["overlap"] == "b"].iloc[0]
    assert int(row_b["n"]) == 1
    assert row_b["si_sdri"] != row_b["si_sdri"]  # nanmean([nan]) -> nan

    row_all = summary[summary["overlap"] == "ALL"].iloc[0]
    assert int(row_all["n"]) == 3
    assert row_all["si_sdr"] == pytest.approx(7.5)  # nanmean([10, 5])


def test_summarize_without_overlap_column_is_single_all_row():
    df = pd.DataFrame({"si_sdr": [1.0, 2.0, 3.0]})
    summary = summarize(df, None)
    assert len(summary) == 1
    assert summary.iloc[0]["group"] == "ALL"
    assert int(summary.iloc[0]["n"]) == 3
    assert summary.iloc[0]["si_sdr"] == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────
# cli.py — light end-to-end
# ─────────────────────────────────────────────────────────────────────────
def test_cli_writes_output_and_summary_csv(tmp_path, wav_triple, write_input_csv):
    rows = []
    for i in range(2):
        est_p, ref_p, mix_p = wav_triple(idx=i)
        rows.append({
            "file_id": f"utt{i}", "estimate": est_p, "reference": ref_p,
            "mixture": mix_p, "overlap": "40%",
        })
    csv_path = write_input_csv(rows)
    out_csv = tmp_path / "out.csv"

    rc = cli_main(["--input", csv_path, "--output", str(out_csv), "--no-progress"])

    assert rc == 0
    assert out_csv.exists()
    summary_csv = tmp_path / "out_summary.csv"
    assert summary_csv.exists()

    per_row = pd.read_csv(out_csv)
    assert len(per_row) == 2
    summary = pd.read_csv(summary_csv)
    assert "ALL" in summary["overlap"].astype(str).tolist()


def test_cli_summary_output_none_skips_summary_file(tmp_path, wav_triple, write_input_csv):
    est_p, ref_p, mix_p = wav_triple(idx=0)
    rows = [{"file_id": "u0", "estimate": est_p, "reference": ref_p, "mixture": mix_p}]
    csv_path = write_input_csv(rows)
    out_csv = tmp_path / "out2.csv"

    rc = cli_main([
        "--input", csv_path, "--output", str(out_csv),
        "--summary-output", "none", "--no-progress",
    ])

    assert rc == 0
    assert out_csv.exists()
    assert not (tmp_path / "out2_summary.csv").exists()


def test_cli_metrics_subset_restricts_computed_columns(tmp_path, wav_triple, write_input_csv):
    est_p, ref_p, mix_p = wav_triple(idx=0)
    rows = [{"file_id": "u0", "estimate": est_p, "reference": ref_p, "mixture": mix_p}]
    csv_path = write_input_csv(rows)
    out_csv = tmp_path / "out3.csv"

    rc = cli_main([
        "--input", csv_path, "--output", str(out_csv),
        "--metrics", "si_sdr,pesq", "--no-progress",
    ])

    assert rc == 0
    per_row = pd.read_csv(out_csv)
    assert math.isfinite(per_row.loc[0, "si_sdr"])
    assert math.isfinite(per_row.loc[0, "pesq"])
    assert math.isnan(per_row.loc[0, "stoi"])
    assert math.isnan(per_row.loc[0, "dnsmos_sig"])
