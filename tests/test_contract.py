"""
Tests for the input contract and the summary contract.

Covers the things a reviewer would check, and the things that silently broke
before: est-column auto-detection on the real sibling manifests, the
``si_sdri == si_sdr − input_si_sdr`` identity, 3-way length alignment, corpus
micro-WER aggregation, and multi-axis stratification.

No audio model is loaded here — WER/Speaker-Similarity are exercised at the
aggregation and schema level only, so the suite stays CPU-only and offline.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tse_eval.audio import align_triple
from tse_eval.evaluate import (add_derived_axes, evaluate_csv, resolve_columns,
                               summarize)
from tse_eval.metrics import (DEFAULT_METRICS, METRIC_COLUMNS,
                              MODEL_BACKED_METRICS, compute_row_metrics,
                              micro_wer, si_sdr, si_sdr_family)

# The 17 columns every sibling manifest shares (paths + PORTE-v3 metadata).
COMMON_MANIFEST_COLUMNS = [
    "file_id", "pred_path", "mixed_path", "target_path", "sample_rate", "task",
    "overlap_ratio", "prompt", "prompt_category", "target_gender", "infer_gender",
    "first_speak", "target_start_time", "target_end_time", "length_sec",
    "overlap_start_time", "overlap_end_time",
]

# How each project's manifest differs — see the project inference scripts.
MANIFEST_SHAPES = {
    # TPEX keeps the source name `infer_path` and appends run metadata.
    "tpex": COMMON_MANIFEST_COLUMNS + [
        "infer_path", "target_sentence", "split", "cropped", "out_len_sec",
        "steps", "cfg", "sway", "seed"],
    "llmtse": COMMON_MANIFEST_COLUMNS + ["interference_path", "target_sentence"],
    # StyleTSE has no transcript column, so WER is unavailable without a join.
    "styletse": COMMON_MANIFEST_COLUMNS + ["interference_path"],
}


# ─────────────────────────────────────────────────────────────────────────
# Column auto-detection on the real manifest shapes
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("project", sorted(MANIFEST_SHAPES))
def test_auto_detect_resolves_every_sibling_manifest(project):
    """`pred_path` used to be missing from the candidates, so est detection
    failed on all three manifests and the tool could not be run without
    --est-col. This pins the fix."""
    cols = resolve_columns(MANIFEST_SHAPES[project])
    assert cols.est == "pred_path"
    assert cols.ref == "target_path"
    assert cols.mix == "mixed_path"
    assert cols.ovr == "overlap_ratio"
    assert cols.id == "file_id"


def test_interference_alias_is_recognised_under_either_name():
    """TPEX says `infer_path`; the others say `interference_path`."""
    assert resolve_columns(MANIFEST_SHAPES["tpex"]).itf == "infer_path"
    assert resolve_columns(MANIFEST_SHAPES["llmtse"]).itf == "interference_path"


def test_transcript_column_present_only_where_the_manifest_has_it():
    assert resolve_columns(MANIFEST_SHAPES["tpex"]).txt == "target_sentence"
    assert resolve_columns(MANIFEST_SHAPES["styletse"]).txt is None


def test_legacy_example_header_still_resolves_the_same_way():
    """Backward compatibility: appended candidates must not steal a match."""
    cols = resolve_columns(["file_id", "estimate", "reference", "mixture", "overlap"])
    assert (cols.est, cols.ref, cols.mix, cols.ovr) == (
        "estimate", "reference", "mixture", "overlap")


# ─────────────────────────────────────────────────────────────────────────
# SI-SDR family: identity + 3-way alignment
# ─────────────────────────────────────────────────────────────────────────
def _triple(n=16_000, est_len=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16_000
    ref = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
           + 0.05 * rng.standard_normal(n))
    mix = ref + 0.6 * rng.standard_normal(n)
    est = ref + 0.05 * rng.standard_normal(n)
    if est_len is not None:
        est = est[:est_len]
    return est, ref, mix


def test_si_sdri_identity_holds_exactly():
    """`si_sdri == si_sdr − input_si_sdr` — the identity the reviewer flagged."""
    est, ref, mix = _triple()
    fam = si_sdr_family(est, ref, mix)
    assert fam["si_sdri"] == pytest.approx(
        fam["si_sdr"] - fam["input_si_sdr"], abs=1e-9)


def test_identity_holds_in_compute_row_metrics_output():
    est, ref, mix = _triple()
    row = compute_row_metrics(est, ref, mix, 16_000, metrics={"si_sdri"})
    assert row["si_sdri"] == pytest.approx(
        row["si_sdr"] - row["input_si_sdr"], abs=1e-9)


def test_align_triple_trims_all_three_to_the_shortest():
    est, ref, mix = _triple(est_len=12_345)
    e, r, m = align_triple(est, ref, mix)
    assert len(e) == len(r) == len(m) == 12_345


def test_pairwise_baseline_is_independent_of_the_estimate():
    """`input_si_sdr_pairwise` must not move when only the estimate changes.

    That independence is the direct evidence that one pipeline scored all three
    projects: for a given sample the value is identical everywhere, however long
    or short each system's output happens to be.
    """
    _, ref, mix = _triple()
    full, _, _ = _triple()
    short, _, _ = _triple(est_len=9_000)

    a = compute_row_metrics(full, ref, mix, 16_000, metrics={"si_sdri"})
    b = compute_row_metrics(short, ref, mix, 16_000, metrics={"si_sdri"})

    assert a["input_si_sdr_pairwise"] == pytest.approx(
        b["input_si_sdr_pairwise"], abs=1e-12)
    # The 3-way-trimmed baseline, by contrast, follows the estimate's length.
    assert a["input_si_sdr"] != pytest.approx(b["input_si_sdr"], abs=1e-6)
    # ...and equals the pairwise value when nothing needed trimming.
    assert a["input_si_sdr"] == pytest.approx(a["input_si_sdr_pairwise"], abs=1e-12)


def test_trimmed_baseline_matches_a_manual_trim():
    """Sanity: the 3-way path is exactly 'trim, then compute'."""
    est, ref, mix = _triple(est_len=9_000)
    row = compute_row_metrics(est, ref, mix, 16_000, metrics={"si_sdri"})
    _, ref_t, mix_t = align_triple(est, ref, mix)
    assert row["input_si_sdr"] == pytest.approx(si_sdr(mix_t, ref_t), abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# Metric set defaults
# ─────────────────────────────────────────────────────────────────────────
def test_model_backed_metrics_are_off_by_default():
    """A plain run must not need Whisper (3 GB) or ECAPA."""
    assert MODEL_BACKED_METRICS == {"wer", "spk_sim"}
    assert not (set(DEFAULT_METRICS) & MODEL_BACKED_METRICS)
    est, ref, mix = _triple()
    row = compute_row_metrics(est, ref, mix, 16_000)      # metrics=None
    assert math.isnan(row["wer"]) and math.isnan(row["spk_sim"])


def test_input_si_sdr_columns_exist_in_the_schema():
    for key in ("input_si_sdr", "input_si_sdr_pairwise"):
        assert key in METRIC_COLUMNS


# ─────────────────────────────────────────────────────────────────────────
# Corpus micro-WER
# ─────────────────────────────────────────────────────────────────────────
def test_micro_wer_differs_from_the_row_average():
    """Micro weights by length; the row mean over-weights short utterances.

    Row 1: 1 error in 100 words (1%). Row 2: 5 errors in 5 words (100%).
    micro = 6/105 ≈ 5.7%, row mean = 50.5% — a large, deliberate difference.
    """
    edits = np.array([1.0, 5.0])
    words = np.array([100.0, 5.0])
    micro = micro_wer(edits, words)
    row_mean = float(np.mean(edits / words))
    assert micro == pytest.approx(6 / 105)
    assert row_mean == pytest.approx(0.505)
    assert abs(micro - row_mean) > 0.4


def test_micro_wer_skips_rows_without_a_reference():
    """StyleTSE has no transcript column, so such rows must drop out."""
    micro = micro_wer(np.array([2.0, np.nan]), np.array([10.0, np.nan]))
    assert micro == pytest.approx(0.2)
    assert math.isnan(micro_wer(np.array([np.nan]), np.array([np.nan])))
    assert math.isnan(micro_wer(np.array([0.0]), np.array([0.0])))


def test_summary_aggregates_wer_as_micro_not_mean():
    df = pd.DataFrame({
        "axis_col": ["a", "a"],
        "wer": [0.01, 1.0],
        "wer_edits": [1.0, 5.0],
        "wer_words": [100.0, 5.0],
    })
    summary = summarize(df, "axis_col")
    row = summary[summary["group"] == "a"].iloc[0]
    assert row["wer"] == pytest.approx(6 / 105)          # micro, not 0.505


# ─────────────────────────────────────────────────────────────────────────
# Multi-axis stratification
# ─────────────────────────────────────────────────────────────────────────
def test_same_gender_axis_is_derived_from_the_gender_columns():
    df = pd.DataFrame({"target_gender": ["F", "M", "f"], "infer_gender": ["F", "F", " F "]})
    created = add_derived_axes(df)
    assert created == ["same_gender"]
    assert list(df["same_gender"]) == ["same", "diff", "same"]   # case/space-insensitive


def test_summary_is_long_format_with_one_all_row_per_axis():
    df = pd.DataFrame({
        "overlap_ratio": [0.0, 0.0, 1.0, 1.0],
        "prompt_category": ["gender_female", "order_first", "gender_female", "pitch_lower"],
        "si_sdr": [1.0, 2.0, 3.0, 4.0],
    })
    summary = summarize(df, ["overlap_ratio", "prompt_category"], model_name="tpex")

    assert list(summary.columns)[:4] == ["model_name", "axis", "group", "n"]
    assert set(summary["axis"]) == {"overlap_ratio", "prompt_category"}
    assert (summary["model_name"] == "tpex").all()
    # Each axis carries its own ALL row over the full set.
    for axis in ("overlap_ratio", "prompt_category"):
        sub = summary[summary["axis"] == axis]
        all_row = sub[sub["group"] == "ALL"].iloc[0]
        assert int(all_row["n"]) == 4
        assert all_row["si_sdr"] == pytest.approx(2.5)
    # Group counts add up to the total within an axis.
    ovr = summary[(summary["axis"] == "overlap_ratio") & (summary["group"] != "ALL")]
    assert int(ovr["n"].sum()) == 4


def test_continuous_axis_is_quartile_binned_but_discrete_axis_is_not():
    """`snr_db` is continuous → quartiles. `overlap_ratio` has 6 levels → verbatim."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "snr_db": rng.uniform(-10, 10, 400),
        "overlap_ratio": np.tile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 400 // 6 + 1)[:400],
        "si_sdr": rng.normal(size=400),
    })
    summary = summarize(df, ["snr_db", "overlap_ratio"])

    snr_groups = set(summary[(summary["axis"] == "snr_db")
                             & (summary["group"] != "ALL")]["group"])
    assert len(snr_groups) == 4
    assert all(g.startswith("Q") for g in snr_groups)

    ovr_groups = set(summary[(summary["axis"] == "overlap_ratio")
                             & (summary["group"] != "ALL")]["group"])
    assert len(ovr_groups) == 6
    assert not any(g.startswith("Q") for g in ovr_groups)


def test_evaluate_csv_multi_axis_end_to_end(wav_triple, write_input_csv):
    """Axes present in the manifest work without any source join."""
    rows = []
    for i, (ovr, tg, ig) in enumerate([(0.0, "F", "F"), (1.0, "M", "F"), (1.0, "M", "M")]):
        est_p, ref_p, mix_p = wav_triple(idx=i)
        rows.append({
            "file_id": f"utt{i}", "pred_path": est_p, "target_path": ref_p,
            "mixed_path": mix_p, "overlap_ratio": ovr,
            "target_gender": tg, "infer_gender": ig,
            "prompt_category": "gender_female",
        })
    csv_path = write_input_csv(rows)

    per_row, summary, cols, info = evaluate_csv(
        csv_path, target_sr=16_000, metrics={"si_sdr", "si_sdri"},
        group_by="overlap_ratio,same_gender", model_name="demo", progress=False)

    assert cols.est == "pred_path"                       # auto-detected
    assert info["derived_axes"] == ["same_gender"]
    assert info["group_by"] == ["overlap_ratio", "same_gender"]
    assert set(summary["axis"]) == {"overlap_ratio", "same_gender"}
    assert set(summary[summary["axis"] == "same_gender"]["group"]) == {"same", "diff", "ALL"}
    assert (summary["model_name"] == "demo").all()
    # The identity survives the full pipeline.
    for _, row in per_row.iterrows():
        assert row["si_sdri"] == pytest.approx(
            row["si_sdr"] - row["input_si_sdr"], abs=1e-9)


def test_unknown_axis_is_reported_not_fatal(wav_triple, write_input_csv):
    est_p, ref_p, mix_p = wav_triple(idx=0)
    csv_path = write_input_csv([{
        "file_id": "u0", "pred_path": est_p, "target_path": ref_p, "mixed_path": mix_p}])

    _per_row, summary, _cols, info = evaluate_csv(
        csv_path, target_sr=16_000, metrics={"si_sdr"},
        group_by="overlap_ratio,nonexistent_axis", progress=False)

    assert info["group_by_missing"] == ["overlap_ratio", "nonexistent_axis"]
    assert len(summary) == 1 and summary.iloc[0]["axis"] == "ALL"
