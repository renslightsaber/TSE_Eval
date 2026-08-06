"""
Tests for the shipped scoring policy and the onnxruntime setup.

The metric set in ``configs/config.yaml`` decides what a plain run computes and
how long it takes, so it is pinned here: changing it should be a deliberate act
that updates this test, not a silent edit.

No ONNX session is created and no model is downloaded — the provider logic is
exercised through pure resolution helpers.
"""
from __future__ import annotations

import os

import pytest

from tse_eval.config import DEFAULT_CONFIG_PATH, DEFAULTS, load_config
from tse_eval.metrics import METRIC_COLUMNS, MODEL_BACKED_METRICS
from tse_eval.ort_setup import (CPU_ONLY_PROVIDERS, DEFAULT_INTRA_OP_THREADS,
                                DEFAULT_PROVIDERS, provenance, resolve_providers)


# ─────────────────────────────────────────────────────────────────────────
# The shipped policy file
# ─────────────────────────────────────────────────────────────────────────
def test_shipped_config_exists_and_parses():
    assert os.path.isfile(DEFAULT_CONFIG_PATH), "configs/config.yaml is missing"
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg["target_sr"] == 24_000
    assert cfg["si_sdr_backend"] == "native"


def test_shipped_config_metric_set_is_exactly_what_we_expect():
    """Pins the default metric set — the thing that sets a run's cost.

    ``spk_sim`` is on (~20 min per 15,000 utterances). ``wer`` is off (~178 min);
    it is enabled explicitly for the paper run. If this assertion fails, the
    policy changed — update it deliberately and re-check the runtime estimates
    in the docs.
    """
    metrics = load_config(DEFAULT_CONFIG_PATH)["metrics"]
    assert "spk_sim" in metrics
    assert "wer" not in metrics
    # Everything that is not model-backed should be present.
    for m in METRIC_COLUMNS:
        if m not in MODEL_BACKED_METRICS:
            assert m in metrics, f"{m} dropped out of the default metric set"
    assert len(set(metrics)) == len(metrics), "duplicate entries in metrics"


def test_shipped_config_pins_dnsmos_acceleration():
    """DNSMOS defaults must stay accelerated: it dominates the runtime."""
    dns = load_config(DEFAULT_CONFIG_PATH)["dnsmos"]
    assert dns["intra_op_threads"] == 4, (
        "unbounded threads cost ~2.7x here (1475 vs 550 ms/utt, measured)")
    assert "CUDAExecutionProvider" in dns["providers"]
    assert "CPUExecutionProvider" in dns["providers"], "keep a CPU fallback"


def test_unknown_metric_in_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("metrics: [si_sdr, not_a_metric]\n")
    with pytest.raises(SystemExit, match="unknown metric"):
        load_config(str(bad))


def test_missing_explicit_config_path_raises(tmp_path):
    with pytest.raises(SystemExit, match="Config file not found"):
        load_config(str(tmp_path / "nope.yaml"))


def test_config_file_overrides_defaults_per_key(tmp_path):
    """A partial file must override only what it names, nested dicts merging."""
    partial = tmp_path / "p.yaml"
    partial.write_text("target_sr: 16000\ndnsmos:\n  intra_op_threads: 1\n")
    cfg = load_config(str(partial))
    assert cfg["target_sr"] == 16_000
    assert cfg["dnsmos"]["intra_op_threads"] == 1
    # providers were not named, so the default survives the merge
    assert cfg["dnsmos"]["providers"] == list(DEFAULT_PROVIDERS)
    assert cfg["si_sdr_backend"] == DEFAULTS["si_sdr_backend"]


# ─────────────────────────────────────────────────────────────────────────
# Provider resolution (no session is built)
# ─────────────────────────────────────────────────────────────────────────
def test_provider_aliases():
    assert resolve_providers("cuda") == DEFAULT_PROVIDERS
    assert resolve_providers("gpu") == DEFAULT_PROVIDERS
    assert resolve_providers("cpu") == CPU_ONLY_PROVIDERS
    assert resolve_providers("CPU") == CPU_ONLY_PROVIDERS      # case-insensitive
    assert resolve_providers(None) == DEFAULT_PROVIDERS


def test_provider_explicit_list_passes_through():
    explicit = ["CUDAExecutionProvider"]
    assert resolve_providers(explicit) == ("CUDAExecutionProvider",)
    # A bare non-alias string is treated as a provider name.
    assert resolve_providers("TensorrtExecutionProvider") == ("TensorrtExecutionProvider",)


def test_default_threads_is_bounded():
    """0 would restore onnxruntime's host-wide default, which is the slow path."""
    assert DEFAULT_INTRA_OP_THREADS == 4


def test_provenance_shape_is_json_safe():
    """The sidecar needs plain types, and None before any session exists."""
    p = provenance()
    assert set(p) == {"configured", "intra_op_threads",
                      "requested_providers", "actual_providers"}
    for value in p.values():
        assert value is None or isinstance(value, (bool, int, list))
