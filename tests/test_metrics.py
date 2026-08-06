"""
Tests for tse_eval.metrics.

Validates invariants of SI-SDR/SI-SDRi (native implementation) and the
graceful, exception-free behaviour of the perceptual metrics (PESQ / STOI /
ESTOI / DNSMOS), plus the row-level aggregation in ``compute_row_metrics``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tse_eval.audio import align_pair
from tse_eval.metrics import (
    METRIC_COLUMNS,
    compute_row_metrics,
    dnsmos,
    pesq_wb,
    si_sdr,
    si_sdri,
    stoi_metric,
)

DNSMOS_KEYS = ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"]


# ─────────────────────────────────────────────────────────────────────────
# si_sdr
# ─────────────────────────────────────────────────────────────────────────
def test_si_sdr_self_is_very_high(speech_signal):
    """SI-SDR of a signal against itself should be extremely high (near-perfect)."""
    x = speech_signal(seed=1)
    value = si_sdr(x, x)
    assert math.isfinite(value)
    assert value > 40.0


def test_si_sdr_scale_invariant(speech_signal, noise_signal):
    """A pure gain on the estimate must not change SI-SDR (scale invariance).

    Uses an imperfect estimate (ref + small noise) rather than est == ref:
    when est == ref exactly, SI-SDR sits right at the log(~0) singularity,
    where float32 rounding noise makes the dB value itself unstable and
    unsuitable for a tight equality check.
    """
    ref = speech_signal(seed=1)
    est = ref + 0.05 * noise_signal(seed=50)
    base = si_sdr(est, ref)
    scaled = si_sdr(2.0 * est, ref)
    assert scaled == pytest.approx(base, abs=1e-3)


def test_si_sdr_shift_invariant_to_dc_offset(speech_signal, noise_signal):
    """Zero-meaning both signals absorbs a DC offset on the estimate."""
    ref = speech_signal(seed=1)
    est = ref + 0.05 * noise_signal(seed=50)
    base = si_sdr(est, ref)
    shifted = si_sdr(est + 0.3, ref)
    assert shifted == pytest.approx(base, abs=1e-3)


def test_si_sdr_low_for_independent_noise(speech_signal, noise_signal):
    """Independent broadband noise vs. the reference should score far lower
    than the reference against itself."""
    x = speech_signal(seed=1)
    noise = noise_signal(seed=99)
    self_score = si_sdr(x, x)
    noise_score = si_sdr(noise, x)
    assert math.isfinite(noise_score)
    assert noise_score < 5.0
    assert noise_score < self_score


def test_si_sdr_nan_for_silent_reference(speech_signal):
    """A (near-)silent reference makes SI-SDR undefined -> nan, not a crash."""
    x = speech_signal(seed=1)
    value = si_sdr(x, np.zeros_like(x))
    assert math.isnan(value)


# ─────────────────────────────────────────────────────────────────────────
# si_sdri
# ─────────────────────────────────────────────────────────────────────────
def test_si_sdri_positive_for_perfect_estimate(speech_signal, noise_signal):
    """A perfect estimate (est == ref) against a noisy mixture improves a lot."""
    ref = speech_signal(seed=1)
    interferer = noise_signal(seed=2)
    mix = ref + 0.8 * interferer
    value = si_sdri(ref, ref, mix)
    assert math.isfinite(value)
    assert value > 0.0


def test_si_sdri_zero_when_estimate_equals_mixture(speech_signal, noise_signal):
    """If the estimate IS the mixture, there is no improvement: SI-SDRi ~ 0."""
    ref = speech_signal(seed=1)
    interferer = noise_signal(seed=2)
    mix = ref + 0.8 * interferer
    value = si_sdri(mix, ref, mix)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_si_sdri_nan_when_reference_silent(speech_signal):
    """Propagates si_sdr's nan-on-silent-reference behaviour."""
    x = speech_signal(seed=1)
    value = si_sdri(x, np.zeros_like(x), x)
    assert math.isnan(value)


# ─────────────────────────────────────────────────────────────────────────
# align_pair
# ─────────────────────────────────────────────────────────────────────────
def test_align_pair_trims_to_shorter_length():
    a = np.arange(10, dtype=np.float64)
    b = np.arange(7, dtype=np.float64)
    a2, b2 = align_pair(a, b)
    assert len(a2) == 7
    assert len(b2) == 7
    np.testing.assert_array_equal(a2, a[:7])
    np.testing.assert_array_equal(b2, b)


# ─────────────────────────────────────────────────────────────────────────
# Perceptual metrics: pesq_wb / stoi_metric / dnsmos
# ─────────────────────────────────────────────────────────────────────────
def test_pesq_wb_finite_and_in_range(speech_signal):
    x = speech_signal(seed=1)
    value = pesq_wb(x, x)
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert 1.0 <= value <= 4.65  # PESQ-WB range is [-0.5, 4.5]; self-compare tops out near there


@pytest.mark.parametrize("extended", [False, True])
def test_stoi_metric_finite_and_in_range(speech_signal, extended):
    x = speech_signal(seed=1)
    value = stoi_metric(x, x, extended=extended)
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert -1.5 <= value <= 1.5  # (E)STOI is nominally in [0, 1]; self-compare should be ~1


def test_dnsmos_keys_and_range(speech_signal):
    x = speech_signal(seed=1)
    result = dnsmos(x)
    assert isinstance(result, dict)
    for key in DNSMOS_KEYS:
        assert key in result
        value = result[key]
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert 0.5 <= value <= 5.5


def test_pesq_wb_graceful_nan_on_degenerate_input():
    """All-zero (silent) signals are degenerate for PESQ -> nan, not a crash."""
    z = np.zeros(16_000, dtype=np.float32)
    value = pesq_wb(z, z)
    assert math.isnan(value)


def test_stoi_and_dnsmos_do_not_crash_on_degenerate_input():
    """All-zero input must never raise; values stay finite or become nan."""
    z = np.zeros(16_000, dtype=np.float32)

    stoi_value = stoi_metric(z, z, extended=False)
    estoi_value = stoi_metric(z, z, extended=True)
    assert isinstance(stoi_value, float)
    assert isinstance(estoi_value, float)
    assert math.isnan(stoi_value) or math.isfinite(stoi_value)
    assert math.isnan(estoi_value) or math.isfinite(estoi_value)

    dns = dnsmos(z)
    for key in DNSMOS_KEYS:
        assert key in dns
        assert math.isnan(dns[key]) or math.isfinite(dns[key])


# ─────────────────────────────────────────────────────────────────────────
# compute_row_metrics
# ─────────────────────────────────────────────────────────────────────────
def test_compute_row_metrics_returns_all_metric_columns(speech_signal, noise_signal):
    ref = speech_signal(seed=1)
    interferer = noise_signal(seed=2)
    mix = ref + 0.6 * interferer
    est = ref + 0.05 * interferer

    result = compute_row_metrics(est, ref, mix, sr=16_000)

    assert set(result.keys()) == set(METRIC_COLUMNS)
    for key in METRIC_COLUMNS:
        assert isinstance(result[key], float)
    assert math.isfinite(result["si_sdr"])


def test_compute_row_metrics_subset_leaves_others_nan(speech_signal, noise_signal):
    ref = speech_signal(seed=1)
    interferer = noise_signal(seed=2)
    mix = ref + 0.6 * interferer
    est = ref + 0.05 * interferer

    result = compute_row_metrics(est, ref, mix, sr=16_000, metrics={"si_sdr"})

    assert math.isfinite(result["si_sdr"])
    for key in METRIC_COLUMNS:
        if key != "si_sdr":
            assert math.isnan(result[key])


def test_compute_row_metrics_target_sr_24000_does_not_crash(speech_signal, noise_signal):
    """Every metric must produce a finite value at the 24 kHz working sr."""
    sr24 = 24_000
    ref = speech_signal(seed=1, sr=sr24)
    interferer = noise_signal(seed=2, sr=sr24)
    mix = ref + 0.6 * interferer
    est = ref + 0.05 * interferer

    result = compute_row_metrics(est, ref, mix, sr=sr24)

    assert set(result.keys()) == set(METRIC_COLUMNS)
    assert math.isfinite(result["si_sdr"])
    assert math.isfinite(result["si_sdri"])
    # STOI / ESTOI stay at the native 24 kHz (pystoi resamples to 10k itself).
    assert math.isfinite(result["stoi"])
    assert math.isfinite(result["estoi"])
    # PESQ / DNSMOS went through an internal 24k -> 16k resample.
    assert math.isfinite(result["pesq"])
    for key in DNSMOS_KEYS:
        assert math.isfinite(result[key])


def test_stoi_family_computed_at_native_sr_not_16k(speech_signal, noise_signal):
    """STOI/ESTOI must be computed at the working sr, not pre-resampled to 16 kHz.

    Pins the sample-rate protocol shared with the sibling TSE projects
    ("SI-SDR/SI-SDRi/ESTOI = native 24k, PESQ = 16k").  A regression that routes
    STOI/ESTOI through the 16 kHz path would make the row value match a
    ``sr=16000`` call on the same samples instead of the native-rate one.
    """
    sr24 = 24_000
    ref = speech_signal(seed=1, sr=sr24)
    interferer = noise_signal(seed=2, sr=sr24)
    mix = ref + 0.6 * interferer
    est = ref + 0.05 * interferer

    result = compute_row_metrics(est, ref, mix, sr=sr24)

    for extended, key in ((False, "stoi"), (True, "estoi")):
        native = stoi_metric(ref, est, sr24, extended=extended)
        # Same samples, but pystoi told the wrong rate -> resamples by a
        # different factor -> clearly different score (~0.05 apart).
        as_if_16k = stoi_metric(ref, est, 16_000, extended=extended)

        # rel_tol, not ==: the two call paths can differ in the last ULP.
        assert math.isclose(result[key], native, rel_tol=1e-9), (
            f"{key} is not computed at the native sr "
            f"({result[key]} vs native {native})")
        assert not math.isclose(native, as_if_16k, rel_tol=1e-3), (
            f"{key}: the native-sr and 16 kHz calls are indistinguishable, so "
            "this test could not catch a regression")


def test_stoi_metric_sr_defaults_to_16k_for_backward_compat(speech_signal):
    """``sr`` defaults to 16 kHz so pre-existing two-positional-arg calls hold."""
    x = speech_signal(seed=1)          # fixture default is 16 kHz
    assert stoi_metric(x, x, extended=True) == stoi_metric(x, x, 16_000, extended=True)
