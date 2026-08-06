"""
Tests for tse_eval.backends — SI-SDR definition equivalence.

This file is the evidence that ``native`` and ``asteroid`` compute the *same*
quantity, which is what lets a paper claim one pipeline scored every system.

Two layers, on purpose:

1. **Golden values (always run).** Numbers captured from
   ``asteroid.metrics`` / ``pb_bss_eval`` on this exact signal set and hardcoded
   below. They guard the definition even in environments without asteroid — and
   the runtime install deliberately omits asteroid (22 extra packages).
2. **Live comparison (runs only where asteroid is installed.)** Guards against
   the goldens themselves drifting from the library.

The degenerate cases matter most: an earlier implementation added ``eps`` to both
sides of the ratio and returned ``0.0 dB`` for a silent / DC-only / near-zero
estimate, which quietly inflated the mean. asteroid's floor is
``10·log10(1e-8) = -80 dB``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tse_eval.audio import align_pair
from tse_eval.backends import (BACKENDS, DEFAULT_BACKEND, available_backends,
                               si_sdr_family, si_sdr_value)
from tse_eval.metrics import si_sdr

SR = 16_000
N = 16_000  # 1.0 s

# SI-SDR values from asteroid/pb_bss_eval for the signals built by _cases().
# Provenance: pb_bss_eval 0.0.2 via asteroid 0.7.0, captured 2026-08-06.
ASTEROID_GOLDEN = {
    "normal": 8.616265210448,
    "exact": 110.624654927760,
    "scaled_2x": 116.645254840927,
    "silent_est": -80.000000000000,
    "tiny_1e8": -49.371585565792,
    "dc_only": -80.000000000000,
    "reversed": 10.297963508471,
    "noise_only": -73.428065097617,
}

# 10·log10(pb_bss_eval.EPS) — the floor a degenerate estimate must land on.
ASTEROID_FLOOR_DB = -80.0


def _reference() -> np.ndarray:
    """The fixed reference signal the goldens were captured against."""
    t = np.arange(N) / SR
    return (0.30 * np.sin(2 * np.pi * 220 * t)
            + 0.20 * np.sin(2 * np.pi * 440 * t)
            + 0.10 * np.sin(2 * np.pi * 660 * t)
            + 0.05 * np.random.default_rng(0).standard_normal(N))


def _cases(ref: np.ndarray) -> "dict[str, np.ndarray]":
    """Estimates covering normal operation and every degenerate shape."""
    return {
        "normal": ref + 0.1 * np.random.default_rng(1).standard_normal(N),
        "exact": ref.copy(),
        "scaled_2x": ref * 2.0,            # scale invariance
        "silent_est": np.zeros(N),         # → floor
        "tiny_1e8": ref * 1e-8,            # near-zero
        "dc_only": np.full(N, 0.5),        # zero after mean removal → floor
        "reversed": ref[::-1].copy(),
        "noise_only": np.random.default_rng(2).standard_normal(N),
    }


# ─────────────────────────────────────────────────────────────────────────
# Layer 1 — golden values (no asteroid needed)
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", sorted(ASTEROID_GOLDEN))
def test_native_matches_asteroid_golden_value(case):
    ref = _reference()
    got = si_sdr(_cases(ref)[case], ref)
    assert got == pytest.approx(ASTEROID_GOLDEN[case], abs=1e-6), (
        f"{case}: native SI-SDR drifted from the asteroid definition")


@pytest.mark.parametrize("case", ["silent_est", "dc_only"])
def test_degenerate_estimate_hits_the_floor_not_zero(case):
    """A silent/DC estimate must floor at -80 dB, never look like 0 dB.

    0 dB would read as "as much signal as noise" and pull the mean up; the whole
    point of the eps fix is that these land on the floor instead.
    """
    ref = _reference()
    got = si_sdr(_cases(ref)[case], ref)
    assert got == pytest.approx(ASTEROID_FLOOR_DB, abs=1e-9)
    assert got < -50.0


def test_scale_invariance():
    """Scaling the estimate must not change SI-SDR (that is the 'SI').

    Invariance is exact in the algebra but only approximate in floating point,
    because ``eps`` is a fixed absolute term while the signal energy scales: the
    deviation grows like ``1/gain²``. Measured here — gain 1000 → 2.7e-10,
    gain 0.01 → 2.7e-6, gain 0.001 → 2.7e-4. asteroid shares the property since
    it is the same formula, so this is a documented characteristic, not drift.
    """
    ref = _reference()
    est = ref + 0.1 * np.random.default_rng(1).standard_normal(N)
    base = si_sdr(est, ref)
    for gain in (0.5, 2.0, 1000.0):
        assert si_sdr(est * gain, ref) == pytest.approx(base, abs=1e-8), (
            f"gain {gain} broke scale invariance")
    # Heavily attenuated estimates drift at the eps level — bounded, not exact.
    assert si_sdr(est * 0.01, ref) == pytest.approx(base, abs=1e-5)


def test_silent_reference_returns_nan_not_a_floor():
    """The one deliberate divergence from asteroid.

    asteroid yields a finite floor for a silent reference; we return nan so the
    row drops out of the mean instead of dragging it toward -80 dB.
    """
    assert math.isnan(si_sdr(_reference(), np.zeros(N)))


def test_length_mismatch_is_trimmed_not_an_error():
    """asteroid raises on unequal lengths; we trim to the common length.

    The trimmed result must equal what the backend gives for pre-trimmed input.
    """
    ref = _reference()
    est = (ref + 0.1 * np.random.default_rng(1).standard_normal(N))[: N - 1234]
    trimmed_est, trimmed_ref = align_pair(est, ref)
    assert si_sdr(est, ref) == pytest.approx(si_sdr(trimmed_est, trimmed_ref), abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# Layer 2 — live comparison against the real library
# ─────────────────────────────────────────────────────────────────────────
def test_golden_values_still_match_installed_asteroid():
    """Catch the goldens drifting from the library (skips without asteroid)."""
    pytest.importorskip("asteroid", reason="asteroid is a test-only extra")
    from pb_bss_eval.evaluation.module_si_sdr import si_sdr as reference_impl

    ref = _reference()
    for case, est in _cases(ref).items():
        live = float(reference_impl(ref, est))
        assert live == pytest.approx(ASTEROID_GOLDEN[case], abs=1e-6), (
            f"{case}: hardcoded golden no longer matches installed asteroid")
        assert si_sdr(est, ref) == pytest.approx(live, abs=1e-6)


def test_both_backends_agree_on_the_whole_family():
    """native and asteroid must agree on si_sdr / si_sdri / input_si_sdr."""
    pytest.importorskip("asteroid", reason="asteroid is a test-only extra")
    rng = np.random.default_rng(7)
    ref = _reference()
    mix = ref + 0.6 * rng.standard_normal(N)
    est = ref + 0.05 * rng.standard_normal(N)

    a = si_sdr_family(est, ref, mix, backend="native")
    b = si_sdr_family(est, ref, mix, backend="asteroid")
    for key in ("si_sdr", "si_sdri", "input_si_sdr"):
        assert a[key] == pytest.approx(b[key], abs=1e-6), f"{key} differs"


# ─────────────────────────────────────────────────────────────────────────
# Dispatch behaviour
# ─────────────────────────────────────────────────────────────────────────
def test_default_backend_is_native_and_needs_no_extras():
    assert DEFAULT_BACKEND == "native"
    assert available_backends()["native"] is True
    assert set(BACKENDS) == {"native", "asteroid"}


def test_unknown_backend_raises():
    ref = _reference()
    with pytest.raises(ValueError, match="Unknown si_sdr backend"):
        si_sdr_family(ref, ref, ref, backend="nope")
    with pytest.raises(ValueError, match="Unknown si_sdr backend"):
        si_sdr_value(ref, ref, backend="nope")


def test_missing_asteroid_raises_instead_of_falling_back():
    """No silent fallback: which backend ran is part of the provenance."""
    if available_backends()["asteroid"]:
        pytest.skip("asteroid is installed here, so the error path cannot trigger")
    ref = _reference()
    with pytest.raises(RuntimeError, match="asteroid"):
        si_sdr_family(ref, ref, ref, backend="asteroid")
