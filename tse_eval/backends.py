"""
tse_eval.backends — interchangeable SI-SDR implementations.

Two backends compute the same quantity:

    native    the in-repo implementation (:func:`tse_eval.metrics.si_sdr`).
              No extra dependency. This is the default.
    asteroid  ``asteroid.metrics.get_metrics``, i.e. ``pb_bss_eval`` underneath.

They agree to ~1e-13 because :func:`tse_eval.metrics.si_sdr` copies
``pb_bss_eval``'s epsilon placement exactly; ``tests/test_backends.py`` pins that
with golden values captured from asteroid, and additionally compares against the
real library when it is installed.

Why ``native`` is the default: the two are numerically interchangeable, and
``asteroid`` pulls in 22 further packages (pytorch-lightning, torchmetrics,
mir_eval, …). Keeping it out of the runtime install leaves the environment lean;
it is declared as a *test* extra so it can still serve as the equivalence oracle.

Only ``si_sdr`` is requested from asteroid — never ``sdr``/``sir``/``sar``
(≈1 s per utterance, and SIR degenerates without an interference signal), and
never asteroid's ``stoi``/``pesq`` (its STOI is non-extended, unlike the ESTOI we
report, and its PESQ carries 8/16 kHz constraints we handle ourselves).

There is no silent fallback: asking for a backend that is not installed raises.
Which backend produced a number is part of the provenance we record, so guessing
would defeat the purpose.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .audio import align_pair, align_triple
from .metrics import si_sdr as _native_si_sdr, si_sdr_family as _native_family

BACKENDS = ("native", "asteroid")
DEFAULT_BACKEND = "native"


def available_backends() -> Dict[str, bool]:
    """Map backend name -> importable right now (for logging / provenance)."""
    try:
        import asteroid.metrics  # noqa: F401
        has_asteroid = True
    except Exception:                                       # noqa: BLE001
        has_asteroid = False
    return {"native": True, "asteroid": has_asteroid}


def _asteroid_family(est: np.ndarray, ref: np.ndarray, mix: np.ndarray
                     ) -> Dict[str, float]:
    """SI-SDR family via asteroid.

    ★ Inputs are trimmed to a common length first. ``get_metrics`` asserts that
    all three arrays match and raises ``AssertionError`` otherwise, whereas this
    project's contract is "unequal lengths are trimmed, never padded"
    (:func:`tse_eval.audio.align_triple`). Without this trim the asteroid backend
    would blow up on rows the native backend handles fine, which breaks the
    interchangeability the two backends are supposed to guarantee.
    """
    try:
        from asteroid.metrics import get_metrics
    except ImportError as exc:                              # pragma: no cover
        raise RuntimeError(
            "si_sdr backend 'asteroid' requested but asteroid is not installed. "
            "Install the test extra (`pip install -e '.[test]'`) or use "
            "--si-sdr-backend native (numerically equivalent)."
        ) from exc

    est, ref, mix = align_triple(np.asarray(est, dtype=np.float64),
                                 np.asarray(ref, dtype=np.float64),
                                 np.asarray(mix, dtype=np.float64))
    res = get_metrics(
        mix[None, :], ref[None, :], est[None, :],
        sample_rate=16_000,          # SI-SDR is rate-agnostic; value is unused
        metrics_list=["si_sdr"],     # ★ never sdr/sir/sar/stoi/pesq
    )
    out, inp = float(res["si_sdr"]), float(res["input_si_sdr"])
    return {"si_sdr": out, "si_sdri": out - inp, "input_si_sdr": inp}


def si_sdr_family(est: np.ndarray, ref: np.ndarray, mix: np.ndarray,
                  backend: str = DEFAULT_BACKEND) -> Dict[str, float]:
    """Dispatch the SI-SDR family to ``backend``.

    Args:
        est/ref/mix: 1-D waveforms at a common rate (already length-aligned).
        backend: ``"native"`` or ``"asteroid"``.

    Returns:
        Dict with ``si_sdr``, ``si_sdri``, ``input_si_sdr``.

    Raises:
        ValueError: unknown backend name.
        RuntimeError: asteroid requested but not installed.
    """
    if backend == "native":
        return _native_family(est, ref, mix)
    if backend == "asteroid":
        return _asteroid_family(est, ref, mix)
    raise ValueError(f"Unknown si_sdr backend {backend!r}. Valid: {list(BACKENDS)}")


def si_sdr_value(est: np.ndarray, ref: np.ndarray,
                 backend: str = DEFAULT_BACKEND) -> float:
    """Single SI-SDR value between two signals, via ``backend``.

    Used for ``input_si_sdr_pairwise``, where the caller passes ``(mix, ref)``.
    ★ Alignment here is **pairwise on purpose** — trimming to
    ``min(len(a), len(b))`` and nothing else. That is what makes the pairwise
    baseline independent of the estimate: bring a third signal into the window
    and the value would start moving with the estimate's length, which is exactly
    the property this column exists to avoid.
    """
    if backend == "native":
        return _native_si_sdr(est, ref)             # aligns pairwise internally
    if backend == "asteroid":
        # asteroid's entry point always wants a mixture; reuse ref as a stand-in
        # since only the est-vs-ref term is read back. Align pairwise *first* so
        # the stand-in cannot widen or narrow the window.
        a, b = align_pair(np.asarray(est, dtype=np.float64),
                          np.asarray(ref, dtype=np.float64))
        return _asteroid_family(a, b, b)["si_sdr"]
    raise ValueError(f"Unknown si_sdr backend {backend!r}. Valid: {list(BACKENDS)}")
