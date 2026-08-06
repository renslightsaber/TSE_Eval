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
    """SI-SDR family via asteroid. Requires equal-length 1-D inputs."""
    try:
        from asteroid.metrics import get_metrics
    except ImportError as exc:                              # pragma: no cover
        raise RuntimeError(
            "si_sdr backend 'asteroid' requested but asteroid is not installed. "
            "Install the test extra (`pip install -e '.[test]'`) or use "
            "--si-sdr-backend native (numerically equivalent)."
        ) from exc

    res = get_metrics(
        np.asarray(mix, dtype=np.float64)[None, :],
        np.asarray(ref, dtype=np.float64)[None, :],
        np.asarray(est, dtype=np.float64)[None, :],
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
    """Single SI-SDR value via ``backend`` (used for the pairwise baseline)."""
    if backend == "native":
        return _native_si_sdr(est, ref)
    if backend == "asteroid":
        # asteroid's entry point always wants a mixture; reuse ref as a stand-in
        # since only the est-vs-ref term is read back.
        return _asteroid_family(est, ref, ref)["si_sdr"]
    raise ValueError(f"Unknown si_sdr backend {backend!r}. Valid: {list(BACKENDS)}")
