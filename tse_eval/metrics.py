"""
tse_eval.metrics — core TSE evaluation metrics.

Implemented (all reference-based unless noted):

    si_sdr     SI-SDR  (scale-invariant SDR, dB)         est vs ref
    si_sdri    SI-SDR improvement over the mixture (dB)  est/ref/mix
    stoi       STOI                                      est vs ref  @16 kHz
    estoi      Extended STOI                             est vs ref  @16 kHz
    pesq       PESQ (wideband)                           est vs ref  @16 kHz
    dnsmos     DNSMOS SIG / BAK / OVRL (+P.808)          est only    @16 kHz (no-reference)

SI-SDR / SI-SDRi are implemented natively (no `asteroid` dependency).
Perceptual metrics (PESQ/STOI/ESTOI/DNSMOS) always operate at 16 kHz, so inputs
are resampled internally when the working sample rate differs.

Every metric degrades gracefully: on any failure the value is ``float('nan')``
and the key is still present, so a whole CSV never aborts because of one bad row.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .audio import align_pair, resample_np

# Perceptual metrics (PESQ / STOI / ESTOI / DNSMOS) are defined at 16 kHz.
PERCEPTUAL_SR = 16_000
_EPS = 1e-8

# Metric column order used by the output CSV.
METRIC_COLUMNS = [
    "si_sdr",
    "si_sdri",
    "stoi",
    "estoi",
    "pesq",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
    "dnsmos_p808",
]


# ─────────────────────────────────────────────────────────────────────────
# SI-SDR family (native, sample-rate agnostic)
# ─────────────────────────────────────────────────────────────────────────
def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = _EPS) -> float:
    """Scale-Invariant SDR in dB between an estimate and a reference.

    Zero-mean both signals, project ``est`` onto ``ref``, then take
    ``10·log10(||s_target||² / ||e_noise||²)``.  Scale-invariant, so a gain
    applied to ``est`` leaves the value unchanged.

    Args:
        est: Estimated (extracted) waveform, 1-D ``[N]``.
        ref: Reference (ground-truth target) waveform, 1-D ``[N]``.
        eps: Numerical floor.

    Returns:
        SI-SDR in dB (float). ``nan`` if the reference is (near-)silent.
    """
    est, ref = align_pair(np.asarray(est, np.float64), np.asarray(ref, np.float64))
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = float(np.dot(ref, ref))
    if ref_energy < eps:                                   # silent reference
        return float("nan")
    alpha = float(np.dot(est, ref)) / (ref_energy + eps)
    s_target = alpha * ref
    e_noise = est - s_target
    num = float(np.dot(s_target, s_target))
    den = float(np.dot(e_noise, e_noise))
    return 10.0 * np.log10((num + eps) / (den + eps))


def si_sdri(est: np.ndarray, ref: np.ndarray, mix: np.ndarray) -> float:
    """SI-SDR improvement: ``SI-SDR(est, ref) − SI-SDR(mix, ref)`` (dB).

    Measures how much the extraction improves over the unprocessed mixture.
    """
    out = si_sdr(est, ref)
    inp = si_sdr(mix, ref)
    if np.isnan(out) or np.isnan(inp):
        return float("nan")
    return out - inp


# ─────────────────────────────────────────────────────────────────────────
# Perceptual metrics (16 kHz)
# ─────────────────────────────────────────────────────────────────────────
def pesq_wb(ref16k: np.ndarray, est16k: np.ndarray) -> float:
    """PESQ wideband (P.862.2) at 16 kHz. ``nan`` on failure."""
    try:
        from pesq import pesq as _pesq
        ref16k, est16k = align_pair(ref16k, est16k)
        return float(_pesq(PERCEPTUAL_SR, ref16k, est16k, "wb"))
    except Exception:
        return float("nan")


def stoi_metric(ref16k: np.ndarray, est16k: np.ndarray, extended: bool) -> float:
    """STOI (``extended=False``) or ESTOI (``extended=True``) at 16 kHz."""
    try:
        from pystoi import stoi as _stoi
        ref16k, est16k = align_pair(ref16k, est16k)
        return float(_stoi(ref16k, est16k, PERCEPTUAL_SR, extended=extended))
    except Exception:
        return float("nan")


def dnsmos(est16k: np.ndarray) -> Dict[str, float]:
    """DNSMOS (P.835) on the estimate only — no reference needed.

    Returns SIG / BAK / OVRL and the P.808 MOS. Input is clipped to [-1, 1]
    (speechmos requirement). All ``nan`` if speechmos is unavailable.
    """
    keys = {"dnsmos_sig": "sig_mos", "dnsmos_bak": "bak_mos",
            "dnsmos_ovrl": "ovrl_mos", "dnsmos_p808": "p808_mos"}
    try:
        from speechmos import dnsmos as _dnsmos
        audio = np.clip(np.asarray(est16k, dtype=np.float32), -1.0, 1.0)
        res = _dnsmos.run(audio, sr=PERCEPTUAL_SR)
        return {out_key: float(res[src_key]) for out_key, src_key in keys.items()}
    except Exception:
        return {out_key: float("nan") for out_key in keys}


# ─────────────────────────────────────────────────────────────────────────
# Row-level aggregation
# ─────────────────────────────────────────────────────────────────────────
def compute_row_metrics(
    est: np.ndarray,
    ref: np.ndarray,
    mix: np.ndarray,
    sr: int,
    metrics: "set[str] | None" = None,
) -> Dict[str, float]:
    """Compute all core metrics for a single (est, ref, mix) triple.

    Args:
        est:     Estimated waveform at ``sr``, 1-D.
        ref:     Reference waveform at ``sr``, 1-D.
        mix:     Mixture waveform at ``sr``, 1-D.
        sr:      Working sample rate of the three inputs (Hz).
        metrics: Optional subset of :data:`METRIC_COLUMNS` to compute; ``None``
                 computes all. Skipped metrics are still present as ``nan``.

    Returns:
        Dict keyed by :data:`METRIC_COLUMNS` (all floats).
    """
    want = set(METRIC_COLUMNS) if metrics is None else set(metrics)
    out: Dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}

    # SI-SDR family — sample-rate agnostic, computed at the working sr.
    if "si_sdr" in want:
        out["si_sdr"] = si_sdr(est, ref)
    if "si_sdri" in want:
        out["si_sdri"] = si_sdri(est, ref, mix)

    # Perceptual metrics — resample to 16 kHz once, reuse.
    needs_ref_perceptual = want & {"stoi", "estoi", "pesq"}
    needs_est_perceptual = needs_ref_perceptual or (want & {
        "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"})
    if needs_est_perceptual:
        est16 = resample_np(est, sr, PERCEPTUAL_SR) if sr != PERCEPTUAL_SR else est
    if needs_ref_perceptual:
        ref16 = resample_np(ref, sr, PERCEPTUAL_SR) if sr != PERCEPTUAL_SR else ref
        if "stoi" in want:
            out["stoi"] = stoi_metric(ref16, est16, extended=False)
        if "estoi" in want:
            out["estoi"] = stoi_metric(ref16, est16, extended=True)
        if "pesq" in want:
            out["pesq"] = pesq_wb(ref16, est16)
    if want & {"dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"}:
        out.update(dnsmos(est16))

    return out
