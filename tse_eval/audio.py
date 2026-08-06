"""
tse_eval.audio — audio loading and preprocessing helpers.

All metric code consumes 1-D float32 numpy arrays at a single working sample
rate.  This module centralises the load → mono → resample → align steps so the
metric functions stay simple.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torchaudio


def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file as a 1-D float32 numpy array at ``target_sr``.

    Steps: read → average channels to mono → resample if needed → float32.

    Args:
        path:       Path to an audio file (any format torchaudio can read).
        target_sr:  Sample rate the returned waveform is resampled to (Hz).

    Returns:
        1-D ``np.ndarray`` of shape ``[N]``, dtype float32.
    """
    wav, sr = torchaudio.load(path)                       # [C, N]
    if wav.shape[0] > 1:                                  # stereo/multi → mono
        wav = wav.mean(dim=0, keepdim=True)               # [1, N]
    wav = wav.squeeze(0)                                  # [N]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.detach().cpu().float().numpy()             # [N]


def resample_np(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D numpy waveform from ``orig_sr`` to ``target_sr``."""
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    out = torchaudio.functional.resample(t, orig_sr, target_sr)
    return out.numpy()


def align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim two 1-D waveforms to their common (minimum) length.

    Reference-based metrics (SI-SDR, PESQ, STOI) require identical-length
    signals; small end offsets are handled by trimming, never padding.
    """
    n = min(len(a), len(b))
    return a[:n], b[:n]


def align_triple(est: np.ndarray, ref: np.ndarray, mix: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim estimate / reference / mixture to one common length.

    ``n = min(len(est), len(ref), len(mix))``, then all three are sliced — the
    convention the sibling TSE projects already use (``llmtse/eval.py``), so
    SI-SDR and SI-SDRi are computed over the same window as their numbers.

    Trimming all three together (rather than pairwise per metric) is what makes
    ``si_sdri == si_sdr − input_si_sdr`` hold over one consistent window.
    """
    n = min(len(est), len(ref), len(mix))
    return est[:n], ref[:n], mix[:n]
