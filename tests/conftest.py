"""
Shared pytest fixtures for the tse_eval test suite.

All synthetic audio is generated in-process (no real data, no network, no
GPU) and written to ``tmp_path`` with ``soundfile``. Signals are broadband
and speech-like (a few harmonics + noise) rather than pure tones, since
PESQ/STOI are meaningless (and can go negative) on pure sinusoids.
"""
from __future__ import annotations

import csv
from typing import Callable, List, Sequence

import numpy as np
import pytest
import soundfile as sf

SR = 16_000
DUR = 1.5  # seconds — long enough for PESQ/STOI, short enough to stay fast


def _speech_like(seed: int, n: int, sr: int, amp: float = 1.0) -> np.ndarray:
    """Broadband, speech-like synthetic signal: a few harmonics + noise.

    Deliberately NOT a pure tone — PESQ/STOI degrade meaninglessly on a
    single sinusoid, so every perceptual-metric test uses this instead.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    sig = (
        0.30 * np.sin(2 * np.pi * 220.0 * t)
        + 0.20 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 660.0 * t)
    )
    sig = sig + 0.05 * rng.standard_normal(n)
    return (amp * sig).astype(np.float32)


def _white_noise(seed: int, n: int) -> np.ndarray:
    """Independent broadband noise, uncorrelated across different seeds."""
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(n)).astype(np.float32)


@pytest.fixture(scope="session")
def sr() -> int:
    return SR


@pytest.fixture(scope="session")
def duration() -> float:
    return DUR


@pytest.fixture
def speech_signal() -> Callable[..., np.ndarray]:
    """Factory: speech_signal(seed=1, seconds=DUR, sr=SR, amp=1.0) -> np.ndarray [N]."""

    def _make(seed: int = 1, seconds: float = DUR, sr: int = SR, amp: float = 1.0) -> np.ndarray:
        return _speech_like(seed, int(seconds * sr), sr, amp)

    return _make


@pytest.fixture
def noise_signal() -> Callable[..., np.ndarray]:
    """Factory: noise_signal(seed=2, seconds=DUR, sr=SR) -> np.ndarray [N]."""

    def _make(seed: int = 2, seconds: float = DUR, sr: int = SR) -> np.ndarray:
        return _white_noise(seed, int(seconds * sr))

    return _make


@pytest.fixture
def write_wav(tmp_path):
    """Factory: write_wav(name, wav, sr=SR) -> str path (written under tmp_path)."""

    def _write(name: str, wav: np.ndarray, sr: int = SR) -> str:
        path = tmp_path / name
        sf.write(str(path), np.clip(wav, -1.0, 1.0), sr)
        return str(path)

    return _write


@pytest.fixture
def wav_triple(write_wav, speech_signal, noise_signal):
    """Factory building one (est_path, ref_path, mix_path) wav triple.

    ``est`` is a decent (mostly-clean) extraction of ``ref`` from a noisier
    ``mix``, so SI-SDRi is positive and perceptual metrics behave sensibly.
    """

    def _make(idx: int = 0, sr: int = SR, seconds: float = DUR):
        target = speech_signal(seed=100 + idx, seconds=seconds, sr=sr)
        interferer = noise_signal(seed=200 + idx, seconds=seconds, sr=sr)
        mixture = target + 0.6 * interferer
        estimate = target + 0.05 * interferer  # decent extraction
        ref_p = write_wav(f"utt{idx}_ref.wav", target, sr)
        mix_p = write_wav(f"utt{idx}_mix.wav", mixture, sr)
        est_p = write_wav(f"utt{idx}_est.wav", estimate, sr)
        return est_p, ref_p, mix_p

    return _make


@pytest.fixture
def write_input_csv(tmp_path):
    """Factory: write_input_csv(rows, name='input.csv') -> str csv path.

    ``rows`` is a list of dicts; column names come from the first row's keys.
    """

    def _write(rows: Sequence[dict], name: str = "input.csv") -> str:
        csv_path = tmp_path / name
        fieldnames: List[str] = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(csv_path)

    return _write
