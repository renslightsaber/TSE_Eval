"""
Generate a tiny synthetic example so you can run TSE_Eval end-to-end without
any real data.

Creates 6 short 16 kHz wav files under ``examples/wavs/`` and an
``examples/sample_input.csv`` with two utterances at two overlap ratios.

Usage:
    python examples/make_example.py
    python -m tse_eval -i examples/sample_input.csv -o examples/results.csv
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SR = 16_000
DUR = 2.0                      # seconds
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
WAV_DIR = os.path.join(HERE, "wavs")


def _tone(freq: float, seconds: float = DUR, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds: float = DUR, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.05 * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def _save(name: str, wav: np.ndarray) -> str:
    """Write a wav and return its path RELATIVE to the repo root (portable CSV).

    The sample CSV therefore stays valid on any machine, as long as the tool is
    run from the repo root after regenerating the wavs with this script.
    """
    abs_path = os.path.join(WAV_DIR, name)
    sf.write(abs_path, np.clip(wav, -1.0, 1.0), SR)
    return os.path.relpath(abs_path, REPO_ROOT)


def main() -> None:
    os.makedirs(WAV_DIR, exist_ok=True)
    rows = []

    for idx, (f_tgt, f_intf) in enumerate([(220.0, 440.0), (330.0, 550.0)]):
        target = _tone(f_tgt)                       # ground-truth target speaker
        interferer = _tone(f_intf)                  # competing speaker
        mixture = target + interferer + _noise(seed=idx)

        # A decent extraction (target + a little residual) and the mixture as GT paths.
        estimate = target + 0.1 * interferer + _noise(seed=idx + 10)

        ref_p = _save(f"utt{idx}_ref.wav", target)
        mix_p = _save(f"utt{idx}_mix.wav", mixture)
        est_p = _save(f"utt{idx}_est.wav", estimate)

        rows.append({
            "file_id": f"utt{idx}",
            "estimate": est_p,
            "reference": ref_p,
            "mixture": mix_p,
            "overlap": ["40%", "100%"][idx],        # demonstrates the summary grouping
        })

    csv_path = os.path.join(HERE, "sample_input.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} wav triples to {WAV_DIR}")
    print(f"Wrote sample CSV → {csv_path}")
    print("\nNow run:")
    print(f"  python -m tse_eval -i {csv_path} -o {os.path.join(HERE, 'results.csv')}")


if __name__ == "__main__":
    main()
