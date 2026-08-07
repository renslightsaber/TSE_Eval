"""
tse_eval.config — scoring *policy* in a file, run-specific values on the CLI.

The split is deliberate:

* **config file** holds what must be *identical* across the three projects for
  the comparison to be fair — working sample rate, the metric set, the SI-SDR
  backend, the per-metric sample-rate policy, and the model ids.
* **CLI** holds what changes every run — which manifest, where to write, which
  model is being scored, which axes to stratify by.

CLI beats config, config beats these built-in defaults. Whatever the three
layers resolve to is written next to the results as ``<output>_config.json`` so
a paper can cite the exact configuration a number came from.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any, Dict, Optional

from .metrics import (DEFAULT_METRICS, METRIC_COLUMNS, METRIC_SR_POLICY,
                      SPK_SIM_MODEL_ID, WER_LANGUAGE, WER_MODEL_ID)
from .ort_setup import DEFAULT_INTRA_OP_THREADS, DEFAULT_PROVIDERS

# Shipped policy file, used when --config is not given and it exists.
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "config.yaml")

DEFAULTS: Dict[str, Any] = {
    "target_sr": 24_000,
    "si_sdr_backend": "native",
    "metrics": list(DEFAULT_METRICS),
    # ``normalize`` is descriptive, not a switch: the implementation lives in
    # metrics.dnsmos_normalize / metrics.wer. It is recorded here (and therefore
    # in every run's sidecar) because both normalisations change the reported
    # numbers and every system in a comparison must share them.
    "wer": {"model_id": WER_MODEL_ID, "language": WER_LANGUAGE,
            "normalize": "whisper_english"},
    "spk_sim": {"model_id": SPK_SIM_MODEL_ID},
    "dnsmos": {"intra_op_threads": DEFAULT_INTRA_OP_THREADS,
               "providers": list(DEFAULT_PROVIDERS),
               "normalize": "rms_-26dbov"},
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the policy file merged over :data:`DEFAULTS`.

    Args:
        path: Explicit config path. ``None`` uses the shipped
            ``configs/config.yaml`` when present, otherwise pure defaults.

    Raises:
        SystemExit: an explicitly-requested path does not exist, or the metric
            list names something unknown (failing loudly beats scoring with a
            silently different metric set).
    """
    cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULTS))   # deep copy
    chosen = path or (DEFAULT_CONFIG_PATH if os.path.isfile(DEFAULT_CONFIG_PATH) else None)

    if path and not os.path.isfile(path):
        raise SystemExit(f"Config file not found: {path}")

    if chosen:
        try:
            import yaml
        except ImportError as exc:                          # pragma: no cover
            raise SystemExit("PyYAML is required to read a config file.") from exc
        with open(chosen, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    cfg["_config_path"] = os.path.abspath(chosen) if chosen else None

    unknown = [m for m in cfg.get("metrics", []) if m not in METRIC_COLUMNS]
    if unknown:
        raise SystemExit(
            f"Config lists unknown metric(s): {unknown}. Valid: {METRIC_COLUMNS}")
    return cfg


def write_sidecar(path: str, resolved: Dict[str, Any]) -> None:
    """Write the resolved configuration + provenance as JSON.

    This is the file to cite: it records the metric set, the SI-SDR backend, the
    per-metric sample rates, model ids, the input manifest, and the versions of
    every library whose value could move a number.
    """
    payload = dict(resolved)
    payload["sample_rate_policy"] = {k: v for k, v in METRIC_SR_POLICY.items()}
    payload["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def _package_versions() -> Dict[str, Optional[str]]:
    """Versions of the libraries that can change a metric value."""
    from importlib.metadata import PackageNotFoundError, version
    names = ["numpy", "scipy", "pandas", "torch", "torchaudio", "pesq", "pystoi",
             "speechmos", "librosa", "onnxruntime-gpu", "onnxruntime",
             "speechbrain", "transformers", "jiwer", "asteroid", "tse-eval"]
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out
