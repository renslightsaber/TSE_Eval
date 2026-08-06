"""
tse_eval.cli — command-line entrypoint.

Examples:
    python -m tse_eval --input preds.csv --output results.csv
    tse-eval --input preds.csv --output results.csv --summary-output summary.csv
    tse-eval --input preds.csv --output results.csv --est-col my_est --ovr-col overlap
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from .backends import available_backends
from .config import load_config, write_sidecar
from .metrics import METRIC_COLUMNS, configure_models
from .ort_setup import DEFAULT_INTRA_OP_THREADS
from .evaluate import evaluate_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tse-eval",
        description="Unified evaluation for Target Speech Extraction (TSE). "
                    "Reads a CSV of audio paths, writes per-row metrics and an "
                    "overlap-stratified summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="Input CSV with id / estimate / reference / mixture paths.")
    p.add_argument("--output", "-o", required=True,
                   help="Output per-row CSV path.")
    p.add_argument("--summary-output", "-s", default=None,
                   help="Summary CSV path. Default: '<output>_summary.csv'. "
                        "Use 'none' to skip the summary.")
    p.add_argument("--target-sr", type=int, default=None,
                   help="Working sample rate. Default comes from the config "
                        "(24000 = PORTE-v3 native). SI-SDR/SI-SDRi/STOI/ESTOI are "
                        "computed at this rate; PESQ/DNSMOS/WER/spk_sim always run "
                        "at 16 kHz internally.")

    # Metric selection. 'all' means the config's metric set (model-backed extras
    # excluded unless asked for); name them explicitly to add wer / spk_sim.
    p.add_argument("--metrics", default=None,
                   help="Comma-separated subset of "
                        f"[{','.join(METRIC_COLUMNS)}], or 'all' for the config's "
                        "set. 'wer' and 'spk_sim' need model downloads and are "
                        "off unless requested.")

    # Policy file + backend.
    p.add_argument("--config", default=None,
                   help="Scoring policy YAML. Default: configs/config.yaml if present.")
    p.add_argument("--si-sdr-backend", default=None, choices=["native", "asteroid"],
                   help="SI-SDR implementation. Equivalent to ~1e-13; 'native' "
                        "avoids asteroid's 22 extra packages.")
    p.add_argument("--dnsmos-providers", default=None,
                   help="onnxruntime execution providers for DNSMOS: 'cuda' "
                        "(default, ~30x faster) or 'cpu'. ★ CUDA shifts DNSMOS "
                        "values by ~1.5e-3, so every system in a comparison must "
                        "use the same setting (recorded in the sidecar).")
    p.add_argument("--dnsmos-threads", type=int, default=None,
                   help="DNSMOS intra-op thread count (0 = onnxruntime default, "
                        "which is ~3x slower here and spams affinity warnings).")

    # Run-specific inputs.
    p.add_argument("--model-name", default=None,
                   help="Label for the system being scored (e.g. tpex, llmtse, "
                        "styletse). Recorded in the summary and the sidecar.")
    p.add_argument("--source-csv", default=None,
                   help="Optional PORTE-v3 source CSV, left-joined on the id "
                        "column to supply target_sentence and continuous axes.")
    p.add_argument("--group-by", default=None,
                   help="Comma-separated stratification axes, e.g. "
                        "'overlap_ratio,prompt_category,same_gender,first_speak'. "
                        "'same_gender' is derived from target/infer gender. "
                        "Default: the detected overlap column.")

    # Column overrides (auto-detected if omitted).
    p.add_argument("--id-col", default=None, help="Column with the utterance id.")
    p.add_argument("--est-col", default=None, help="Column with the extracted/estimated wav path.")
    p.add_argument("--ref-col", default=None, help="Column with the ground-truth target wav path.")
    p.add_argument("--mix-col", default=None, help="Column with the mixture wav path.")
    p.add_argument("--ovr-col", default=None, help="Column with the overlap ratio (optional).")
    p.add_argument("--txt-col", default=None, help="Column with the WER reference transcript.")

    p.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    return p


def _resolve_metrics(arg: Optional[str], config_metrics: list) -> Optional[set]:
    """CLI metric selection, falling back to the config's set.

    ``None`` (flag omitted) or ``'all'`` -> the config's metric set.
    An explicit list wins and may name the model-backed extras.
    """
    if arg is None or arg.strip().lower() == "all":
        return set(config_metrics)
    requested = [m.strip() for m in arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in METRIC_COLUMNS]
    if unknown:
        raise SystemExit(
            f"Unknown metric(s): {unknown}. Valid: {METRIC_COLUMNS} or 'all'.")
    return set(requested)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input CSV not found: {args.input}")

    # Layered configuration: CLI > config file > built-in defaults.
    cfg = load_config(args.config)
    target_sr = args.target_sr if args.target_sr is not None else cfg["target_sr"]
    backend = args.si_sdr_backend or cfg["si_sdr_backend"]
    metrics = _resolve_metrics(args.metrics, cfg["metrics"])

    # Fail fast on a backend that cannot run: otherwise every row would just
    # record the same error and the output would be a CSV full of NaNs.
    if not available_backends().get(backend, False):
        raise SystemExit(
            f"si_sdr backend '{backend}' is not available. Install the test "
            f"extra (pip install -e '.[test]') or use --si-sdr-backend native "
            f"(equivalent to ~1e-13).")
    configure_models(wer_model_id=cfg.get("wer", {}).get("model_id"),
                     wer_language=cfg.get("wer", {}).get("language"),
                     spk_sim_model_id=cfg.get("spk_sim", {}).get("model_id"))

    dns_cfg = cfg.get("dnsmos", {}) or {}
    dnsmos_threads = (args.dnsmos_threads if args.dnsmos_threads is not None
                      else dns_cfg.get("intra_op_threads", DEFAULT_INTRA_OP_THREADS))
    dnsmos_providers = args.dnsmos_providers or dns_cfg.get("providers")

    # Resolve summary path.
    if args.summary_output is None:
        base, ext = os.path.splitext(args.output)
        summary_path = f"{base}_summary{ext or '.csv'}"
    elif args.summary_output.strip().lower() == "none":
        summary_path = None
    else:
        summary_path = args.summary_output

    per_row, summary, cols, info = evaluate_csv(
        input_csv=args.input,
        target_sr=target_sr,
        metrics=metrics,
        id_col=args.id_col,
        est_col=args.est_col,
        ref_col=args.ref_col,
        mix_col=args.mix_col,
        ovr_col=args.ovr_col,
        txt_col=args.txt_col,
        progress=not args.no_progress,
        source_csv=args.source_csv,
        group_by=args.group_by,
        model_name=args.model_name,
        si_sdr_backend=backend,
        dnsmos_threads=dnsmos_threads,
        dnsmos_providers=dnsmos_providers,
    )

    for d in (args.output, summary_path):
        parent = os.path.dirname(os.path.abspath(d)) if d else None
        if parent:
            os.makedirs(parent, exist_ok=True)

    per_row.to_csv(args.output, index=False)
    print(f"[tse-eval] wrote per-row metrics → {args.output}  ({len(per_row)} rows)")

    if summary_path is not None:
        summary.to_csv(summary_path, index=False)
        print(f"[tse-eval] wrote summary      → {summary_path}")

    # Provenance sidecar — the file to cite in the paper.
    base, _ = os.path.splitext(args.output)
    sidecar_path = f"{base}_config.json"
    write_sidecar(sidecar_path, {
        "input_csv": os.path.abspath(args.input),
        "output_csv": os.path.abspath(args.output),
        "model_name": args.model_name,
        "target_sr": target_sr,
        "si_sdr_backend": backend,
        "metrics": sorted(metrics) if metrics else [],
        "config_file": cfg.get("_config_path"),
        "wer": cfg.get("wer"),
        "spk_sim": cfg.get("spk_sim"),
        **info,
    })
    print(f"[tse-eval] wrote run config   → {sidecar_path}")

    if info.get("source_join"):
        j = info["source_join"]
        print(f"[tse-eval] joined source CSV  → {j['matched_rows']}/{info['n_rows']} "
              f"rows matched; {len(j['columns_from_source'])} column(s) taken from "
              f"source, {len(j['columns_kept_from_manifest'])} kept from manifest")
    if info.get("group_by_missing"):
        print(f"[tse-eval] ⚠ axis not in data, skipped: {info['group_by_missing']}")

    # Console preview of the summary.
    _print_summary(summary)
    return 0


def _print_summary(summary) -> None:
    print("\n===== Summary =====")
    try:
        shown = summary.copy()
        label_cols = {"model_name", "axis", "group", "n"}
        for c in [c for c in shown.columns if c not in label_cols]:
            shown[c] = shown[c].map(lambda v: f"{v:.4f}" if v == v else "nan")
        if not shown["model_name"].astype(bool).any():
            shown = shown.drop(columns=["model_name"])
        print(shown.to_string(index=False))
    except Exception:
        print(summary)


if __name__ == "__main__":
    sys.exit(main())
