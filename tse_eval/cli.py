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

from .metrics import METRIC_COLUMNS
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
    p.add_argument("--target-sr", type=int, default=16_000,
                   help="Working sample rate. SI-SDR is SR-agnostic; perceptual "
                        "metrics always run at 16 kHz internally.")

    # Metric selection (all on by default).
    p.add_argument("--metrics", default="all",
                   help="Comma-separated subset of "
                        f"[{','.join(METRIC_COLUMNS)}], or 'all'.")

    # Column overrides (auto-detected if omitted).
    p.add_argument("--id-col", default=None, help="Column with the utterance id.")
    p.add_argument("--est-col", default=None, help="Column with the extracted/estimated wav path.")
    p.add_argument("--ref-col", default=None, help="Column with the ground-truth target wav path.")
    p.add_argument("--mix-col", default=None, help="Column with the mixture wav path.")
    p.add_argument("--ovr-col", default=None, help="Column with the overlap ratio (optional).")

    p.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    return p


def _resolve_metrics(arg: str) -> Optional[set]:
    if arg.strip().lower() == "all":
        return None
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

    metrics = _resolve_metrics(args.metrics)

    # Resolve summary path.
    if args.summary_output is None:
        base, ext = os.path.splitext(args.output)
        summary_path = f"{base}_summary{ext or '.csv'}"
    elif args.summary_output.strip().lower() == "none":
        summary_path = None
    else:
        summary_path = args.summary_output

    per_row, summary, cols = evaluate_csv(
        input_csv=args.input,
        target_sr=args.target_sr,
        metrics=metrics,
        id_col=args.id_col,
        est_col=args.est_col,
        ref_col=args.ref_col,
        mix_col=args.mix_col,
        ovr_col=args.ovr_col,
        progress=not args.no_progress,
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

    # Console preview of the summary.
    _print_summary(summary, cols.ovr)
    return 0


def _print_summary(summary, ovr_col: Optional[str]) -> None:
    print("\n===== Summary =====")
    try:
        with_pd = summary.copy()
        num_cols = [c for c in with_pd.columns if c not in (ovr_col, "group", "n")]
        for c in num_cols:
            with_pd[c] = with_pd[c].map(lambda v: f"{v:.4f}" if v == v else "nan")
        print(with_pd.to_string(index=False))
    except Exception:
        print(summary)


if __name__ == "__main__":
    sys.exit(main())
