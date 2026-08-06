"""
tse_eval.ort_setup — make ``speechmos``'s ONNX sessions fast and explicit.

``speechmos`` builds its DNSMOS sessions with a bare
``ort.InferenceSession(path)`` (``speechmos/dnsmos.py:19-20``) — no
``SessionOptions``, no ``providers``. Two consequences, both measured on this
machine:

1. **Thread oversubscription.** onnxruntime sizes its intra-op pool from the
   *host* CPU topology (``/proc/cpuinfo`` shows 224 cores) and pins each thread
   to a core, but the container's cgroup only grants 24. Every pin fails with a
   red ``pthread_setaffinity_np failed`` line, and the oversubscribed pool is
   ~3x slower than an explicit 4-thread pool.
2. **CPU-only execution.** Since onnxruntime 1.9, omitting ``providers`` selects
   the CPU EP alone — CUDA is never even attempted. Installing
   ``onnxruntime-gpu`` therefore does nothing on its own.

This module wraps ``ort.InferenceSession`` so sessions created *without*
explicit options get both a bounded thread pool and a provider list. Sessions
that pass their own ``sess_options`` are left untouched.

Measured end-to-end through this pipeline on real PORTE-v3 audio (mean 11 s,
lengths all different). Marginal cost = (60-row run − 20-row run) / 40, which
cancels fixed startup; DNSMOS was the only metric requested, so the figure also
carries the per-row wav I/O both settings share. Extrapolated to 15,000
utterances (5,000 rows x 3 systems):

    setting                          marginal      15,000 utt   vs as-is
    ──────────────────────────────   ───────────   ──────────   ────────
    as-is (threads=0, CPU)           1475 ms/utt      369 min      1.0x
    intra_op=4 (CPU)                  550 ms/utt      137 min      2.7x
    intra_op=4 + CUDA  ← default      175 ms/utt       44 min      8.4x

CUDA additionally pays ~40 s of one-time context/kernel setup (about 0.5 % of a
5,000-row run).

★ ``OMP_NUM_THREADS`` does **not** help: onnxruntime's intra-op pool is its own,
not OpenMP's. ``SessionOptions.intra_op_num_threads`` is the only lever.

★ CUDA changes DNSMOS values by up to ~3e-03 versus CPU (two CPU settings differ
by only 4e-07). DNSMOS is normally reported to two decimals so this sits below
the reported precision, but **every system in a comparison must use the same
providers**. :func:`actual_providers` records what really ran, and the CLI writes
it into the sidecar, so a mismatch is detectable after the fact.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Sequence, Tuple

# Sensible defaults for this container: 4 threads was fastest on 24 allowed CPUs,
# and CUDA-first with a CPU fallback in the list.
DEFAULT_INTRA_OP_THREADS = 4
DEFAULT_PROVIDERS: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
CPU_ONLY_PROVIDERS: Tuple[str, ...] = ("CPUExecutionProvider",)

# Friendly aliases for the CLI.
PROVIDER_ALIASES = {
    "cuda": DEFAULT_PROVIDERS,
    "gpu": DEFAULT_PROVIDERS,
    "cpu": CPU_ONLY_PROVIDERS,
}

_configured = False
_requested_providers: Optional[Tuple[str, ...]] = None
_actual_providers: Optional[Tuple[str, ...]] = None
_intra_op_threads: Optional[int] = None
_original_session_cls = None      # kept so the wrapper can be undone (tests)


def resolve_providers(spec: "str | Sequence[str] | None") -> Tuple[str, ...]:
    """Turn a CLI/config value into an onnxruntime provider tuple.

    Accepts ``"cuda"`` / ``"gpu"`` / ``"cpu"``, or an explicit list such as
    ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
    """
    if spec is None:
        return DEFAULT_PROVIDERS
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in PROVIDER_ALIASES:
            return PROVIDER_ALIASES[key]
        return (spec.strip(),)
    return tuple(spec)


def configure_onnxruntime(threads: int = DEFAULT_INTRA_OP_THREADS,
                          providers: "str | Sequence[str] | None" = None,
                          verbose: bool = True) -> None:
    """Install the session wrapper. Idempotent; the first call wins.

    Args:
        threads: intra-op thread count. ``0`` leaves onnxruntime's default in
            place (slow here, plus the affinity warnings).
        providers: provider spec — see :func:`resolve_providers`.
        verbose: print a one-line notice. Worth keeping on for CUDA, whose first
            session takes ~40 s to build and otherwise looks like a hang.

    Does nothing (silently) if onnxruntime is not importable — DNSMOS will then
    degrade to NaN with its own warning.
    """
    global _configured, _requested_providers, _intra_op_threads, _original_session_cls
    if _configured:
        return

    try:
        import onnxruntime as ort
    except ImportError:
        return

    wanted = resolve_providers(providers)
    available = set(ort.get_available_providers())
    usable = tuple(p for p in wanted if p in available)
    if not usable:
        usable = CPU_ONLY_PROVIDERS
        if verbose and wanted != CPU_ONLY_PROVIDERS:
            print(f"[tse-eval] ⚠ requested ONNX providers {list(wanted)} are "
                  f"unavailable; falling back to CPU", file=sys.stderr)

    _requested_providers = wanted
    _intra_op_threads = threads
    original_cls = ort.InferenceSession
    _original_session_cls = original_cls

    def _session(path_or_bytes, sess_options=None, providers=None, **kwargs):
        """Fill in options/providers only when the caller supplied none."""
        global _actual_providers
        opts = sess_options
        if opts is None and threads > 0:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        chosen = providers if providers is not None else list(usable)

        first = _actual_providers is None
        if first and verbose and "CUDAExecutionProvider" in chosen:
            # CUDA context + kernel setup costs ~40 s the first time, spread over
            # session construction and the first inference. Say so, or the run
            # looks hung. Amortised to ~0.5 % over 5,000 rows.
            print("[tse-eval] initialising DNSMOS on CUDA (~40 s, one time)…",
                  flush=True)
        session = original_cls(path_or_bytes, sess_options=opts,
                               providers=chosen, **kwargs)
        # Record what onnxruntime actually accepted, not what we asked for.
        got = tuple(session.get_providers())
        if _actual_providers is None:
            _actual_providers = got
            if verbose:
                print(f"[tse-eval] DNSMOS ONNX providers: {list(got)} "
                      f"(intra_op={threads})", flush=True)
        return session

    ort.InferenceSession = _session
    _configured = True


def actual_providers() -> Optional[Sequence[str]]:
    """Providers onnxruntime really used, or ``None`` before the first session."""
    return list(_actual_providers) if _actual_providers else None


def provenance() -> dict:
    """Settings + outcome, for the run's sidecar JSON."""
    return {
        "configured": _configured,
        "intra_op_threads": _intra_op_threads,
        "requested_providers": list(_requested_providers) if _requested_providers else None,
        "actual_providers": actual_providers(),
    }


def _reset_for_tests() -> None:
    """Restore the unwrapped session class and clear state, so a test can
    configure again without stacking wrappers."""
    global _configured, _requested_providers, _actual_providers
    global _intra_op_threads, _original_session_cls
    if _original_session_cls is not None:
        try:
            import onnxruntime as ort
            ort.InferenceSession = _original_session_cls
        except ImportError:
            pass
    _configured = False
    _requested_providers = _actual_providers = _intra_op_threads = None
    _original_session_cls = None
