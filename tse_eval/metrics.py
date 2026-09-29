"""
tse_eval.metrics — core TSE evaluation metrics.

Implemented (all reference-based unless noted):

    si_sdr                SI-SDR (scale-invariant SDR, dB)     est vs ref   @native
    si_sdri               SI-SDR improvement over the mixture  est/ref/mix  @native
    input_si_sdr          SI-SDR of the *mixture* vs ref       mix vs ref   @native
    input_si_sdr_pairwise same, without the 3-way length trim  mix vs ref   @native
    stoi                  STOI                                 est vs ref   @native
    estoi                 Extended STOI                        est vs ref   @native
    pesq                  PESQ (wideband)                      est vs ref   @16 kHz
    dnsmos_*              DNSMOS SIG/BAK/OVRL (+P.808)         est only     @16 kHz
    wer                   Word Error Rate (Whisper + jiwer)    est vs text  @16 kHz
    spk_sim               ECAPA speaker-embedding cosine       est vs ref   @16 kHz

Sample-rate protocol — declared once in :data:`METRIC_SR_POLICY` and matching the
sibling TSE projects so numbers are directly comparable (see ``llmtse/eval.py``
docstring: "SI-SDR/SI-SDRi/ESTOI = native 24k, PESQ = 16k"):

    metric                 working rate  reason
    ─────────────────────  ────────────  ──────────────────────────────────────────
    si_sdr / si_sdri       native        scale- and rate-agnostic definition
    input_si_sdr*          native        same family
    stoi / estoi           native        pystoi resamples to its own 10 kHz rate;
                                         passing native avoids a 24k->16k->10k hop
    pesq                   16 kHz        ITU-T P.862 defines 8/16 kHz only
    dnsmos_*               16 kHz        speechmos raises ValueError if sr != 16000
    wer                    16 kHz        Whisper feature extractor rejects other rates
    spk_sim                16 kHz        ECAPA is 16 kHz and does NOT resample —
                                         a 24 kHz input silently yields wrong
                                         embeddings (requirements/h200/requirements_h200.txt [B-9])

All resampling goes through :func:`tse_eval.audio.resample_np` — the single
conversion point.  ``mix`` is never resampled: no 16 kHz metric consumes it.

SI-SDR is implemented natively and is **numerically equivalent to asteroid**
(``asteroid.metrics.get_metrics`` -> ``pb_bss_eval``) to ~1e-14; see
:func:`si_sdr` and ``tests/test_backends.py``.  Only one deliberate difference:
a (near-)silent reference returns ``nan`` here so it drops out of means, whereas
asteroid returns a finite floor value.

Not implemented on purpose — SDR / SIR / SAR (``mir_eval.bss_eval_sources``):
~1 s per utterance (~91 min for 5,000 rows) and, being a blind-source-separation
measure, SIR degenerates to ``inf`` without an explicit interference signal, so
it is not meaningful for target-speaker extraction.

Every metric degrades gracefully: on any failure the value is ``float('nan')``
and the key is still present, so a whole CSV never aborts because of one bad row.
"""

from __future__ import annotations

import sys
from typing import Dict

import numpy as np

from .audio import align_pair, align_triple, resample_np

# The rate every 16 kHz-only metric runs at (PESQ, DNSMOS, WER, Speaker Sim).
PERCEPTUAL_SR = 16_000

# ``pb_bss_eval``'s epsilon. Sharing the exact value is what makes the native
# SI-SDR bit-comparable with asteroid — do not change it independently.
_EPS = 1e-8

# The DNSMOS output columns, as one set — used by the resample guard in
# ``compute_row_metrics`` and by the onnxruntime setup trigger in
# ``evaluate.evaluate_csv``. Kept in sync with the keys returned by
# :func:`dnsmos`, including the ``_clipped`` companions: asking for only those
# still needs an ONNX session.
_DNSMOS_COLUMNS = frozenset(
    {f"dnsmos_{n}{sfx}" for n in ("sig", "bak", "ovrl", "p808")
     for sfx in ("", "_clipped")})

# Metric column order used by the output CSV.
METRIC_COLUMNS = [
    "si_sdr",
    "si_sdri",
    "input_si_sdr",
    "input_si_sdr_pairwise",
    "stoi",
    "estoi",
    "pesq",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
    "dnsmos_p808",
    "wer",
    "spk_sim",
    # ── Companion columns (see COMPANION_METRICS) ────────────────────────────
    # Emitted automatically alongside their primary metric, so `--metrics` never
    # has to name them. They are listed here so ``summarize`` reports them too:
    # showing both variants is what demonstrates that a conclusion does not rest
    # on the normalisation choice.
    "dnsmos_sig_clipped",
    "dnsmos_bak_clipped",
    "dnsmos_ovrl_clipped",
    "dnsmos_p808_clipped",
    "wer_raw",
]

# primary metric -> companions it drags along. Same idea as "asking for si_sdri
# implies its two terms": the pair is only meaningful together.
COMPANION_METRICS: Dict[str, "frozenset[str]"] = {
    "dnsmos_sig":  frozenset({"dnsmos_sig_clipped"}),
    "dnsmos_bak":  frozenset({"dnsmos_bak_clipped"}),
    "dnsmos_ovrl": frozenset({"dnsmos_ovrl_clipped"}),
    "dnsmos_p808": frozenset({"dnsmos_p808_clipped"}),
    "wer":         frozenset({"wer_raw"}),
}


def expand_companions(metrics: "set[str]") -> "set[str]":
    """Add every companion of every requested metric. Idempotent.

    The companions cost nothing extra to produce — the same Whisper pass and the
    same ONNX session yield both variants — so requesting one and silently
    dropping the other would only hide information.
    """
    out = set(metrics)
    for primary, extra in COMPANION_METRICS.items():
        if primary in out:
            out |= extra
    return out

# ── Sample-rate policy, in one place (see module docstring for the reasons) ──
# "native" = compute at the caller's working rate; an int = resample to that rate.
NATIVE = "native"
METRIC_SR_POLICY: Dict[str, "str | int"] = {
    "si_sdr": NATIVE,
    "si_sdri": NATIVE,
    "input_si_sdr": NATIVE,
    "input_si_sdr_pairwise": NATIVE,
    "stoi": NATIVE,
    "estoi": NATIVE,
    "pesq": PERCEPTUAL_SR,
    "dnsmos_sig": PERCEPTUAL_SR,
    "dnsmos_bak": PERCEPTUAL_SR,
    "dnsmos_ovrl": PERCEPTUAL_SR,
    "dnsmos_p808": PERCEPTUAL_SR,
    "wer": PERCEPTUAL_SR,
    "spk_sim": PERCEPTUAL_SR,
    "dnsmos_sig_clipped": PERCEPTUAL_SR,
    "dnsmos_bak_clipped": PERCEPTUAL_SR,
    "dnsmos_ovrl_clipped": PERCEPTUAL_SR,
    "dnsmos_p808_clipped": PERCEPTUAL_SR,
    "wer_raw": PERCEPTUAL_SR,
}

# Metrics needing a model download; excluded from the default metric set so a
# plain run stays light. Opt in with ``--metrics``. ``wer_raw`` is here too: it is
# a companion of ``wer`` and needs the very same Whisper pass.
MODEL_BACKED_METRICS = frozenset({"wer", "wer_raw", "spk_sim"})

# Default metric set: everything except the model-backed extras.
DEFAULT_METRICS = [m for m in METRIC_COLUMNS if m not in MODEL_BACKED_METRICS]

# Extra per-row columns that are written alongside the metrics but are not
# metrics themselves (WER needs the counts for corpus-level micro aggregation,
# and ``wer_hyp`` is what makes a WER recomputable without re-running Whisper).
WER_SUPPORT_COLUMNS = ["wer_edits", "wer_words",
                       "wer_raw_edits", "wer_raw_words", "wer_hyp"]


# ─────────────────────────────────────────────────────────────────────────
# SI-SDR family (native, sample-rate agnostic)
# ─────────────────────────────────────────────────────────────────────────
def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = _EPS) -> float:
    """Scale-Invariant SDR in dB between an estimate and a reference.

    Zero-mean both signals, project ``est`` onto ``ref``, then take
    ``10·log10(||s_target||² / ||e_noise||²)``.  Scale-invariant, so a gain
    applied to ``est`` leaves the value unchanged.

    ★ Epsilon placement is copied exactly from ``pb_bss_eval`` (the library
    behind ``asteroid.metrics.get_metrics``): ``eps`` goes into the reference
    energy, into the *denominator* of the ratio, and onto the ratio before the
    log — but **never into the numerator**.  That placement is what produces
    asteroid's ``10·log10(eps) = -80 dB`` floor for degenerate inputs.  An
    earlier version added ``eps`` to both sides, which returned a misleading
    ``0.0 dB`` for a silent/DC-only/near-zero estimate and inflated the mean.
    Verified equal to asteroid within ~1e-14 (``tests/test_backends.py``).

    Args:
        est: Estimated (extracted) waveform, 1-D ``[N]``.
        ref: Reference (ground-truth target) waveform, 1-D ``[N]``.
        eps: Numerical floor. Must match ``pb_bss_eval.EPS`` for equivalence.

    Returns:
        SI-SDR in dB (float). ``nan`` if the reference is (near-)silent — the
        one deliberate divergence from asteroid, so such rows drop out of means
        instead of dragging them toward the floor.
    """
    est, ref = align_pair(np.asarray(est, np.float64), np.asarray(ref, np.float64))
    est = est - est.mean()
    ref = ref - ref.mean()
    if float(np.dot(ref, ref)) < eps:                      # silent reference
        return float("nan")
    ref_energy = float(np.dot(ref, ref)) + eps
    alpha = float(np.dot(est, ref)) / ref_energy
    s_target = alpha * ref
    e_noise = est - s_target
    ratio = float(np.dot(s_target, s_target)) / (float(np.dot(e_noise, e_noise)) + eps)
    return float(10.0 * np.log10(ratio + eps))


def si_sdr_family(est: np.ndarray, ref: np.ndarray, mix: np.ndarray) -> Dict[str, float]:
    """The whole SI-SDR family for one triple, including the mixture baseline.

    ``input_si_sdr`` is normally discarded by ``si_sdri``; exposing it lets a
    reviewer check the identity ``si_sdri == si_sdr − input_si_sdr`` directly.

    ``input_si_sdr`` here is computed on exactly the ``mix``/``ref`` passed in.
    Callers hand over 3-way-trimmed signals (matching the sibling projects), so
    the value depends on the estimate's length.  The estimate-independent
    counterpart, ``input_si_sdr_pairwise``, is produced by
    :func:`compute_row_metrics` from the *untrimmed* mixture and reference.

    ★ The three signals are trimmed to one common window here, not left to each
    metric's own pairwise alignment. Both terms must be measured over the *same*
    samples or ``si_sdri`` compares two different windows — and the two backends
    would disagree whenever the mixture happened to be the shortest of the three.
    :func:`compute_row_metrics` already trims before calling, so this is a no-op
    on that path and a safety net for direct callers.

    Args:
        est: Estimate, ref: reference, mix: mixture — all 1-D at the same rate.

    Returns:
        Dict with ``si_sdr``, ``si_sdri``, ``input_si_sdr``.
    """
    est, ref, mix = align_triple(est, ref, mix)
    out = si_sdr(est, ref)
    inp = si_sdr(mix, ref)
    improvement = float("nan") if (np.isnan(out) or np.isnan(inp)) else out - inp
    return {"si_sdr": out, "si_sdri": improvement, "input_si_sdr": inp}


def si_sdri(est: np.ndarray, ref: np.ndarray, mix: np.ndarray) -> float:
    """SI-SDR improvement: ``SI-SDR(est, ref) − SI-SDR(mix, ref)`` (dB).

    Kept for backward compatibility; :func:`si_sdr_family` also returns the
    ``input_si_sdr`` term that this function throws away.
    """
    return si_sdr_family(est, ref, mix)["si_sdri"]


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


def stoi_metric(ref: np.ndarray, est: np.ndarray, sr: int = PERCEPTUAL_SR,
                extended: bool = False) -> float:
    """STOI (``extended=False``) or ESTOI (``extended=True``) at ``sr``.

    Args:
        ref:      Reference waveform, 1-D.
        est:      Estimated waveform, 1-D.
        sr:       Sample rate of *both* inputs (Hz). Any rate is valid — pystoi
                  resamples internally to its own 10 kHz working rate — so pass
                  the native rate rather than pre-resampling to 16 kHz. That
                  avoids a 24k->16k->10k double resample and matches the sibling
                  protocol (``styletse/eval.py``, ``tpex/eval.py``).
        extended: ``True`` for ESTOI, ``False`` for plain STOI.

    Returns:
        The score, or ``float('nan')`` on any failure.
    """
    try:
        from pystoi import stoi as _stoi
        ref, est = align_pair(ref, est)
        return float(_stoi(ref, est, sr, extended=extended))
    except Exception:
        return float("nan")


_dnsmos_warned = False


DNSMOS_TARGET_DBOV = -26.0      # ITU-T P.56 convention
_DNSMOS_PEAK_CEILING = 0.99     # speechmos rejects |x| > 1 outright


def dnsmos_normalize(x: np.ndarray, dbov: float = DNSMOS_TARGET_DBOV) -> np.ndarray:
    """Bring a waveform to ``dbov`` RMS, scaling down further if it would clip.

    ★ This is a *scale*, never a clip — that distinction is the whole point.
    ``speechmos`` refuses input outside [-1, 1] (``ValueError: np.ndarray values
    must be between -1 and 1.``). An earlier version of this module satisfied
    that with ``np.clip``, which meant DNSMOS scored a **clipped** waveform
    whenever a model's output ran hot. Measured on LLM-TSE predictions, whose
    peaks exceed 1.0 on 100 % of rows (median 4.94, max 17.25): the median row
    had 5.5 % of its samples clipped (worst 40.3 %), dragging ``dnsmos_ovrl``
    from 2.91 to 2.55 — a level artefact reported as if it were audio quality.

    RMS normalisation alone is not enough: crest factor here runs to 31.7 dB, so
    ~2 % of files still peak above 1.0 after hitting -26 dBov. Those get one
    further proportional reduction to ``_DNSMOS_PEAK_CEILING``. They end up a
    little quieter than -26 dBov, which costs far less than clipping would.

    DNSMOS is level-sensitive because it is a no-reference regressor trained on
    conventionally-levelled speech; feeding it a signal 24 dB hot is out of
    distribution. Every system in a comparison must therefore use the *same*
    normalisation — see ``configs/config.yaml``.
    """
    a = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(a * a)))
    if not np.isfinite(rms) or rms <= 0.0:
        return a                                    # silent/degenerate: leave alone
    a = a * (10.0 ** (dbov / 20.0) / rms)
    peak = float(np.max(np.abs(a)))
    if peak > _DNSMOS_PEAK_CEILING:
        a = a * (_DNSMOS_PEAK_CEILING / peak)
    return a


def dnsmos(est16k: np.ndarray) -> Dict[str, float]:
    """DNSMOS (P.835) on the estimate only — no reference needed.

    Returns SIG / BAK / OVRL and the P.808 MOS **twice**:

    * ``dnsmos_*``          — on a level-normalised signal (:func:`dnsmos_normalize`).
      These are the values to report.
    * ``dnsmos_*_clipped``  — on the signal hard-clipped to [-1, 1], reproducing
      this project's pre-fix behaviour so older numbers stay reconcilable.

    Why ``_clipped`` and not ``_raw``: speechmos rejects anything outside
    [-1, 1], so a genuinely un-normalised DNSMOS value cannot be computed at all.
    Calling the clipped figure "raw" would suggest it is the untouched
    measurement when it is in fact the *most* altered one.

    All ``nan`` if speechmos is unavailable. That is the most common cause of an
    all-NaN DNSMOS column: ``speechmos`` declares no dependencies of its own yet
    imports ``librosa`` and ``onnxruntime`` at module level. Because this
    function swallows exceptions by design, that used to look like "DNSMOS just
    returns NaN"; an ``ImportError`` now prints one warning so the cause shows.
    """
    global _dnsmos_warned
    src = {"sig": "sig_mos", "bak": "bak_mos", "ovrl": "ovrl_mos", "p808": "p808_mos"}
    blank = {f"dnsmos_{n}{sfx}": float("nan")
             for n in src for sfx in ("", "_clipped")}
    try:
        from speechmos import dnsmos as _dnsmos

        def score(audio: np.ndarray, suffix: str) -> Dict[str, float]:
            res = _dnsmos.run(np.asarray(audio, dtype=np.float32), sr=PERCEPTUAL_SR)
            return {f"dnsmos_{n}{suffix}": float(res[k]) for n, k in src.items()}

        a = np.asarray(est16k, dtype=np.float64)
        out = score(dnsmos_normalize(a), "")
        out.update(score(np.clip(a, -1.0, 1.0), "_clipped"))
        return out
    except ImportError as exc:
        if not _dnsmos_warned:
            _dnsmos_warned = True
            print(f"[tse-eval] ⚠ DNSMOS unavailable → every dnsmos_* value will "
                  f"be NaN. Missing dependency: {exc}. `speechmos` needs "
                  f"`librosa` and an onnxruntime build, neither of which it "
                  f"declares. Fix: pip install librosa==0.11.0 "
                  f"'onnxruntime-gpu>=1.19.2,<1.21'", file=sys.stderr, flush=True)
        return blank
    except Exception:
        return blank


# ─────────────────────────────────────────────────────────────────────────
# Model-backed metrics (16 kHz) — opt-in, loaded once and reused
# ─────────────────────────────────────────────────────────────────────────
# Model ids are overridable from configs/config.yaml via ``configure_models``.
WER_MODEL_ID = "openai/whisper-large-v3"
WER_LANGUAGE = "en"
SPK_SIM_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

_asr_cache: Dict[str, object] = {}          # lazily-built Whisper singleton
_spk_cache: Dict[str, object] = {}          # lazily-built ECAPA singleton


def configure_models(wer_model_id: "str | None" = None,
                     wer_language: "str | None" = None,
                     spk_sim_model_id: "str | None" = None) -> None:
    """Override the model ids/language before evaluation (from config).

    Changing an id clears the corresponding cache so the next call reloads.
    """
    global WER_MODEL_ID, WER_LANGUAGE, SPK_SIM_MODEL_ID
    if wer_model_id is not None and wer_model_id != WER_MODEL_ID:
        WER_MODEL_ID = wer_model_id
        _asr_cache.clear()
    if wer_language is not None:
        WER_LANGUAGE = wer_language
    if spk_sim_model_id is not None and spk_sim_model_id != SPK_SIM_MODEL_ID:
        SPK_SIM_MODEL_ID = spk_sim_model_id
        _spk_cache.clear()


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                       # noqa: BLE001
        return "cpu"


def _ort_session_first() -> None:
    """Make sure the DNSMOS ONNX session exists before torch loads cuDNN.

    Both model-backed metrics run CUDA convolutions, which pull in torch's
    bundled cuDNN; if that happens first, onnxruntime's CUDA EP fails to load and
    DNSMOS silently falls back to the CPU (``tse_eval.ort_setup`` docstring).
    ``compute_row_metrics`` already scores DNSMOS before these two, but a caller
    using the metrics directly has no such guarantee — so pin it here as well.
    No-op unless onnxruntime was configured, i.e. unless DNSMOS is in play.
    """
    from . import ort_setup
    ort_setup.prime_dnsmos_session(verbose=False)


def _get_asr():
    """Load Whisper once (bf16 on GPU, fp32 on CPU) and cache it.

    The cache also carries ``normalizer`` — the text normaliser used for the
    reported WER. It comes from the tokenizer we already load, so it costs no
    extra dependency and no extra download: the model's own ``normalizer.json``
    (1 740 spelling variants) ships in the checkpoint. Using Whisper's own
    normaliser rather than a hand-rolled lowercase/strip-punctuation pass is what
    makes the number comparable with published ASR results.
    """
    if "model" not in _asr_cache:
        _ort_session_first()
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        device = _device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WER_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(WER_MODEL_ID)
        # ``normalize`` is the English normaliser; ``basic_normalize`` is the
        # language-agnostic fallback for non-English checkpoints.
        tk = processor.tokenizer
        normalizer = getattr(tk, "normalize", None) or getattr(
            tk, "basic_normalize", None) or (lambda s: str(s).strip().lower())
        _asr_cache.update(model=model, processor=processor, normalizer=normalizer,
                          device=device, dtype=dtype)
    return _asr_cache


def _get_spk():
    """Load ECAPA once and cache it.

    ★ The device must be passed via ``run_opts``. Calling ``.to(device)`` later
    moves the weights but not the input — ``encode_batch`` sends the waveform to
    the device recorded at construction time, so a mismatched pair raises
    "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)".
    """
    if "model" not in _spk_cache:
        _ort_session_first()
        from speechbrain.inference import EncoderClassifier
        device = _device()
        _spk_cache.update(
            model=EncoderClassifier.from_hparams(
                source=SPK_SIM_MODEL_ID, run_opts={"device": device}),
            device=device)
    return _spk_cache


def wer(est16k: np.ndarray, reference_text: "str | None") -> Dict[str, float]:
    """Word Error Rate of a Whisper transcription of ``est16k`` vs ``reference_text``.

    Input must already be at 16 kHz (Whisper's feature extractor rejects any
    other rate outright).

    Scored **twice** against the same transcription:

    * ``wer`` / ``wer_edits`` / ``wer_words``
      after Whisper's own English text normaliser (case, punctuation, numbers,
      contractions, 1 740 spelling variants). These are the values to report.
    * ``wer_raw`` / ``wer_raw_edits`` / ``wer_raw_words``
      on the strings as-is, reproducing this project's pre-fix behaviour — which
      is also what ``llmtse/eval.py:43-45`` does, so the baseline's published
      numbers stay comparable.

    Without normalisation, Whisper emitting a row without punctuation costs edits
    even when every word is right::

        REF: Although plump and healthy, Josiana was, we repeat, a perfect prude.
        HYP: although plump and healthy josianna was we repeat a perfect prude
        -> 6 edits / 11 words = 0.545, on a row whose SI-SDR is 50.75 dB

    Measured over 5 000 LLM-TSE rows, corpus micro-WER falls 0.5932 -> 0.5240.

    ``wer_hyp`` always carries Whisper's untouched output, which is what makes
    any of this recomputable later without re-running the model.

    The counts exist because the paper table reports **corpus micro-WER**
    (``sum(edits) / sum(words)``); averaging per-row WER over-weights short
    utterances and yields a different figure, so the summary aggregates counts.

    All ``nan`` (and ``wer_hyp=""``) when there is no reference text or on any
    failure, so such rows drop out of the micro sum.
    """
    blank: Dict[str, float] = {k: float("nan") for k in
                               ("wer", "wer_edits", "wer_words",
                                "wer_raw", "wer_raw_edits", "wer_raw_words")}
    blank["wer_hyp"] = ""
    if not reference_text or not str(reference_text).strip():
        return blank
    try:
        import torch
        import jiwer
        bundle = _get_asr()
        feats = bundle["processor"](
            np.asarray(est16k, dtype=np.float32), sampling_rate=PERCEPTUAL_SR,
            return_tensors="pt").input_features.to(bundle["device"]).to(bundle["dtype"])
        with torch.no_grad():
            ids = bundle["model"].generate(feats, language=WER_LANGUAGE)
        hyp = bundle["processor"].batch_decode(ids, skip_special_tokens=True)[0]
        ref = str(reference_text).strip()

        def score(r: str, h: str, prefix: str) -> Dict[str, float]:
            # An empty hypothesis is legitimate (silence in, nothing out) and
            # jiwer handles it; an empty *reference* is not scorable.
            if not r.strip():
                return {prefix: float("nan"), f"{prefix}_edits": float("nan"),
                        f"{prefix}_words": float("nan")}
            o = jiwer.process_words(r, h)
            return {prefix: float(o.wer),
                    f"{prefix}_edits": float(o.substitutions + o.deletions + o.insertions),
                    f"{prefix}_words": float(o.substitutions + o.deletions + o.hits)}

        norm = bundle["normalizer"]
        out = score(norm(ref), norm(hyp), "wer")
        out.update(score(ref, hyp.strip(), "wer_raw"))
        out["wer_hyp"] = hyp.strip()
        return out
    except Exception:                                       # noqa: BLE001
        return blank


def spk_sim(est16k: np.ndarray, ref16k: np.ndarray) -> float:
    """Cosine similarity of ECAPA speaker embeddings for estimate vs reference.

    Both inputs must already be at 16 kHz. ECAPA is a 16 kHz model and
    ``encode_batch`` does **not** resample, so feeding 24 kHz audio silently
    produces wrong embeddings (requirements/h200/requirements_h200.txt [B-9]) — hence the rate is
    the caller's responsibility and is enforced by the sample-rate policy.

    Returns the cosine in [-1, 1], or ``nan`` on failure / too-short input.
    """
    try:
        import torch
        if est16k is None or ref16k is None:
            return float("nan")
        # ECAPA needs a meaningful span; below ~0.5 s the embedding is unstable.
        if min(len(est16k), len(ref16k)) < PERCEPTUAL_SR // 2:
            return float("nan")
        model = _get_spk()["model"]
        with torch.no_grad():
            emb = [
                model.encode_batch(
                    torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
                ).squeeze().cpu().numpy().ravel()
                for x in (est16k, ref16k)
            ]
        a, b = emb
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return float("nan")
        return float(np.dot(a, b) / denom)
    except Exception:                                       # noqa: BLE001
        return float("nan")


def micro_wer(edits: np.ndarray, words: np.ndarray) -> float:
    """Corpus-level (micro) WER: ``sum(edits) / sum(words)``, NaN rows excluded.

    This is the figure that belongs in the paper table. Rows without a reference
    transcript carry ``nan`` counts and are dropped.
    """
    e = np.asarray(edits, dtype=float)
    w = np.asarray(words, dtype=float)
    keep = ~(np.isnan(e) | np.isnan(w))
    total_words = float(w[keep].sum())
    if not keep.any() or total_words <= 0:
        return float("nan")
    return float(e[keep].sum() / total_words)


# ─────────────────────────────────────────────────────────────────────────
# Row-level aggregation
# ─────────────────────────────────────────────────────────────────────────
def compute_row_metrics(
    est: np.ndarray,
    ref: np.ndarray,
    mix: np.ndarray,
    sr: int,
    metrics: "set[str] | None" = None,
    reference_text: "str | None" = None,
    si_sdr_backend: str = "native",
) -> Dict[str, float]:
    """Compute all core metrics for a single (est, ref, mix) triple.

    Args:
        est:     Estimated waveform at ``sr``, 1-D.
        ref:     Reference waveform at ``sr``, 1-D.
        mix:     Mixture waveform at ``sr``, 1-D.
        sr:      Working sample rate of the three inputs (Hz).
        metrics: Optional subset of :data:`METRIC_COLUMNS` to compute; ``None``
                 uses :data:`DEFAULT_METRICS` (everything except the
                 model-backed extras). Skipped metrics are present as ``nan``.
        reference_text: Ground-truth transcript for ``wer`` (the manifest's
                 ``target_sentence``). ``None`` leaves WER as ``nan``.
        si_sdr_backend: ``"native"`` (default) or ``"asteroid"``; see
                 :mod:`tse_eval.backends`. The two agree to ~1e-13.

    Returns:
        Dict keyed by :data:`METRIC_COLUMNS`, plus :data:`WER_SUPPORT_COLUMNS`
        when ``wer`` is requested.
    """
    want = expand_companions(
        set(DEFAULT_METRICS) if metrics is None else set(metrics))
    out: Dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}

    # ★ 3-way length alignment FIRST, so every metric below sees one common
    # window — the sibling-project convention (see audio.align_triple).
    est_t, ref_t, mix_t = align_triple(est, ref, mix)

    # ── @native: SI-SDR family — scale- and sample-rate agnostic.
    # Asking for si_sdri implies its two terms: a reviewer must be able to check
    # si_sdri == si_sdr - input_si_sdr straight from the CSV, so they come along.
    si_keys = {"si_sdr", "si_sdri", "input_si_sdr"}
    emit = set(want & si_keys)
    if "si_sdri" in want:
        emit |= si_keys
    if emit or "input_si_sdr_pairwise" in want or "si_sdri" in want:
        from .backends import si_sdr_family as _family, si_sdr_value as _value
    if emit:
        fam = _family(est_t, ref_t, mix_t, backend=si_sdr_backend)
        for key in emit:
            out[key] = fam[key]

    # Estimate-independent baseline: untrimmed mix/ref, pairwise aligned only.
    if "input_si_sdr_pairwise" in want or "si_sdri" in want:
        out["input_si_sdr_pairwise"] = _value(mix, ref, backend=si_sdr_backend)

    # ── @native: STOI / ESTOI — pystoi resamples to 10 kHz internally, so the
    # native rate is passed straight through (no 24k->16k->10k double resample).
    if "stoi" in want:
        out["stoi"] = stoi_metric(ref_t, est_t, sr, extended=False)
    if "estoi" in want:
        out["estoi"] = stoi_metric(ref_t, est_t, sr, extended=True)

    # ── @16 kHz: PESQ, DNSMOS, WER, Speaker Similarity.
    # Resample est/ref once and reuse. ``mix`` is deliberately never resampled:
    # no 16 kHz metric consumes it.
    want_dnsmos = bool(want & _DNSMOS_COLUMNS)
    want_wer = bool(want & {"wer", "wer_raw"})
    needs_est16 = want_dnsmos or want_wer or bool(want & {"pesq", "spk_sim"})
    needs_ref16 = bool(want & {"pesq", "spk_sim"})
    if needs_est16:
        est16 = resample_np(est_t, sr, PERCEPTUAL_SR) if sr != PERCEPTUAL_SR else est_t
        ref16 = None
        if needs_ref16:
            ref16 = resample_np(ref_t, sr, PERCEPTUAL_SR) if sr != PERCEPTUAL_SR else ref_t
        if "pesq" in want:
            out["pesq"] = pesq_wb(ref16, est16)
        if want_dnsmos:
            out.update(dnsmos(est16))
        if "spk_sim" in want:
            out["spk_sim"] = spk_sim(est16, ref16)
        if want_wer:
            out.update(wer(est16, reference_text))

    return out
