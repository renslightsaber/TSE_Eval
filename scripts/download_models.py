#!/usr/bin/env python
"""scripts/download_models.py — TSE_Eval 확장 지표용 사전학습 모델 1회 다운로드 + 검증.

다운로드 대상 (2개):
  1. Speaker Similarity : speechbrain/spkrec-ecapa-voxceleb   ECAPA-TDNN, 16 kHz, 192-dim, ~89 MB
  2. WER                : openai/whisper-large-v3             transformers, 16 kHz, ~3.1 GB

  ※ DNSMOS 는 `speechmos` 휠에 ONNX 가 번들되어 있어 다운로드가 필요 없습니다.
  ※ whisper-large-v3 은 llmtse/eval.py:126-128 과 동일 모델 → WER 숫자를 바로 비교할 수 있습니다.

★ 두 모델 모두 16 kHz 전용입니다. 우리 데이터(PORTE-v3)는 24 kHz 이므로 지표 계산 시
  16 kHz 로 다운샘플해야 합니다. --verify 단계에서 이를 실제로 보여줍니다:
    - Whisper : 24 kHz 를 넘기면 ValueError 로 거부
    - ECAPA   : 24 kHz 를 넘기면 ★ 에러 없이 다른 임베딩 ★ (조용한 실패 — 가장 위험)

실행:
    export HF_HOME=/home/work/my-checkpoints/hf_cache      # ★ 영속 vFolder (미설정 시 자동 사용)
    python scripts/download_models.py                      # 둘 다 + 검증
    python scripts/download_models.py --only spk           # ECAPA 만
    python scripts/download_models.py --only wer           # Whisper 만
    python scripts/download_models.py --no-verify          # 다운로드만
    python scripts/download_models.py --device cpu         # 검증을 CPU 로

재실행은 캐시 히트로 즉시 끝납니다(idempotent). 성공 시 마지막 줄이 ALL CHECKS PASSED,
실패 시 exit code 1.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

# speechbrain 1.0.3 + torch 2.5.1 조합이 내는 FutureWarning 들(torch.cuda.amp.custom_fwd,
# torch.load(weights_only=False))을 숨깁니다 — 동작에 영향이 없고, 진짜 검증 결과를 가립니다.
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────
ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
WHISPER_ID = "openai/whisper-large-v3"

# 두 모델의 학습/추론 샘플레이트. 이 값 외의 입력은 (Whisper) 거부되거나
# (ECAPA) 조용히 틀린 특징을 만듭니다.
MODEL_SR = 16_000
# PORTE-v3 원본 샘플레이트 — 조용한 실패 실증용.
DATA_SR = 24_000

# 영속 저장소(vFolder). /home/work 루트는 세션 종료 시 삭제되므로 반드시 이 아래로.
DEFAULT_HF_HOME = "/home/work/my-checkpoints/hf_cache"
VFOLDER_PREFIXES = ("/home/work/my-checkpoints", "/home/work/my-code",
                    "/home/work/my-datasets", "/home/work/my-outputs")


# ─────────────────────────────────────────────────────────────────────────
# 출력 헬퍼 (추가 의존성 0 — colorama 없이)
# ─────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}", flush=True)


def kv(key: str, value: object) -> None:
    print(f"  {key:<22}: {value}", flush=True)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}", file=sys.stderr, flush=True)


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr, flush=True)


def du_mb(path: Path) -> float:
    """디렉토리의 실제 디스크 사용량(MB). 존재하지 않으면 0.

    ★ 심볼릭 링크는 건너뜁니다. HuggingFace 캐시는 실제 바이트를 `blobs/` 에 한 번만
      두고 `snapshots/` 에는 링크만 두기 때문에, 링크를 따라가면 같은 파일을 두 번
      세어 사용량이 2배로 보입니다(whisper 3 GB → 6.4 GB 로 보이던 원인).
    """
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_symlink() or not p.is_file():
                continue
            total += p.stat().st_size
        except OSError:                                   # 깨진 링크 / 권한 등
            continue
    return total / 1e6


# ─────────────────────────────────────────────────────────────────────────
# 환경 준비
# ─────────────────────────────────────────────────────────────────────────
def resolve_hf_home() -> Path:
    """HF_HOME 을 결정하고 환경변수에 반영한다.

    미설정이면 영속 vFolder 기본값을 쓰고, 설정돼 있으나 vFolder 밖이면 경고한다.
    ★ transformers / huggingface_hub 를 import 하기 전에 호출해야 한다
      (import 시점에 캐시 경로가 고정되기 때문).
    """
    env = os.environ.get("HF_HOME")
    if not env:
        os.environ["HF_HOME"] = DEFAULT_HF_HOME
        warn(f"HF_HOME 이 설정되지 않아 기본값을 사용합니다: {DEFAULT_HF_HOME}")
        warn("다음부터는 셸에서 미리 지정하세요:  "
             f"export HF_HOME={DEFAULT_HF_HOME}")
    hf_home = Path(os.environ["HF_HOME"])

    if not str(hf_home).startswith(VFOLDER_PREFIXES):
        warn("=" * 66)
        warn(f"HF_HOME 이 영속 저장소(vFolder) 밖입니다: {hf_home}")
        warn("/home/work 루트는 세션 종료 시 삭제됩니다 → whisper 3 GB 를 매 세션 재다운로드")
        warn(f"권장:  export HF_HOME={DEFAULT_HF_HOME}")
        warn("=" * 66)

    hf_home.mkdir(parents=True, exist_ok=True)
    return hf_home


def print_environment(hf_home: Path, device: str) -> None:
    section("환경")
    kv("python", sys.version.split()[0])
    kv("HF_HOME", hf_home)
    usage = shutil.disk_usage(hf_home)
    kv("HF_HOME 여유 공간", f"{usage.free / 1e9:.1f} GB / {usage.total / 1e9:.1f} GB")
    kv("HF_HOME 현재 사용량", f"{du_mb(hf_home):.0f} MB")
    kv("검증 device", device)
    try:
        import torch
        kv("torch", f"{torch.__version__} (cuda {torch.version.cuda})")
        if torch.cuda.is_available():
            kv("gpu", f"{torch.cuda.get_device_name(0)} "
                      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    except ImportError:
        fail("torch 가 없습니다. requirements/h200/requirements_h200.txt 를 먼저 설치하세요.")
        raise


def sine_speechlike(sr: int, seconds: float = 3.0):
    """검증용 합성 신호 — 순음이 아니라 하모닉 + 미세 노이즈 (무음/순음은 모델이 싫어함).

    같은 파형을 두 SR 로 만들 수 있도록 시간축 기준으로 생성한다.
    """
    import numpy as np
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    wav = np.zeros_like(t)
    for k, amp in enumerate([0.5, 0.25, 0.12, 0.06], start=1):
        wav += amp * np.sin(2 * np.pi * 140.0 * k * t)
    wav += 0.01 * np.random.default_rng(0).standard_normal(t.shape).astype(np.float32)
    # 발화처럼 보이도록 5 Hz 진폭 변조
    wav *= (0.6 + 0.4 * np.sin(2 * np.pi * 5.0 * t))
    return (wav / np.abs(wav).max() * 0.5).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# 1) Speaker Similarity — ECAPA-TDNN
# ─────────────────────────────────────────────────────────────────────────
def download_ecapa(hf_home: Path, device: str):
    """ECAPA-TDNN 을 내려받아 EncoderClassifier 를 반환한다.

    ★ device 는 `run_opts` 로 넘겨야 합니다. 나중에 `model.to("cuda")` 를 부르면
      가중치만 옮겨지고, speechbrain 의 `encode_batch()` 는 입력을 자기가 기억하는
      `self.device`(기본 "cpu")로 되돌려 보내기 때문에
      "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
       should be the same" 로 죽습니다. (실측 확인)
    """
    section(f"1/2  Speaker Similarity — {ECAPA_ID}")
    # speechbrain 1.x: speechbrain.pretrained → speechbrain.inference 로 이전됨
    from speechbrain.inference import EncoderClassifier

    savedir = hf_home / "speechbrain" / "spkrec-ecapa-voxceleb"
    kv("savedir", savedir)
    kv("device", device)
    before = du_mb(savedir)

    t0 = time.time()
    model = EncoderClassifier.from_hparams(
        source=ECAPA_ID, savedir=str(savedir), run_opts={"device": device})
    elapsed = time.time() - t0

    after = du_mb(savedir)
    kv("소요 시간", f"{elapsed:.1f} s")
    kv("디스크", f"{after:.0f} MB  (이번에 {max(0.0, after - before):.0f} MB 추가)")
    if elapsed < 3.0 and before > 0:
        ok("캐시 히트 — 재다운로드 없음")
    ok("ECAPA-TDNN 로드 완료")
    return model


def verify_ecapa(model, device: str) -> bool:
    """16 kHz 임베딩 정상 동작 + 24 kHz 조용한 실패를 실증한다."""
    import numpy as np
    import torch

    print("\n  ── 검증: ECAPA ──")
    passed = True

    def embed(wav_np):
        # 입력은 CPU 텐서로 넘겨도 됩니다 — encode_batch 가 model 의 device 로 옮깁니다
        # (그 device 는 from_hparams 의 run_opts 로 지정됨).
        t = torch.from_numpy(wav_np).unsqueeze(0)          # [1, T]
        with torch.no_grad():
            return model.encode_batch(t).squeeze(1).cpu().numpy()   # [1, 192]

    # (a) 16 kHz — 정상 경로
    wav16 = sine_speechlike(MODEL_SR)
    emb_a = embed(wav16)
    if emb_a.shape == (1, 192):
        ok(f"16 kHz 임베딩 shape {emb_a.shape}")
    else:
        fail(f"임베딩 shape 가 (1, 192) 가 아닙니다: {emb_a.shape}")
        passed = False

    # (b) 자기 자신과의 코사인 유사도 == 1
    self_sim = float(np.dot(emb_a[0], emb_a[0]) /
                     (np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_a[0])))
    if abs(self_sim - 1.0) < 1e-4:
        ok(f"self cosine similarity = {self_sim:.6f}")
    else:
        fail(f"self cosine similarity 가 1.0 이 아닙니다: {self_sim}")
        passed = False

    # (c) ★ 조용한 실패 실증: 같은 소리의 24 kHz 판을 그대로 넣으면
    #     에러 없이 다른 임베딩이 나온다 → 반드시 16 kHz 로 내려서 써야 한다.
    wav24 = sine_speechlike(DATA_SR)
    emb_c = embed(wav24)
    cross = float(np.dot(emb_a[0], emb_c[0]) /
                  (np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_c[0])))
    print(f"  [INFO] 같은 소리의 24 kHz 판 → 예외 없이 통과, "
          f"cos(16k, 24k) = {cross:.4f}")
    if cross < 0.99:
        ok("★ 24 kHz 입력이 조용히 다른 임베딩을 만드는 것 확인 "
           "→ 지표 계산 시 반드시 16 kHz 로 리샘플할 것")
    else:
        warn(f"24 kHz 와 16 kHz 임베딩이 거의 같습니다(cos={cross:.4f}). "
             "합성 신호 특성일 수 있으나, 16 kHz 리샘플 규칙은 그대로 지켜야 합니다.")

    # (d) 실제 사용 패턴: torchaudio 로 24k → 16k 리샘플 후 임베딩
    import torchaudio.functional as AF
    wav24_to16 = AF.resample(torch.from_numpy(wav24), DATA_SR, MODEL_SR).numpy()
    emb_d = embed(wav24_to16)
    proper = float(np.dot(emb_a[0], emb_d[0]) /
                   (np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_d[0])))
    if proper > cross:
        ok(f"24k→16k 리샘플 후 cos = {proper:.4f}  (리샘플 없이 {cross:.4f} 보다 높음)")
    else:
        warn(f"리샘플 후 cos = {proper:.4f} (리샘플 전 {cross:.4f}) — 합성 신호로는 "
             "차이가 뚜렷하지 않을 수 있습니다.")

    return passed


# ─────────────────────────────────────────────────────────────────────────
# 2) WER — Whisper large-v3
# ─────────────────────────────────────────────────────────────────────────
def download_whisper(hf_home: Path, device: str):
    """Whisper large-v3 을 내려받아 (model, processor) 를 반환한다.

    ★ snapshot_download 를 쓰지 않는다 — 해당 repo 는 flax/tf/fp32 샤드까지 포함해
      총 24.7 GB 이고, from_pretrained 는 model.safetensors(3.09 GB) + config/tokenizer
      만 가져와 ~3.1 GB 로 끝난다.
    """
    section(f"2/2  WER — {WHISPER_ID}")
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    hub = hf_home / "hub"
    before = du_mb(hub)
    # llmtse/eval.py:109 과 동일: bf16 (fp16 금지)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kv("torch_dtype", str(dtype))

    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_ID, torch_dtype=dtype, low_cpu_mem_usage=True)   # accelerate 필요
    processor = AutoProcessor.from_pretrained(WHISPER_ID)
    elapsed = time.time() - t0
    model = model.to(device).eval()

    after = du_mb(hub)
    kv("소요 시간", f"{elapsed:.1f} s")
    kv("hub 디스크", f"{after:.0f} MB  (이번에 {max(0.0, after - before):.0f} MB 추가)")
    if elapsed < 30.0 and before > 1000:
        ok("캐시 히트 — 재다운로드 없음")
    ok("Whisper large-v3 로드 완료")
    return model, processor


def verify_whisper(model, processor, device: str) -> bool:
    """16 kHz 전사 정상 동작 + 24 kHz 가 ValueError 로 거부되는 것을 확인한다."""
    import torch

    print("\n  ── 검증: Whisper ──")
    passed = True
    wav16 = sine_speechlike(MODEL_SR, seconds=2.0)

    # (a) 16 kHz — 정상 경로 (llmtse/eval.py:158-160 과 동일한 호출 형태)
    try:
        feat = processor(wav16, sampling_rate=MODEL_SR,
                         return_tensors="pt").input_features.to(device).to(model.dtype)
        with torch.no_grad():
            ids = model.generate(feat, language="en")
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        ok(f"16 kHz 전사 성공 — mel {tuple(feat.shape)}, 출력 {len(hyp)} chars")
        print(f"  [INFO] 전사 결과(합성음이라 내용은 의미 없음): {hyp[:60]!r}")
    except Exception as exc:                                # noqa: BLE001
        fail(f"16 kHz 전사 실패: {type(exc).__name__}: {exc}")
        passed = False

    # (b) ★ 24 kHz — 반드시 거부되어야 한다
    try:
        processor(sine_speechlike(DATA_SR, seconds=2.0),
                  sampling_rate=DATA_SR, return_tensors="pt")
        fail("24 kHz 입력이 거부되지 않았습니다 — Whisper 가정이 깨졌습니다")
        passed = False
    except ValueError as exc:
        ok(f"★ 24 kHz 입력을 ValueError 로 거부 확인: {str(exc)[:70]}...")

    # (c) jiwer 경로 sanity
    try:
        import jiwer
        w = jiwer.wer("hello world", "hello word")
        if abs(w - 0.5) < 1e-9:
            ok(f"jiwer.wer 동작 확인 = {w}")
        else:
            fail(f"jiwer.wer 가 0.5 가 아닙니다: {w}")
            passed = False
    except ImportError:
        fail("jiwer 가 없습니다. `pip install -r requirements/h200/requirements_h200.txt` 를 다시 실행하세요.")
        passed = False

    return passed


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TSE_Eval 확장 지표(Speaker Similarity / WER) 모델 다운로드 + 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["spk", "wer"], default=None,
                   help="한쪽만 처리 (spk=ECAPA, wer=Whisper). 기본은 둘 다.")
    p.add_argument("--no-verify", action="store_true",
                   help="다운로드만 하고 검증은 생략")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="검증 device (기본: cuda 가용 시 cuda)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # ★ HF_HOME 은 transformers/huggingface_hub import 전에 확정해야 한다.
    hf_home = resolve_hf_home()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print_environment(hf_home, device)

    do_spk = args.only in (None, "spk")
    do_wer = args.only in (None, "wer")
    results: dict[str, bool] = {}

    if do_spk:
        try:
            ecapa = download_ecapa(hf_home, device)
            results["ECAPA 다운로드"] = True
        except Exception as exc:                            # noqa: BLE001
            fail(f"ECAPA 다운로드 실패: {type(exc).__name__}: {exc}")
            results["ECAPA 다운로드"] = False
        else:
            if not args.no_verify:
                try:
                    results["ECAPA 검증"] = verify_ecapa(ecapa, device)
                except Exception as exc:                    # noqa: BLE001
                    fail(f"ECAPA 검증 실패: {type(exc).__name__}: {exc}")
                    results["ECAPA 검증"] = False

    if do_wer:
        try:
            model, processor = download_whisper(hf_home, device)
            results["Whisper 다운로드"] = True
        except Exception as exc:                            # noqa: BLE001
            fail(f"Whisper 다운로드 실패: {type(exc).__name__}: {exc}")
            results["Whisper 다운로드"] = False
        else:
            if not args.no_verify:
                try:
                    results["Whisper 검증"] = verify_whisper(model, processor, device)
                except Exception as exc:                    # noqa: BLE001
                    fail(f"Whisper 검증 실패: {type(exc).__name__}: {exc}")
                    results["Whisper 검증"] = False

    section("요약")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL':<5} {name}")
    kv("HF_HOME 총 사용량", f"{du_mb(hf_home):.0f} MB")
    print()
    print("  다음 세션에서도 캐시를 재사용하려면 셸에서 아래를 먼저 실행하세요:")
    print(f"      export HF_HOME={hf_home}")
    print()
    print("  ★ 두 모델 모두 16 kHz 전용입니다. 지표 계산 시 24 kHz 오디오는")
    print("    torchaudio.functional.resample(wav, 24000, 16000) 으로 내려서 넣으세요.")
    print("    (SI-SDR / SI-SDRi / STOI / ESTOI 는 24 kHz native 로 계산합니다.)")

    if all(results.values()) and results:
        print("\nALL CHECKS PASSED")
        return 0
    print("\nSOME CHECKS FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
