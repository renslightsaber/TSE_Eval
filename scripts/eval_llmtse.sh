#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# scripts/eval_llmtse.sh — LLM-TSE 두 체크포인트(best / last)를 순서대로 채점
# ════════════════════════════════════════════════════════════════════════════
#
# 이 스크립트가 존재하는 이유는 편의가 아니라 **선행 검사**입니다. 전체 런이 약 3.6시간
# (WER 이 그중 2시간)이라, 아래 중 하나라도 빠지면 몇 시간을 버립니다:
#
#   · HF_HOME 미설정          → whisper-large-v3 3 GB 를 39 GB 남은 /home/work 에 재다운로드
#   · asteroid 미설치         → 첫 행에서 RuntimeError
#   · GPU 메모리 부족         → whisper 로딩 시점(= SI-SDR·STOI·PESQ 를 다 계산한 뒤)에 OOM
#   · 모델 캐시 불완전        → 두 시간 뒤 다운로드 시도
#
# preflight 는 이것들을 시작 10초 안에 잡아냅니다.
#
# ── 양보 모드 (같은 박스에서 CosyVoice·StyleTSE 학습이 동시에 돌아감) ──────────
# 24 코어 · H200 1장 · /dev/shm 2 GB 를 셋이 나눠 씁니다. 학습을 밀지 않도록
# nice 로 CPU 우선권을 양보하고 스레드 수를 묶습니다. 자세한 근거는 아래 [양보] 절.
#
# ── 사용법 ───────────────────────────────────────────────────────────────────
#   bash scripts/eval_llmtse.sh --check-only     # 선행 검사만 (약 10초)
#   bash scripts/eval_llmtse.sh                  # best → last 순서대로 채점
#
#   ONLY=best   bash scripts/eval_llmtse.sh      # 한쪽만
#   FORCE=1     bash scripts/eval_llmtse.sh      # 기존 결과 무시하고 재실행
#   NICE=15 THREADS=2 bash scripts/eval_llmtse.sh          # 더 많이 양보
#   DNSMOS_PROVIDERS=cpu bash scripts/eval_llmtse.sh       # GPU 를 학습에 완전히 양보
#
# 산출물 (ckpt 당 4개):
#   <추론디렉터리>/eval/llmtse_<tag>.csv           per-row 지표 (원본 컬럼 + 13지표 + error)
#   <추론디렉터리>/eval/llmtse_<tag>_summary.csv   long-format 다축 요약
#   <추론디렉터리>/eval/llmtse_<tag>_config.json   provenance sidecar ← 논문에 인용할 파일
#   <추론디렉터리>/eval/eval.log                   실행 로그 (append)
# ════════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

trap 'echo "" >&2; echo "✗ 실패: ${BASH_SOURCE[0]}:${LINENO} — 종료코드 $?" >&2' ERR

# ── 설정 (환경변수로 전부 덮어쓰기 가능) ─────────────────────────────────────
ENV_NAME="${ENV_NAME:-tseeval}"
BACKEND="${BACKEND:-asteroid}"
TARGET_SR="${TARGET_SR:-24000}"
NICE="${NICE:-10}"
THREADS="${THREADS:-4}"
ONLY="${ONLY:-}"                       # 빈 값 = 둘 다, 또는 best | last
FORCE="${FORCE:-0}"
DNSMOS_PROVIDERS="${DNSMOS_PROVIDERS:-}"   # 빈 값 = config 기본(cuda), 또는 cpu

# ★ 영속 vFolder. 비어 있으면 whisper 3 GB 가 /home/work(49 GB) 로 다시 내려옵니다.
export HF_HOME="${HF_HOME:-/home/work/my-checkpoints/hf_cache}"

# 13개 지표 전부.
# ★ `--metrics all,wer` 는 동작하지 않습니다 — cli._resolve_metrics 는 인자 *전체*가
#   정확히 'all' 일 때만 config 세트로 해석하므로 'all,wer' 는 Unknown metric(s): ['all']
#   로 종료합니다. 그래서 여기서는 열거합니다(sh 만 봐도 무엇을 쟀는지 읽히는 이점도 있음).
METRICS="${METRICS:-si_sdr,si_sdri,input_si_sdr,input_si_sdr_pairwise,stoi,estoi,pesq,dnsmos_sig,dnsmos_bak,dnsmos_ovrl,dnsmos_p808,wer,spk_sim}"

# 계층화 축. same_gender 는 target_gender/infer_gender 에서 파생됩니다(evaluate.add_derived_axes).
GROUP_BY="${GROUP_BY:-overlap_ratio,prompt_category,same_gender,first_speak}"

# 채점 대상 — "태그|추론 CSV" 를 순서대로. 출력은 각 CSV 옆 eval/ 에 들어갑니다.
# RUNS 환경변수(공백/줄바꿈 구분)로 덮어쓸 수 있습니다 — 소규모 스모크 테스트나
# 세 번째 체크포인트 추가 채점에 사용:
#   RUNS="smoke|/tmp/subset.csv" bash scripts/eval_llmtse.sh
if [[ -n "${RUNS:-}" ]]; then
  read -r -a RUNS <<< "$RUNS"
else
  RUNS=(
    "best|/home/work/my-outputs/llmtse/llmtse_baseline_v1/inference_test.csv"
    "last|/home/work/my-outputs/llmtse/llmtse_baseline_v1_step50000/inference_test.csv"
  )
fi

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────
hr()   { printf '─%.0s' {1..76}; echo; }
head1() { echo; hr; echo "  $*"; hr; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
die()  { echo "  ✗ $*" >&2; exit 1; }

# ════════════════════════════════════════════════════════════════════════════
# ① preflight
# ════════════════════════════════════════════════════════════════════════════
head1 "① 선행 검사"

# 1. conda env
CONDA_BASE="$(conda info --base 2>/dev/null)" || die "conda 를 찾을 수 없습니다."
[[ -d "$CONDA_BASE/envs/$ENV_NAME" ]] \
  || die "conda env '$ENV_NAME' 가 없습니다. (conda env list 로 확인)"
# conda 의 셸 함수는 미정의 변수를 건드리므로 set -u 를 잠시 끕니다.
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
ok "conda env: $ENV_NAME ($(python -V 2>&1))"

# 2·3. 필수 패키지 + asteroid 백엔드 가용성
#      CLI 도 fail-fast 하지만 여기서 잡으면 모델 로딩 전에 끝납니다.
python - "$BACKEND" <<'PY' || die "필수 패키지 검사 실패 (위 메시지 참조)"
import importlib, sys
need = ["tse_eval", "numpy", "pandas", "soundfile", "librosa", "scipy",
        "pystoi", "pesq", "speechmos", "onnxruntime", "torch",
        "transformers", "jiwer", "speechbrain"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception as exc:                                   # noqa: BLE001
        missing.append(f"{m} ({exc.__class__.__name__})")
if missing:
    print("  ✗ import 실패:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)

from tse_eval.backends import available_backends
avail = available_backends()
want = sys.argv[1]
if not avail.get(want, False):
    print(f"  ✗ si_sdr 백엔드 '{want}' 사용 불가. available={avail}\n"
          f"    → pip install -e '.[test]' 또는 BACKEND=native", file=sys.stderr)
    sys.exit(1)

import torch, onnxruntime as ort
print(f"  ✓ 패키지 {len(need)}개 import 성공")
print(f"  ✓ si_sdr 백엔드 '{want}' 사용 가능  (available={avail})")
print(f"  ✓ torch {torch.__version__} · cuda={torch.cuda.is_available()} · "
      f"onnxruntime {ort.__version__}")
print(f"  ✓ onnxruntime providers: {ort.get_available_providers()}")
PY

# 4. 모델 캐시 — 여기서 막지 않으면 WER 단계(2시간 지점)에서 3 GB 를 내려받습니다.
[[ -d "$HF_HOME" ]] || die "HF_HOME 이 없습니다: $HF_HOME"
ok "HF_HOME: $HF_HOME"
for spec in "models--openai--whisper-large-v3|model.safetensors" \
            "models--speechbrain--spkrec-ecapa-voxceleb|embedding_model.ckpt"; do
  repo="${spec%%|*}"; key="${spec##*|}"
  snap="$HF_HOME/hub/$repo/snapshots"
  [[ -d "$snap" ]] || die "모델 캐시 없음: $repo
    → python scripts/download_models.py  를 먼저 실행하세요."
  find "$snap" -name "$key" | grep -q . \
    || die "모델 캐시 불완전: $repo 에 $key 가 없습니다.
    → python scripts/download_models.py  를 다시 실행하세요."
  ok "모델 캐시: ${repo#models--}  ($(du -sh --exclude='*.incomplete' \
        "$HF_HOME/hub/$repo" 2>/dev/null | cut -f1))"
done
if find "$HF_HOME" -name '*.incomplete' 2>/dev/null | grep -q .; then
  die "다운로드가 중단된 파일(.incomplete)이 있습니다.
    → python scripts/download_models.py  로 마무리하세요."
fi

# 5. 입력 CSV
for spec in "${RUNS[@]}"; do
  tag="${spec%%|*}"; csv="${spec#*|}"
  [[ -f "$csv" ]] || die "입력 CSV 없음 ($tag): $csv"
  ok "입력 [$tag]: $(( $(wc -l < "$csv") - 1 ))행  $csv"
done

# 6. GPU 여유 — 평가는 약 5 GB 를 씁니다
#    (whisper-large-v3 bf16 ≈3.1 · ECAPA ≈0.05 · DNSMOS onnx ≈0.5 · CUDA context ≈1)
#    8 GB 를 요구해 여유를 둡니다. 부족하면 학습이 GPU 를 채운 상태입니다.
GPU_MIN_FREE_MIB=8000
if command -v nvidia-smi >/dev/null 2>&1; then
  read -r gpu_total gpu_used < <(nvidia-smi \
      --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
      | head -1 | tr -d ',')
  gpu_free=$(( gpu_total - gpu_used ))
  if (( gpu_free < GPU_MIN_FREE_MIB )); then
    echo "  현재 GPU 점유 프로세스:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sed 's/^/    /'
    die "GPU 여유 ${gpu_free} MiB < 필요 ${GPU_MIN_FREE_MIB} MiB.
    → 학습이 끝나길 기다리거나, DNSMOS_PROVIDERS=cpu 로도 WER/spk_sim 은 GPU 가 필요합니다."
  fi
  ok "GPU 여유: ${gpu_free} MiB / ${gpu_total} MiB (필요 ≈5000)"
else
  warn "nvidia-smi 없음 — GPU 확인을 건너뜁니다 (WER/spk_sim 이 CPU 로 떨어지면 매우 느립니다)"
fi

# 7. 동시 실행 상황 — 중단하지 않고 보고만
# ★ grep 은 매치가 없으면 종료코드 1 을 냅니다. pipefail 이 걸려 있으므로 `|| true`
#   없이는 "학습이 안 돌고 있다"는 정상 상황에서 스크립트가 죽습니다.
train_ps=$(ps -eo pid,pcpu,args --sort=-pcpu 2>/dev/null \
  | grep -E 'train\.py|accelerate launch' | grep -v grep || true)
if [[ -n "$train_ps" ]]; then
  n_train=$(wc -l <<< "$train_ps")
  train_cores=$(awk '{s+=$2} END {printf "%.1f", s/100}' <<< "$train_ps")
else
  n_train=0
  train_cores="0.0"
fi
loadavg=$(cut -d' ' -f1 /proc/loadavg)
ncore=$(python -c 'import os;print(len(os.sched_getaffinity(0)))')
if (( n_train > 0 )); then
  warn "학습 프로세스 ${n_train}개 감지 — 약 ${train_cores} 코어 사용 중"
  head -4 <<< "$train_ps" | cut -c1-110 | sed 's/^/      /'
else
  ok "학습 프로세스 없음 — 평가가 자원을 독점합니다"
fi
ok "코어 ${ncore}개 · loadavg ${loadavg} · shm $(df -h /dev/shm | awk 'NR==2{print $4}') 여유"
# ★ 이 박스의 loadavg 는 실제 CPU 경합보다 크게 나옵니다 — 데이터가 전부 NFS 라
#   D-state(I/O 대기) 프로세스가 loadavg 에 포함되기 때문입니다(측정 예: %CPU 합계
#   5 코어인데 loadavg 34). 그래서 아래는 "느려질 수 있다"는 참고 신호일 뿐이고,
#   실제 CPU 경합은 위의 '학습 프로세스 N개 — 약 X 코어' 쪽을 보는 게 정확합니다.
awk -v l="$loadavg" -v n="$ncore" 'BEGIN{ if (l > n*0.85)
  print "  ⚠ loadavg 가 코어 수의 85% 를 넘었습니다 (NFS I/O 대기 포함이라 과대 표시될 수 있음)" }'

# 8. 출력 볼륨
ok "출력 볼륨 여유: $(df -h /home/work/my-outputs | awk 'NR==2{print $4}')"

if (( CHECK_ONLY )); then
  head1 "선행 검사 통과 — 채점은 하지 않았습니다 (--check-only)"
  echo "  실제 실행:  bash scripts/eval_llmtse.sh"
  echo
  exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# [양보] 스레드·우선순위
# ════════════════════════════════════════════════════════════════════════════
# 이 컨테이너의 torch 기본 intra-op 스레드 수 = 코어 수(24). 묶지 않으면 whisper 의
# CPU 측 전처리와 numpy/BLAS 가 24 스레드를 잡으려 들어 학습 DataLoader 와 부딪힙니다.
#
# ★★ OMP_NUM_THREADS 는 onnxruntime(DNSMOS)에는 **아무 효과가 없습니다** — 실측으로
#    확인된 사실입니다(346 vs 333 ms/utt, 차이 없음). onnxruntime 의 intra-op 풀은
#    자체 관리이므로 SessionOptions.intra_op_num_threads 로만 제어되며, 그 경로는
#    tse_eval/ort_setup.py 가 config 의 dnsmos.intra_op_threads(=4) 로 처리합니다.
#    아래 export 들과 중복이 아니니 "정리" 하지 마세요 — 담당 대상이 다릅니다.
export OMP_NUM_THREADS="$THREADS"        # torch (whisper·ECAPA) intra-op
export MKL_NUM_THREADS="$THREADS"        # numpy/BLAS — PESQ·STOI 주변
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false      # whisper 토크나이저의 fork 경고 억제

# ionice 는 넣지 않습니다: 데이터 3경로(my-datasets·my-outputs·my-checkpoints)가 전부
# 동일 NFS 서버라 로컬 블록 큐가 없어 무효입니다. 평가의 NFS 읽기는 행당 3파일 ≈1.5 MB
# ÷ 1.3 s ≈ 1.2 MB/s 로, 학습 데이터 로딩과 경합할 수준이 아닙니다.
#
# nice 는 CPU 만 양보합니다. GPU 는 커널 스케줄러가 관여하지 않으므로 학습과 서로
# 조금씩 느려지는 것은 감수합니다(DNSMOS 의 GPU 발자국은 ≈0.5 GB 로 작습니다).

head1 "② 채점 시작"
echo "  백엔드    : $BACKEND        (sidecar 에 기록됩니다)"
echo "  target_sr : $TARGET_SR"
echo "  지표      : $(tr ',' ' ' <<< "$METRICS" | wc -w)개"
echo "  축        : $GROUP_BY"
echo "  양보      : nice $NICE · threads $THREADS"
[[ -n "$DNSMOS_PROVIDERS" ]] && echo "  DNSMOS EP : $DNSMOS_PROVIDERS (명시)"
echo "  예상      : ckpt 당 약 110분 (단독 실행 기준. 학습과 동시면 더 걸립니다)"

TOTAL_START=$SECONDS
DONE_TAGS=()

for spec in "${RUNS[@]}"; do
  tag="${spec%%|*}"; in_csv="${spec#*|}"

  if [[ -n "$ONLY" && "$ONLY" != "$tag" ]]; then
    echo; echo "  ⏭  [$tag] ONLY=$ONLY 이므로 건너뜀"
    continue
  fi

  eval_dir="$(dirname "$in_csv")/eval"
  out_csv="$eval_dir/llmtse_${tag}.csv"
  summary_csv="$eval_dir/llmtse_${tag}_summary.csv"
  sidecar="$eval_dir/llmtse_${tag}_config.json"
  log="$eval_dir/eval.log"
  mkdir -p "$eval_dir"

  if [[ -f "$out_csv" && "$FORCE" != "1" ]]; then
    echo; echo "  ⏭  [$tag] 이미 존재하므로 건너뜀: $out_csv"
    echo "        다시 돌리려면 FORCE=1"
    DONE_TAGS+=("$tag")
    continue
  fi

  head1 "[$tag] $(basename "$(dirname "$in_csv")")"
  {
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  run   : llmtse_${tag}"
    echo "  start : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "  input : $in_csv"
    echo "  cmd   : backend=$BACKEND sr=$TARGET_SR nice=$NICE threads=$THREADS"
    echo "════════════════════════════════════════════════════════════════"
  } >> "$log"

  dnsmos_args=()
  [[ -n "$DNSMOS_PROVIDERS" ]] && dnsmos_args=(--dnsmos-providers "$DNSMOS_PROVIDERS")

  RUN_START=$SECONDS
  nice -n "$NICE" python -m tse_eval \
      --input  "$in_csv" \
      --output "$out_csv" \
      --model-name "llmtse_${tag}" \
      --target-sr "$TARGET_SR" \
      --si-sdr-backend "$BACKEND" \
      --metrics "$METRICS" \
      --group-by "$GROUP_BY" \
      "${dnsmos_args[@]}" \
      2>&1 | tee -a "$log"
  elapsed=$(( SECONDS - RUN_START ))
  printf '  ⏱  [%s] 소요 %dh %dm %ds\n' \
      "$tag" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) | tee -a "$log"

  # ── ③⑤ 검증 ─────────────────────────────────────────────────────────────
  # 숫자를 믿을 수 있는지 그 자리에서 판정합니다. 세 시간 뒤에 "전부 NaN 이었다"를
  # 발견하는 것보다 지금 아는 편이 낫습니다. 문제가 있어도 exit 하지 않고(뒤 ckpt 는
  # 계속 돌려야 하므로) 눈에 띄게 경고만 남깁니다.
  echo "  ── 검증 ──────────────────────────────────────────────────────"
  python - "$out_csv" "$summary_csv" "$sidecar" "$BACKEND" <<'PY' 2>&1 | tee -a "$log"
import json
import sys

import numpy as np
import pandas as pd

out_csv, summary_csv, sidecar, want_backend = sys.argv[1:5]
problems = []

df = pd.read_csv(out_csv)
print(f"    행 수                       : {len(df)}")

# 1. 에러 행 — _evaluate_row 가 예외를 삼켜 error 컬럼에 담습니다.
if "error" in df.columns:
    bad = df[df["error"].notna() & (df["error"].astype(str).str.strip() != "")]
    print(f"    error 컬럼 비어있지 않은 행  : {len(bad)}")
    if len(bad):
        problems.append(f"{len(bad)}개 행이 에러로 채점되지 않았습니다")
        for _, r in bad.head(3).iterrows():
            print(f"      · {r.iloc[0]}: {str(r['error'])[:90]}")

# 2. SI-SDRi 항등식 — 리뷰어가 지적했던 바로 그 지점.
if {"si_sdr", "si_sdri", "input_si_sdr"} <= set(df.columns):
    resid = (df["si_sdri"] - (df["si_sdr"] - df["input_si_sdr"])).abs()
    m = float(np.nanmax(resid.to_numpy())) if len(resid) else float("nan")
    print(f"    max |si_sdri-(si_sdr-input)| : {m:.3e}   (기대 ≈0)")
    if not (m < 1e-9):
        problems.append(f"SI-SDRi 항등식이 깨졌습니다 (max {m:.3e})")

# 3. 두 baseline 의 일치 — 이 데이터는 pred/target/mixed 프레임 수가 완전히 같으므로
#    3-way 창과 pairwise 창이 동일해져 **정확히** 0 이어야 합니다. 0 이 아니면
#    길이 가정이 깨진 것(= 추론 산출물이 바뀐 것)입니다.
if {"input_si_sdr", "input_si_sdr_pairwise"} <= set(df.columns):
    d = (df["input_si_sdr"] - df["input_si_sdr_pairwise"]).abs()
    m = float(np.nanmax(d.to_numpy())) if len(d) else float("nan")
    n_nz = int((d > 1e-12).sum())
    print(f"    max |input - input_pairwise| : {m:.3e}   (기대 정확히 0, 불일치 {n_nz}행)")
    if n_nz:
        problems.append(
            f"{n_nz}개 행에서 input_si_sdr != input_si_sdr_pairwise — "
            "est/ref/mix 길이가 더 이상 같지 않습니다")

# 4. 지표별 NaN
metric_cols = [c for c in df.columns
               if c.startswith(("si_sdr", "input_si_sdr", "stoi", "estoi", "pesq",
                                "dnsmos_", "wer", "spk_sim"))
               and c not in ("wer_hyp",)]
nan_counts = {c: int(df[c].isna().sum()) for c in metric_cols
              if pd.api.types.is_numeric_dtype(df[c])}
nz = {c: v for c, v in nan_counts.items() if v}
print(f"    NaN 있는 지표                : {nz if nz else '없음'}")
if nz:
    problems.append(f"NaN 이 있는 지표: {nz}")

# 5. sidecar — 출처. 두 ckpt 가 서로 다른 설정으로 채점되면 비교가 무의미합니다.
with open(sidecar) as fh:
    sc = json.load(fh)
print(f"    sidecar si_sdr_backend       : {sc.get('si_sdr_backend')}   (요청 {want_backend})")
if sc.get("si_sdr_backend") != want_backend:
    problems.append(f"백엔드 불일치: 기록 {sc.get('si_sdr_backend')} != 요청 {want_backend}")
print(f"    sidecar target_sr            : {sc.get('target_sr')}")

dns = sc.get("dnsmos") or {}
actual = dns.get("actual_providers")
print(f"    DNSMOS 실사용 EP             : {actual}")
if actual and not any("CUDA" in str(p) for p in actual):
    # 학습이 GPU 를 채운 상태에서 세션을 만들면 조용히 CPU 로 떨어집니다.
    # CPU/CUDA 의 DNSMOS 값은 ~3e-3 차이가 나므로 두 ckpt 가 섞이면 비교가 깨집니다.
    problems.append("DNSMOS 가 CUDA 가 아닌 EP 로 실행됐습니다 — "
                    "다른 ckpt 와 EP 가 다르면 DNSMOS 비교가 무효입니다")

# 6. 전체 성능 — WER 은 corpus micro-WER 로 집계됩니다(행별 WER 의 평균이 아님).
#    ★ 축을 지정하면 summarize() 는 축별 group="ALL" 행만 만들고 axis="ALL" 행은
#      만들지 않습니다(evaluate.summarize: `if not axes` 일 때만 추가). 축별 ALL 행은
#      모두 전체 프레임에 대한 동일한 집계이므로 아무거나 하나 집으면 됩니다.
s = pd.read_csv(summary_csv)
allrow = s[s["group"] == "ALL"]
if len(allrow):
    show = [c for c in ("n", "si_sdr", "si_sdri", "input_si_sdr", "estoi",
                        "pesq", "dnsmos_ovrl", "wer", "spk_sim") if c in allrow.columns]
    print(f"    ── 전체 ({len(df)}행) ──")
    for c in show:
        v = allrow.iloc[0][c]
        print(f"      {c:14s} {v:.4f}" if isinstance(v, float) else f"      {c:14s} {v}")
else:
    problems.append("요약에 group='ALL' 행이 없습니다")

if problems:
    print()
    print("    ⚠⚠ 검증 경고 " + "─" * 40)
    for p in problems:
        print(f"      · {p}")
else:
    print("    ✓ 모든 검증 통과")
PY
  echo
  DONE_TAGS+=("$tag")
done

total=$(( SECONDS - TOTAL_START ))
head1 "완료 — ${DONE_TAGS[*]:-없음}  (총 $((total/3600))h $((total%3600/60))m)"
echo "  결과 위치:"
for spec in "${RUNS[@]}"; do
  tag="${spec%%|*}"; in_csv="${spec#*|}"
  [[ -n "$ONLY" && "$ONLY" != "$tag" ]] && continue
  echo "    $(dirname "$in_csv")/eval/"
done
echo
echo "  다음 단계 — 완료 확인 + best/last 비교 + 출처 일치 검사:"
echo "    python scripts/check_eval_done.py"
echo
echo "  논문에 인용할 파일은 per-row CSV 가 아니라 *_config.json (sidecar) 입니다 —"
echo "  백엔드·샘플레이트·모델 id·DNSMOS 실사용 EP·라이브러리 버전이 들어 있습니다."
echo
