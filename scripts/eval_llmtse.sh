#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# scripts/eval_llmtse.sh — LLM-TSE 채점 (엔진 scripts/eval_tse.sh 의 얇은 래퍼)
# ════════════════════════════════════════════════════════════════════════════
#
# 이 파일이 아는 것은 **LLM-TSE 의 경로 규약뿐**입니다. 선행 검사·양보 모드·채점·
# 검증은 전부 엔진에 있습니다. 추론 산출물 규약:
#
#   <LLMTSE_ROOT>/llmtse_baseline_<VERSION>/inference_test.csv            ← best
#   <LLMTSE_ROOT>/llmtse_baseline_<VERSION>_step50000/inference_test.csv  ← last
#
# 사용법:
#   bash scripts/eval_llmtse.sh --check-only     # 선행 검사만 (약 10초)
#   bash scripts/eval_llmtse.sh                  # best → last 순서대로 채점
#   VERSION=v1 bash scripts/eval_llmtse.sh       # 다른 학습 버전
#   ONLY=best FORCE=1 NICE=15 THREADS=2 ...      # 나머지 옵션은 엔진 참조
#
# ★ V1 만 명명 규칙이 다릅니다. V1 은 VERSION 이 생기기 전에 채점돼 산출물이
#   llmtse_best.csv / llmtse_last.csv (버전 접두사 없음)입니다. `VERSION=v1` 로 돌리면
#   llmtse_v1_best.csv 가 **새로** 생겨 기존 파일과 공존합니다. V1 을 그대로 재현하려면:
#     RUNS="best|/home/work/my-outputs/llmtse/llmtse_baseline_v1/inference_test.csv" \
#       bash scripts/eval_llmtse.sh
# ════════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

VERSION="${VERSION:-v2}"
LLMTSE_ROOT="${LLMTSE_ROOT:-/home/work/my-outputs/llmtse}"

export PROJECT="llmtse"
export RUNS="${RUNS:-${VERSION}_best|$LLMTSE_ROOT/llmtse_baseline_${VERSION}/inference_test.csv ${VERSION}_last|$LLMTSE_ROOT/llmtse_baseline_${VERSION}_step50000/inference_test.csv}"

# ★ readlink -f 로 실경로를 푼다. dirname "${BASH_SOURCE[0]}" 만 쓰면 이 래퍼를
#   심볼릭 링크로 걸었을 때(예: ~/bin) 링크가 있는 디렉터리에서 엔진을 찾다가
#   "No such file or directory" 로 죽는다.
_ENGINE_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
exec bash "$_ENGINE_DIR/eval_tse.sh" "$@"
