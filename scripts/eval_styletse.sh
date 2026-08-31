#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# scripts/eval_styletse.sh — StyleTSE 채점 (엔진 scripts/eval_tse.sh 의 얇은 래퍼)
# ════════════════════════════════════════════════════════════════════════════
#
# 이 파일이 아는 것은 **StyleTSE 의 경로 규약뿐**입니다. 매니페스트 스키마(19열)는
# LLM-TSE 와 동일하므로 채점 로직에 손댈 것이 없습니다 — 열 순서만 다르고 자동 감지는
# 이름 기반이라 무관합니다. 추론 산출물 규약:
#
#   <STYLETSE_ROOT>/styletse_<VERSION>_best/manifest.csv   ← Stage-2 best
#   <STYLETSE_ROOT>/styletse_<VERSION>_last/manifest.csv   ← Stage-2 last
#
#   ⚠️ LLM-TSE 와 다른 점: 파일명이 inference_test.csv 가 아니라 manifest.csv 이고,
#      디렉터리에 _step50000 접미사 대신 _best / _last 가 붙습니다. 예측 wav 도
#      <file_id>_est.wav 입니다(LLM-TSE 는 <file_id>.wav). 전부 매니페스트에 절대경로로
#      적혀 있어 채점기는 신경 쓰지 않습니다.
#
# 사용법:
#   bash scripts/eval_styletse.sh --check-only   # 선행 검사만 (약 10초)
#   bash scripts/eval_styletse.sh                # best → last 순서대로 채점
#   VERSION=v2 bash scripts/eval_styletse.sh     # 다른 학습 버전
#   ONLY=best FORCE=1 NICE=15 THREADS=2 ...      # 나머지 옵션은 엔진 참조
#
# 산출물: <추론디렉터리>/eval/styletse_<VERSION>_{best,last}{,_summary,_config.json}.csv
# ════════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

VERSION="${VERSION:-v1}"
STYLETSE_ROOT="${STYLETSE_ROOT:-/home/work/my-outputs/styletse}"

export PROJECT="styletse"
export RUNS="${RUNS:-${VERSION}_best|$STYLETSE_ROOT/styletse_${VERSION}_best/manifest.csv ${VERSION}_last|$STYLETSE_ROOT/styletse_${VERSION}_last/manifest.csv}"

# ★ readlink -f 로 실경로를 푼다. dirname "${BASH_SOURCE[0]}" 만 쓰면 이 래퍼를
#   심볼릭 링크로 걸었을 때(예: ~/bin) 링크가 있는 디렉터리에서 엔진을 찾다가
#   "No such file or directory" 로 죽는다.
_ENGINE_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
exec bash "$_ENGINE_DIR/eval_tse.sh" "$@"
