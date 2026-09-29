"""DNSMOS ONNX 세션이 torch 의 cuDNN 보다 **먼저** 만들어지는지 고정한다.

배경 (``tse_eval/ort_setup.py`` 모듈 docstring): onnxruntime-gpu 의 CUDA EP 는
provider 라이브러리를 올릴 때 cuDNN 9 를 dlopen 한다. torch 가 먼저 자기 번들
cuDNN 을 올려 두면 시스템 cuDNN 과 섞여 로드가 실패하고, onnxruntime 은 **경고 한 줄만
찍고 CPU EP 로 폴백**한다. 값은 계속 나오되 약 3배 느리고 최대 3e-03 달라지므로,
"조용히 다른 값"이 되는 종류의 사고다.

여기 테스트는 GPU 없이도 의미가 있다: 확인하는 것은 "CUDA 가 붙었는가"가 아니라
**세션을 만드는 시점**이기 때문이다.
"""
from __future__ import annotations

import sys

import pytest

from tse_eval import metrics, ort_setup


def _forget_speechmos() -> None:
    """speechmos 를 sys.modules 에서 떼어낸다.

    speechmos 는 ONNX 세션을 모듈 안에 캐시하므로, 한 번 만들어진 뒤에는 다시
    호출해도 **새 세션이 생기지 않는다**(래퍼도 그래서 아무것도 기록하지 못한다).
    테스트끼리 그 캐시를 물려받으면 검사 대상이 사라지므로 매번 떼어낸다.
    """
    for name in [m for m in sys.modules if m.split(".")[0] == "speechmos"]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _clean_ort():
    """각 테스트가 깨끗한 ort_setup·speechmos 상태에서 시작하도록 한다."""
    ort_setup._reset_for_tests()
    _forget_speechmos()
    yield
    ort_setup._reset_for_tests()
    _forget_speechmos()


def test_prime_is_noop_when_not_configured():
    """DNSMOS 를 요청하지 않은 실행(configure 미호출)에서는 세션을 만들지 않는다.

    만들어 버리면 wer/spk_sim 만 돌리는 실행이 쓰지도 않을 CUDA 초기화 ~40초를
    떠안는다.
    """
    assert ort_setup.prime_dnsmos_session(verbose=False) is None
    assert ort_setup.provenance()["actual_providers"] is None


def test_prime_builds_the_session_up_front():
    """configure 직후 프라이밍하면 첫 행을 채점하기 전에 이미 세션이 존재한다."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("speechmos")
    pytest.importorskip("librosa")           # 없으면 speechmos import 자체가 실패

    ort_setup.configure_onnxruntime(threads=1, providers="cpu", verbose=False)
    assert ort_setup.actual_providers() is None          # 아직 세션 없음

    got = ort_setup.prime_dnsmos_session(verbose=False)
    assert got == ["CPUExecutionProvider"]
    assert ort_setup.provenance()["actual_providers"] == ["CPUExecutionProvider"]


def test_prime_is_idempotent():
    """두 번째 호출은 새 세션을 만들지 않는다(래퍼가 기록한 EP 가 그대로)."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("speechmos")
    pytest.importorskip("librosa")

    ort_setup.configure_onnxruntime(threads=1, providers="cpu", verbose=False)
    first = ort_setup.prime_dnsmos_session(verbose=False)
    second = ort_setup.prime_dnsmos_session(verbose=False)
    assert first == second == ["CPUExecutionProvider"]
    # 두 번째 호출은 speechmos 를 다시 부르지 않는다 — 이미 세션이 있으면 즉시 반환.


def test_prime_survives_missing_speechmos(monkeypatch):
    """speechmos/librosa 가 없어도 예외로 실행을 깨지 않는다 (DNSMOS 는 나중에 nan)."""
    pytest.importorskip("onnxruntime")
    ort_setup.configure_onnxruntime(threads=1, providers="cpu", verbose=False)

    import builtins
    real_import = builtins.__import__

    def _fail(name, *a, **k):
        if name.startswith("speechmos"):
            raise ImportError("simulated: speechmos missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fail)
    assert ort_setup.prime_dnsmos_session(verbose=False) is None


@pytest.mark.parametrize("loader", ["_get_asr", "_get_spk"])
def test_model_loaders_prime_before_touching_torch(monkeypatch, loader):
    """WER·Speaker-Sim 모델을 올리기 **전에** 프라이밍이 먼저 일어난다.

    ``compute_row_metrics`` 는 이미 DNSMOS 를 먼저 계산하지만, 지표 함수를 직접
    쓰는 코드에는 그런 보장이 없다. 그래서 두 로더가 스스로 순서를 지켜야 한다.
    """
    order: list[str] = []

    def fake_prime(verbose: bool = True):
        order.append("prime")
        return None

    monkeypatch.setattr(ort_setup, "prime_dnsmos_session", fake_prime)
    # 모델 로딩은 여기서 멈춘다 — 순서만 확인하면 되므로 가중치는 건드리지 않는다.
    monkeypatch.setattr(metrics, "WER_MODEL_ID", "/nonexistent/tse-eval-test-model")
    monkeypatch.setattr(metrics, "SPK_SIM_MODEL_ID", "/nonexistent/tse-eval-test-model")
    metrics._asr_cache.clear()
    metrics._spk_cache.clear()

    with pytest.raises(Exception):
        getattr(metrics, loader)()

    assert order == ["prime"], "모델 로더가 프라이밍보다 먼저 torch 를 건드렸습니다"
