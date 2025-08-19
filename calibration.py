# === calibration.py (FINAL) ==============================================
"""
확률 보정 모듈 (가볍고 안전)
- 기본 OFF: config.get_CALIB()["enabled"]가 False면 모든 API는 원본 그대로 반환.
- 지원 기법:
    * Temperature Scaling (다중클래스 가능)
    * Platt Scaling (이진 전용)
- 저장/로드: /persistent/calib/{symbol}_{strategy}_{model}.json
- 외부에서 쓰는 핵심 함수:
    * apply_calibration(raw_prob, meta) -> prob
    * fit_and_save(y_true, y_pred_proba, meta)  # 훈련 끝난 뒤 혹은 주기적으로
"""

from __future__ import annotations
import os, json, math
import numpy as np

# ---- Config 안전 로더 ----------------------------------------------------
try:
    from config import get_NUM_CLASSES
except Exception:
    def get_NUM_CLASSES(): return 2

try:
    from config import _default_config  # 선택적
except Exception:
    _default_config = {}

def _get_CALIB():
    # 기본값(OFF)
    base = {
        "enabled": False,          # 전역 스위치
        "method": "temperature",   # "temperature" | "platt"
        "min_samples": 200,        # 학습 최소 표본
        "clip_eps": 1e-6           # 확률 안정화
    }
    try:
        from config import CALIB  # 있으면 사용
        base.update(CALIB if isinstance(CALIB, dict) else {})
    except Exception:
        pass
    return base

# ---- 파일 경로 -----------------------------------------------------------
CALIB_DIR = "/persistent/calib"
os.makedirs(CALIB_DIR, exist_ok=True)

def _calib_path(symbol: str, strategy: str, model: str) -> str:
    fname = f"{symbol}_{strategy}_{model}.json".replace("/", "_")
    return os.path.join(CALIB_DIR, fname)

# ---- 수학 유틸 -----------------------------------------------------------
def _softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(T, 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _to_onehot(y: np.ndarray, K: int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((y.shape[0], K), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

# ---- Temperature Scaling (multi-class) -----------------------------------
def _fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter=100, lr=0.01) -> float:
    """
    간단한 1D GD로 NLL 최소화(안정/가벼움). 반환: T>0
    logits: (N,K) — 모델의 로짓(또는 logit 비례 점수). 확률이 들어온 경우 logit으로 변환 권장.
    """
    T = 1.0
    K = logits.shape[1]
    y_oh = _to_onehot(y_true, K)

    for _ in range(max_iter):
        p = _softmax(logits, T)  # (N,K)
        # NLL의 d/dT 근사: (∂log softmax/∂T) * (p - y)
        # 안정적 근사를 위해 수치적 gradient 사용
        eps = 1e-3
        p_eps = _softmax(logits, T + eps)
        nll = -np.mean(np.sum(y_oh * np.log(p + 1e-12), axis=1))
        nll_eps = -np.mean(np.sum(y_oh * np.log(p_eps + 1e-12), axis=1))
        g = (nll_eps - nll) / eps
        T_new = max(1e-3, T - lr * g)
        if abs(T_new - T) < 1e-6:
            T = T_new
            break
        T = T_new
    return float(T)

# ---- Platt Scaling (binary) ----------------------------------------------
def _fit_platt(scores: np.ndarray, y_true: np.ndarray, max_iter=100, lr=0.01) -> tuple[float, float]:
    """
    Platt: sigmoid(A*x + B). 여기서 x는 logit 혹은 점수(양수=긍정 쪽).
    간단한 GD (A,B) 학습.
    """
    A, B = 1.0, 0.0
    y = y_true.astype(np.float64)

    for _ in range(max_iter):
        s = A * scores + B
        p = _sigmoid(s)
        # 로스 = -[y log p + (1-y) log(1-p)]
        # dA = (p - y)*x, dB = (p - y)
        grad = (p - y)
        dA = np.mean(grad * scores)
        dB = np.mean(grad)
        A -= lr * dA
        B -= lr * dB
    return float(A), float(B)

# ---- 저장/로드 ------------------------------------------------------------
def load(symbol: str, strategy: str, model: str) -> dict | None:
    path = _calib_path(symbol, strategy, model)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj
    except Exception:
        return None

def save(params: dict, symbol: str, strategy: str, model: str) -> None:
    path = _calib_path(symbol, strategy, model)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---- 외부 API -------------------------------------------------------------
def fit_and_save(y_true: np.ndarray,
                 y_pred_proba_or_logits: np.ndarray,
                 meta: dict) -> dict | None:
    """
    훈련/주기 업데이트용.
    meta = {"symbol":.., "strategy":.., "model":.., "is_logits":bool}
    """
    cfg = _get_CALIB()
    if not cfg.get("enabled", False):
        return None

    symbol   = meta.get("symbol", "UNK")
    strategy = meta.get("strategy", "UNK")
    model    = meta.get("model", "UNK")
    is_logits= bool(meta.get("is_logits", False))

    y_true = np.asarray(y_true)
    X = np.asarray(y_pred_proba_or_logits, dtype=np.float64)

    if X.ndim == 1:
        # binary 점수로 간주 → Platt
        if y_true.shape[0] < cfg["min_samples"]:
            return None
        A, B = _fit_platt(X, y_true)
        params = {"method": "platt", "A": A, "B": B, "ver": 1}
        save(params, symbol, strategy, model)
        return params

    # multi-class
    if y_true.shape[0] < cfg["min_samples"]:
        return None

    if is_logits:
        logits = X
    else:
        # 확률 → 로짓 근사
        P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
        logits = np.log(P)
    T = _fit_temperature(logits, y_true)
    params = {"method": "temperature", "T": T, "ver": 1}
    save(params, symbol, strategy, model)
    return params

def apply_calibration(raw_prob_or_scores: np.ndarray,
                      meta: dict) -> np.ndarray:
    """
    예측 직후 호출.
    - meta: {"symbol","strategy","model","is_logits":bool}
    - 반환값: 보정된 확률 배열 (shape 유지)
    - 설정 OFF 또는 파라미터 없음: 입력을 안정화(CLIP)만 해서 그대로 반환.
    """
    cfg = _get_CALIB()
    X = np.asarray(raw_prob_or_scores, dtype=np.float64)
    if not cfg.get("enabled", False):
        # 안정화만
        if X.ndim == 2:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            return P
        # 1D는 sigmoid 확률로 가정하지 않고 그대로 반환(메타러너가 처리)
        return X

    symbol   = meta.get("symbol", "UNK")
    strategy = meta.get("strategy", "UNK")
    model    = meta.get("model", "UNK")
    is_logits= bool(meta.get("is_logits", False))

    params = load(symbol, strategy, model)
    if params is None:
        # 파라미터 없음 → 입력 안정화 후 반환
        if X.ndim == 2:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            return P
        return X

    method = params.get("method", "temperature")
    if method == "temperature":
        # multi-class
        if X.ndim == 1:
            # 1D가 들어오면 그대로 반환
            return X
        T = max(1e-3, float(params.get("T", 1.0)))
        if is_logits:
            P = _softmax(X, T)
        else:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            # 확률에 온도 적용은 로짓 변환 후 softmax가 더 안정적
            logits = np.log(P)
            P = _softmax(logits, T)
        return P

    elif method == "platt":
        # binary 전용: 1D 점수/로짓 → 확률
        if X.ndim != 1:
            # shape (N,2) 확률이 들어오면 그대로
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            return P
        A = float(params.get("A", 1.0)); B = float(params.get("B", 0.0))
        p = _sigmoid(A * X + B)
        return p

    # 알 수 없는 메서드 → 안정화만
    if X.ndim == 2:
        P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        return P
    return X

# ---- 품질 리포트(선택) ----------------------------------------------------
def expected_calibration_error(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    간단 ECE(멀티클래스는 argmax 기준).
    """
    y_true = np.asarray(y_true).astype(int)
    if y_prob.ndim == 1:
        conf = y_prob
        pred = (y_prob >= 0.5).astype(int)
    else:
        conf = y_prob.max(axis=1)
        pred = y_prob.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.any():
            acc = (pred[m] == y_true[m]).mean()
            conf_mean = conf[m].mean()
            ece += (m.mean()) * abs(acc - conf_mean)
    return float(ece)
# ========================================================================
