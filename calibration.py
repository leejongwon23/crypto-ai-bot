# === calibration.py (patched: atomic save/load, file lock, meta, validation) ===
from __future__ import annotations
import os, json, math, time
import numpy as np
from typing import Optional, Tuple

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
    base = {
        "enabled": False,
        "method": "temperature",
        "min_samples": 200,
        "clip_eps": 1e-6,
        "lock_timeout": 5.0
    }
    try:
        from config import CALIB
        base.update(CALIB if isinstance(CALIB, dict) else {})
    except Exception:
        pass
    return base

# ---- 파일 경로 / 원자적 쓰기 / 간단 파일락 --------------------------------
CALIB_DIR = "/persistent/calib"
os.makedirs(CALIB_DIR, exist_ok=True)

def _calib_path(symbol: str, strategy: str, model: str) -> str:
    fname = f"{symbol}_{strategy}_{model}.json".replace("/", "_")
    return os.path.join(CALIB_DIR, fname)

def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

class _SimpleFileLock:
    def __init__(self, path: str, timeout: float = 5.0, poll: float = 0.05):
        self.lockfile = path + ".lock"
        self.timeout = float(timeout)
        self.poll = float(poll)
    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} ts={time.time()}\n")
                return self
            except FileExistsError:
                try:
                    mtime = os.path.getmtime(self.lockfile)
                    if time.time() - mtime > max(60.0, self.timeout*2):
                        os.remove(self.lockfile)
                        continue
                except Exception:
                    pass
                if time.time() >= deadline:
                    raise TimeoutError("lock timeout")
                time.sleep(self.poll)
    def __exit__(self, exc_type, exc, tb):
        try:
            if os.path.exists(self.lockfile):
                os.remove(self.lockfile)
        except Exception:
            pass

# ---- 수학 유틸 ------------------------------------------------------------
def _softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(T, 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # 안정화
    x = np.clip(x, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-x))

def _to_onehot(y: np.ndarray, K: int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((y.shape[0], K), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

# ---- Temperature Scaling (multi-class) -----------------------------------
def _fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter=200, lr=0.01) -> float:
    """
    안정된 1D GD (finite-diff 기반)로 NLL 최소화. 반환 T>0.
    logits: (N,K)
    """
    logits = np.asarray(logits, dtype=np.float64)
    N, K = logits.shape
    y_oh = _to_onehot(y_true, K)
    T = 1.0
    for _ in range(max_iter):
        P = _softmax(logits, T)  # (N,K)
        nll = -np.mean(np.sum(y_oh * np.log(P + 1e-12), axis=1))
        # 중앙차분 근사 gradient (수치적으로 안정)
        eps = max(1e-4, 1e-3 * T)
        Pp = _softmax(logits, T + eps)
        Pm = _softmax(logits, max(1e-8, T - eps))
        nll_p = -np.mean(np.sum(y_oh * np.log(Pp + 1e-12), axis=1))
        nll_m = -np.mean(np.sum(y_oh * np.log(Pm + 1e-12), axis=1))
        g = (nll_p - nll_m) / (2 * eps)
        T_new = max(1e-3, T - lr * g)
        if abs(T_new - T) < 1e-6:
            T = T_new
            break
        T = T_new
    return float(T)

# ---- Platt Scaling (binary) ----------------------------------------------
def _fit_platt(scores: np.ndarray, y_true: np.ndarray, max_iter=200, lr=0.01) -> tuple[float, float]:
    A, B = 1.0, 0.0
    y = y_true.astype(np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    for _ in range(max_iter):
        s = A * scores + B
        p = _sigmoid(s)
        # gradient of BCE
        grad = (p - y)
        dA = np.mean(grad * scores)
        dB = np.mean(grad)
        A -= lr * dA
        B -= lr * dB
    return float(A), float(B)

# ---- 저장/로드 ------------------------------------------------------------
def load(symbol: str, strategy: str, model: str) -> Optional[dict]:
    path = _calib_path(symbol, strategy, model)
    if not os.path.exists(path):
        return None
    cfg = _get_CALIB()
    timeout = float(cfg.get("lock_timeout", 5.0))
    try:
        with _SimpleFileLock(path, timeout=timeout):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        # 간단 검증
        if not isinstance(obj, dict):
            return None
        if "method" not in obj:
            return None
        return obj
    except Exception:
        return None

def save(params: dict, symbol: str, strategy: str, model: str) -> None:
    path = _calib_path(symbol, strategy, model)
    cfg = _get_CALIB()
    timeout = float(cfg.get("lock_timeout", 5.0))
    meta = {
        "meta": {
            "symbol": symbol, "strategy": strategy, "model": model,
            "num_classes": int(params.get("num_classes", -1)),
            "created_at": time.time()
        },
        "params": params
    }
    try:
        with _SimpleFileLock(path, timeout=timeout):
            _atomic_write_json(path, meta)
    except Exception:
        # best-effort: try direct write
        try:
            _atomic_write_json(path, meta)
        except Exception:
            pass

# ---- 외부 API -------------------------------------------------------------
def fit_and_save(y_true: np.ndarray,
                 y_pred_proba_or_logits: np.ndarray,
                 meta: dict) -> Optional[dict]:
    """
    meta = {"symbol":.., "strategy":.., "model":.., "is_logits":bool (optional)}
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

    if y_true.ndim != 1 or y_true.shape[0] != X.shape[0]:
        return None

    if X.ndim == 1:
        # binary -> Platt
        if y_true.shape[0] < int(cfg.get("min_samples", 200)):
            return None
        A, B = _fit_platt(X, y_true)
        params = {"method": "platt", "A": A, "B": B, "ver": 1, "num_classes": 2}
        save(params, symbol, strategy, model)
        return params

    # multiclass
    if y_true.shape[0] < int(cfg.get("min_samples", 200)):
        return None

    if is_logits:
        logits = X
    else:
        P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
        # avoid log(0)
        logits = np.log(P + 1e-12)

    T = _fit_temperature(logits, y_true)
    params = {"method": "temperature", "T": T, "ver": 1, "num_classes": int(logits.shape[1])}
    save(params, symbol, strategy, model)
    return params

def _normalize_meta_kwargs(symbol, strategy, model, model_meta, meta):
    sym = (meta or {}).get("symbol", None) if isinstance(meta, dict) else None
    strat = (meta or {}).get("strategy", None) if isinstance(meta, dict) else None
    mdl = (meta or {}).get("model", None) if isinstance(meta, dict) else None

    if isinstance(model_meta, dict):
        sym = sym or model_meta.get("symbol")
        strat = strat or model_meta.get("strategy")
        mdl = mdl or model_meta.get("model")

    sym = symbol or sym or "UNK"
    strat = strategy or strat or "UNK"
    mdl = model or mdl or (model_meta.get("model") if isinstance(model_meta, dict) else "UNK")
    return sym, strat, mdl

def apply_calibration(raw_prob_or_scores: np.ndarray,
                      meta: dict = None,
                      *,
                      symbol: str = None, strategy: str = None, regime: str = None,
                      model_meta: dict = None, model: str = None,
                      is_logits: Optional[bool] = None) -> np.ndarray:
    cfg = _get_CALIB()
    X = np.asarray(raw_prob_or_scores, dtype=np.float64)

    # 전역 OFF → 안정화만
    if not cfg.get("enabled", False):
        if X.ndim == 2:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            return P
        return X

    sym, strat, mdl = _normalize_meta_kwargs(symbol, strategy, model, model_meta, meta)
    loaded = load(sym, strat, mdl)
    params = (loaded.get("params") if isinstance(loaded, dict) else loaded) if loaded else None

    if params is None:
        if X.ndim == 2:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            return P
        return X

    method = params.get("method", "temperature")

    if is_logits is None:
        is_logits = bool((meta or {}).get("is_logits")) if isinstance(meta, dict) else False

    if method == "temperature":
        if X.ndim == 1:
            return X
        T = max(1e-3, float(params.get("T", 1.0)))
        if is_logits:
            P = _softmax(X, T)
        else:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            logits = np.log(P + 1e-12)
            P = _softmax(logits, T)
        return P

    elif method == "platt":
        if X.ndim != 1:
            P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            return P
        A = float(params.get("A", 1.0)); B = float(params.get("B", 0.0))
        p = _sigmoid(A * X + B)
        return p

    # unknown method -> stabilize
    if X.ndim == 2:
        P = np.clip(X, cfg["clip_eps"], 1.0 - cfg["clip_eps"])
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        return P
    return X

def get_calibration_version() -> str:
    cfg = _get_CALIB()
    if not cfg.get("enabled", False):
        return "off"
    m = str(cfg.get("method", "temperature")).lower()
    return f"{m}@1"

def expected_calibration_error(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
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

# ---- train.py 호환용 얇은 래퍼 (best-effort: 파일 기반 학습 데이터 찾아 적용) ----
def learn_and_save(symbol: str, strategy: str, model_name: str):
    """
    기본적으로 외부 검증 예측 데이터를 자동으로 찾는 로직은 제공하지 않음.
    여기서는 /persistent/calib_sources/{symbol}_{strategy}_{model}.json 형식의
    검증 결과 파일이 있으면 읽어 fit_and_save를 수행하도록 시도함.
    """
    try:
        src_dir = "/persistent/calib_sources"
        fn = f"{symbol}_{strategy}_{model_name}.json".replace("/", "_")
        path = os.path.join(src_dir, fn)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        y_true = np.asarray(obj.get("y_true", []))
        y_scores = np.asarray(obj.get("y_scores", []))
        is_logits = bool(obj.get("is_logits", False))
        if y_true.size == 0 or y_scores.size == 0:
            return None
        meta = {"symbol": symbol, "strategy": strategy, "model": model_name, "is_logits": is_logits}
        return fit_and_save(y_true, y_scores, meta)
    except Exception:
        return None

def learn_and_save_from_checkpoint(symbol: str, strategy: str, model_name: str):
    return learn_and_save(symbol, strategy, model_name)
