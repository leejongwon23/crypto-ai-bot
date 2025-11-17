import sitecustomize
import os, sys, json, datetime, pytz, random, time, tempfile, shutil, csv, glob, inspect, threading
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import gc
from sklearn.preprocessing import MinMaxScaler
from logger import (
    log_prediction,
    extract_candle_returns,
    make_return_histogram,
)
# =========================================================
# 🔐 쓰기 가능한 루트 디렉터리
# =========================================================
PERSISTENT_ROOT = os.getenv("PERSISTENT_DIR", "/persistent")
os.makedirs(PERSISTENT_ROOT, exist_ok=True)

def _p(*parts: str) -> str:
    return os.path.join(PERSISTENT_ROOT, *parts)

# 경로 고정
from data.utils import get_kline_by_strategy, compute_features

__all__ = [
    "predict", "is_predict_gate_open", "open_predict_gate", "close_predict_gate",
    "run_evaluation_once", "run_evaluation_loop"
]

# =========================================================
# ✅ 부팅 시 게이트 자동 오픈
# =========================================================
def _ensure_gate_open_on_boot():
    try:
        block_path = _p("predict.block")
        run_dir = _p("run")
        os.makedirs(run_dir, exist_ok=True)
        gate_file = _p("run", "predict_gate.json")

        if os.path.exists(block_path):
            os.remove(block_path)

        state = {}
        if os.path.exists(gate_file):
            with open(gate_file, "r", encoding="utf-8") as f:
                state = json.load(f)

        if not state.get("open", False):
            with open(gate_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "open": True,
                        "opened_at": datetime.datetime.now(
                            pytz.timezone("Asia/Seoul")
                        ).isoformat(),
                        "note": "auto_open_on_boot"
                    },
                    f,
                    ensure_ascii=False
                )
            print("[AUTO] 예측 게이트 자동 오픈 완료")
    except Exception as e:
        print(f"[AUTO-GATE ERROR] 게이트 자동 오픈 실패: {e}")

_ensure_gate_open_on_boot()

# =========================================================
# 공용 메모리 정리
# =========================================================
def _safe_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _release_memory(*objs):
    try:
        for o in objs:
            try:
                del o
            except Exception:
                pass
    finally:
        gc.collect()
        _safe_empty_cache()

# =========================================================
# 런타임 경로/게이트
# =========================================================
RUN_DIR = _p("run")
os.makedirs(RUN_DIR, exist_ok=True)
PREDICT_GATE = os.path.join(RUN_DIR, "predict_gate.json")
PREDICT_BLOCK = _p("predict.block")
GROUP_ACTIVE = os.path.join(RUN_DIR, "group_predict.active")
GROUP_ACTIVE_STALE_SEC = int(os.getenv("GROUP_ACTIVE_STALE_SEC", "600"))

def _clean(s: str) -> str:
    s = (s or "None")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def _lock_path_for(symbol: str, strategy: str) -> str:
    return os.path.join(RUN_DIR, f"predict_{_clean(symbol)}_{_clean(strategy)}.lock")

PREDICT_LOCK_TTL = int(os.getenv("PREDICT_LOCK_TTL", "600"))
PREDICT_LOCK_STALE_TRAIN_SEC = int(os.getenv("PREDICT_LOCK_STALE_TRAIN_SEC", "600"))

def _now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def is_predict_gate_open():
    try:
        if os.getenv("FORCE_PREDICT_CLOSE", "0") == "1":
            return False
        if os.path.exists(PREDICT_BLOCK):
            return False
        if os.path.exists(PREDICT_GATE):
            with open(PREDICT_GATE, "r", encoding="utf-8") as f:
                o = json.load(f)
            return bool(o.get("open", True))
        return True
    except Exception:
        return True

def _bypass_gate_for_source(source: str) -> bool:
    s = str(source or "")
    if "그룹직후" in s:
        return True
    bl = os.getenv("PREDICT_GATE_BYPASS_SOURCES", "")
    return any(t and t in s for t in [x.strip() for x in bl.split(",") if x.strip()])

def _group_active() -> bool:
    """
    그룹 학습 중인지 여부만 확인한다.
    - train.py에서 GROUP_ACTIVE 파일을 만들면 -> True (학습 중)
    - train.py에서 파일을 지우기 전까지는 계속 True 유지
    """
    try:
        return os.path.exists(GROUP_ACTIVE)
    except Exception:
        return False
def open_predict_gate(note=""):
    try:
        with open(PREDICT_GATE, "w", encoding="utf-8") as f:
            json.dump(
                {"open": True, "opened_at": _now_kst().isoformat(), "note": note},
                f,
                ensure_ascii=False
            )
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        if os.path.exists(PREDICT_BLOCK):
            try:
                os.remove(PREDICT_BLOCK)
            except Exception:
                pass
    except Exception:
        pass

def close_predict_gate(note=""):
    try:
        with open(PREDICT_GATE, "w", encoding="utf-8") as f:
            json.dump(
                {"open": False, "closed_at": _now_kst().isoformat(), "note": note},
                f,
                ensure_ascii=False
            )
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            with open(PREDICT_BLOCK, "a") as bf:
                try:
                    bf.flush()
                    os.fsync(bf.fileno())
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

def _is_stale_lock(path: str, ttl_sec: int) -> bool:
    try:
        if not os.path.exists(path):
            return False
        mtime = os.path.getmtime(path)
        return (time.time() - float(mtime)) > max(30, int(ttl_sec))
    except Exception:
        return False

def _clear_stale_lock(path: str, ttl_sec: int, tag: str = ""):
    try:
        if os.path.exists(path) and _is_stale_lock(path, ttl_sec):
            os.remove(path)
            print(f"[LOCK] stale predict lock removed ({os.path.basename(path)}) > {ttl_sec}s {tag}")
    except Exception:
        pass

def _acquire_predict_lock(path: str):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "ts": _now_kst().isoformat()}, ensure_ascii=False))
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

def _release_predict_lock(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

PREDICT_HEARTBEAT_SEC = int(os.getenv("PREDICT_HEARTBEAT_SEC", "3"))

def _predict_hb_loop(stop_evt: threading.Event, tag: str, lock_path: str):
    last_note = ""
    while not stop_evt.is_set():
        try:
            gate = "open" if is_predict_gate_open() else "closed"
            lock = os.path.exists(lock_path)
            note = f"[HB] predict alive ({tag}) gate={gate} lock={'1' if lock else '0'} ts={_now_kst().strftime('%H:%M:%S')}"
            if note != last_note:
                print(note)
                last_note = note
        except Exception:
            pass
        hb = int(os.getenv('PREDICT_HEARTBEAT_SEC', '3'))
        stop_evt.wait(max(1, hb))

# =========================================================
# 외부 컴포넌트 폴백
# =========================================================
try:
    from window_optimizer import find_best_windows
except Exception:
    try:
        from window_optimizer import find_best_window
    except Exception:
        find_best_window = None
    def find_best_windows(symbol, strategy):
        try:
            best = int(find_best_window(symbol, strategy, window_list=[10, 20, 30, 40, 60], group_id=None)) if callable(find_best_window) else 60
        except Exception:
            best = 60
        return [best, best, best]

try:
    from regime_detector import detect_regime
except Exception:
    def detect_regime(symbol, strategy, now=None): return "unknown"

try:
    from calibration import apply_calibration, get_calibration_version
except Exception:
    def apply_calibration(probs, *, symbol=None, strategy=None, regime=None, model_meta=None): return probs
    def get_calibration_version(): return "none"

STRICT_SAME_BOUNDS = os.getenv("STRICT_SAME_BOUNDS", "0") == "1"

PREDICT_MODEL_LOADER_TTL = int(os.getenv("MODEL_LOADER_TTL", "600"))
try:
    from model_weight_loader import load_model_cached as _raw_load_model
except Exception:
    _raw_load_model = None
if _raw_load_model is None:
    try:
        from model_io import load_model as _raw_load_model
    except Exception:
        _raw_load_model = None

def load_model_any(path, model=None, **kwargs):
    ttl = kwargs.pop("ttl_sec", PREDICT_MODEL_LOADER_TTL)
    if _raw_load_model is not None:
        try:
            sig = inspect.signature(_raw_load_model)
            params = list(sig.parameters.values())
            if len(params) >= 2:
                try:
                    return _raw_load_model(path, model, ttl_sec=ttl, **kwargs)
                except TypeError:
                    return _raw_load_model(path, model, **kwargs)
            else:
                return _raw_load_model(path)
        except Exception as e:
            print(f"[load_model_any: primary loader failed] {e}")

    try:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and model is not None:
            try:
                model.load_state_dict(sd, strict=False)
                model.eval()
                return model
            except Exception as e:
                print(f"[load_model_any: state_dict load failed] {e}")
                return sd
        return sd
    except Exception as e:
        print(f"[load_model_any: torch.load 실패, 경로={path}] {e}")
        return model

from logger import log_prediction, update_model_success, PREDICTION_HEADERS, ensure_prediction_log_exists
from failure_db import insert_failure_record, ensure_failure_db
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
try:
    from model.base_model import get_model
except Exception:
    from base_model import get_model
from config import (
    get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups,
    get_class_return_range, class_to_expected_return, get_CLASS_BIN, get_PUBLISH_RUNTIME
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 모델 경로 다중 스캔 설정
# =========================================================
_DEFAULT_MODEL_ROOTS = [
    os.getenv("MODEL_DIR", _p("models")),
    _p("models"),
    "./models",
    "/mnt/data/models",
    "/workspace/models",
    "/data/models",
]
MODEL_DIRS = []
for p in _DEFAULT_MODEL_ROOTS:
    if isinstance(p, str) and p not in MODEL_DIRS:
        MODEL_DIRS.append(p)
MODEL_DIR = MODEL_DIRS[0] if MODEL_DIRS else _p("models")

PREDICTION_LOG_PATH = _p("prediction_log.csv")
NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

MIN_RET_THRESHOLD = float(os.getenv("PREDICT_MIN_RETURN", "0.01"))
ABSTAIN_PROB_MIN = float(os.getenv("ABSTAIN_PROB_MIN", "0.25"))
PREDICT_SOFT_ABORT = int(os.getenv("PREDICT_SOFT_ABORT", "1"))

PREDICT_WINDOW_ENSEMBLE = os.getenv("PREDICT_WINDOW_ENSEMBLE", "mean_var").lower()
ENSEMBLE_VAR_GAMMA = float(os.getenv("ENSEMBLE_VAR_GAMMA", "1.0"))
ADJUST_WITH_DIVERSITY = os.getenv("ADJUST_WITH_DIVERSITY", "0") == "1"

EXP_STATE = _p("logs", "meta_explore_state.json")
EXP_EPS = float(os.getenv("EXPLORE_EPS_BASE", "0.15"))
EXP_DEC_MIN = float(os.getenv("EXPLORE_DECAY_MIN", "120"))
EXP_NEAR = float(os.getenv("EXPLORE_NEAR_GAP", "0.07"))
EXP_GAMMA = float(os.getenv("EXPLORE_GAMMA", "0.05"))

RG_ENABLE = os.getenv("RG_ENABLE", "1") == "1"
RG_VOL_MULT = float(os.getenv("RG_VOL_MULT", "3.0"))
RG_LOOKBACK_SHORT = int(os.getenv("RG_LOOKBACK_SHORT", "48"))
RG_LOOKBACK_MID = int(os.getenv("RG_LOOKBACK_MID", "96"))
RG_LOOKBACK_LONG = int(os.getenv("RG_LOOKBACK_LONG", "336"))
RG_VOL_METHOD = os.getenv("RG_VOL_METHOD", "std").lower()
RG_MIN_ABS_MID_FOR_VOLCHECK = float(os.getenv("RG_MIN_ABS_MID_FOR_VOLCHECK", "0.004"))

FORCE_PUBLISH_ON_ABSTAIN = os.getenv("FORCE_PUBLISH_ON_ABSTAIN", "0") == "1"

# =========================================================
# 📍 (추가) 시장 컨텍스트 / 최근 성공률
# =========================================================
def _load_market_context(symbol: str, strategy: str) -> dict:
    """
    외부에서 실시간 시장요약을 뿌려주는 경우를 위한 약한 훅.
    없으면 기본 0 들고 간다.
    """
    base = {"dominance": 0.0, "volume_bias": 0.0, "risk": 0.0}
    try:
        ctx_path = _p("run", "market_ctx", f"{symbol.upper()}__{strategy}.json")
        if os.path.exists(ctx_path):
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in ("dominance", "volume_bias", "risk"):
                if k in data:
                    base[k] = float(data[k])
    except Exception:
        pass
    base["risk"] = float(base.get("risk", 0.0))
    return base

def _get_recent_success_weight(symbol: str, strategy: str) -> float:
    """
    logger 쪽에 최근 성공률 함수가 있으면 그걸 쓰고,
    없으면 0.5로.
    반환은 0.3~1.0 사이로 잘라서 너무 안 흔들리게.
    """
    base = 0.5
    try:
        import logger as _lg
        if hasattr(_lg, "get_recent_success_rate"):
            sr = float(_lg.get_recent_success_rate(symbol, strategy, window=10))
            base = max(0.0, min(1.0, sr))
    except Exception:
        pass
    return 0.3 + 0.7 * base

# =========================================================
# 유틸리티
# =========================================================
def _norm_model_type(mt: str) -> str:
    s = str(mt or "").lower()
    if "transformer" in s:
        return "TRANSFORMER"
    if "cnn" in s:
        return "CNN_LSTM"
    return "LSTM"

def _sanitize_range(lo: float, hi: float) -> tuple[float, float]:
    try:
        lo_f, hi_f = float(lo), float(hi)
        if abs(lo_f) > 1 or abs(hi_f) > 1:
            lo_f /= 100.0
            hi_f /= 100.0
        return lo_f, hi_f
    except Exception:
        return float(lo or 0.0), float(hi or 0.0)

def _ranges_from_meta(meta):
    try:
        cr = meta.get("class_ranges", None)
        if isinstance(cr, list) and len(cr) >= 2 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in cr):
            return [(float(a), float(b)) for a, b in cr]
    except Exception:
        pass
    return None

def _validate_meta_class_ranges(meta, *, num_classes: int, symbol: str, strategy: str):
    raw = _ranges_from_meta(meta) if isinstance(meta, dict) else None
    if not raw:
        return None
    cr = [_sanitize_range(a, b) for (a, b) in raw]
    if len(cr) < max(2, min(num_classes, 3)):
        return None
    uniq = set((round(lo, 6), round(hi, 6)) for (lo, hi) in cr)
    if len(uniq) <= 1:
        print(f"[메타무시] {symbol}-{strategy} meta.class_ranges 전 클래스 동일 → config 사용")
        return None
    if any((hi - lo) <= 0 for (lo, hi) in cr):
        print(f"[메타무시] {symbol}-{strategy} meta.class_ranges hi<=lo 존재 → config 사용")
        return None
    mids = [abs((lo + hi) / 2.0) for (lo, hi) in cr]
    widths = [abs(hi - lo) for (lo, hi) in cr]
    if sum(w > 0.4 for w in widths) >= max(2, len(widths) // 2):
        print(f"[메타무시] {symbol}-{strategy} meta.class_ranges 폭 과다(과반) → config 사용")
        return None
    if sum(m > 0.25 for m in mids) >= max(2, len(mids) // 2):
        print(f"[메타무시] {symbol}-{strategy} meta.class_ranges 중앙값 과다(과반) → config 사용")
        return None
    return cr

def _class_range_by_meta_or_cfg(cls_id: int, meta, symbol: str, strategy: str):
    """
    클래스별 수익률 구간을 결정할 때, 학습 시점의 동적 경계까지 반영하도록 수정된 버전.
    우선순위:
      1️⃣ 모델(meta.json)에 class_ranges가 있으면 그걸 최우선으로 사용
      2️⃣ labels.py가 생성한 동적 경계파일(/persistent/label_edges/...)을 사용
      3️⃣ 위 두 개 다 없을 경우 config.py 기본 구간으로 fallback
    """
    cid = int(cls_id)

    # 1️⃣ 모델 메타정보에서 class_ranges가 있으면 최우선 사용
    try:
        if isinstance(meta, dict):
            cr = meta.get("class_ranges")
            if isinstance(cr, list) and cid < len(cr):
                lo, hi = cr[cid]
                return _sanitize_range(lo, hi)
    except Exception as e:
        print(f"[class_range/meta 예외] {e}")

    # 2️⃣ labels.py가 만든 동적 라벨 경계 사용
    try:
        key = f"{symbol.strip().upper()}__{strategy.strip()}"
        lp = os.path.join(PERSISTENT_ROOT, "label_edges", f"{key}.json")
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                data = json.load(f)
            edges = data.get("edges") or []
            if edges and (cid + 1) < len(edges):
                lo = float(edges[cid])
                hi = float(edges[cid + 1])
                return _sanitize_range(lo, hi)
    except Exception as e:
        print(f"[class_range/label_edges 예외] {e}")

    # 3️⃣ 마지막 fallback: config.py 기본 구간
    try:
        lo, hi = get_class_return_range(cid, symbol, strategy)
        return _sanitize_range(lo, hi)
    except Exception as e:
        print(f"[class_range/config 예외] {e}")
        return (0.0, 0.0)

def _position_from_range(lo: float, hi: float) -> str:
    try:
        lo = float(lo)
        hi = float(hi)
        if hi <= 0 and lo < 0:
            return "short"
        if lo >= 0 and hi > 0:
            return "long"
        return "neutral"
    except Exception:
        return "neutral"

def _meets_minret_with_hint(lo: float, hi: float, allow_long: bool, allow_short: bool, thr: float) -> bool:
    try:
        lo = float(lo)
        hi = float(hi)
        thr = float(thr)
        long_ok = allow_long and (hi > 0.0) and (hi >= thr)
        short_ok = allow_short and (lo < 0.0) and ((-lo) >= thr)
        if allow_long and allow_short:
            return (hi >= thr) or ((-lo) >= thr)
        return long_ok or short_ok
    except Exception:
        return False

def _load_json(p, default):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(p, obj):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ✅ 원본에 있던 해시 유틸 복원
def _feature_hash(row):
    try:
        import hashlib
        if isinstance(row, torch.Tensor):
            arr = row.detach().cpu().flatten().numpy().astype(float)
        elif isinstance(row, np.ndarray):
            arr = row.flatten().astype(float)
        elif isinstance(row, (list, tuple)):
            arr = np.array(row, dtype=float).flatten()
        else:
            arr = np.array([float(row)], dtype=float)
        r = [round(float(x), 2) for x in arr]
        return hashlib.sha1(",".join(map(str, r)).encode()).hexdigest()
    except Exception:
        return "hash_error"

_KNOWN_EXTS = (".pt", ".ptz", ".safetensors")
def _stem(fn):
    for e in _KNOWN_EXTS:
        if fn.endswith(e):
            return fn[:-len(e)]
    return os.path.splitext(fn)[0]

def _resolve_meta_from_any_root(weight_abs: str) -> str | None:
    try:
        base = _stem(weight_abs)
        m1 = f"{base}.meta.json"
        if os.path.exists(m1):
            return m1
        target = os.path.basename(_stem(weight_abs))
        for root in MODEL_DIRS:
            pattern = os.path.join(root, "**", f"{target}.meta.json")
            cands = glob.glob(pattern, recursive=True)
            if cands:
                return cands[0]
        return None
    except Exception:
        return None

def _parse_group_cls_from_filename(path: str):
    try:
        base = os.path.basename(_stem(path))
        parts = base.split("_")
        grp = None
        cls = None
        for p in parts:
            if p.startswith("group") and p[5:].isdigit():
                grp = int(p[5:])
            if p.startswith("cls") and p[3:].isdigit():
                cls = int(p[3:])
        return grp, cls
    except Exception:
        return None, None

def _infer_group_id(symbol: str, strategy: str) -> int:
    try:
        group_map = {
            "BTCUSDT": 0, "ETHUSDT": 1, "XRPUSDT": 2, "SOLUSDT": 3,
            "ADAUSDT": 4, "DOGEUSDT": 5, "DOTUSDT": 6, "MATICUSDT": 7
        }
        gid = group_map.get((symbol or "").upper(), None)
        if gid is None:
            gid = int(abs(hash(f"{symbol}|{strategy}")) % 8)
        return max(0, min(7, int(gid)))
    except Exception:
        return 0

# =========================================================
# 유사도 기반 하이브리드
# =========================================================
from sklearn.metrics.pairwise import cosine_similarity

def _compute_similarity_class_probs(current_vec: np.ndarray,
                                    lib_vecs: np.ndarray | None,
                                    lib_labels: np.ndarray | None,
                                    num_classes: int,
                                    top_k: int = 200):
    try:
        if lib_vecs is None or lib_labels is None or len(lib_vecs) == 0:
            sim_probs = np.ones(num_classes, dtype=float) / float(num_classes)
            return sim_probs, {"m": 0, "w_sim": 0.0}
        lib_vecs = np.asarray(lib_vecs, dtype=float)
        lib_labels = np.asarray(lib_labels)
        d = lib_vecs.shape[1]
        v = np.asarray(current_vec, dtype=float).reshape(1, -1)
        if v.shape[1] != d:
            sim_probs = np.ones(num_classes, dtype=float) / float(num_classes)
            return sim_probs, {"m": 0, "w_sim": 0.0}

        sims = cosine_similarity(lib_vecs, v).ravel()
        k = int(min(max(10, top_k), len(sims)))
        idx = np.argsort(-sims)[:k]
        m = len(idx)
        if m < 20:
            w_sim = 0.2
        elif m < 80:
            w_sim = 0.5
        else:
            w_sim = 0.6

        s = sims[idx]
        s = s - s.min() + 1e-6
        w = s / (s.sum() + 1e-12)

        sim_probs = np.zeros(num_classes, dtype=float)
        for weight, lab in zip(w, lib_labels[idx]):
            try:
                ci = int(lab)
            except Exception:
                continue
            ci = max(0, min(num_classes - 1, ci))
            sim_probs[ci] += float(weight)
        sim_probs = np.nan_to_num(sim_probs, nan=0.0, posinf=0.0, neginf=0.0)
        sim_sum = sim_probs.sum()
        if sim_sum <= 0:
            sim_probs = np.ones(num_classes, dtype=float) / float(num_classes)
        else:
            sim_probs = sim_probs / sim_sum
        return sim_probs, {"m": m, "w_sim": w_sim}
    except Exception:
        sim_probs = np.ones(num_classes, dtype=float) / float(num_classes)
        return sim_probs, {"m": 0, "w_sim": 0.0}

# =========================================================
# 모델 탐색 (네 원본 그대로)
# =========================================================
def get_available_models(symbol, strategy):
    diag = {"symbol": symbol, "strategy": strategy, "roots": [], "found": []}
    try:
        results = []
        for root in MODEL_DIRS:
            root_info = {"root": root, "exists": os.path.isdir(root), "matches": 0}
            if not root_info["exists"]:
                diag["roots"].append(root_info)
                continue

            search_patterns = [
                os.path.join(root, f"{symbol}_{strategy}_*.*"),
                os.path.join(root, f"{symbol}_*{strategy}_*.*"),
                os.path.join(root, f"{symbol}_{strategy}_*_*.*"),
                os.path.join(root, symbol, strategy, "*"),
                os.path.join(root, symbol, f"{strategy}_*"),
                os.path.join(root, "**", f"{symbol}_{strategy}_*.*"),
            ]

            for pattern in search_patterns:
                for ext in [".pt", ".ptz", ".safetensors"]:
                    for w in glob.glob(f"{pattern}{ext}", recursive=True):
                        if not os.path.isfile(w):
                            continue
                        meta_path = _resolve_meta_from_any_root(w)
                        if not meta_path:
                            meta_tmp = {
                                "symbol": symbol, "strategy": strategy,
                                "group_id": _infer_group_id(symbol, strategy),
                                "input_size": FEATURE_INPUT_SIZE,
                                "num_classes": NUM_CLASSES,
                                "created_at": time.time(),
                            }
                            meta_path = _stem(w) + ".meta.json"
                            try:
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(meta_tmp, f, ensure_ascii=False, indent=2)
                            except Exception:
                                continue
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                meta = json.load(mf)
                        except Exception:
                            continue

                        gid = meta.get("group_id", _infer_group_id(symbol, strategy))
                        results.append({
                            "pt_abs": os.path.abspath(w),
                            "meta_path": os.path.abspath(meta_path),
                            "group_id": gid,
                            "root": root
                        })
                        root_info["matches"] += 1
            diag["roots"].append(root_info)

        if not results:
            for root in MODEL_DIRS:
                if not os.path.isdir(root):
                    continue
                cands = []
                for ext in [".pt", ".ptz", ".safetensors"]:
                    cands += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
                if not cands:
                    continue
                cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                for w in cands[:3]:
                    meta_path = _resolve_meta_from_any_root(w)
                    if not meta_path:
                        meta_tmp = {
                            "symbol": symbol, "strategy": strategy,
                            "group_id": _infer_group_id(symbol, strategy),
                            "input_size": FEATURE_INPUT_SIZE, "num_classes": NUM_CLASSES,
                            "created_at": time.time(),
                        }
                        meta_path = _stem(w) + ".meta.json"
                        try:
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(meta_tmp, f, ensure_ascii=False, indent=2)
                            except Exception:
                                continue
                    results.append({
                        "pt_abs": os.path.abspath(w),
                        "meta_path": os.path.abspath(meta_path),
                        "group_id": _infer_group_id(symbol, strategy),
                        "root": root
                    })

        seen, uniq = set(), []
        for r in results:
            if r["pt_abs"] not in seen:
                uniq.append(r)
                seen.add(r["pt_abs"])
        try:
            uniq.sort(key=lambda x: os.path.getmtime(x["pt_abs"]), reverse=True)
        except Exception:
            pass

        try:
            os.makedirs(_p("logs"), exist_ok=True)
            diag["found"] = [{"pt_abs": it["pt_abs"], "meta": it["meta_path"], "root": it["root"]}
                             for it in uniq]
            diag_path = _p("logs", "model_discovery_diag.json")
            payload = _load_json(diag_path, [])
            payload = payload[-200:] + [diag]
            _save_json(diag_path, payload)
        except Exception:
            pass

        if not uniq:
            print(f"[⚠️ 모델 탐색 실패] {symbol}-{strategy} → 모델 없음")
        else:
            print(f"[✅ 모델 탐색 성공] {symbol}-{strategy} → {len(uniq)}개 모델 발견 (첫 root={uniq[0].get('root')})")

        return uniq

    except Exception as e:
        print(f"[get_available_models 오류] {e}")
        try:
            os.makedirs(_p("logs"), exist_ok=True)
            diag["error"] = str(e)
            payload = _load_json(_p("logs", "model_discovery_diag.json"), [])
            payload = payload[-200:] + [diag]
            _save_json(_p("logs", "model_discovery_diag.json"), payload)
        except Exception:
            pass
        return []

# =========================================================
# 실패/보류 결과
# =========================================================
def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    t = _now_kst().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[predict] skip {symbol}-{strategy} :: {reason}")
    res = {"symbol": symbol, "strategy": strategy, "success": False, "reason": reason, "model": str(model_type or "unknown"), "rate": 0.0, "class": -1, "timestamp": t, "source": source, "predicted_class": -1, "label": -1}
    try:
        ensure_prediction_log_exists()
        log_prediction(symbol=symbol, strategy=strategy, direction="예측실패", entry_price=0, target_price=0, model=str(model_type or "unknown"), success=False, reason=reason, rate=0.0, expected_return=0.0, position="neutral", timestamp=t, return_value=0.0, volatility=True, source=source, predicted_class=-1, label=-1, class_return_min=0.0, class_return_max=0.0, class_return_text="")
    except Exception as e:
        print(f"[failed_result log_prediction 오류] {e}")
    try:
        if X_input is not None:
            insert_failure_record(res, feature_vector=np.array(X_input).flatten().tolist(), context="prediction")
    except Exception as e:
        print(f"[failed_result insert_failure_record 오류] {e}")
    return res

def _soft_abstain(symbol, strategy, *, reason, meta_choice="abstain", regime="unknown", X_last=None, group_id=None, df=None, source="보류"):
    try:
        ensure_prediction_log_exists()
        cur = float((df["close"].iloc[-1] if df is not None and len(df) else 0.0))
        note = {"reason": reason, "abstain_prob_min": float(ABSTAIN_PROB_MIN), "max_calib_prob": None, "meta_choice": meta_choice, "regime": regime}

        log_prediction(
            symbol=symbol, strategy=strategy, direction="예측보류",
            entry_price=cur, target_price=cur, model="meta", model_name=str(meta_choice),
            predicted_class=-1, label=-1, note=json.dumps(note, ensure_ascii=False),
            top_k=[], success=False, reason=reason, rate=0.0, expected_return=0.0,
            position="neutral", return_value=0.0, source=source, group_id=group_id,
            feature_vector=(torch.tensor(X_last, dtype=torch.float32).numpy() if X_last is not None else None),
            regime=regime, meta_choice="abstain", raw_prob=None, calib_prob=None,
            calib_ver=get_calibration_version(), class_return_min=0.0,
            class_return_max=0.0, class_return_text=""
        )

        log_prediction(
            symbol=symbol, strategy=strategy, direction="예측(보류)",
            entry_price=cur, target_price=cur, model="meta", model_name=str(meta_choice),
            predicted_class=-1, label=-1,
            note=json.dumps({"reason": reason, "summary": True}, ensure_ascii=False),
            top_k=[], success=False, reason=reason, rate=0.0, expected_return=0.0,
            position="neutral", return_value=0.0, source=source, group_id=group_id,
            feature_vector=None, regime=regime, meta_choice="abstain",
            raw_prob=None, calib_prob=None, calib_ver=get_calibration_version(),
            class_return_min=0.0, class_return_max=0.0, class_return_text=""
        )

    except Exception as e:
        print(f"[soft_abstain 예외] {e}")

    print(f"[predict] abstain {symbol}-{strategy} :: {reason}")
    return {
        "symbol": symbol, "strategy": strategy, "model": "meta", "class": -1,
        "expected_return": 0.0, "class_return_min": 0.0, "class_return_max": 0.0,
        "class_return_text": "", "position": "neutral",
        "timestamp": _now_kst().isoformat(), "source": source, "regime": regime,
        "reason": reason, "success": False, "predicted_class": -1, "label": -1
    }

# =========================================================
# 보조
# =========================================================
def _acquire_predict_lock_with_retry(path: str, max_wait_sec: int):
    deadline = time.time() + max(1, int(max_wait_sec))
    while time.time() < deadline:
        if _acquire_predict_lock(path):
            return True
        time.sleep(random.uniform(0.5, 2.0))
    return False

def _prep_lock_for_source(source: str):
    src = str(source or "")
    if "그룹직후" in src:
        try:
            return int(os.getenv("PREDICT_LOCK_WAIT_GROUP_SEC", "30"))
        except Exception:
            return 30
    return int(os.getenv("PREDICT_LOCK_WAIT_MAX_SEC", "15"))

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    s = pd.Series(arr, dtype=float)
    return s.ewm(span=span, adjust=False).mean().values

def _position_hint_from_market(df: pd.DataFrame) -> dict:
    try:
        close = df["close"].astype(float).values
        if close.size < 70:
            return {"allow_long": True, "allow_short": True, "ma_fast": None, "ma_slow": None, "slope": 0.0}
        ma_fast = _ema(close, 20)
        ma_slow = _ema(close, 60)
        y = close[-30:]
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0]) / (np.mean(y) + 1e-12)
        diff = float(ma_fast[-1] - ma_slow[-1]) / (close[-1] + 1e-12)
        strong_up = (diff > 0.0015) and (slope > 0)
        strong_dn = (diff < -0.0015) and (slope < 0)
        if strong_up and not strong_dn:
            return {"allow_long": True, "allow_short": False, "ma_fast": float(ma_fast[-1]), "ma_slow": float(ma_slow[-1]), "slope": float(slope)}
        if strong_dn and not strong_up:
            return {"allow_long": False, "allow_short": True, "ma_fast": float(ma_fast[-1]), "ma_slow": float(ma_slow[-1]), "slope": float(slope)}
        return {"allow_long": True, "allow_short": True, "ma_fast": float(ma_fast[-1]), "ma_slow": float(ma_slow[-1]), "slope": float(slope)}
    except Exception:
        return {"allow_long": True, "allow_short": True, "ma_fast": None, "ma_slow": None, "slope": 0.0}

def _recent_volatility(df: pd.DataFrame, strategy: str) -> float:
    try:
        if df is None or df.empty or "close" not in df.columns:
            return 0.0
        close = df["close"].astype(float).values
        if close.size < 10:
            return 0.0
        if strategy == "단기":
            lb = RG_LOOKBACK_SHORT
        elif strategy == "중기":
            lb = RG_LOOKBACK_MID
        else:
            lb = RG_LOOKBACK_LONG
        lb = max(10, min(int(lb), len(close) - 1))
        ret = pd.Series(close).pct_change().dropna().values[-lb:]
        if ret.size == 0:
            return 0.0
        if RG_VOL_METHOD == "mad":
            vol = float(np.mean(np.abs(ret - np.median(ret))) * 1.2533)
        else:
            vol = float(np.std(ret))
        return max(0.0, vol)
    except Exception:
        return 0.0

def _reality_guard_check(df, strategy, hint, lo_sel, hi_sel, exp_mid) -> tuple[bool, str]:
    try:
        pos = _position_from_range(lo_sel, hi_sel)
        if pos == "long" and not bool(hint.get("allow_long", True)):
            return True, "warn_position_conflict_long"
        if pos == "short" and not bool(hint.get("allow_short", True)):
            return True, "warn_position_conflict_short"
        if abs(exp_mid) >= RG_MIN_ABS_MID_FOR_VOLCHECK:
            vol = _recent_volatility(df, strategy)
            if vol > 0:
                if abs(exp_mid) > RG_VOL_MULT * vol:
                    return False, f"reality_guard_overclaim(vol={vol:.4f}, mid={exp_mid:.4f})"
        return True, "ok"
    except Exception as e:
        return True, f"rg_exception:{e}"

def _exit_guard_check(lo_sel: float, hi_sel: float, exp_ret: float) -> tuple[bool, str]:
    try:
        pub_conf = get_PUBLISH_RUNTIME()
        min_er = float(pub_conf.get("min_expected_return", 0.01))
        if abs(float(exp_ret)) < (min_er * 0.5):
            return False, f"exit_guard_min_expected_return(mid={float(exp_ret):.4f}, min={min_er:.4f})"
        return True, "ok"
    except Exception as e:
        return True, f"exit_guard_exception:{e}"

# =========================================================
# 윈도우 앙상블
# =========================================================
def _combine_windows(calib_stack: np.ndarray, raw_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_c = calib_stack.mean(axis=0)
    mean_r = raw_stack.mean(axis=0)
    if PREDICT_WINDOW_ENSEMBLE == "mean":
        cc = mean_c
        rr = mean_r
    else:
        var_c = calib_stack.var(axis=0)
        var_r = raw_stack.var(axis=0)
        cc = 0.5 * mean_c + 0.5 * (mean_c / (1.0 + ENSEMBLE_VAR_GAMMA * var_c))
        rr = 0.5 * mean_r + 0.5 * (mean_r / (1.0 + ENSEMBLE_VAR_GAMMA * var_r))
    cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
    rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
    cc = (cc + 1e-9)
    rr = (rr + 1e-9)
    cc = cc / (cc.sum() + 1e-12)
    rr = rr / (rr.sum() + 1e-12)
    return cc.astype(float), rr.astype(float)

# =========================================================
# 모델 추론 루틴 (네 원본 방식 + 유지)
# =========================================================
def get_model_predictions(symbol, strategy, models, df, feat_scaled, window_list, recent_freq, regime="unknown"):
    outs = []
    allpreds = []
    for info in models:
        try:
            model_path = info.get("pt_abs")
            meta_path = info.get("meta_path")
            if not model_path or not meta_path:
                continue
            if not os.path.exists(model_path):
                print(f"[⚠️ 모델 파일 소실] {model_path}")
                continue
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)

            fname_grp, _fname_cls = _parse_group_cls_from_filename(model_path)
            if fname_grp is not None:
                meta["group_id"] = int(fname_grp)
            elif "group_id" not in meta or not str(meta.get("group_id")).isdigit():
                meta["group_id"] = _infer_group_id(symbol, strategy)

            if STRICT_SAME_BOUNDS:
                cr_meta = _ranges_from_meta(meta)
                if not (cr_meta and len(cr_meta) >= 2):
                    print(f"[STRICT] no class_ranges in meta → {os.path.basename(model_path)} (skip)")
                    continue

            mtype_raw = meta.get("model", "lstm")
            gid = meta.get("group_id", 0)
            inp_size = int(meta.get("input_size", feat_scaled.shape[1]))
            cr_meta = _ranges_from_meta(meta)
            num_cls = int(meta.get("num_classes", (len(cr_meta) if cr_meta else NUM_CLASSES)))

            preds_c_list, preds_r_list = [], []
            used_windows = []
            for win in list(dict.fromkeys([int(w) for w in window_list if int(w) > 0])):
                if feat_scaled.shape[0] < win:
                    preds_c_list = []
                    preds_r_list = []
                    used_windows = []
                    break
                seq = feat_scaled[-win:]
                if seq.shape[1] < inp_size:
                    seq = np.pad(seq, ((0, 0), (0, inp_size - seq.shape[1])), mode="constant")
                elif seq.shape[1] > inp_size:
                    seq = seq[:, :inp_size]
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

                model = get_model(mtype_raw, input_size=inp_size, output_size=num_cls)
                loaded = load_model_any(model_path, model, ttl_sec=PREDICT_MODEL_LOADER_TTL)
                if loaded is None:
                    try:
                        obj = torch.load(model_path, map_location="cpu")
                        if isinstance(obj, dict):
                            try:
                                model.load_state_dict(obj, strict=False)
                                model.eval()
                                loaded = model
                            except Exception:
                                loaded = None
                        else:
                            loaded = obj
                    except Exception:
                        loaded = None
                if isinstance(loaded, dict) and model is not None:
                    try:
                        model.load_state_dict(loaded, strict=False)
                        model.eval()
                    except Exception:
                        print(f"[모델 주입 실패] {model_path}")
                        preds_c_list = []
                        preds_r_list = []
                        break
                elif loaded is None:
                    print(f"[⚠️ 모델 로딩 실패] {model_path}")
                    preds_c_list = []
                    preds_r_list = []
                    break
                else:
                    if hasattr(loaded, "eval"):
                        model = loaded
                model.to(DEVICE)
                model.eval()

                with torch.no_grad():
                    logits = model(x.to(DEVICE))
                    logits = torch.nan_to_num(logits)
                    probs = F.softmax(logits, dim=1)
                    probs = torch.nan_to_num(probs).squeeze().cpu().numpy()

                probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) + 1e-9
                probs = probs / (probs.sum() + 1e-12)
                cprobs = apply_calibration(probs, symbol=symbol, strategy=strategy, regime=regime, model_meta=meta).astype(float)
                cprobs = np.nan_to_num(cprobs, nan=0.0, posinf=0.0, neginf=0.0) + 1e-9
                cprobs = cprobs / (cprobs.sum() + 1e-12)
                preds_c_list.append(cprobs)
                preds_r_list.append(probs)
                used_windows.append(int(win))

                try:
                    del x, logits
                except Exception:
                    pass
                try:
                    model.cpu()
                except Exception:
                    pass
                try:
                    del model
                except Exception:
                    pass
                _safe_empty_cache()
                gc.collect()

            if not preds_c_list:
                print(f"[모델 스킵] 예측값 없음 → {os.path.basename(model_path)}")
                continue

            calib_stack = np.vstack(preds_c_list)
            raw_stack = np.vstack(preds_r_list)
            comb_c, comb_r = _combine_windows(calib_stack, raw_stack)

            outs.append({
                "raw_probs": comb_r, "calib_probs": comb_c, "predicted_class": int(np.argmax(comb_c)),
                "group_id": gid, "model_type": _norm_model_type(mtype_raw), "model_path": model_path,
                "val_f1": None, "symbol": symbol, "strategy": strategy, "meta": meta,
                "window_ensemble": {"mode": PREDICT_WINDOW_ENSEMBLE, "gamma": ENSEMBLE_VAR_GAMMA, "wins": used_windows}
            })
            entry_price = df["close"].iloc[-1]
            allpreds.append({
                "class": int(np.argmax(comb_c)), "probs": comb_c, "entry_price": float(entry_price),
                "num_classes": num_cls, "group_id": gid, "model_name": _norm_model_type(mtype_raw),
                "model_symbol": symbol, "symbol": symbol, "strategy": strategy
            })

            try:
                del preds_c_list, preds_r_list, calib_stack, raw_stack
            except Exception:
                pass
            _safe_empty_cache()
            gc.collect()

        except Exception as e:
            print(f"[❌ 모델 예측 실패] {info} → {e}")
            _safe_empty_cache()
            gc.collect()
            continue
    return outs, allpreds

# =========================================================
# 보수적 강제발행 선택기
# =========================================================
def _choose_conservative_prediction(outs, symbol, strategy, allow_long, allow_short, min_thr):
    try:
        best = (None, None, -1.0)
        for m in outs:
            probs = m.get("hybrid_probs", m.get("adjusted_probs", m.get("calib_probs")))
            if probs is None:
                continue
            probs = np.asarray(probs, dtype=float)
            mask = np.zeros_like(probs, dtype=float)
            for ci in range(len(probs)):
                try:
                    lo, hi = _class_range_by_meta_or_cfg(ci, m.get("meta"), symbol, strategy)
                    if _meets_minret_with_hint(lo, hi, allow_long, allow_short, min_thr):
                        mask[ci] = 1.0
                except Exception:
                    pass
            cand = probs * mask
            if cand.sum() > 0:
                cand = cand / cand.sum()
                ci = int(np.argmax(cand))
                sc = float(cand[ci])
                if sc > best[2]:
                    best = (m, ci, sc)
        if best[0] is not None:
            return best[0], best[1]

        best = (None, None, -1.0)
        for m in outs:
            probs = m.get("hybrid_probs", m.get("adjusted_probs", m.get("calib_probs")))
            if probs is None:
                continue
            for ci in range(len(probs)):
                try:
                    lo, hi = _class_range_by_meta_or_cfg(ci, m.get("meta"), symbol, strategy)
                    pos = _position_from_range(lo, hi)
                    hint_bonus = 1.0 if ((pos == "long" and allow_long) or (pos == "short" and allow_short) or pos == "neutral") else 0.5
                    mid = abs((float(lo) + float(hi)) / 2.0) * hint_bonus
                    if mid > best[2]:
                        best = (m, ci, mid)
                except Exception:
                    continue
        return (best[0], best[1]) if best[0] is not None else (None, None)
    except Exception:
        return (None, None)

# =========================================================
# 소프트 가드
# =========================================================
def _apply_soft_guard(df, strategy, outs, chosen, final_cls, allow_long, allow_short, min_thr):
    """
    🟢 가드 비활성화 버전
    - 예측이 정상이라는 가정하에 소프트가드는 통과만 시킨다.
    - log용 메시지만 남긴다.
    """
    return "ok", chosen, final_cls, "soft_guard_disabled"
# =========================================================
# 메인 predict
# =========================================================
def predict(symbol, strategy, source="일반", model_type=None):
    # ✅ 0. 그룹학습 중이면 predict 내부 아무 것도 돌리기 전에 즉시 차단
    if _group_active() and not _bypass_gate_for_source(source):
        print(f"[predict] blocked by group_active (source={source})")
        return failed_result(
            symbol or "None",
            strategy or "None",
            reason="group_predict_active",
            source=source,
            X_input=None
        )

    # ✅ 0-2. 전역 게이트(PREDICT_GATE) 닫혀 있으면 바로 차단
    if not (_bypass_gate_for_source(source) or is_predict_gate_open()):
        print(f"[predict] gate closed (source={source})")
        return failed_result(
            symbol or "None",
            strategy or "None",
            reason="predict_gate_closed",
            source=source,
            X_input=None
        )

    # 🔐 심볼/전략별 락 걸기
    lock_path = _lock_path_for(symbol or "None", strategy or "None")
    if "그룹직후" in str(source or ""):
        _clear_stale_lock(lock_path, PREDICT_LOCK_STALE_TRAIN_SEC, tag="(group)")
    else:
        _clear_stale_lock(lock_path, PREDICT_LOCK_TTL, tag="(normal)")
    lock_wait = _prep_lock_for_source(source)
    if not _acquire_predict_lock_with_retry(lock_path, lock_wait):
        print(f"[predict] lock timeout {symbol}-{strategy}")
        return failed_result(
            symbol or "None",
            strategy or "None",
            reason="predict_lock_timeout",
            source=source,
            X_input=None
        )

    # 하트비트 스레드 시작 (운영 상태 찍어주는 용도)
    _hb_stop = threading.Event()
    _hb_tag = f"{symbol}-{strategy}"
    _hb_thread = threading.Thread(
        target=_predict_hb_loop,
        args=(_hb_stop, _hb_tag, lock_path),
        daemon=True
    )
    _hb_thread.start()

    df = feat = X = outs = allpreds = None
    lib_vecs = lib_labels = None

    try:
        # 1) 패턴 라이브러리 로드 (없어도 됨)
        try:
            from evo_meta_dataset import load_pattern_library
            lib_vecs, lib_labels = load_pattern_library(symbol, strategy)
        except Exception:
            lib_vecs = None
            lib_labels = None

        # 2) prediction_log 헤더 보장
        try:
            ensure_prediction_log_exists()
        except Exception as _e:
            print(f"[헤더보장 실패] {_e}")

        # 3) 메타 예측기 임포트
        try:
            from evo_meta_learner import predict_evo_meta
        except Exception:
            predict_evo_meta = None

        try:
            from meta_learning import get_meta_prediction
        except Exception:
            def get_meta_prediction(pl, ft, meta=None):
                return int(np.argmax(np.mean(np.array(pl), axis=0)))

        ensure_failure_db()
        os.makedirs(_p("logs"), exist_ok=True)

        if not symbol or not strategy:
            return failed_result(
                symbol or "None",
                strategy or "None",
                reason="invalid_symbol_strategy",
                source=source,
                X_input=None
            )

        # 4) 레짐 감지 + 캘리브레이션 버전 체크
        regime = detect_regime(symbol, strategy, now=_now_kst())
        _ = get_calibration_version()
        print(f"[predict] start {symbol}-{strategy} regime={regime} source={source}")

        # 5) 윈도우 선택
        windows = find_best_windows(symbol, strategy)
        if not windows:
            if PREDICT_SOFT_ABORT:
                return _soft_abstain(
                    symbol, strategy,
                    reason="window_list_none",
                    meta_choice="abstain",
                    regime=regime,
                    df=None
                )
            return failed_result(
                symbol, strategy,
                reason="window_list_none",
                source=source,
                X_input=None
            )

        # 6) 캔들/피처 준비
        df = get_kline_by_strategy(symbol, strategy)
        log_return_distribution_for_run(symbol, strategy, df)

        if df is None or len(df) < max(windows) + 1:
            if PREDICT_SOFT_ABORT:
                return _soft_abstain(
                    symbol, strategy,
                    reason="df_short",
                    meta_choice="abstain",
                    regime=regime,
                    df=df
                )
            return failed_result(
                symbol, strategy,
                reason="df_short",
                source=source,
                X_input=None
            )

        feat = compute_features(symbol, df, strategy)
        if feat is None:
            if PREDICT_SOFT_ABORT:
                return _soft_abstain(
                    symbol, strategy,
                    reason="feature_short",
                    meta_choice="abstain",
                    regime=regime,
                    df=df
                )
            return failed_result(
                symbol, strategy,
                reason="feature_short",
                source=source,
                X_input=None
            )

        X = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        X = MinMaxScaler().fit_transform(X)
        feat_dim = int(X.shape[1])

        if X.shape[0] < max(windows):
            if PREDICT_SOFT_ABORT:
                return _soft_abstain(
                    symbol, strategy,
                    reason="insufficient_recent_rows",
                    meta_choice="abstain",
                    regime=regime,
                    df=df
                )
            return failed_result(
                symbol, strategy,
                reason="insufficient_recent_rows",
                source=source,
                X_input=None
            )

        # 7) 모델 탐색
        models = get_available_models(symbol, strategy)
        if not models:
            if PREDICT_SOFT_ABORT:
                return _soft_abstain(
                    symbol, strategy,
                    reason="no_models",
                    meta_choice="abstain",
                    regime=regime,
                    X_last=X[-1],
                    df=df
                )
            return failed_result(
                symbol, strategy,
                reason="no_models",
                source=source,
                X_input=X[-1]
            )

        # 8) 최근 클래스 분포 / 특징 벡터
        rec_freq = get_recent_class_frequencies(strategy)
        feat_row = torch.tensor(X[-1], dtype=torch.float32)

        # 9) 모델 추론
        outs, allpreds = get_model_predictions(
            symbol, strategy,
            models, df, X, windows,
            rec_freq, regime=regime
        )
        if not outs:
            if PREDICT_SOFT_ABORT:
                return _soft_abstain(
                    symbol, strategy,
                    reason="no_valid_model",
                    meta_choice="abstain",
                    regime=regime,
                    X_last=X[-1],
                    df=df
                )
            return failed_result(
                symbol, strategy,
                reason="no_valid_model",
                source=source,
                X_input=X[-1]
            )

        # 10) 시장 컨텍스트 / 최근 성공률
        market_ctx = _load_market_context(symbol, strategy)
        recent_succ_w = _get_recent_success_weight(symbol, strategy)

        # 11) 방향 힌트
        hint = _position_hint_from_market(df)
        allow_long, allow_short = bool(hint["allow_long"]), bool(hint["allow_short"])

        final_cls = None
        meta_choice = "best_single"
        chosen = None
        used_minret = False

        # (선택) 진화형 메타러너
        if glob.glob(os.path.join(MODEL_DIR, "evo_meta_learner*")):
            try:
                from evo_meta_learner import predict_evo_meta
                if callable(predict_evo_meta):
                    pred = int(predict_evo_meta(
                        feat_row.unsqueeze(0),
                        input_size=feat_dim
                    ))
                    cmin, cmax = _class_range_by_meta_or_cfg(
                        pred, (chosen or {}).get("meta"),
                        symbol, strategy
                    )
                    if _meets_minret_with_hint(
                        cmin, cmax, allow_long, allow_short,
                        MIN_RET_THRESHOLD
                    ):
                        final_cls = pred
                        meta_choice = "evo_meta_learner"
            except Exception as e:
                print(f"[evo_meta 예외] {e}")

        def _maybe_adjust(probs, recent):
            if ADJUST_WITH_DIVERSITY:
                return adjust_probs_with_diversity(
                    probs, recent, class_counts=None,
                    alpha=0.10, beta=0.10
                )
            return np.asarray(probs, dtype=float)

        # 12) 여러 모델 중 최종 선택 (유사도+성공률+리스크)
        sim_cache = {}
        if final_cls is None:
            best_i, best_score, best_pred = -1, -1.0, None
            scores = []
            current_vec = X[-1]

            for i, m in enumerate(outs):
                adj = _maybe_adjust(m["calib_probs"], rec_freq)
                num_classes = len(adj)

                # 유사도 기반 보조 확률
                if num_classes not in sim_cache:
                    sim_probs, info = _compute_similarity_class_probs(
                        current_vec, lib_vecs, lib_labels,
                        num_classes=num_classes, top_k=200
                    )
                    sim_cache[num_classes] = (sim_probs, info)
                else:
                    sim_probs, info = sim_cache[num_classes]

                w_sim = float(info.get("w_sim", 0.0))
                w_prob = 1.0 - w_sim

                hybrid = w_prob * adj + w_sim * sim_probs
                hybrid = np.nan_to_num(hybrid, nan=0.0, posinf=0.0, neginf=0.0)
                if hybrid.sum() > 0:
                    hybrid = hybrid / hybrid.sum()
                else:
                    hybrid = adj

                # 1% 미만 수익 구간 제거 필터
                mask = np.zeros_like(hybrid, dtype=float)
                for ci in range(num_classes):
                    try:
                        lo, hi = _class_range_by_meta_or_cfg(
                            ci, m.get("meta"),
                            symbol, strategy
                        )
                        if _meets_minret_with_hint(
                            lo, hi, allow_long, allow_short,
                            MIN_RET_THRESHOLD
                        ):
                            mask[ci] = 1.0
                    except Exception:
                        pass

                filt = hybrid * mask
                fused = False
                if filt.sum() > 0:
                    filt = filt / filt.sum()
                    pred = int(np.argmax(filt))
                    p = float(filt[pred])
                    fused = True
                else:
                    pred = int(np.argmax(hybrid))
                    p = float(hybrid[pred])

                # 시장 위험도 / 최근 성공률 반영
                risk_penalty = 1.0 - min(0.3, float(market_ctx.get("risk", 0.0)))
                p = p * (0.7 + 0.3 * recent_succ_w) * risk_penalty

                m.update({
                    "adjusted_probs": adj,
                    "sim_probs": sim_probs,
                    "hybrid_probs": hybrid,
                    "filtered_probs": (filt if fused else None),
                    "candidate_pred": pred,
                    "success_score": p,
                    "filtered_used": fused,
                    "hybrid_w_sim": w_sim,
                    "hybrid_w_prob": w_prob,
                    "sim_topk": int(info.get("m", 0)),
                    "symbol": symbol,
                    "strategy": strategy
                })

                scores.append((i, p, pred))
                if p > best_score:
                    best_i, best_score, best_pred = i, p, pred
                    used_minret = fused

            # 점수 차이가 작으면 탐색(epislon-explore) 조금
            if len(scores) >= 2:
                ss = sorted(scores, key=lambda x: x[1], reverse=True)
                top1, top2 = ss[0], ss[1]
                gap = float(top1[1] - top2[1])

                st = _load_json(EXP_STATE, {}).get(f"{symbol}|{strategy}", {})
                last = max(
                    [v.get("last_explore_ts", 0.0) for v in st.values()],
                    default=0.0
                ) if st else 0.0
                minutes = (time.time() - last) / 60.0 if last > 0 else 1e9
                eps = EXP_EPS * (0.5 if minutes < EXP_DEC_MIN else 1.0)

                if gap <= EXP_NEAR and random.random() < eps:
                    cands = []
                    for i, base, _ in ss[:min(3, len(ss))]:
                        mp = outs[i].get("model_path", "")
                        n = _load_json(EXP_STATE, {}).get(
                            f"{symbol}|{strategy}", {}
                        ).get(mp, {"n": 0}).get("n", 0)
                        bonus = EXP_GAMMA / np.sqrt(1.0 + float(n))
                        cands.append((i, base + bonus))
                    cands.sort(key=lambda x: x[1], reverse=True)
                    if cands and cands[0][0] != top1[0]:
                        best_i = cands[0][0]
                        best_pred = outs[best_i]["candidate_pred"]
                        meta_choice = "best_single_explore"

        final_cls = int(best_pred)
        chosen = outs[best_i]
        if meta_choice != "best_single_explore":
            meta_choice = os.path.basename(chosen["model_path"])

        # 13) 탐색 상태 기록
        try:
            st = _load_json(EXP_STATE, {})
            key = f"{symbol}|{strategy}"
            rec = st.setdefault(key, {}).setdefault(
                chosen.get("model_path", ""),
                {"n": 0, "n_explore": 0, "last_explore_ts": 0.0}
            )
            rec["n"] += 1
            if "best_single_explore" in meta_choice:
                rec["n_explore"] += 1
                rec["last_explore_ts"] = float(time.time())
            st[key][chosen.get("model_path", "")] = rec
            _save_json(EXP_STATE, st)
        except Exception:
            pass

        # 14) 1% 미만 필터 재확인 → 대체 후보 탐색
        try:
            cmin_sel, cmax_sel = _class_range_by_meta_or_cfg(
                final_cls, (chosen or {}).get("meta"),
                symbol, strategy
            )
            if not _meets_minret_with_hint(
                cmin_sel, cmax_sel,
                allow_long, allow_short,
                MIN_RET_THRESHOLD
            ):
                best_m, best_sc, best_cls = None, -1.0, None
                for m in outs:
                    base_probs = m.get(
                        "hybrid_probs",
                        m.get("adjusted_probs", m["calib_probs"])
                    )
                    for ci in range(len(base_probs)):
                        try:
                            lo, hi = _class_range_by_meta_or_cfg(
                                ci, m.get("meta"),
                                symbol, strategy
                            )
                        except Exception:
                            continue
                        if not _meets_minret_with_hint(
                            lo, hi, allow_long, allow_short,
                            MIN_RET_THRESHOLD
                        ):
                            continue
                        sc = float(base_probs[ci])
                        sc = sc * (0.7 + 0.3 * recent_succ_w)
                        if sc > best_sc:
                            best_sc, best_m, best_cls = sc, m, int(ci)
                if best_cls is not None:
                    final_cls = best_cls
                    chosen = best_m
                    used_minret = True
        except Exception as e:
            print(f"[임계 가드 예외] {e}")

        # 15) StopLoss risk guard
        try:
            ok_sl, why_sl, alt_m, alt_c = _stoploss_risk_guard(
                symbol, strategy,
                final_cls, outs,
                allow_long, allow_short,
                MIN_RET_THRESHOLD
            )
            if not ok_sl:
                if alt_m is not None:
                    chosen, final_cls = alt_m, int(alt_c)
                    meta_choice = (str(meta_choice) + "|stoploss_conservative")
                else:
                    return _soft_abstain(
                        symbol, strategy,
                        reason=why_sl,
                        meta_choice=str(meta_choice),
                        regime=regime,
                        X_last=X[-1],
                        group_id=(
                            chosen.get("group_id")
                            if isinstance(chosen, dict) else None
                        ),
                        df=df,
                        source="보류"
                    )
        except Exception as e:
            print(f"[StopLossRiskGuard 예외] {e}")

        # 16) Exit guard (수익률이 너무 약하면 보류)
        try:
            lo_sel, hi_sel = _class_range_by_meta_or_cfg(
                final_cls, (chosen or {}).get("meta"),
                symbol, strategy
            )
            exp_ret = (float(lo_sel) + float(hi_sel)) / 2.0
            ok, why = _exit_guard_check(lo_sel, hi_sel, exp_ret)
            if not ok:
                if FORCE_PUBLISH_ON_ABSTAIN:
                    alt_m, alt_c = _choose_conservative_prediction(
                        outs, symbol, strategy,
                        allow_long, allow_short,
                        MIN_RET_THRESHOLD
                    )
                    if alt_m is not None:
                        chosen, final_cls = alt_m, int(alt_c)
                        meta_choice = f"force_publish_exit_guard({why})"
                    else:
                        return _soft_abstain(
                            symbol, strategy,
                            reason=why,
                            meta_choice=str(meta_choice),
                            regime=regime,
                            X_last=X[-1],
                            group_id=(
                                chosen.get("group_id")
                                if isinstance(chosen, dict) else None
                            ),
                            df=df,
                            source="보류"
                        )
                else:
                    return _soft_abstain(
                        symbol, strategy,
                        reason=why,
                        meta_choice=str(meta_choice),
                        regime=regime,
                        X_last=X[-1],
                        group_id=(
                            chosen.get("group_id")
                            if isinstance(chosen, dict) else None
                        ),
                        df=df,
                        source="보류"
                    )
        except Exception as e:
            print(f"[출구 가드 예외] {e}")

        # 17) Reality guard는 현재 off (soft_guard에서 패스만)
        try:
            lo_sel, hi_sel = _class_range_by_meta_or_cfg(
                final_cls, (chosen or {}).get("meta"),
                symbol, strategy
            )
            exp_ret = (float(lo_sel) + float(hi_sel)) / 2.0
            # if RG_ENABLE:
            #     ok, why = _reality_guard_check(
            #         df, strategy, hint,
            #         lo_sel, hi_sel, exp_ret
            #     )
            #     action, new_chosen, new_cls, tag = _apply_soft_guard(
            #         df, strategy,
            #         outs, chosen, final_cls,
            #         allow_long, allow_short,
            #         MIN_RET_THRESHOLD
            #     )
            #     ...
        except Exception as e:
            print(f"[RealityGuard 예외] {e}")

        # 18) 최종 로그 + 리턴
        lo_sel, hi_sel = _class_range_by_meta_or_cfg(
            final_cls, (chosen or {}).get("meta"),
            symbol, strategy
        )
        exp_ret = (float(lo_sel) + float(hi_sel)) / 2.0
        pos_sel = _position_from_range(lo_sel, hi_sel)
        class_text = f"{float(lo_sel)*100:.2f}% ~ {float(hi_sel)*100:.2f}%"
        current = float(df.iloc[-1]["close"])
        entry = current

        def _topk(p, k=3):
            return [int(i) for i in np.argsort(p)[::-1][:k]]

        chosen_probs_for_topk = (
            chosen.get("hybrid_probs")
            if isinstance(chosen, dict) and "hybrid_probs" in chosen
            else (chosen or outs[0])["calib_probs"]
        )
        topk = _topk(chosen_probs_for_topk) if (chosen or outs) else []

        raw_pred = float(np.nan_to_num(
            (chosen or outs[0])["raw_probs"][final_cls],
            nan=0.0, posinf=0.0, neginf=0.0
        )) if (chosen or outs) else None

        calib_pred = float(np.nan_to_num(
            (chosen or outs[0])["calib_probs"][final_cls],
            nan=0.0, posinf=0.0, neginf=0.0
        )) if (chosen or outs) else None

        note = {
            "regime": regime,
            "meta_choice": meta_choice,
            "raw_prob_pred": raw_pred,
            "calib_prob_pred": calib_pred,
            "calib_ver": get_calibration_version(),
            "min_return_threshold": float(MIN_RET_THRESHOLD),
            "used_minret_filter": bool(used_minret),
            "explore_used": ("best_single_explore" in str(meta_choice)),
            "class_range_lo": float(lo_sel),
            "class_range_hi": float(hi_sel),
            "expected_return_mid": float(exp_ret),
            "position": pos_sel,
            "hint_allow_long": allow_long,
            "hint_allow_short": allow_short,
            "hint_ma_fast": hint.get("ma_fast"),
            "hint_ma_slow": hint.get("ma_slow"),
            "hint_slope": hint.get("slope"),
            "reality_guard": {
                "enabled": bool(RG_ENABLE),
                "vol_mult": float(RG_VOL_MULT),
                "method": RG_VOL_METHOD,
            },
            "soft_guard": "disabled_reality_guard",
            "market_ctx": market_ctx,
            "recent_success_weight": recent_succ_w,
        }

        ensure_prediction_log_exists()
        log_prediction(
            symbol=symbol,
            strategy=strategy,
            direction="예측",
            entry_price=entry,
            target_price=entry * (1 + exp_ret),
            model="meta",
            model_name=(
                "evo_meta_learner"
                if meta_choice == "evo_meta_learner"
                else str(meta_choice)
            ),
            predicted_class=final_cls,
            label=final_cls,
            note=json.dumps(note, ensure_ascii=False),
            top_k=topk,
            success=False,
            reason="predicted",
            rate=float(exp_ret),
            expected_return=float(exp_ret),
            position=pos_sel,
            return_value=0.0,
            source=(
                "진화형"
                if meta_choice == "evo_meta_learner"
                else "기본"
            ),
            group_id=(
                chosen.get("group_id")
                if isinstance(chosen, dict) else None
            ),
            feature_vector=torch.tensor(X[-1], dtype=torch.float32).numpy(),
            regime=regime,
            meta_choice=meta_choice,
            raw_prob=raw_pred,
            calib_prob=calib_pred,
            calib_ver=get_calibration_version(),
            class_return_min=float(lo_sel),
            class_return_max=float(hi_sel),
            class_return_text=class_text,
        )

        # 19) 섀도우 예측 로깅 (선택된 모델 제외 나머지)
        try:
            for m in outs:
                if chosen and m.get("model_path") == chosen.get("model_path"):
                    continue

                src_probs = m.get(
                    "hybrid_probs",
                    m.get("adjusted_probs", m["calib_probs"])
                )
                filt = m.get("filtered_probs", None)

                if filt is not None and np.sum(filt) > 0:
                    pred_i = int(np.argmax(filt))
                    src = filt
                else:
                    mask = np.zeros_like(src_probs, dtype=float)
                    for ci in range(len(src_probs)):
                        try:
                            lo_i, hi_i = _class_range_by_meta_or_cfg(
                                ci, m.get("meta"),
                                symbol, strategy
                            )
                            if _meets_minret_with_hint(
                                lo_i, hi_i,
                                allow_long, allow_short,
                                MIN_RET_THRESHOLD
                            ):
                                mask[ci] = 1.0
                        except Exception:
                            pass
                    adj2 = src_probs * mask
                    if np.sum(adj2) == 0:
                        continue
                    adj2 = adj2 / np.sum(adj2)
                    pred_i = int(np.argmax(adj2))
                    src = adj2

                lo_i, hi_i = _class_range_by_meta_or_cfg(
                    pred_i, m.get("meta"),
                    symbol, strategy
                )
                exp_i = (float(lo_i) + float(hi_i)) / 2.0
                pos_i = _position_from_range(lo_i, hi_i)
                top_i = [int(i) for i in np.argsort(src)[::-1][:3]]
                class_text_i = f"{float(lo_i)*100:.2f}% ~ {float(hi_i)*100:.2f}%"
                raw_i = float(np.nan_to_num(
                    m["raw_probs"][pred_i],
                    nan=0.0, posinf=0.0, neginf=0.0
                ))
                calib_i = float(np.nan_to_num(
                    m["calib_probs"][pred_i],
                    nan=0.0, posinf=0.0, neginf=0.0
                ))

                note_s = {
                    "regime": regime,
                    "shadow": True,
                    "model_path": os.path.basename(
                        m.get("model_path", "")
                    ),
                    "model_type": _norm_model_type(
                        m.get("model_type", "")
                    ),
                    "val_f1": (
                        None if m.get("val_f1") is None
                        else float(m.get("val_f1"))
                    ),
                    "calib_ver": get_calibration_version(),
                    "min_return_threshold": float(MIN_RET_THRESHOLD),
                    "class_range_lo": float(lo_i),
                    "class_range_hi": float(hi_i),
                    "expected_return_mid": float(exp_i),
                    "position": pos_i,
                    "hint_allow_long": allow_long,
                    "hint_allow_short": allow_short,
                }

                log_prediction(
                    symbol=symbol,
                    strategy=strategy,
                    direction="예측(섀도우)",
                    entry_price=entry,
                    target_price=entry * (1 + exp_i),
                    model=_norm_model_type(m.get("model_type", "")),
                    model_name=os.path.basename(
                        m.get("model_path", "")
                    ),
                    predicted_class=pred_i,
                    label=pred_i,
                    note=json.dumps(note_s, ensure_ascii=False),
                    top_k=top_i,
                    success=False,
                    reason="shadow",
                    rate=float(exp_i),
                    expected_return=float(exp_i),
                    position=pos_i,
                    return_value=0.0,
                    source="섀도우",
                    group_id=m.get("group_id", 0),
                    feature_vector=torch.tensor(
                        X[-1], dtype=torch.float32
                    ).numpy(),
                    regime=regime,
                    meta_choice="shadow",
                    raw_prob=raw_i,
                    calib_prob=calib_i,
                    calib_ver=get_calibration_version(),
                    class_return_min=float(lo_i),
                    class_return_max=float(hi_i),
                    class_return_text=class_text_i,
                )
        except Exception as e:
            print(f"[섀도우 로깅 예외] {e}")

        return {
            "symbol": symbol,
            "strategy": strategy,
            "model": "meta",
            "class": final_cls,
            "expected_return": float(exp_ret),
            "class_return_min": float(lo_sel),
            "class_return_max": float(hi_sel),
            "class_return_text": class_text,
            "position": pos_sel,
            "timestamp": _now_kst().isoformat(),
            "source": source,
            "regime": regime,
            "reason": (
                "진화형 메타 최종 선택"
                if meta_choice == "evo_meta_learner"
                else f"선택 모델: {meta_choice}"
            ),
        }

    finally:
        try:
            _hb_stop.set()
            _hb_thread.join(timeout=2)
        except Exception:
            pass

        _release_predict_lock(_lock_path_for(symbol or "None", strategy or "None"))
        try:
            _release_memory(df, feat, X, outs, allpreds, lib_vecs, lib_labels)
        finally:
            gc.collect()
            _safe_empty_cache()
# =========================================================
# 평가 루프 (네 원본 그대로)
# =========================================================
def evaluate_predictions(get_price_fn):
    from failure_db import check_failure_exists
    ensure_failure_db()
    ensure_prediction_log_exists()
    P = PREDICTION_LOG_PATH
    now_local = lambda: _now_kst()
    date_str = now_local().strftime("%Y-%m-%d")
    LOG_DIR = _p("logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    EVAL = os.path.join(LOG_DIR, f"evaluation_{date_str}.csv")
    WRONG = os.path.join(LOG_DIR, f"wrong_{date_str}.csv")
    tmp = None
    try:
        with open(P, "r", encoding="utf-8-sig", newline="") as f_in:
            rd = csv.DictReader(f_in)
            if rd.fieldnames is None:
                print("[오류] prediction_log.csv 헤더 없음")
                return
            base = list(PREDICTION_HEADERS)
            extras = ["status", "return"]
            fields = base + [c for c in extras if c not in base]
            dir_name = os.path.dirname(P) or "."
            fd, tmp = tempfile.mkstemp(prefix="predlog_", suffix=".csv", dir=dir_name, text=True)
            os.close(fd)
            with (open(tmp, "w", encoding="utf-8-sig", newline="") as f_tmp,
                  open(EVAL, "w", encoding="utf-8-sig", newline="") as f_eval,
                  open(WRONG, "w", encoding="utf-8-sig", newline="") as f_wrong):
                w_all = csv.DictWriter(f_tmp, fieldnames=fields)
                w_all.writeheader()
                eval_written = False
                wrong_written = False
                for r in rd:
                    try:
                        if r.get("status") not in [None, "", "pending", "v_pending"] and "섀도우" not in str(r.get("direction", "")):
                            w_all.writerow({k: r.get(k, "") for k in fields})
                            continue
                        sym = r.get("symbol", "UNKNOWN")
                        strat = r.get("strategy", "알수없음")
                        model = r.get("model", "unknown")
                        try:
                            gid = int(float(r.get("group_id", 0)))
                        except Exception:
                            gid = 0
                        def to_int(x, d):
                            try:
                                if x in [None, ""]:
                                    return d
                                return int(float(x))
                            except Exception:
                                return d
                        pred_cls = to_int(r.get("predicted_class", -1), -1)
                        label = to_int(r.get("label", -1), -1)
                        r["label"] = label
                        try:
                            entry = float(r.get("entry_price", 0) or 0)
                        except Exception:
                            entry = 0.0
                        if entry <= 0 or label == -1:
                            r.update({"status": "invalid", "reason": "invalid_entry_or_label", "return": 0.0, "return_value": 0.0})
                            if not check_failure_exists(r):
                                insert_failure_record(r)
                            w_all.writerow({k: r.get(k, "") for k in fields})
                            if not wrong_written:
                                wrong_writer = csv.DictWriter(f_wrong, fieldnames=sorted(r.keys()))
                                wrong_writer.writeheader()
                                wrong_written = True
                            wrong_writer.writerow({k: r.get(k, "") for k in r.keys()})
                            continue
                        ts = pd.to_datetime(r.get("timestamp"), errors="coerce")
                        if ts is None or pd.isna(ts):
                            r.update({"status": "invalid", "reason": "timestamp_parse_error", "return": 0.0, "return_value": 0.0})
                            w_all.writerow({k: r.get(k, "") for k in fields})
                            if not wrong_written:
                                wrong_writer = csv.DictWriter(f_wrong, fieldnames=sorted(r.keys()))
                                wrong_writer.writeheader()
                                wrong_written = True
                            wrong_writer.writerow({k: r.get(k, "") for k in r.keys()})
                            continue
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("Asia/Seoul")
                        else:
                            ts = ts.tz_convert("Asia/Seoul")
                        hours = {"단기": 4, "중기": 24, "장기": 168}.get(strat, 6)
                        deadline = ts + pd.Timedelta(hours=hours)
                        dfp = get_price_fn(sym, strat)
                        if dfp is None or "timestamp" not in dfp.columns:
                            r.update({"status": "invalid", "reason": "no_price_data", "return": 0.0, "return_value": 0.0})
                            w_all.writerow({k: r.get(k, "") for k in fields})
                            if not wrong_written:
                                wrong_writer = csv.DictWriter(f_wrong, fieldnames=sorted(r.keys()))
                                wrong_writer.writeheader()
                                wrong_written = True
                            wrong_writer.writerow({k: r.get(k, "") for k in r.keys()})
                            continue
                        dfp = dfp.copy()
                        dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], errors="coerce").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                        fut = dfp.loc[(dfp["timestamp"] >= ts) & (dfp["timestamp"] <= deadline)]
                        if fut.empty:
                            if _now_kst() < deadline:
                                r.update({"status": "pending", "reason": "⏳ 평가 대기 중(마감 전 데이터 없음)", "return": 0.0, "return_value": 0.0})
                                w_all.writerow({k: r.get(k, "") for k in fields})
                                continue
                            else:
                                r.update({"status": "invalid", "reason": "no_data_until_deadline", "return": 0.0, "return_value": 0.0})
                                w_all.writerow({k: r.get(k, "") for k in fields})
                                if not wrong_written:
                                    wrong_writer = csv.DictWriter(f_wrong, fieldnames=sorted(r.keys()))
                                    wrong_writer.writeheader()
                                    wrong_written = True
                                wrong_writer.writerow({k: r.get(k, "") for k in r.keys()})
                                continue
                        actual_max = float(fut["high"].max())
                        gain = (actual_max - entry) / (entry + 1e-12)
                        if pred_cls >= 0:
                            try:
                                cmin, cmax = get_class_return_range(pred_cls, sym, strat)
                                cmin, cmax = _sanitize_range(cmin, cmax)
                            except Exception:
                                cmin, cmax = (0.0, 0.0)
                        else:
                            cmin, cmax = (0.0, 0.0)
                        reached = gain >= cmin
                        if _now_kst() < deadline and reached:
                            status = "v_success" if str(r.get("volatility", "")).strip().lower() in ["1", "true"] else "success"
                            r.update({"status": status, "reason": f"[조기성공 pred_class={pred_cls}] gain={gain:.3f} (cls_min={cmin}, cls_max={cmax})", "return": 0.0, "return_value": 0.0})
                            log_prediction(symbol=sym, strategy=strat, direction=f"평가:{status}", entry_price=entry, target_price=entry * (1 + gain), timestamp=_now_kst().isoformat(), model=model, predicted_class=pred_cls, success=True, reason=r["reason"], rate=gain, expected_return=gain, position=("long" if cmax > 0 else "short" if cmin < 0 else "neutral"), return_value=gain, volatility=(status == "v_success"), source="평가", label=label, group_id=gid)
                            if model == "meta":
                                update_model_success(sym, strat, model, True)
                            w_all.writerow({k: r.get(k, "") for k in fields})
                            if not eval_written:
                                eval_writer = csv.DictWriter(f_eval, fieldnames=sorted(r.keys()))
                                eval_writer.writeheader()
                                eval_written = True
                            eval_writer.writerow({k: r.get(k, "") for k in r.keys()})
                            continue
                        if _now_kst() < deadline and not reached:
                            r.update({"status": "pending", "reason": "⏳ 평가 대기 중", "return": 0.0, "return_value": 0.0})
                            w_all.writerow({k: r.get(k, "") for k in fields})
                            continue
                        status = "success" if reached else "fail"
                        if str(r.get("volatility", "")).strip().lower() in ["1", "true"]:
                            status = "v_success" if status == "success" else "v_fail"
                        r.update({"status": status, "reason": f"[pred_class={pred_cls}] gain={gain:.3f} (cls_min={cmin}, cls_max={cmax})", "return": 0.0, "return_value": 0.0})
                        log_prediction(symbol=sym, strategy=strat, direction=f"평가:{status}", entry_price=entry, target_price=entry * (1 + gain), timestamp=_now_kst().isoformat(), model=model, predicted_class=pred_cls, success=(status in ["success", "v_success"]), reason=r["reason"], rate=gain, expected_return=gain, position=("long" if cmax > 0 else "short" if cmin < 0 else "neutral"), return_value=gain, volatility=("v_" in status), source="평가", label=label, group_id=gid)
                        if status in ["fail", "v_fail"]:
                            if not check_failure_exists(r):
                                insert_failure_record(r)
                        if model == "meta":
                            update_model_success(sym, strat, model, status in ["success", "v_success"])
                        w_all.writerow({k: r.get(k, "") for k in fields})
                        if not eval_written:
                            eval_writer = csv.DictWriter(f_eval, fieldnames=sorted(r.keys()))
                            eval_writer.writeheader()
                            eval_written = True
                        eval_writer.writerow({k: r.get(k, "") for k in r.keys()})
                        if status in ["fail", "v_fail"]:
                            if not wrong_written:
                                wrong_writer = csv.DictWriter(f_wrong, fieldnames=sorted(r.keys()))
                                wrong_writer.writeheader()
                                wrong_written = True
                            wrong_writer.writerow({k: r.get(k, "") for k in r.keys()})
                    except Exception as e:
                        r.update({"status": "invalid", "reason": f"exception:{e}", "return": 0.0, "return_value": 0.0})
                        w_all.writerow({k: r.get(k, "") for k in fields})
                        if not wrong_written:
                            wrong_writer = csv.DictWriter(f_wrong, fieldnames=sorted(r.keys()))
                            wrong_writer.writeheader()
                            wrong_written = True
                        wrong_writer.writerow({k: r.get(k, "") for k in r.keys()})
            shutil.move(tmp, P)
            print("[✅ 평가 완료] 스트리밍 재작성 성공")

            try:
                import pandas as _pd, os as _os
                _agg_path = _p("logs", "evaluation_result.csv")
                _df_today = _pd.read_csv(EVAL, encoding="utf-8-sig") if _os.path.exists(EVAL) else _pd.DataFrame()
                _df_old = _pd.read_csv(_agg_path, encoding="utf-8-sig") if _os.path.exists(_agg_path) else _pd.DataFrame()
                _df_all = _pd.concat([_df_old, _df_today], ignore_index=True)
                if "timestamp" in _df_all.columns:
                    _df_all["timestamp"] = _pd.to_datetime(_df_all["timestamp"], errors="coerce")
                    _df_all = _df_all.sort_values("timestamp", ascending=False)
                _df_all = _df_all.head(100)
                _df_all.to_csv(_agg_path, index=False, encoding="utf-8-sig")
                print(f"[✅ 평가 집계 갱신] {_agg_path} rows={len(_df_all)}")
            except Exception as _e:
                print(f"[⚠️ 평가 집계 갱신 실패] {_e}")

    except FileNotFoundError:
        print(f"[정보] {P} 없음 → 평가 스킵")
    except Exception as e:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        print(f"[오류] evaluate_predictions 스트리밍 실패 → {e}")
    finally:
        _release_memory()
        gc.collect()
        _safe_empty_cache()

# =========================================================
# 실행 헬퍼
# =========================================================
def _get_price_df_for_eval(symbol, strategy): return get_kline_by_strategy(symbol, strategy)
def run_evaluation_once(): evaluate_predictions(_get_price_df_for_eval)
def run_evaluation_loop(interval_minutes=None):
    try:
        iv = int(os.getenv("EVAL_INTERVAL_MIN", "30")) if interval_minutes is None else int(interval_minutes)
    except Exception:
        iv = 30
    iv = max(1, iv)
    print(f"[EVAL_LOOP] 시작 — {iv}분 주기")
    while True:
        try:
            run_evaluation_once()
        except Exception as e:
            print(f"[EVAL_LOOP] evaluate_predictions 예외 → {e}")
        time.sleep(iv * 60)

def _load_label_stats(symbol: str, strategy: str):
    try:
        key = f"{symbol.strip().upper()}__{strategy.strip()}"
        base = _p("labels", key)
        path_parquet = f"{base}.parquet"
        path_csv = f"{base}.csv"

        if os.path.exists(path_parquet):
            df = pd.read_parquet(path_parquet)
        elif os.path.exists(path_csv):
            df = pd.read_csv(path_csv)
        else:
            return {"class_count": {}, "risk_long": {}, "risk_short": {}, "risk_conflict": {}}

        req_cols = {"class_id", "up_ge_2pct", "dn_le_-2pct", "conflict_2pct"}
        if not req_cols.issubset(set(df.columns)):
            return {"class_count": {}, "risk_long": {}, "risk_short": {}, "risk_conflict": {}}

        g = df.groupby("class_id", dropna=False)
        cnt = g.size().to_dict()

        def _rate(series):
            n = float(len(series))
            return float(series.sum()) / n if n > 0 else 0.0

        risk_long = {}
        risk_short = {}
        risk_conflict = {}

        for k, grp in g:
            try:
                risk_long[int(k)] = _rate(grp["dn_le_-2pct"].astype(int))
            except Exception:
                risk_long[int(k)] = 0.0
            try:
                risk_short[int(k)] = _rate(grp["up_ge_2pct"].astype(int))
            except Exception:
                risk_short[int(k)] = 0.0
            try:
                risk_conflict[int(k)] = _rate(grp["conflict_2pct"].astype(int))
            except Exception:
                risk_conflict[int(k)] = 0.0

        return {
            "class_count": {int(k): int(v) for k, v in cnt.items()},
            "risk_long": risk_long,
            "risk_short": risk_short,
            "risk_conflict": risk_conflict,
        }
    except Exception:
        return {"class_count": {}, "risk_long": {}, "risk_short": {}, "risk_conflict": {}}

def _stoploss_risk_guard(symbol: str, strategy: str, final_cls: int,
                         outs, allow_long: bool, allow_short: bool, min_thr: float):
    try:
        SL_RISK_MAX = float(os.getenv("SL_RISK_MAX", "0.40"))
        SL_RISK_SIDE_MAX = float(os.getenv("SL_RISK_SIDE_MAX", "0.35"))
        SL_MIN_CLASS_SAMPLES = int(os.getenv("SL_MIN_CLASS_SAMPLES", "50"))
        SL_LOW_SAMPLE_RELAX = float(os.getenv("SL_LOW_SAMPLE_RELAX", "0.10"))

        stats = _load_label_stats(symbol, strategy)
        cls_n = int(stats["class_count"].get(int(final_cls), 0))

        relax = SL_LOW_SAMPLE_RELAX if cls_n < SL_MIN_CLASS_SAMPLES else 0.0
        side_thr = min(0.95, SL_RISK_SIDE_MAX + relax)
        mix_thr = min(0.95, SL_RISK_MAX + relax)

        try:
            lo_sel, hi_sel = _class_range_by_meta_or_cfg(int(final_cls), None, symbol, strategy)
            pos = _position_from_range(lo_sel, hi_sel)
        except Exception:
            pos = "neutral"

        r_conf = float(stats["risk_conflict"].get(int(final_cls), 0.0))
        r_long = float(stats["risk_long"].get(int(final_cls), 0.0))
        r_short = float(stats["risk_short"].get(int(final_cls), 0.0))

        too_mixed = (r_conf >= mix_thr)
        too_side = ((pos == "long" and r_long >= side_thr) or
                    (pos == "short" and r_short >= side_thr))

        if not (too_mixed or too_side):
            return True, "ok", None, None

        alt_m, alt_c = _choose_conservative_prediction(
            outs, symbol, strategy, allow_long, allow_short, min_thr
        )
        if alt_m is not None:
            why = f"stoploss_risk_conservative(r_conf={r_conf:.2f}, r_long={r_long:.2f}, r_short={r_short:.2f}, pos={pos})"
            return False, why, alt_m, int(alt_c)

        why = f"stoploss_risk_abstain(r_conf={r_conf:.2f}, r_long={r_long:.2f}, r_short={r_short:.2f}, pos={pos})"
        return False, why, None, None

    except Exception as e:
        return True, f"stoploss_risk_guard_exception:{e}", None, None

def log_return_distribution_for_run(symbol: str, strategy: str, df):
    if df is None or df.empty:
        return
    rets = extract_candle_returns(df, max_rows=1000)
    if not rets:
        return
    hist = make_return_histogram(rets, bins=20)
    log_prediction(
        symbol=symbol,
        strategy=strategy,
        direction="운영수익분포",
        entry_price=0.0,
        target_price=0.0,
        model="predictor",
        model_name="predictor",
        predicted_class=-1,
        label=-1,
        note=json.dumps(
            {
                "sample_size": len(rets),
                "bin_edges": hist["bin_edges"],
                "bin_counts": hist["bin_counts"],
            },
            ensure_ascii=False,
        ),
        top_k=[],
        success=True,
        reason="run_return_distribution",
        rate=0.0,
        expected_return=0.0,
        position="neutral",
        return_value=0.0,
        source="run",
    )

if __name__ == "__main__":
    res = predict("BTCUSDT", "단기", source="테스트")
    print(res)
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig")
        print("[✅ prediction_log.csv 상위 20줄 출력]")
        print(df.head(20))
    except Exception as e:
        print(f"[오류] prediction_log.csv 로드 실패] {e}")
    if str(os.getenv("EVAL_LOOP", "0")).strip() == "1":
        run_evaluation_loop()
