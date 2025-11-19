# train.py — SPEED v2.6 FINAL (전략별 DF/라벨 분리 적용판, 그룹쪼개기 잠시 OFF, 캔들수익분포 로그 유지)
# -*- coding: utf-8 -*-
import sitecustomize
import os, time, glob, shutil, json, random, traceback, threading, gc, csv, re
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import numpy as np, pandas as pd, pytz, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from logger import (
    log_prediction,
    ensure_prediction_log_exists,
    extract_candle_returns,      # ← 방금 만든 거
    make_return_histogram,       # ← 방금 만든 거
)

# ---------- 공용 메모리 유틸 ----------
def _safe_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _release_memory(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    _safe_empty_cache()


# ---------- 기본 환경/시드 ----------
def _set_default_thread_env(n: str, v: int):
    if os.getenv(n) is None:
        os.environ[n] = str(v)


for _n in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    _set_default_thread_env(_n, int(os.getenv("CPU_THREAD_CAP", "1")))
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
except:
    pass

# --- CPU 최적화 ---
try:
    torch.set_num_interop_threads(1)
except:
    pass
try:
    torch.backends.mkldnn.enabled = True
except:
    pass
try:
    torch.use_deterministic_algorithms(False)
    torch.set_deterministic_debug(False)
except:
    pass


def set_global_seed(s: int = 20240101):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass


set_global_seed(int(os.getenv("GLOBAL_SEED", "20240101")))

# ---------- 외부 의존 ----------
from model_io import save_model

# [풀백] data.utils → utils
try:
    from data.utils import (
        get_kline_by_strategy,
        compute_features,
        SYMBOL_GROUPS,
        should_train_symbol,
        mark_symbol_trained,
        ready_for_group_predict,
        mark_group_predicted,
        reset_group_order,
        CacheManager as DataCacheManager,
        compute_features_multi,
    )
except Exception:
    from utils import (
        get_kline_by_strategy,
        compute_features,
        SYMBOL_GROUPS,
        should_train_symbol,
        mark_symbol_trained,
        ready_for_group_predict,
        mark_group_predicted,
        reset_group_order,
        CacheManager as DataCacheManager,
    )

    # 선택적 신규 API 폴백
    def compute_features_multi(symbol: str, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        out = {}
        for s in ("단기", "중기", "장기"):
            try:
                out[s] = compute_features(symbol, df, s)
            except Exception:
                out[s] = None
        return out

# ===== [ADD] 보강 임포트: data.utils 에서 현재 그룹 조회 =====
try:
    from data.utils import get_current_group_index, get_current_group_symbols
except Exception:
    try:
        from utils import get_current_group_index, get_current_group_symbols
    except Exception:
        def get_current_group_index():
            return 0

        def get_current_group_symbols():
            return SYMBOL_GROUPS[0] if SYMBOL_GROUPS else []

# NOTE: 리포 구조에 맞춰 경로 정정 (robust dual import)
try:
    from model.base_model import get_model, freeze_backbone, unfreeze_last_k_layers
except Exception:
    from base_model import get_model

    def freeze_backbone(model):
        return None

    def unfreeze_last_k_layers(model, k: int = 1):
        return None

from feature_importance import compute_feature_importance, save_feature_importance
from failure_db import insert_failure_record, ensure_failure_db
import logger
from logger import log_prediction, ensure_prediction_log_exists  # <<< [ADD] 운영로그 찍기용

ensure_prediction_log_exists()  # <<< prediction_log.csv 없으면 만들어 둔다
from config import (
    get_NUM_CLASSES,
    get_FEATURE_INPUT_SIZE,
    get_class_groups,
    get_class_ranges,
    set_NUM_CLASSES,
    STRATEGY_CONFIG,
    get_QUALITY,
    get_LOSS,
    BOUNDARY_BAND,
    get_TRAIN_LOG_PATH,
)

# ===================== [ADD] 클래스 불균형 보정 유틸 =====================
# YOPO 철학 준수: 라벨 "병합" 없음. 오로지 학습 데이터에서 소수 클래스만 살짝 늘리거나(오버샘플) 가중 샘플링.
try:
    from data_augmentation import balance_classes, compute_class_weights, make_weighted_sampler
except Exception:
    # 안전 폴백: 아무 것도 하지 않음
    def balance_classes(X, y, *a, **k): return (X, y)
    def compute_class_weights(y, *a, **k): return np.ones((int(np.max(y))+1 if len(y)>0 else 0,), dtype=np.float32)
    def make_weighted_sampler(*a, **k): return None
# ======================================================================

# ================================================
# ⚠️ 여기서부터 경로를 전부 "안전 경로"로 바꿈
# 기본은 /tmp/persistent 밑으로 저장. 환경변수 PERSIST_DIR 있으면 그거 씀.
# ================================================
BASE_PERSIST_DIR = os.getenv("PERSIST_DIR", "/tmp/persistent")
try:
    os.makedirs(BASE_PERSIST_DIR, exist_ok=True)
except Exception:
    # 최후 폴백
    BASE_PERSIST_DIR = "/tmp/persistent-fallback"
    os.makedirs(BASE_PERSIST_DIR, exist_ok=True)

LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_PERSIST_DIR, "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_PERSIST_DIR, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# run/ 락파일 위치도 안전 경로로
RUN_DIR = os.getenv("RUN_DIR", os.path.join(BASE_PERSIST_DIR, "run"))
os.makedirs(RUN_DIR, exist_ok=True)

# GROUP_ACTIVE 도 여기로
GROUP_ACTIVE_PATH = os.path.join(BASE_PERSIST_DIR, "GROUP_ACTIVE")
PERSIST_DIR = BASE_PERSIST_DIR  # 아래 코드들이 쓰는 이름 그대로 둠
# ================================================

# ==== [ADD] 학습 때 캔들 수익분포 운영로그로 남기는 함수 ====
def log_return_distribution_for_train(symbol: str, strategy: str, df: pd.DataFrame, max_rows: int = 1000):
    """
    학습 때 사용한 캔들의 수익률 분포를 운영로그와 '똑같은 방식'으로 남긴다.
    즉, 공통 함수만 쓴다.
    """
    try:
        if df is None or df.empty:
            return

        # 1) 공통함수로 수익률 전부 뽑기
        # ✅ 전략까지 같이 넘겨서, labels / 운영로그와 완전히 동일한 방식으로 계산
        rets = extract_candle_returns(df, max_rows=max_rows, strategy=strategy)

        if not rets:
            return

        # 2) 공통함수로 히스토그램 만들기
        hist_info = make_return_histogram(rets, bins=20)

        # 3) 콘솔에도 찍어주고
        print(f"[수익분포: {symbol}-{strategy}] sample={len(rets)}", flush=True)
        edges = hist_info["bin_edges"]
        counts = hist_info["bin_counts"]
        for i, cnt in enumerate(counts):
            if cnt == 0:
                continue
            lo = edges[i]
            hi = edges[i + 1]
            print(f"  {lo:.4f} ~ {hi:.4f} : {cnt}", flush=True)

        # 4) 운영로그(csv)에도 남기기
        log_prediction(
            symbol=symbol,
            strategy=strategy,
            direction="학습수익분포",
            entry_price=0.0,
            target_price=0.0,
            model="trainer",
            model_name="trainer",
            predicted_class=-1,
            label=-1,
            note=json.dumps(
                {
                    "sample_size": len(rets),
                    "bin_edges": hist_info["bin_edges"],
                    "bin_counts": hist_info["bin_counts"],
                },
                ensure_ascii=False,
            ),
            top_k=[],
            success=True,
            reason="train_return_distribution",
            rate=0.0,
            expected_return=0.0,
            position="neutral",
            return_value=0.0,
            source="train",
        )
    except Exception as e:
        print(f"[train.return-dist warn] {e}", flush=True)


# ==== [ADD] train 로그 경로/헤더 보장 ====
DEFAULT_TRAIN_HEADERS = [
    "timestamp",
    "symbol",
    "strategy",
    "model",
    "val_acc",
    "val_f1",
    "val_loss",
    "engine",
    "window",
    "recent_cap",
    "rows",
    "limit",
    "min",
    "augment_needed",
    "enough_for_training",
    "note",
    "source_exchange",
    "status",
    # 여분(옵셔널)
    "accuracy",
    "f1",
    "loss",
    "y_true",
    "y_pred",
    "num_classes",
    # === 진단 5종 ===
    "NUM_CLASSES",
    "class_counts_label_freeze",
    "usable_samples",
    "class_counts_after_assemble",
    "batch_stratified_ok",
]
try:
    from logger import TRAIN_HEADERS

    TRAIN_HEADERS = list(dict.fromkeys(list(TRAIN_HEADERS) + DEFAULT_TRAIN_HEADERS))
except Exception:
    TRAIN_HEADERS = DEFAULT_TRAIN_HEADERS

_raw_train_log_path = get_TRAIN_LOG_PATH()
if _raw_train_log_path.startswith("/persistent"):
    TRAIN_LOG = os.path.join(
        BASE_PERSIST_DIR, os.path.relpath(_raw_train_log_path, "/persistent")
    )
else:
    TRAIN_LOG = _raw_train_log_path
try:
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
except Exception:
    TRAIN_LOG = os.path.join(BASE_PERSIST_DIR, "train_log.csv")
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)


def _ensure_train_log():
    try:
        if not os.path.exists(TRAIN_LOG):
            with open(TRAIN_LOG, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(TRAIN_HEADERS)
    except Exception as e:
        print(f"[경고] train_log 초기화 실패: {e}")


def _normalize_train_row(row: dict) -> dict:
    r = {k: row.get(k, None) for k in TRAIN_HEADERS}
    if r.get("val_acc") is None and row.get("accuracy") is not None:
        r["val_acc"] = row.get("accuracy")
    if r.get("val_f1") is None and row.get("f1") is not None:
        r["val_f1"] = row.get("f1")
    if r.get("val_loss") is None and row.get("loss") is not None:
        r["val_loss"] = row.get("loss")
    r.setdefault("engine", row.get("engine", "manual"))
    r.setdefault("source_exchange", row.get("source_exchange", "BYBIT"))
    return r


def _append_train_log(row: dict):
    try:
        _ensure_train_log()
        with open(TRAIN_LOG, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=TRAIN_HEADERS, extrasaction="ignore"
            )
            w.writerow(_normalize_train_row(row))
    except Exception as e:
        print(f"[경고] train_log 기록 실패: {e}")


if not getattr(logger, "_patched_train_log", False):
    _orig_ltr = getattr(logger, "log_training_result", None)

    def _log_training_result_patched(*args, **kw):
        if callable(_orig_ltr):
            try:
                _orig_ltr(*args, **kw)
            except Exception as e:
                print(f"[경고] logger.log_training_result 실패: {e}")
        row = dict(kw)
        row.setdefault(
            "timestamp", datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
        )
        _append_train_log(row)

    logger.log_training_result = _log_training_result_patched
    logger._patched_train_log = True

# ✅ 예측 게이트: 안전 임포트(없으면 no-op)
try:
    from predict import close_predict_gate
except Exception:
    def close_predict_gate(*a, **k):
        return None

# ✅ 학습 직후 자동 예측 트리거 (없으면 no-op)
try:
    from predict_trigger import run_after_training
except Exception:
    def run_after_training(symbol, strategy, *a, **k):
        return False

# [가드] data_augmentation (없으면 원본 그대로 통과)  # (위에서 임포트 추가됨)

# [가드] focal_loss (없으면 CE Loss 대체)
try:
    from focal_loss import FocalLoss
except Exception:
    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(weight=weight)

        def forward(self, logits, targets):
            return self.ce(logits, targets)

# [풀백] data.labels → labels (Render용 엄격 fallback)
try:
    from data.labels import make_labels, make_all_horizon_labels
except ModuleNotFoundError:
    from labels import make_labels, make_all_horizon_labels

try:
    from window_optimizer import find_best_window, find_best_windows
except Exception:
    from window_optimizer import find_best_window

    def find_best_windows(symbol, strategy, window_list, top_k=3, group_id=None):
        return [find_best_window(symbol, strategy, window_list=window_list, group_id=group_id)]

# ───────── predict_lock: per-key 락 사용 ─────────
try:
    from predict_lock import (
        clear_stale_predict_lock as pl_clear_stale,
        wait_until_free as pl_wait_free,
    )
except Exception:
    def pl_clear_stale(lock_key=None):
        return None

    def pl_wait_free(max_wait_sec: int, lock_key=None):
        return True

# ───────── 파인튜닝 로더(선택) ─────────
try:
    from model_io import load_for_finetune as _load_for_finetune
except Exception:
    _load_for_finetune = None

# ---------- 전역 상수 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

_MAX_ROWS_FOR_TRAIN = int(os.getenv("TRAIN_MAX_ROWS", "1200"))
_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "128"))
_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", "0"))
_PIN_MEMORY = False
_PERSISTENT = False
SMART_TRAIN = os.getenv("SMART_TRAIN", "1") == "1"
LABEL_SMOOTH = float(os.getenv("LABEL_SMOOTH", "0.05"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP_NORM", "1.0"))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", "2.0"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "4"))
EARLY_STOP_MIN_DELTA = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0001"))

USE_AMP = os.getenv("USE_AMP", "1") == "1"
TRAIN_CUDA_EMPTY_EVERY_EP = os.getenv("TRAIN_CUDA_EMPTY_EVERY_EP", "1") == "1"

# ===== [ADD] 클래스 불균형 제어용 ENV =====
BALANCE_CLASSES_FLAG = os.getenv("BALANCE_CLASSES", "1") == "1"      # 기본 ON
WEIGHTED_SAMPLER_FLAG = os.getenv("WEIGHTED_SAMPLER", "0") == "1"    # 기본 OFF
# ===================================================================

def _as_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


COST_SENSITIVE_ARGMAX = _as_bool_env("COST_SENSITIVE_ARGMAX", True)
CS_ARG_BETA = float(os.getenv("CS_ARG_BETA", "1.0"))


def _epochs_for(strategy: str) -> int:
    """
    전략별 기본 학습 에포크 수
    - 단기: 조금 더 많이
    - 중기: 단기보다는 적게, 장기보다는 많게
    - 장기: 기존 수준 유지
    """
    if strategy == "단기":
        # 기존 24 → 기본 36 (env 로 덮어쓸 수 있음)
        return int(os.getenv("EPOCHS_SHORT", "36"))
    if strategy == "중기":
        # 기존 12 → 기본 24
        return int(os.getenv("EPOCHS_MID", "24"))
    if strategy == "장기":
        # 장기는 현재도 성능이 잘 나오므로 그대로 유지
        return int(os.getenv("EPOCHS_LONG", "12"))
    return int(os.getenv("EPOCHS_DEFAULT", "24"))



EVAL_MIN_F1_SHORT = float(os.getenv("EVAL_MIN_F1_SHORT", "0.10"))
EVAL_MIN_F1_MID = float(os.getenv("EVAL_MIN_F1_MID", "0.50"))
EVAL_MIN_F1_LONG = float(os.getenv("EVAL_MIN_F1_LONG", "0.45"))
_SHORT_RETRY = int(os.getenv("SHORT_STRATEGY_RETRY", "3"))


def _min_f1_for(strategy: str) -> float:
    return (
        EVAL_MIN_F1_SHORT
        if strategy == "단기"
        else (EVAL_MIN_F1_MID if strategy == "중기" else EVAL_MIN_F1_LONG)
    )


now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))

PREDICT_OVERRIDE_ON_GROUP_END = _as_bool_env("PREDICT_OVERRIDE_ON_GROUP_END", True)

PREDICT_FORCE_AFTER_GROUP = _as_bool_env("PREDICT_FORCE_AFTER_GROUP", True)
PREDICT_TIMEOUT_SEC = float(os.getenv("PREDICT_TIMEOUT_SEC", "180"))

IMPORTANCE_ENABLE = os.getenv("IMPORTANCE_ENABLE", "1") == "1"

GROUP_TRAIN_LOCK = os.path.join(RUN_DIR, "group_training.lock")


def _set_group_active(active: bool, group_idx: int | None = None, symbols: list | None = None):
    try:
        if active:
            with open(GROUP_ACTIVE_PATH, "w", encoding="utf-8") as f:
                ts = datetime.utcnow().isoformat()
                syms = ",".join(symbols or [])
                f.write(f"ts={ts}\n")
                if group_idx is not None:
                    f.write(f"group={int(group_idx)}\n")
                f.write(f"symbols={syms}\n")
        else:
            if os.path.exists(GROUP_ACTIVE_PATH):
                os.remove(GROUP_ACTIVE_PATH)
    except Exception as e:
        try:
            print(f"[GROUP_ACTIVE warn] {e}", flush=True)
        except:
            pass


def _set_group_train_lock(active: bool, group_idx: int | None = None, symbols: list | None = None):
    try:
        if active:
            with open(GROUP_TRAIN_LOCK, "w", encoding="utf-8") as f:
                f.write(f"group={int(group_idx) if group_idx is not None else -1}\n")
                f.write(f"ts={datetime.utcnow().isoformat()}\n")
                f.write(f"symbols={','.join(symbols or [])}\n")
        else:
            if os.path.exists(GROUP_TRAIN_LOCK):
                os.remove(GROUP_TRAIN_LOCK)
    except Exception as e:
        try:
            print(f"[GROUP_LOCK warn] {e}", flush=True)
        except:
            pass


def _is_group_active_file() -> bool:
    try:
        return os.path.exists(GROUP_ACTIVE_PATH)
    except Exception:
        return False


def _is_group_lock_file() -> bool:
    try:
        return os.path.exists(GROUP_TRAIN_LOCK)
    except Exception:
        return False


def _maybe_insert_failure(payload: dict, feature_vector: Optional[List[Any]] = None):
    try:
        if not ready_for_group_predict():
            return
        insert_failure_record(payload, feature_vector=(feature_vector or []))
    except Exception as e:
        try:
            print(f"[FAILREC skip] {e}", flush=True)
        except:
            pass


def _safe_print(msg):
    try:
        print(msg, flush=True)
    except:
        pass


def _stem(p: str) -> str:
    return os.path.splitext(p)[0]


def _save_model_and_meta(model: nn.Module, path_pt: str, meta: dict):
    os.makedirs(os.path.dirname(path_pt), exist_ok=True)
    weight = _stem(path_pt) + ".ptz"
    save_model(weight, model.state_dict())
    meta_path = _stem(path_pt) + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    return weight, meta_path


def coverage_split_indices(
    y,
    val_frac=0.20,
    min_coverage=0.60,
    stride=50,
    max_windows=200,
    num_classes=None,
):
    y = np.asarray(y).astype(int)
    n = len(y)
    val_len = max(1, int(round(n * val_frac)))
    if num_classes is None:
        uniq = np.unique(y)
        num_classes = (
            max(len(uniq), int(uniq.max()) + 1)
            if (uniq.size and uniq.min() >= 0)
            else len(uniq)
        )
    tried = 0
    best = None
    end = n
    while end - val_len >= 0 and tried < max_windows:
        start = end - val_len
        yv = y[start:end]
        cnt = Counter(yv.tolist())
        coverage = len([1 for v in cnt.values() if v > 0]) / max(1, num_classes)
        if best is None or coverage > best[0]:
            best = (coverage, start, end, cnt)
        if coverage >= float(min_coverage):
            break
        end -= int(max(1, stride))
        tried += 1
    if best is None:
        start, end = max(0, n - val_len), n
        cnt = Counter(y[start:end].tolist())
        coverage = len(cnt) / max(1, num_classes)
    else:
        coverage, start, end, cnt = best
    val_idx = np.arange(start, end)
    train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)], axis=0)
    _safe_print(
        f"[VAL COVER] {len(cnt)}/{num_classes} ({coverage:.2f}) window={start}:{end} size={len(val_idx)}"
    )
    return train_idx, val_idx


def _log_skip(symbol, strategy, reason):
    logger.log_training_result(
        symbol,
        strategy,
        model="all",
        accuracy=0.0,
        f1=0.0,
        loss=0.0,
        val_acc=0.0,
        val_f1=0.0,
        val_loss=0.0,
        engine="manual",
        window=None,
        recent_cap=None,
        rows=None,
        limit=None,
        min=None,
        augment_needed=None,
        enough_for_training=None,
        note=reason,
        source_exchange="BYBIT",
        status="skipped",
    )
    _maybe_insert_failure(
        {
            "symbol": symbol,
            "strategy": strategy,
            "model": "all",
            "predicted_class": -1,
            "success": False,
            "rate": 0.0,
            "reason": reason,
        },
        feature_vector=[],
    )


def _log_fail(symbol, strategy, reason):
    logger.log_training_result(
        symbol,
        strategy,
        model="all",
        accuracy=0.0,
        f1=0.0,
        loss=0.0,
        val_acc=0.0,
        val_f1=0.0,
        val_loss=0.0,
        engine="manual",
        window=None,
        recent_cap=None,
        rows=None,
        limit=None,
        min=None,
        augment_needed=None,
        enough_for_training=None,
        note=reason,
        source_exchange="BYBIT",
        status="failed",
    )
    _maybe_insert_failure(
        {
            "symbol": symbol,
            "strategy": strategy,
            "model": "all",
            "predicted_class": -1,
            "success": False,
            "rate": 0.0,
            "reason": reason,
        },
        feature_vector=[],
    )


def _has_any_model_for_symbol(symbol: str) -> bool:
    exts = (".ptz", ".safetensors", ".pt")
    try:
        if any(glob.glob(os.path.join(MODEL_DIR, f"{symbol}_*{e}")) for e in exts):
            return True
        d = os.path.join(MODEL_DIR, symbol)
        return (
            any(glob.glob(os.path.join(d, "*", "*" + e)) for e in exts)
            if os.path.isdir(d)
            else False
        )
    except:
        return False


def _has_model_for(symbol: str, strategy: str) -> bool:
    exts = (".ptz", ".safetensors", ".pt")
    try:
        if any(glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*{e}")) for e in exts):
            return True
        d = os.path.join(MODEL_DIR, symbol, strategy)
        return (
            any(glob.glob(os.path.join(d, "*" + e)) for e in exts)
            if os.path.isdir(d)
            else False
        )
    except:
        return False


# ---------- 전략 간 피처/라벨 패스다운 (수정본) ----------
def _build_precomputed(symbol: str) -> tuple[Dict[str, Optional[pd.DataFrame]], Dict[str, Any], Dict[str, Any]]:
    """
    전략마다 자기 df를 불러서 → 그걸로 피처/라벨을 미리 계산해 둔다.
    이렇게 해야 단기 df 한 장으로 중기/장기 라벨을 재활용하는 구멍이 안 생김.
    """
    dfs: Dict[str, Optional[pd.DataFrame]] = {}
    feats: Dict[str, Any] = {}
    pre_lbl: Dict[str, Any] = {}

    for strat in ("단기", "중기", "장기"):
        try:
            df_s = get_kline_by_strategy(symbol, strat)
        except Exception:
            df_s = None

        if df_s is None or df_s.empty:
            dfs[strat] = None
            feats[strat] = None
            pre_lbl[strat] = None
            continue

        dfs[strat] = df_s

        # 피처도 전략별 df로 계산
        try:
            feats[strat] = compute_features(symbol, df_s, strat)
        except Exception:
            feats[strat] = None

        # 라벨도 전략별 df로 직접 계산 (labels.py가 전략→캔들수로 이미 나눠둠)
        try:
            pre_lbl[strat] = make_labels(df=df_s, symbol=symbol, strategy=strat, group_id=None)
        except Exception:
            pre_lbl[strat] = None

    return dfs, feats, pre_lbl


def _find_prev_model_for(symbol: str, prev_strategy: str) -> Optional[str]:
    try:
        candidates = []
        for p in glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{prev_strategy}_*.ptz")):
            candidates.append((os.path.getmtime(p), p))
        if not candidates:
            for p in glob.glob(os.path.join(MODEL_DIR, symbol, prev_strategy, "*.ptz")):
                candidates.append((os.path.getmtime(p), p))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception:
        return None


# === [ADD] 라벨 유효성/재시도 유틸 ===
def _uniq_nonneg(labels: np.ndarray) -> int:
    try:
        v = labels[np.asarray(labels) >= 0]
        return int(np.unique(v).size) if v.size else 0
    except Exception:
        return 0


def _rebuild_labels_once(df: pd.DataFrame, symbol: str, strategy: str):
    try:
        return make_labels(df=df, symbol=symbol, strategy=strategy, group_id=None)
    except Exception:
        return None

def _rebuild_samples_with_keepset(
    fv: np.ndarray,
    labels: np.ndarray,
    window: int,
    keep_set: set[int] | None,
    min_samples: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    그룹 정보(keep_set)를 이용해서 시계열 샘플을 다시 만드는 함수.

    - labels 는 "전역 클래스 인덱스"(0~NUM_CLASSES-1)를 그대로 유지한다.
    - keep_set 이 주어지면 그 클래스들 위주로 샘플을 골라오되,
      너무 적은 클래스(샘플 < min_samples)는 자동으로 약하게 반영.
    - 최소 2개 이상의 클래스가 실제로 남아 있지 않으면 빈 배열을 반환한다.
    """
    from collections import Counter

    # 방어 코드: 입력이 비었으면 바로 종료
    if fv is None or labels is None or len(fv) == 0 or len(labels) == 0:
        return (
            np.empty(
                (0, window, fv.shape[1] if fv is not None and fv.ndim == 2 else 0),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
        )

    labels = np.asarray(labels).astype(int)

    # 1) 전체 라벨 분포 (전역 클래스 기준)
    cnt = Counter(int(l) for l in labels.tolist() if int(l) >= 0)

    # 2) 이번 그룹에서 우선 보고 싶은 클래스 집합
    base_keep = set(int(c) for c in (keep_set or set(cnt.keys())))
    # 실제로 데이터가 있는 클래스만 남긴다
    base_keep = {c for c in base_keep if cnt.get(c, 0) > 0}

    if len(base_keep) == 0:
        return (
            np.empty(
                (0, window, fv.shape[1] if fv is not None and fv.ndim == 2 else 0),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
        )

    # 3) 샘플이 min_samples 이상인 클래스 위주로 우선 사용
    strong = [c for c in sorted(base_keep) if cnt.get(c, 0) >= min_samples]

    if len(strong) >= 2:
        eff_keep = set(strong)
    else:
        # min_samples 를 만족하는 클래스가 2개 미만이면
        # 일단 base_keep 에 있는 모든 클래스를 사용해서 6칸 구조를 유지
        eff_keep = set(base_keep)
        if len(eff_keep) < 2:
            # 정말로 클래스가 1개뿐이면 학습 불가
            return (
                np.empty(
                    (0, window, fv.shape[1] if fv is not None and fv.ndim == 2 else 0),
                    dtype=np.float32,
                ),
                np.empty((0,), dtype=np.int64),
            )

    # 4) 시계열 윈도우를 돌면서 X,y 구성 (라벨은 전역 인덱스를 그대로 사용)
    X_raw: list[np.ndarray] = []
    y_out: list[int] = []
    n = len(fv)

    for i in range(n - window):
        yi = i + window - 1
        if yi < 0 or yi >= len(labels):
            continue

        lab_g = int(labels[yi])  # 전역 클래스 인덱스(C0~C5)
        if lab_g < 0:
            continue
        if lab_g not in eff_keep:
            continue

        X_raw.append(fv[i : i + window])
        y_out.append(lab_g)

    if not X_raw:
        return (
            np.empty(
                (0, window, fv.shape[1] if fv is not None and fv.ndim == 2 else 0),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(X_raw, dtype=np.float32),
        np.asarray(y_out, dtype=np.int64),
    )
    
def _synthesize_minority_if_needed(
    X_raw: np.ndarray,
    y: np.ndarray,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # 지금 버전: 합성 안 한다
    return X_raw, y, False


def _ensure_val_has_two_classes(train_idx, val_idx, y, min_classes=2):
    vy = y[val_idx]
    if len(np.unique(vy)) >= min_classes:
        return train_idx, val_idx, False
    ty = y[train_idx]
    classes = np.unique(y)
    if len(classes) < min_classes:
        return train_idx, val_idx, False
    want = [c for c in classes if c not in set(vy)]
    moved = False
    for c in want[:2]:
        cand = np.where(ty == c)[0]
        if len(cand) == 0:
            continue
        take = cand[0]
        g_take = train_idx[take]
        train_idx = np.delete(train_idx, take)
        val_idx = np.append(val_idx, g_take)
        moved = True
        vy = y[val_idx]
        if len(np.unique(vy)) >= min_classes:
            break
    return train_idx, val_idx, moved
    
def train_one_model(
    symbol,
    strategy,
    group_id=None,
    max_epochs: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    pre_feat: Optional[pd.DataFrame] = None,
    pre_lbl: Optional[tuple] = None,
    df_hint: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    한 심볼-전략-그룹에 대해 실제로 모델을 1개 이상 학습해서 저장하는 함수.
    (device_type 누락되는 거 고친 버전 + 그룹별 클래스 제대로 분리한 버전)
    """
    HAS_CUDA = torch.cuda.is_available()
    device_type = "cuda" if HAS_CUDA else "cpu"
    use_amp_here = (os.getenv("USE_AMP", "1") == "1") and HAS_CUDA

    if max_epochs is None:
        max_epochs = _epochs_for(strategy)

    res = {
        "symbol": symbol,
        "strategy": strategy,
        "group_id": int(group_id or 0),
        "windows": [],
        "models": [],
    }

    try:
        ensure_failure_db()
        _safe_print(f"✅ train_one_model {symbol}-{strategy}-g{group_id}")

        # ===== 1. 데이터 불러오기 =====
        if df_hint is not None:
            df = df_hint
        else:
            try:
                df = get_kline_by_strategy(symbol, strategy)
            except Exception:
                df = None

        if df is None or df.empty:
            _log_skip(symbol, strategy, "데이터 없음")
            return res

        log_return_distribution_for_train(symbol, strategy, df)

        cfg = STRATEGY_CONFIG.get(strategy, {})
        _limit = int(cfg.get("limit", 300))
        _min_required = max(60, int(_limit * 0.90))
        _attrs = getattr(df, "attrs", {}) if df is not None else {}
        augment_needed = bool(_attrs.get("augment_needed", len(df) < _limit))
        enough_for_training = bool(_attrs.get("enough_for_training", len(df) >= _min_required))

        # ===== 2. 피처 만들기 =====
        if isinstance(pre_feat, pd.DataFrame):
            feat = pre_feat
        elif isinstance(pre_feat, dict) and pre_feat.get(strategy, None) is not None:
            feat = pre_feat[strategy]
        else:
            feat = compute_features(symbol, df, strategy)

        if feat is None or getattr(feat, "empty", True):
            _log_skip(symbol, strategy, "피처 없음")
            return res

        # ===== 3. 라벨 만들기 =====
        bin_info = None
        if isinstance(pre_lbl, tuple) and len(pre_lbl) in (3, 4, 6):
            # 기존 처리 그대로 유지
            if len(pre_lbl) == 6:
                gains, labels, class_ranges_used_global, be, bc, bs = pre_lbl
                bin_info = {
                    "bin_edges": be.tolist() if hasattr(be, "tolist") else list(be),
                    "bin_counts": bc.tolist() if hasattr(bc, "tolist") else list(bc),
                    "bin_spans": bs.tolist() if hasattr(bs, "tolist") else list(bs),
                }
            elif len(pre_lbl) == 4:
                gains, labels, class_ranges_used_global, bin_info = pre_lbl
            else:
                gains, labels, class_ranges_used_global = pre_lbl
        elif isinstance(pre_lbl, dict) and pre_lbl.get(strategy, None) is not None:
            val = pre_lbl[strategy]
            if isinstance(val, (list, tuple)) and len(val) in (3, 4, 6):
                if len(val) == 6:
                    gains, labels, class_ranges_used_global, be, bc, bs = val
                    bin_info = {
                        "bin_edges": be.tolist() if hasattr(be, "tolist") else list(be),
                        "bin_counts": bc.tolist() if hasattr(bc, "tolist") else list(bc),
                        "bin_spans": bs.tolist() if hasattr(bs, "tolist") else list(bs),
                    }
                elif len(val) == 4:
                    gains, labels, class_ranges_used_global, bin_info = val
                else:
                    gains, labels, class_ranges_used_global = val
            else:
                _log_skip(symbol, strategy, "사전 라벨 구조 오류")
                return res
        else:
            res_labels = make_labels(df=df, symbol=symbol, strategy=strategy, group_id=None)
            if isinstance(res_labels, (list, tuple)) and len(res_labels) in (3, 4, 6):
                if len(res_labels) == 6:
                    gains, labels, class_ranges_used_global, be, bc, bs = res_labels
                    bin_info = {
                        "bin_edges": be.tolist() if hasattr(be, "tolist") else list(be),
                        "bin_counts": bc.tolist() if hasattr(bc, "tolist") else list(bc),
                        "bin_spans": bs.tolist() if hasattr(bs, "tolist") else list(bs),
                    }
                elif len(res_labels) == 4:
                    gains, labels, class_ranges_used_global, bin_info = res_labels
                else:
                    gains, labels, class_ranges_used_global = res_labels
            else:
                _log_skip(symbol, strategy, "라벨 생성 실패")
                return res

        if (not isinstance(labels, np.ndarray)) or labels.size == 0:
            _log_skip(symbol, strategy, "라벨 없음")
            return res

        # ===== 4. 라벨 RETRY =====
        uniq0 = _uniq_nonneg(labels)
        if uniq0 <= 1:
            _safe_print(f"[LABEL RETRY] uniq<=1 → rebuild via make_labels() once ({symbol}-{strategy})")
            res_try = _rebuild_labels_once(df=df, symbol=symbol, strategy=strategy)
            if isinstance(res_try, (list, tuple)) and len(res_try) in (3, 4, 6):
                if len(res_try) == 6:
                    gains2, labels2, class_ranges2, be2, bc2, bs2 = res_try
                    bin_info2 = {
                        "bin_edges": be2.tolist() if hasattr(be2, "tolist") else list(be2),
                        "bin_counts": bc2.tolist() if hasattr(bc2, "tolist") else list(bc2),
                        "bin_spans": bs2.tolist() if hasattr(bs2, "tolist") else list(bs2),
                    }
                elif len(res_try) == 4:
                    gains2, labels2, class_ranges2, bin_info2 = res_try
                else:
                    gains2, labels2, class_ranges2 = res_try
                    bin_info2 = bin_info
                uniq1 = _uniq_nonneg(labels2)
                if uniq1 > uniq0 and uniq1 >= 2:
                    gains, labels = gains2, labels2
                    class_ranges_used_global = class_ranges2
                    bin_info = bin_info2
                    _safe_print(f"[LABEL RETRY OK] uniq {uniq0}→{uniq1}")
                else:
                    _safe_print(f"[LABEL RETRY NO-IMPROVE] uniq {uniq0}→{uniq1}")

        # ===== 5. 실제 클래스 구간 (★ 그룹 반영 핵심) =====
        # 5-1) 전체(전역) 클래스 구간 (C0~C5)
        if "class_ranges_used_global" in locals() and class_ranges_used_global is not None:
            full_ranges = class_ranges_used_global
        else:
            full_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=None)

        num_total_classes = len(full_ranges) if full_ranges is not None else 0

        # 5-2) 그룹별 클래스 인덱스 (전역 인덱스 리스트)
        from config import get_class_groups

        if num_total_classes > 0:
            groups = get_class_groups(num_classes=max(2, num_total_classes)) or []
        else:
            groups = []

        if group_id is not None and groups:
            try:
                gid = int(group_id)
            except Exception:
                gid = 0
            if gid < 0 or gid >= len(groups):
                gid = 0
            cls_in_group = list(groups[gid])  # 전역 클래스 인덱스들
        else:
            # 그룹 개념이 없으면 전체 클래스 사용
            cls_in_group = list(range(num_total_classes))

        if num_total_classes <= 0:
            _log_skip(symbol, strategy, f"전역 클래스 정보 없음")
            return res

        if not cls_in_group:
            _log_skip(symbol, strategy, f"그룹 내 클래스 없음(group_id={group_id})")
            return res

        # 5-3) 이번 학습에 사용할 "클래스 구간"은 전역 6칸 전체
        #      → 모델 출력은 항상 C0~C5 전체를 본다.
        class_ranges = full_ranges or []
        #      대신 keep_set 으로 "이 그룹에서 우선 보고 싶은 칸"을 지정
        keep_set = set(cls_in_group) if cls_in_group else set(range(num_total_classes))

        # ===== 6. 로그 출력 =====
        mask_cnt = int((labels < 0).sum())
        _safe_print(
            f"[LABELS] total={len(labels)} masked={mask_cnt} ({mask_cnt/max(1,len(labels)):.2%}) "
            f"BOUNDARY_BAND=±{BOUNDARY_BAND}"
        )
        try:
            # 전역 기준 분포
            cnt_before = np.bincount(labels[labels >= 0],
                                     minlength=num_total_classes).astype(int).tolist()
        except Exception:
            cnt_before = []

        num_classes_effective = int(np.unique(labels[labels >= 0]).size) if labels.size else 0
        empty_idx = [i for i, c in enumerate(cnt_before) if int(c) == 0]

        return_note = ""
        if isinstance(bin_info, dict):
            edges = bin_info.get("bin_edges", [])
            counts = bin_info.get("bin_counts", [])
            return_note = f" ; [ReturnDist] edges={edges[:20]}, counts={counts[:20]}"

        try:
            logger.log_training_result(
                symbol,
                strategy,
                model="all",
                accuracy=None,
                f1=None,
                loss=None,
                val_acc=None,
                val_f1=None,
                val_loss=None,
                engine="manual",
                window=None,
                recent_cap=None,
                rows=int(len(df)),
                limit=int(_limit),
                min=int(_min_required),
                augment_needed=bool(augment_needed),
                enough_for_training=bool(enough_for_training),
                note=f"[LabelStats] bins_total={num_total_classes}, bins_group={len(cls_in_group)}, empty={len(empty_idx)}, "
                     f"classes={num_classes_effective}, empty_idx={empty_idx[:8]}"
                     + return_note,
                source_exchange="BYBIT",
                status="info",
                NUM_CLASSES=int(num_total_classes),
                class_counts_label_freeze=cnt_before,
            )
        except Exception:
            pass

        # ===== 7. 피처 정제 =====
        drop_cols = [c for c in ("timestamp", "strategy", "symbol") if c in feat.columns]
        feat_num = feat.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        features_only = feat_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feat_dim = int(getattr(features_only, "shape", [0, FEATURE_INPUT_SIZE])[1]) or int(FEATURE_INPUT_SIZE)

        if len(features_only) > _MAX_ROWS_FOR_TRAIN or len(labels) > _MAX_ROWS_FOR_TRAIN:
            cut = min(_MAX_ROWS_FOR_TRAIN, len(features_only), len(labels))
            features_only = features_only.iloc[-cut:, :]
            labels = labels[-cut:]

        # ===== 8. 윈도우 후보 =====
        try:
            top_windows = find_best_windows(
                symbol, strategy,
                window_list=[16, 20, 24, 28, 32],
                top_k=3,
                group_id=group_id,
            )
        except Exception:
            try:
                top_windows = [
                    int(find_best_window(
                        symbol, strategy,
                        window_list=[16, 20, 24, 28, 32],
                        group_id=group_id,
                    ))
                ]
            except Exception:
                top_windows = [20]

        top_windows = [
            int(max(5, w))
            for w in top_windows
            if isinstance(w, (int, float)) and w == w
        ] or [20]
        _safe_print(f"[WINDOWS] {top_windows}")

        # ===== 9. 윈도우별 학습 =====
        for window in top_windows:
            if stop_event is not None and stop_event.is_set():
                break

            window = min(window, max(6, len(features_only) - 1))
            fv = features_only.values.astype(np.float32)

            # ★ 여기서 그룹 정보(keep_set)를 사용하지만,
            #   y 라벨은 전역 6클래스를 그대로 유지한다.
            X_raw, y = _rebuild_samples_with_keepset(
                fv=fv,
                labels=labels,
                window=window,
                keep_set=keep_set,
                min_samples=8,
            )

            repaired_info = {
                "neighbor_expansion": False,
                "synthetic_labels": False,
            }

            if X_raw.size and len(np.unique(y)) < 2:
                X_raw, y, syn = _synthesize_minority_if_needed(
                    X_raw, y, num_classes=len(class_ranges)
                )
                repaired_info["synthetic_labels"] = syn

            usable_samples = int(len(y))
            note_msg = ""
            if usable_samples == 0:
                _log_skip(symbol, strategy, f"유효 라벨 샘플 없음(w={window})")
                continue
            if y.min() < 0:
                _log_skip(symbol, strategy, f"음수 라벨 유입 감지(w={window})")
                continue
            if usable_samples < 20:
                note_msg = f"⚠️ 희소 학습 (샘플 {usable_samples})"
                _safe_print(f"[WARN] {symbol}-{strategy}-w{window}: {note_msg}")

            # 이번 학습에서 사용하는 클래스 수 = 전역 6클래스 전체
            set_NUM_CLASSES(len(class_ranges))

            # ===== train/val split =====
            strat_ok = False
            try:
                if len(y) >= 40 and len(np.unique(y)) >= 2:
                    splitter = StratifiedShuffleSplit(
                        n_splits=1,
                        test_size=0.20,
                        random_state=int(os.getenv("GLOBAL_SEED", "20240101")),
                    )
                    tr_idx, val_idx = next(splitter.split(X_raw, y))
                    strat_ok = True
            except Exception:
                strat_ok = False

            if not strat_ok:
                try:
                    train_idx, val_idx = coverage_split_indices(
                        y, val_frac=0.20,
                        min_coverage=0.60,
                        stride=50,
                        num_classes=len(class_ranges),
                    )
                except Exception:
                    n = len(y)
                    if n <= 1:
                        train_idx = np.array([0], dtype=int)
                        val_idx = np.array([0], dtype=int)
                    else:
                        train_idx = np.arange(0, n - 1, dtype=int)
                        val_idx = np.array([n - 1], dtype=int)
            else:
                train_idx, val_idx = tr_idx, val_idx

            if len(train_idx) == 0 and len(val_idx) == 0:
                n = len(y)
                if n <= 1:
                    train_idx = np.array([0], dtype=int)
                    val_idx = np.array([0], dtype=int)
                else:
                    train_idx = np.arange(0, n - 1, dtype=int)
                    val_idx = np.array([n - 1], dtype=int)
            elif len(train_idx) == 0:
                val_take = int(val_idx[0])
                train_idx = np.array([val_take], dtype=int)
                val_idx = np.array([val_idx[-1]], dtype=int)
            elif len(val_idx) == 0:
                val_idx = np.array([train_idx[-1]], dtype=int)
                train_idx = train_idx[:-1] if len(train_idx) > 1 else train_idx

            train_idx, val_idx, _ = _ensure_val_has_two_classes(train_idx, val_idx, y, min_classes=2)

            try:
                # 전역 6클래스 기준 분포
                cnt_after = np.bincount(y, minlength=len(class_ranges)).astype(int).tolist()
            except Exception:
                cnt_after = []
            batch_stratified_ok = bool(strat_ok)

            # ===== 데이터로더 준비 =====
            X_train = X_raw[train_idx]
            y_train = y[train_idx]
            X_val = X_raw[val_idx]
            y_val = y[val_idx]

            # ─────────────── [ADD] 희소 클래스 보정 ───────────────
            if BALANCE_CLASSES_FLAG:
                try:
                    X_train, y_train = balance_classes(X_train, y_train)
                    _safe_print(f"[BALANCE] applied: train={len(y_train)}, val={len(y_val)}")
                except Exception as e:
                    _safe_print(f"[BALANCE skip] {e}")

            # ─────────────── [ADD] 증강 후 클래스 분포 출력 ───────────────
            try:
                from collections import Counter
                cls_dist = Counter(y_train.tolist())
                _safe_print("[BALANCE RESULT] 클래스별 샘플 수:")
                for cls_id in sorted(cls_dist.keys()):
                    _safe_print(f"  - class {cls_id}: {cls_dist[cls_id]}개")
            except Exception as e:
                _safe_print(f"[BALANCE RESULT skip] {e}")
            # ────────────────────────────────────────────────────────

            sampler = None
            if WEIGHTED_SAMPLER_FLAG:
                try:
                    w_cls = compute_class_weights(y_train, method="effective", beta=0.999)
                    sampler = make_weighted_sampler(y_train, class_weights=w_cls, replacement=True)
                    if sampler is not None:
                        _safe_print("[SAMPLER] WeightedRandomSampler enabled")
                except Exception as e:
                    _safe_print(f"[SAMPLER skip] {e}")

            train_ds = TensorDataset(
                torch.from_numpy(X_train).to(torch.float32),
                torch.from_numpy(y_train).to(torch.long),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_val).to(torch.float32),
                torch.from_numpy(y_val).to(torch.long),
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=32,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )

            # ===== 모델/손실/옵티마 =====
            model = get_model(
                num_classes=len(class_ranges),  # 이제 전역 6클래스 전체
                input_size=feat_dim,
            ).to(DEVICE)

            model_type = getattr(model, "model_type", None) or model.__class__.__name__.lower()

            loss_cfg = get_LOSS()
            if isinstance(loss_cfg, dict):
                loss_name = (loss_cfg.get("name") or "").lower()
            else:
                loss_name = (loss_cfg or "").lower()

            if loss_name == "focal":
                criterion = FocalLoss(gamma=FOCAL_GAMMA).to(DEVICE)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH).to(DEVICE)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(os.getenv("TRAIN_LR", "1e-3")),
                weight_decay=float(os.getenv("TRAIN_WD", "1e-4")),
            )

            scaler = torch.amp.GradScaler(device=device_type) if use_amp_here else None

            best_f1 = -1.0
            best_state = None
            no_improve = 0
            loss_sum = 0.0

            # ===== 학습 루프 =====
            for epoch in range(max_epochs):
                if stop_event is not None and stop_event.is_set():
                    break

                model.train()
                running_loss = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp_here:
                        with torch.amp.autocast(device_type=device_type, enabled=True):
                            logits = model(xb)
                            loss = criterion(logits, yb)
                        scaler.scale(loss).backward()
                        if GRAD_CLIP > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        if GRAD_CLIP > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        optimizer.step()

                    running_loss += float(loss.item())

                # ===== 검증 =====
                model.eval()
                all_preds = []
                all_lbls = []
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(DEVICE, non_blocking=True)
                        yb = yb.to(DEVICE, non_blocking=True)
                        if use_amp_here:
                            with torch.amp.autocast(device_type=device_type, enabled=True):
                                logits = model(xb)
                                loss = criterion(logits, yb)
                        else:
                            logits = model(xb)
                            loss = criterion(logits, yb)
                        val_loss += float(loss.item())
                        preds = torch.argmax(logits, dim=1)
                        all_preds.append(preds.cpu().numpy())
                        all_lbls.append(yb.cpu().numpy())

                if all_preds:
                    preds = np.concatenate(all_preds, axis=0)
                    lbls = np.concatenate(all_lbls, axis=0)
                    try:
                        acc = accuracy_score(lbls, preds)
                    except Exception:
                        acc = 0.0
                    try:
                        f1_val = f1_score(lbls, preds, average="macro", zero_division=0)
                    except Exception:
                        f1_val = 0.0
                else:
                    preds = np.zeros(0, dtype=np.int64)
                    lbls = np.zeros(0, dtype=np.int64)
                    acc = 0.0
                    f1_val = 0.0

                loss_sum = float(running_loss)
                val_loss = float(val_loss)

                _safe_print(
                    f"[EPOCH {epoch+1}/{max_epochs}] {symbol}-{strategy}-w{window} "
                    f"loss={running_loss:.4f} val_loss={val_loss:.4f} acc={acc:.4f} f1={f1_val:.4f}"
                )

                # early stopping
                if f1_val > best_f1 + EARLY_STOP_MIN_DELTA:
                    best_f1 = f1_val
                    best_state = {
                        "model": model.state_dict(),
                        "acc": acc,
                        "f1": f1_val,
                        "val_loss": val_loss,
                        "preds": preds,
                        "lbls": lbls,
                        "val_y": y_val,
                    }
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= EARLY_STOP_PATIENCE:
                    _safe_print(f"[EARLY STOP] {symbol}-{strategy}-w{window} no_improve={no_improve}")
                    break

                if TRAIN_CUDA_EMPTY_EVERY_EP:
                    _safe_empty_cache()

            if best_state is None:
                _log_fail(symbol, strategy, "학습 실패(best_state 없음)")
                continue

            model.load_state_dict(best_state["model"])
            acc = best_state["acc"]
            f1_val = best_state["f1"]
            val_loss = best_state["val_loss"]
            preds = best_state["preds"]
            lbls = best_state["lbls"]
            val_y = best_state["val_y"]

            # ===== bin 정보 정리 및 저장 (전역 6클래스 기준) =====
            try:
                full_ranges_for_bins = full_ranges or get_class_ranges(
                    symbol=symbol, strategy=strategy, group_id=None
                )

                if isinstance(bin_info, dict) and "bin_edges" in bin_info:
                    bin_edges = [float(x) for x in bin_info.get("bin_edges", [])]
                    bin_spans = bin_info.get("bin_spans", [])
                    if bin_edges and (not bin_spans or len(bin_spans) != len(bin_edges) - 1):
                        bin_spans = [
                            float(bin_edges[i + 1] - bin_edges[i])
                            for i in range(len(bin_edges) - 1)
                        ]
                else:
                    bin_edges = [float(lo) for (lo, _) in full_ranges_for_bins] + [
                        float(full_ranges_for_bins[-1][1])
                    ]
                    bin_spans = [float(hi - lo) for (lo, hi) in full_ranges_for_bins]

                cnt_global = np.bincount(val_y, minlength=len(full_ranges_for_bins)).astype(int)
                bin_counts = cnt_global.tolist()
            except Exception:
                bin_edges, bin_spans, bin_counts = [], [], []

            bin_cfg = {
                "TARGET_BINS": int(os.getenv("TARGET_BINS", "8")),
                "OUTLIER_Q_LOW": float(os.getenv("OUTLIER_Q_LOW", "0.01")),
                "OUTLIER_Q_HIGH": float(os.getenv("OUTLIER_Q_HIGH", "0.99")),
                "MAX_BIN_SPAN_PCT": float(os.getenv("MAX_BIN_SPAN_PCT", "8.0")),
                "MIN_BIN_COUNT_FRAC": float(os.getenv("MIN_BIN_COUNT_FRAC", "0.05")),
            }

            stem = os.path.join(
                MODEL_DIR,
                f"{symbol}_{strategy}_{model_type}_w{int(window)}_group{int(group_id) if group_id is not None else 0}_cls{int(len(class_ranges))}",
            )
            meta = {
                "symbol": symbol,
                "strategy": strategy,
                "model": model_type,
                "group_id": int(group_id or 0),
                "num_classes": int(len(class_ranges)),  # 전역 6클래스 수
                "class_ranges": [[float(lo), float(hi)] for (lo, hi) in class_ranges],
                "input_size": int(feat_dim),
                "metrics": {
                    "val_acc": acc,
                    "val_f1": f1_val,
                    "val_loss": val_loss,
                },
                "timestamp": now_kst().isoformat(),
                "model_name": os.path.basename(stem) + ".ptz",
                "window": int(window),
                "recent_cap": int(len(features_only)),
                "engine": "manual",
                "data_flags": {
                    "rows": int(len(df)),
                    "limit": int(_limit),
                    "min": int(_min_required),
                    "augment_needed": bool(augment_needed),
                    "enough_for_training": bool(enough_for_training),
                },
                "train_loss_sum": float(loss_sum),
                "boundary_band": float(BOUNDARY_BAND),
                "cs_argmax": {"enabled": bool(COST_SENSITIVE_ARGMAX), "beta": float(CS_ARG_BETA)},
                "eval_gate": "none",
                "label_repair": repaired_info,
                "bin_edges": bin_edges,
                "bin_counts": bin_counts,
                "bin_spans": bin_spans,
                "bin_cfg": bin_cfg,
            }
            wpath, mpath = _save_model_and_meta(model, stem + ".pt", meta)

            try:
                final_note = f"train_one_model(window={window}, cap={len(features_only)}, engine=manual)"
                if note_msg:
                    final_note += f" | {note_msg}"
                final_note = final_note + return_note

                logger.log_training_result(
                    symbol,
                    strategy,
                    model=os.path.basename(wpath),
                    accuracy=acc,
                    f1=f1_val,
                    loss=val_loss,
                    val_acc=acc,
                    val_f1=f1_val,
                    val_loss=val_loss,
                    engine="manual",
                    window=int(window),
                    recent_cap=int(len(features_only)),
                    rows=int(len(df)),
                    limit=int(_limit),
                    min=int(_min_required),
                    augment_needed=bool(augment_needed),
                    enough_for_training=bool(enough_for_training),
                    note=final_note,
                    source_exchange="BYBIT",
                    status="success",
                    y_true=lbls.tolist() if isinstance(lbls, np.ndarray) else lbls,
                    y_pred=preds.tolist() if isinstance(preds, np.ndarray) else preds,
                    num_classes=len(class_ranges),
                    NUM_CLASSES=int(num_total_classes),
                    class_counts_label_freeze=cnt_before,
                    usable_samples=usable_samples,
                    class_counts_after_assemble=cnt_after,
                    batch_stratified_ok=batch_stratified_ok,
                )
            except Exception:
                pass

            res["windows"].append(int(window))
            res["models"].append(os.path.basename(wpath))

            if IMPORTANCE_ENABLE:
                try:
                    fi = compute_feature_importance(model, features_only, device=DEVICE)
                    save_feature_importance(
                        fi,
                        symbol=symbol,
                        strategy=strategy,
                        window=window,
                        model_name=os.path.basename(wpath),
                    )
                except Exception as e:
                    _safe_print(f"[FI warn] {e}")

            _release_memory(
                train_ds,
                val_ds,
                train_loader,
                val_loader,
                X_train,
                X_val,
                y_train,
                y_val,
                model,
            )

        return res

    except Exception as e:
        _safe_print(f"[EXC] train_one_model {symbol}-{strategy}-g{group_id} → {e}\n{traceback.format_exc()}")
        _log_fail(symbol, strategy, str(e))
        return res

_ENFORCE_FULL_STRATEGY = False
_STRICT_HALT_ON_INCOMPLETE = False
_REQUIRE_AT_LEAST_ONE_MODEL_PER_GROUP = False
_SYMBOL_RETRY_LIMIT = int(os.getenv("SYMBOL_RETRY_LIMIT", "1"))


def _train_full_symbol(
    symbol: str, stop_event: Optional[threading.Event] = None
) -> Tuple[bool, Dict[str, Any]]:
    strategies = ["단기", "중기", "장기"]
    detail = {}
    any_saved = False

    # 수정된 _build_precomputed 는 전략별 df 도 같이 준다
    pre_dfs, pre_feats, pre_lbls = _build_precomputed(symbol)

    for strategy in strategies:
        if stop_event is not None and stop_event.is_set():
            return any_saved, detail
        try:
            cr = get_class_ranges(symbol=symbol, strategy=strategy)
            num_classes = len(cr) if cr else 0
            groups = get_class_groups(num_classes=max(2, num_classes))
            max_gid = len(groups) - 1
            detail[strategy] = {}
            for gid in range(max_gid + 1):
                if stop_event is not None and stop_event.is_set():
                    return any_saved, detail
                gr = get_class_ranges(
                    symbol=symbol, strategy=strategy, group_id=gid
                )
                if not gr or len(gr) < 2:
                    _safe_print(
                        f"[FORCE-TRAIN] {symbol}-{strategy}-g{gid}: cls<2 → 학습 강행"
                    )

                attempts = _SHORT_RETRY if strategy == "단기" else 1
                ok_once = False
                for _ in range(attempts):
                    pf = pre_feats.get(strategy) if isinstance(pre_feats, dict) else None
                    pl = pre_lbls.get(strategy) if isinstance(pre_lbls, dict) else None
                    df_hint = pre_dfs.get(strategy) if isinstance(pre_dfs, dict) else None

                    res = train_one_model(
                        symbol,
                        strategy,
                        group_id=gid,
                        max_epochs=_epochs_for(strategy),
                        stop_event=stop_event,
                        pre_feat=pf,
                        pre_lbl=pl,
                        df_hint=df_hint,
                    )
                    if bool(res and isinstance(res, dict) and res.get("models")):
                        ok_once = True
                        any_saved = True
                        break
                detail[strategy][gid] = ok_once
                time.sleep(0.01)
        except Exception as e:
            logger.log_training_result(
                symbol,
                strategy,
                model="all",
                accuracy=0.0,
                f1=0.0,
                loss=0.0,
                val_acc=0.0,
                val_f1=0.0,
                val_loss=0.0,
                engine="manual",
                window=None,
                recent_cap=None,
                rows=None,
                limit=None,
                min=None,
                augment_needed=None,
                enough_for_training=None,
                note=f"전략 실패: {e}",
                status="failed",
                source_exchange="BYBIT",
            )
            detail[strategy] = {-1: False}
    return any_saved, detail


def train_models(
    symbol_list, stop_event: Optional[threading.Event] = None, ignore_should: bool = False
):
    completed_symbols = []
    partial_symbols = []
    env_force = os.getenv("TRAIN_FORCE_IGNORE_SHOULD", "0") == "1"
    for symbol in symbol_list:
        if stop_event is not None and stop_event.is_set():
            break
        symbol_has_model = _has_any_model_for_symbol(symbol)
        local_ignore = ignore_should or env_force or (not symbol_has_model)
        if not local_ignore:
            if not should_train_symbol(symbol):
                _safe_print(
                    f"[ORDER] skip {symbol} (should_train_symbol=False, models_exist={symbol_has_model})"
                )
                continue
        else:
            _safe_print(f"[order-override] {symbol}: force train")

        trained_complete = False
        for _ in range(max(1, _SYMBOL_RETRY_LIMIT)):
            if stop_event is not None and stop_event.is_set():
                break
            complete, detail = _train_full_symbol(symbol, stop_event=stop_event)
            _safe_print(f"[ORDER] {symbol} → complete={complete} detail={detail}")
            if complete:
                trained_complete = True
                break

        if trained_complete:
            completed_symbols.append(symbol)
            try:
                mark_symbol_trained(symbol)
            except Exception as e:
                _safe_print(f"[mark_symbol_trained err] {e}")
        else:
            partial_symbols.append(symbol)
    return completed_symbols, partial_symbols


def _scan_symbols_from_model_dir() -> List[str]:
    syms = set()
    try:
        for p in glob.glob(os.path.join(MODEL_DIR, f"*_*_*.*")):
            b = os.path.basename(p)
            m = re.match(r"^([A-Z0-9]+)_[^_]+_", b)
            if m:
                syms.add(m.group(1))
        for d in glob.glob(os.path.join(MODEL_DIR, "*")):
            if os.path.isdir(d):
                syms.add(os.path.basename(d))
    except Exception:
        pass
    return sorted(syms)


def _pick_smoke_symbol(candidates: List[str]) -> Optional[str]:
    cand = [s for s in candidates if _has_any_model_for_symbol(s)]
    if cand:
        return sorted(cand)[0]
    pool = _scan_symbols_from_model_dir()
    pool = [s for s in pool if _has_any_model_for_symbol(s)]
    return pool[0] if pool else None


def _run_smoke_predict(predict_fn, symbol: str):
    ok_any = False
    for strat in ["단기", "중기", "장기"]:
        if _has_model_for(symbol, strat):
            try:
                predict_fn(symbol, strat, source="그룹직후(스모크)", model_type=None)
                ok_any = True
            except Exception as e:
                _safe_print(f"[SMOKE fail] {symbol}-{strat}: {e}")
    return ok_any


def _safe_predict_with_timeout(
    predict_fn,
    symbol: str,
    strategy: str,
    source: str = "group_end",
    model_type: str | None = None,
    timeout: float = PREDICT_TIMEOUT_SEC,
) -> bool:
    try:
        pl_clear_stale(lock_key=(symbol, strategy))
    except Exception:
        pass
    try:
        pl_wait_free(max_wait_sec=int(max(1, timeout / 2)), lock_key=(symbol, strategy))
    except Exception:
        pass

    ok = [False]
    err = [None]

    def _run():
        try:
            predict_fn(symbol, strategy, source=source, model_type=model_type)
            ok[0] = True
        except Exception as e:
            err[0] = e

    th = threading.Thread(target=_run, daemon=True, name=f"predict-{symbol}-{strategy}")
    th.start()
    th.join(timeout=float(timeout))
    if th.is_alive():
        return False
    if err[0] is not None:
        raise err[0]
    return ok[0]

def train_symbol_group_loop(
    sleep_sec: int = 0, stop_event: Optional[threading.Event] = None
):
    env_force_ignore = os.getenv("TRAIN_FORCE_IGNORE_SHOULD", "0") == "1"
    env_reset = os.getenv("RESET_GROUP_ORDER_ON_START", "0") == "1"
    force_full_pass = env_force_ignore
    if force_full_pass or env_reset:
        try:
            reset_group_order(0)
        except Exception as e:
            _safe_print(f"[group reset skip] {e}")

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            from predict import predict

            if hasattr(logger, "ensure_train_log_exists"):
                logger.ensure_train_log_exists()
            if hasattr(logger, "ensure_prediction_log_exists"):
                logger.ensure_prediction_log_exists()

            groups = [list(g) for g in SYMBOL_GROUPS]
            if not groups:
                _safe_print("[group] SYMBOL_GROUPS 비어 있음 → 대기")
            else:
                # ✅ 현재 그룹 index 에 맞춰 한 그룹만 처리
                try:
                    cur_idx = get_current_group_index()
                except Exception:
                    cur_idx = 0
                try:
                    cur_idx = int(cur_idx)
                except Exception:
                    cur_idx = 0
                if cur_idx < 0 or cur_idx >= len(groups):
                    cur_idx = 0

                idx = cur_idx
                group = groups[idx]

                if stop_event is not None and stop_event.is_set():
                    break
                _safe_print(f"🚀 [group] {idx+1}/{len(groups)} → {group}")

                try:
                    _set_group_active(True, group_idx=idx, symbols=group)
                    _set_group_train_lock(True, group_idx=idx, symbols=group)
                except Exception as e:
                    _safe_print(f"[GROUP mark warn] {e}")

                # 1) 이 그룹 학습부터
                completed_syms, partial_syms = train_models(
                    group, stop_event=stop_event, ignore_should=force_full_pass
                )
                if stop_event is not None and stop_event.is_set():
                    break

                # 2) 예측 게이트 확인
                try:
                    gate_ok = ready_for_group_predict()
                except Exception as e:
                    _safe_print(f"[PREDICT-GATE warn] {e} -> 게이트 실패로 간주하고 예측 생략")
                    gate_ok = False

                if not gate_ok:
                    _safe_print(
                        f"[PREDICT-SKIP] group{idx+1}: ready_for_group_predict()==False → 학습만 하고 예측은 안 함"
                    )
                else:
                    # 3) 게이트가 True인 경우에만 예측 실행
                    ran_any = False
                    for symbol in group:
                        for strategy in ["단기", "중기", "장기"]:
                            if not _has_model_for(symbol, strategy):
                                _safe_print(f"[PREDICT-SKIP] {symbol}-{strategy}: 모델 없음")
                                continue
                            try:
                                ok = _safe_predict_with_timeout(
                                    predict_fn=predict,
                                    symbol=symbol,
                                    strategy=strategy,
                                    source="그룹직후",
                                    model_type=None,
                                    timeout=PREDICT_TIMEOUT_SEC,
                                )
                                if ok:
                                    ran_any = True
                                else:
                                    _safe_print(
                                        f"[PREDICT TIMEOUT] {symbol}-{strategy} (> {PREDICT_TIMEOUT_SEC}s)"
                                    )
                            except Exception as e:
                                _safe_print(f"[PREDICT FAIL] {symbol}-{strategy}: {e}")

                    # 하나도 실행이 안 됐을 때만 스모크
                    if not ran_any:
                        cand_symbol = _pick_smoke_symbol(group)
                        if cand_symbol:
                            _safe_print(f"[SMOKE] fallback predict for {cand_symbol}")
                            try:
                                _run_smoke_predict(predict, cand_symbol)
                            except Exception as e:
                                _safe_print(f"[SMOKE fail] {e}")

                    if ran_any:
                        try:
                            mark_group_predicted()
                        except Exception as e:
                            _safe_print(f"[mark_group_predicted err] {e}")

                # 4) 그룹 단위 정리
                try:
                    from logger import flush_gwanwoo_summary
                    flush_gwanwoo_summary()
                except Exception:
                    pass

                try:
                    # 예측을 했든 안 했든 여기선 게이트만 닫아준다
                    close_predict_gate(note=f"train:group{idx+1}_end")
                except Exception as e:
                    _safe_print(f"[gate close warn] {e}")

                try:
                    _set_group_active(False)
                    _set_group_train_lock(False)
                except Exception as e:
                    _safe_print(f"[GROUP clear warn] {e}")

                # 그룹 사이 휴식
                if sleep_sec > 0:
                    for _ in range(sleep_sec):
                        if stop_event is not None and stop_event.is_set():
                            break
                        time.sleep(1)
                    if stop_event is not None and stop_event.is_set():
                        break

            _safe_print("✅ group pass done")
            try:
                from logger import flush_gwanwoo_summary
                flush_gwanwoo_summary()
            except Exception:
                pass
            try:
                close_predict_gate(note="train:group_pass_done")
            except Exception as e:
                _safe_print(f"[gate close warn] {e}")

            if force_full_pass and not env_force_ignore:
                force_full_pass = False
        except Exception as e:
            _safe_print(f"[group loop err] {e}\n{traceback.format_exc()}")

        _safe_print("💓 heartbeat")
        time.sleep(max(1, int(os.getenv("TRAIN_LOOP_IDLE_SEC", "3"))))


_TRAIN_LOOP_THREAD: Optional[threading.Thread] = None
_TRAIN_LOOP_STOP: Optional[threading.Event] = None
_TRAIN_LOOP_LOCK = threading.Lock()


def start_train_loop(force_restart: bool = False, sleep_sec: int = 0):
    global _TRAIN_LOOP_THREAD, _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive():
            if not force_restart:
                _safe_print("ℹ️ already running")
                return False
            stop_train_loop(timeout=30)
        _TRAIN_LOOP_STOP = threading.Event()

        def _runner():
            try:
                train_symbol_group_loop(sleep_sec=sleep_sec, stop_event=_TRAIN_LOOP_STOP)
            finally:
                _safe_print("ℹ️ train loop exit")

        _TRAIN_LOOP_THREAD = threading.Thread(
            target=_runner, daemon=True
        )
        _TRAIN_LOOP_THREAD.start()
        _safe_print("✅ train loop started")
        return True


def stop_train_loop(timeout: int | float | None = 30):
    global _TRAIN_LOOP_THREAD, _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        # 이미 루프가 없으면 그대로 종료
        if _TRAIN_LOOP_THREAD is None or not _TRAIN_LOOP_THREAD.is_alive():
            _safe_print("ℹ️ no loop")
            return True
        if _TRAIN_LOOP_STOP is None:
            _safe_print("⚠️ no stop event")
            return False

        # 멈추기 요청
        _TRAIN_LOOP_STOP.set()
        _TRAIN_LOOP_THREAD.join(timeout=timeout)

        # ⛔️ 수정 핵심: 아직 멈추지 않았으면 절대 QWIPE 하지 않는다
        if _TRAIN_LOOP_THREAD.is_alive():
            _safe_print("⚠️ stop timeout — 학습이 완전히 멈추지 않음 → 초기화(QWIPE) 생략")
            return False

        # 완전히 멈춘 경우에만 종료 처리
        _TRAIN_LOOP_THREAD = None
        _TRAIN_LOOP_STOP = None
        _safe_print("✅ loop stopped (safe state)")
        return True


def request_stop() -> bool:
    global _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_STOP is None:
            return True
        _TRAIN_LOOP_STOP.set()
        return True


def is_loop_running() -> bool:
    with _TRAIN_LOOP_LOCK:
        return bool(
            _TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive()
        )


def train_symbol(symbol: str, strategy: str, group_id: int | None = None) -> dict:
    res = train_one_model(symbol=symbol, strategy=strategy, group_id=group_id)
    try:
        if res.get("models"):
            # 학습 완료 표시는 그대로
            mark_symbol_trained(symbol)

            # ✅ 바로 예측하지 말고, 게이트가 열려 있을 때만 예측
            try:
                gate_ok = ready_for_group_predict()
            except Exception:
                gate_ok = False

            if gate_ok and not (_is_group_active_file() or _is_group_lock_file()):
                try:
                    from predict import predict
                    _safe_predict_with_timeout(
                        predict_fn=predict,
                        symbol=symbol,
                        strategy=strategy,
                        source="train_symbol",
                        model_type=None,
                        timeout=PREDICT_TIMEOUT_SEC,
                    )
                except Exception:
                    # 예측 실패해도 학습은 성공이므로 조용히 패스
                    pass
            else:
                _safe_print(
                    f"[PREDICT-SKIP] {symbol}-{strategy}: 게이트 닫힘이거나 그룹 학습 중이라 예측 생략"
                )
    except Exception:
        pass
    return res


def train_group(group_id: int | None = None) -> dict:
    idx = get_current_group_index() if group_id is None else int(group_id)
    symbols = (
        get_current_group_symbols()
        if group_id is None
        else (SYMBOL_GROUPS[idx] if 0 <= idx < len(SYMBOL_GROUPS) else [])
    )
    out = {"group_index": idx, "symbols": symbols, "results": {}}

    try:
        _set_group_active(True)
        _set_group_train_lock(True, group_idx=idx, symbols=symbols)
    except Exception as e:
        _safe_print(f"[GROUP mark warn] {e}")

    completed, partial = train_models(
        symbols, stop_event=None, ignore_should=False
    )
    out["completed"] = completed
    out["partial"] = partial

    # 🔒 여기서도 게이트가 True일 때만 예측
    try:
        gate_ok = ready_for_group_predict()
    except Exception:
        gate_ok = False

    if gate_ok:
        try:
            from predict import predict

            ran_any = False
            for s in symbols:
                for strat in ["단기", "중기", "장기"]:
                    if _has_model_for(s, strat):
                        try:
                            ok = _safe_predict_with_timeout(
                                predict_fn=predict,
                                symbol=s,
                                strategy=strat,
                                source="train_group",
                                model_type=None,
                                timeout=PREDICT_TIMEOUT_SEC,
                            )
                            ran_any = ran_any or ok
                        except Exception:
                            pass
            if ran_any:
                try:
                    mark_group_predicted()
                except Exception:
                    pass
        finally:
            try:
                from logger import flush_gwanwoo_summary
                flush_gwanwoo_summary()
            except Exception:
                pass
            try:
                close_predict_gate(note=f"train_group:idx{idx}_end")
            except Exception:
                pass
    else:
        _safe_print(f"[PREDICT-SKIP] train_group idx={idx}: 게이트가 False라 예측은 생략")

    try:
        _set_group_active(False)
        _set_group_train_lock(False)
    except Exception as e:
        _safe_print(f"[GROUP clear warn] {e}")

    return out


def train_all() -> dict:
    summary = {"groups": []}
    for gid, group in enumerate(SYMBOL_GROUPS):
        res = train_group(group_id=gid)
        summary["groups"].append(res)
    return summary


def continue_from_failure(limit: int = 50) -> dict:
    tried = []
    ok = False
    err = None
    try:
        import failure_learn as FL

        tried.append("failure_learn.run")
        ok = bool(FL.run(limit=limit))
    except Exception as e1:
        err = str(e1)
        try:
            import failure_trainer as FT

            tried.append("failure_trainer.retrain_failures")
            ok = bool(FT.retrain_failures(limit=limit))
        except Exception as e2:
            err = f"{err} | {e2}"
    return {"ok": ok, "tried": tried, "error": err}

def _apply_real_balance(X_train: np.ndarray, y_train: np.ndarray,
                        min_count: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    YOPO 학습 안정화를 위한 '진짜' 클래스 균형 함수.
    - 너무 적은 클래스(샘플 < min_count)는 자동 제외
    - 남은 클래스는 최소 min_count까지 오버샘플링
    """
    if len(y_train) == 0:
        return X_train, y_train

    # 클래스 분포 계산
    from collections import Counter
    cnt = Counter(y_train.tolist())

    # 1) 너무 적은 클래스 제거
    valid_classes = [c for c, v in cnt.items() if v >= min_count]
    if len(valid_classes) < 2:
        # 학습 가능한 최소 클래스가 안 되면 원본 유지
        return X_train, y_train

    # 2) 유효한 클래스만 남기기
    mask = np.isin(y_train, valid_classes)
    X_filtered = X_train[mask]
    y_filtered = y_train[mask]

    # 3) 최소 샘플 수까지 오버샘플
    X_bal, y_bal = [], []
    max_count = max([cnt[c] for c in valid_classes])

    for cls in valid_classes:
        idx = np.where(y_filtered == cls)[0]
        cur = len(idx)

        if cur == 0:
            continue

        # 필요한 만큼 반복해서 붙여넣기 (오버샘플링)
        reps = max_count // cur
        rem = max_count % cur

        X_bal.append(np.repeat(X_filtered[idx], reps, axis=0))
        y_bal.append(np.repeat(y_filtered[idx], reps, axis=0))

        if rem > 0:
            extra = np.random.choice(idx, rem, replace=True)
            X_bal.append(X_filtered[extra])
            y_bal.append(y_filtered[extra])

    X_bal = np.concatenate(X_bal, axis=0)
    y_bal = np.concatenate(y_bal, axis=0)

    return X_bal.astype(np.float32), y_bal.astype(np.int64)



if __name__ == "__main__":
    try:
        start_train_loop(force_restart=True, sleep_sec=0)
    except Exception as e:
        _safe_print(f"[MAIN] err: {e}")
