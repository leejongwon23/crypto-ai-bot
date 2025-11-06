# train.py — SPEED v2.6 FINAL (NO-SKIP AUTO-TRAIN 강화판, 희소구간 강제학습, GROUP_ACTIVE + GROUP_TRAIN_LOCK, 라벨/분포 복구 강화, 로그스키마 정합 + 5종 진단로그)
# -*- coding: utf-8 -*-
import sitecustomize
import os, time, glob, shutil, json, random, traceback, threading, gc, csv
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import numpy as np, pandas as pd, pytz, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# ---------- 공용 메모리 유틸 ----------
def _safe_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _release_memory(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    gc.collect()
    _safe_empty_cache()

# ---------- 기본 환경/시드 ----------
def _set_default_thread_env(n: str, v: int):
    if os.getenv(n) is None: os.environ[n] = str(v)
for _n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","TORCH_NUM_THREADS"):
    _set_default_thread_env(_n, int(os.getenv("CPU_THREAD_CAP","1")))
try: torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","1")))
except: pass

# --- CPU 최적화 ---
try: torch.set_num_interop_threads(1)
except: pass
try: torch.backends.mkldnn.enabled = True
except: pass
try:
    torch.use_deterministic_algorithms(False)
    torch.set_deterministic_debug(False)
except: pass

def set_global_seed(s:int=20240101):
    os.environ["PYTHONHASHSEED"]=str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    try:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    except: pass
set_global_seed(int(os.getenv("GLOBAL_SEED","20240101")))

# ---------- 외부 의존 ----------
from model_io import save_model

# [풀백] data.utils → utils
try:
    from data.utils import (
        get_kline_by_strategy, compute_features, SYMBOL_GROUPS,
        should_train_symbol, mark_symbol_trained, ready_for_group_predict, mark_group_predicted,
        reset_group_order, CacheManager as DataCacheManager,
        compute_features_multi
    )
except Exception:
    from utils import (
        get_kline_by_strategy, compute_features, SYMBOL_GROUPS,
        should_train_symbol, mark_symbol_trained, ready_for_group_predict, mark_group_predicted,
        reset_group_order, CacheManager as DataCacheManager
    )
    # 선택적 신규 API 폴백
    def compute_features_multi(symbol: str, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        out = {}
        for s in ("단기","중기","장기"):
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
        def get_current_group_index(): return 0
        def get_current_group_symbols(): return SYMBOL_GROUPS[0] if SYMBOL_GROUPS else []

# NOTE: 리포 구조에 맞춰 경로 정정 (robust dual import)
try:
    from model.base_model import get_model, freeze_backbone, unfreeze_last_k_layers
except Exception:
    from base_model import get_model
    def freeze_backbone(model): return None
    def unfreeze_last_k_layers(model, k:int=1): return None

from feature_importance import compute_feature_importance, save_feature_importance
from failure_db import insert_failure_record, ensure_failure_db
import logger
from config import (
    get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups, get_class_ranges, set_NUM_CLASSES,
    STRATEGY_CONFIG, get_QUALITY, get_LOSS, BOUNDARY_BAND, get_TRAIN_LOG_PATH
)

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

# ==== [ADD] train 로그 경로/헤더 보장 ====
# 운영 CSV 스키마(사용자 제공)와 정합되도록 확장 + 진단 5종 필드 추가
DEFAULT_TRAIN_HEADERS = [
    "timestamp","symbol","strategy","model",
    "val_acc","val_f1","val_loss","engine","window","recent_cap",
    "rows","limit","min","augment_needed","enough_for_training",
    "note","source_exchange","status",
    # 여분(옵셔널)
    "accuracy","f1","loss","y_true","y_pred","num_classes",
    # === 진단 5종 ===
    "NUM_CLASSES","class_counts_label_freeze","usable_samples","class_counts_after_assemble","batch_stratified_ok"
]
try:
    from logger import TRAIN_HEADERS
    TRAIN_HEADERS = list(dict.fromkeys(list(TRAIN_HEADERS) + DEFAULT_TRAIN_HEADERS))
except Exception:
    TRAIN_HEADERS = DEFAULT_TRAIN_HEADERS

# config 가 /persistent/... 를 돌려줘도 여기서 안전폴더로 치환
_raw_train_log_path = get_TRAIN_LOG_PATH()
if _raw_train_log_path.startswith("/persistent"):
    TRAIN_LOG = os.path.join(BASE_PERSIST_DIR, os.path.relpath(_raw_train_log_path, "/persistent"))
else:
    TRAIN_LOG = _raw_train_log_path
try:
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
except Exception:
    # 마지막 폴백: BASE_PERSIST_DIR 밑에 두기
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
    """사용자 CSV 스키마에 최대한 맞춰 필드 매핑/보정"""
    r = {k: row.get(k, None) for k in TRAIN_HEADERS}
    # 기존 키 -> 새 키 매핑
    if r.get("val_acc") is None and row.get("accuracy") is not None:
        r["val_acc"] = row.get("accuracy")
    if r.get("val_f1") is None and row.get("f1") is not None:
        r["val_f1"] = row.get("f1")
    if r.get("val_loss") is None and row.get("loss") is not None:
        r["val_loss"] = row.get("loss")
    # 기본값 보정
    r.setdefault("engine", row.get("engine", "manual"))
    r.setdefault("source_exchange", row.get("source_exchange", "BYBIT"))
    return r

def _append_train_log(row: dict):
    try:
        _ensure_train_log()
        with open(TRAIN_LOG, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRAIN_HEADERS, extrasaction="ignore")
            w.writerow(_normalize_train_row(row))
    except Exception as e:
        print(f"[경고] train_log 기록 실패: {e}")

# logger.log_training_result 패치: 원래 호출 + 파일 기록(예외는 경고)
if not getattr(logger, "_patched_train_log", False):
    _orig_ltr = getattr(logger, "log_training_result", None)
    def _log_training_result_patched(*args, **kw):
        if callable(_orig_ltr):
            try: _orig_ltr(*args, **kw)
            except Exception as e: print(f"[경고] logger.log_training_result 실패: {e}")
        row = dict(kw)
        row.setdefault("timestamp", datetime.now(pytz.timezone("Asia/Seoul")).isoformat())
        _append_train_log(row)
    logger.log_training_result = _log_training_result_patched
    logger._patched_train_log = True
# ====================================================

# ✅ 예측 게이트: 안전 임포트(없으면 no-op)
try:
    from predict import close_predict_gate
except Exception:
    def close_predict_gate(*a, **k): return None

# ✅ 학습 직후 자동 예측 트리거 (없으면 no-op)
try:
    from predict_trigger import run_after_training
except Exception:
    def run_after_training(symbol, strategy, *a, **k): return False

# [가드] data_augmentation (없으면 원본 그대로 통과)
try:
    from data_augmentation import balance_classes
except Exception:
    def balance_classes(X: np.ndarray, y: np.ndarray, num_classes: int):
        return X, y

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
    def pl_clear_stale(lock_key=None): return None
    def pl_wait_free(max_wait_sec: int, lock_key=None): return True

# ───────── 파인튜닝 로더(선택) ─────────
try:
    from model_io import load_for_finetune as _load_for_finetune
except Exception:
    _load_for_finetune = None

# ---------- 전역 상수 ----------
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_DIR 은 위에서 안전경로로 이미 만들었음
NUM_CLASSES=get_NUM_CLASSES()
FEATURE_INPUT_SIZE=get_FEATURE_INPUT_SIZE()

_MAX_ROWS_FOR_TRAIN=int(os.getenv("TRAIN_MAX_ROWS","1200"))
_BATCH_SIZE=int(os.getenv("TRAIN_BATCH_SIZE","128"))
_NUM_WORKERS=int(os.getenv("TRAIN_NUM_WORKERS","0"))  # 누수 방지: 0 권장
_PIN_MEMORY=False                                     # 누수 방지: False 권장
_PERSISTENT=False
SMART_TRAIN = os.getenv("SMART_TRAIN","1")=="1"
LABEL_SMOOTH = float(os.getenv("LABEL_SMOOTH","0.05"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP_NORM","1.0"))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA","2.0"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE","2"))
EARLY_STOP_MIN_DELTA = float(os.getenv("EARLY_STOP_MIN_DELTA","0.0001"))

# AMP 옵션
USE_AMP = os.getenv("USE_AMP","1")=="1"
TRAIN_CUDA_EMPTY_EVERY_EP = os.getenv("TRAIN_CUDA_EMPTY_EVERY_EP","1")=="1"

def _as_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1","true","yes","on")

COST_SENSITIVE_ARGMAX = _as_bool_env("COST_SENSITIVE_ARGMAX", True)
CS_ARG_BETA = float(os.getenv("CS_ARG_BETA","1.0"))

def _epochs_for(strategy:str)->int:
    if strategy=="단기": return int(os.getenv("EPOCHS_SHORT","24"))
    if strategy=="중기": return int(os.getenv("EPOCHS_MID","12"))
    if strategy=="장기": return int(os.getenv("EPOCHS_LONG","12"))
    return 24

# (참고용: 운영 게이트엔 미사용)
EVAL_MIN_F1_SHORT = float(os.getenv("EVAL_MIN_F1_SHORT", "0.10"))
EVAL_MIN_F1_MID   = float(os.getenv("EVAL_MIN_F1_MID",   "0.50"))
EVAL_MIN_F1_LONG  = float(os.getenv("EVAL_MIN_F1_LONG",  "0.45"))
_SHORT_RETRY      = int(os.getenv("SHORT_STRATEGY_RETRY", "3"))
def _min_f1_for(strategy:str)->float:
    return EVAL_MIN_F1_SHORT if strategy=="단기" else (EVAL_MIN_F1_MID if strategy=="중기" else EVAL_MIN_F1_LONG)

now_kst=lambda: datetime.now(pytz.timezone("Asia/Seoul"))

# ✅ 그룹 끝난 직후 예측 허용 스위치
PREDICT_OVERRIDE_ON_GROUP_END = _as_bool_env("PREDICT_OVERRIDE_ON_GROUP_END", True)

# ───────── 예측 타임아웃/강제 옵션 ─────────
PREDICT_FORCE_AFTER_GROUP = _as_bool_env("PREDICT_FORCE_AFTER_GROUP", True)
PREDICT_TIMEOUT_SEC = float(os.getenv("PREDICT_TIMEOUT_SEC","180"))

# === 중요도 저장 플래그 ===
IMPORTANCE_ENABLE = os.getenv("IMPORTANCE_ENABLE", "1") == "1"

# === GROUP_ACTIVE + GROUP_TRAIN_LOCK ===
# PERSIST_DIR, RUN_DIR 은 위에서 이미 안전하게 만들었음
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
        try: print(f"[GROUP_ACTIVE warn] {e}", flush=True)
        except: pass

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
        try: print(f"[GROUP_LOCK warn] {e}", flush=True)
        except: pass

def _is_group_active_file() -> bool:
    try: return os.path.exists(GROUP_ACTIVE_PATH)
    except Exception: return False

def _is_group_lock_file() -> bool:
    try: return os.path.exists(GROUP_TRAIN_LOCK)
    except Exception: return False

def _maybe_insert_failure(payload:dict, feature_vector:Optional[List[Any]] = None):
    try:
        if not ready_for_group_predict(): return
        insert_failure_record(payload, feature_vector=(feature_vector or []))
    except Exception as e: 
        try: print(f"[FAILREC skip] {e}", flush=True)
        except: pass

def _safe_print(msg): 
    try: print(msg, flush=True)
    except: pass

def _stem(p:str)->str: 
    import os
    return os.path.splitext(p)[0]

def _save_model_and_meta(model:nn.Module,path_pt:str,meta:dict):
    os.makedirs(os.path.dirname(path_pt),exist_ok=True)
    weight=_stem(path_pt)+".ptz"
    save_model(weight, model.state_dict())
    meta_path = _stem(path_pt)+".meta.json"
    with open(meta_path,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=None,separators=(",",":"))
    return weight, meta_path

def coverage_split_indices(y, val_frac=0.20, min_coverage=0.60, stride=50, max_windows=200, num_classes=None):
    y = np.asarray(y).astype(int); n=len(y); val_len=max(1,int(round(n*val_frac)))
    if num_classes is None:
        uniq=np.unique(y); num_classes=(max(len(uniq), int(uniq.max())+1) if (uniq.size and uniq.min()>=0) else len(uniq))
    tried=0; best=None; end=n
    while end-val_len>=0 and tried<max_windows:
        start=end-val_len; yv=y[start:end]; cnt=Counter(yv.tolist())
        coverage=len([1 for v in cnt.values() if v>0]) / max(1,num_classes)
        if best is None or coverage>best[0]: best=(coverage,start,end,cnt)
        if coverage>=float(min_coverage): break
        end-=int(max(1,stride)); tried+=1
    if best is None:
        start, end = max(0,n-val_len), n; cnt=Counter(y[start:end].tolist()); coverage=len(cnt)/max(1,num_classes)
    else:
        coverage,start,end,cnt=best
    val_idx=np.arange(start,end); train_idx=np.concatenate([np.arange(0,start), np.arange(end,n)],axis=0)
    _safe_print(f"[VAL COVER] {len(cnt)}/{num_classes} ({coverage:.2f}) window={start}:{end} size={len(val_idx)}")
    return train_idx, val_idx

def _log_skip(symbol,strategy,reason):
    logger.log_training_result(
        symbol, strategy, model="all",
        accuracy=0.0, f1=0.0, loss=0.0,
        val_acc=0.0, val_f1=0.0, val_loss=0.0,
        engine="manual", window=None, recent_cap=None,
        rows=None, limit=None, min=None,
        augment_needed=None, enough_for_training=None,
        note=reason, source_exchange="BYBIT", status="skipped"
    )
    _maybe_insert_failure({"symbol":symbol,"strategy":strategy,"model":"all","predicted_class":-1,"success":False,"rate":0.0,"reason":reason},feature_vector=[])

def _log_fail(symbol,strategy,reason):
    logger.log_training_result(
        symbol, strategy, model="all",
        accuracy=0.0, f1=0.0, loss=0.0,
        val_acc=0.0, val_f1=0.0, val_loss=0.0,
        engine="manual", window=None, recent_cap=None,
        rows=None, limit=None, min=None,
        augment_needed=None, enough_for_training=None,
        note=reason, source_exchange="BYBIT", status="failed"
    )
    _maybe_insert_failure({"symbol":symbol,"strategy":strategy,"model":"all","predicted_class":-1,"success":False,"rate":0.0,"reason":reason},feature_vector=[])

def _has_any_model_for_symbol(symbol: str) -> bool:
    exts=(".ptz",".safetensors",".pt")
    try:
        if any(glob.glob(os.path.join(MODEL_DIR, f"{symbol}_*{e}")) for e in exts): return True
        d=os.path.join(MODEL_DIR, symbol)
        return any(glob.glob(os.path.join(d,"*","*"+e)) for e in exts) if os.path.isdir(d) else False
    except: return False

def _has_model_for(symbol: str, strategy: str) -> bool:
    exts=(".ptz",".safetensors",".pt")
    try:
        if any(glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*{e}")) for e in exts): return True
        d=os.path.join(MODEL_DIR, symbol, strategy)
        return any(glob.glob(os.path.join(d,"*"+e)) for e in exts) if os.path.isdir(d) else False
    except: return False

# ---------- 전략 간 피처/라벨 패스다운 ----------
def _build_precomputed(symbol: str) -> tuple[Optional[pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    try:
        df = get_kline_by_strategy(symbol, "단기")
        if df is None or df.empty:
            return None, {}, {}
        feats = compute_features_multi(symbol, df)
        lbls_all = make_all_horizon_labels(df=df, symbol=symbol, group_id=None)  # keys: "4h","1d","7d"
        pre_lbl = {}
        for strat, key in (("단기","4h"),("중기","1d"),("장기","7d")):
            v = lbls_all.get(key, None)
            pre_lbl[strat] = v if v is not None else None
        return df, feats, pre_lbl
    except Exception:
        return None, {}, {}

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

# === [ADD] 최소 2클래스 확보 보조 ===
def _expand_to_neighbor_classes(gidx: List[int], present_g: set[int]) -> List[int]:
    """
    원래 버전: 현재 그룹에 실제로 존재하는 클래스가 1개뿐이면 양옆 bin을 끌어와서 2개 이상으로 맞췄음.
    지금 버전: 희소클래스 자체가 의미 있는 케이스이므로 '확장'하지 않고 원래 gidx 그대로 쓴다.
    즉, 그룹 설계대로 간다.
    """
    return gidx

def _rebuild_samples_with_keepset(fv: np.ndarray, labels: np.ndarray, window: int,
                                  keep_set: set[int], to_local: Dict[int,int]) -> Tuple[np.ndarray, np.ndarray]:
    X_raw, y = [], []
    n = len(fv)
    for i in range(n - window):
        yi = i + window - 1
        if yi < 0 or yi >= len(labels): continue
        lab_g = int(labels[yi])
        if (lab_g < 0) or (lab_g not in keep_set): continue
        lab = to_local.get(lab_g, None)
        if lab is None: continue
        X_raw.append(fv[i:i + window]); y.append(lab)
    if not X_raw: return np.empty((0, window, fv.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.array(X_raw, dtype=np.float32), np.array(y, dtype=np.int64)

def _synthesize_minority_if_needed(X_raw: np.ndarray, y: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    원래 버전: 클래스가 하나만 남으면 synthetic 시퀀스를 만들어서 최소 2개 클래스로 맞춤.
    지금 버전: '클래스가 하나만 있는 것 자체가 시장구간의 정보'라고 보는 설정이므로 절대 synthetic을 추가하지 않는다.
    그대로 돌려보내고, 세 번째 리턴값도 False로.
    """
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

def train_one_model(symbol, strategy, group_id=None, max_epochs: Optional[int] = None,
                    stop_event: Optional[threading.Event] = None,
                    pre_feat: Optional[pd.DataFrame] = None,
                    pre_lbl: Optional[tuple] = None) -> Dict[str, Any]:
    if max_epochs is None: max_epochs = _epochs_for(strategy)
    res={"symbol":symbol,"strategy":strategy,"group_id":int(group_id or 0),"windows":[], "models": []}
    try:
        ensure_failure_db()
        _safe_print(f"✅ train_one_model {symbol}-{strategy}-g{group_id}")

        # 데이터
        df = None
        if pre_feat is not None or pre_lbl is not None:
            try: df = get_kline_by_strategy(symbol, "단기")
            except Exception: df = None
        if df is None:
            df=get_kline_by_strategy(symbol,strategy)
        if df is None or df.empty: _log_skip(symbol,strategy,"데이터 없음"); return res

        cfg=STRATEGY_CONFIG.get(strategy,{})
        _limit=int(cfg.get("limit",300)); _min_required=max(60,int(_limit*0.90))
        _attrs=getattr(df,"attrs",{}) if df is not None else {}
        augment_needed=bool(_attrs.get("augment_needed", len(df)<_limit))
        enough_for_training=bool(_attrs.get("enough_for_training", len(df)>=_min_required))

        # 피처: 사전계산 우선
        if isinstance(pre_feat, pd.DataFrame):
            feat = pre_feat
        elif isinstance(pre_feat, dict) and pre_feat.get(strategy, None) is not None:
            feat = pre_feat[strategy]
        else:
            feat=compute_features(symbol, df, strategy)
        if feat is None or getattr(feat,"empty",True): _log_skip(symbol,strategy,"피처 없음"); return res

        # 라벨: 글로벌 기준 (가변 반환 대응: 3 or 4 or 6)
        bin_info = None
        if isinstance(pre_lbl, tuple) and len(pre_lbl) in (3,4,6):
            if len(pre_lbl) == 6:
                gains, labels, class_ranges_used_global, be, bc, bs = pre_lbl
                bin_info = {
                    "bin_edges": be.tolist() if hasattr(be,"tolist") else list(be),
                    "bin_counts": bc.tolist() if hasattr(bc,"tolist") else list(bc),
                    "bin_spans": bs.tolist() if hasattr(bs,"tolist") else list(bs),
                }
            elif len(pre_lbl) == 4:
                gains, labels, class_ranges_used_global, bin_info = pre_lbl
            else:
                gains, labels, class_ranges_used_global = pre_lbl
        elif isinstance(pre_lbl, dict) and pre_lbl.get(strategy, None) is not None:
            val = pre_lbl[strategy]
            if isinstance(val, (list, tuple)) and len(val) in (3,4,6):
                if len(val) == 6:
                    gains, labels, class_ranges_used_global, be, bc, bs = val
                    bin_info = {
                        "bin_edges": be.tolist() if hasattr(be,"tolist") else list(be),
                        "bin_counts": bc.tolist() if hasattr(bc,"tolist") else list(bc),
                        "bin_spans": bs.tolist() if hasattr(bs,"tolist") else list(bs),
                    }
                elif len(val) == 4:
                    gains, labels, class_ranges_used_global, bin_info = val
                else:
                    gains, labels, class_ranges_used_global = val
            else:
                _log_skip(symbol,strategy,"사전 라벨 구조 오류"); return res
        else:
            res_labels = make_labels(df=df, symbol=symbol, strategy=strategy, group_id=None)
            if isinstance(res_labels, (list, tuple)) and len(res_labels) in (3,4,6):
                if len(res_labels) == 6:
                    gains, labels, class_ranges_used_global, be, bc, bs = res_labels
                    bin_info = {
                        "bin_edges": be.tolist() if hasattr(be,"tolist") else list(be),
                        "bin_counts": bc.tolist() if hasattr(bc,"tolist") else list(bc),
                        "bin_spans": bs.tolist() if hasattr(bs,"tolist") else list(bs),
                    }
                elif len(res_labels) == 4:
                    gains, labels, class_ranges_used_global, bin_info = res_labels
                else:
                    gains, labels, class_ranges_used_global = res_labels
            else:
                _log_skip(symbol,strategy,"라벨 생성 실패"); return res

        if (not isinstance(labels, np.ndarray)) or labels.size == 0:
            _log_skip(symbol,strategy,"라벨 없음"); return res

        # === [ADD] 유효 클래스 부족 시 1회 재라벨링 시도 ===
        try:
            uniq0 = _uniq_nonneg(labels)
        except Exception:
            uniq0 = 0
        if uniq0 <= 1:
            _safe_print(f"[LABEL RETRY] uniq<=1 → rebuild via make_labels() once ({symbol}-{strategy})")
            res_try = _rebuild_labels_once(df=df, symbol=symbol, strategy=strategy)
            if isinstance(res_try, (list, tuple)) and len(res_try) in (3,4,6):
                if len(res_try) == 6:
                    gains2, labels2, class_ranges2, be2, bc2, bs2 = res_try
                    bin_info2 = {
                        "bin_edges": be2.tolist() if hasattr(be2,"tolist") else list(be2),
                        "bin_counts": bc2.tolist() if hasattr(bc2,"tolist") else list(bc2),
                        "bin_spans": bs2.tolist() if hasattr(bs2,"tolist") else list(bs2),
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
            else:
                _safe_print("[LABEL RETRY FAIL] make_labels() second call failed")

        # --------- 그룹 로컬 재매핑 ---------
        all_ranges_full = get_class_ranges(symbol=symbol, strategy=strategy, group_id=None)
        groups_full = get_class_groups(num_classes=len(all_ranges_full))
        gid = int(group_id or 0)
        gidx = groups_full[gid] if 0 <= gid < len(groups_full) else list(range(len(all_ranges_full)))
        keep_set = set(gidx)
        class_ranges = [all_ranges_full[i] for i in gidx]
        to_local = {g:i for i, g in enumerate(gidx)}
        # -----------------------------------

        # 마스크/분포 진단(라벨 프리즈 단계 분포)
        mask_cnt=int((labels<0).sum())
        _safe_print(f"[LABELS] total={len(labels)} masked={mask_cnt} ({mask_cnt/max(1,len(labels)):.2%}) BOUNDARY_BAND=±{BOUNDARY_BAND}")
        try:
            cnt_before = np.bincount(labels[labels>=0], minlength=len(all_ranges_full)).astype(int).tolist()
        except Exception:
            cnt_before = []

        # ★ LabelStats 로그
        try:
            n_bins = int(len(all_ranges_full))
            empty_idx = [i for i, c in enumerate(cnt_before) if int(c) == 0] if cnt_before else []
            num_classes_effective = int(np.unique(labels[labels >= 0]).size) if labels.size else 0
            logger.log_training_result(
                symbol, strategy, model="all",
                accuracy=None, f1=None, loss=None,
                val_acc=None, val_f1=None, val_loss=None,
                engine="manual", window=None, recent_cap=None,
                rows=int(len(df)), limit=int(STRATEGY_CONFIG.get(strategy,{}).get("limit",300)),
                min=int(max(60,int(STRATEGY_CONFIG.get(strategy,{}).get("limit",300)*0.90))),
                augment_needed=bool(augment_needed), enough_for_training=bool(enough_for_training),
                note=f"[LabelStats] bins={n_bins}, empty={len(empty_idx)}, classes={num_classes_effective}, empty_idx={empty_idx[:8]}",
                source_exchange="BYBIT", status="info",
                NUM_CLASSES=int(n_bins),
                class_counts_label_freeze=cnt_before
            )
        except Exception:
            pass

        # 특징행렬 정제
        drop_cols = [c for c in ("timestamp","strategy","symbol") if c in feat.columns]
        feat_num = feat.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        features_only = feat_num.replace([np.inf,-np.inf], np.nan).fillna(0.0)
        feat_dim = int(getattr(features_only,"shape",[0,FEATURE_INPUT_SIZE])[1]) or int(FEATURE_INPUT_SIZE)
        if len(features_only)>_MAX_ROWS_FOR_TRAIN or len(labels)>_MAX_ROWS_FOR_TRAIN:
            cut=min(_MAX_ROWS_FOR_TRAIN,len(features_only),len(labels))
            features_only=features_only.iloc[-cut:,:]; labels=labels[-cut:]

        # 윈도우 후보
        try:
            top_windows = find_best_windows(symbol, strategy, window_list=[16,20,24,28,32], top_k=3, group_id=group_id)
        except Exception:
            try: top_windows=[int(find_best_window(symbol,strategy,window_list=[16,20,24,28,32],group_id=group_id))]
            except: top_windows=[20]
        top_windows=[int(max(5,w)) for w in top_windows if isinstance(w,(int,float)) and w==w] or [20]
        _safe_print(f"[WINDOWS] {top_windows}")

        # 각 윈도우 학습
        for window in top_windows:
            window=min(window, max(6,len(features_only)-1))

            # 샘플 생성 (1차)
            fv=features_only.values.astype(np.float32)
            X_raw, y = _rebuild_samples_with_keepset(fv, labels, window, keep_set, to_local)

            # [보강] 최소 2클래스 확보: 이웃 bin 확장 재생성 → 합성 minority
            repaired_info = {"neighbor_expansion": False, "synthetic_labels": False}
            if X_raw.size == 0 or len(np.unique(y)) < 2:
                # 여기서도 이젠 확장 안 함 (위에서 _expand_to_neighbor_classes 가 no-op이므로)
                # 희소클래스 그대로 유지
                repaired_info["neighbor_expansion"] = False

            if X_raw.size and len(np.unique(y)) < 2:
                # 여기도 synthetic 안 만듦
                X_raw, y, syn = _synthesize_minority_if_needed(X_raw, y, num_classes=len(class_ranges))
                repaired_info["synthetic_labels"] = syn

            usable_samples = int(len(y))
            note_msg = ""
            if usable_samples == 0:
                _log_skip(symbol,strategy,f"유효 라벨 샘플 없음(w={window})")
                continue
            if y.min() < 0:
                _log_skip(symbol,strategy,f"음수 라벨 유입 감지(w={window})")
                continue

            if usable_samples < 20:
                note_msg = f"⚠️ 희소 학습 (샘플 {usable_samples})"
                _safe_print(f"[WARN] {symbol}-{strategy}-w{window}: {note_msg}")

            set_NUM_CLASSES(len(class_ranges))

            strat_ok=False
            try:
                if len(y)>=40 and len(np.unique(y))>=2:
                    splitter=StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=int(os.getenv("GLOBAL_SEED","20240101")))
                    tr_idx, val_idx = next(splitter.split(X_raw, y)); strat_ok=True
            except: strat_ok=False
            if not strat_ok:
                try:
                    train_idx, val_idx = coverage_split_indices(y, val_frac=0.20, min_coverage=0.60, stride=50, num_classes=len(class_ranges))
                except Exception:
                    n=len(y)
                    if n<=1:
                        train_idx=np.array([0],dtype=int); val_idx=np.array([0],dtype=int)
                    else:
                        train_idx=np.arange(0, n-1, dtype=int); val_idx=np.array([n-1], dtype=int)
            else:
                train_idx, val_idx = tr_idx, val_idx

            if len(train_idx)==0 and len(val_idx)==0:
                n=len(y)
                if n<=1:
                    train_idx=np.array([0],dtype=int); val_idx=np.array([0],dtype=int)
                else:
                    train_idx=np.arange(0, n-1, dtype=int); val_idx=np.array([n-1], dtype=int)
            elif len(train_idx)==0:
                val_take = int(val_idx[0])
                train_idx = np.array([val_take], dtype=int)
                val_idx = np.array([val_idx[-1]], dtype=int)
            elif len(val_idx)==0:
                val_idx = np.array([train_idx[-1]], dtype=int)
                train_idx = train_idx[:-1] if len(train_idx)>1 else train_idx

            train_idx, val_idx, _ = _ensure_val_has_two_classes(train_idx, val_idx, y, min_classes=2)

            try:
                cnt_after = np.bincount(y, minlength=len(class_ranges)).astype(int).tolist()
            except Exception:
                cnt_after = []
            batch_stratified_ok = bool(strat_ok)

            try:
                base_note = "prep(stats)"
                if note_msg:
                    base_note += f" | {note_msg}"
                logger.log_training_result(
                    symbol, strategy, model="all",
                    accuracy=None, f1=None, loss=None,
                    val_acc=None, val_f1=None, val_loss=None,
                    engine="manual", window=int(window), recent_cap=int(len(features_only)),
                    rows=int(len(df)), limit=int(STRATEGY_CONFIG.get(strategy,{}).get("limit",300)), min=int(max(60,int(STRATEGY_CONFIG.get(strategy,{}).get("limit",300)*0.90))),
                    augment_needed=bool(augment_needed), enough_for_training=bool(enough_for_training),
                    note=base_note,
                    source_exchange="BYBIT", status="prep",
                    NUM_CLASSES=int(len(class_ranges)),
                    class_counts_label_freeze=cnt_before,
                    usable_samples=usable_samples,
                    class_counts_after_assemble=cnt_after,
                    batch_stratified_ok=batch_stratified_ok
                )
            except Exception:
                pass

            scaler=MinMaxScaler()
            Xtr_flat=X_raw[train_idx].reshape(-1, feat_dim); scaler.fit(Xtr_flat)
            train_X=scaler.transform(Xtr_flat).reshape(len(train_idx),window,feat_dim)
            val_X  =scaler.transform(X_raw[val_idx].reshape(-1,feat_dim)).reshape(len(val_idx),window,feat_dim)
            train_y, val_y = y[train_idx], y[val_idx]

            local_epochs=_epochs_for(strategy)
            if len(train_X)<200: local_epochs=max(6, int(round(local_epochs*0.7)))
            try:
                if len(train_X)<200:
                    train_X,train_y=balance_classes(train_X,train_y,num_classes=len(class_ranges))
            except Exception as e: _safe_print(f"[balance warn] {e}")

            try:
                loss_cfg = get_LOSS()
                cw_cfg = loss_cfg.get("class_weight", {}) if isinstance(loss_cfg, dict) else {}
                mode = str(cw_cfg.get("mode", "inverse_freq_clip")).lower()
                cw_min = float(cw_cfg.get("min", 0.5))
                cw_max = float(cw_cfg.get("max", 2.0))
                eps    = float(cw_cfg.get("eps", 1e-6))
                cc = np.bincount(train_y, minlength=len(class_ranges)).astype(np.float32)
                if mode == "none":
                    w_full = np.ones(len(class_ranges), dtype=np.float32)
                elif mode == "inverse_freq":
                    w_full = (1.0 / np.sqrt(cc + eps)).astype(np.float32)
                else:
                    w = (1.0 / np.sqrt(cc + eps))
                    w = np.clip(w, cw_min, cw_max)
                    w_full = w.astype(np.float32)
                zero = (cc == 0)
                if zero.any():
                    w_full[zero] = max(cw_max, float(np.max(w_full)) if w_full.size else 1.0)
            except Exception:
                w_full=np.ones(len(class_ranges),dtype=np.float32)
            try:
                if np.mean(w_full)>0: w_full=w_full/float(np.mean(w_full))
            except: pass
            w=torch.tensor(w_full,dtype=torch.float32,device=DEVICE)

            priors=np.bincount(train_y, minlength=len(class_ranges)).astype(np.float32)
            priors = priors / max(1.0, float(priors.sum()))
            priors[priors <= 0] = 1e-6
            priors_t=torch.tensor(priors,dtype=torch.float32,device=DEVICE)

            def _make_train_loader():
                base_ds=TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
                if SMART_TRAIN:
                    cc=np.bincount(train_y, minlength=len(class_ranges)).astype(np.float64)
                    inv=1.0/np.clip(cc,1.0,None)
                    sw=inv[train_y].astype(np.float32); sw=np.nan_to_num(sw, nan=1.0, posinf=1.0, neginf=1.0)
                    sampler=torch.utils.data.WeightedRandomSampler(sw.tolist(), num_samples=len(train_y), replacement=True)
                    kw={"batch_size":_BATCH_SIZE,"sampler":sampler,"num_workers":max(0,_NUM_WORKERS),"pin_memory":_PIN_MEMORY}
                else:
                    kw={"batch_size":_BATCH_SIZE,"shuffle":True,"num_workers":max(0,_NUM_WORKERS),"pin_memory":_PIN_MEMORY}
                if _NUM_WORKERS>0 and _PERSISTENT: kw["persistent_workers"]=True
                return DataLoader(base_ds, **kw)
            train_loader=_make_train_loader()
            vds=TensorDataset(torch.tensor(val_X,dtype=torch.float32), torch.tensor(val_y,dtype=torch.long))
            vkw={"batch_size":_BATCH_SIZE,"shuffle":False,"num_workers":max(0,_NUM_WORKERS),"pin_memory":_PIN_MEMORY}
            if _NUM_WORKERS>0 and _PERSISTENT: vkw["persistent_workers"]=True
            val_loader=DataLoader(vds, **vkw)

            scaler_amp = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type=="cuda"))

            for model_type in ["lstm","cnn_lstm","transformer"]:
                base=get_model(model_type,input_size=feat_dim,output_size=len(class_ranges)).to(DEVICE)
                model=base

                if strategy != "단기":
                    prev_strat = "단기" if strategy=="중기" else "중기"
                    prev_path = _find_prev_model_for(symbol, prev_strat)
                    if prev_path and _load_for_finetune is not None:
                        try:
                            _load_for_finetune(model, prev_path, strict=False)
                            freeze_backbone(model)
                            unfreeze_last_k_layers(model, k=1)
                            _safe_print(f"[FT] loaded {prev_path} → freeze backbone, tune last layers")
                        except Exception as e:
                            _safe_print(f"[FT warn] {e}")

                opt=torch.optim.Adam(model.parameters(), lr=1e-3)
                crit=FocalLoss(gamma=FOCAL_GAMMA, weight=w)
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=1e-5) if SMART_TRAIN else None
                patience=EARLY_STOP_PATIENCE; min_delta=EARLY_STOP_MIN_DELTA
                best_f1=-1.0; best_state=None; bad=0; loss_sum=0.0

                _safe_print(f"🟦 TRAIN {symbol}-{strategy}-g{group_id} w={window} model={model_type} epochs={local_epochs}")
                for ep in range(local_epochs):
                    if stop_event is not None and stop_event.is_set(): break
                    model.train()
                    for xb,yb in train_loader:
                        xb=xb.to(DEVICE,dtype=torch.float32); yb=yb.to(DEVICE,dtype=torch.long)
                        with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type=="cuda")):
                            logits=model(xb); loss=crit(logits,yb)
                        if not np.isfinite(float(loss.item())): continue
                        opt.zero_grad(set_to_none=True)
                        if scaler_amp.is_enabled():
                            scaler_amp.scale(loss).backward()
                            if SMART_TRAIN and GRAD_CLIP>0: scaler_amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                            scaler_amp.step(opt); scaler_amp.update()
                        else:
                            loss.backward()
                            if SMART_TRAIN and GRAD_CLIP>0: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                            opt.step()
                        loss_sum += float(loss.item())
                        _release_memory(xb,yb,loss,logits)

                    model.eval(); preds=[]; lbls=[]
                    with torch.no_grad():
                        for xb,yb in val_loader:
                            xb=xb.to(DEVICE,dtype=torch.float32)
                            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type=="cuda")):
                                logits=model(xb)
                            if COST_SENSITIVE_ARGMAX:
                                adj=logits-(CS_ARG_BETA*torch.log(priors_t.unsqueeze(0)))
                                p=torch.argmax(adj,dim=1).cpu().numpy()
                            else:
                                p=torch.argmax(logits,dim=1).cpu().numpy()
                            preds.extend(p); lbls.extend(yb.numpy())
                            _release_memory(xb,logits)
                    try:
                        cur_f1=float(f1_score(lbls,preds,average="macro",
                                              labels=list(range(len(class_ranges))),
                                              zero_division=0)) if len(lbls) else 0.0
                    except Exception:
                        cur_f1=0.0
                    if scheduler is not None:
                        try: scheduler.step(cur_f1)
                        except: pass
                    improved=(cur_f1-best_f1)>min_delta if best_f1>=0 else True
                    if improved:
                        best_f1=cur_f1
                        try: best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        except: best_state=None
                        bad=0
                    else:
                        bad+=1
                    if TRAIN_CUDA_EMPTY_EVERY_EP: _safe_empty_cache()
                    if bad>=patience:
                        _safe_print(f"🛑 early stop @ ep{ep+1} best_f1={best_f1:.4f}")
                        break

                if best_state is not None:
                    try: model.load_state_dict(best_state)
                    except: pass
                _release_memory(best_state)

                model.eval(); preds=[]; lbls=[]; val_loss_sum=0.0; n_val=0
                crit_eval=nn.CrossEntropyLoss(weight=w)
                with torch.no_grad():
                    for xb,yb in val_loader:
                        xb=xb.to(DEVICE,dtype=torch.float32)
                        with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type=="cuda")):
                            logits=model(xb)
                            loss_eval = crit_eval(logits, yb.to(DEVICE,dtype=torch.long))
                        try:
                            val_loss_sum += float(loss_eval.item()) * xb.size(0)
                        except Exception:
                            pass
                        n_val += xb.size(0)
                        if COST_SENSITIVE_ARGMAX:
                            adj=logits-(CS_ARG_BETA*torch.log(priors_t.unsqueeze(0)))
                            p=torch.argmax(adj,dim=1).cpu().numpy()
                        else:
                            p=torch.argmax(logits,dim=1).cpu().numpy()
                        preds.extend(p); lbls.extend(yb.numpy())
                        _release_memory(xb,logits,loss_eval)
                try:
                    acc=float(accuracy_score(lbls,preds)) if len(lbls) else 0.0
                except Exception:
                    acc=0.0
                try:
                    f1_val=float(f1_score(lbls,preds,average="macro",
                                           labels=list(range(len(class_ranges))),
                                           zero_division=0)) if len(lbls) else 0.0
                except Exception:
                    f1_val=0.0
                val_loss=float(val_loss_sum/max(1,n_val))

                try:
                    if isinstance(bin_info, dict) and "bin_edges" in bin_info:
                        bin_edges = [float(x) for x in bin_info.get("bin_edges", [])]
                        bin_spans = bin_info.get("bin_spans", [])
                        if bin_edges and (not bin_spans or len(bin_spans) != len(bin_edges)-1):
                            bin_spans = [float(bin_edges[i+1]-bin_edges[i]) for i in range(len(bin_edges)-1)]
                        full_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=None)
                        cnt_local = np.bincount(val_y, minlength=len(class_ranges))
                        counts_map = np.zeros(len(full_ranges), dtype=int)
                        for g, l in to_local.items():
                            if g < len(counts_map) and l < len(cnt_local):
                                counts_map[g] = int(cnt_local[l])
                        bin_counts = counts_map.tolist()
                    else:
                        full_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=None)
                        bin_edges = [float(lo) for (lo, _) in full_ranges] + [float(full_ranges[-1][1])]
                        bin_spans = [float(hi-lo) for (lo, hi) in full_ranges]
                        cnt_local = np.bincount(val_y, minlength=len(full_ranges))
                        counts_map = np.zeros(len(full_ranges), dtype=int)
                        for g, l in to_local.items():
                            if g < len(counts_map) and l < len(cnt_local):
                                counts_map[g] = int(cnt_local[l])
                        bin_counts = counts_map.tolist()
                except Exception:
                    bin_edges, bin_spans, bin_counts = [], [], []

                bin_cfg = {
                    "TARGET_BINS": int(os.getenv("TARGET_BINS", "8")),
                    "OUTLIER_Q_LOW": float(os.getenv("OUTLIER_Q_LOW", "0.01")),
                    "OUTLIER_Q_HIGH": float(os.getenv("OUTLIER_Q_HIGH", "0.99")),
                    "MAX_BIN_SPAN_PCT": float(os.getenv("MAX_BIN_SPAN_PCT", "8.0")),
                    "MIN_BIN_COUNT_FRAC": float(os.getenv("MIN_BIN_COUNT_FRAC", "0.05")),
                }

                stem=os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_w{int(window)}_group{int(group_id) if group_id is not None else 0}_cls{int(len(class_ranges))}")
                meta={"symbol":symbol,"strategy":strategy,"model":model_type,"group_id":int(group_id or 0),
                      "num_classes":int(len(class_ranges)),
                      "class_ranges":[[float(lo),float(hi)] for (lo,hi) in class_ranges],
                      "input_size":int(feat_dim),
                      "metrics":{"val_acc":acc,"val_f1":f1_val,"val_loss":val_loss},
                      "timestamp":now_kst().isoformat(),"model_name":os.path.basename(stem)+".ptz",
                      "window":int(window),"recent_cap":int(len(features_only)),"engine":"manual",
                      "data_flags":{"rows":int(len(df)),"limit":int(_limit),"min":int(_min_required),"augment_needed":bool(augment_needed),"enough_for_training":bool(enough_for_training)},
                      "train_loss_sum":float(loss_sum),
                      "boundary_band": float(BOUNDARY_BAND),
                      "cs_argmax":{"enabled":bool(COST_SENSITIVE_ARGMAX),"beta":float(CS_ARG_BETA)},
                      "eval_gate":"none",
                      "label_repair": repaired_info,
                      "bin_edges": bin_edges,
                      "bin_counts": bin_counts,
                      "bin_spans": bin_spans,
                      "bin_cfg": bin_cfg}
                wpath,mpath=_save_model_and_meta(model, stem+".pt", meta)

                try:
                    DataCacheManager.delete(f"{symbol}-{strategy}")
                    DataCacheManager.delete(f"{symbol}-{strategy}-features")
                except: pass

                try:
                    final_note = f"train_one_model(window={window}, cap={len(features_only)}, engine=manual)"
                    if note_msg:
                        final_note += f" | {note_msg}"
                    logger.log_training_result(
                        symbol, strategy,
                        model=os.path.basename(wpath),
                        accuracy=acc, f1=f1_val, loss=val_loss,
                        val_acc=acc, val_f1=f1_val, val_loss=val_loss,
                        engine="manual", window=int(window), recent_cap=int(len(features_only)),
                        rows=int(len(df)), limit=int(_limit), min=int(_min_required),
                        augment_needed=bool(augment_needed), enough_for_training=bool(enough_for_training),
                        note=final_note,
                        source_exchange="BYBIT", status="success",
                        y_true=lbls, y_pred=preds, num_classes=len(class_ranges),
                        NUM_CLASSES=int(len(class_ranges)),
                        class_counts_label_freeze=cnt_before,
                        usable_samples=usable_samples,
                        class_counts_after_assemble=cnt_after,
                        batch_stratified_ok=batch_stratified_ok
                    )
                except Exception:
                    pass

                res["models"].append({"window":int(window),"type":model_type,"acc":acc,"f1":f1_val, "val_loss":val_loss,
                                      "loss_sum":float(loss_sum),"pt":wpath,"meta":mpath,"passed":True})
                _safe_print(f"🟩 DONE w={window} {model_type} acc={acc:.4f} f1={f1_val:.4f} val_loss={val_loss:.5f} (no gate)")

                try:
                    if IMPORTANCE_ENABLE:
                        X_val_t = torch.tensor(val_X, dtype=torch.float32, device="cpu")
                        y_val_t = torch.tensor(val_y, dtype=torch.long, device="cpu")
                        fi = compute_feature_importance(
                            model.to("cpu"),
                            X_val=X_val_t, y_val=y_val_t,
                            feature_names=list(features_only.columns),
                            max_seconds=30
                        )
                        try:
                            save_feature_importance(fi, symbol, strategy, model_type, method="permutation")
                        except OSError as e:
                            import errno
                            if getattr(e, "errno", None) == errno.Enospc:
                                _safe_print("[경고] 디스크 부족으로 feature importance 저장 스킵")
                            else:
                                _safe_print(f"[경고] feature importance 저장 실패: {e}")
                        finally:
                            try: model.to(DEVICE)
                            except Exception: pass
                except Exception as e:
                    _safe_print(f"[경고] feature importance 비활성화/실패: {e}")

                _release_memory(model, base, opt, crit, scheduler)
                _release_memory(preds, lbls)
                if DEVICE.type=="cuda": _safe_empty_cache()

            _release_memory(train_loader, val_loader, vds)
            _release_memory(train_X, val_X, train_y, val_y, X_raw, y, train_idx, val_idx, scaler)
            if DEVICE.type=="cuda": _safe_empty_cache()

            res["windows"].append({"window":int(window), "results":[m for m in res["models"] if m["window"]==window]})

        res["ok"]=bool(res.get("models"))
        _safe_print(f"[RESULT] {symbol}-{strategy}-g{group_id} ok={res['ok']}")

        if _is_group_active_file() or _is_group_lock_file():
            _safe_print(f"[AUTO-PREDICT SKIP] group-active/lock → skip {symbol}-{strategy}")
        else:
            try:
                pl_clear_stale(lock_key=(symbol, strategy))
                pl_wait_free(max_wait_sec=10, lock_key=(symbol, strategy))
                run_after_training(symbol, strategy)
                _safe_print(f"[AUTO-PREDICT] triggered after training {symbol}-{strategy}")
            except Exception as e:
                _safe_print(f"[AUTO-PREDICT FAIL] {symbol}-{strategy} → {e}")

        try:
            from logger import flush_gwanwoo_summary
            flush_gwanwoo_summary()
        except Exception:
            pass

        _release_memory(feat, features_only, df)
        return res

    except Exception as e:
        _safe_print(f"[EXC] train_one_model {symbol}-{strategy}-g{group_id} → {e}\n{traceback.format_exc()}")
        _log_fail(symbol,strategy,str(e)); return res


# ---------- 심볼 전체/그룹 순서 ----------
_ENFORCE_FULL_STRATEGY = False
_STRICT_HALT_ON_INCOMPLETE = False
_REQUIRE_AT_LEAST_ONE_MODEL_PER_GROUP = False
_SYMBOL_RETRY_LIMIT = int(os.getenv("SYMBOL_RETRY_LIMIT","1"))

def _train_full_symbol(symbol:str, stop_event: Optional[threading.Event] = None) -> Tuple[bool, Dict[str, Any]]:
    strategies=["단기","중기","장기"]; detail={}; any_saved=False
    base_df, pre_feats, pre_lbls = _build_precomputed(symbol)

    for strategy in strategies:
        if stop_event is not None and stop_event.is_set(): return any_saved, detail
        try:
            cr=get_class_ranges(symbol=symbol,strategy=strategy)
            num_classes=len(cr) if cr else 0
            groups=get_class_groups(num_classes=max(2,num_classes))
            max_gid=len(groups)-1
            detail[strategy]={}
            for gid in range(max_gid+1):
                if stop_event is not None and stop_event.is_set(): return any_saved, detail
                gr=get_class_ranges(symbol=symbol,strategy=strategy,group_id=gid)
                if not gr or len(gr)<2:
                    _safe_print(f"[FORCE-TRAIN] {symbol}-{strategy}-g{gid}: cls<2 → 학습 강행")

                attempts=(_SHORT_RETRY if strategy=="단기" else 1); ok_once=False
                for _ in range(attempts):
                    pf = pre_feats.get(strategy) if isinstance(pre_feats, dict) else None
                    pl = pre_lbls.get(strategy)  if isinstance(pre_lbls, dict) else None
                    res=train_one_model(symbol,strategy,group_id=gid, max_epochs=_epochs_for(strategy),
                                        stop_event=stop_event, pre_feat=pf, pre_lbl=pl)
                    if bool(res and isinstance(res,dict) and res.get("models")):
                        ok_once=True; any_saved=True; break
                detail[strategy][gid]=ok_once
                time.sleep(0.01)
        except Exception as e:
            logger.log_training_result(symbol,strategy,model="all",
                accuracy=0.0,f1=0.0,loss=0.0,val_acc=0.0,val_f1=0.0,val_loss=0.0,
                engine="manual",window=None,recent_cap=None,rows=None,limit=None,min=None,
                augment_needed=None,enough_for_training=None,
                note=f"전략 실패: {e}",status="failed",source_exchange="BYBIT")
            detail[strategy]={-1:False}
    return any_saved, detail

def train_models(symbol_list, stop_event: Optional[threading.Event] = None, ignore_should: bool = False):
    completed_symbols=[]; partial_symbols=[]
    env_force = (os.getenv("TRAIN_FORCE_IGNORE_SHOULD","0") == "1")
    for symbol in symbol_list:
        if stop_event is not None and stop_event.is_set(): break
        symbol_has_model=_has_any_model_for_symbol(symbol)
        local_ignore = ignore_should or env_force or (not symbol_has_model)
        if not local_ignore:
            if not should_train_symbol(symbol):
                _safe_print(f"[ORDER] skip {symbol} (should_train_symbol=False, models_exist={symbol_has_model})")
                continue
        else:
            _safe_print(f"[order-override] {symbol}: force train")

        trained_complete=False
        for _ in range(max(1,_SYMBOL_RETRY_LIMIT)):
            if stop_event is not None and stop_event.is_set(): break
            complete, detail = _train_full_symbol(symbol, stop_event=stop_event)
            _safe_print(f"[ORDER] {symbol} → complete={complete} detail={detail}")
            if complete: trained_complete=True; break

        if trained_complete:
            completed_symbols.append(symbol)
            try: mark_symbol_trained(symbol)
            except Exception as e: _safe_print(f"[mark_symbol_trained err] {e}")
        else:
            partial_symbols.append(symbol)
    return completed_symbols, partial_symbols

def _scan_symbols_from_model_dir() -> List[str]:
    syms=set()
    try:
        for p in glob.glob(os.path.join(MODEL_DIR, f"*_*_*.*")):
            b=os.path.basename(p); import re
            m=re.match(r"^([A-Z0-9]+)_[^_]+_", b)
            if m: syms.add(m.group(1))
        for d in glob.glob(os.path.join(MODEL_DIR, "*")):
            if os.path.isdir(d): syms.add(os.path.basename(d))
    except: pass
    return sorted(syms)

def _pick_smoke_symbol(candidates: List[str]) -> Optional[str]:
    cand=[s for s in candidates if _has_any_model_for_symbol(s)]
    if cand: return sorted(cand)[0]
    pool=_scan_symbols_from_model_dir(); pool=[s for s in pool if _has_any_model_for_symbol(s)]
    return pool[0] if pool else None

def _run_smoke_predict(predict_fn, symbol: str):
    ok_any=False
    for strat in ["단기","중기","장기"]:
        if _has_model_for(symbol, strat):
            try:
                predict_fn(symbol, strat, source="그룹직후(스모크)", model_type=None); ok_any=True
            except Exception as e:
                _safe_print(f"[SMOKE fail] {symbol}-{strat}: {e}")
    return ok_any

def _safe_predict_with_timeout(predict_fn, symbol: str, strategy: str, source: str = "group_end",
                               model_type: str | None = None, timeout: float = PREDICT_TIMEOUT_SEC) -> bool:
    try: pl_clear_stale(lock_key=(symbol, strategy))
    except Exception: pass
    try: pl_wait_free(max_wait_sec=int(max(1, timeout/2)), lock_key=(symbol, strategy))
    except Exception: pass

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

def train_symbol_group_loop(sleep_sec:int=0, stop_event: Optional[threading.Event] = None):
    env_force_ignore = (os.getenv("TRAIN_FORCE_IGNORE_SHOULD","0") == "1")
    env_reset = (os.getenv("RESET_GROUP_ORDER_ON_START","0") == "1")
    force_full_pass = env_force_ignore
    if force_full_pass or env_reset:
        try: reset_group_order(0)
        except Exception as e: _safe_print(f"[group reset skip] {e}")

    while True:
        if stop_event is not None and stop_event.is_set(): break
        try:
            from predict import predict
            if hasattr(logger,"ensure_train_log_exists"): logger.ensure_train_log_exists()
            if hasattr(logger,"ensure_prediction_log_exists"): logger.ensure_prediction_log_exists()

            groups=[list(g) for g in SYMBOL_GROUPS]
            for idx, group in enumerate(groups):
                if stop_event is not None and stop_event.is_set(): break
                _safe_print(f"🚀 [group] {idx+1}/{len(groups)} → {group}")

                try:
                    _set_group_active(True, group_idx=idx, symbols=group)
                    _set_group_train_lock(True, group_idx=idx, symbols=group)
                except Exception as e:
                    _safe_print(f"[GROUP mark warn] {e}")

                completed_syms, partial_syms = train_models(group, stop_event=stop_event, ignore_should=force_full_pass)
                if stop_event is not None and stop_event.is_set(): break

                gate_ok = True
                try:
                    gate_ok = ready_for_group_predict()
                except Exception as e:
                    _safe_print(f"[PREDICT-GATE warn] {e} -> 게이트 무시하고 진행")

                if (not gate_ok) and (not PREDICT_FORCE_AFTER_GROUP):
                    _safe_print(f"[PREDICT-BLOCK] group{idx+1} ready_for_group_predict()==False (강제 실행 비활성)")
                else:
                    if (not gate_ok) and PREDICT_FORCE_AFTER_GROUP:
                        _safe_print(f"[PREDICT-OVERRIDE] group{idx+1} 게이트 False지만 강제 실행")

                    ran_any=False
                    for symbol in group:
                        for strategy in ["단기","중기","장기"]:
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
                                    ran_any=True
                                else:
                                    _safe_print(f"[PREDICT TIMEOUT] {symbol}-{strategy} (> {PREDICT_TIMEOUT_SEC}s)")
                            except Exception as e:
                                _safe_print(f"[PREDICT FAIL] {symbol}-{strategy}: {e}")

                    if not ran_any:
                        cand_symbol=_pick_smoke_symbol(group)
                        if cand_symbol:
                            _safe_print(f"[SMOKE] fallback predict for {cand_symbol}")
                            try: _run_smoke_predict(predict, cand_symbol)
                            except Exception as e: _safe_print(f"[SMOKE fail] {e}")

                    if ran_any:
                        try: mark_group_predicted()
                        except Exception as e: _safe_print(f"[mark_group_predicted err] {e}")

                try:
                    from logger import flush_gwanwoo_summary
                    flush_gwanwoo_summary()
                except Exception:
                    pass

                try: close_predict_gate(note=f"train:group{idx+1}_end")
                except Exception as e: _safe_print(f"[gate close warn] {e}")

                try:
                    _set_group_active(False)
                    _set_group_train_lock(False)
                except Exception as e:
                    _safe_print(f"[GROUP clear warn] {e}")

                if sleep_sec>0:
                    for _ in range(sleep_sec):
                        if stop_event is not None and stop_event.is_set(): break
                        time.sleep(1)
                    if stop_event is not None and stop_event.is_set(): break

            _safe_print("✅ group pass done")
            try:
                from logger import flush_gwanwoo_summary
                flush_gwanwoo_summary()
            except Exception:
                pass
            try: close_predict_gate(note="train:group_pass_done")
            except Exception as e: _safe_print(f"[gate close warn] {e}")

            if force_full_pass and not env_force_ignore:
                force_full_pass=False
        except Exception as e:
            _safe_print(f"[group loop err] {e}\n{traceback.format_exc()}")

        _safe_print("💓 heartbeat"); time.sleep(max(1,int(os.getenv("TRAIN_LOOP_IDLE_SEC","3"))))

# ---------- 루프 제어 ----------
_TRAIN_LOOP_THREAD: Optional[threading.Thread] = None
_TRAIN_LOOP_STOP: Optional[threading.Event] = None
_TRAIN_LOOP_LOCK=threading.Lock()

def start_train_loop(force_restart:bool=False, sleep_sec:int=0):
    global _TRAIN_LOOP_THREAD,_TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive():
            if not force_restart: _safe_print("ℹ️ already running"); return False
            stop_train_loop(timeout=30)
        _TRAIN_LOOP_STOP=threading.Event()
        def _runner():
            try: train_symbol_group_loop(sleep_sec=sleep_sec, stop_event=_TRAIN_LOOP_STOP)
            finally: _safe_print("ℹ️ train loop exit")
        _TRAIN_LOOP_THREAD=threading.Thread(target=_runner,daemon=True); _TRAIN_LOOP_THREAD.start()
        _safe_print("✅ train loop started"); return True

def stop_train_loop(timeout:int|float|None=30):
    global _TRAIN_LOOP_THREAD,_TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is None or not _TRAIN_LOOP_THREAD.is_alive():
            _safe_print("ℹ️ no loop"); return True
        if _TRAIN_LOOP_STOP is None: _safe_print("⚠️ no stop event"); return False
        _TRAIN_LOOP_STOP.set(); _TRAIN_LOOP_THREAD.join(timeout=timeout)
        if _TRAIN_LOOP_THREAD.is_alive(): _safe_print("⚠️ stop timeout"); return False
        _TRAIN_LOOP_THREAD=None; _TRAIN_LOOP_STOP=None; _safe_print("✅ loop stopped"); return True

def request_stop()->bool:
    global _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_STOP is None: return True
        _TRAIN_LOOP_STOP.set(); return True

def is_loop_running()->bool:
    with _TRAIN_LOOP_LOCK:
        return bool(_TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive())

# ===== 공개 API =====
def train_symbol(symbol: str, strategy: str, group_id: int | None = None) -> dict:
    res = train_one_model(symbol=symbol, strategy=strategy, group_id=group_id)
    try:
        if res.get("models"):
            mark_symbol_trained(symbol)
            if not (_is_group_active_file() or _is_group_lock_file()):
                try:
                    from predict import predict
                    _safe_predict_with_timeout(predict_fn=predict, symbol=symbol, strategy=strategy,
                                               source="train_symbol", model_type=None, timeout=PREDICT_TIMEOUT_SEC)
                except Exception:
                    pass
    except Exception:
        pass
    return res

def train_group(group_id: int | None = None) -> dict:
    idx = get_current_group_index() if group_id is None else int(group_id)
    symbols = get_current_group_symbols() if group_id is None else (SYMBOL_GROUPS[idx] if 0 <= idx < len(SYMBOL_GROUPS) else [])
    out = {"group_index": idx, "symbols": symbols, "results": {}}

    try:
        _set_group_active(True, group_idx=idx, symbols=symbols)
        _set_group_train_lock(True, group_idx=idx, symbols=symbols)
    except Exception as e:
        _safe_print(f"[GROUP mark warn] {e}")

    completed, partial = train_models(symbols, stop_event=None, ignore_should=False)
    out["completed"] = completed; out["partial"] = partial

    try:
        gate_ok = ready_for_group_predict()
    except Exception:
        gate_ok = True
    if gate_ok:
        try:
            from predict import predict
            ran_any = False
            for s in symbols:
                for strat in ["단기", "중기", "장기"]:
                    if _has_model_for(s, strat):
                        try:
                            ok = _safe_predict_with_timeout(predict_fn=predict, symbol=s, strategy=strat,
                                                            source="train_group", model_type=None,
                                                            timeout=PREDICT_TIMEOUT_SEC)
                            ran_any = ran_any or ok
                        except Exception:
                            pass
            if ran_any:
                try: mark_group_predicted()
                except Exception: pass
        finally:
            try:
                from logger import flush_gwanwoo_summary
                flush_gwanwoo_summary()
            except Exception:
                pass
            try: close_predict_gate(note=f"train_group:idx{idx}_end")
            except Exception: pass

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

if __name__=="__main__":
    try: start_train_loop(force_restart=True, sleep_sec=0)
    except Exception as e: _safe_print(f"[MAIN] err: {e}")
