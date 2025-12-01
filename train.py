# train.py — SPEED v2.6 FINAL (희소클래스 제거 없이 그룹 클래스 전부 학습 버전)
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
        compute_features_multi,
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
# YOPO 철학: 라벨 "병합/삭제" 없음. 희소 클래스도 그대로 유지.
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
    # === 수익률/클래스 구간 요약 추가 ===
    "near_zero_band",
    "near_zero_count",
    "masked_count",
    "class_ranges",
    "bin_edges",
    "bin_counts",
    "bin_spans",
    # 🔥 프론트에서 찾는 이름들(호환용)
    "class_edges",    # = bin_edges
    "class_counts",   # = bin_counts
    "bins",           # = len(bin_edges) - 1
]

try:
    from logger import TRAIN_HEADERS

    # logger 쪽 기본 헤더와 우리가 추가한 헤더를 합침(중복 제거)
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
    """
    - 파일이 없으면: 최신 TRAIN_HEADERS 로 새로 생성
    - 파일이 있되, 예전 헤더(분포 컬럼 없음)이면:
      → 백업(.bak) 만들고, 헤더를 TRAIN_HEADERS 기준으로 업그레이드
    """
    try:
        # 1) 파일이 아예 없으면 최신 헤더로 생성
        if not os.path.exists(TRAIN_LOG):
            with open(TRAIN_LOG, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(TRAIN_HEADERS)
            return

        # 2) 파일이 있는데, 헤더가 비었으면 다시 작성
        with open(TRAIN_LOG, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()

        if not first_line:
            with open(TRAIN_LOG, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(TRAIN_HEADERS)
            return

        # 3) 기존 헤더 파싱
        try:
            reader = csv.reader([first_line])
            old_headers = next(reader)
        except Exception:
            old_headers = []

        # 이미 충분히 최신이면 아무 것도 안 함
        important_new_cols = [
            "class_edges",
            "class_counts",
            "bins",
            "bin_edges",
            "bin_counts",
            "bin_spans",
            "near_zero_band",
            "near_zero_count",
            "masked_count",
            "NUM_CLASSES",
            "class_counts_label_freeze",
            "usable_samples",
            "class_counts_after_assemble",
            "batch_stratified_ok",
        ]
        need_upgrade = any(col not in old_headers for col in important_new_cols)

        if not need_upgrade:
            return

        # 4) 업그레이드: pandas 로 전체 읽어서, 누락 컬럼 추가 후 저장
        try:
            df = pd.read_csv(TRAIN_LOG, encoding="utf-8-sig")
        except Exception:
            # 읽기 실패하면, 기존 파일은 백업해 두고 새로 생성
            backup_path = TRAIN_LOG + ".bak"
            try:
                shutil.move(TRAIN_LOG, backup_path)
            except Exception:
                pass
            with open(TRAIN_LOG, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(TRAIN_HEADERS)
            print("[train_log] 기존 파일 손상 → 새 train_log.csv 생성", flush=True)
            return

        # df에 TRAIN_HEADERS 에 있는 모든 컬럼을 보장
        for col in TRAIN_HEADERS:
            if col not in df.columns:
                df[col] = None

        # 컬럼 순서는 TRAIN_HEADERS 순서로 맞춤
        df = df[TRAIN_HEADERS]

        backup_path = TRAIN_LOG + ".bak"
        try:
            shutil.move(TRAIN_LOG, backup_path)
        except Exception:
            pass

        df.to_csv(TRAIN_LOG, index=False, encoding="utf-8-sig")
        print(f"[train_log] 헤더 업그레이드 완료 → backup={backup_path}", flush=True)

    except Exception as e:
        print(f"[경고] train_log 초기화/업그레이드 실패: {e}", flush=True)


def _normalize_train_row(row: dict) -> dict:
    # 모든 헤더에 대해 기본값 채우기
    r = {k: row.get(k, None) for k in TRAIN_HEADERS}

    # 옛날 키 이름과 호환
    if r.get("val_acc") is None and row.get("accuracy") is not None:
        r["val_acc"] = row.get("accuracy")
    if r.get("val_f1") is None and row.get("f1") is not None:
        r["val_f1"] = row.get("f1")
    if r.get("val_loss") is None and row.get("loss") is not None:
        r["val_loss"] = row.get("loss")

    r.setdefault("engine", row.get("engine", "manual"))
    r.setdefault("source_exchange", row.get("source_exchange", "BYBIT"))

    # 수익률 구간 호환 필드 채우기
    # - bin_edges/bin_counts 가 있으면 class_edges/class_counts/bins 도 같이 채워줌
    be = row.get("bin_edges") or r.get("bin_edges")
    bc = row.get("bin_counts") or r.get("bin_counts")

    if be is not None and r.get("class_edges") is None:
        r["class_edges"] = be
    if bc is not None and r.get("class_counts") is None:
        r["class_counts"] = bc
    if r.get("bins") is None:
        try:
            if isinstance(be, (list, tuple)) and len(be) >= 2:
                r["bins"] = len(be) - 1
        except Exception:
            pass

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

# logger.log_training_result 를 패치해서
# → 원래 로깅 + train_log.csv 에 한 줄 더 쓰도록
if not getattr(logger, "_patched_train_log", False):
    _orig_ltr = getattr(logger, "log_training_result", None)

    def _log_training_result_patched(*args, **kw):
        # 1) 원래 logger 로깅 먼저 시도
        if callable(_orig_ltr):
            try:
                _orig_ltr(*args, **kw)
            except Exception as e:
                print(f"[경고] logger.log_training_result 실패: {e}")

        # 2) train_log.csv 에도 기록
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
BALANCE_CLASSES_FLAG = os.getenv("BALANCE_CLASSES", "1") == "1"      # 기본 ON (증강/리샘플용)
WEIGHTED_SAMPLER_FLAG = os.getenv("WEIGHTED_SAMPLER", "0") == "1"    # 기본 OFF

# ★ 희소 클래스 보강용 ENV (삭제 없음, 단순 복사)
MIN_CLASS_SAMPLES = int(os.getenv("MIN_CLASS_SAMPLES", "4"))   # 각 클래스 최소 유지 샘플 수
MAX_CLASS_UPSAMPLE = int(os.getenv("MAX_CLASS_UPSAMPLE", "64"))  # 전체 데이터가 너무 비대해지는 것 방지용 상한
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
    keep_set: set[int],
    to_local: Dict[int, int],
    min_samples: int = 1,  # ← 형식만 남기고, "클래스 컷"에는 쓰지 않는다
) -> Tuple[np.ndarray, np.ndarray]:
    """
    그룹에 속한 클래스는 '전부' 그대로 쓰는 시계열 샘플 재구성 함수.

    - labels: 전역 클래스 인덱스(0~전체-1)
    - keep_set: 이번 그룹에 해당하는 전역 클래스 집합
    - to_local: (필요시) 전역→로컬 매핑(지금은 local_map 새로 구성하므로 필수는 아님)

    ❗ 중요한 점:
    - '샘플이 적다'는 이유로 특정 클래스를 잘라내지 않는다.
    - 단지 이번 그룹에 속하지 않는 클래스만 제외.
    - 나중에 전체 샘플이 0개면 그때만 학습 스킵.
    """
    # 방어 코드: 입력이 비었으면 바로 종료
    if fv is None or labels is None or len(fv) == 0 or len(labels) == 0:
        feat_dim = fv.shape[1] if fv is not None and fv.ndim == 2 else 0
        return (
            np.empty((0, window, feat_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    n = len(fv)

    # 이번 그룹에 사용할 전역 클래스 집합
    base_keep: set[int] = set(int(c) for c in (keep_set or set()))
    if not base_keep:
        # 혹시라도 비어 있으면, 관측된 모든 클래스를 후보로 사용
        base_keep = set(int(l) for l in labels.tolist() if int(l) >= 0)

    if not base_keep:
        feat_dim = fv.shape[1] if fv is not None and fv.ndim == 2 else 0
        return (
            np.empty((0, window, feat_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # 전역 → 로컬 인덱스 매핑 (정렬된 순서로 0~k-1)
    sorted_keep = sorted(base_keep)
    local_map: Dict[int, int] = {g: i for i, g in enumerate(sorted_keep)}

    X_raw: list[np.ndarray] = []
    y_out: list[int] = []

    # 윈도우 단위로 샘플 생성
    for i in range(n - window):
        yi = i + window - 1
        if yi < 0 or yi >= len(labels):
            continue

        lab_g = int(labels[yi])
        if lab_g < 0:
            continue
        if lab_g not in base_keep:
            # 이번 그룹에 속하지 않는 클래스는 제외
            continue

        lab_local = local_map.get(lab_g, None)
        if lab_local is None:
            continue

        X_raw.append(fv[i : i + window])
        y_out.append(lab_local)

    if not X_raw:
        feat_dim = fv.shape[1] if fv is not None and fv.ndim == 2 else 0
        return (
            np.empty((0, window, feat_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X_arr = np.asarray(X_raw, dtype=np.float32)
    y_arr = np.asarray(y_out, dtype=np.int64)

    return X_arr, y_arr


def _synthesize_minority_if_needed(
    X_raw: np.ndarray,
    y: np.ndarray,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    희소 클래스를 '삭제'하지 않고, 단순 복사(오버샘플)로만 최소 개수까지 보강하는 함수.

    - 어떤 클래스가 1~몇 개밖에 없어도 절대 버리지 않는다.
    - 각 클래스 샘플 수가 MIN_CLASS_SAMPLES 보다 작으면, 그 클래스 샘플을 복사해서 늘린다.
    - 전체 샘플 수가 MAX_CLASS_UPSAMPLE 를 크게 넘지 않도록 안전장치도 둔다.
    """
    if X_raw is None or y is None or len(y) == 0 or num_classes <= 0:
        return X_raw, y, False

    y = np.asarray(y, dtype=np.int64)
    X_raw = np.asarray(X_raw, dtype=np.float32)

    # 잘못된 라벨은 건드리지 않는다
    if y.min() < 0:
        return X_raw, y, False

    # 클래스별 개수
    counts = np.bincount(y, minlength=num_classes).astype(int)
    if counts.sum() <= 0:
        return X_raw, y, False

    # 이미 충분히 많은 데이터면 굳이 안 건드림
    if counts.min() >= MIN_CLASS_SAMPLES:
        return X_raw, y, False

    rng = np.random.default_rng(int(os.getenv("GLOBAL_SEED", "20240101")))
    X_list = [X_raw]
    y_list = [y]
    changed = False

    for cls_id, cnt in enumerate(counts):
        if cnt <= 0:
            # 실제로 한 번도 등장하지 않은 클래스는 어쩔 수 없음 (데이터가 0이니까 보강 불가)
            continue
        if cnt >= MIN_CLASS_SAMPLES:
            continue

        # 이 클래스에 해당하는 인덱스
        idx = np.where(y == cls_id)[0]
        if idx.size == 0:
            continue

        # 얼마나 더 채울지
        need = MIN_CLASS_SAMPLES - cnt
        # 전체 길이 상한 보호
        max_extra = max(0, MAX_CLASS_UPSAMPLE - len(y))
        if max_extra <= 0:
            break
        need = min(need, max_extra)
        if need <= 0:
            continue

        # 동일 샘플을 랜덤 복사
        dup_idx = rng.choice(idx, size=need, replace=True)
        X_extra = X_raw[dup_idx]
        y_extra = y[dup_idx]

        X_list.append(X_extra)
        y_list.append(y_extra)
        changed = True

    if not changed:
        return X_raw, y, False

    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)

    # 섞어서 순서 편향 제거
    perm = rng.permutation(len(y_new))
    X_new = X_new[perm]
    y_new = y_new[perm]

    return X_new.astype(np.float32), y_new.astype(np.int64), True


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
    (device_type 누락되는 거 고친 버전 + 그룹별 클래스 제대로 분리한 버전 + window 후보 성능 비교/최적화 버전)
    """
    # ▷ NEAR_ZERO_BAND(0% 근처 수익률 구간) 기준 값 가져오기
    #   - config에 NEAR_ZERO_BAND가 있으면 그 값을 우선 사용
    #   - 없으면 환경변수 NEAR_ZERO_BAND → 마지막으로는 0.0
    try:
        from config import NEAR_ZERO_BAND as _NEAR_ZERO_BAND  # type: ignore
    except Exception:
        try:
            _NEAR_ZERO_BAND = float(os.getenv("NEAR_ZERO_BAND", "0.0"))
        except Exception:
            _NEAR_ZERO_BAND = 0.0

    HAS_CUDA = torch.cuda.is_available()
    device_type = "cuda" if HAS_CUDA else "cpu"
    use_amp_here = (os.getenv("USE_AMP", "1") == "1") and HAS_CUDA

    if max_epochs is None:
        max_epochs = _epochs_for(strategy)

    res: Dict[str, Any] = {
        "symbol": symbol,
        "strategy": strategy,
        "group_id": int(group_id or 0),
        "windows": [],
        "models": [],
        # ★ window 최적화 결과를 리턴에도 남김
        "best_window": None,
        "best_f1": None,
        "best_acc": None,
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

        cfg = STRATEGY_CONFIG.get(strategy, {})  # ← 여기서 전략별 설정 사용
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
        # 5-1) 전체(전역) 클래스 구간
        if "class_ranges_used_global" in locals() and class_ranges_used_global is not None:
            full_ranges = class_ranges_used_global
        else:
            full_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=None)

        num_total_classes = len(full_ranges) if full_ranges is not None else 0

        # 5-2) 그룹별 클래스 인덱스 (전역 인덱스 리스트)
        from config import get_class_groups  # 재임포트 안전

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

        if not cls_in_group:
            _log_skip(symbol, strategy, f"그룹 내 클래스 없음(group_id={group_id})")
            return res

        # 5-3) 이번 학습에 사용할 "로컬 클래스 구간" (그룹에 해당하는 구간만)
        class_ranges = [full_ranges[c] for c in cls_in_group]
        keep_set = set(cls_in_group)
        to_local = {g: i for i, g in enumerate(sorted(cls_in_group))}

        # ===== 6. 라벨/분포 로그 출력 =====
        mask_cnt = int((labels < 0).sum())

        # ▷ NEAR_ZERO_BAND 기준으로 0% 근처 수익률 개수 계산
        nz_band = float(abs(_NEAR_ZERO_BAND)) if _NEAR_ZERO_BAND is not None else 0.0
        if nz_band <= 0.0:
            # 설정이 0이면, 최소한 BOUNDARY_BAND라도 같이 찍어줌
            try:
                nz_band = float(abs(BOUNDARY_BAND))
            except Exception:
                nz_band = 0.0

        if nz_band > 0.0:
            near_zero_mask = np.abs(np.asarray(gains, dtype=np.float32)) <= nz_band
            near_zero_cnt = int(near_zero_mask.sum())
            near_zero_ratio = near_zero_cnt / max(1, len(gains))
        else:
            near_zero_cnt = 0
            near_zero_ratio = 0.0

        total_labels = max(1, len(labels))
        mask_ratio = mask_cnt / total_labels

        _safe_print(
            f"[LABELS] total={len(labels)} "
            f"masked={mask_cnt} ({mask_ratio:.2%}) "
            f"BOUNDARY_BAND=±{BOUNDARY_BAND} "
            f"NEAR_ZERO_BAND=±{nz_band} "
            f"near_zero={near_zero_cnt} ({near_zero_ratio:.2%})"
        )

        try:
            # 전역 기준 분포 (full_ranges 크기만큼)
            cnt_before = np.bincount(
                labels[labels >= 0],
                minlength=num_total_classes
            ).astype(int).tolist()
        except Exception:
            cnt_before = []

        num_classes_effective = int(np.unique(labels[labels >= 0]).size) if labels.size else 0
        empty_idx = [i for i, c in enumerate(cnt_before) if int(c) == 0]

        return_note = ""
        if isinstance(bin_info, dict):
            edges = bin_info.get("bin_edges", [])
            counts = bin_info.get("bin_counts", [])
            return_note = (
                f" ; [ReturnDist] edges={edges[:20]}, counts={counts[:20]}"
            )

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
                note=(
                    f"[LabelStats] bins_total={num_total_classes}, "
                    f"bins_group={len(class_ranges)}, empty={len(empty_idx)}, "
                    f"classes={num_classes_effective}, empty_idx={empty_idx[:8]}, "
                    f"masked={mask_cnt}, near_zero={near_zero_cnt}, "
                    f"NEAR_ZERO_BAND=±{nz_band}"
                    + return_note
                ),
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

        # ===== 8. 윈도우 후보 (config 기반 + window_optimizer) =====
        base_windows = [16, 20, 24, 28, 32]
        try:
            cfg_windows = cfg.get("windows") or cfg.get("window_list")
            if isinstance(cfg_windows, (list, tuple)) and cfg_windows:
                base_windows = [
                    int(w) for w in cfg_windows
                    if isinstance(w, (int, float)) and w == w
                ]
        except Exception:
            pass

        if not base_windows:
            base_windows = [16, 20, 24, 28, 32]

        try:
            top_windows = find_best_windows(
                symbol,
                strategy,
                window_list=base_windows,
                top_k=3,
                group_id=group_id,
            )
        except Exception:
            try:
                top_windows = [
                    int(
                        find_best_window(
                            symbol,
                            strategy,
                            window_list=base_windows,
                            group_id=group_id,
                        )
                    )
                ]
            except Exception:
                top_windows = base_windows[:1]

        top_windows = [
            int(max(5, w))
            for w in top_windows
            if isinstance(w, (int, float)) and w == w
        ] or [base_windows[0]]
        _safe_print(f"[WINDOWS] candidates(base={base_windows}) → top={top_windows}")

        # ★ 윈도우 전체 중 "최고" 성능 추적
        best_window_overall: Optional[int] = None
        best_f1_overall: float = -1.0
        best_acc_overall: float = 0.0

        # ===== 9. 윈도우별 학습 =====
        for window in top_windows:
            if stop_event is not None and stop_event.is_set():
                break

            window = min(window, max(6, len(features_only) - 1))
            fv = features_only.values.astype(np.float32)

            # ★ 여기서 그룹 정보(keep_set, to_local)를 사용해서 그룹 클래스만 학습
            X_raw, y = _rebuild_samples_with_keepset(
                fv=fv,
                labels=labels,
                window=window,
                keep_set=keep_set,
                to_local=to_local,
                min_samples=1,  # ← 더 이상 컷 기준으로 쓰지 않음
            )

            repaired_info = {
                "neighbor_expansion": False,
                "synthetic_labels": False,
            }

            # ★ 희소 클래스 보강: 삭제가 아니라 "복사"로만 채운다
            if X_raw.size:
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

            # 이번 학습에서 사용하는 클래스 수 = 그룹 내 클래스 수
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
                        y,
                        val_frac=0.20,
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

            train_idx, val_idx, _ = _ensure_val_has_two_classes(
                train_idx, val_idx, y, min_classes=2
            )

            try:
                # y 는 "그룹 로컬 클래스" 기준
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
                from collections import Counter as _Ctr
                cls_dist = _Ctr(y_train.tolist())
                _safe_print("[BALANCE RESULT] 클래스별 샘플 수:")
                for cls_id in sorted(cls_dist.keys()):
                    _safe_print(f"  - class {cls_id}: {cls_dist[cls_id]}개")
            except Exception as e:
                _safe_print(f"[BALANCE RESULT skip] {e}")
            # ────────────────────────────────────────────────────────

            sampler = None
            if WEIGHTED_SAMPLER_FLAG:
                try:
                    w_cls = compute_class_weights(
                        y_train, method="effective", beta=0.999
                    )
                    sampler = make_weighted_sampler(
                        y_train, class_weights=w_cls, replacement=True
                    )
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
                num_classes=len(class_ranges),  # 그룹 내 클래스 수만큼 출력
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
                all_preds: List[np.ndarray] = []
                all_lbls: List[np.ndarray] = []
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
                        f1_val = f1_score(
                            lbls, preds, average="macro", zero_division=0
                        )
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
                    _safe_print(
                        f"[EARLY STOP] {symbol}-{strategy}-w{window} no_improve={no_improve}"
                    )
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

            # ★ 윈도우 전체 중 best 업데이트
            try:
                if f1_val > best_f1_overall or (
                    f1_val == best_f1_overall and acc > best_acc_overall
                ):
                    best_f1_overall = float(f1_val)
                    best_acc_overall = float(acc)
                    best_window_overall = int(window)
            except Exception:
                pass

            # ===== bin 정보 정리 및 저장 =====
            try:
                if isinstance(bin_info, dict) and "bin_edges" in bin_info:
                    bin_edges = [float(x) for x in bin_info.get("bin_edges", [])]
                    bin_spans = bin_info.get("bin_spans", [])
                    if bin_edges and (not bin_spans or len(bin_spans) != len(bin_edges) - 1):
                        bin_spans = [
                            float(bin_edges[i + 1] - bin_edges[i])
                            for i in range(len(bin_edges) - 1)
                        ]
                    full_ranges_for_bins = get_class_ranges(
                        symbol=symbol, strategy=strategy, group_id=None
                    )
                    cnt_local = np.bincount(
                        val_y, minlength=len(class_ranges)
                    ).astype(int)
                    counts_map = np.zeros(len(full_ranges_for_bins), dtype=int)
                    for g, l in to_local.items():
                        if g < len(counts_map) and l < len(cnt_local):
                            counts_map[g] = int(cnt_local[l])
                    bin_counts = counts_map.tolist()
                else:
                    full_ranges_for_bins = get_class_ranges(
                        symbol=symbol, strategy=strategy, group_id=None
                    )
                    bin_edges = [float(lo) for (lo, _) in full_ranges_for_bins] + [
                        float(full_ranges_for_bins[-1][1])
                    ]
                    bin_spans = [
                        float(hi - lo) for (lo, hi) in full_ranges_for_bins
                    ]
                    cnt_local = np.bincount(
                        val_y, minlength=len(class_ranges)
                    ).astype(int)
                    counts_map = np.zeros(len(full_ranges_for_bins), dtype=int)
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

            stem = os.path.join(
                MODEL_DIR,
                f"{symbol}_{strategy}_{model_type}_w{int(window)}_group{int(group_id) if group_id is not None else 0}_cls{int(len(class_ranges))}",
            )
            meta = {
                "symbol": symbol,
                "strategy": strategy,
                "model": model_type,
                "group_id": int(group_id or 0),
                "num_classes": int(len(class_ranges)),
                "class_ranges": [
                    [float(lo), float(hi)] for (lo, hi) in class_ranges
                ],
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
                "cs_argmax": {
                    "enabled": bool(COST_SENSITIVE_ARGMAX),
                    "beta": float(CS_ARG_BETA),
                },
                "eval_gate": "none",
                "label_repair": repaired_info,
                "bin_edges": bin_edges,
                "bin_counts": bin_counts,
                "bin_spans": bin_spans,
                "bin_cfg": bin_cfg,
                "near_zero_band": float(nz_band),
                "near_zero_count": int(near_zero_cnt),
            }
            wpath, mpath = _save_model_and_meta(model, stem + ".pt", meta)

            # ==== 🔁 여기서부터: logger 로 넘길 값들을 JSON 문자열로 변환 ====
            try:
                class_ranges_for_log = [
                    [float(lo), float(hi)] for (lo, hi) in class_ranges
                ]
            except Exception:
                class_ranges_for_log = class_ranges

            def _to_json_safe(obj):
                try:
                    return json.dumps(obj, ensure_ascii=False)
                except Exception:
                    return str(obj)

            class_ranges_json = _to_json_safe(class_ranges_for_log)
            bin_edges_json = _to_json_safe(bin_edges)
            bin_counts_json = _to_json_safe(bin_counts)
            bin_spans_json = _to_json_safe(bin_spans)
            bins_value = len(bin_edges) if isinstance(bin_edges, (list, tuple)) else None
            # ============================================================

            try:
                # 한글 요약 note: 이 한 줄만 봐도 상태를 바로 이해할 수 있게
                summary_parts = [
                    f"[학습요약] {symbol}-{strategy}",
                    f"윈도우={int(window)}",
                    f"클래스={len(class_ranges)}개",
                    f"샘플={usable_samples}개",
                    f"정확도={acc:.4f}",
                    f"F1={f1_val:.4f}",
                ]
                if note_msg:
                    summary_parts.append(note_msg)
                if bin_edges and bin_counts:
                    try:
                        nz_ratio_pct = near_zero_ratio * 100.0
                    except Exception:
                        nz_ratio_pct = 0.0
                    summary_parts.append(
                        f"수익률구간={len(bin_edges)-1}개, 0%근처(±{nz_band:.4f}) 비중≈{nz_ratio_pct:.1f}%"
                    )
                final_note = " | ".join(summary_parts) + return_note

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
                    y_true=lbls.tolist()
                    if isinstance(lbls, np.ndarray)
                    else lbls,
                    y_pred=preds.tolist()
                    if isinstance(preds, np.ndarray)
                    else preds,
                    num_classes=len(class_ranges),
                    NUM_CLASSES=int(num_total_classes),
                    class_counts_label_freeze=cnt_before,
                    usable_samples=usable_samples,
                    class_counts_after_assemble=cnt_after,
                    batch_stratified_ok=batch_stratified_ok,
                    # 추가: 클래스/수익률 구간도 로그에 포함 (JSON 문자열)
                    class_ranges=class_ranges_json,
                    bin_edges=bin_edges_json,
                    bin_counts=bin_counts_json,
                    bin_spans=bin_spans_json,
                    # 🔁 프론트 호환용 별칭 필드도 같이 기록
                    class_edges=bin_edges_json,
                    class_counts=bin_counts_json,
                    bins=bins_value,
                    # 추가: near-zero 통계도 로그에 포함
                    near_zero_band=float(nz_band),
                    near_zero_count=int(near_zero_cnt),
                    masked_count=int(mask_cnt),
                )
            except Exception:
                pass

            res["windows"].append(int(window))
            res["models"].append(os.path.basename(wpath))

            if IMPORTANCE_ENABLE:
                try:
                    fi = compute_feature_importance(
                        model, features_only, device=DEVICE
                    )
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

        # ===== 10. 전체 window 중 BEST 한 번 더 요약 로그 =====
        try:
            if best_window_overall is not None:
                _safe_print(
                    f"[WINDOW BEST] {symbol}-{strategy}-g{group_id} "
                    f"best_window={best_window_overall} "
                    f"f1={best_f1_overall:.4f} acc={best_acc_overall:.4f}"
                )
                logger.log_training_result(
                    symbol,
                    strategy,
                    model="best_window",
                    accuracy=best_acc_overall,
                    f1=best_f1_overall,
                    loss=None,
                    val_acc=best_acc_overall,
                    val_f1=best_f1_overall,
                    val_loss=None,
                    engine="manual",
                    window=int(best_window_overall),
                    recent_cap=int(len(features_only)),
                    rows=None,
                    limit=None,
                    min=None,
                    augment_needed=None,
                    enough_for_training=None,
                    note=(
                        f"[WindowBest] window={best_window_overall} "
                        f"f1={best_f1_overall:.4f} acc={best_acc_overall:.4f}"
                    ),
                    source_exchange="BYBIT",
                    status="best",
                )
                # 리턴값에도 반영
                res["best_window"] = int(best_window_overall)
                res["best_f1"] = float(best_f1_overall)
                res["best_acc"] = float(best_acc_overall)
        except Exception:
            pass

        return res

    except Exception as e:
        _safe_print(
            f"[EXC] train_one_model {symbol}-{strategy}-g{group_id} → {e}\n{traceback.format_exc()}"
        )
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
    (현재 미사용) YOPO 학습 안정화를 위한 '진짜' 클래스 균형 함수 틀.
    - 여기서는 호출하지 않는다. 희소 클래스 삭제 방지를 위해 비활성 유지.
    """
    if len(y_train) == 0:
        return X_train, y_train
    return X_train.astype(np.float32), y_train.astype(np.int64)



if __name__ == "__main__":
    try:
        start_train_loop(force_restart=True, sleep_sec=0)
    except Exception as e:
        _safe_print(f"[MAIN] err: {e}")
