# train.py — SPEED v2.6 FINAL (희소클래스 제거 없이 그룹 클래스 전부 학습 버전)
# -*- coding: utf-8 -*-
import sitecustomize
import os, time, glob, shutil, json, random, traceback, threading, gc, csv, re
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from logger import ensure_train_log_exists

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
# ✅ 경로 통일(핵심)
# - train_log.csv가 있는 "persistent"를 기준으로 logs/models/run 모두 고정
# - /tmp/persistent 같은 다른 루트로 갈라지지 않게 막음
# ================================================
TRAIN_LOG = get_TRAIN_LOG_PATH()  # /persistent/logs/train_log.csv 같은 "진짜" 경로

# train_log.csv -> .../persistent/logs/train_log.csv
# BASE는 .../persistent
_base_from_trainlog = os.path.dirname(os.path.dirname(TRAIN_LOG))

BASE_PERSIST_DIR = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or _base_from_trainlog
)

try:
    os.makedirs(BASE_PERSIST_DIR, exist_ok=True)
except Exception:
    pass

# logs는 train_log.csv가 있는 폴더로 고정
LOG_DIR = os.path.dirname(TRAIN_LOG)
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    pass

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_PERSIST_DIR, "models"))
RUN_DIR   = os.getenv("RUN_DIR",   os.path.join(BASE_PERSIST_DIR, "run"))

for _d in (MODEL_DIR, RUN_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass

GROUP_ACTIVE_PATH = os.path.join(BASE_PERSIST_DIR, "GROUP_ACTIVE")
PERSIST_DIR = BASE_PERSIST_DIR
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

    # ✅ train.py 패치가 실제로 쓰는 필드 (누락되면 CSV에 안 써짐)
    "group",
    "epoch",
    "train_loss",
    "notes",

    # 추가 메트릭
    "accuracy",
    "f1",
    "loss",
    "y_true",
    "y_pred",
    "num_classes",

    # ✅ 라벨/분포 관련 (패치가 채우는 키)
    "label_total",
    "label_masked",
    "label_masked_ratio",
    "near_zero",
    "boundary_band",
    "val_coverage",

    # === 진단 5종 ===
    "NUM_CLASSES",
    "class_counts_label_freeze",
    "usable_samples",
    "class_counts_after_assemble",
    "batch_stratified_ok",

    # === 수익률/클래스 구간 요약 ===
    "near_zero_band",
    "near_zero_count",
    "masked_count",
    "class_ranges",
    "bin_edges",
    "bin_counts",
    "bin_spans",

    # 🔥 프론트엔드 호환 필드
    "class_edges",      # = bin_edges
    "class_counts",     # = bin_counts
    "bins",             # = len(bin_edges) - 1

    # 🔥 per-class 성능 저장용
    "per_class_f1",

    # 🔥 학습로그 카드 UI용 필수 6개
    "ui_data_summary",
    "ui_dist_summary",
    "ui_class_range_summary",
    "ui_class_count_summary",
    "ui_usable_summary",
    "ui_performance_summary",

    # ✅ _log_training_result_patched()가 실제로 채우는 "구형 UI 요약"
    "ui_status",
    "ui_data_amount",
    "ui_return_summary",
    "ui_coverage_summary",
]

try:
    from logger import TRAIN_HEADERS

    # logger 쪽 기본 헤더와 우리가 추가한 헤더를 합침(중복 제거)
    TRAIN_HEADERS = list(dict.fromkeys(list(TRAIN_HEADERS) + DEFAULT_TRAIN_HEADERS))
except Exception:
    TRAIN_HEADERS = DEFAULT_TRAIN_HEADERS

# ✅✅✅ 핵심 수정: train_log 경로는 절대 재작성하지 말고, config.get_TRAIN_LOG_PATH() 그대로 쓴다.
TRAIN_LOG = get_TRAIN_LOG_PATH()
try:
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
except Exception:
    TRAIN_LOG = get_TRAIN_LOG_PATH()


def _ensure_train_log():
    """
    ✅ train.py에서는 헤더를 '직접' 새로 만들지 않는다.
    ✅ 오직 logger.py의 ensure_train_log_exists()만 사용한다.
    """
    try:
        ensure_train_log_exists()
    except Exception as e:
        print(f"[FATAL] ensure_train_log_exists() failed: {e}", flush=True)
        raise


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

    if r.get("engine") in (None, ""):
        r["engine"] = row.get("engine", "manual")
    if r.get("source_exchange") in (None, ""):
        r["source_exchange"] = row.get("source_exchange", "BYBIT")

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


def _compute_bin_info_from_labels(
    labels: np.ndarray,
    class_ranges: list,
    to_local: dict,
    bin_edges: list | None = None,
):
    """
    - y(전체 labels) 기준으로 클래스 분포 계산 ✅
    """
    num_global = len(class_ranges)
    if num_global <= 0:
        return {"bin_edges": [], "bin_counts": [], "bin_spans": [], "bins": 0}

    counts_global = np.zeros(num_global, dtype=int)
    for g, l in to_local.items():
        if g < num_global:
            counts_global[g] = int((labels == g).sum())

    if not bin_edges:
        bin_edges = [float(lo) for (lo, _) in class_ranges]
        bin_edges.append(float(class_ranges[-1][1]))

    bin_spans = [float(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)]

    return {
        "bin_edges": list(bin_edges),
        "bin_counts": counts_global.tolist(),
        "bin_spans": bin_spans,
        "bins": len(bin_edges) - 1,
    }


def _append_train_log(row: dict):
    """
    ✅✅✅ 핵심: 무조건 get_TRAIN_LOG_PATH()로만 기록한다.
    """
    try:
        _ensure_train_log()

        _path = get_TRAIN_LOG_PATH()
        try:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
        except Exception:
            pass

        with open(_path, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRAIN_HEADERS, extrasaction="ignore")
            w.writerow(_normalize_train_row(row))

        try:
            print(f"[TRAIN_LOG APPEND OK] path={_path} symbol={row.get('symbol')} strategy={row.get('strategy')} status={row.get('status')}", flush=True)
        except Exception:
            pass

    except Exception as e:
        print(f"[FATAL] train_log append failed: {e}", flush=True)
        raise


# =========================================================
# ✅✅✅ [핵심 수정] logger 패치는 "전역에서 1번만" 적용되어야 한다
# - 원래 함수 백업(_orig_ltr) 없으면 패치 함수가 깨짐
# - 패치 적용 코드가 함수 안(들여쓰기)으로 들어가면 적용이 불안정함
# =========================================================

# 1) 원본 백업 (딱 1번)
if not hasattr(logger, "_orig_log_training_result"):
    logger._orig_log_training_result = logger.log_training_result

_orig_ltr = logger._orig_log_training_result


def _log_training_result_patched(*args, **kw):
    """
    logger.log_training_result 를 가로채서:
    1) 원래 logger 로깅 실행
    2) train_log.csv 에 기록할 row 생성(운영로그 요약 + 분포/구간/성능 포함)
    """
    # 1) 원래 logger 호출
    if callable(_orig_ltr):
        try:
            _orig_ltr(*args, **kw)
        except Exception as e:
            print(f"[경고] logger.log_training_result 실패: {e}", flush=True)

    # 2) symbol/strategy/model 복구
    symbol = args[0] if len(args) > 0 else kw.get("symbol")
    strategy = args[1] if len(args) > 1 else kw.get("strategy")
    model = args[2] if len(args) > 2 else kw.get("model", "")

    result_dict = None
    if "result" in kw and isinstance(kw["result"], dict):
        result_dict = kw["result"]

    row: Dict[str, Any] = {
        "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "model": model,

        "note": kw.get("note", ""),
        "notes": kw.get("notes", ""),

        "status": kw.get("status", ""),
        "window": kw.get("window", ""),
        "group": kw.get("group", ""),
        "epoch": kw.get("epoch", ""),

        "val_acc": kw.get("val_acc", ""),
        "val_f1": kw.get("val_f1", ""),
        "val_loss": kw.get("val_loss", ""),
        "train_loss": kw.get("train_loss", ""),

        "engine": kw.get("engine", "manual"),
        "source_exchange": kw.get("source_exchange", "BYBIT"),
        "rows": kw.get("rows", ""),
        "limit": kw.get("limit", ""),
        "min": kw.get("min", ""),
        "augment_needed": kw.get("augment_needed", ""),
        "enough_for_training": kw.get("enough_for_training", ""),

        "label_total": kw.get("label_total", ""),
        "label_masked": kw.get("label_masked", ""),
        "label_masked_ratio": kw.get("label_masked_ratio", ""),
        "near_zero": kw.get("near_zero", ""),
        "near_zero_band": kw.get("near_zero_band", ""),
        "near_zero_count": kw.get("near_zero_count", ""),
        "boundary_band": kw.get("boundary_band", ""),
        "masked_count": kw.get("masked_count", ""),
        "bin_edges": kw.get("bin_edges", ""),
        "bin_counts": kw.get("bin_counts", ""),
        "bin_spans": kw.get("bin_spans", ""),
        "class_ranges": kw.get("class_ranges", ""),
        "class_edges": kw.get("class_edges", ""),
        "class_counts": kw.get("class_counts", ""),
        "bins": kw.get("bins", ""),
        "val_coverage": kw.get("val_coverage", ""),

        "num_classes": kw.get("num_classes", ""),
        "NUM_CLASSES": kw.get("NUM_CLASSES", ""),
        "usable_samples": kw.get("usable_samples", ""),
        "class_counts_label_freeze": kw.get("class_counts_label_freeze", ""),
        "class_counts_after_assemble": kw.get("class_counts_after_assemble", ""),
        "batch_stratified_ok": kw.get("batch_stratified_ok", ""),

        "per_class_f1": kw.get("per_class_f1", ""),

        "ui_status": kw.get("ui_status", ""),
        "ui_data_amount": kw.get("ui_data_amount", ""),
        "ui_return_summary": kw.get("ui_return_summary", ""),
        "ui_coverage_summary": kw.get("ui_coverage_summary", ""),

        "ui_data_summary": kw.get("ui_data_summary", ""),
        "ui_dist_summary": kw.get("ui_dist_summary", ""),
        "ui_class_range_summary": kw.get("ui_class_range_summary", ""),
        "ui_class_count_summary": kw.get("ui_class_count_summary", ""),
        "ui_usable_summary": kw.get("ui_usable_summary", ""),
        "ui_performance_summary": kw.get("ui_performance_summary", ""),
    }

    if isinstance(result_dict, dict):
        for k, v in result_dict.items():
            if k in row:
                row[k] = v

    for k, v in kw.items():
        if k in row and v not in (None, ""):
            row[k] = v

    def _jsonify_if_needed(v):
        if v is None:
            return None
        if isinstance(v, (dict, list, tuple)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return v

    def _restore_json(val):
        if val in (None, "", "null"):
            return None
        if isinstance(val, (list, dict)):
            return val
        if isinstance(val, str):
            s = val.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    return json.loads(s)
                except Exception:
                    return val
        return val

    for k in ("bin_edges", "bin_counts", "bin_spans", "class_ranges", "class_edges", "class_counts", "val_coverage", "per_class_f1"):
        row[k] = _restore_json(row.get(k))

    try:
        be = row.get("bin_edges")
        bc = row.get("bin_counts")
        if row.get("bins") in (None, "", 0):
            if isinstance(be, list) and len(be) >= 2:
                row["bins"] = len(be) - 1
            elif isinstance(bc, list):
                row["bins"] = len(bc)
    except Exception:
        pass

    try:
        if row.get("num_classes") in (None, "", 0):
            cr = row.get("class_ranges")
            be2 = row.get("bin_edges")
            cc2 = row.get("class_counts")
            if isinstance(cr, list) and cr:
                row["num_classes"] = int(len(cr))
            elif isinstance(be2, list) and len(be2) >= 2:
                row["num_classes"] = int(len(be2) - 1)
            elif isinstance(cc2, list) and cc2:
                row["num_classes"] = int(len(cc2))
    except Exception:
        pass

    try:
        status = (row.get("status") or "").strip()
        if not row.get("ui_status"):
            if status in ("success", "best"):
                row["ui_status"] = f"✅ 정상 학습 (status={status})"
            elif status in ("info",):
                row["ui_status"] = f"ℹ️ 참고용 로그 (status={status})"
            elif status in ("warn", "warning"):
                row["ui_status"] = f"🟠 경고 (status={status})"
            elif status in ("error", "fail", "failed"):
                row["ui_status"] = f"🔴 실패/오류 (status={status})"
            else:
                row["ui_status"] = f"상태: {status or '미상'}"

        need_new_ui = any(not row.get(k) for k in (
            "ui_data_summary", "ui_dist_summary", "ui_class_range_summary",
            "ui_class_count_summary", "ui_usable_summary", "ui_performance_summary"
        ))

        if need_new_ui:
            rows = int(row.get("rows") or 0)
            usable = int(row.get("usable_samples") or 0)
            acc = float(row.get("val_acc") or row.get("accuracy") or 0.0)
            f1v = float(row.get("val_f1") or row.get("f1") or 0.0)
            be = row.get("bin_edges") or []
            bc = row.get("bin_counts") or []
            cr = row.get("class_ranges") or []

            summary = make_training_summary_fields(
                rows=rows,
                bin_edges=be if isinstance(be, list) else [],
                bin_counts=bc if isinstance(bc, list) else [],
                class_ranges=cr if isinstance(cr, list) else [],
                usable_samples=usable,
                acc=acc,
                f1=f1v,
            )
            for k, v in summary.items():
                if not row.get(k):
                    row[k] = v

        if not row.get("ui_data_amount"):
            rows_v = row.get("rows")
            row["ui_data_amount"] = f"학습 데이터: {rows_v}개" if rows_v not in (None, "", 0) else "학습 데이터 정보 없음"
        if not row.get("ui_return_summary"):
            be = row.get("bin_edges")
            if isinstance(be, list) and len(be) >= 2:
                row["ui_return_summary"] = f"수익률 범위: {float(be[0]):.4f} ~ {float(be[-1]):.4f}"
            else:
                row["ui_return_summary"] = "수익률 분포 정보 없음"
        if not row.get("ui_coverage_summary"):
            cov = row.get("val_coverage")
            if isinstance(cov, dict) and cov.get("total"):
                covered = int(cov.get("covered", 0))
                total = int(cov.get("total", 0))
                row["ui_coverage_summary"] = f"검증 커버리지: {covered}/{total}"
            else:
                row["ui_coverage_summary"] = "검증 커버리지 정보 없음"
    except Exception as e:
        print(f"[train_log summary warn] {e}", flush=True)

    for k in list(row.keys()):
        row[k] = _jsonify_if_needed(row[k])

    _append_train_log(row)

    try:
        logger.update_train_dashboard(
            symbol=row.get("symbol"),
            strategy=row.get("strategy"),
            model=row.get("model", "")
        )
    except Exception as e:
        print(f"[경고] train_dashboard 업데이트 실패: {e}", flush=True)


# ✅ 패치 적용은 여기서 "전역 1번"만
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
BALANCE_CLASSES_FLAG = os.getenv("BALANCE_CLASSES", "1") == "1"
WEIGHTED_SAMPLER_FLAG = os.getenv("WEIGHTED_SAMPLER", "0") == "1"

MIN_CLASS_SAMPLES = int(os.getenv("MIN_CLASS_SAMPLES", "4"))
MAX_CLASS_UPSAMPLE = int(os.getenv("MAX_CLASS_UPSAMPLE", "64"))
# ===================================================================

def _as_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


COST_SENSITIVE_ARGMAX = _as_bool_env("COST_SENSITIVE_ARGMAX", True)
CS_ARG_BETA = float(os.getenv("CS_ARG_BETA", "1.0"))


def _epochs_for(strategy: str) -> int:
    if strategy == "단기":
        return int(os.getenv("EPOCHS_SHORT", "36"))
    if strategy == "중기":
        return int(os.getenv("EPOCHS_MID", "24"))
    if strategy == "장기":
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
        covered = len([1 for v in cnt.values() if v > 0])
        coverage = covered / max(1, int(num_classes))

        if best is None or coverage > best[0]:
            best = (coverage, start, end, cnt, covered)

        if coverage >= float(min_coverage):
            break

        end -= int(max(1, stride))
        tried += 1

    if best is None:
        start, end = max(0, n - val_len), n
        cnt = Counter(y[start:end].tolist())
        covered = len(cnt)
        coverage = covered / max(1, int(num_classes))
    else:
        coverage, start, end, cnt, covered = best

    val_idx = np.arange(start, end)
    train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)], axis=0)

    # ✅ 원하는 출력 포맷으로 변경
    _safe_print(
        f"[VAL][COVER] num_classes={int(num_classes)} covered={int(covered)} coverage={coverage*100:.1f}%"
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




_ENFORCE_FULL_STRATEGY = False
_STRICT_HALT_ON_INCOMPLETE = False
_REQUIRE_AT_LEAST_ONE_MODEL_PER_GROUP = False
_SYMBOL_RETRY_LIMIT = int(os.getenv("SYMBOL_RETRY_LIMIT", "1"))


def make_training_summary_fields(
    rows, 
    bin_edges, 
    bin_counts, 
    class_ranges, 
    usable_samples,
    acc, 
    f1
):
    """웹 UI가 읽을 수 있는 매우 쉬운 요약본문 생성"""

    # (1) 데이터 개수
    data_summary = f"학습에 사용된 데이터: 총 {rows}개"

    # (2) 수익률 구간 요약
    try:
        if isinstance(bin_edges, list) and len(bin_edges) >= 2:
            lo = float(bin_edges[0])
            hi = float(bin_edges[-1])
            dist_summary = f"수익률 분포: 최소 {lo:.4f} ~ 최대 {hi:.4f}"
        else:
            dist_summary = "수익률 분포 정보 없음"
    except:
        dist_summary = "수익률 분포 정보 없음"

    # (3) 클래스별 구간 요약
    try:
        if isinstance(class_ranges, list) and class_ranges:
            cr = "; ".join([f"{float(lo):.4f}~{float(hi):.4f}" for lo, hi in class_ranges])
            class_range_summary = f"클래스별 수익률 구간: {cr}"
        else:
            class_range_summary = "클래스별 구간 정보 없음"
    except:
        class_range_summary = "클래스별 구간 정보 없음"

    # (4) 클래스 개수
    num_classes = len(class_ranges) if class_ranges else 0
    class_count_summary = f"클래스 개수: {num_classes}개"

    # (5) 실제 usable 샘플 수
    usable_summary = f"실제 학습에 사용한 샘플: {usable_samples}개"

    # (6) 모델 성능
    perf_summary = f"정확도 {acc:.4f}, F1 {f1:.4f}"

    return {
        "ui_data_summary": data_summary,
        "ui_dist_summary": dist_summary,
        "ui_class_range_summary": class_range_summary,
        "ui_class_count_summary": class_count_summary,
        "ui_usable_summary": usable_summary,
        "ui_performance_summary": perf_summary,
    }


def _train_full_symbol(
    symbol: str, stop_event: Optional[threading.Event] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    ✅ 수정 포인트
    - train_one_model()이 3모델을 모두 시도하므로,
      여기서는 "저장된 모델이 1개라도 있으면 ok_once=True"로 처리(기존과 동일)
    - detail에 모델 타입도 같이 남김(어떤 모델이 실제로 저장됐는지 확인용)
    """
    strategies = ["단기", "중기", "장기"]
    detail = {}
    any_saved = False

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

                attempts = _SHORT_RETRY if strategy == "단기" else 1
                ok_once = False
                trained_types = []

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
                        trained_types = list(res.get("trained_model_types") or [])
                        break

                detail[strategy][gid] = {"ok": ok_once, "trained_model_types": trained_types}
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
            detail[strategy] = {-1: {"ok": False, "trained_model_types": []}}

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
