# YOPO v1.7.2 — model_weight_loader.py
# (LRU+TTL 캐시, VRAM 누수 차단, meta.bin_edges 자동보정 + class_ranges 정규화 주입)
# - 변경점 (v1.7.2)
#   1) _sanitize_range(), _normalize_class_ranges(): meta.class_ranges를 분수(±1==±100%)로 정규화
#   2) _preflight(): bin_edges 자동보정 + class_ranges 정규화 반영
#   3) _attach_bin_info(): class_ranges/num_classes/meta 전체를 모델 객체 속성으로 주입
#   4) 반환 시그니처는 유지(return model_obj)해 predict.py와 완전 호환

import os
import json
import glob
import time
import re
import threading
from typing import Optional, Dict, Any, Tuple, List
from collections import OrderedDict

import torch
import pandas as pd

# 프로젝트 표준 I/O (PT / PTZ / safetensors 지원)
from model_io import load_model as _load_with_policy, ModelLoadError, load_meta as _load_meta

__all__ = [
    "load_model_cached",
    "find_models_fuzzy",
    "locate_best_weight",
    "get_model_weight",
    "model_exists",
    "count_models_per_strategy",
    "clear_cache",
]

MODEL_DIR = "/persistent/models"
EVAL_RESULT_SINGLE = "/persistent/evaluation_result.csv"
SUPPORTED_WEIGHTS = (".pt", ".ptz", ".safetensors")

# ===== 캐시 파라미터 =====
_MAX_ITEMS = int(os.getenv("MODEL_CACHE_MAX_ITEMS", "32"))
_TTL = int(os.getenv("MODEL_CACHE_TTL_SEC", "900"))  # 15분
_MAX_BYTES = int(os.getenv("MODEL_CACHE_MAX_BYTES", "536870912"))  # 512MB

# meta.bin_edges 엄격 여부 (1=필수, 0=자동보정 허용[기본])
_REQUIRE_BIN_EDGES_STRICT = os.getenv("REQUIRE_BIN_EDGES_STRICT", "0").strip() == "1"
# 자동보정 생성 시 기본 구간 폭 (num_classes 기반 균등분할용)
_DEFAULT_EDGE_MIN = float(os.getenv("DEFAULT_EDGE_MIN", "-0.05"))
_DEFAULT_EDGE_MAX = float(os.getenv("DEFAULT_EDGE_MAX", "0.05"))

# ===== 운영 로그 레벨 제어 =====
# OP_LOG_LEVEL = ERROR / WARN / INFO / DEBUG
# 기본값: WARN (운영용 최소 로그)
_OP_LOG_LEVEL = os.getenv("OP_LOG_LEVEL", "WARN").strip().upper()
_LOG_RANK = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}
_MIN_RANK = _LOG_RANK.get(_OP_LOG_LEVEL, 1)

def _log(level: str, msg: str) -> None:
    try:
        lv = (level or "INFO").strip().upper()
        if _LOG_RANK.get(lv, 2) <= _MIN_RANK:
            print(msg)
    except Exception:
        pass

_cache_lock = threading.Lock()
# value: {"state":state_dict, "meta":dict, "ts":float, "ttl":int, "bytes":int}
_model_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_cache_bytes = 0

# ===== 유틸 =====
def _now() -> float:
    return time.time()

def _is_weight_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and path.lower().endswith(SUPPORTED_WEIGHTS)

def _stem(path: str) -> str:
    base = os.path.basename(path)
    return base.rsplit(".", 1)[0]

def _meta_path_for(weight_path: str) -> str:
    return os.path.splitext(os.path.abspath(weight_path))[0] + ".meta.json"

def _obj_bytes(obj: Any) -> int:
    try:
        if isinstance(obj, dict):
            return sum(v.numel() * v.element_size() for v in obj.values() if isinstance(v, torch.Tensor))
        if isinstance(obj, torch.nn.Module):
            return sum(p.numel() * p.element_size() for p in obj.parameters())
    except Exception:
        pass
    return 0

def _ensure_cpu_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    try:
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                sd[k] = v.cpu()
    except Exception:
        pass
    return sd

def _evict_if_needed(extra_bytes: int = 0):
    """TTL 만료 제거 + LRU/용량 한도 제거"""
    global _cache_bytes
    now = _now()
    # TTL
    for k in list(_model_cache.keys()):
        ent = _model_cache[k]
        if now - ent["ts"] > ent["ttl"]:
            _cache_bytes -= ent.get("bytes", 0)
            _model_cache.pop(k, None)
    # LRU/용량
    while _model_cache and (
        len(_model_cache) > _MAX_ITEMS
        or (_cache_bytes + max(0, extra_bytes)) > _MAX_BYTES
    ):
        old_key, ent = _model_cache.popitem(last=False)
        _cache_bytes -= ent.get("bytes", 0)

def clear_cache(pattern: str | None = None):
    """전역 캐시 비움. pattern 정규식으로 부분 삭제 가능."""
    global _cache_bytes
    with _cache_lock:
        if pattern is None:
            _model_cache.clear()
            _cache_bytes = 0
        else:
            rgx = re.compile(pattern)
            for k in list(_model_cache.keys()):
                if rgx.search(k):
                    ent = _model_cache.pop(k)
                    _cache_bytes -= ent.get("bytes", 0)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# ===== 메타/가중치 탐색 =====
def _resolve_meta(weight_path: str) -> Optional[str]:
    try:
        exact = _meta_path_for(weight_path)
        if os.path.exists(exact):
            return exact
        dirn = os.path.dirname(weight_path) or MODEL_DIR
        stem = _stem(weight_path)
        cands = sorted(glob.glob(os.path.join(dirn, f"{stem}*.meta.json")),
                       key=lambda p: os.path.getmtime(p), reverse=True)
        if cands:
            return cands[0]
        root_cands = sorted(glob.glob(os.path.join(MODEL_DIR, f"{os.path.basename(stem)}*.meta.json")),
                            key=lambda p: os.path.getmtime(p), reverse=True)
        if root_cands:
            return root_cands[0]
    except Exception as e:
        _log("DEBUG", f"[⚠️ META 탐색 실패] {weight_path} → {e}")
    return None

def _load_meta_dict(weight_path: str) -> Dict[str, Any]:
    """연결된 meta.json을 로드. 실패 시 {}.
    ✅ 운영로그 최소화를 위해 model_io.load_meta() 호출을 피하고, 여기서 직접 읽는다.
       (model_io 내부의 '손상 감지/재생성/가중치 미발견' 류 print를 원천 차단)
    """
    try:
        meta_path = _resolve_meta(weight_path)
        if not meta_path:
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _safe_write_meta(weight_path: str, meta: Dict[str, Any]) -> None:
    """보정된 meta를 원래 경로(.meta.json)에 덮어쓰기."""
    try:
        meta_path = _resolve_meta(weight_path) or _meta_path_for(weight_path)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _log("WARN", f"[⚠️ meta 저장 실패] {weight_path} → {e}")

# ===== 클래스 구간 정규화/보정 =====
def _sanitize_range(lo: float, hi: float) -> Tuple[float, float]:
    """절대값이 1을 넘으면 퍼센트로 간주하여 100으로 나눔."""
    try:
        lo_f, hi_f = float(lo), float(hi)
        if abs(lo_f) > 1 or abs(hi_f) > 1:
            lo_f /= 100.0
            hi_f /= 100.0
        return lo_f, hi_f
    except Exception:
        return float(lo or 0.0), float(hi or 0.0)

def _normalize_class_ranges(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """meta.class_ranges를 분수 기준으로 정규화."""
    try:
        cr = meta.get("class_ranges", None)
        if not (isinstance(cr, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in cr or [])):
            return meta, False
        fixed: List[List[float]] = []
        for lo, hi in cr:
            s_lo, s_hi = _sanitize_range(lo, hi)
            fixed.append([float(s_lo), float(s_hi)])
        if fixed != cr:
            meta["class_ranges"] = fixed
            return meta, True
        return meta, False
    except Exception:
        return meta, False

def _fill_meta_bin_edges(weight_path: str, meta: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    bin_edges 누락 시 합리적인 기본값을 생성해 meta에 주입.
    우선순위:
      1) class_ranges로부터 경계 추정 (연속/겹침 허용, 정렬 후 경계화)
      2) num_classes 기반 균등 분할 (DEFAULT_EDGE_MIN~DEFAULT_EDGE_MAX)
      3) 최후: 2구간 [-0.01, 0.01]
    반환: (meta_updated, was_modified)
    """
    try:
        be = meta.get("bin_edges", None)
        if isinstance(be, list) and len(be) >= 2:
            return meta, False

        # 1) class_ranges 기반
        cr = meta.get("class_ranges")
        if isinstance(cr, list) and len(cr) >= 1 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in cr):
            try:
                lows = [float(lo) for lo, _ in cr]
                his = [float(hi) for _, hi in cr]
                edges = sorted(set(lows + [max(his)]))
                if len(edges) >= 2:
                    meta["bin_edges"] = edges
                    return meta, True
            except Exception:
                pass

        # 2) num_classes 기반 균등 분할
        ncls = meta.get("num_classes", None)
        try:
            n = int(ncls) if ncls is not None else None
        except Exception:
            n = None
        if n is not None and n >= 1:
            import numpy as _np
            edges = _np.linspace(_DEFAULT_EDGE_MIN, _DEFAULT_EDGE_MAX, num=int(n) + 1, dtype=float).tolist()
            meta["bin_edges"] = edges
            return meta, True

        # 3) 최후: 2구간
        meta["bin_edges"] = [-0.01, 0.01]
        return meta, True
    except Exception:
        try:
            meta["bin_edges"] = [-0.01, 0.01]
            return meta, True
        except Exception:
            return meta, False

def _preflight(weight_path: str, model_type: str = "unknown", input_size: Optional[int] = None) -> Tuple[bool, str, Dict[str, Any]]:
    if not _is_weight_file(weight_path):
        return False, "weight_missing", {}
    meta = _load_meta_dict(weight_path)
    if not meta:
        return False, "meta_missing", {}

    if input_size is not None:
        mi = meta.get("input_size")
        if mi not in (None, input_size):
            return False, f"input_size_mismatch(meta={mi}, in={input_size})", meta

    # class_ranges 정규화
    meta, _ = _normalize_class_ranges(meta)

    # bin_edges 강제/자동보정
    be = meta.get("bin_edges", None)
    if not (isinstance(be, list) and len(be) >= 2):
        if _REQUIRE_BIN_EDGES_STRICT:
            return False, "bin_edges_missing", meta
        # 자동 보정 허용
        meta, fixed = _fill_meta_bin_edges(weight_path, meta)
        if fixed:
            _safe_write_meta(weight_path, meta)
            return True, "ok_autofixed", meta
        else:
            return False, "bin_edges_autofix_failed", meta

    return True, "ok", meta

def _find_weight_candidates(symbol: str, strategy: str, model_type: str) -> List[str]:
    patt = os.path.join(MODEL_DIR, "**", f"{symbol}_{strategy}_{model_type}*")
    found: List[str] = [p for p in glob.iglob(patt, recursive=True) if _is_weight_file(p)]
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found

def find_models_fuzzy(symbol: str, strategy: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        for root, _, files in os.walk(MODEL_DIR):
            for fn in files:
                if not fn.lower().endswith(SUPPORTED_WEIGHTS):
                    continue
                if not fn.startswith(f"{symbol}_"):
                    continue
                if f"_{strategy}_" not in fn:
                    continue
                wpath = os.path.join(root, fn)
                mpath = _resolve_meta(wpath)
                if not mpath:
                    continue
                out.append({
                    "pt_file": os.path.relpath(wpath, MODEL_DIR),
                    "meta_file": os.path.relpath(mpath, MODEL_DIR),
                })
        out.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x["pt_file"])), reverse=True)
    except Exception as e:
        _log("DEBUG", f"[⚠️ find_models_fuzzy 실패] {symbol}-{strategy} → {e}")
    return out

def locate_best_weight(symbol: str, strategy: str, model_type: str) -> Optional[str]:
    for w in _find_weight_candidates(symbol, strategy, model_type):
        ok, _, _ = _preflight(w, model_type=model_type, input_size=None)
        if ok:
            return w
    return None

# ===== 평가 로그 수집/가중치 계산 =====
def _load_all_eval_logs() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    eval_daily = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
    for fp in eval_daily:
        try:
            frames.append(pd.read_csv(fp, encoding="utf-8-sig"))
        except Exception as e:
            _log("DEBUG", f"[⚠️ 평가 파일 읽기 실패] {fp} → {e}")
    if os.path.exists(EVAL_RESULT_SINGLE):
        try:
            frames.append(pd.read_csv(EVAL_RESULT_SINGLE, encoding="utf-8-sig"))
        except Exception as e:
            _log("DEBUG", f"[⚠️ 단일 평가 파일 읽기 실패] {EVAL_RESULT_SINGLE} → {e}")
    if not frames:
        return pd.DataFrame()
    try:
        df = pd.concat(frames, ignore_index=True)
        if "timestamp" in df.columns:
            dedup_cols = [c for c in ["timestamp", "symbol", "strategy", "model", "status"] if c in df.columns]
            df = df.drop_duplicates(subset=dedup_cols)
        else:
            df = df.drop_duplicates()
        return df
    except Exception as e:
        _log("WARN", f"[⚠️ 평가 로그 병합 실패] → {e}")
        return pd.DataFrame()

def _symbol_from_meta_path(meta_path: str) -> Optional[str]:
    try:
        base = os.path.basename(meta_path).replace(".meta.json", "")
        parts = base.split("_")
        if parts:
            return parts[0]
    except Exception:
        pass
    return None

def _infer_model_type_from_fname(path: str) -> str:
    core = _stem(os.path.basename(path))
    parts = core.split("_")
    return parts[2] if len(parts) >= 3 else "unknown"

def get_model_weight(model_type: str, strategy: str, symbol: str = "ALL", min_samples: int = 3, input_size: Optional[int] = None) -> float:
    if symbol != "ALL":
        pattern = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json")
        meta_files = glob.glob(pattern) or glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}*.meta.json"))
    else:
        meta_files = glob.glob(os.path.join(MODEL_DIR, f"*_{strategy}_{model_type}*.meta.json"))

    if not meta_files:
        _log("DEBUG", f"[⚠️ meta 파일 없음] symbol={symbol}, strategy={strategy}, type={model_type} → weight=0.2")
        return 0.2

    df_all = _load_all_eval_logs()
    if df_all.empty:
        _log("DEBUG", "[INFO] 평가 데이터 없음 → cold-start weight=0.2")
        return 0.2

    need_cols = {"model", "strategy", "symbol", "status"}
    if not need_cols.issubset(df_all.columns):
        _log("DEBUG", "[INFO] 평가 로그 컬럼 부족 → cold-start weight=0.2")
        return 0.2

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        meta_symbol = meta.get("symbol") or _symbol_from_meta_path(meta_path)

        if input_size is not None and meta.get("input_size") not in (None, input_size):
            _log("DEBUG", f"[⚠️ input_size 불일치] meta={meta.get('input_size')} vs input={input_size} → weight=0.2")
            return 0.2

        base = meta_path.replace(".meta.json", "")
        if not any(os.path.exists(base + ext) for ext in SUPPORTED_WEIGHTS):
            _log("DEBUG", f"[⚠️ 모델 파일 없음] {base}.[pt|ptz|safetensors] → weight=0.2")
            return 0.2

        df = df_all[
            (df_all["model"] == model_type)
            & (df_all["strategy"] == strategy)
            & (df_all["symbol"] == meta_symbol)
            & (df_all["status"].isin(["success", "fail", "v_success", "v_fail"]))
        ].copy()

        n = len(df)
        if n < min_samples:
            _log("DEBUG", f"[INFO] 평가 샘플 부족 (len={n} < {min_samples}) → weight=0.2")
            return 0.2

        sr = (df["status"].isin(["success", "v_success"])).mean()
        if sr >= 0.7:
            _log("DEBUG", f"[INFO] success_rate={sr:.4f} → weight=1.0")
            return 1.0
        if sr < 0.3:
            _log("DEBUG", f"[INFO] success_rate={sr:.4f} → weight=0.0")
            return 0.0
        w = max(0.0, round((sr - 0.3) / 0.4, 4))
        _log("DEBUG", f"[INFO] success_rate={sr:.4f}, weight={w}")
        return w

    _log("DEBUG", "[INFO] 조건 충족 모델 없음 → weight=0.2")
    return 0.2

# ===== 공개 로더(API) =====
def _attach_bin_info(model_obj: torch.nn.Module, meta: Dict[str, Any]) -> None:
    """meta 정보를 모델 객체 속성에 주입 (predict.py 호환을 위해 class_ranges 포함)."""
    try:
        # bin 정보
        setattr(model_obj, "bin_edges", meta.get("bin_edges", []))
        setattr(model_obj, "bin_counts", meta.get("bin_counts", []))
        setattr(model_obj, "bin_spans", meta.get("bin_spans", []))
        setattr(model_obj, "bin_cfg", meta.get("bin_cfg", {}))
        # 클래스 구간/개수
        setattr(model_obj, "class_ranges", meta.get("class_ranges", []))
        setattr(model_obj, "num_classes", meta.get("num_classes", None))
        # 원본 메타 전체 접근도 제공
        setattr(model_obj, "meta", dict(meta))
    except Exception:
        pass

def load_model_cached(pt_path: str, model_obj: torch.nn.Module, ttl_sec: int = _TTL) -> Optional[torch.nn.Module]:
    """정책 로더 + LRU+TTL 캐시. 항상 CPU 텐서로 저장해 VRAM 누수 차단.
    추가: meta.bin_edges 자동 보정 + class_ranges 정규화 후 모델 객체에 주입.
    반환: model_obj (predict.py의 기존 시그니처와 100% 호환)
    """
    global _cache_bytes
    if not pt_path or not os.path.exists(pt_path):
        _log("ERROR", f"[❌ load_model_cached] 파일 없음: {pt_path}")
        return None

    ok, why, meta_pf = _preflight(pt_path, model_type=_infer_model_type_from_fname(pt_path), input_size=None)
    if not ok:
        # 운영에서는 한 줄만
        if why == "bin_edges_missing":
            _log("WARN", f"[❌ load_model_cached] bin_edges 누락 → 예측 비활성: {pt_path}")
        else:
            _log("WARN", f"[❌ load_model_cached 사전점검 실패] {pt_path} → {why}")
        return None
    # ok / ok_autofixed 모두 허용
    meta_final = meta_pf if meta_pf else _load_meta_dict(pt_path)

    with _cache_lock:
        ent = _model_cache.get(pt_path)
        if ent and (_now() - ent["ts"]) <= ent["ttl"]:
            _model_cache.move_to_end(pt_path)  # LRU 갱신
            try:
                model_obj.load_state_dict(ent["state"], strict=False)
                _attach_bin_info(model_obj, ent.get("meta", {}))
                model_obj.eval()
                return model_obj
            except Exception as e:
                _model_cache.pop(pt_path, None)
                _cache_bytes -= ent.get("bytes", 0)
                _log("WARN", f"[⚠️ 캐시 로드 실패, 디스크 재시도] {pt_path} → {e}")

    try:
        loaded = _load_with_policy(pt_path, model=model_obj, map_location="cpu", strict=False)
        state = loaded.state_dict() if hasattr(loaded, "state_dict") else None
        if not state:
            _log("ERROR", f"[❌ 로더 반환 state_dict 없음] {pt_path}")
            return None
        state = _ensure_cpu_state(state)
        size_b = _obj_bytes(state)

        # STRICT 모드일 경우 bin_edges 재확인
        be = meta_final.get("bin_edges", [])
        if _REQUIRE_BIN_EDGES_STRICT and not (isinstance(be, list) and len(be) >= 2):
            _log("WARN", f"[❌ meta.bin_edges 누락(STRICT) → 예측 비활성] {pt_path}")
            return None

        with _cache_lock:
            _evict_if_needed(extra_bytes=size_b)
            _model_cache[pt_path] = {
                "state": dict(state),
                "meta": dict(meta_final),
                "ts": _now(),
                "ttl": int(ttl_sec),
                "bytes": int(size_b),
            }
            _model_cache.move_to_end(pt_path)
            _cache_bytes += int(size_b)

        model_obj.load_state_dict(state, strict=False)
        _attach_bin_info(model_obj, meta_final)
        model_obj.eval()
        _log("INFO", f"[✅ 모델 로드 완료] {pt_path}")
        return model_obj
    except ModelLoadError as e:
        _log("WARN", f"[❌ ModelLoadError] {pt_path} → {e.reason}")
    except Exception as e:
        _log("WARN", f"[❌ load_model_cached 오류] {pt_path} → {e}")
    return None

# ===== 기타 유틸 =====
def model_exists(symbol: str, strategy: str) -> bool:
    try:
        for _, _, files in os.walk(MODEL_DIR):
            for file in files:
                if file.startswith(f"{symbol}_{strategy}_") and file.lower().endswith(SUPPORTED_WEIGHTS):
                    return True
    except Exception as e:
        _log("WARN", f"[오류] 모델 존재 확인 실패: {e}")
    return False

def count_models_per_strategy() -> Dict[str, int]:
    counts = {"단기": 0, "중기": 0, "장기": 0}
    try:
        for _, _, files in os.walk(MODEL_DIR):
            for file in files:
                if not file.lower().endswith(SUPPORTED_WEIGHTS):
                    continue
                parts = file.split("_")
                if len(parts) >= 3:
                    strat = parts[1]
                    if strat in counts:
                        counts[strat] += 1
    except Exception as e:
        _log("WARN", f"[오류] 모델 수 계산 실패: {e}")
    return counts

def get_similar_symbol(symbol: str) -> list:
    return []
