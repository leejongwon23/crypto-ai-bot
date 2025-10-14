# YOPO v1.5 — robust weight+meta loader
import os
import json
import glob
import time
from typing import Optional, Dict, Any, Tuple, List

import torch
import pandas as pd

# 프로젝트 표준 I/O 사용 (PT / PTZ / safetensors 지원, prefix 자동보정)
from model_io import load_model as _load_with_policy, ModelLoadError, load_meta as _load_meta

MODEL_DIR = "/persistent/models"
EVAL_RESULT_SINGLE = "/persistent/evaluation_result.csv"  # 있을 수도 있어서 추가 확인용

SUPPORTED_WEIGHTS = (".pt", ".ptz", ".safetensors")

# ============== 캐시 ==============
_model_cache: Dict[str, Dict[str, torch.Tensor]] = {}
_model_cache_ttl: Dict[str, float] = {}

def _is_weight_file(path: str) -> bool:
    return path and os.path.splitext(path)[1].lower() in SUPPORTED_WEIGHTS and os.path.isfile(path)

def _now() -> float:
    return time.time()

def _stem(path: str) -> str:
    base = os.path.basename(path)
    return base.rsplit(".", 1)[0]

def _meta_path_for(weight_path: str) -> str:
    return os.path.splitext(os.path.abspath(weight_path))[0] + ".meta.json"

def _find_weight_candidates(symbol: str, strategy: str, model_type: str) -> List[str]:
    """
    {SYMBOL}_{STRATEGY}_{TYPE}.* (pt/ptz/safetensors) 을 전 디렉터리에서 탐색
    가장 최근 mtime 순으로 정렬하여 반환
    """
    patt = os.path.join(MODEL_DIR, "**", f"{symbol}_{strategy}_{model_type}*")
    found: List[str] = []
    for p in glob.iglob(patt, recursive=True):
        if _is_weight_file(p):
            found.append(p)
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found

def _resolve_meta(weight_path: str) -> Optional[str]:
    """
    가중치 옆의 {stem}.meta.json 우선, 없으면 동일 디렉·루트에서 prefix 매칭 최신본
    """
    try:
        exact = _meta_path_for(weight_path)
        if os.path.exists(exact):
            return exact
        dirn = os.path.dirname(weight_path) or MODEL_DIR
        stem = _stem(weight_path)
        cands = sorted(
            glob.glob(os.path.join(dirn, f"{stem}*.meta.json")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if cands:
            return cands[0]
        # 루트에서도 fallback
        root_cands = sorted(
            glob.glob(os.path.join(MODEL_DIR, f"{os.path.basename(stem)}*.meta.json")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if root_cands:
            return root_cands[0]
    except Exception as e:
        print(f"[⚠️ META 탐색 실패] {weight_path} → {e}")
    return None

def _preflight(weight_path: str, model_type: str, input_size: Optional[int]) -> Tuple[bool, str]:
    """
    사전 점검: 파일 존재, 메타 일관성(input_size 등) 확인
    """
    if not _is_weight_file(weight_path):
        return False, "weight_missing"
    meta_path = _resolve_meta(weight_path)
    if not meta_path or not os.path.isfile(meta_path):
        return False, "meta_missing"

    try:
        meta = _load_meta(weight_path, default={}) if os.path.samefile(_meta_path_for(weight_path), meta_path) else json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        meta = {}

    if input_size is not None:
        mi = meta.get("input_size")
        if mi not in (None, input_size):
            return False, f"input_size_mismatch(meta={mi}, in={input_size})"

    mt = str(meta.get("model_type") or meta.get("type") or "").lower()
    if mt and mt != str(model_type).lower():
        # 경고만 — 다른 타입 메타가 붙었어도 로딩은 시도
        print(f"[⚠️ 경고] meta.model_type={mt} ≠ req={model_type}")

    return True, "ok"

def _drop_expired(ttl_sec: int):
    now = _now()
    expired = [k for k, t in _model_cache_ttl.items() if now - t >= ttl_sec]
    for k in expired:
        _model_cache.pop(k, None)
        _model_cache_ttl.pop(k, None)

def load_model_cached(pt_path: str, model_obj: torch.nn.Module, ttl_sec: int = 600) -> Optional[torch.nn.Module]:
    """
    표준 로더: 정책 준수(model_io.load_model) + module. prefix 자동보정 + 캐시
    predict.py가 기대하는 시그니처 유지
    """
    _drop_expired(ttl_sec)

    if not pt_path or not os.path.exists(pt_path):
        print(f"[❌ load_model_cached] 파일 없음: {pt_path}")
        return None

    ok, why = _preflight(pt_path, model_type=_infer_model_type_from_fname(pt_path), input_size=None)
    if not ok:
        print(f"[❌ load_model_cached 사전점검 실패] {pt_path} → {why}")
        return None

    try:
        # 캐시 hit
        if pt_path in _model_cache:
            try:
                model_obj.load_state_dict(_model_cache[pt_path], strict=False)
                model_obj.eval()
                return model_obj
            except Exception as e:
                print(f"[⚠️ 캐시 로드 실패, 디스크 재시도] {pt_path} → {e}")
                _model_cache.pop(pt_path, None)
                _model_cache_ttl.pop(pt_path, None)

        # 디스크에서 정책 준수 로드 (PT/PTZ/safetensors & prefix 보정 포함)
        loaded = _load_with_policy(pt_path, model=model_obj, map_location="cpu", strict=False)
        # _load_with_policy가 nn.Module을 반환 — state_dict를 캐시에 저장
        state_dict = loaded.state_dict() if hasattr(loaded, "state_dict") else None
        if state_dict:
            _model_cache[pt_path] = dict(state_dict)
            _model_cache_ttl[pt_path] = _now()
        model_obj.eval()
        print(f"[✅ 모델 로드 완료] {pt_path}")
        return model_obj

    except ModelLoadError as e:
        print(f"[❌ ModelLoadError] {pt_path} → {e.reason} :: {e.detail}")
        return None
    except Exception as e:
        print(f"[❌ load_model_cached 오류] {pt_path} → {e}")
        return None

# ============== 가중치 후보/리스트 ==============
def find_models_fuzzy(symbol: str, strategy: str) -> List[Dict[str, str]]:
    """
    YOPO 탐색 표준: 심볼/전략 기준으로 가중치 파일을 찾고, 유효 META를 붙여 반환
    """
    out: List[Dict[str, str]] = []
    try:
        for root, _, files in os.walk(MODEL_DIR):
            for fn in files:
                if not fn.endswith(SUPPORTED_WEIGHTS):
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
        print(f"[⚠️ find_models_fuzzy 실패] {symbol}-{strategy} → {e}")
    return out

def locate_best_weight(symbol: str, strategy: str, model_type: str) -> Optional[str]:
    """
    가장 최근 mtime 가중치 후보 중 메타 일치/사전점검 통과한 첫 번째 경로 반환
    """
    for w in _find_weight_candidates(symbol, strategy, model_type):
        ok, _ = _preflight(w, model_type, input_size=None)
        if ok:
            return w
    return None

# ============== 평가 로그 수집 ==============
def _load_all_eval_logs() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    eval_daily = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
    for fp in eval_daily:
        try:
            frames.append(pd.read_csv(fp, encoding="utf-8-sig"))
        except Exception as e:
            print(f"[⚠️ 평가 파일 읽기 실패] {fp} → {e}")
    if os.path.exists(EVAL_RESULT_SINGLE):
        try:
            frames.append(pd.read_csv(EVAL_RESULT_SINGLE, encoding="utf-8-sig"))
        except Exception as e:
            print(f"[⚠️ 단일 평가 파일 읽기 실패] {EVAL_RESULT_SINGLE} → {e}")
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
        print(f"[⚠️ 평가 로그 병합 실패] → {e}")
        return pd.DataFrame()

# ============== 가중치 추정(메타+로그) ==============
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
    base = os.path.basename(path)
    core = _stem(base)
    parts = core.split("_")
    return parts[2] if len(parts) >= 3 else ""

def get_model_weight(model_type: str, strategy: str, symbol: str = "ALL", min_samples: int = 3, input_size: Optional[int] = None) -> float:
    """
    메타파일 및 최근 평가 로그 기반 가중치(0.0~1.0) 추정
    """
    if symbol != "ALL":
        pattern = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json")
        meta_files = glob.glob(pattern) or glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}*.meta.json"))
    else:
        meta_files = glob.glob(os.path.join(MODEL_DIR, f"*_{strategy}_{model_type}*.meta.json"))

    if not meta_files:
        print(f"[⚠️ meta 파일 없음] symbol={symbol}, strategy={strategy}, type={model_type} → weight=0.2")
        return 0.2

    df_all = _load_all_eval_logs()
    if df_all.empty:
        print("[INFO] 평가 데이터 없음 → cold-start weight=0.2")
        return 0.2

    need_cols = {"model", "strategy", "symbol", "status"}
    if not need_cols.issubset(df_all.columns):
        print("[INFO] 평가 로그 컬럼 부족 → cold-start weight=0.2")
        return 0.2

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        meta_symbol = meta.get("symbol") or _symbol_from_meta_path(meta_path)
        if input_size is not None and meta.get("input_size") not in (None, input_size):
            print(f"[⚠️ input_size 불일치] meta={meta.get('input_size')} vs input={input_size} → weight=0.2")
            return 0.2

        # 가중치 파일 존재 확인(모든 포맷)
        base = meta_path.replace(".meta.json", "")
        candidates = [base + ext for ext in [".pt", ".ptz", ".safetensors"]]
        if not any(os.path.exists(c) for c in candidates):
            print(f"[⚠️ 모델 파일 없음] {base}.[pt|ptz|safetensors] → weight=0.2")
            return 0.2

        df = df_all[
            (df_all["model"] == model_type)
            & (df_all["strategy"] == strategy)
            & (df_all["symbol"] == meta_symbol)
            & (df_all["status"].isin(["success", "fail", "v_success", "v_fail"]))
        ].copy()

        sample_len = len(df)
        if sample_len < min_samples:
            print(f"[INFO] 평가 샘플 부족 (len={sample_len} < {min_samples}) → weight=0.2")
            return 0.2

        success_rate = (df["status"].isin(["success", "v_success"])).mean()
        if success_rate >= 0.7:
            print(f"[INFO] success_rate={success_rate:.4f} → weight=1.0")
            return 1.0
        elif success_rate < 0.3:
            print(f"[INFO] success_rate={success_rate:.4f} → weight=0.0")
            return 0.0
        else:
            w = max(0.0, round((success_rate - 0.3) / 0.4, 4))
            print(f"[INFO] success_rate={success_rate:.4f}, weight={w}")
            return w

    print("[INFO] 조건 충족 모델 없음 → weight=0.2")
    return 0.2

# ============== 기타 유틸 ==============
def model_exists(symbol: str, strategy: str) -> bool:
    try:
        for _, _, files in os.walk(MODEL_DIR):
            for file in files:
                if file.startswith(f"{symbol}_{strategy}_") and file.endswith(SUPPORTED_WEIGHTS):
                    return True
    except Exception as e:
        print(f"[오류] 모델 존재 확인 실패: {e}")
    return False

def count_models_per_strategy() -> Dict[str, int]:
    counts = {"단기": 0, "중기": 0, "장기": 0}
    try:
        for _, _, files in os.walk(MODEL_DIR):
            for file in files:
                if not file.endswith(SUPPORTED_WEIGHTS):
                    continue
                parts = file.split("_")
                if len(parts) >= 3:
                    strat = parts[1]
                    if strat in counts:
                        counts[strat] += 1
    except Exception as e:
        print(f"[오류] 모델 수 계산 실패: {e}")
    return counts

def get_similar_symbol(symbol: str) -> list:
    return []
