# model_weight_loader.py (YOPO v1.4 fixed)
import os
import json
import glob
import time
import torch
import pandas as pd

MODEL_DIR = "/persistent/models"
EVAL_RESULT_SINGLE = "/persistent/evaluation_result.csv"  # 있을 수도 있어서 추가 확인용

# ============== 캐시 ==============
_model_cache = {}
_model_cache_ttl = {}

def _safe_state_dict(obj):
    """
    다양한 저장 포맷(torch.save(state), torch.save(model), ckpt dict 등)을
    최대한 state_dict으로 정규화.
    """
    try:
        # state_dict 형태
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            return obj
        # 모듈 객체
        try:
            return obj.state_dict()
        except Exception:
            pass
        # ckpt dict 안에 state_dict 키
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    except Exception:
        return obj

def load_model_cached(pt_path, model_obj, ttl_sec=600):
    """
    프로젝트 표준: 주어진 model_obj에 pt_path의 state_dict를 로드하여 반환
    - predict.py가 기대하는 시그니처: (pt_path, model_obj, ttl_sec)
    - 내부 캐시는 state_dict 기준 (메모리 절약 + 호환성 ↑)
    """
    now = time.time()

    # TTL 만료 캐시 정리
    expired = [k for k, t in _model_cache_ttl.items() if now - t >= ttl_sec]
    for k in expired:
        _model_cache.pop(k, None)
        _model_cache_ttl.pop(k, None)

    if not pt_path or not os.path.exists(pt_path):
        print(f"[❌ load_model_cached] 파일 없음: {pt_path}")
        return None

    try:
        # 캐시 히트
        if pt_path in _model_cache:
            try:
                model_obj.load_state_dict(_model_cache[pt_path], strict=False)
                model_obj.eval()
                return model_obj
            except Exception as e:
                print(f"[⚠️ 캐시 로드 실패, 디스크 재시도] {pt_path} → {e}")
                _model_cache.pop(pt_path, None)
                _model_cache_ttl.pop(pt_path, None)

        # 디스크에서 state_dict 로드
        state = torch.load(pt_path, map_location="cpu")
        state_dict = _safe_state_dict(state)
        model_obj.load_state_dict(state_dict, strict=False)
        model_obj.eval()

        # 캐시에 저장
        _model_cache[pt_path] = state_dict
        _model_cache_ttl[pt_path] = now

        print(f"[✅ 모델 state_dict 로드 완료] {pt_path}")
        return model_obj

    except Exception as e:
        print(f"[❌ load_model_cached 오류] {pt_path} → {e}")
        return None

# ============== META 유연 탐색 ==============
def _stem(path: str) -> str:
    base = os.path.basename(path)
    return base.rsplit(".", 1)[0]

def resolve_meta_for_pt(pt_path: str):
    """
    PT/PTZ 파일에 대응하는 META 파일을 유연 규칙으로 탐색.
    우선순위:
      1) 동일 디렉터리에서 {base}.meta.json
      2) 동일 디렉터리에서 {base}*.meta.json (mtime 최신)
      3) MODEL_DIR 루트에서 {base}.meta.json
      4) MODEL_DIR 루트에서 {base}*.meta.json (mtime 최신)
      5) 심볼/전략 추정 시 {MODEL_DIR}/{SYMBOL}/{STRATEGY}/(lstm|cnn_lstm|transformer).meta.json
    """
    try:
        if not pt_path:
            return None
        base = os.path.basename(pt_path)
        if not (base.endswith(".pt") or base.endswith(".ptz")):
            return None

        dirn = os.path.dirname(pt_path) or MODEL_DIR
        stem = _stem(base)

        # 1) 동일 디렉터리 완전 일치
        exact_local = os.path.join(dirn, f"{stem}.meta.json")
        if os.path.exists(exact_local):
            return exact_local

        # 2) 동일 디렉터리 prefix 후보
        cand_local = sorted(
            glob.glob(os.path.join(dirn, f"{stem}*.meta.json")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if cand_local:
            return cand_local[0]

        # 3) 루트 완전 일치
        exact_root = os.path.join(MODEL_DIR, f"{stem}.meta.json")
        if os.path.exists(exact_root):
            return exact_root

        # 4) 루트 prefix 후보
        cand_root = sorted(
            glob.glob(os.path.join(MODEL_DIR, f"{stem}*.meta.json")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if cand_root:
            return cand_root[0]

        # 5) {SYMBOL}_{STRATEGY}_{TYPE} 패턴 역추정
        parts = stem.split("_")
        if len(parts) >= 3:
            sym, strat, mtype = parts[0], parts[1], parts[2]
            guess = os.path.join(MODEL_DIR, sym, strat, f"{mtype}.meta.json")
            if os.path.exists(guess):
                return guess
    except Exception as e:
        print(f"[⚠️ resolve_meta_for_pt 실패] {pt_path} → {e}")
    return None

# ============== 모델 리스트 찾기 (리커시브) ==============
def find_models_fuzzy(symbol: str, strategy: str):
    """
    심볼/전략 기준으로 PT/PTZ를 먼저 찾고, META를 유연 규칙으로 붙여 반환.
    - MODEL_DIR 하위 전체(서브폴더 포함) 탐색
    """
    out = []
    try:
        prefix = f"{symbol}_"
        needle = f"_{strategy}_"
        for root, _, files in os.walk(MODEL_DIR):
            for fn in files:
                if not (fn.endswith(".pt") or fn.endswith(".ptz")):
                    continue
                if not fn.startswith(prefix):
                    continue
                if needle not in fn:
                    continue
                pt_path = os.path.join(root, fn)
                meta_path = resolve_meta_for_pt(pt_path)
                if not meta_path:
                    continue
                out.append({
                    "pt_file": os.path.relpath(pt_path, MODEL_DIR),
                    "meta_file": os.path.relpath(meta_path, MODEL_DIR)
                })
        out.sort(key=lambda x: x["pt_file"])
    except Exception as e:
        print(f"[⚠️ find_models_fuzzy 실패] {symbol}-{strategy} → {e}")
    return out

# ============== 평가 로그 수집 ==============
def _load_all_eval_logs():
    frames = []
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
            df = df.drop_duplicates(subset=[c for c in df.columns if c in ["timestamp","symbol","strategy","model"]])
        else:
            df = df.drop_duplicates()
        return df
    except Exception as e:
        print(f"[⚠️ 평가 로그 병합 실패] → {e}")
        return pd.DataFrame()

# ============== 가중치 추정 ==============
def _symbol_from_meta_path(meta_path: str):
    """
    meta에 symbol 키가 없을 때 파일명에서 추정
    """
    try:
        base = os.path.basename(meta_path).replace(".meta.json","")
        parts = base.split("_")
        if len(parts) >= 1:
            return parts[0]
    except Exception:
        pass
    return None

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=3, input_size=None):
    """
    메타파일 및 최근 평가 로그 기반 가중치(0.0~1.0) 추정
    """
    # 메타 파일 후보
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
        if input_size is not None and meta.get("input_size") not in [None, input_size]:
            print(f"[⚠️ input_size 불일치] meta={meta.get('input_size')} vs input={input_size} → weight=0.2")
            return 0.2

        # PT/PTZ 존재 확인
        pt_path = meta_path.replace(".meta.json", ".pt")
        if not os.path.exists(pt_path):
            ptz_path = meta_path.replace(".meta.json", ".ptz")
            if os.path.exists(ptz_path):
                pt_path = ptz_path
        if not os.path.exists(pt_path):
            print(f"[⚠️ 모델 파일 없음] {pt_path} → weight=0.2")
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
def model_exists(symbol, strategy):
    try:
        for root, _, files in os.walk(MODEL_DIR):
            for file in files:
                if file.startswith(f"{symbol}_{strategy}_") and (file.endswith(".pt") or file.endswith(".ptz")):
                    return True
    except Exception as e:
        print(f"[오류] 모델 존재 확인 실패: {e}")
    return False

def count_models_per_strategy():
    counts = {"단기": 0, "중기": 0, "장기": 0}
    try:
        for root, _, files in os.walk(MODEL_DIR):
            for file in files:
                if not (file.endswith(".pt") or file.endswith(".ptz")):
                    continue
                parts = file.split("_")
                if len(parts) >= 3:
                    strat = parts[1]
                    if strat in counts:
                        counts[strat] += 1
    except Exception as e:
        print(f"[오류] 모델 수 계산 실패: {e}")
    return counts

def get_similar_symbol(symbol: str):
    return []
