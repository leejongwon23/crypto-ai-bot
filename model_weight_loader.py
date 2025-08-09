import os
import pandas as pd
import json
import glob
import torch
import time

MODEL_DIR = "/persistent/models"
EVAL_RESULT_SINGLE = "/persistent/evaluation_result.csv"  # 있을 수도 있어서 추가 확인용

# ✅ 모델 캐시 (state_dict TTL 캐싱)
#  - 키: pt 경로, 값: state_dict
_model_cache = {}
_model_cache_ttl = {}

def load_model_cached(pt_path, model_obj, ttl_sec=600):
    """
    ✅ (프로젝트 표준) 주어진 model_obj에 pt_path의 state_dict를 로드하여 반환
    - predict.py가 기대하는 시그니처: (pt_path, model_obj, ttl_sec)
    - 내부 캐시는 state_dict 기준 (메모리 절약 + 호환성 ↑)
    """
    now = time.time()

    # TTL 만료 캐시 정리
    expired = [k for k, t in _model_cache_ttl.items() if now - t >= ttl_sec]
    for k in expired:
        _model_cache.pop(k, None)
        _model_cache_ttl.pop(k, None)

    if not os.path.exists(pt_path):
        print(f"[❌ load_model_cached] 파일 없음: {pt_path}")
        return None

    try:
        # 캐시 히트 → 바로 로드
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
        # (torch 1.x/2.x 겸용: state_dict 저장된 파일/모델 전체 저장 모두 허용)
        state = torch.load(pt_path, map_location="cpu")
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            state_dict = state
        else:
            try:
                state_dict = state.state_dict()
            except Exception:
                state_dict = state  # 마지막 안전책

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


def _load_all_eval_logs():
    """
    ✅ 평가 로그를 가능한 폭넓게 수집해서 하나의 DataFrame으로 반환
    - /persistent/logs/evaluation_*.csv (일자별)
    - /persistent/evaluation_result.csv (있으면)
    """
    frames = []

    # 1) 일자별 평가 로그
    eval_daily = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
    for fp in eval_daily:
        try:
            frames.append(pd.read_csv(fp, encoding="utf-8-sig"))
        except Exception as e:
            print(f"[⚠️ 평가 파일 읽기 실패] {fp} → {e}")

    # 2) 단일 평가 로그(있을 수 있음)
    if os.path.exists(EVAL_RESULT_SINGLE):
        try:
            frames.append(pd.read_csv(EVAL_RESULT_SINGLE, encoding="utf-8-sig"))
        except Exception as e:
            print(f"[⚠️ 단일 평가 파일 읽기 실패] {EVAL_RESULT_SINGLE} → {e}")

    if not frames:
        return pd.DataFrame()

    try:
        df = pd.concat(frames, ignore_index=True)
        # 중복 제거 (있을 경우 대비)
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=[c for c in df.columns if c in ["timestamp","symbol","strategy","model"]])
        else:
            df = df.drop_duplicates()
        return df
    except Exception as e:
        print(f"[⚠️ 평가 로그 병합 실패] → {e}")
        return pd.DataFrame()


def get_model_weight(model_type, strategy, symbol="ALL", min_samples=3, input_size=None):
    """
    ✅ 메타파일 및 최근 평가 로그 기반 가중치(0.0~1.0) 추정
    - 메타/평가로그가 부족하면 0.2로 보수적 fallback
    - 일자별 로그와 단일 로그를 모두 탐색하여 샘플 부족 문제 완화
    """
    pattern = (
        os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json")
        if symbol != "ALL"
        else os.path.join(MODEL_DIR, f"*_{strategy}_{model_type}.meta.json")
    )
    meta_files = glob.glob(pattern)

    if not meta_files:
        print(f"[⚠️ meta 파일 없음] {pattern} → fallback weight=0.2 (모델 없음)")
        return 0.2

    # 평가 로그 로드
    df_all = _load_all_eval_logs()
    if df_all.empty:
        print("[INFO] 평가 데이터 없음 → cold-start weight=0.2")
        return 0.2

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if input_size is not None and meta.get("input_size") != input_size:
                print(f"[⚠️ input_size 불일치] meta={meta.get('input_size')} vs input={input_size} → weight=0.2")
                return 0.2

            pt_path = meta_path.replace(".meta.json", ".pt")
            if not os.path.exists(pt_path):
                print(f"[⚠️ 모델 파일 없음] {pt_path} → weight=0.2")
                return 0.2

            # 평가 레코드 필터
            df = df_all.copy()
            need_cols = {"model", "strategy", "symbol", "status"}
            if not need_cols.issubset(df.columns):
                print("[INFO] 평가 로그 컬럼 부족 → cold-start weight=0.2")
                return 0.2

            df = df[
                (df["model"] == model_type)
                & (df["strategy"] == strategy)
                & (df["symbol"] == meta.get("symbol"))
                & (df["status"].isin(["success", "fail", "v_success", "v_fail"]))
            ]

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

        except Exception as e:
            print(f"[get_model_weight 예외] {e}")
            continue

    print("[INFO] 조건 충족 모델 없음 → weight=0.2")
    return 0.2


def model_exists(symbol, strategy):
    try:
        for file in os.listdir(MODEL_DIR):
            if file.startswith(f"{symbol}_{strategy}_") and file.endswith(".pt"):
                return True
    except Exception as e:
        print(f"[오류] 모델 존재 확인 실패: {e}")
    return False


def count_models_per_strategy():
    counts = {"단기": 0, "중기": 0, "장기": 0}
    try:
        for file in os.listdir(MODEL_DIR):
            if not file.endswith(".pt"):
                continue
            parts = file.split("_")
            if len(parts) >= 3:
                strategy = parts[1]
                if strategy in counts:
                    counts[strategy] += 1
    except Exception as e:
        print(f"[오류] 모델 수 계산 실패: {e}")
    return counts


# ✅ logger.get_available_models()에서 import하는 유틸 (호환용)
def get_similar_symbol(symbol: str):
    """
    간단 호환용 스텁: 타 심볼을 허용하지 않음 (필요 시 확장)
    """
    return []

# ⚠️ 주의:
#  - 과거 이 파일에 있었던 `get_class_return_range`는 제거했습니다.
#    클래스 경계/수익률 관련 로직은 반드시 `config.py`의
#    `get_class_return_range(class_id, symbol, strategy)`를 사용하세요.
