# evo_meta_dataset.py  — v5 (step 5 완료)
# - 안전한 CSV 로드(on_bad_lines='skip', 인코딩 폴백)
# - 필수/선택 컬럼 점검 및 NaN 제거
# - 리스트 필드 안전 파싱( ast.literal_eval → json.loads )
# - 피처 구성: [softmax | expected_return | actual_return | empirical_prior? | knn_vote_dist?]
# - 길이 불일치 시 0 패딩/트림
# - 반환: X(float32), y(int64: 전략 인덱스), num_strategies(int)

import os
import json
import ast
import pandas as pd
import numpy as np

# 선택 컬럼(있으면 피처에 포함)
OPTIONAL_LIST_COLS = ["empirical_prior", "knn_vote_dist", "knn_vote", "prior"]

def _safe_parse_list_field(v):
    """문자열 리스트를 안전하게 파싱. 실패/결측은 None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        # 1) literal_eval
        try:
            return list(ast.literal_eval(s))
        except Exception:
            # 2) json
            try:
                return list(json.loads(s))
            except Exception:
                return None
    return None

def _pad_or_trim(arr, target_len):
    """길이를 target_len으로 맞춤(부족→0.0 패딩 / 초과→자르기)."""
    arr = list(arr) if arr is not None else []
    if len(arr) < target_len:
        return arr + [0.0] * (target_len - len(arr))
    if len(arr) > target_len:
        return arr[:target_len]
    return arr

def _pick_optional_lists(row):
    """OPTIONAL_LIST_COLS 중 존재하는 것만 (이름, 리스트) 튜플로 반환"""
    out = []
    for c in OPTIONAL_LIST_COLS:
        if c in row and row[c] is not None:
            parsed = _safe_parse_list_field(row[c])
            if isinstance(parsed, list):
                out.append((c, parsed))
    return out

def prepare_evo_meta_dataset(csv_path="/persistent/wrong_predictions.csv"):
    """
    반환: (X, y, num_strategies)
      - X: float32, shape (N, D)  — D는 사용 가능한 리스트 컬럼에 따라 가변
      - y: int64,  shape (N,)      — 'strategy'를 인덱싱한 라벨(메타가 전략 선택 학습)
      - num_strategies: int        — 전략 개수(메타러너 출력 크기 결정 시 사용)
    """
    if not os.path.exists(csv_path):
        print(f"❌ [evo_meta_dataset] 파일 없음: {csv_path}")
        return None, None, None

    # 1) CSV 로드(인코딩/깨진 줄 대비)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(csv_path, on_bad_lines="skip")
        except Exception as e:
            print(f"❌ [evo_meta_dataset] CSV 읽기 실패: {e}")
            return None, None, None

    if df is None or df.empty:
        print("❌ [evo_meta_dataset] 데이터가 비어 있음")
        return None, None, None

    # 2) 필수 컬럼 점검
    required = ["strategy", "softmax", "expected_return", "actual_return"]
    for col in required:
        if col not in df.columns:
            print(f"❌ [evo_meta_dataset] 필수 컬럼 누락: {col}")
            return None, None, None

    # 3) 기본 결측 제거
    df = df.dropna(subset=["strategy", "softmax"])
    if df.empty:
        print("❌ [evo_meta_dataset] NaN 제거 후 데이터 없음")
        return None, None, None

    # 4) 전략 → 인덱스
    try:
        strategies = sorted(df["strategy"].dropna().unique())
        strategy_map = {s: i for i, s in enumerate(strategies)}
        num_strategies = len(strategy_map)
        if num_strategies <= 0:
            print("❌ [evo_meta_dataset] 전략 수가 0")
            return None, None, None
    except Exception as e:
        print(f"❌ [evo_meta_dataset] 전략 매핑 실패: {e}")
        return None, None, None

    X_list, y_list = [], []

    # 5) 행 단위 파싱 → 피처 벡터 구성
    for _, row in df.iterrows():
        try:
            sm = _safe_parse_list_field(row.get("softmax"))
            er = _safe_parse_list_field(row.get("expected_return"))
            ar = _safe_parse_list_field(row.get("actual_return"))
            if not (isinstance(sm, list) and isinstance(er, list) and isinstance(ar, list)):
                continue

            # 기준 길이 = softmax 길이(클래스 개수로 간주)
            C = int(len(sm))
            if C <= 0:
                continue

            # 선택 피처 수집
            opt_feats = _pick_optional_lists(row)

            # 각 리스트를 동일 길이(C)로 정렬
            sm = _pad_or_trim(sm, C)
            er = _pad_or_trim(er, C)
            ar = _pad_or_trim(ar, C)
            aligned_optional = [(name, _pad_or_trim(vals, C)) for name, vals in opt_feats]

            # 최종 피처: [sm | er | ar | (empirical_prior?) | (knn_vote_dist/knn_vote/prior?)]
            feats = []
            feats.extend(sm)
            feats.extend(er)
            feats.extend(ar)
            for _, vals in aligned_optional:
                feats.extend(vals)

            # 라벨 = 전략 인덱스
            strat = row.get("strategy")
            if strat not in strategy_map:
                continue

            X_list.append([float(x) if x is not None else 0.0 for x in feats])
            y_list.append(int(strategy_map[strat]))
        except Exception:
            # 행 단위 에러는 건너뜀
            continue

    if not X_list or not y_list:
        print("❌ [evo_meta_dataset] 유효 샘플 없음")
        return None, None, None

    try:
        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int64)
    except Exception as e:
        print(f"❌ [evo_meta_dataset] ndarray 변환 실패: {e}")
        return None, None, None

    print(f"[✅ prepare_evo_meta_dataset] X: {X.shape}, y: {y.shape}, num_strategies: {num_strategies}")
    return X, y, num_strategies
