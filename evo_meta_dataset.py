# evo_meta_dataset.py (PATCHED: 안전한 파싱 + 입력 검증 + CSV 로드 견고화)
import os
import json
import ast
import pandas as pd
import numpy as np

def _safe_parse_list_field(v):
    """
    안전하게 리스트/튜플/ndarray 형태로 파싱.
    문자열이면 ast.literal_eval 우선, 실패 시 json.loads 시도.
    None/NaN이면 None 반환.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    try:
        # pandas NA handling
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            return list(ast.literal_eval(s))
        except Exception:
            try:
                return list(json.loads(s))
            except Exception:
                return None
    # 숫자 등 비리스트 타입이면 None
    return None

def prepare_evo_meta_dataset(csv_path="/persistent/wrong_predictions.csv"):
    """
    반환: (X, y, num_strategies) 또는 (None, None, None) on failure.
    X: numpy float32, shape (N, num_strategies*3)
    y: numpy int64, shape (N,)
    """
    if not os.path.exists(csv_path):
        print(f"❌ [evo_meta_dataset] 파일 없음: {csv_path}")
        return None, None, None

    # 안전히 읽기
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(csv_path, on_bad_lines="skip")
        except Exception as e:
            print(f"❌ [evo_meta_dataset] CSV 읽기 실패: {e}")
            return None, None, None

    if df.empty:
        print("❌ [evo_meta_dataset] 실패 예측 데이터가 비어 있음")
        return None, None, None

    # 필수 컬럼 체크
    required_cols = ["strategy", "softmax", "expected_return", "actual_return", "predicted_class", "true_class"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ [evo_meta_dataset] 필수 컬럼 누락: {col}")
            return None, None, None

    # NaN 제거 (필수 컬럼 기준)
    df = df.dropna(subset=["strategy", "softmax"])
    if df.empty:
        print("❌ [evo_meta_dataset] NaN 제거 후 데이터 없음")
        return None, None, None

    # 전략 → 인덱스 매핑
    try:
        strategies = sorted(df["strategy"].dropna().unique())
        strategy_map = {s: i for i, s in enumerate(strategies)}
        num_strategies = len(strategy_map)
        if num_strategies <= 0:
            print("❌ [evo_meta_dataset] 전략 수가 0임")
            return None, None, None
    except Exception as e:
        print(f"❌ [evo_meta_dataset] 전략 매핑 실패: {e}")
        return None, None, None

    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            sm = _safe_parse_list_field(row.get("softmax"))
            er = _safe_parse_list_field(row.get("expected_return"))
            ar = _safe_parse_list_field(row.get("actual_return"))

            # 모두 리스트 형태여야 함
            if not (isinstance(sm, list) and isinstance(er, list) and isinstance(ar, list)):
                continue

            # 합친 feature 벡터
            features = list(sm) + list(er) + list(ar)

            # 길이 검사: 전략 개수 * 3 이어야 함
            expected_len = num_strategies * 3
            if len(features) < expected_len:
                # 부족하면 0으로 패딩
                features = features + [0.0] * (expected_len - len(features))
            elif len(features) > expected_len:
                # 넘치면 잘라냄
                features = features[:expected_len]

            # strategy label mapping
            strat = row.get("strategy")
            if strat not in strategy_map:
                continue
            X_list.append([float(x) if x is not None else 0.0 for x in features])
            y_list.append(strategy_map[strat])

        except Exception:
            continue

    if not X_list or not y_list:
        print("❌ [evo_meta_dataset] 유효 샘플 없음")
        return None, None, None

    try:
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
    except Exception as e:
        print(f"❌ [evo_meta_dataset] ndarray 변환 실패: {e}")
        return None, None, None

    print(f"[✅ prepare_evo_meta_dataset] X: {X.shape}, y: {y.shape}, num_strategies: {num_strategies}")
    return X, y, num_strategies
