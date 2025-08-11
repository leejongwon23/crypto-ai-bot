# ✅ 파일명: evo_meta_dataset.py
import os
import pandas as pd
import numpy as np

def prepare_evo_meta_dataset(csv_path="/persistent/wrong_predictions.csv"):
    import pandas as pd
    import numpy as np

    if not os.path.exists(csv_path):
        print(f"❌ [evo_meta_dataset] 파일 없음: {csv_path}")
        return None, None, None

    df = pd.read_csv(csv_path)

    if df.empty:
        print("❌ [evo_meta_dataset] 실패 예측 데이터가 비어 있음")
        return None, None, None

    # 필수 컬럼 체크
    required_cols = ["strategy", "softmax", "expected_return", "actual_return", "predicted_class", "true_class"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ [evo_meta_dataset] 필수 컬럼 누락: {col}")
            return None, None, None

    # NaN 제거
    df = df.dropna(subset=required_cols)
    if df.empty:
        print("❌ [evo_meta_dataset] NaN 제거 후 데이터 없음")
        return None, None, None

    # 전략 → 인덱스 매핑
    strategy_map = {s: i for i, s in enumerate(sorted(df["strategy"].unique()))}
    num_strategies = len(strategy_map)

    # 입력(X), 출력(y) 구성
    X, y = [], []
    for _, row in df.iterrows():
        try:
            # softmax, expected_return, actual_return이 문자열 리스트면 변환
            sm = eval(row["softmax"]) if isinstance(row["softmax"], str) else row["softmax"]
            er = eval(row["expected_return"]) if isinstance(row["expected_return"], str) else row["expected_return"]
            ar = eval(row["actual_return"]) if isinstance(row["actual_return"], str) else row["actual_return"]

            if not (isinstance(sm, (list, tuple)) and isinstance(er, (list, tuple)) and isinstance(ar, (list, tuple))):
                continue  # 형식 불일치 시 스킵

            # 전략별 softmax + expected_return + actual_return → 하나의 feature 벡터로 결합
            features = list(sm) + list(er) + list(ar)
            if len(features) != num_strategies * 3:
                continue  # 전략 개수와 맞지 않으면 스킵

            X.append(features)
            y.append(strategy_map[row["strategy"]])

        except Exception as e:
            print(f"[⚠️ prepare_evo_meta_dataset] 변환 예외: {e}")
            continue

    if not X or not y:
        print("❌ [evo_meta_dataset] 유효 샘플 없음")
        return None, None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"[✅ prepare_evo_meta_dataset] X: {X.shape}, y: {y.shape}, num_strategies: {num_strategies}")
    return X, y, num_strategies
