import os
import pandas as pd
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from logger import get_feature_hash
from failure_db import load_existing_failure_hashes
from config import get_NUM_CLASSES

NUM_CLASSES = get_NUM_CLASSES()

WRONG_CSV = "/persistent/wrong_predictions.csv"

# 속도 최적화: 최근 구간만 사용
_MAX_ROWS_FOR_SAMPLING = 800  # 필요 시 600~1000 범위에서 조정 가능

def load_training_prediction_data(symbol, strategy, input_size, window, group_id=None, min_per_class=10):
    import random
    from collections import Counter
    from config import FAIL_AUGMENT_RATIO, get_class_ranges, set_NUM_CLASSES

    sequences = []

    # ✅ 클래스 범위 계산
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    num_classes = len(class_ranges)
    set_NUM_CLASSES(num_classes)

    # ========= 0) 데이터 로드 (최근 구간만) =========
    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline_by_strategy → 데이터 없음")
        return None, None
    df_price = df_price.tail(_MAX_ROWS_FOR_SAMPLING).copy()

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        print(f"[❌ 실패] {symbol}-{strategy}: compute_features → 데이터 없음/NaN")
        return None, None
    # 동일하게 최근 구간만 사용
    df_feat = df_feat.tail(_MAX_ROWS_FOR_SAMPLING).dropna().reset_index(drop=True)

    # ✅ 타임스탬프 컬럼 통일
    if "timestamp" not in df_feat.columns:
        if "datetime" in df_feat.columns:
            df_feat.rename(columns={"datetime": "timestamp"}, inplace=True)
        else:
            raise Exception("timestamp 또는 datetime 컬럼 없음")

    # ========= 1) 라벨링(간단 pct_change 기반, class_ranges에 매핑) =========
    #  - df_price와 df_feat가 같은 최근 구간으로 맞춰졌으므로 길이 불일치 최소화
    returns = df_price["close"].pct_change().fillna(0).to_numpy()
    labels = []
    for r in returns:
        idx = 0
        for i, rng in enumerate(class_ranges):
            if isinstance(rng, tuple) and len(rng) == 2:
                low, high = rng
                if low <= r <= high:
                    idx = i
                    break
        labels.append(idx)
    # feat 길이에 맞춰 자르기
    df_feat["label"] = labels[:len(df_feat)]

    used_hashes = set()
    existing_hashes = load_existing_failure_hashes()

    fail_count, normal_count = 0, 0

    # ========= 2) 실패 샘플 우선 수집 (최근 구간만) =========
    if os.path.exists(WRONG_CSV):
        try:
            _df_all = pd.read_csv(WRONG_CSV, encoding="utf-8-sig", on_bad_lines="skip")

            base_cols = ["timestamp", "symbol", "strategy", "predicted_class"]
            for c in base_cols:
                if c not in _df_all.columns:
                    raise ValueError(f"'{c}' 컬럼 없음")

            opt_cols = [c for c in ["regime", "raw_prob", "calib_prob"] if c in _df_all.columns]
            dfw = _df_all[base_cols + opt_cols].copy()

            # 심볼/전략 필터
            dfw = dfw[(dfw["symbol"] == symbol) & (dfw["strategy"] == strategy)]
            dfw["timestamp"] = pd.to_datetime(dfw["timestamp"], errors="coerce")
            dfw = dfw[dfw["timestamp"].notna()]

            # 최근 구간만: feat의 최소 시각 이후만 사용 (조인 비용 없이 빠른 컷)
            feat_min_ts = pd.to_datetime(df_feat["timestamp"], errors="coerce").min()
            if pd.notna(feat_min_ts):
                dfw = dfw[dfw["timestamp"] >= feat_min_ts]

            # 타입 보정
            if "raw_prob" not in dfw.columns:  dfw["raw_prob"] = np.nan
            if "calib_prob" not in dfw.columns: dfw["calib_prob"] = np.nan
            if "regime" not in dfw.columns:     dfw["regime"] = "unknown"

            dfw["raw_prob"] = pd.to_numeric(dfw["raw_prob"], errors="coerce").clip(0, 1)
            dfw["calib_prob"] = pd.to_numeric(dfw["calib_prob"], errors="coerce").clip(0, 1)
            dfw["label"] = pd.to_numeric(dfw.get("predicted_class", -1), errors="coerce").astype("Int64")
            dfw = dfw[(dfw["label"] >= 0) & (dfw["label"] < num_classes)]

            # 실패 샘플 → 시퀀스화
            feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "label")]
            for _, row in dfw.iterrows():
                entry_time = row["timestamp"]
                label = int(row["label"])

                past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
                if past_window.empty:
                    continue

                xb = past_window[feat_cols].to_numpy(dtype=np.float32)

                # 좌측 패딩(길이 부족 시)
                if len(xb) < window:
                    pad_top = window - len(xb)
                    xb = np.pad(xb, ((pad_top, 0), (0, 0)), mode="constant")

                # 피처 차원 패딩
                if xb.shape[1] < input_size:
                    xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
                elif xb.shape[1] > input_size:
                    xb = xb[:, :input_size]

                if xb.shape != (window, input_size):
                    continue

                h = get_feature_hash(xb[-1])
                if h in used_hashes or h in existing_hashes:
                    continue
                used_hashes.add(h)

                for _ in range(FAIL_AUGMENT_RATIO * 2):
                    sequences.append((xb.copy(), label))
                    fail_count += 1
        except Exception as e:
            print(f"[⚠️ 실패 로드 예외] {symbol}-{strategy}: {e}")

    # ========= 3) 정규 학습 샘플 수집 (최근 구간만) =========
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "label")]
    for i in range(window, len(df_feat)):
        try:
            window_df = df_feat.iloc[i - window:i]
            label = int(df_feat.iloc[i].get("label", -1))
            if not (0 <= label < num_classes):
                continue

            xb = window_df[feat_cols].to_numpy(dtype=np.float32)

            # 피처 차원 패딩/절단
            if xb.shape[1] < input_size:
                xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
            elif xb.shape[1] > input_size:
                xb = xb[:, :input_size]

            if xb.shape != (window, input_size):
                continue

            h = get_feature_hash(xb[-1])
            if h in used_hashes:
                continue
            used_hashes.add(h)

            sequences.append((xb.copy(), label))
            normal_count += 1
        except Exception:
            continue

    # ========= 4) 클래스별 최소 샘플 보장 =========
    label_counts = Counter([s[1] for s in sequences])
    all_by_label = {cls: [] for cls in range(num_classes)}
    for xb, y in sequences:
        all_by_label[y].append(xb)

    for cls in range(num_classes):
        while len(all_by_label[cls]) < min_per_class:
            # 인접 클래스에서 보충
            neighbors = [c for c in (cls - 1, cls + 1) if 0 <= c < num_classes and all_by_label[c]]
            candidates = sum((all_by_label[c] for c in neighbors), [])
            if not candidates:
                dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
                all_by_label[cls].append(dummy)
            else:
                xb = random.choice(candidates)
                noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                all_by_label[cls].append(xb + noise)

    # ========= 5) 최종 시퀀스 구성 =========
    sequences = [(xb, cls) for cls, xb_list in all_by_label.items() for xb in xb_list]

    # 데이터 전무 시 더미 보강
    if not sequences:
        for _ in range(FAIL_AUGMENT_RATIO * 2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = random.randint(0, num_classes - 1)
            sequences.append((dummy, random_label))

    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)

    print(f"[✅ load_training_prediction_data 완료] {symbol}-{strategy} → 실패데이터 {fail_count} / 정상데이터 {normal_count} / 최종 {len(y)} (클래스별 최소 {min_per_class} 보장, 최근 {_MAX_ROWS_FOR_SAMPLING}행 기준)")
    return X, y
