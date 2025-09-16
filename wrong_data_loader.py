# wrong_data_loader.py (PATCHED: 안전성 보강 + 폴백 처리 + 타임스탬프/라벨 정합성)
import os
import pandas as pd
import numpy as np

# 안전한 imports: data.utils, logger, failure_db, config
try:
    from data.utils import get_kline_by_strategy, compute_features
except Exception:
    def get_kline_by_strategy(symbol, strategy):
        return None
    def compute_features(symbol, df, strategy):
        return None

try:
    from logger import get_feature_hash
except Exception:
    # fallback hash using numpy bytes -> md5
    import hashlib
    def get_feature_hash(vec):
        try:
            b = np.ascontiguousarray(vec).tobytes()
            return hashlib.md5(b).hexdigest()
        except Exception:
            return str(hash(bytes(np.ascontiguousarray(vec).flatten())))

try:
    from failure_db import load_existing_failure_hashes
except Exception:
    def load_existing_failure_hashes():
        return set()

try:
    from config import get_NUM_CLASSES
except Exception:
    def get_NUM_CLASSES():
        return int(os.getenv("NUM_CLASSES", "3"))

NUM_CLASSES = get_NUM_CLASSES()

WRONG_CSV = "/persistent/wrong_predictions.csv"

# 속도 최적화: 최근 구간만 사용
_MAX_ROWS_FOR_SAMPLLING_DEFAULT = 800  # 필요 시 600~1000 범위에서 조정 가능
_MAX_ROWS_FOR_SAMPLING = int(os.getenv("YOPO_MAX_ROWS_FOR_SAMPLING", _MAX_ROWS_FOR_SAMPLLING_DEFAULT))
if _MAX_ROWS_FOR_SAMPLING <= 0:
    _MAX_ROWS_FOR_SAMPLING = _MAX_ROWS_FOR_SAMPLLING_DEFAULT

def _map_to_class_idx(r: float, class_ranges) -> int:
    """누적 수익률 r을 동적 클래스 경계(class_ranges)에 매핑."""
    idx = 0
    for i, rng in enumerate(class_ranges):
        if isinstance(rng, tuple) and len(rng) == 2:
            low, high = rng
            try:
                if low <= r <= high:
                    idx = i
                    break
            except Exception:
                continue
    return idx

def _future_cum_return(close: pd.Series, k_future: int) -> np.ndarray:
    """
    미래 k 스텝 누적수익률: (close[t+k]/close[t]) - 1
    길이 보전을 위해 마지막 k 구간은 0으로 채운다(학습 시 무해).
    """
    future = close.shift(-k_future)
    ret = (future / close - 1.0).fillna(0.0).to_numpy()
    return ret

def load_training_prediction_data(symbol, strategy, input_size, window, group_id=None, min_per_class=10):
    import random
    from collections import Counter
    from config import FAIL_AUGMENT_RATIO, get_class_ranges, set_NUM_CLASSES

    sequences = []

    # ✅ 클래스 범위 계산
    try:
        class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    except Exception as e:
        print(f"[ERROR] get_class_ranges 실패: {e}")
        class_ranges = [( -1.0, 1.0 )]  # 기본 안전 범위

    num_classes = len(class_ranges) if class_ranges else get_NUM_CLASSES()
    try:
        set_NUM_CLASSES(num_classes)
    except Exception:
        pass

    # ========= 0) 데이터 로드 (최근 구간만) =========
    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or (hasattr(df_price, "empty") and df_price.empty):
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline_by_strategy → 데이터 없음")
        return None, None

    # ensure df_price is DataFrame
    if not isinstance(df_price, pd.DataFrame):
        try:
            df_price = pd.DataFrame(df_price)
        except Exception:
            print(f"[❌ 실패] {symbol}-{strategy}: get_kline_by_strategy 반환 형식 오류")
            return None, None

    df_price = df_price.tail(_MAX_ROWS_FOR_SAMPLING).copy().reset_index(drop=True)

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or (hasattr(df_feat, "empty") and df_feat.empty):
        print(f"[❌ 실패] {symbol}-{strategy}: compute_features → 데이터 없음/NaN")
        return None, None

    # ensure df_feat is DataFrame
    if not isinstance(df_feat, pd.DataFrame):
        try:
            df_feat = pd.DataFrame(df_feat)
        except Exception:
            print(f"[❌ 실패] {symbol}-{strategy}: compute_features 반환 형식 오류")
            return None, None

    # 동일하게 최근 구간만 사용
    df_feat = df_feat.tail(_MAX_ROWS_FOR_SAMPLING).dropna(axis=0, how='any').reset_index(drop=True)

    # ✅ 타임스탬프 컬럼 통일
    if "timestamp" not in df_feat.columns:
        if "datetime" in df_feat.columns:
            df_feat = df_feat.rename(columns={"datetime": "timestamp"})
        else:
            # attempt to infer timestamp from df_price if possible
            if "timestamp" in df_price.columns:
                df_feat["timestamp"] = df_price["timestamp"].tail(len(df_feat)).values
            elif "datetime" in df_price.columns:
                df_feat["timestamp"] = df_price["datetime"].tail(len(df_feat)).values
            else:
                # give synthetic monotonic timestamps to preserve indexing
                df_feat["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_feat), freq="T")

    # ========= 1) 라벨링(미래 k-스텝 누적수익률 기반 → class_ranges 매핑) =========
    k_future = max(2, min(8, max(1, int(window // 4))))
    try:
        returns = _future_cum_return(df_price["close"], k_future=k_future)
    except Exception:
        # fallback: zeros
        returns = np.zeros(len(df_price), dtype=float)

    # create labels aligned to df_feat length safely
    labels_all = [_map_to_class_idx(float(r), class_ranges) for r in returns]
    if len(labels_all) < len(df_feat):
        # pad with last label
        pad_val = labels_all[-1] if labels_all else 0
        labels_all = labels_all + [pad_val] * (len(df_feat) - len(labels_all))
    labels = labels_all[:len(df_feat)]
    df_feat["label"] = labels

    used_hashes = set()
    try:
        existing_hashes = load_existing_failure_hashes() or set()
    except Exception:
        existing_hashes = set()

    fail_count, normal_count = 0, 0

    # 공통 피처 컬럼
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "label")]

    # ========= 2) 실패 샘플 우선 수집 (최근 구간만, 국소 시프트 증강) =========
    if os.path.exists(WRONG_CSV):
        try:
            _df_all = pd.read_csv(WRONG_CSV, encoding="utf-8-sig", on_bad_lines="skip")
            base_cols = ["timestamp", "symbol", "strategy", "predicted_class"]
            for c in base_cols:
                if c not in _df_all.columns:
                    raise ValueError(f"'{c}' 컬럼 없음")
            opt_cols = [c for c in ["regime", "raw_prob", "calib_prob"] if c in _df_all.columns]
            dfw = _df_all[base_cols + opt_cols].copy()
            dfw = dfw[(dfw["symbol"] == symbol) & (dfw["strategy"] == strategy)].copy()
            dfw["timestamp"] = pd.to_datetime(dfw["timestamp"], errors="coerce")
            dfw = dfw[dfw["timestamp"].notna()]

            # 최근 구간만: feat의 최소 시각 이후만 사용
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

            # 인덱스 매핑 준비
            ts_series = pd.to_datetime(df_feat["timestamp"], errors="coerce").values
            for _, row in dfw.iterrows():
                try:
                    entry_time = row["timestamp"]
                    label = int(row["label"])
                    # entry_time may be Timestamp or str
                    if pd.isna(entry_time):
                        continue
                    # find first index where ts_series >= entry_time
                    idx_candidates = np.where(ts_series >= np.datetime64(entry_time))[0]
                    if len(idx_candidates) == 0:
                        continue
                    end_idx0 = int(idx_candidates[0])

                    offsets = [-2, -1, 0, 1, 2]
                    appended_once = False
                    for off in offsets:
                        end_idx = end_idx0 + off
                        start_idx = end_idx - window
                        if end_idx <= 0 or start_idx < 0:
                            continue
                        if end_idx > len(df_feat):
                            continue

                        window_df = df_feat.iloc[start_idx:end_idx]
                        if len(window_df) != window:
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
                        if h in used_hashes or h in existing_hashes:
                            continue
                        used_hashes.add(h)

                        sequences.append((xb.copy(), label))
                        appended_once = True
                        for _ in range(max(0, FAIL_AUGMENT_RATIO - 1)):
                            noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                            sequences.append((xb.copy() + noise, label))
                        fail_count += 1

                    # 보완 시도
                    if not appended_once:
                        past_window = df_feat[df_feat["timestamp"] < row["timestamp"]].tail(window)
                        if past_window.shape[0] == window:
                            xb = past_window[feat_cols].to_numpy(dtype=np.float32)
                            if xb.shape[1] < input_size:
                                xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
                            elif xb.shape[1] > input_size:
                                xb = xb[:, :input_size]
                            if xb.shape == (window, input_size):
                                h = get_feature_hash(xb[-1])
                                if h not in used_hashes and h not in existing_hashes:
                                    used_hashes.add(h)
                                    sequences.append((xb.copy(), label))
                                    for _ in range(max(0, FAIL_AUGMENT_RATIO - 1)):
                                        noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                                        sequences.append((xb.copy() + noise, label))
                                    fail_count += 1
                except Exception:
                    continue

        except Exception as e:
            print(f"[⚠️ 실패 로드 예외] {symbol}-{strategy}: {e}")

    # ========= 3) 정규 학습 샘플 수집 (최근 구간만) =========
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
            neighbors = [c for c in (cls - 1, cls + 1) if 0 <= c < num_classes and all_by_label.get(c)]
            candidates = sum((all_by_label[c] for c in neighbors), []) if neighbors else []
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
        for _ in range(max(1, int(FAIL_AUGMENT_RATIO)) * 2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = random.randint(0, max(0, num_classes - 1))
            sequences.append((dummy, random_label))

    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)

    print(
        f"[✅ load_training_prediction_data 완료] {symbol}-{strategy} → "
        f"실패데이터 {fail_count} / 정상데이터 {normal_count} / 최종 {len(y)} "
        f"(클래스별 최소 {min_per_class} 보장, 최근 {_MAX_ROWS_FOR_SAMPLING}행, k_future={k_future})"
    )
    return X, y
