# wrong_data_loader.py (PATCHED v2025-09-17a: 소수클래스 실패우선 + 오프셋±5 + 증강상향 + 해시가드강화)
import os
import pandas as pd
import numpy as np

# 안전한 imports: data.utils, logger, failure_db, config
try:
    from data.utils import get_kline_by_strategy, compute_features
except Exception:
    def get_kline_by_strategy(symbol, strategy): return None
    def compute_features(symbol, df, strategy): return None

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
    def load_existing_failure_hashes(): return set()

try:
    from config import get_NUM_CLASSES
except Exception:
    def get_NUM_CLASSES(): return int(os.getenv("NUM_CLASSES", "3"))

NUM_CLASSES = get_NUM_CLASSES()

WRONG_CSV = "/persistent/wrong_predictions.csv"

# 속도 최적화: 최근 구간만 사용
_MAX_ROWS_FOR_SAMPLLING_DEFAULT = 800
_MAX_ROWS_FOR_SAMPLING = int(os.getenv("YOPO_MAX_ROWS_FOR_SAMPLING", _MAX_ROWS_FOR_SAMPLLING_DEFAULT))
if _MAX_ROWS_FOR_SAMPLING <= 0:
    _MAX_ROWS_FOR_SAMPLING = _MAX_ROWS_FOR_SAMPLLING_DEFAULT

# 🔧 증강/오프셋/부스트 파라미터 (env로 조정 가능)
_FAIL_AUG_MULT = int(os.getenv("FAIL_AUG_MULT", "2"))                 # 실패샘플 증강 배수(기존 FAIL_AUGMENT_RATIO에 곱)
_MINOR_CLASS_BOOST = int(os.getenv("MINOR_CLASS_BOOST", "2"))         # 소수클래스 추가 복제 배수
_OFFSET_MAX = int(os.getenv("FAIL_OFFSET_MAX", "5"))                   # 시간 오프셋 최대(±)
_OFFSET_MIN = int(os.getenv("FAIL_OFFSET_MIN", "3"))                   # 시간 오프셋 최소
_USE_FULL_WINDOW_HASH = os.getenv("USE_FULL_WINDOW_HASH", "1") == "1" # 윈도우 전체 해시도 중복가드

def _map_to_class_idx(r: float, class_ranges) -> int:
    """누적 수익률 r을 동적 클래스 경계(class_ranges)에 매핑."""
    idx = 0
    for i, rng in enumerate(class_ranges):
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            low, high = rng
            try:
                if float(low) <= float(r) <= float(high):
                    idx = i
                    break
            except Exception:
                continue
    return idx

def _future_cum_return(close: pd.Series, k_future: int) -> np.ndarray:
    """미래 k 스텝 누적수익률: (close[t+k]/close[t]) - 1 (끝 k개는 0으로 채움)."""
    future = close.shift(-k_future)
    ret = (future / close - 1.0).fillna(0.0).to_numpy()
    return ret

def _hash_window(xb: np.ndarray) -> str:
    """윈도우 전체를 이용한 강한 해시(옵션)."""
    try:
        import hashlib
        arr = np.ascontiguousarray(xb).tobytes()
        return hashlib.sha256(arr).hexdigest()
    except Exception:
        return get_feature_hash(xb[-1])

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
        class_ranges = [(-1.0, 1.0)]

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

    if not isinstance(df_price, pd.DataFrame):
        try: df_price = pd.DataFrame(df_price)
        except Exception:
            print(f"[❌ 실패] {symbol}-{strategy}: get_kline_by_strategy 반환 형식 오류")
            return None, None

    df_price = df_price.tail(_MAX_ROWS_FOR_SAMPLING).copy().reset_index(drop=True)

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or (hasattr(df_feat, "empty") and df_feat.empty):
        print(f"[❌ 실패] {symbol}-{strategy}: compute_features → 데이터 없음/NaN")
        return None, None

    if not isinstance(df_feat, pd.DataFrame):
        try: df_feat = pd.DataFrame(df_feat)
        except Exception:
            print(f"[❌ 실패] {symbol}-{strategy}: compute_features 반환 형식 오류")
            return None, None

    df_feat = df_feat.tail(_MAX_ROWS_FOR_SAMPLING).dropna(axis=0, how='any').reset_index(drop=True)

    # ✅ 타임스탬프 컬럼 통일
    if "timestamp" not in df_feat.columns:
        if "datetime" in df_feat.columns:
            df_feat = df_feat.rename(columns={"datetime": "timestamp"})
        else:
            if "timestamp" in df_price.columns:
                df_feat["timestamp"] = df_price["timestamp"].tail(len(df_feat)).values
            elif "datetime" in df_price.columns:
                df_feat["timestamp"] = df_price["datetime"].tail(len(df_feat)).values
            else:
                df_feat["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_feat), freq="T")

    # ========= 1) 라벨링 =========
    k_future = max(2, min(8, max(1, int(window // 4))))
    try:
        returns = _future_cum_return(df_price["close"], k_future=k_future)
    except Exception:
        returns = np.zeros(len(df_price), dtype=float)

    labels_all = [_map_to_class_idx(float(r), class_ranges) for r in returns]
    if len(labels_all) < len(df_feat):
        pad_val = labels_all[-1] if labels_all else 0
        labels_all += [pad_val] * (len(df_feat) - len(labels_all))
    labels = labels_all[:len(df_feat)]
    df_feat["label"] = labels

    # 중복 해시 가드 세트 (기존 + 강화)
    used_hashes = set()
    try:
        existing_hashes = load_existing_failure_hashes() or set()
    except Exception:
        existing_hashes = set()

    fail_count, normal_count = 0, 0
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "label")]

    # ========= 2) 실패 샘플 우선 수집 =========
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

            # feat 최소 시각 이후만
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

            # 오프셋 범위: ±3~±_OFFSET_MAX
            offset_min = max(1, int(_OFFSET_MIN))
            offset_max = max(offset_min, int(_OFFSET_MAX))
            offsets = list(range(-offset_max, -offset_min + 1)) + [0] + list(range(offset_min, offset_max + 1))

            ts_series = pd.to_datetime(df_feat["timestamp"], errors="coerce").values
            for _, row in dfw.iterrows():
                try:
                    entry_time = row["timestamp"]
                    label = int(row["label"])
                    if pd.isna(entry_time):
                        continue
                    idx_candidates = np.where(ts_series >= np.datetime64(entry_time))[0]
                    if len(idx_candidates) == 0:
                        continue
                    end_idx0 = int(idx_candidates[0])

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

                        # 강화된 중복가드: 마지막행 + 전체윈도우 해시
                        h_last = get_feature_hash(xb[-1])
                        h_full = _hash_window(xb) if _USE_FULL_WINDOW_HASH else None
                        if (h_last in used_hashes or h_last in existing_hashes) or (h_full and (h_full in used_hashes or h_full in existing_hashes)):
                            continue
                        used_hashes.add(h_last)
                        if h_full: used_hashes.add(h_full)

                        sequences.append((xb.copy(), label))
                        appended_once = True

                        # 증강 비율 상향: FAIL_AUGMENT_RATIO * _FAIL_AUG_MULT
                        aug_times = max(0, int(FAIL_AUGMENT_RATIO) * max(1, _FAIL_AUG_MULT) - 1)
                        for _ in range(aug_times):
                            noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                            sequences.append((xb.copy() + noise, label))
                        fail_count += 1

                    # 매칭 실패 시 과거 윈도우 보완
                    if not appended_once:
                        past_window = df_feat[df_feat["timestamp"] < row["timestamp"]].tail(window)
                        if past_window.shape[0] == window:
                            xb = past_window[feat_cols].to_numpy(dtype=np.float32)
                            if xb.shape[1] < input_size:
                                xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
                            elif xb.shape[1] > input_size:
                                xb = xb[:, :input_size]
                            if xb.shape == (window, input_size):
                                h_last = get_feature_hash(xb[-1])
                                h_full = _hash_window(xb) if _USE_FULL_WINDOW_HASH else None
                                if (h_last not in used_hashes and h_last not in existing_hashes) and (not h_full or (h_full not in used_hashes and h_full not in existing_hashes)):
                                    used_hashes.add(h_last); 
                                    if h_full: used_hashes.add(h_full)
                                    sequences.append((xb.copy(), label))
                                    aug_times = max(0, int(FAIL_AUGMENT_RATIO) * max(1, _FAIL_AUG_MULT) - 1)
                                    for _ in range(aug_times):
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

            if xb.shape[1] < input_size:
                xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
            elif xb.shape[1] > input_size:
                xb = xb[:, :input_size]

            if xb.shape != (window, input_size):
                continue

            h_last = get_feature_hash(xb[-1])
            h_full = _hash_window(xb) if _USE_FULL_WINDOW_HASH else None
            if (h_last in used_hashes) or (h_full and h_full in used_hashes):
                continue
            used_hashes.add(h_last); 
            if h_full: used_hashes.add(h_full)

            sequences.append((xb.copy(), label))
            normal_count += 1
        except Exception:
            continue

    # ========= 4) 클래스별 최소 샘플 보장 + 소수클래스 추가부스트 =========
    label_counts = Counter([s[1] for s in sequences]) if sequences else Counter()
    all_by_label = {cls: [] for cls in range(num_classes)}
    for xb, y in sequences:
        all_by_label[y].append(xb)

    # 소수클래스 기준: 중앙값 미만인 클래스들
    if label_counts:
        med = np.median([c for c in label_counts.values()])
        minority = [cls for cls, c in label_counts.items() if c < med]
    else:
        minority = []

    # 최소 보장
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

    # 소수클래스 추가 부스트
    for cls in minority:
        base = list(all_by_label[cls])
        boost_times = max(0, _MINOR_CLASS_BOOST - 1)
        for xb in base:
            for _ in range(boost_times):
                noise = np.random.normal(0, 0.012, xb.shape).astype(np.float32)
                all_by_label[cls].append(xb + noise)

    # ========= 5) 최종 시퀀스 구성 =========
    sequences = [(xb, cls) for cls, xb_list in all_by_label.items() for xb in xb_list]

    if not sequences:
        for _ in range(2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = random.randint(0, max(0, num_classes - 1))
            sequences.append((dummy, random_label))

    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)

    print(
        f"[✅ load_training_prediction_data 완료] {symbol}-{strategy} → "
        f"실패데이터 {fail_count} / 정상데이터 {normal_count} / 최종 {len(y)} "
        f"(소수클래스 {minority}, 최소 {min_per_class} 보장, 최근 {_MAX_ROWS_FOR_SAMPLING}행, k_future={k_future}, "
        f"offset±{_OFFSET_MAX}, fail_aug×{_FAIL_AUG_MULT}, full_hash={int(_USE_FULL_WINDOW_HASH)})"
    )
    return X, y
