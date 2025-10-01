# wrong_data_loader.py (FINAL v2025-10-01a)
# - CV_CONFIG 연동(min_per_class 자동)
# - 클래스 경계 포함 규칙 일치(마지막 구간만 우측 포함)
# - label == -1 전면 배제 가드
# - 실패샘플 우선/오프셋±/중복가드/소수클래스 부스트 유지

import os
import pandas as pd
import numpy as np

# 안전 import
try:
    from data.utils import get_kline_by_strategy, compute_features
except Exception:
    def get_kline_by_strategy(symbol, strategy): return None
    def compute_features(symbol, df, strategy): return None

try:
    from logger import get_feature_hash
except Exception:
    import hashlib
    def get_feature_hash(vec):
        try:
            b = np.ascontiguousarray(np.asarray(vec, dtype=np.float32)).tobytes()
            return hashlib.md5(b).hexdigest()
        except Exception:
            return hashlib.md5(str(vec).encode()).hexdigest()

try:
    from failure_db import load_existing_failure_hashes
except Exception:
    def load_existing_failure_hashes(): return set()

# --- config 연동 ---
try:
    from config import get_NUM_CLASSES, get_class_ranges, set_NUM_CLASSES, FAIL_AUGMENT_RATIO, get_CV_CONFIG
except Exception:
    def get_NUM_CLASSES(): return int(os.getenv("NUM_CLASSES", "3"))
    def get_class_ranges(**kwargs): return [(-1.0, 0.0), (0.0, 1.0)]
    def set_NUM_CLASSES(n): pass
    FAIL_AUGMENT_RATIO = int(os.getenv("FAIL_AUGMENT_RATIO", "2"))
    def get_CV_CONFIG(): return {"min_per_class": 3}

NUM_CLASSES = get_NUM_CLASSES()
WRONG_CSV = "/persistent/wrong_predictions.csv"

# 최근 구간만 사용
_MAX_ROWS_FOR_SAMPLLING_DEFAULT = 800
_MAX_ROWS_FOR_SAMPLING = int(os.getenv("YOPO_MAX_ROWS_FOR_SAMPLING", _MAX_ROWS_FOR_SAMPLLING_DEFAULT))
if _MAX_ROWS_FOR_SAMPLING <= 0:
    _MAX_ROWS_FOR_SAMPLING = _MAX_ROWS_FOR_SAMPLLING_DEFAULT

# 증강/오프셋/부스트 파라미터
_FAIL_AUG_MULT = int(os.getenv("FAIL_AUG_MULT", "2"))
_MINOR_CLASS_BOOST = int(os.getenv("MINOR_CLASS_BOOST", "2"))
_OFFSET_MAX = int(os.getenv("FAIL_OFFSET_MAX", "5"))
_OFFSET_MIN = int(os.getenv("FAIL_OFFSET_MIN", "3"))
_USE_FULL_WINDOW_HASH = os.getenv("USE_FULL_WINDOW_HASH", "1") == "1"

def _future_cum_return(close: pd.Series, k_future: int) -> np.ndarray:
    """미래 k 스텝 누적수익률: (close[t+k]/close[t]) - 1 (끝 k개는 0)."""
    try:
        c = pd.to_numeric(close, errors="coerce")
        future = c.shift(-k_future)
        ret = (future / c - 1.0).fillna(0.0).to_numpy(dtype=float)
        ret[~np.isfinite(ret)] = 0.0
        return ret
    except Exception:
        return np.zeros(len(close), dtype=float)

def _hash_window(xb: np.ndarray) -> str:
    """윈도우 전체를 이용한 강한 해시(옵션)."""
    try:
        import hashlib
        arr = np.ascontiguousarray(xb, dtype=np.float32).tobytes()
        return hashlib.sha256(arr).hexdigest()
    except Exception:
        return get_feature_hash(xb[-1])

def _ensure_ts(series_like) -> pd.Series:
    """타임스탬프(UTC) 시리즈 생성."""
    return pd.to_datetime(series_like, errors="coerce", utc=True)

def _map_to_class_idx_inclusive_last(r: float, class_ranges) -> int:
    """
    config의 경계 규칙과 일치:
    - 모든 구간은 [lo, hi) (좌포함/우미포함X)
    - '마지막' 구간만 [lo, hi] (우측 포함)
    """
    try:
        C = len(class_ranges)
        if C == 0:
            return -1
        for i, (lo, hi) in enumerate(class_ranges[:-1]):
            if (r >= float(lo)) and (r < float(hi)):
                return i
        lo, hi = class_ranges[-1]
        if (r >= float(lo)) and (r <= float(hi)):
            return C - 1
        return -1
    except Exception:
        return -1

def load_training_prediction_data(
    symbol: str,
    strategy: str,
    input_size: int,
    window: int,
    group_id: int | None = None,
    min_per_class: int | None = None,
):
    """
    반환: X(np.float32: [N, window, input_size]), y(np.int64: [N])
    - 실패샘플 가중 수집 + 정규 샘플
    - label==-1 전면 배제
    - 클래스별 최소 샘플수 보장(CV_CONFIG.min_per_class 기본)
    """
    import random
    from collections import Counter

    # --- 클래스 경계 ---
    try:
        class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    except Exception as e:
        print(f"[ERROR] get_class_ranges 실패: {e}")
        class_ranges = [(-1.0, 0.0), (0.0, 1.0)]

    num_classes = len(class_ranges) if class_ranges else get_NUM_CLASSES()
    try:
        set_NUM_CLASSES(num_classes)
    except Exception:
        pass

    # --- CV_CONFIG에서 min_per_class 기본값 수급 ---
    if min_per_class is None:
        try:
            cv_cfg = get_CV_CONFIG() or {}
            min_per_class = int(cv_cfg.get("min_per_class", 3))
        except Exception:
            min_per_class = 3
    min_per_class = max(1, int(min_per_class))

    # ========= 0) 데이터 로드 =========
    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or (hasattr(df_price, "empty") and df_price.empty):
        print(f"[❌] {symbol}-{strategy}: 가격 데이터 없음")
        return None, None
    if not isinstance(df_price, pd.DataFrame):
        try: df_price = pd.DataFrame(df_price)
        except Exception:
            print(f"[❌] {symbol}-{strategy}: 가격 데이터 형식 오류")
            return None, None

    df_price = df_price.tail(_MAX_ROWS_FOR_SAMPLING).copy().reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_price.columns:
            df_price[col] = pd.to_numeric(df_price[col], errors="coerce")
    df_price = df_price.dropna(subset=["close"]).reset_index(drop=True)

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or (hasattr(df_feat, "empty") and df_feat.empty):
        print(f"[❌] {symbol}-{strategy}: 피처 없음/NaN")
        return None, None
    if not isinstance(df_feat, pd.DataFrame):
        try: df_feat = pd.DataFrame(df_feat)
        except Exception:
            print(f"[❌] {symbol}-{strategy}: 피처 형식 오류")
            return None, None
    df_feat = df_feat.tail(_MAX_ROWS_FOR_SAMPLING).reset_index(drop=True)

    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    for drop_c in ["label", "strategy"]:
        if drop_c in num_cols:
            num_cols.remove(drop_c)

    if "timestamp" not in df_feat.columns:
        if "datetime" in df_feat.columns:
            df_feat = df_feat.rename(columns={"datetime": "timestamp"})
        elif "timestamp" in df_price.columns:
            df_feat["timestamp"] = df_price["timestamp"].tail(len(df_feat)).values
        elif "datetime" in df_price.columns:
            df_feat["timestamp"] = df_price["datetime"].tail(len(df_feat)).values
        else:
            df_feat["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df_feat), freq="T")

    if not num_cols:
        num_cols = [c for c in df_feat.columns if c not in ("timestamp", "strategy", "label")]

    df_feat[num_cols] = (
        df_feat[num_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    # ========= 1) 라벨링 (경계 규칙 일치) =========
    k_future = max(2, min(8, max(1, int(window // 4))))
    try:
        returns = _future_cum_return(df_price["close"], k_future=k_future)
    except Exception:
        returns = np.zeros(len(df_price), dtype=float)

    # df_feat 길이에 맞추기
    if len(returns) < len(df_feat):
        pad_val = returns[-1] if len(returns) else 0.0
        returns = np.concatenate([returns, np.full(len(df_feat) - len(returns), pad_val, dtype=float)])
    returns = returns[:len(df_feat)]

    labels = [_map_to_class_idx_inclusive_last(float(r), class_ranges) for r in returns]
    df_feat["label"] = labels

    # **-1 전면 배제** (이후 모든 단계에서 제외)
    mask_valid = df_feat["label"].astype(int) >= 0
    if not mask_valid.any():
        print(f"[❌] {symbol}-{strategy}: 유효 라벨 없음(-1 과다)")
        return None, None
    df_feat = df_feat.loc[mask_valid].reset_index(drop=True)

    ts_feat = _ensure_ts(df_feat["timestamp"]).values

    # ========= 2) 실패 샘플 우선 수집 =========
    used_hashes = set()
    try:
        existing_hashes = set(load_existing_failure_hashes() or set())
    except Exception:
        existing_hashes = set()

    sequences = []
    fail_count, normal_count = 0, 0
    feat_cols = [c for c in num_cols if c not in ("timestamp", "label")]

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
            dfw["timestamp"] = _ensure_ts(dfw["timestamp"])
            dfw = dfw[dfw["timestamp"].notna()]

            # feat 최소 시각 이후만
            if len(ts_feat) > 0:
                feat_min_ts = pd.to_datetime(ts_feat.min())
                dfw = dfw[dfw["timestamp"] >= feat_min_ts]

            dfw["raw_prob"] = pd.to_numeric(dfw.get("raw_prob", np.nan), errors="coerce").clip(0, 1)
            dfw["calib_prob"] = pd.to_numeric(dfw.get("calib_prob", np.nan), errors="coerce").clip(0, 1)
            dfw["label"] = pd.to_numeric(dfw["predicted_class"], errors="coerce").astype("Int64")
            dfw = dfw[(dfw["label"] >= 0) & (dfw["label"] < num_classes)]

            offset_min = max(1, int(_OFFSET_MIN))
            offset_max = max(offset_min, int(_OFFSET_MAX))
            offsets = list(range(-offset_max, -offset_min + 1)) + [0] + list(range(offset_min, offset_max + 1))

            ts_series = ts_feat  # numpy datetime64[ns, UTC]
            for _, row in dfw.iterrows():
                try:
                    entry_time = np.datetime64(row["timestamp"].to_datetime64())
                    label = int(row["label"])
                    idx_candidates = np.where(ts_series >= entry_time)[0]
                    if len(idx_candidates) == 0:
                        continue
                    end_idx0 = int(idx_candidates[0])

                    appended_once = False
                    for off in offsets:
                        end_idx = end_idx0 + off
                        start_idx = end_idx - window
                        if end_idx <= 0 or start_idx < 0 or end_idx > len(df_feat):
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

                        # 강화 중복가드
                        h_last = get_feature_hash(xb[-1])
                        h_full = _hash_window(xb) if _USE_FULL_WINDOW_HASH else None
                        if (h_last in used_hashes or h_last in existing_hashes) or (h_full and (h_full in used_hashes or h_full in existing_hashes)):
                            continue
                        used_hashes.add(h_last)
                        if h_full: used_hashes.add(h_full)

                        sequences.append((xb.copy(), label))
                        appended_once = True

                        # 실패샘플 증강
                        aug_times = max(0, int(FAIL_AUGMENT_RATIO) * max(1, _FAIL_AUG_MULT) - 1)
                        for _ in range(aug_times):
                            noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                            sequences.append((xb.copy() + noise, label))
                        fail_count += 1

                    # 매칭 실패 시 과거 윈도우 보완
                    if not appended_once:
                        past_mask = ts_series < entry_time
                        if past_mask.any():
                            last_idx = int(np.where(past_mask)[0][-1])
                            start_idx = last_idx - window + 1
                            if start_idx >= 0 and (last_idx + 1) <= len(df_feat):
                                past_window = df_feat.iloc[start_idx:last_idx + 1]
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
                                            used_hashes.add(h_last)
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

    # ========= 3) 정규 학습 샘플 =========
    for i in range(window, len(df_feat)):
        try:
            window_df = df_feat.iloc[i - window:i]
            label = int(df_feat.iloc[i].get("label", -1))
            if label < 0 or label >= num_classes:
                continue  # -1 완전 배제
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
            used_hashes.add(h_last)
            if h_full: used_hashes.add(h_full)

            sequences.append((xb.copy(), label))
            normal_count += 1
        except Exception:
            continue

    # ========= 4) 클래스별 최소 샘플 보장 + 소수클래스 부스트 =========
    from collections import Counter
    label_counts = Counter([s[1] for s in sequences]) if sequences else Counter()
    all_by_label = {cls: [] for cls in range(num_classes)}
    for xb, y in sequences:
        if 0 <= y < num_classes:
            all_by_label[y].append(xb)

    # 소수클래스(중앙값 미만)
    minority = []
    if label_counts:
        med = float(np.median([c for c in label_counts.values()])) if len(label_counts) > 0 else 0.0
        minority = [cls for cls, c in label_counts.items() if c < med]

    # 최소치 보장
    for cls in range(num_classes):
        while len(all_by_label[cls]) < min_per_class:
            neighbors = [c for c in (cls - 1, cls + 1) if 0 <= c < num_classes and all_by_label.get(c)]
            candidates = sum((all_by_label[c] for c in neighbors), []) if neighbors else []
            if not candidates:
                dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
                all_by_label[cls].append(dummy)
            else:
                xb = candidates[np.random.randint(0, len(candidates))]
                noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                all_by_label[cls].append(xb + noise)

    # 소수클래스 부스트
    for cls in minority:
        base = list(all_by_label[cls])
        boost_times = max(0, _MINOR_CLASS_BOOST - 1)
        for xb in base:
            for _ in range(boost_times):
                noise = np.random.normal(0, 0.012, xb.shape).astype(np.float32)
                all_by_label[cls].append(xb + noise)

    # ========= 5) 최종 구성 =========
    sequences = [(xb, cls) for cls, xb_list in all_by_label.items() for xb in xb_list]
    if not sequences:
        for _ in range(2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            rnd = np.random.randint(0, max(1, num_classes))
            sequences.append((dummy, int(rnd)))

    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)

    print(
        f"[✅ load_training_prediction_data] {symbol}-{strategy} → "
        f"실패 {fail_count} / 정상 {normal_count} / 최종 {len(y)} "
        f"(min_per_class={min_per_class}, 소수클래스={sorted(minority)}, "
        f"최근 {_MAX_ROWS_FOR_SAMPLING}행, k_future={k_future}, "
        f"offset±{_OFFSET_MAX}, fail_aug×{_FAIL_AUG_MULT}, full_hash={int(_USE_FULL_WINDOW_HASH)})"
    )
    return X, y
