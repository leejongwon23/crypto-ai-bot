# === train.py (FINAL, speed-tuned: recent rows cap + lean DataLoader + narrow window search) ===
import os, json, time, traceback, tempfile, io, errno
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ⬇️ 불필요한 SYMBOLS/SYMBOLS_GROUPS 의존 제거
from data.utils import get_kline_by_strategy, compute_features, create_dataset, SYMBOL_GROUPS

from model.base_model import get_model
from feature_importance import compute_feature_importance, save_feature_importance  # 호환용
from failure_db import insert_failure_record, ensure_failure_db
import logger  # log_* 및 ensure_prediction_log_exists 사용
from config import (
    get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups,
    get_class_ranges, set_NUM_CLASSES  # ⬅️ get_SYMBOL_GROUPS 제거
)
from data_augmentation import balance_classes

# --- window_optimizer ---
from window_optimizer import find_best_window

# --- ssl_pretrain (옵션) ---
try:
    from ssl_pretrain import masked_reconstruction
except Exception:
    def masked_reconstruction(symbol, strategy, input_size):
        return None

# --- evo meta learner (옵션) ---
try:
    from evo_meta_learner import train_evo_meta_loop
except Exception:
    def train_evo_meta_loop(*args, **kwargs):
        return None

# === (6번) 자동 후처리 훅: 학습 직후 캘리브레이션/실패학습 ===
def _safe_print(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _try_auto_calibration(symbol, strategy, model_name):
    try:
        import calibration  # 4번에서 추가되는 파일(없으면 스킵)
    except Exception as e:
        _safe_print(f"[CALIB] 모듈 없음/로드 실패 → 스킵 ({e})")
        return
    for fn_name in ("learn_and_save_from_checkpoint", "learn_and_save"):
        try:
            fn = getattr(calibration, fn_name, None)
            if callable(fn):
                fn(symbol=symbol, strategy=strategy, model_name=model_name)
                _safe_print(f"[CALIB] {symbol}-{strategy}-{model_name} → {fn_name} 실행")
                return
        except Exception as ce:
            _safe_print(f"[CALIB] {fn_name} 예외 → {ce}")
    _safe_print("[CALIB] 사용가능한 API 없음 → 스킵")

try:
    _orig_log_training_result = logger.log_training_result
    def _wrapped_log_training_result(symbol, strategy, model="", accuracy=0.0, f1=0.0, loss=0.0,
                                     note="", source_exchange="BYBIT", status="success"):
        try:
            _orig_log_training_result(symbol, strategy, model, accuracy, f1, loss, note, source_exchange, status)
        finally:
            try:
                _try_auto_calibration(symbol, strategy, model or "")
            except Exception as e:
                _safe_print(f"[HOOK] 캘리브레이션 훅 예외 → {e}")
    logger.log_training_result = _wrapped_log_training_result
    _safe_print("[HOOK] logger.log_training_result → 캘리 훅 장착 완료")
except Exception as _e:
    _safe_print(f"[HOOK] 장착 실패(원본 미탐) → {_e}")

def _maybe_run_failure_learn(background=True):
    import threading
    def _job():
        try:
            import failure_learn  # 7번 단계(없으면 스킵)
        except Exception as e:
            _safe_print(f"[FAIL-LEARN] 모듈 없음/로드 실패 → 스킵 ({e})")
            return
        for name in ("mini_retrain", "run_once", "run"):
            try:
                fn = getattr(failure_learn, name, None)
                if callable(fn):
                    fn()
                    _safe_print(f"[FAIL-LEARN] {name} 실행 완료")
                    return
            except Exception as e:
                _safe_print(f"[FAIL-LEARN] {name} 예외 → {e}")
        _safe_print("[FAIL-LEARN] 실행 가능한 API 없음 → 스킵")
    if background:
        threading.Thread(target=_job, daemon=True).start()
    else:
        _job()

# 초기 1회 조용히 시도(있으면 수행/없으면 스킵)
try:
    _maybe_run_failure_learn(background=True)
except Exception as _e:
    _safe_print(f"[FAIL-LEARN] 초기 시도 예외 → {_e}")
# === 자동 후처리 훅 끝 ===

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "/persistent/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ 학습 데이터 최근 구간 상한 (정확도 영향 최소 / 속도 향상)
_MAX_ROWS_FOR_TRAIN = int(os.getenv("TRAIN_MAX_ROWS", "1200"))

# ✅ DataLoader 튜닝(안전): CPU 기준
_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "128"))
_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", "0"))   # CPU 경량 파이프라인: 0 권장
_PIN_MEMORY = False
_PERSISTENT = False

now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))
training_in_progress = {"단기": False, "중기": False, "장기": False}

# --------------------------------------------------
# 유틸
# --------------------------------------------------
def _atomic_write(path: str, bytes_or_str, mode: str = "wb"):
    """쓰기 실패/중단 대비 원자적 저장."""
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(dir=dirpath, prefix=".tmp_", suffix=".swap")
    try:
        with os.fdopen(fd, mode) as f:
            if "b" in mode:
                data = bytes_or_str if isinstance(bytes_or_str, (bytes, bytearray)) else bytes_or_str.encode("utf-8")
                f.write(data)
            else:
                f.write(bytes_or_str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmppath, path)
    finally:
        try:
            if os.path.exists(tmppath):
                os.remove(tmppath)
        except Exception:
            pass

def _log_skip(symbol, strategy, reason):
    logger.log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                               loss=0.0, note=reason, status="skipped")
    insert_failure_record({
        "symbol": symbol, "strategy": strategy, "model": "all",
        "predicted_class": -1, "success": False, "rate": 0.0,
        "reason": reason
    }, feature_vector=[])

def _log_fail(symbol, strategy, reason):
    logger.log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                               loss=0.0, note=reason, status="failed")
    insert_failure_record({
        "symbol": symbol, "strategy": strategy, "model": "all",
        "predicted_class": -1, "success": False, "rate": 0.0,
        "reason": reason
    }, feature_vector=[])

def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

def _future_returns_by_timestamp(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return np.zeros(0 if df is None else len(df), dtype=np.float32)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    close = df["close"].astype(float).values
    high = (df["high"] if "high" in df.columns else df["close"]).astype(float).values

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")

    out = np.zeros(len(df), dtype=np.float32)
    horizon = pd.Timedelta(hours=horizon_hours)

    j_start = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]; t1 = t0 + horizon
        j = max(j_start, i)
        max_h = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > max_h: max_h = high[j]
            j += 1
        j_start = max(j_start, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((max_h - base) / (base + 1e-12))
    return out.astype(np.float32)

def _save_model_and_meta(model: nn.Module, path_pt: str, meta: dict):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    _atomic_write(path_pt, buffer.getvalue(), mode="wb")
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    _atomic_write(path_pt.replace(".pt", ".meta.json"), meta_json, mode="w")

# --------------------------------------------------
# 단일 모델 학습
# --------------------------------------------------
def train_one_model(symbol, strategy, group_id=None, max_epochs=12):
    result = {
        "symbol": symbol, "strategy": strategy, "group_id": int(group_id or 0),
        "models": []
    }
    try:
        print(f"✅ [train_one_model 시작] {symbol}-{strategy}-group{group_id}")
        ensure_failure_db()

        try:
            masked_reconstruction(symbol, strategy, FEATURE_INPUT_SIZE)
        except Exception as e:
            print(f"[⚠️ SSL 사전학습 실패] {e}")

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            _log_skip(symbol, strategy, "데이터 없음"); return result

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.empty or feat.isnull().any().any():
            _log_skip(symbol, strategy, "피처 없음"); return result

        # 1) 동적 클래스 경계
        try:
            class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
        except Exception as e:
            _log_fail(symbol, strategy, "클래스 계산 실패"); return result

        num_classes = len(class_ranges)
        set_NUM_CLASSES(num_classes)

        if not class_ranges or len(class_ranges) < 2:
            try:
                logger.log_class_ranges(symbol, strategy, group_id=group_id,
                                        class_ranges=class_ranges or [], note="train_skip(<2 classes)")
                logger.log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0, loss=0.0,
                                           note=f"스킵: group_id={group_id}, 클래스<2", status="skipped")
            except Exception:
                pass
            return result

        # 경계 로그
        try:
            logger.log_class_ranges(symbol, strategy, group_id=group_id,
                                    class_ranges=class_ranges, note="train_one_model")
            print(f"[📏 클래스경계 로그] {symbol}-{strategy}-g{group_id} → {class_ranges}")
        except Exception as e:
            print(f"[⚠️ log_class_ranges 실패/미구현] {e}")

        # 2) 미래 수익률 + 요약 로그
        horizon_hours = _strategy_horizon_hours(strategy)
        future_gains = _future_returns_by_timestamp(df, horizon_hours=horizon_hours)
        try:
            fg = future_gains[np.isfinite(future_gains)]
            if fg.size > 0:
                q = np.nanpercentile(fg, [0,25,50,75,90,95,99])
                print(f"[📈 수익률분포] {symbol}-{strategy}-g{group_id} "
                      f"min={q[0]:.4f}, p25={q[1]:.4f}, p50={q[2]:.4f}, p75={q[3]:.4f}, "
                      f"p90={q[4]:.4f}, p95={q[5]:.4f}, p99={q[6]:.4f}, max={np.nanmax(fg):.4f}")
                try:
                    logger.log_return_distribution(symbol, strategy, group_id=group_id,
                        horizon_hours=int(horizon_hours),
                        summary={"min":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),
                                 "p75":float(q[3]),"p90":float(q[4]),"p95":float(q[5]),
                                 "p99":float(q[6]),"max":float(np.nanmax(fg)),"count":int(fg.size)},
                        note="train_one_model")
                except Exception as le:
                    print(f"[⚠️ log_return_distribution 실패/미구현] {le}")
        except Exception as e:
            print(f"[⚠️ 수익률분포 요약 실패] {e}")

        # 3) 라벨링 + 분포 로그  ── ★ 경계 이탈 보정(클리핑) 추가
        labels = []
        clipped_low, clipped_high, unmatched = 0, 0, 0

        lo0 = class_ranges[0][0]
        hi_last = class_ranges[-1][1]

        for r in future_gains:
            if not np.isfinite(r):
                r = lo0
            if r < lo0:
                labels.append(0); clipped_low += 1; continue
            if r > hi_last:
                labels.append(len(class_ranges) - 1); clipped_high += 1; continue
            idx = None
            for i, (lo, hi) in enumerate(class_ranges):
                if lo <= r <= hi:
                    idx = i; break
            if idx is None:
                idx = len(class_ranges) - 1 if r > hi_last else 0
                unmatched += 1
            labels.append(idx)

        if clipped_low or clipped_high or unmatched:
            print(f"[🔧 라벨 보정] {symbol}-{strategy}-g{group_id} "
                  f"low_clip={clipped_low}, high_clip={clipped_high}, unmatched={unmatched}")

        labels = np.array(labels, dtype=np.int64)

        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)

        # ✅ 속도 개선: 최근 구간만 사용(라벨과 정렬 유지)
        if len(feat_scaled) > _MAX_ROWS_FOR_TRAIN or len(labels) > _MAX_ROWS_FOR_TRAIN:
            cut = min(_MAX_ROWS_FOR_TRAIN, len(feat_scaled), len(labels))
            feat_scaled = feat_scaled[-cut:]
            labels = labels[-cut:]

        # 4) 최적 윈도우(탐색 폭 축소)
        try:
            best_window = find_best_window(symbol, strategy, window_list=[20, 40], group_id=group_id)
        except Exception:
            best_window = 40
        window = int(max(5, best_window))
        # 🔒 데이터 길이를 넘지 않도록 클램프
        window = int(min(window, max(6, len(feat_scaled) - 1)))

        # 5) 시퀀스 생성
        X, y = [], []
        for i in range(len(feat_scaled) - window):
            X.append(feat_scaled[i:i+window])
            y_idx = i + window - 1
            y.append(labels[y_idx] if 0 <= y_idx < len(labels) else 0)
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

        # 라벨/분포 로그
        try:
            label_counts = Counter(y.tolist())
            total_labels = int(len(y))
            probs = np.array(list(label_counts.values()), dtype=np.float64)
            entropy = float(-(probs / max(1, probs.sum()) * np.log2((probs / max(1, probs.sum())) + 1e-12)).sum()) if probs.sum() > 0 else 0.0
            logger.log_label_distribution(symbol, strategy, group_id=group_id,
                                          counts=dict(label_counts), total=total_labels,
                                          n_unique=int(len(label_counts)), entropy=float(entropy),
                                          note=f"window={window}, recent_cap={len(feat_scaled)}")
            print(f"[🧮 라벨분포 로그] {symbol}-{strategy}-g{group_id} total={total_labels}, "
                  f"classes={len(label_counts)}, H={entropy:.4f}")
        except Exception as e:
            print(f"[⚠️ log_label_distribution 실패/미구현] {e}")

        if len(X) < 20:
            try:
                res = create_dataset(feat.to_dict(orient="records"), window=window, strategy=strategy, input_size=FEATURE_INPUT_SIZE)
                if isinstance(res, tuple) and len(res) >= 2:
                    X_fb, y_fb = res[0], res[1]
                else:
                    X_fb, y_fb = res
                if isinstance(X_fb, np.ndarray) and len(X_fb) > 0:
                    X, y = X_fb.astype(np.float32), y_fb.astype(np.int64)
            except Exception as e:
                print(f"[⚠️ fallback 실패] {e}")

        if len(X) < 10:
            _log_skip(symbol, strategy, "최종 샘플 부족"); return result

        try:
            if len(X) < 200:
                X, y = balance_classes(X, y, num_classes=num_classes)
        except Exception as e:
            print(f"[⚠️ 밸런싱 실패] {e}")

        # 6) 학습/평가/저장
        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size=FEATURE_INPUT_SIZE, output_size=num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            val_len = max(1, int(len(X) * 0.2))
            if len(X) - val_len < 1: val_len = len(X) - 1
            train_X, val_X = X[:-val_len], X[-val_len:]
            train_y, val_y = y[:-val_len], y[-val_len:]

            train_loader = DataLoader(
                TensorDataset(torch.tensor(train_X), torch.tensor(train_y)),
                batch_size=_BATCH_SIZE, shuffle=True,
                num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY, persistent_workers=_PERSISTENT
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(val_X), torch.tensor(val_y)),
                batch_size=_BATCH_SIZE,
                num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY, persistent_workers=_PERSISTENT
            )

            total_loss = 0.0
            for epoch in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    if not torch.isfinite(loss): continue
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total_loss += float(loss.item())

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = torch.argmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
                    all_preds.extend(preds); all_labels.extend(yb.numpy())
            acc = float(accuracy_score(all_labels, all_preds))
            f1 = float(f1_score(all_labels, all_preds, average="macro"))

            model_name = f"{symbol}_{strategy}_{model_type}_group{int(group_id) if group_id is not None else 0}_cls{int(num_classes)}.pt"
            model_path = os.path.join(MODEL_DIR, model_name)
            meta = {
                "symbol": symbol, "strategy": strategy, "model": model_type,
                "group_id": int(group_id) if group_id is not None else 0,
                "num_classes": int(num_classes), "input_size": int(FEATURE_INPUT_SIZE),
                "metrics": {"val_acc": acc, "val_f1": f1, "train_loss_sum": float(total_loss)},
                "timestamp": now_kst().isoformat(), "model_name": model_name,
                "window": int(window), "recent_cap": int(len(feat_scaled))
            }
            _save_model_and_meta(model, model_path, meta)

            logger.log_training_result(
                symbol, strategy, model=model_name, accuracy=acc, f1=f1,
                loss=float(total_loss), note=f"train_one_model(window={window}, cap={len(feat_scaled)})",
                source_exchange="BYBIT", status="success"
            )
            result["models"].append({
                "type": model_type, "acc": acc, "f1": f1,
                "loss_sum": float(total_loss), "pt": model_path,
                "meta": model_path.replace(".pt", ".meta.json")
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    except Exception as e:
        _log_fail(symbol, strategy, str(e))
        return result

# --------------------------------------------------
# 전체 학습 루틴
# --------------------------------------------------
def train_models(symbol_list):
    strategies = ["단기", "중기", "장기"]
    for symbol in symbol_list:
        for strategy in strategies:
            try:
                class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
                if not class_ranges:
                    raise ValueError("빈 클래스 경계")
                num_classes = len(class_ranges)
                groups = get_class_groups(num_classes=num_classes)
                max_gid = len(groups) - 1
            except Exception as e:
                _log_fail(symbol, strategy, f"클래스 계산 실패: {e}")
                continue

            for gid in range(max_gid + 1):
                # 각 그룹별 클래스 수 2개 미만이면 스킵
                try:
                    grp_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=gid)
                    if not grp_ranges or len(grp_ranges) < 2:
                        try:
                            logger.log_class_ranges(symbol, strategy, group_id=gid,
                                                    class_ranges=grp_ranges or [], note="train_skip(<2 classes)")
                            logger.log_training_result(symbol, strategy, model=f"group{gid}",
                                                       accuracy=0.0, f1=0.0, loss=0.0,
                                                       note=f"스킵: group_id={gid}, 클래스<2", status="skipped")
                        except Exception:
                            pass
                        continue
                except Exception as e:
                    try:
                        logger.log_training_result(symbol, strategy, model=f"group{gid}",
                                                   accuracy=0.0, f1=0.0, loss=0.0,
                                                   note=f"스킵: group_id={gid}, 경계계산실패 {e}", status="skipped")
                    except Exception:
                        pass
                    continue

                train_one_model(symbol, strategy, group_id=gid)
                time.sleep(0.5)

    try:
        import maintenance_fix_meta
        maintenance_fix_meta.fix_all_meta_json()
    except Exception as e:
        print(f"[⚠️ meta 보정 실패] {e}")

    try:
        import failure_trainer
        failure_trainer.run_failure_training()
    except Exception as e:
        print(f"[⚠️ 실패학습 루프 예외] {e}")

    try:
        train_evo_meta_loop()
    except Exception as e:
        print(f"[⚠️ 진화형 메타러너 학습 실패] {e}")

# --------------------------------------------------
# 그룹 루프(그룹 완료 후 예측 1회)
# --------------------------------------------------
def train_symbol_group_loop(sleep_sec: int = 0):
    try:
        from predict import predict  # 예측 함수 불러오기

        # ✅ 예측 로그 파일/헤더 보장
        try:
            logger.ensure_prediction_log_exists()
        except Exception as e:
            print(f"[경고] prediction_log 준비 실패: {e}")

        groups = SYMBOL_GROUPS  # ⬅️ 동적 호출 제거, data.utils 기준 고정 그룹 사용
        for idx, group in enumerate(groups):
            print(f"🚀 [train_symbol_group_loop] 그룹 #{idx+1}/{len(groups)} → {group} | mode=per_symbol_all_horizons")

            # 1) 그룹 학습 (심볼별 단→중→장 → 다음 심볼)
            train_models(group)

            # ✅ 모델 저장 직후 I/O 안정화
            time.sleep(0.2)

            # 2) 그룹 학습 완료 후 단 한 번씩 예측
            for symbol in group:
                for strategy in ["단기", "중기", "장기"]:
                    try:
                        print(f"🔮 [즉시예측] {symbol}-{strategy}")
                        predict(symbol, strategy, source="그룹직후", model_type=None)
                    except Exception as e:
                        print(f"[⚠️ 예측 실패] {symbol}-{strategy}: {e}")

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        print("✅ train_symbol_group_loop 완료")
    except Exception as e:
        print(f"[❌ train_symbol_group_loop 예외] {e}")
