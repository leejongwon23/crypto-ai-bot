# === train.py (FINAL, speed-tuned + SSL cache + CPU thread cap + Lightning Trainer + predict timeout) ===
# ✅ CPU 스레드 상한(기본 2). 기존 환경변수 설정이 있으면 그대로 둠.
import os
def _set_default_thread_env(name: str, val: int):
    if os.getenv(name) is None:
        os.environ[name] = str(val)
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
           "TORCH_NUM_THREADS"):
    _set_default_thread_env(_n, int(os.getenv("CPU_THREAD_CAP", "2")))

import json, time, traceback, tempfile, io, errno, glob
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
import shutil
import gc
import threading

# ✅ 무손실 모델 압축 유틸
from model_io import convert_pt_to_ptz, save_model

# ✅ torch 내부 스레드도 제한
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
except Exception:
    pass

# (선택) Lightning 사용: 설치 안 되어 있으면 폴백 (+ 환경변수로 비활성 가능)
_DISABLE_LIGHTNING = os.getenv("DISABLE_LIGHTNING", "0") == "1"
_HAS_LIGHTNING = False
if not _DISABLE_LIGHTNING:
    try:
        import pytorch_lightning as pl
        _HAS_LIGHTNING = True
    except Exception:
        _HAS_LIGHTNING = False

# ⬇️ 불필요한 SYMBOLS/SYMBOLS_GROUPS 의존 제거
from data.utils import get_kline_by_strategy, compute_features, create_dataset, SYMBOL_GROUPS

from model.base_model import get_model
from feature_importance import compute_feature_importance, save_feature_importance  # 호환용
from failure_db import insert_failure_record, ensure_failure_db
import logger  # log_* 및 ensure_prediction_log_exists 사용
from config import (
    get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups,
    get_class_ranges, set_NUM_CLASSES, STRATEGY_CONFIG
)
from data_augmentation import balance_classes

# --- window_optimizer ---
from window_optimizer import find_best_window

# --- ssl_pretrain (옵션) ---
try:
    from ssl_pretrain import masked_reconstruction, get_ssl_ckpt_path
except Exception:
    def masked_reconstruction(symbol, strategy, input_size):
        return None
    def get_ssl_ckpt_path(symbol: str, strategy: str) -> str:
        base = os.getenv("SSL_CACHE_DIR", "/persistent/ssl_models")
        os.makedirs(base, exist_ok=True)
        return f"{base}/{symbol}_{strategy}_ssl.pt"

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
        import calibration
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
            import failure_learn
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
    logger.log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0, loss=0.0,
                               note=reason, status="failed")
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

# ===== 저장/별칭/아카이브 =====
def _stem(path: str) -> str:
    return os.path.splitext(path)[0]

def _save_model_and_meta(model: nn.Module, path_pt: str, meta: dict):
    stem = _stem(path_pt)
    weight_path = stem + ".ptz"
    save_model(weight_path, model.state_dict())
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    _atomic_write(stem + ".meta.json", meta_json, mode="w")
    return weight_path, (stem + ".meta.json")

def _safe_alias(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except Exception:
        pass
    try:
        os.link(src, dst)
    except Exception:
        shutil.copyfile(src, dst)

def _emit_aliases(model_path: str, meta_path: str, symbol: str, strategy: str, model_type: str):
    ext = os.path.splitext(model_path)[1]  # .ptz
    flat_pt   = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}{ext}")
    flat_meta = _stem(flat_pt) + ".meta.json"
    _safe_alias(model_path, flat_pt)
    _safe_alias(meta_path, flat_meta)
    dir_pt   = os.path.join(MODEL_DIR, symbol, strategy, f"{model_type}{ext}")
    dir_meta = _stem(dir_pt) + ".meta.json"
    _safe_alias(model_path, dir_pt)
    _safe_alias(meta_path, dir_meta)

def _archive_old_checkpoints(symbol: str, strategy: str, model_type: str, keep_n: int = 1):
    patt_pt  = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group*_cls*.pt")
    patt_ptz = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group*_cls*.ptz")
    paths = sorted(glob.glob(patt_pt) + glob.glob(patt_ptz), key=lambda p: os.path.getmtime(p), reverse=True)
    if not paths:
        return
    survivors = set(paths[:max(1, int(keep_n))])
    for p in paths[max(1, int(keep_n)):]:
        try:
            if p.endswith(".pt"):
                ptz = os.path.splitext(p)[0] + ".ptz"
                if not os.path.exists(ptz):
                    convert_pt_to_ptz(p, ptz)
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            print(f"[ARCHIVE] {os.path.basename(p)} 압축 실패 → {e}")

# --------------------------------------------------
# Lightning 래퍼
# --------------------------------------------------
if _HAS_LIGHTNING:
    class LitSeqModel(pl.LightningModule):
        def __init__(self, base_model: nn.Module, lr: float = 1e-3):
            super().__init__()
            self.model = base_model
            self.criterion = nn.CrossEntropyLoss()
            self.lr = lr

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            xb, yb = batch
            logits = self(xb)
            loss = self.criterion(logits, yb)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

# --------------------------------------------------
# 단일 모델 학습
# --------------------------------------------------
def train_one_model(symbol, strategy, group_id=None, max_epochs=12):
    result = {
        "symbol": symbol, "strategy": strategy, "group_id": int(group_id or 0),
        "models": []
    }
    try:
        print(f"✅ [train_one_model 시작] {symbol}-{strategy}-group{group_id}", flush=True)
        ensure_failure_db()

        # ✅ SSL 사전학습 캐시 스킵
        try:
            ssl_ckpt = get_ssl_ckpt_path(symbol, strategy)
            if not os.path.exists(ssl_ckpt):
                masked_reconstruction(symbol, strategy, FEATURE_INPUT_SIZE)
            else:
                print(f"[SSL] cache found → skip: {ssl_ckpt}", flush=True)
        except Exception as e:
            print(f"[⚠️ SSL 사전학습 실패] {e}", flush=True)

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            _log_skip(symbol, strategy, "데이터 없음"); return result

        try:
            cfg = STRATEGY_CONFIG.get(strategy, {})
            _limit = int(cfg.get("limit", 300))
        except Exception:
            _limit = 300
        _min_required = max(60, int(_limit * 0.90))

        _attrs = getattr(df, "attrs", {}) if df is not None else {}
        augment_needed = bool(_attrs.get("augment_needed", len(df) < _limit))
        enough_for_training = bool(_attrs.get("enough_for_training", len(df) >= _min_required))
        print(f"[DATA] {symbol}-{strategy} rows={len(df)} limit={_limit} "
              f"min_required={_min_required} augment_needed={augment_needed} "
              f"enough_for_training={enough_for_training}", flush=True)

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
            print(f"[📏 클래스경계 로그] {symbol}-{strategy}-g{group_id} → {class_ranges}", flush=True)
        except Exception as e:
            print(f"[⚠️ log_class_ranges 실패/미구현] {e}", flush=True)

        # 2) 미래 수익률 + 요약 로그
        horizon_hours = _strategy_horizon_hours(strategy)
        future_gains = _future_returns_by_timestamp(df, horizon_hours=horizon_hours)
        try:
            fg = future_gains[np.isfinite(future_gains)]
            if fg.size > 0:
                q = np.nanpercentile(fg, [0,25,50,75,90,95,99])
                print(f"[📈 수익률분포] {symbol}-{strategy}-g{group_id} "
                      f"min={q[0]:.4f}, p25={q[1]:.4f}, p50={q[2]:.4f}, p75={q[3]:.4f}, "
                      f"p90={q[4]:.4f}, p95={q[5]:.4f}, p99={q[6]:.4f}, max={np.nanmax(fg):.4f}", flush=True)
                try:
                    logger.log_return_distribution(symbol, strategy, group_id=group_id,
                        horizon_hours=int(horizon_hours),
                        summary={"min":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),
                                 "p75":float(q[3]),"p90":float(q[4]),"p95":float(q[5]),
                                 "p99":float(q[6]),"max":float(np.nanmax(fg)),"count":int(fg.size)},
                        note="train_one_model")
                except Exception as le:
                    print(f"[⚠️ log_return_distribution 실패/미구현] {le}", flush=True)
        except Exception as e:
            print(f"[⚠️ 수익률분포 요약 실패] {e}", flush=True)

        # 3) 라벨링 + 분포 로그  ── ★ 경계 이탈 보정(클리핑)
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
                  f"low_clip={clipped_low}, high_clip={clipped_high}, unmatched={unmatched}", flush=True)

        labels = np.array(labels, dtype=np.int64)

        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)

        # ✅ 속도 개선: 최근 구간만 사용
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
                  f"classes={len(label_counts)}, H={entropy:.4f}", flush=True)
        except Exception as e:
            print(f"[⚠️ log_label_distribution 실패/미구현] {e}", flush=True)

        # 데이터 부족시 fallback
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
                print(f"[⚠️ fallback 실패] {e}", flush=True)

        if len(X) < 10:
            _log_skip(symbol, strategy, f"최종 샘플 부족 (rows={len(df)}, limit={_limit}, min_required={_min_required})")
            return result

        try:
            if len(X) < 200:
                X, y = balance_classes(X, y, num_classes=num_classes)
        except Exception as e:
            print(f"[⚠️ 밸런싱 실패] {e}", flush=True)

        # 6) 학습/평가/저장
        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            base_model = get_model(model_type, input_size=FEATURE_INPUT_SIZE, output_size=num_classes).to(DEVICE)

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

            if _HAS_LIGHTNING:
                lit = LitSeqModel(base_model, lr=1e-3)
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    enable_checkpointing=False,
                    logger=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )
                trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
                model = lit.model.to(DEVICE)
            else:
                model = base_model
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(max_epochs):
                    model.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        if not torch.isfinite(loss): continue
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        total_loss += float(loss.item())

            # 검증
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = torch.argmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
                    all_preds.extend(preds); all_labels.extend(yb.numpy())
            acc = float(accuracy_score(all_labels, all_preds))
            f1 = float(f1_score(all_labels, all_preds, average="macro"))

            # 저장
            base_stem = os.path.join(
                MODEL_DIR,
                f"{symbol}_{strategy}_{model_type}_group{int(group_id) if group_id is not None else 0}_cls{int(num_classes)}"
            )
            meta = {
                "symbol": symbol, "strategy": strategy, "model": model_type,
                "group_id": int(group_id) if group_id is not None else 0,
                "num_classes": int(num_classes), "input_size": int(FEATURE_INPUT_SIZE),
                "metrics": {"val_acc": acc, "val_f1": f1, "train_loss_sum": float(total_loss)},
                "timestamp": now_kst().isoformat(), "model_name": os.path.basename(base_stem) + ".ptz",
                "window": int(window), "recent_cap": int(len(feat_scaled)),
                "engine": "lightning" if _HAS_LIGHTNING else "manual",
                "data_flags": {
                    "rows": int(len(df)), "limit": int(_limit), "min_required": int(_min_required),
                    "augment_needed": bool(augment_needed), "enough_for_training": bool(enough_for_training)
                }
            }
            weight_path, meta_path = _save_model_and_meta(model, base_stem + ".pt", meta)
            _archive_old_checkpoints(symbol, strategy, model_type, keep_n=1)
            _emit_aliases(weight_path, meta_path, symbol, strategy, model_type)

            logger.log_training_result(
                symbol, strategy, model=os.path.basename(weight_path), accuracy=acc, f1=f1,
                loss=float(total_loss),
                note=(f"train_one_model(window={window}, cap={len(feat_scaled)}, "
                      f"engine={'lightning' if _HAS_LIGHTNING else 'manual'}, "
                      f"data_flags={{rows:{len(df)},limit:{_limit},min:{_min_required},"
                      f"aug:{int(augment_needed)},enough:{int(enough_for_training)}}})"),
                source_exchange="BYBIT", status="success"
            )
            result["models"].append({
                "type": model_type, "acc": acc, "f1": f1,
                "loss_sum": float(total_loss), "pt": weight_path,
                "meta": meta_path
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    except Exception as e:
        _log_fail(symbol, strategy, str(e))
        return result

# --------------------------------------------------
# 경량 정리 유틸 — 그룹 종료 시 호출
# --------------------------------------------------
def _prune_caches_and_gc():
    try:
        from cache import CacheManager as _CM
        try:
            before = _CM.stats()
        except Exception:
            before = None
        pruned = _CM.prune()
        try:
            after = _CM.stats()
        except Exception:
            after = None
        print(f"[CACHE] prune ok: before={before}, after={after}, pruned={pruned}", flush=True)
    except Exception as e:
        print(f"[CACHE] 모듈 없음/스킵 ({e})", flush=True)
    try:
        from safe_cleanup import trigger_light_cleanup
        trigger_light_cleanup()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

# --------------------------------------------------
# 그룹 배열 회전: BTCUSDT 그룹을 맨 앞으로
# --------------------------------------------------
def _rotate_groups_starting_with(groups, anchor_symbol="BTCUSDT"):
    norm = [list(g) for g in groups]
    anchor_gid = None
    for i, g in enumerate(norm):
        if anchor_symbol in g:
            anchor_gid = i
            break
    if anchor_gid is not None and anchor_gid != 0:
        norm = norm[anchor_gid:] + norm[:anchor_gid]
    if norm and anchor_symbol in norm[0]:
        norm[0] = [anchor_symbol] + [s for s in norm[0] if s != anchor_symbol]
    return norm

# --------------------------------------------------
# 🔒 예측 타임아웃 래퍼 — 예측 단계 블로킹 방지 (기본 30초)
# --------------------------------------------------
_PREDICT_TIMEOUT_SEC = float(os.getenv("PREDICT_TIMEOUT_SEC", "30"))

def _safe_predict_with_timeout(predict_fn, symbol, strategy, source, model_type=None, timeout=_PREDICT_TIMEOUT_SEC):
    err = []
    done = threading.Event()
    def _run():
        try:
            predict_fn(symbol, strategy, source=source, model_type=model_type)
        except Exception as e:
            err.append(e)
        finally:
            done.set()
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    finished = done.wait(timeout)
    if not finished:
        print(f"[⏱️ 예측 타임아웃] {symbol}-{strategy} ({timeout}s) → 스킵", flush=True)
        return False
    if err:
        print(f"[⚠️ 예측 실패] {symbol}-{strategy}: {err[0]}", flush=True)
        return False
    return True

# --------------------------------------------------
# 전체 학습 루틴  (✅ stop_event 지원)
# --------------------------------------------------
def train_models(symbol_list, stop_event: threading.Event | None = None):
    strategies = ["단기", "중기", "장기"]
    for symbol in symbol_list:
        if stop_event is not None and stop_event.is_set():
            print("[STOP] train_models: stop_event 감지 → 조기 종료", flush=True); return
        for strategy in strategies:
            if stop_event is not None and stop_event.is_set():
                print("[STOP] train_models: stop_event 감지(strategy loop) → 조기 종료", flush=True); return
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
                if stop_event is not None and stop_event.is_set():
                    print("[STOP] train_models: stop_event 감지(group loop) → 조기 종료", flush=True); return
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
                if stop_event is not None and stop_event.is_set():
                    print("[STOP] train_models: stop_event 감지(after one model) → 조기 종료", flush=True); return
                time.sleep(0.5)

    try:
        import maintenance_fix_meta
        maintenance_fix_meta.fix_all_meta_json()
    except Exception as e:
        print(f"[⚠️ meta 보정 실패] {e}", flush=True)

    try:
        import failure_trainer
        failure_trainer.run_failure_training()
    except Exception as e:
        print(f"[⚠️ 실패학습 루프 예외] {e}", flush=True)

    try:
        train_evo_meta_loop()
    except Exception as e:
        print(f"[⚠️ 진화형 메타러너 학습 실패] {e}", flush=True)

# --------------------------------------------------
# 그룹 루프(그룹 완료 후 예측 1회)  (✅ stop_event 지원 + 예측 타임아웃)
# --------------------------------------------------
def train_symbol_group_loop(sleep_sec: int = 0, stop_event: threading.Event | None = None):
    try:
        from predict import predict  # evaluate 호출 없음

        # 로그 파일/헤더 보장(존재 시만)
        try:
            if hasattr(logger, "ensure_train_log_exists"):
                logger.ensure_train_log_exists()
        except Exception:
            pass
        try:
            if hasattr(logger, "ensure_prediction_log_exists"):
                logger.ensure_prediction_log_exists()
        except Exception:
            pass

        # 원본 그룹 → BTCUSDT 그룹을 맨 앞으로 회전
        groups = _rotate_groups_starting_with(SYMBOL_GROUPS, anchor_symbol="BTCUSDT")

        for idx, group in enumerate(groups):
            # ⛔️ 새 그룹에 들어가기 전에만 stop 체크
            if stop_event is not None and stop_event.is_set():
                print("[STOP] train_symbol_group_loop: stop_event 감지(다음 그룹 진입 전) → 종료", flush=True); break

            print(f"🚀 [train_symbol_group_loop] 그룹 #{idx+1}/{len(groups)} → {group} | mode=per_symbol_all_horizons", flush=True)

            # 1) 그룹 학습
            train_models(group, stop_event=stop_event)

            # stop이면 예측 생략 후 종료
            if stop_event is not None and stop_event.is_set():
                print("🛑 stop 요청 반영 → 그룹 학습 직후 즉시 종료(예측 생략)", flush=True)
                break

            # ✅ 모델 저장 직후 I/O 안정화
            time.sleep(0.2)

            # 2) 그룹 학습 완료 후 단 한 번씩 **예측만** 수행 (타임아웃 보호)
            for symbol in group:
                for strategy in ["단기", "중기", "장기"]:
                    _safe_predict_with_timeout(predict, symbol, strategy, source="그룹직후", model_type=None)

            # 3) 그룹 종료 정리
            _prune_caches_and_gc()

            if sleep_sec > 0:
                for _ in range(sleep_sec):
                    if stop_event is not None and stop_event.is_set():
                        print("[STOP] train_symbol_group_loop: stop_event 감지(sleep) → 종료", flush=True); break
                    time.sleep(1)
                if stop_event is not None and stop_event.is_set():
                    break

        print("✅ train_symbol_group_loop 완료", flush=True)
    except Exception as e:
        print(f"[❌ train_symbol_group_loop 예외] {e}", flush=True)

# --------------------------------------------------
# ✅ 루프 제어 유틸: 중복 방지용 (단일 루프 보장)
# --------------------------------------------------
_TRAIN_LOOP_THREAD: threading.Thread | None = None
_TRAIN_LOOP_STOP: threading.Event | None = None
_TRAIN_LOOP_LOCK = threading.Lock()

def start_train_loop(force_restart: bool = False, sleep_sec: int = 0):
    """학습 루프를 1개만 실행. force_restart=True면 기존 루프를 먼저 정지."""
    global _TRAIN_LOOP_THREAD, _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive():
            if not force_restart:
                print("ℹ️ start_train_loop: 기존 루프가 실행 중 → 재시작 생략", flush=True); return False
            print("🛑 start_train_loop: 기존 루프 정지 시도", flush=True)
            stop_train_loop(timeout=30)

        _TRAIN_LOOP_STOP = threading.Event()
        def _runner():
            try:
                train_symbol_group_loop(sleep_sec=sleep_sec, stop_event=_TRAIN_LOOP_STOP)
            finally:
                print("ℹ️ train loop thread 종료", flush=True)
        _TRAIN_LOOP_THREAD = threading.Thread(target=_runner, daemon=True)
        _TRAIN_LOOP_THREAD.start()
        print("✅ train loop 시작됨 (단일 인스턴스 보장)", flush=True)
        return True

def stop_train_loop(timeout: int | float | None = 30):
    """실행 중 루프를 안전하게 중단 요청하고 대기."""
    global _TRAIN_LOOP_THREAD, _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is None or not _TRAIN_LOOP_THREAD.is_alive():
            print("ℹ️ stop_train_loop: 실행 중인 루프 없음", flush=True); return True
        if _TRAIN_LOOP_STOP is None:
            print("⚠️ stop_train_loop: stop_event 없음(비정상 상태)", flush=True); return False
        _TRAIN_LOOP_STOP.set()
        _TRAIN_LOOP_THREAD.join(timeout=timeout)
        if _TRAIN_LOOP_THREAD.is_alive():
            print("⚠️ stop_train_loop: 타임아웃 — 여전히 실행 중", flush=True)
            return False
        _TRAIN_LOOP_THREAD = None
        _TRAIN_LOOP_STOP = None
        print("✅ stop_train_loop: 정상 종료", flush=True)
        return True

# ✅ app.py의 reset-all이 즉시 사용할 수 있도록 보조 API
def request_stop() -> bool:
    """외부에서 중단 신호만 보내고 즉시 반환(폴링은 호출측에서)."""
    global _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_STOP is None:
            return True
        _TRAIN_LOOP_STOP.set()
        return True

def is_loop_running() -> bool:
    """학습 루프 스레드가 살아있는지 반환."""
    with _TRAIN_LOOP_LOCK:
        return bool(_TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive())

if __name__ == "__main__":
    try:
        start_train_loop(force_restart=True, sleep_sec=0)
    except Exception as e:
        print(f"[MAIN] 예외: {e}", flush=True)
