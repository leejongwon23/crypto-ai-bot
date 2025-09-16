# ssl_pretrain.py (PATCHED: input-dim autosync + cache verify + cooperative stop + import fallbacks)
import os
import json
import time
import threading
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# 안전한 import: model.base_model, data.utils, config
try:
    from model.base_model import TransformerPricePredictor
except Exception:
    TransformerPricePredictor = None

try:
    from data.utils import get_kline_by_strategy, compute_features
except Exception:
    def get_kline_by_strategy(symbol, strategy):
        return None
    def compute_features(symbol, df, strategy):
        return None

try:
    from config import get_FEATURE_INPUT_SIZE, get_SSL_CACHE_DIR
except Exception:
    def get_FEATURE_INPUT_SIZE():
        try:
            return int(os.getenv("FEATURE_INPUT_SIZE", "0"))
        except Exception:
            return 0
    def get_SSL_CACHE_DIR():
        return os.getenv("SSL_CACHE_DIR", "/persistent/ssl_cache")

# DEVICE selection safe-guard
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

def get_ssl_ckpt_path(symbol: str, strategy: str) -> str:
    """SSL 체크포인트 표준 경로(메타는 .meta.json에 저장)"""
    ssl_dir = get_SSL_CACHE_DIR() or "/persistent/ssl_cache"
    os.makedirs(ssl_dir, exist_ok=True)
    base = os.path.join(ssl_dir, f"{symbol}_{strategy}_ssl.pt")
    return base

def _meta_path(ckpt_path: str) -> str:
    return ckpt_path + ".meta.json"

def _load_meta(ckpt_path: str) -> dict | None:
    mp = _meta_path(ckpt_path)
    if not os.path.exists(mp):
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_meta(ckpt_path: str, **meta):
    mp = _meta_path(ckpt_path)
    try:
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass

def _check_stop(ev: threading.Event | None, where: str = ""):
    if ev is not None and ev.is_set():
        print(f"[SSL] stop detected → {where}")
        raise KeyboardInterrupt  # 상위에서 조용히 중단

@torch.no_grad() if torch is not None else (lambda f: f)
def _to_tensor_3d(feat_df) -> "torch.Tensor | None":
    """(1, T, F) 텐서 변환"""
    if feat_df is None:
        return None
    try:
        X = feat_df.drop(columns=["timestamp", "strategy"], errors="ignore").to_numpy(dtype=np.float32)
        X = np.expand_dims(X, axis=0)  # (1, T, F)
        if torch is None:
            return X
        return torch.tensor(X, dtype=torch.float32, device=DEVICE)
    except Exception:
        return None

def masked_reconstruction(
    symbol: str,
    strategy: str,
    input_size: int | None = None,
    mask_ratio: float = 0.2,
    epochs: int = 10,
    min_rows: int = 50,
    stop_event: threading.Event | None = None,
    max_seconds: float | None = None,
):
    """
    마스크드 재구성 사전학습.
    - 입력차원 자동 동기화: feature F를 감지해 model 입/출력 크기로 사용
    - 캐시 검증: 저장된 메타(input_size)가 현재 F와 같을 때만 스킵
    - 협조적 중단 및 시간 제한 지원
    """
    start_ts = time.time()
    try:
        _check_stop(stop_event, "entry")

        # --- 데이터 준비
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < min_rows:
            print(f"[SSL] 데이터 부족 → 스킵 (rows={len(df) if df is not None else 0}, 필요={min_rows})")
            return None

        _check_stop(stop_event, "before features")
        feat = compute_features(symbol, df, strategy)
        if feat is None or getattr(feat, "empty", False) or (hasattr(feat, "isnull") and feat.isnull().any().any()):
            print("[SSL] feature 생성 실패 또는 NaN 포함 → 스킵")
            return None

        X_tensor = _to_tensor_3d(feat)  # (1, T, F) or numpy if torch missing
        if X_tensor is None:
            print("[SSL] 텐서 생성 실패 → 스킵")
            return None

        # handle numpy fallback
        if torch is None:
            # cannot train without torch
            print("[SSL] torch 미설치 → 사전학습 불가능")
            return None

        T, F = int(X_tensor.shape[1]), int(X_tensor.shape[2])

        # --- 캐시 확인(입력차원 일치 시에만 스킵)
        ckpt_path = get_ssl_ckpt_path(symbol, strategy)
        meta = _load_meta(ckpt_path)
        if os.path.exists(ckpt_path) and meta and int(meta.get("input_size", -1)) == F:
            print(f"[SSL] cache found & dim matched(F={F}) → skip: {ckpt_path}")
            return ckpt_path

        # --- 입력차원 자동 동기화
        cfg_F = int(input_size or get_FEATURE_INPUT_SIZE() or 0)
        if cfg_F != F:
            print(f"[SSL] input_size 자동 동기화: config={cfg_F} → features={F}")
        input_size = F  # 강제 동기화
        output_size = F

        print(f"[SSL] {symbol}-{strategy} pretraining 시작 (T={T}, F={F})")

        # --- 모델/학습
        if TransformerPricePredictor is None:
            print("[SSL] TransformerPricePredictor 미발견 → 사전학습 불가")
            return None

        try:
            model = TransformerPricePredictor(
                input_size=input_size,
                output_size=output_size,
                mode="reconstruction"
            ).to(DEVICE)
        except Exception as e:
            print(f"[SSL] 모델 생성 실패: {e}")
            return None

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lossfn = nn.MSELoss()

        # 마스크 개수: 최대 절반
        num_mask = max(1, int(T * mask_ratio))
        num_mask = min(num_mask, max(1, T // 2))

        for epoch in range(epochs):
            _check_stop(stop_event, f"epoch {epoch}")
            if max_seconds is not None and (time.time() - start_ts) > max_seconds:
                print(f"[SSL] time limit reached ({max_seconds}s) → stop")
                return None

            X_masked = X_tensor.clone()
            mask_idx = np.random.choice(T, num_mask, replace=False)
            X_masked[:, mask_idx, :] = 0

            pred = model(X_masked)
            # 모양 보정(안전장치)
            if pred.shape != X_tensor.shape:
                min_len = min(pred.shape[1], X_tensor.shape[1])
                pred = pred[:, :min_len, :]
                target = X_tensor[:, :min_len, :]
            else:
                target = X_tensor

            loss = lossfn(pred, target)
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                # 살짝 안정화
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            print(f"[SSL] epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")

        # --- 저장(메타 포함)
        try:
            dirp = os.path.dirname(ckpt_path)
            if dirp:
                os.makedirs(dirp, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            _save_meta(ckpt_path, input_size=input_size, feature_dim=F, timestamp=time.time())
            print(f"[SSL] {symbol}-{strategy} pretraining 완료 → 저장: {ckpt_path} (F={F})")
            return ckpt_path
        except Exception as e:
            print(f"[SSL] 저장 실패: {e}")
            return None

    except KeyboardInterrupt:
        print("[SSL] canceled by stop signal")
        return None
    except Exception as e:
        print(f"[SSL] 예외 발생: {e}")
        return None
