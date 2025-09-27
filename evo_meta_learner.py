# evo_meta_learner.py (FINAL+GUARDS)
# (2025-09-27) — BTC 앵커 가드 + 기간 상호모순 가드 추가
# - predict_evo_meta(..., symbol=None, strategy=None, ...) 인자 확장(하위호환)
# - 앵커/모순 가드가 트리거되면 None 반환(상위 predict.py가 기본 경로로 진행)
# - 로그/CSV 안전 처리

import os
import json
import ast
import numpy as np
import pandas as pd

# torch optional safe import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

# ── optional: config에서 클래스 수익구간 가져오기(가드에 사용)
try:
    from config import get_class_return_range
except Exception:
    def get_class_return_range(cls_id: int, symbol: str = None, strategy: str = None):
        # 보수적 폴백: 양/음/중립 3클래스 가정
        if int(cls_id) == 0: return (-0.30, -0.05)
        if int(cls_id) == 1: return (-0.02, 0.02)
        return (0.05, 0.30)

MODEL_PATH = "/persistent/models/evo_meta_learner.pt"
META_PATH  = MODEL_PATH.replace(".pt", ".meta.json")
PRED_LOG   = "/persistent/prediction_log.csv"

# ── 앙상블 합성 규칙 (predict.py와 일관)
EVO_META_AGG = os.getenv("EVO_META_AGG", "mean_var").lower()   # mean | varpen | mean_var
EVO_META_VAR_GAMMA = float(os.getenv("EVO_META_VAR_GAMMA", "1.0"))

# ── 가드 파라미터 (환경변수로 미세 조정 가능)
ANCHOR_ENABLE         = os.getenv("EVO_ANCHOR_ENABLE", "1") == "1"
ANCHOR_LOOKBACK_HRS   = float(os.getenv("EVO_ANCHOR_LOOKBACK_HRS", "6"))
ANCHOR_MIN_CONF       = float(os.getenv("EVO_ANCHOR_MIN_CONF", "0.55"))  # BTC 예측 신뢰 컷(=calib_prob)
ANCHOR_REQUIRE_LONG   = os.getenv("EVO_ANCHOR_REQUIRE_STRATEGY", "장기")  # BTC 앵커 기준 기간
CONFLICT_ENABLE       = os.getenv("EVO_CONFLICT_ENABLE", "1") == "1"
CONFLICT_LOOKBACK_MIN = float(os.getenv("EVO_CONFLICT_LOOKBACK_MIN", "90"))
CONFLICT_MIN_CONF     = float(os.getenv("EVO_CONFLICT_MIN_CONF", "0.55"))
MINRET_THR            = float(os.getenv("PREDICT_MIN_RETURN", "0.01"))   # predict.py와 맞춤

# =========================================
# 기본 모델
# =========================================
class EvoMetaModel(nn.Module if nn is not None else object):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        if nn is None:
            return
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if F is None:
            return x
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def _safe_lit(x, default):
    try:
        if x is None or str(x).strip() == "":
            return default
        return ast.literal_eval(str(x))
    except Exception:
        return default

def _df_from_path_or_df(path_or_df):
    if path_or_df is None:
        return None
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    if isinstance(path_or_df, str):
        if not os.path.exists(path_or_df):
            return None
        for enc in ("utf-8-sig", None):
            try:
                return pd.read_csv(path_or_df, encoding=enc, on_bad_lines="skip")
            except Exception:
                continue
    return None

# ── (NEW) 앙상블 확률 합성 유틸: (W, C) -> (C,)
def aggregate_probs_for_meta(probs_stack: np.ndarray,
                             mode: str = None,
                             gamma: float = None) -> np.ndarray:
    if probs_stack is None:
        return None
    ps = np.asarray(probs_stack, dtype=float)
    if ps.ndim != 2 or ps.shape[0] == 0 or ps.shape[1] == 0:
        return None
    mode = (mode or EVO_META_AGG).lower()
    gamma = EVO_META_VAR_GAMMA if gamma is None else float(gamma)
    eps = 1e-12

    mean = ps.mean(axis=0)
    if mode == "mean":
        out = mean
    else:
        var = ps.var(axis=0)
        penal = mean / (1.0 + gamma * var)
        out = penal if mode == "varpen" else (0.5 * mean + 0.5 * penal)

    out = out / (out.sum() + eps)
    return out.astype(float)

# =========================================
# 데이터 준비/학습 루틴(원본 유지)
# =========================================
def prepare_evo_meta_dataset(path_or_df="/persistent/wrong_predictions.csv", min_samples=50):
    df = _df_from_path_or_df(path_or_df)
    if df is None:
        print(f"[❌ prepare_evo_meta_dataset] 파일/데이터 없음 또는 읽기 실패: {path_or_df}")
        return None, None

    try:
        if len(df) < min_samples:
            print(f"[❌ prepare_evo_meta_dataset] 샘플 부족: {len(df)}개 (min={min_samples})")
            return None, None
    except Exception:
        print("[❌ prepare_evo_meta_dataset] 데이터 길이 확인 실패")
        return None, None

    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            sm = _safe_lit(row.get("softmax"), [])
            er = _safe_lit(row.get("expected_returns"), [0, 0, 0])
            mp = _safe_lit(row.get("model_predictions"), [0, 0, 0])
            if not sm or len(sm) < 3:
                continue
            label = int(float(row.get("label", -1))) if pd.notnull(row.get("label")) else -1
            feats = []
            for i in range(3):
                s_val = float(sm[i]) if i < len(sm) else 0.0
                e_val = float(er[i]) if i < len(er) else 0.0
                hit = 1 if (i < len(mp) and int(mp[i]) == label and label >= 0) else 0
                feats.extend([s_val, e_val, hit])
            if len(feats) != 9:
                continue
            X_list.append(feats)
            try:
                y_list.append(int(float(row.get("best_strategy", 0))))
            except Exception:
                y_list.append(0)
        except Exception:
            continue

    if not X_list or not y_list:
        print("[❌ prepare_evo_meta_dataset] 유효 샘플 부족")
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[✅ prepare_evo_meta_dataset] X:{X.shape}, y:{y.shape}")
    return X, y

def train_evo_meta(X, y, input_size, output_size=3, epochs=10, batch_size=32, lr=1e-3, task="strategy"):
    if torch is None:
        raise RuntimeError("torch is required for train_evo_meta")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvoMetaModel(input_size, output_size=output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[evo_meta_learner] 학습 시작 → 샘플:{len(dataset)}, output_size:{output_size}, input:{input_size}, task={task}")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.item())
        print(f"[evo_meta_learner] Epoch {epoch+1}/{epochs} → Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        model_cpu = model.to("cpu")
        torch.save(model_cpu.state_dict(), MODEL_PATH)
    finally:
        try:
            model.to(DEVICE)
        except Exception:
            pass

    meta_info = {
        "task": str(task),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "version": 1,
        "agg_mode": EVO_META_AGG,
        "agg_gamma": EVO_META_VAR_GAMMA,
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] meta save 실패: {e}")

    print(f"[✅ evo_meta_learner] 학습 완료 → {MODEL_PATH} (task={task}, output_size={output_size})")
    return model

def train_evo_meta_loop(min_samples=50, auto_trigger=False, task="strategy", path_or_df="/persistent/wrong_predictions.csv"):
    if task == "strategy":
        X, y = prepare_evo_meta_dataset(path_or_df, min_samples=min_samples)
    else:
        X, y = None, None

    if X is None or y is None:
        if auto_trigger:
            print("[⏭️ evo_meta_learner] 데이터 부족 → 자동 학습 스킵")
        return

    input_size = int(X.shape[1])
    output_size = int(len(np.unique(y))) if len(np.unique(y)) > 0 else 3
    train_evo_meta(X, y, input_size=input_size, output_size=output_size, task=task)
    print("[✅ evo_meta_learner] 학습 완료 및 모델 저장됨")

def _load_meta():
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    return meta

# =========================================
# 가드 유틸
# =========================================
def _class_sign(cls_id: int, symbol: str = None, strategy: str = None) -> str:
    """클래스 수익구간으로 long/short/neutral 판정"""
    try:
        lo, hi = get_class_return_range(int(cls_id), symbol, strategy)
    except Exception:
        lo, hi = (-0.01, 0.01)
    if hi <= 0 and lo < 0:  return "short"
    if lo >= 0 and hi > 0:  return "long"
    return "neutral"

def _load_recent_predictions(symbol: str = None,
                             strategy: str = None,
                             lookback_minutes: float = 120) -> pd.DataFrame | None:
    if not os.path.exists(PRED_LOG):
        return None
    try:
        df = pd.read_csv(PRED_LOG, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(PRED_LOG, on_bad_lines="skip")
        except Exception:
            return None
    if df.empty:
        return None
    # 시간 파싱
    ts = pd.to_datetime(df.get("timestamp"), errors="coerce")
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    df = df.assign(_ts=ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT", errors="coerce"))
    df = df.dropna(subset=["_ts"])
    df = df[df["_ts"] >= (now - pd.Timedelta(minutes=float(lookback_minutes)))]
    if symbol:
        df = df[df.get("symbol") == symbol]
    if strategy:
        df = df[df.get("strategy") == strategy]
    return df if not df.empty else None

def _sign_from_row(row) -> str:
    try:
        lo = float(row.get("class_return_min", 0.0))
        hi = float(row.get("class_return_max", 0.0))
        if hi <= 0 and lo < 0:  return "short"
        if lo >= 0 and hi > 0:  return "long"
    except Exception:
        pass
    # 보조: note에 position이 기록되는 경우
    try:
        note = str(row.get("note", ""))
        if '"position": "short"' in note: return "short"
        if '"position": "long"'  in note: return "long"
    except Exception:
        pass
    return "neutral"

def _btc_anchor() -> dict | None:
    """BTCUSDT 장기 예측 앵커 로드"""
    if not ANCHOR_ENABLE:
        return None
    df = _load_recent_predictions(symbol="BTCUSDT",
                                  strategy=str(ANCHOR_REQUIRE_LONG),
                                  lookback_minutes=ANCHOR_LOOKBACK_HRS * 60.0)
    if df is None:
        return None
    # 최신 한 건
    row = df.sort_values("_ts").iloc[-1]
    sign = _sign_from_row(row)
    try:
        conf = float(row.get("calib_prob", np.nan))
    except Exception:
        conf = np.nan
    return {"sign": sign, "conf": conf}

def _has_cross_horizon_conflict(symbol: str, strategy: str, proposed_sign: str) -> bool:
    """같은 심볼의 다른 기간 예측과 정면 충돌 여부(최근 N분)"""
    if not CONFLICT_ENABLE:
        return False
    other_map = {"단기": ["중기", "장기"], "중기": ["단기", "장기"], "장기": ["단기", "중기"]}
    others = other_map.get(str(strategy), ["단기", "중기", "장기"])
    df = _load_recent_predictions(symbol=symbol, strategy=None,
                                  lookback_minutes=CONFLICT_LOOKBACK_MIN)
    if df is None:
        return False
    df = df[df["strategy"].isin(others)]
    if df.empty:
        return False
    # 신뢰도 필터
    if "calib_prob" in df.columns:
        df = df[pd.to_numeric(df["calib_prob"], errors="coerce") >= CONFLICT_MIN_CONF]
    if df.empty:
        return False
    # 반대 방향이 강하게 존재?
    opp = "short" if proposed_sign == "long" else ("long" if proposed_sign == "short" else "neutral")
    if opp == "neutral":
        return False
    signs = df.apply(_sign_from_row, axis=1).values.tolist()
    return opp in signs

# =========================================
# 예측(진입점) — 가드 적용
# =========================================
def predict_evo_meta(X_new,
                     input_size,
                     probs_stack: np.ndarray = None,
                     *,
                     symbol: str = None,
                     strategy: str = None):
    """
    저장된 메타의 task가 'class' 또는 'strategy'이면 사용.
    가드 로직:
      - BTC 앵커 가드(장기): BTC 장기 신뢰 높은 방향과 정면 충돌 시 None(패스)
      - 기간 상호모순 가드: 동일 심볼의 최근 타 기간 예측과 강한 충돌 시 None(패스)
    ※ None을 반환하면 상위 predict.py가 기본 경로로 진행(메타 비사용).
    """
    if torch is None:
        print("[❌ evo_meta_learner] torch 미설치")
        return None
    if not os.path.exists(MODEL_PATH):
        print("[❌ evo_meta_learner] 모델 없음")
        return None

    meta = _load_meta()
    task = meta.get("task", "class")
    if task not in ("class", "strategy"):
        print(f"[ℹ️ evo_meta_learner] task={task} → 사용하지 않음")
        return None

    # (선택) 앙상블 확률 합성 규칙 로그
    if probs_stack is not None:
        vec = aggregate_probs_for_meta(probs_stack, mode=meta.get("agg_mode", EVO_META_AGG),
                                       gamma=float(meta.get("agg_gamma", EVO_META_VAR_GAMMA)))
        if vec is not None:
            try:
                top = np.argsort(vec)[::-1][:3].tolist()
                print(f"[evo_meta_agg] mode={meta.get('agg_mode', EVO_META_AGG)}, "
                      f"gamma={meta.get('agg_gamma', EVO_META_VAR_GAMMA)}, top3={top}, "
                      f"max={float(vec.max()):.3f}")
            except Exception:
                pass

    out_size = int(meta.get("output_size", 3))
    model = EvoMetaModel(input_size, output_size=out_size)
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[❌ evo_meta_learner] 모델 로드 실패: {e}")
        return None

    # 입력 모양 가드
    try:
        x = torch.tensor(X_new, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != int(input_size):
            x = x.reshape(x.shape[0], int(input_size))
    except Exception as e:
        print(f"[❌ evo_meta_learner] 입력 가공 실패: {e}")
        return None

    model.eval()
    with torch.no_grad():
        try:
            logits = model(x)
            if F is None:
                print("[❌ evo_meta_learner] torch.nn.functional 없음")
                return None
            probs = F.softmax(logits, dim=1)
            best = int(torch.argmax(probs, dim=1).item())
        except Exception as e:
            print(f"[❌ evo_meta_learner] 예측 실패: {e}")
            return None

    # ==========================
    # 가드 적용 (symbol/strategy가 제공된 경우)
    # ==========================
    try:
        if symbol is not None and strategy is not None:
            proposed_sign = _class_sign(best, symbol, strategy)

            # 1) BTC 앵커 가드 (장기 예측에만 적용)
            if ANCHOR_ENABLE and str(strategy) == str(ANCHOR_REQUIRE_LONG) and str(symbol) != "BTCUSDT":
                anchor = _btc_anchor()
                if anchor and anchor.get("sign") in ("long", "short"):
                    conf = float(anchor.get("conf") or 0.0)
                    if conf >= ANCHOR_MIN_CONF:
                        # 강한 충돌 시: None 반환(메타 패스)
                        if (anchor["sign"] == "long" and proposed_sign == "short") or \
                           (anchor["sign"] == "short" and proposed_sign == "long"):
                            print(f"[ANCHOR_GUARD] BTC({ANCHOR_REQUIRE_LONG})={anchor['sign']}@{conf:.2f} "
                                  f"↔ {symbol}-{strategy}:{proposed_sign} → PASS(meta)")
                            return None

            # 2) 기간 상호모순 가드 (최근 N분 내 타 기간 강한 반대 신호)
            if CONFLICT_ENABLE:
                if _has_cross_horizon_conflict(symbol, strategy, proposed_sign):
                    print(f"[CONFLICT_GUARD] {symbol}-{strategy}:{proposed_sign} "
                          f"↔ 최근 타기간 강한 반대 → PASS(meta)")
                    return None
    except Exception as e:
        print(f"[WARN] 가드 적용 중 예외: {e}")

    return best

# =========================================
# 보조 휴리스틱(원본 유지)
# =========================================
def get_best_strategy_by_failure_probability(symbol, current_strategy, feature_tensor, model_outputs):
    try:
        path = PRED_LOG
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["symbol"] == symbol]
        df = df[df["status"].isin(["success","fail","v_success","v_fail"])]
        if df.empty:
            return None
        sr = df.pivot_table(index="strategy", values="status",
                            aggfunc=lambda s: (s.isin(["success","v_success"])).mean())
        sr = sr["status"].to_dict()
        curr = float(sr.get(current_strategy, 0.0))
        if curr < 0.25:
            alt = sorted([(k, float(v)) for k, v in sr.items() if k != current_strategy],
                         key=lambda x: x[1], reverse=True)
            if alt and alt[0][1] >= 0.45:
                return alt[0][0]
        return None
    except Exception:
        return None
