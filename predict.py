# predict.py (FINAL — canonical rewrite + numeric sanitation + safe top_k + header-locked rewrite + KST timestamp normalization)

import os, sys, json, datetime, pytz
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from data.utils import get_kline_by_strategy, compute_features

# --- window_optimizer 호환 임포트 ---
try:
    from window_optimizer import find_best_windows  # 선호
except Exception:
    try:
        from window_optimizer import find_best_window
    except Exception:
        find_best_window = None
    def find_best_windows(symbol, strategy):
        try:
            if callable(find_best_window):
                best = int(find_best_window(symbol, strategy, window_list=[10, 20, 30, 40, 60], group_id=None))
            else:
                best = 60
        except Exception:
            best = 60
        return [best, best, best]

# --- (옵션) 레짐/캘리브레이션 모듈: 없으면 안전 패스 ---
try:
    from regime_detector import detect_regime
except Exception:
    def detect_regime(symbol, strategy, now=None):
        return "unknown"

try:
    from calibration import apply_calibration, get_calibration_version
except Exception:
    def apply_calibration(probs, *, symbol=None, strategy=None, regime=None, model_meta=None):
        return probs  # no-op
    def get_calibration_version():
        return "none"

# logger
from logger import log_prediction, update_model_success, PREDICTION_HEADERS, ensure_prediction_log_exists
from failure_db import insert_failure_record, load_existing_failure_hashes, ensure_failure_db
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
from model.base_model import get_model
from model_weight_loader import load_model_cached
from config import (
    get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups,
    get_class_return_range, class_to_expected_return
)

DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
PREDICTION_LOG_PATH = "/persistent/prediction_log.csv"

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ✅ 최소 예측 기대수익률 임계치(기본 1%) — 이보다 작은 클래스는 선택/기록하지 않음
MIN_RET_THRESHOLD = float(os.getenv("PREDICT_MIN_RETURN", "0.01"))

# -----------------------------
# 로컬 헬퍼: feature hash
# -----------------------------
def _get_feature_hash(feature_row) -> str:
    try:
        import hashlib
        if feature_row is None:
            return "none"
        if isinstance(feature_row, torch.Tensor):
            arr = feature_row.detach().cpu().flatten().numpy().astype(float)
        elif isinstance(feature_row, np.ndarray):
            arr = feature_row.flatten().astype(float)
        elif isinstance(feature_row, (list, tuple)):
            arr = np.array(feature_row, dtype=float).flatten()
        else:
            arr = np.array([float(feature_row)], dtype=float)
        rounded = [round(float(x), 2) for x in arr]
        joined = ",".join(map(str, rounded))
        return hashlib.sha1(joined.encode()).hexdigest()
    except Exception:
        return "hash_error"

# -----------------------------
# 로컬 헬퍼: 모델 탐색
# -----------------------------
def get_available_models(symbol: str, strategy: str):
    try:
        if not os.path.isdir(MODEL_DIR):
            return []
        items = []
        prefix = f"{symbol}_"
        needle = f"_{strategy}_"
        for fn in os.listdir(MODEL_DIR):
            if not fn.endswith(".pt"):
                continue
            if not fn.startswith(prefix):
                continue
            if needle not in fn:
                continue
            meta = os.path.join(MODEL_DIR, fn.replace(".pt", ".meta.json"))
            if not os.path.exists(meta):
                continue
            items.append({"pt_file": fn})
        items.sort(key=lambda x: x["pt_file"])
        return items
    except Exception as e:
        print(f"[get_available_models 오류] {e}")
        return []

# -----------------------------
# 실패 결과 빠른 기록
# -----------------------------
def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    from datetime import datetime as _dt
    t = _dt.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "symbol": symbol, "strategy": strategy, "success": False, "reason": reason,
        "model": str(model_type or "unknown"), "rate": 0.0, "class": -1,
        "timestamp": t, "source": source, "predicted_class": -1, "label": -1
    }
    try:
        log_prediction(
            symbol=symbol, strategy=strategy, direction="예측실패",
            entry_price=0, target_price=0, model=str(model_type or "unknown"),
            success=False, reason=reason, rate=0.0, timestamp=t,
            return_value=0.0, volatility=True, source=source,
            predicted_class=-1, label=-1
        )
    except Exception as e:
        print(f"[failed_result log_prediction 오류] {e}")
    try:
        if X_input is not None:
            feature_hash = _get_feature_hash(X_input)
            insert_failure_record(result, feature_hash, feature_vector=np.array(X_input).flatten().tolist(), label=-1)
    except Exception as e:
        print(f"[failed_result insert_failure_record 오류] {e}")
    return result

# -----------------------------
# 메인 예측
# -----------------------------
def predict(symbol, strategy, source="일반", model_type=None):
    """
    - 저장된 모델 출력 취합
    - 진화형 메타러너가 있으면 사용, 없으면 '캘리브레이션 확률이 가장 높은 단일 모델' 선택
      success_score = adjusted_calib_prob[pred] × (0.5 + 0.5 × val_f1)
    - ✅ MIN_RET_THRESHOLD(기본 1%) 미만 클래스는 선택/기록하지 않음
    """
    try:
        from evo_meta_learner import predict_evo_meta
    except Exception:
        predict_evo_meta = None
    try:
        from meta_learning import get_meta_prediction
    except Exception:
        def get_meta_prediction(probs_list, feature_tensor, meta_info=None):
            avg = np.mean(np.array(probs_list), axis=0)
            return int(np.argmax(avg))

    ensure_failure_db()
    os.makedirs("/persistent/logs", exist_ok=True)

    if not symbol or not strategy:
        return failed_result(symbol or "None", strategy or "None", reason="invalid_symbol_strategy", X_input=None)

    regime = detect_regime(symbol, strategy, now=now_kst())
    calib_ver = get_calibration_version()

    # 1) 준비
    window_list = find_best_windows(symbol, strategy)
    if not window_list:
        return failed_result(symbol, strategy, reason="window_list_none", X_input=None)

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < max(window_list) + 1:
        return failed_result(symbol, strategy, reason="df_short", X_input=None)

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.dropna().shape[0] < max(window_list) + 1:
        return failed_result(symbol, strategy, reason="feature_short", X_input=None)

    features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
    feat_scaled = MinMaxScaler().fit_transform(features_only)
    if feat_scaled.shape[1] < FEATURE_INPUT_SIZE:
        feat_scaled = np.pad(feat_scaled, ((0, 0), (0, FEATURE_INPUT_SIZE - feat_scaled.shape[1])), mode="constant")
    else:
        feat_scaled = feat_scaled[:, :FEATURE_INPUT_SIZE]

    models = get_available_models(symbol, strategy)
    if not models:
        return failed_result(symbol, strategy, reason="no_models", X_input=feat_scaled[-1])

    recent_freq = get_recent_class_frequencies(strategy)
    feature_tensor = torch.tensor(feat_scaled[-1], dtype=torch.float32)

    # 2) 모델별 확률 계산
    model_outputs_list, all_model_predictions = get_model_predictions(
        symbol, strategy, models, df, feat_scaled, window_list, recent_freq, regime=regime
    )
    if not model_outputs_list:
        return failed_result(symbol, strategy, reason="no_valid_model", X_input=feat_scaled[-1])

    # 3) (옵션) 진화형 메타 사용 — 임계 미만 클래스면 무시
    final_pred_class = None
    use_evo = False
    evo_model_path = os.path.join(MODEL_DIR, "evo_meta_learner.pt")
    if os.path.exists(evo_model_path):
        try:
            from evo_meta_learner import predict_evo_meta  # 재확인
            if callable(predict_evo_meta):
                evo_pred = predict_evo_meta(feature_tensor.unsqueeze(0), input_size=FEATURE_INPUT_SIZE)
                if evo_pred is not None:
                    evo_pred = int(evo_pred)
                    cls_min_evo, _ = get_class_return_range(evo_pred, symbol, strategy)
                    if cls_min_evo >= MIN_RET_THRESHOLD:
                        final_pred_class = evo_pred
                        use_evo = True
                    else:
                        print(f"[META] 진화형 예측 {evo_pred} 최소수익 {cls_min_evo:.4f} < 임계 {MIN_RET_THRESHOLD:.4f} → 무시")
        except Exception as e:
            print(f"[⚠️ 진화형 메타러너 예외] {e}")

    # 4) '최고 성공확률 단일 모델' (1% 필터 포함)
    meta_choice = "best_single"
    chosen_info = None
    used_minret_filter = False
    if final_pred_class is None:
        best_idx, best_score, best_pred = -1, -1.0, None
        for i, m in enumerate(model_outputs_list):
            calib_probs = m["calib_probs"]
            adj = adjust_probs_with_diversity(calib_probs, recent_freq, class_counts=None, alpha=0.10, beta=0.10)
            val_f1 = float(m.get("val_f1", 0.6))

            # 임계치 필터 마스크
            valid_mask = np.zeros_like(adj, dtype=float)
            for ci in range(len(adj)):
                try:
                    cls_min, _ = get_class_return_range(ci, symbol, strategy)
                    if float(cls_min) >= MIN_RET_THRESHOLD:
                        valid_mask[ci] = 1.0
                except Exception:
                    pass
            adj_filtered = adj * valid_mask
            if adj_filtered.sum() > 0:
                adj_filtered = adj_filtered / adj_filtered.sum()
                pred = int(np.argmax(adj_filtered))
                prob_for_score = float(adj_filtered[pred])
                used_filter_here = True
            else:
                pred = int(np.argmax(adj))
                prob_for_score = float(adj[pred])
                used_filter_here = False

            score = prob_for_score * (0.5 + 0.5 * max(0.0, min(1.0, val_f1)))
            m["adjusted_probs"] = adj
            m["success_score"] = score
            m["filtered_used"] = used_filter_here
            m["filtered_probs"] = adj_filtered if used_filter_here else None
            m["candidate_pred"] = pred

            if score > best_score:
                best_score, best_idx, best_pred = score, i, pred
                used_minret_filter = used_filter_here

        final_pred_class = int(best_pred)
        meta_choice = os.path.basename(model_outputs_list[best_idx]["model_path"])
        chosen_info = model_outputs_list[best_idx]
    else:
        meta_choice = "evo_meta_learner"
        chosen_info = max(model_outputs_list, key=lambda m: m.get("success_score", 0.0)) if model_outputs_list else None

    # 최종 가드: 임계 미만이면 전 모델을 가로질러 대체 후보 탐색
    try:
        cls_min_sel, _ = get_class_return_range(final_pred_class, symbol, strategy)
        if float(cls_min_sel) < MIN_RET_THRESHOLD:
            print(f"[GUARD] 선택 클래스 {final_pred_class} 최소수익 {cls_min_sel:.4f} < 임계 {MIN_RET_THRESHOLD:.4f} → 대체 탐색")
            best_global_idx, best_global_score, best_global_class = None, -1.0, None
            for m in model_outputs_list:
                adj = m.get("adjusted_probs", m["calib_probs"])
                val_f1 = float(m.get("val_f1", 0.6))
                for ci in range(len(adj)):
                    try:
                        cmin, _ = get_class_return_range(ci, symbol, strategy)
                        if float(cmin) < MIN_RET_THRESHOLD:
                            continue
                        score = float(adj[ci]) * (0.5 + 0.5 * max(0.0, min(1.0, val_f1)))
                        if score > best_global_score:
                            best_global_score, best_global_idx, best_global_class = score, m, int(ci)
                    except Exception:
                        continue
            if best_global_class is not None:
                final_pred_class, chosen_info, used_minret_filter = best_global_class, best_global_idx, True
            else:
                return failed_result(symbol, strategy, reason="no_class_ge_min_return", X_input=feat_scaled[-1])
    except Exception as e:
        print(f"[임계치 최종 가드 예외] {e}")

    print(f"[META] {'진화형' if meta_choice=='evo_meta_learner' else '최고확률모델'} 선택: 클래스 {final_pred_class}")

    # 5) 로깅 및 성공판정(메타 최종)
    cls_min, _ = get_class_return_range(final_pred_class, symbol, strategy)
    current_price = float(df.iloc[-1]["close"])
    expected_ret = class_to_expected_return(final_pred_class, symbol, strategy)
    entry_price = float(all_model_predictions[0]["entry_price"])
    actual_return_meta = (current_price / (entry_price + 1e-12)) - 1
    meta_success_flag = actual_return_meta >= cls_min

    if not meta_success_flag:
        try:
            feature_hash = _get_feature_hash(feature_tensor)
            insert_failure_record(
                {"symbol": symbol, "strategy": strategy, "reason": "meta_predicted_fail"},
                feature_hash,
                feature_vector=feature_tensor.numpy().flatten().tolist(),
                label=final_pred_class
            )
        except Exception as e:
            print(f"[meta 실패 기록 오류] {e}")

    def _topk(probs, k=3):
        idx = np.argsort(probs)[::-1][:k]
        return [int(i) for i in idx]

    calib_topk = _topk((chosen_info or model_outputs_list[0])["calib_probs"]) if (chosen_info or model_outputs_list) else []

    note_payload = {
        "regime": regime,
        "meta_choice": meta_choice,
        "raw_prob_pred": float((chosen_info or model_outputs_list[0])["raw_probs"][final_pred_class]) if (chosen_info or model_outputs_list) else None,
        "calib_prob_pred": float((chosen_info or model_outputs_list[0])["calib_probs"][final_pred_class]) if (chosen_info or model_outputs_list) else None,
        "calib_ver": calib_ver,
        "min_return_threshold": float(MIN_RET_THRESHOLD),
        "used_minret_filter": bool(used_minret_filter)
    }

    log_prediction(
        symbol=symbol,
        strategy=strategy,
        direction="예측",
        entry_price=entry_price,
        target_price=entry_price * (1 + expected_ret),
        model="meta",
        model_name="evo_meta_learner" if meta_choice=="evo_meta_learner" else "best_single",
        predicted_class=final_pred_class,
        label=final_pred_class,
        note=json.dumps(note_payload, ensure_ascii=False),
        top_k=calib_topk,
        success=meta_success_flag,
        reason=f"수익률도달:{meta_success_flag}",
        rate=expected_ret,
        return_value=actual_return_meta,
        source="진화형" if meta_choice=="evo_meta_learner" else "기본",
        group_id=(chosen_info.get("group_id") if chosen_info else None) if isinstance(chosen_info, dict) else None,
        feature_vector=feature_tensor.numpy()
    )

    # 🔥 [추가] 메타에 선택되지 않은 "모든 모델"도 섀도우 예측으로 기록 → 평가/실패학습 대상
    try:
        for m in model_outputs_list:
            # 메타가 참조한 chosen_info와 같은 모델이면(이미 위에서 기록됨) 패스
            if chosen_info and m.get("model_path") == chosen_info.get("model_path"):
                continue

            adj = m.get("adjusted_probs", m["calib_probs"])
            filt = m.get("filtered_probs", None)
            # 임계치 만족하는 클래스 우선
            if filt is not None and np.sum(filt) > 0:
                pred_i = int(np.argmax(filt))
                topk_src = filt
            else:
                # 필요한 경우 즉석 필터
                mask = np.zeros_like(adj, dtype=float)
                for ci in range(len(adj)):
                    try:
                        cmin, _ = get_class_return_range(ci, symbol, strategy)
                        if float(cmin) >= MIN_RET_THRESHOLD:
                            mask[ci] = 1.0
                    except Exception:
                        pass
                adj2 = adj * mask
                if np.sum(adj2) == 0:
                    # 이 모델은 임계 이상 클래스가 없음 → 기록 생략(설계상 <1% 미포함)
                    continue
                adj2 = adj2 / np.sum(adj2)
                pred_i = int(np.argmax(adj2))
                topk_src = adj2

            exp_ret_i = class_to_expected_return(pred_i, symbol, strategy)
            top_k_i = [int(i) for i in np.argsort(topk_src)[::-1][:3]]
            note_shadow = {
                "regime": regime,
                "shadow": True,
                "model_path": os.path.basename(m.get("model_path","")),
                "model_type": m.get("model_type",""),
                "val_f1": float(m.get("val_f1",0.0)),
                "calib_ver": calib_ver,
                "min_return_threshold": float(MIN_RET_THRESHOLD)
            }
            log_prediction(
                symbol=symbol,
                strategy=strategy,
                direction="예측(섀도우)",
                entry_price=entry_price,
                target_price=entry_price * (1 + exp_ret_i),
                model=m.get("model_type","model"),
                model_name=os.path.basename(m.get("model_path","")),
                predicted_class=pred_i,
                label=pred_i,
                note=json.dumps(note_shadow, ensure_ascii=False),
                top_k=top_k_i,
                success=False,                   # 평가는 evaluate_predictions에서 판정
                reason="shadow",
                rate=exp_ret_i,
                return_value=actual_return_meta, # 로깅 일관성용(실제 평가는 별도)
                source="섀도우",
                group_id=m.get("group_id", 0),
                feature_vector=feature_tensor.numpy()
            )
    except Exception as e:
        print(f"[섀도우 로깅 예외] {e}")

    return {
        "symbol": symbol,
        "strategy": strategy,
        "model": "meta",
        "class": final_pred_class,
        "expected_return": expected_ret,
        "timestamp": now_kst().isoformat(),
        "reason": "진화형 메타 최종 선택" if meta_choice=="evo_meta_learner" else f"최고 확률 단일 모델: {meta_choice}",
        "source": source,
        "regime": regime
    }

# -----------------------------
# 배치 평가
# -----------------------------
def evaluate_predictions(get_price_fn):
    import csv, os
    import pandas as pd
    from collections import defaultdict
    from failure_db import check_failure_exists

    ensure_failure_db()
    ensure_prediction_log_exists()

    PREDICTION_LOG = PREDICTION_LOG_PATH
    now_local = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    date_str = now_local().strftime("%Y-%m-%d")
    LOG_DIR = "/persistent/logs"
    EVAL_RESULT = os.path.join(LOG_DIR, f"evaluation_{date_str}.csv")
    WRONG = os.path.join(LOG_DIR, f"wrong_{date_str}.csv")

    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}
    updated, evaluated = [], []

    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            return
    except Exception as e:
        print(f"[오류] prediction_log.csv 읽기 실패 → {e}")
        return

    grouped_preds = defaultdict(list)
    for r in rows:
        key = (r.get("symbol"), r.get("strategy"), r.get("timestamp"))
        grouped_preds[key].append(r)

    for key, preds in grouped_preds.items():
        for r in preds:
            try:
                if r.get("status") not in [None, "", "pending", "v_pending"]:
                    updated.append(r); continue

                symbol = r.get("symbol", "UNKNOWN")
                strategy = r.get("strategy", "알수없음")
                model = r.get("model", "unknown")
                group_id = int(float(r.get("group_id", 0))) if str(r.get("group_id", "")).strip().replace(".","",1).isdigit() else 0

                pred_class = int(float(r.get("predicted_class", -1))) if pd.notnull(r.get("predicted_class")) else -1
                label = int(float(r.get("label", -1))) if pd.notnull(r.get("label")) else -1
                r["label"] = label

                try:
                    entry_price = float(r.get("entry_price", 0) or 0)
                except Exception:
                    entry_price = 0.0

                if entry_price <= 0 or label == -1:
                    reason = "entry_price 오류 또는 label=-1"
                    r.update({"status": "fail", "reason": reason, "return": 0.0, "return_value": 0.0})
                    log_prediction(
                        symbol=symbol, strategy=strategy, direction="예측실패",
                        entry_price=entry_price, target_price=entry_price,
                        timestamp=now_local().isoformat(), model=model, predicted_class=pred_class,
                        success=False, reason=reason, rate=0.0, return_value=0.0,
                        volatility=False, source="평가", label=label, group_id=group_id
                    )
                    if not check_failure_exists(r):
                        from failure_db import insert_failure_record
                        insert_failure_record(r, f"{symbol}-{strategy}-{now_local().isoformat()}",
                                              feature_vector=None, label=label)
                    updated.append(r); continue

                timestamp = pd.to_datetime(r.get("timestamp"), errors="coerce")
                if timestamp is None or pd.isna(timestamp):
                    r.update({"status": "fail", "reason": "timestamp 파싱 실패", "return": 0.0, "return_value": 0.0})
                    updated.append(r); continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("Asia/Seoul")
                else:
                    timestamp = timestamp.tz_convert("Asia/Seoul")

                eval_hours = eval_horizon_map.get(strategy, 6)
                deadline = timestamp + pd.Timedelta(hours=eval_hours)

                df = get_price_fn(symbol, strategy)
                if df is None or "timestamp" not in df.columns:
                    r.update({"status": "fail", "reason": "가격 데이터 없음", "return": 0.0, "return_value": 0.0})
                    updated.append(r); continue

                # 🔒 반드시 예측 시점~마감(deadline)까지만 평가
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                mask_window = (df["timestamp"] >= timestamp) & (df["timestamp"] <= deadline)
                future_df = df.loc[mask_window]

                if future_df.empty:
                    if now_local() < deadline:
                        r.update({"status": "pending", "reason": "⏳ 평가 대기 중(마감 전 데이터 없음)", "return": 0.0, "return_value": 0.0})
                        updated.append(r); continue
                    else:
                        r.update({"status": "fail", "reason": "마감까지 데이터 없음", "return": 0.0, "return_value": 0.0})
                        updated.append(r); continue

                actual_max = float(future_df["high"].max())
                gain = (actual_max - entry_price) / (entry_price + 1e-12)

                if pred_class >= 0:
                    cls_min, cls_max = get_class_return_range(pred_class, symbol, strategy)
                else:
                    cls_min, cls_max = (0.0, 0.0)

                reached_target = gain >= cls_min

                if now_local() < deadline:
                    if reached_target:
                        status = "success"
                    else:
                        r.update({"status": "pending", "reason": "⏳ 평가 대기 중", "return": round(gain, 5), "return_value": round(gain, 5)})
                        updated.append(r); continue
                else:
                    status = "success" if reached_target else "fail"

                vol = str(r.get("volatility", "")).strip().lower() in ["1", "true"]
                if vol:
                    status = "v_success" if status == "success" else "v_fail"

                r.update({
                    "status": status,
                    "reason": f"[pred_class={pred_class}] gain={gain:.3f} (cls_min={cls_min}, cls_max={cls_max})",
                    "return": round(gain, 5),
                    "return_value": round(gain, 5),
                    "group_id": group_id
                })

                log_prediction(
                    symbol=symbol, strategy=strategy, direction=f"평가:{status}",
                    entry_price=entry_price, target_price=entry_price * (1 + gain),
                    timestamp=now_local().isoformat(), model=model, predicted_class=pred_class,
                    success=(status in ["success", "v_success"]), reason=r["reason"],
                    rate=gain, return_value=gain, volatility=vol, source="평가",
                    label=label, group_id=group_id
                )

                if status in ["fail", "v_fail"] and not check_failure_exists(r):
                    from failure_db import insert_failure_record
                    insert_failure_record(r, f"{symbol}-{strategy}-{now_local().isoformat()}",
                                          feature_vector=None, label=label)

                # (선택) 통계 집계는 meta 한정 — 현 설계 유지
                if model == "meta":
                    update_model_success(symbol, strategy, model, status in ["success", "v_success"])

                evaluated.append({str(k): (v if v is not None else "") for k, v in r.items()})
            except Exception as e:
                r.update({"status": "fail", "reason": f"예외: {e}", "return": 0.0, "return_value": 0.0})
                updated.append(r)

    # ---------- 안전 재작성 ----------
    def rewrite_prediction_log_canonical(path, rows):
        base = list(PREDICTION_HEADERS)
        extras = ["status", "return"]
        fieldnames = base + [c for c in extras if c not in base]

        def to_float(x, default=0.0):
            try:
                if x in [None, ""]:
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        def to_int(x, default=-1):
            try:
                if x in [None, ""]:
                    return int(default)
                return int(float(x))
            except Exception:
                return int(default)

        sanitized = []
        for r in rows:
            row = {k: "" for k in fieldnames}
            for k, v in r.items():
                if k in row:
                    row[k] = v
            try:
                ts_raw = r.get("timestamp", row.get("timestamp", ""))
                ts = pd.to_datetime(ts_raw, errors="coerce")
                if pd.isna(ts):
                    row["timestamp"] = now_kst().isoformat()
                else:
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("Asia/Seoul")
                    else:
                        ts = ts.tz_convert("Asia/Seoul")
                    row["timestamp"] = ts.isoformat()
            except Exception:
                row["timestamp"] = now_kst().isoformat()

            row["rate"] = to_float(row.get("rate", 0.0), 0.0)
            rv = to_float(row.get("return_value", r.get("return", 0.0)), 0.0)
            row["return_value"] = rv
            row["return"] = rv
            row["entry_price"] = to_float(row.get("entry_price", 0.0), 0.0)
            row["target_price"] = to_float(row.get("target_price", 0.0), 0.0)
            row["predicted_class"] = to_int(row.get("predicted_class", -1), -1)
            row["label"] = to_int(row.get("label", -1), -1)
            row["group_id"] = to_int(row.get("group_id", 0), 0)

            vol = str(r.get("volatility", row.get("volatility", ""))).strip().lower()
            row["volatility"] = "True" if vol in ["1", "true"] else ("False" if vol in ["0", "false"] else str(r.get("volatility", "")))
            sanitized.append(row)

        pd.DataFrame(sanitized, columns=fieldnames).to_csv(path, index=False, encoding="utf-8-sig")

    updated += evaluated
    rewrite_prediction_log_canonical(PREDICTION_LOG, updated)

    def safe_write_csv(path, rows):
        if not rows:
            return
        import csv
        fieldnames = sorted({str(k) for row in rows for k in row.keys() if k is not None})
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    safe_write_csv(EVAL_RESULT, evaluated)
    failed = [r for r in evaluated if r.get("status") in ["fail", "v_fail"]]
    safe_write_csv(WRONG, failed)

    print(f"[✅ 평가 완료] 총 {len(evaluated)}건 평가, 실패 {len(failed)}건")

# -----------------------------
# 개별 모델 예측 취합 (+캘리브레이션)
# -----------------------------
def get_model_predictions(symbol, strategy, models, df, feat_scaled, window_list, recent_freq, regime="unknown"):
    model_outputs_list, all_model_predictions = [], []

    for model_info in models:
        try:
            pt_file = model_info.get("pt_file")
            if not pt_file:
                continue
            model_path = os.path.join(MODEL_DIR, pt_file)
            meta_path = model_path.replace(".pt", ".meta.json")
            if not os.path.exists(meta_path):
                print(f"[⚠️ 메타파일 없음] {meta_path}")
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_type = meta.get("model", "lstm")
            group_id = meta.get("group_id", 0)
            input_size = meta.get("input_size", FEATURE_INPUT_SIZE)
            num_classes = meta.get("num_classes", NUM_CLASSES)
            val_f1 = float(meta.get("metrics", {}).get("val_f1", 0.6))

            idx = min(int(group_id), max(0, len(window_list) - 1))
            window = window_list[idx]
            input_seq = feat_scaled[-window:]
            if input_seq.shape[0] < window:
                print(f"[⚠️ 데이터 부족] {symbol}-{strategy}-group{group_id}")
                continue

            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

            model = get_model(model_type, input_size=input_size, output_size=num_classes)
            model = load_model_cached(model_path, model, ttl_sec=600)
            if model is None:
                print(f"[⚠️ 모델 로딩 실패] {model_path}")
                continue
            model.eval()

            with torch.no_grad():
                out = model(input_tensor.to(DEVICE))
                softmax_probs = F.softmax(out, dim=1)
                raw_probs = softmax_probs.squeeze().cpu().numpy()

            calib_probs = apply_calibration(
                raw_probs,
                symbol=symbol, strategy=strategy, regime=regime, model_meta=meta
            ).astype(float)

            model_outputs_list.append({
                "raw_probs": raw_probs,
                "calib_probs": calib_probs,
                "predicted_class": int(np.argmax(calib_probs)),
                "group_id": group_id,
                "model_type": model_type,
                "model_path": model_path,
                "val_f1": val_f1,
                "symbol": symbol, "strategy": strategy
            })

            entry_price = df["close"].iloc[-1]
            all_model_predictions.append({
                "class": int(np.argmax(calib_probs)),
                "probs": calib_probs, "entry_price": float(entry_price),
                "num_classes": num_classes, "group_id": group_id,
                "model_name": model_type, "model_symbol": symbol,
                "symbol": symbol, "strategy": strategy
            })

        except Exception as e:
            print(f"[❌ 모델 예측 실패] {model_info} → {e}")
            continue

    return model_outputs_list, all_model_predictions


if __name__ == "__main__":
    res = predict("BTCUSDT", "단기")
    print(res)
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig")
        print("[✅ prediction_log.csv 상위 20줄 출력]")
        print(df.head(20))
    except Exception as e:
        print(f"[오류] prediction_log.csv 로드 실패 → {e}")
