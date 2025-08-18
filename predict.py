# predict.py (FINAL PATCHED)

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

# logger 의존 최소화: get_available_models 로컬 구현, 나머지는 그대로 사용
from logger import log_prediction, update_model_success
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
    """
    /persistent/models에서 다음 규칙을 만족하는 pt만 반환:
      - 파일명 시작이 '{symbol}_'
      - 파일명에 '_{strategy}_' 포함
      - 동일 경로에 .meta.json 존재
    반환 포맷: [{"pt_file": "...pt"}]
    """
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
    - 진화형 메타러너가 있으면 사용, 없으면 '성공확률이 가장 높은 단일 모델'을 선택
      success_score = adjusted_prob[pred] × (0.5 + 0.5 × val_f1)
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

    # 2) 각 모델의 확률/메타 읽기
    model_outputs_list, all_model_predictions = get_model_predictions(
        symbol, strategy, models, df, feat_scaled, window_list, recent_freq
    )
    if not model_outputs_list:
        return failed_result(symbol, strategy, reason="no_valid_model", X_input=feat_scaled[-1])

    # 3) (옵션) 진화형 메타 사용
    final_pred_class = None
    use_evo = False
    evo_model_path = os.path.join(MODEL_DIR, "evo_meta_learner.pt")
    if os.path.exists(evo_model_path) and callable(predict_evo_meta):
        try:
            evo_pred = predict_evo_meta(feature_tensor.unsqueeze(0), input_size=FEATURE_INPUT_SIZE)
            if evo_pred is not None:
                final_pred_class = int(evo_pred)
                use_evo = True
        except Exception as e:
            print(f"[⚠️ 진화형 메타러너 예외] {e}")

    # 4) 기본 메타 또는 '최고 성공확률 단일 모델' 선택
    if final_pred_class is None:
        best_idx, best_score = -1, -1.0
        for i, m in enumerate(model_outputs_list):
            pred = int(m["predicted_class"])
            probs = m["probs"]
            # 다양성 보정
            adj = adjust_probs_with_diversity(probs, recent_freq, class_counts=None, alpha=0.10, beta=0.10)
            val_f1 = float(m.get("val_f1", 0.6))  # 메타 없으면 보수적 기본치
            score = float(adj[pred]) * (0.5 + 0.5 * max(0.0, min(1.0, val_f1)))
            model_outputs_list[i]["adjusted_probs"] = adj
            model_outputs_list[i]["success_score"] = score
            if score > best_score:
                best_score, best_idx = score, i
        final_pred_class = int(model_outputs_list[best_idx]["predicted_class"])

    print(f"[META] {'진화형' if use_evo else '최고확률모델'} 선택: 클래스 {final_pred_class}")

    # 5) 로깅 및 성공판정
    cls_min, _ = get_class_return_range(final_pred_class, symbol, strategy)
    current_price = float(df.iloc[-1]["close"])
    expected_ret = class_to_expected_return(final_pred_class, symbol, strategy)
    entry_price = float(all_model_predictions[0]["entry_price"])
    actual_return_meta = (current_price / (entry_price + 1e-12)) - 1
    meta_success_flag = actual_return_meta >= cls_min

    if not meta_success_flag:
        # 실패도 반드시 학습자산으로 남김
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

    log_prediction(
        symbol=symbol,
        strategy=strategy,
        direction="예측",
        entry_price=entry_price,
        target_price=entry_price * (1 + expected_ret),
        model="meta",
        model_name="evo_meta_learner" if use_evo else "best_single",
        predicted_class=final_pred_class,
        label=final_pred_class,
        note="진화형 메타 선택" if use_evo else "최고 확률 단일 모델",
        success=meta_success_flag,
        reason=f"수익률도달:{meta_success_flag}",
        rate=expected_ret,
        return_value=actual_return_meta,
        source="진화형" if use_evo else "기본",
        group_id=all_model_predictions[0].get("group_id"),
        feature_vector=feature_tensor.numpy()
    )

    return {
        "symbol": symbol,
        "strategy": strategy,
        "model": "meta",
        "class": final_pred_class,
        "expected_return": expected_ret,
        "timestamp": now_kst().isoformat(),
        "reason": "진화형 메타 최종 선택" if use_evo else "최고 확률 단일 모델 선택",
        "source": source
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
                group_id = int(r.get("group_id", 0)) if str(r.get("group_id")).isdigit() else 0

                pred_class = int(float(r.get("predicted_class", -1))) if pd.notnull(r.get("predicted_class")) else -1
                label = int(float(r.get("label", -1))) if pd.notnull(r.get("label")) else -1
                r["label"] = label

                entry_price = float(r.get("entry_price", 0))
                if entry_price <= 0 or label == -1:
                    reason = "entry_price 오류 또는 label=-1"
                    r.update({"status": "fail", "reason": reason, "return": 0.0})
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
                    r.update({"status": "fail", "reason": "timestamp 파싱 실패", "return": 0.0})
                    updated.append(r); continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("Asia/Seoul")
                else:
                    timestamp = timestamp.tz_convert("Asia/Seoul")

                eval_hours = eval_horizon_map.get(strategy, 6)
                deadline = timestamp + pd.Timedelta(hours=eval_hours)

                df = get_price_fn(symbol, strategy)
                if df is None or "timestamp" not in df.columns:
                    r.update({"status": "fail", "reason": "가격 데이터 없음", "return": 0.0})
                    updated.append(r); continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                future_df = df[df["timestamp"] >= timestamp]
                if future_df.empty:
                    r.update({"status": "fail", "reason": "미래 데이터 없음", "return": 0.0})
                    updated.append(r); continue

                actual_max = future_df["high"].max()
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
                        r.update({"status": "pending", "reason": "⏳ 평가 대기 중", "return": round(gain, 5)})
                        updated.append(r); continue
                else:
                    status = "success" if reached_target else "fail"

                vol = str(r.get("volatility", "")).lower() in ["1", "true"]
                if vol:
                    status = "v_success" if status == "success" else "v_fail"

                r.update({
                    "status": status,
                    "reason": f"[pred_class={pred_class}] gain={gain:.3f} (cls_min={cls_min}, cls_max={cls_max})",
                    "return": round(gain, 5),
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

                if model == "meta":
                    update_model_success(symbol, strategy, model, status in ["success", "v_success"])

                evaluated.append({str(k): (v if v is not None else "") for k, v in r.items()})
            except Exception as e:
                r.update({"status": "fail", "reason": f"예외: {e}", "return": 0.0})
                updated.append(r)

    def safe_write_csv(path, rows):
        if not rows:
            return
        fieldnames = sorted({str(k) for row in rows for k in row.keys() if k is not None})
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    updated += evaluated
    safe_write_csv(PREDICTION_LOG, updated)
    safe_write_csv(EVAL_RESULT, evaluated)
    failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
    safe_write_csv(WRONG, failed)

    print(f"[✅ 평가 완료] 총 {len(evaluated)}건 평가, 실패 {len(failed)}건")

# -----------------------------
# 개별 모델 예측 취합
# -----------------------------
def get_model_predictions(symbol, strategy, models, df, feat_scaled, window_list, recent_freq):
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

            # 그룹별 윈도우
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
                predicted_class = torch.argmax(softmax_probs, dim=1).item()
                probs = softmax_probs.squeeze().cpu().numpy()

            model_outputs_list.append({
                "probs": probs,
                "predicted_class": predicted_class,
                "group_id": group_id,
                "model_type": model_type,
                "model_path": model_path,
                "val_f1": val_f1,  # ✅ 성능 가중치에 사용
                "symbol": symbol, "strategy": strategy
            })

            entry_price = df["close"].iloc[-1]
            all_model_predictions.append({
                "class": predicted_class, "probs": probs, "entry_price": float(entry_price),
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
