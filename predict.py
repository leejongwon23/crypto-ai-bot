import os, sys, json, datetime, pytz
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from data.utils import get_kline_by_strategy, compute_features
from window_optimizer import find_best_windows
from logger import log_prediction, get_feature_hash, get_available_models, update_model_success
from failure_db import insert_failure_record, load_existing_failure_hashes, ensure_failure_db
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
from model.base_model import get_model
from model_weight_loader import load_model_cached  # ✅ 표준 캐시 로더 사용
from config import (
    get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups,
    get_class_return_range, class_to_expected_return
)

# =========================
# 기본 상수/헬퍼
# =========================
DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
LOG_DIR = "/persistent/logs"
PREDICTION_LOG_PATH = os.path.join(LOG_DIR, "prediction_log.csv")  # ✅ 경로 통일
NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# =========================
# 앙상블 (단순 평균/스태킹)
# =========================
def ensemble_stacking(model_outputs, meta_model=None):
    X_stack = np.array(model_outputs).reshape(1, -1)
    if meta_model is not None:
        pred = meta_model.predict(X_stack)
        return int(pred[0])
    else:
        avg_probs = np.mean(model_outputs, axis=0)
        return int(np.argmax(avg_probs))

# =========================
# 실패 결과 공통 포맷
# =========================
def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    from datetime import datetime as _dt
    import numpy as _np
    t = _dt.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

    pred_class_val = -1
    label_val = -1
    result = {
        "symbol": symbol,
        "strategy": strategy,
        "success": False,
        "reason": reason,
        "model": str(model_type or "unknown"),
        "rate": 0.0,
        "class": pred_class_val,
        "timestamp": t,
        "source": source,
        "predicted_class": pred_class_val,
        "label": label_val
    }

    try:
        log_prediction(
            symbol=symbol,
            strategy=strategy,
            direction="예측실패",
            entry_price=0,
            target_price=0,
            model=str(model_type or "unknown"),
            success=False,
            reason=reason,
            rate=0.0,
            timestamp=t,
            return_value=0.0,
            volatility=True,
            source=source,
            predicted_class=pred_class_val,
            label=label_val
        )
    except Exception as e:
        print(f"[failed_result log_prediction 오류] {e}")

    if X_input is not None and isinstance(X_input, np.ndarray):
        try:
            feature_hash = get_feature_hash(X_input)
            insert_failure_record(result, feature_hash, feature_vector=X_input.tolist(), label=label_val)
        except Exception as e:
            print(f"[failed_result insert_failure_record 오류] {e}")

    return result

# =========================
# 메인 예측
# =========================
def predict(symbol, strategy, source="일반", model_type=None):
    """
    - 전 모델 참여(품질 필터 제거)
    - 메타러너 + (있으면) 진화형 메타러너 적용
    - 성공 판정/목표가 심볼·전략별 클래스 경계와 일관
    """
    from evo_meta_learner import get_best_strategy_by_failure_probability, predict_evo_meta

    ensure_failure_db()
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1) 입력 검증
    if not symbol or not strategy:
        insert_failure_record({"symbol": symbol or "None", "strategy": strategy or "None"},
                              "invalid_symbol_strategy", label=-1)
        return None

    log_strategy = strategy  # 로깅상 원 전략 유지
    try:
        # 2) 최적 윈도우 탐색
        window_list = find_best_windows(symbol, strategy)
        if not window_list:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "window_list_none", label=-1)
            return None

        # 3) 가격 데이터
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 1:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "df_short", label=-1)
            return None

        # 4) 피처
        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < max(window_list) + 1:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "feature_short", label=-1)
            return None

        # 5) 스케일링/패딩
        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)
        if feat_scaled.shape[1] < FEATURE_INPUT_SIZE:
            feat_scaled = np.pad(feat_scaled, ((0, 0), (0, FEATURE_INPUT_SIZE - feat_scaled.shape[1])), mode="constant")
        else:
            feat_scaled = feat_scaled[:, :FEATURE_INPUT_SIZE]

        # 6) 사용 가능한 모든 모델
        models = get_available_models(symbol)
        if not models:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "no_models", label=-1)
            return None

        # 7) 최근 클래스 분포(다양성 보정 등에 사용 가능)
        recent_freq = get_recent_class_frequencies(strategy)
        feature_tensor = torch.tensor(feat_scaled[-1], dtype=torch.float32)

        # 8) 개별 모델 추론
        model_outputs_list, all_model_predictions = get_model_predictions(
            symbol, strategy, models, df, feat_scaled, window_list, recent_freq
        )
        if not model_outputs_list:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "no_valid_model", label=-1)
            return None

        # 9) 진화형 실패확률 기반 전략 교체
        recommended_strategy = get_best_strategy_by_failure_probability(
            symbol=symbol, current_strategy=strategy,
            feature_tensor=feature_tensor, model_outputs=model_outputs_list
        )
        if recommended_strategy and recommended_strategy != strategy:
            print(f"[🔁 전략 교체됨] {strategy} → {recommended_strategy}")
            strategy = recommended_strategy

        # 10) 메타러너 최종 선택
        from meta_learning import get_meta_prediction  # 로컬 임포트(순환 최소화)
        meta_success_rate = {c: 0.5 for c in range(len(model_outputs_list[0]["probs"]))}  # 필터링 제거
        final_pred_class = get_meta_prediction(
            [m["probs"] for m in model_outputs_list],
            feature_tensor,
            meta_info={"success_rate": meta_success_rate}
        )

        # 11) 진화형 메타 적용(있을 때만)
        evo_model_path = os.path.join(MODEL_DIR, "evo_meta_learner.pt")
        use_evo = False
        if os.path.exists(evo_model_path):
            try:
                evo_pred = predict_evo_meta(feature_tensor.unsqueeze(0), input_size=FEATURE_INPUT_SIZE)
                if evo_pred is not None and evo_pred != final_pred_class:
                    print(f"[🔁 진화형 메타러너 전환] {final_pred_class} → {evo_pred}")
                    final_pred_class = evo_pred
                    use_evo = True
            except Exception as e:
                print(f"[⚠️ 진화형 메타러너 예외] {e}")

        print(f"[META] {'진화형' if use_evo else '기본'} 메타 선택: 클래스 {final_pred_class}")

        # 12) 메타 결과 로깅 + 성공 가능성 판단(동일 경계)
        cls_min, _ = get_class_return_range(final_pred_class, symbol, strategy)
        current_price = df.iloc[-1]["close"]
        expected_ret = class_to_expected_return(final_pred_class, symbol, strategy)
        entry_price = all_model_predictions[0]["entry_price"]
        actual_return_meta = (current_price / entry_price) - 1
        meta_success_flag = actual_return_meta >= cls_min

        if not meta_success_flag:
            insert_failure_record(
                {"symbol": symbol, "strategy": log_strategy},
                "meta_predicted_fail", label=final_pred_class, feature_vector=feature_tensor.numpy()
            )

        log_prediction(
            symbol=symbol,
            strategy=log_strategy,
            direction="예측",
            entry_price=entry_price,
            target_price=entry_price * (1 + expected_ret),
            model="meta",
            model_name="evo_meta_learner",
            predicted_class=final_pred_class,
            label=final_pred_class,
            note="진화형 메타 선택" if use_evo else "기본 메타 선택",
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
            "strategy": log_strategy,
            "model": "meta",
            "class": final_pred_class,
            "expected_return": expected_ret,
            "timestamp": now_kst().isoformat(),
            "reason": "진화형 메타 최종 선택" if use_evo else "기본 메타 최종 선택",
            "source": source
        }

    except Exception as e:
        insert_failure_record({"symbol": symbol or "None", "strategy": strategy or "None"}, "exception", label=-1)
        return None


# =========================
# 평가 로직 (심볼·전략별 경계 사용)
# =========================
def evaluate_predictions(get_price_fn):
    import csv, os
    import pandas as pd
    from collections import defaultdict
    from failure_db import check_failure_exists

    ensure_failure_db()

    PREDICTION_LOG = PREDICTION_LOG_PATH  # ✅ 통일 경로 사용
    now_local = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    date_str = now_local().strftime("%Y-%m-%d")
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
                    updated.append(r)
                    continue

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
                    log_prediction(symbol, strategy, "예측실패", entry_price, entry_price, now_local().isoformat(),
                                   model, False, reason, 0.0, 0.0, False, "평가",
                                   predicted_class=pred_class, label=label, group_id=group_id)
                    if not check_failure_exists(r):
                        insert_failure_record(r, f"{symbol}-{strategy}-{now_local().isoformat()}",
                                              feature_vector=None, label=label)
                    updated.append(r)
                    continue

                timestamp = pd.to_datetime(r.get("timestamp"), errors="coerce")
                if timestamp is None or pd.isna(timestamp):
                    r.update({"status": "fail", "reason": "timestamp 파싱 실패", "return": 0.0})
                    updated.append(r)
                    continue
                # 타임존 처리
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("Asia/Seoul")
                else:
                    timestamp = timestamp.tz_convert("Asia/Seoul")

                eval_hours = eval_horizon_map.get(strategy, 6)
                deadline = timestamp + pd.Timedelta(hours=eval_hours)

                df = get_price_fn(symbol, strategy)
                if df is None or "timestamp" not in df.columns:
                    r.update({"status": "fail", "reason": "가격 데이터 없음", "return": 0.0})
                    updated.append(r)
                    continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                future_df = df[df["timestamp"] >= timestamp]
                if future_df.empty:
                    r.update({"status": "fail", "reason": "미래 데이터 없음", "return": 0.0})
                    updated.append(r)
                    continue

                actual_max = future_df["high"].max()
                gain = (actual_max - entry_price) / (entry_price + 1e-6)

                # ✅ 심볼·전략별 경계로 평가
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
                        updated.append(r)
                        continue
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
                    symbol, strategy, f"평가:{status}", entry_price,
                    entry_price * (1 + gain), now_local().isoformat(), model,
                    status in ["success", "v_success"], r["reason"], gain, gain, vol, "평가",
                    predicted_class=pred_class, label=label, group_id=group_id
                )

                if status in ["fail", "v_fail"] and not check_failure_exists(r):
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

# =========================
# 개별 모델 예측 실행
# =========================
def get_model_predictions(symbol, strategy, models, df, feat_scaled, window_list, recent_freq):
    """
    ✅ [YOPO 전용]
    - 주어진 모델 리스트(models)로 예측 수행
    - 각 모델은 meta.json을 통해 정보 추출
    - 결과: model_outputs_list, all_model_predictions 반환
    """
    import json

    model_outputs_list = []
    all_model_predictions = []

    for model_info in models:
        try:
            # logger.get_available_models() 스펙 기준
            pt_file = model_info.get("pt_file")
            if not pt_file:
                print(f"[⚠️ pt_file 누락] {model_info}")
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

            # 윈도우 추출
            window = window_list[group_id]
            input_seq = feat_scaled[-window:]
            if input_seq.shape[0] < window:
                print(f"[⚠️ 데이터 부족] {symbol}-{strategy}-group{group_id}")
                continue

            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # (1, window, input_size)

            # ✅ 표준 캐시 로더 사용 (state_dict)
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
                "symbol": symbol,
                "strategy": strategy
            })

            entry_price = df["close"].iloc[-1]
            all_model_predictions.append({
                "class": predicted_class,
                "probs": probs,
                "entry_price": entry_price,
                "num_classes": num_classes,
                "group_id": group_id,
                "model_name": model_type,
                "model_symbol": symbol,
                "symbol": symbol,
                "strategy": strategy
            })

        except Exception as e:
            print(f"[❌ 모델 예측 실패] {model_info} → {e}")
            continue

    return model_outputs_list, all_model_predictions

# =========================
# 직접 실행 테스트
# =========================
if __name__ == "__main__":
    res = predict("BTCUSDT", "단기")
    print(res)

    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig")
        print("[✅ prediction_log.csv 상위 20줄 출력]")
        print(df.head(20))
    except Exception as e:
        print(f"[오류] prediction_log.csv 로드 실패 → {e}")
