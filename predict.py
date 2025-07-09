import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import log_prediction
from failure_db import insert_failure_record, load_existing_failure_hashes
from logger import get_feature_hash
from config import NUM_CLASSES
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
from logger import get_available_models
import json
from model.base_model import get_model, XGBoostWrapper
from config import FEATURE_INPUT_SIZE
from config import get_class_groups
from collections import OrderedDict
# 변경
from config import get_class_ranges



DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ✅ MODEL_CACHE 수정
MODEL_CACHE = OrderedDict()
MODEL_CACHE_MAX_SIZE = 10  # 최대 10개만 캐싱

def load_model_cached(model_path, model_type, input_size, output_size):
    key = (model_path, model_type)
    if key in MODEL_CACHE:
        # ✅ 사용된 모델은 맨 뒤로 이동 (LRU)
        MODEL_CACHE.move_to_end(key)
        model = MODEL_CACHE[key]
    else:
        model = get_model(model_type, input_size, output_size).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        MODEL_CACHE[key] = model

        # ✅ 캐시 크기 초과 시 가장 오래된 항목 제거
        if len(MODEL_CACHE) > MODEL_CACHE_MAX_SIZE:
            removed_key, removed_model = MODEL_CACHE.popitem(last=False)
            print(f"[🗑️ MODEL_CACHE 제거] {removed_key}")

    # ✅ input_size, output_size 검증 (기존 로직 유지)
    meta_path = model_path.replace(".pt", ".meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_input = meta.get("input_size")
            expected_output = meta.get("output_size")
            if expected_input != input_size or expected_output != output_size:
                print(f"[❌ 모델 크기 불일치] expected input:{expected_input}, output:{expected_output} | got input:{input_size}, output:{output_size}")
                return None
        except Exception as e:
            print(f"[⚠️ meta.json 로드 오류] {meta_path} → {e}")

    return model


    
def class_to_expected_return(cls, recent_days=3):
    import pandas as pd
    import numpy as np

    # ✅ cls 타입 강제 변환
    try:
        cls = int(cls)
    except:
        cls = -1

    try:
        df = pd.read_csv("/persistent/prediction_log.csv", encoding="utf-8-sig")

        # ✅ predicted_class 컬럼 존재 여부 확인
        if "predicted_class" not in df.columns:
            print("[❌ 오류] prediction_log.csv에 predicted_class 컬럼이 없습니다.")
            return -0.01  # ✅ 컬럼 없으면 fallback 기본값 반환

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        df = df[df["predicted_class"].notna() & df["return"].notna()]
        df["predicted_class"] = df["predicted_class"].astype(int)

        centers_dynamic = df.groupby("predicted_class")["return"].mean().to_dict()

        centers_default = [
            -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
            -0.02, 0.0, 0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
            0.75, 1.50, 3.50
        ]

        if 0 <= cls < len(centers_default):
            if cls in centers_dynamic and np.isfinite(centers_dynamic[cls]):
                return centers_dynamic[cls]
            else:
                return centers_default[cls]

        print(f"[⚠️ 예상 수익률 계산 오류] 잘못된 클래스: {cls}")
        return centers_default[0]  # ✅ fallback 첫 번째 값으로 강제

    except Exception as e:
        print(f"[오류] class_to_expected_return 동적 매핑 실패 → {e}")
        centers_default = [
            -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
            -0.02, 0.0, 0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
            0.75, 1.50, 3.50
        ]
        if 0 <= cls < len(centers_default):
            return centers_default[cls]
        return centers_default[0]


# ✅ 수정 요약:
# - failed_result(): label=-1 기본 포함
# - predict(): log_prediction() 호출 시 label 추가

def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
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
        # ✅ entry_price, predicted_class, label 유효성 검사
        valid_entry_price = isinstance(result.get("rate", 0.0), (int, float))
        valid_pred_class = isinstance(pred_class_val, int) and pred_class_val >= -1
        valid_label = isinstance(label_val, int) and label_val >= -1

        if valid_entry_price and valid_pred_class and valid_label:
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
        else:
            print(f"[failed_result 검증 실패] entry_price:{valid_entry_price}, pred_class:{valid_pred_class}, label:{valid_label}")

    except Exception as e:
        print(f"[failed_result log_prediction 오류] {e}")

    # 실패 DB 기록 추가 (feature_hash 검증 후)
    if X_input is not None and isinstance(X_input, np.ndarray):
        try:
            feature_hash = get_feature_hash(X_input)
            insert_failure_record(result, feature_hash, feature_vector=X_input.tolist(), label=label_val)
        except Exception as e:
            print(f"[failed_result insert_failure_record 오류] {e}")

    return result

def predict(symbol, strategy, source="일반", model_type=None):
    from scipy.stats import entropy
    from window_optimizer import find_best_windows
    from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity

    try:
        max_retry = 3
        retry = 0
        class_groups = get_class_groups()

        while retry < max_retry:
            window_list = find_best_windows(symbol, strategy)
            if not window_list:
                retry += 1
                continue

            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < max(window_list) + 1:
                retry += 1
                continue

            feat = compute_features(symbol, df, strategy)
            if feat is None or feat.dropna().shape[0] < max(window_list) + 1:
                retry += 1
                continue

            features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
            feat_scaled = MinMaxScaler().fit_transform(features_only)
            input_size = feat_scaled.shape[1]

            # ✅ FEATURE_INPUT_SIZE로 강제 통일
            if input_size < FEATURE_INPUT_SIZE:
                pad_cols = FEATURE_INPUT_SIZE - input_size
                feat_scaled = np.pad(feat_scaled, ((0,0),(0,pad_cols)), mode="constant", constant_values=0)
                input_size = FEATURE_INPUT_SIZE
            elif input_size > FEATURE_INPUT_SIZE:
                feat_scaled = feat_scaled[:, :FEATURE_INPUT_SIZE]
                input_size = FEATURE_INPUT_SIZE

            models = get_available_models()
            if not models:
                return [failed_result(symbol, strategy, "unknown", "모델 없음 → 학습 필요", source)]

            pred_classes = []
            recent_freq = get_recent_class_frequencies(strategy)
            for trial in range(3):
                ensemble_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
                for window in window_list:
                    if feat_scaled.shape[0] < window:
                        continue
                    X_input = feat_scaled[-window:]
                    X = np.expand_dims(X_input, axis=0)

                    for m in models:
                        if m["symbol"] != symbol or m["strategy"] != strategy:
                            continue
                        if f"_window{window}" not in m["pt_file"]:
                            continue

                        group_id = m.get("group_id")
                        if group_id is None:
                            continue
                        group_classes = class_groups[group_id]

                        model_path = os.path.join(MODEL_DIR, m["pt_file"])
                        meta_path = model_path.replace(".pt", ".meta.json")
                        if not os.path.exists(model_path) or not os.path.exists(meta_path):
                            continue

                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)

                        model_input_size = meta.get("input_size")
                        if model_input_size != FEATURE_INPUT_SIZE:
                            continue

                        # ✅ 캐싱된 모델 로드 적용
                        model = load_model_cached(model_path, m["model"], FEATURE_INPUT_SIZE, len(group_classes))

                        # ✅ [추가] 모델 None 체크
                        if model is None:
                            print(f"[⚠️ skip] {model_path} input/output 크기 불일치로 스킵")
                            continue

                        with torch.no_grad():
                            logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

                        adjusted_probs = adjust_probs_with_diversity(probs, recent_freq)
                        for i, cls in enumerate(group_classes):
                            ensemble_probs[cls] += adjusted_probs[i]

                pred_class = int(ensemble_probs.argmax()) if ensemble_probs.sum() > 0 else -1
                pred_classes.append(pred_class)

            # ✅ 핵심 요약 로그만 출력
            print(f"[predict] {symbol}-{strategy} 결과: {pred_classes}, 최종={pred_classes[0]}")

            if len(set(pred_classes)) == 1 and pred_classes[0] != -1:
                final_pred_class = pred_classes[0]
                expected_return = class_to_expected_return(final_pred_class)

                # ✅ conf_score 안전 처리
                conf_score = 1 - entropy(ensemble_probs + 1e-6) / np.log(len(ensemble_probs))
                if not np.isfinite(conf_score):
                    conf_score = 0.0

                log_prediction(
                    symbol=symbol, strategy=strategy, direction=f"Ensemble-Class-{final_pred_class}",
                    entry_price=df["close"].iloc[-1],
                    target_price=df["close"].iloc[-1] * (1 + expected_return),
                    model="ensemble", success=True, reason=f"Self-Consistency Ensemble | confidence={conf_score:.4f}",
                    rate=expected_return, timestamp=now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                    return_value=expected_return, volatility=True, source=source,
                    predicted_class=final_pred_class, label=final_pred_class
                )

                return [{
                    "symbol": symbol, "strategy": strategy, "model": "ensemble",
                    "class": final_pred_class, "expected_return": expected_return,
                    "success": True, "predicted_class": final_pred_class,
                    "label": final_pred_class, "confidence": round(conf_score, 4)
                }]
            else:
                print(f"[predict] {symbol}-{strategy} Self-Consistency 실패")
                return [failed_result(symbol, strategy, "unknown", "Self-Consistency 실패 (3회 예측 불일치)", source)]

        retry += 1
        return [failed_result(symbol, strategy, "unknown", "다중윈도우 Self-Consistency 실패", source)]

    except Exception as e:
        print(f"[predict 예외] {symbol}-{strategy} → {e}")
        return [failed_result(symbol, strategy, "unknown", f"예외 발생: {e}", source)]


# 📄 predict.py 내부에 추가
import csv, datetime, pytz, os
import pandas as pd
from failure_db import ensure_failure_db, insert_failure_record
from logger import update_model_success


def evaluate_predictions(get_price_fn):
    import csv, os, datetime, pytz
    import pandas as pd
    from failure_db import ensure_failure_db, insert_failure_record
    from logger import update_model_success, log_prediction
    class_ranges = get_class_ranges()

    ensure_failure_db()

    PREDICTION_LOG = "/persistent/prediction_log.csv"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    date_str = now_kst().strftime("%Y-%m-%d")
    EVAL_RESULT = f"/persistent/logs/evaluation_{date_str}.csv"
    WRONG = f"/persistent/logs/wrong_{date_str}.csv"


    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}
    updated, evaluated = [], []

    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            return
    except Exception as e:
        print(f"[오류] prediction_log.csv 읽기 실패 → {e}")
        return

    for r in rows:
        try:
            if r.get("status") not in [None, "", "pending", "v_pending"]:
                updated.append(r)
                continue

            symbol = r.get("symbol", "UNKNOWN")
            strategy = r.get("strategy", "알수없음")
            model = r.get("model", "unknown")
            group_id = r.get("group_id", "")

            pred_class = int(float(r.get("predicted_class", -1))) if pd.notnull(r.get("predicted_class")) else -1
            label = int(float(r.get("label", -1))) if pd.notnull(r.get("label")) else -1
            r["label"] = label

            entry_price = float(r.get("entry_price", 0))
            if entry_price <= 0 or pred_class == -1:
                log_prediction(symbol, strategy, "예측실패", entry_price, entry_price, now_kst().isoformat(),
                               model, False, "entry_price 오류 또는 pred_class=-1", 0.0, 0.0, False, "평가",
                               predicted_class=pred_class, label=label, group_id=group_id)
                r.update({"status": "fail", "reason": "entry_price 오류 또는 pred_class=-1", "return": 0.0})

                # ✅ 실패 샘플 DB insert 추가
                insert_failure_record(r, f"{symbol}-{strategy}-{now_kst().isoformat()}", feature_vector=None, label=label)

                updated.append(r)
                continue

            timestamp = pd.to_datetime(r.get("timestamp"), utc=True).tz_convert("Asia/Seoul")
            deadline = timestamp + pd.Timedelta(hours=eval_horizon_map.get(strategy, 6))
            if now_kst() < deadline:
                r.update({"reason": "⏳ 평가 대기 중", "return": 0.0})
                updated.append(r)
                continue

            df = get_price_fn(symbol, strategy)
            if df is None or "timestamp" not in df.columns:
                r.update({"status": "fail", "reason": "가격 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
            future_df = df[(df["timestamp"] >= timestamp) & (df["timestamp"] <= deadline)]
            if future_df.empty:
                r.update({"status": "fail", "reason": "미래 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            actual_max = future_df["high"].max()
            gain = (actual_max - entry_price) / (entry_price + 1e-6)

            # ✅ 수정된 평가 로직: 예측 클래스 구간(min~max) 내에 들어가야 성공으로 판정
            success = False
            if 0 <= pred_class < len(class_ranges):
                cls_min, cls_max = class_ranges[pred_class]
                if cls_min <= gain < cls_max:
                    success = True

            vol = str(r.get("volatility", "")).lower() in ["1", "true"]
            status = "v_success" if vol and success else \
                     "v_fail" if vol and not success else \
                     "success" if success else "fail"

            confidence = float(r.get("confidence", 0.0)) if "confidence" in r else 0.0

            r.update({
                "status": status,
                "reason": f"[cls={pred_class}] gain={gain:.3f}",
                "return": round(gain, 5),
                "confidence": confidence,
                "label": label,
                "group_id": group_id
            })

            log_prediction(symbol, strategy, f"평가:{status}", entry_price,
                           entry_price * (1 + gain), now_kst().isoformat(), model,
                           success, r["reason"], gain, gain, vol, "평가",
                           predicted_class=pred_class, label=label, group_id=group_id)

            # ✅ 실패 샘플 DB insert 추가
            if not success:
                insert_failure_record(r, f"{symbol}-{strategy}-{now_kst().isoformat()}", feature_vector=None, label=label)

            r_clean = {str(k): (v if v is not None else "") for k, v in r.items() if k is not None}
            update_model_success(symbol, strategy, model, success)
            evaluated.append(r_clean)

        except Exception as e:
            r.update({"status": "fail", "reason": f"예외: {e}", "return": 0.0})
            updated.append(r)

    updated += evaluated

    def safe_write_csv(path, rows):
        if not rows:
            return
        fieldnames = sorted({str(k) for row in rows for k in row.keys() if k is not None})
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    safe_write_csv(PREDICTION_LOG, updated)
    safe_write_csv(EVAL_RESULT, evaluated)
    failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
    safe_write_csv(WRONG, failed)
    print(f"[✅ 평가 완료] 총 {len(evaluated)}건 평가, 실패 {len(failed)}건")

def get_class_distribution(symbol, strategy, model_type):
    import os, json
    meta_path = f"/persistent/models/{symbol}_{strategy}_{model_type}.meta.json"
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("class_counts", {})
    except Exception as e:
        print(f"[⚠️ 클래스 분포 로드 실패] {meta_path} → {e}")
    return {}

if __name__ == "__main__":
    results = predict("BTCUSDT", "단기")
    print(results)

    try:
        df = pd.read_csv("/persistent/prediction_log.csv", encoding="utf-8-sig")
        print("[✅ prediction_log.csv 상위 20줄 출력]")
        print(df.head(20))
    except Exception as e:
        print(f"[오류] prediction_log.csv 로드 실패 → {e}")

