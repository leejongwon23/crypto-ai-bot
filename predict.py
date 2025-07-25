import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
from data.utils import get_kline_by_strategy, compute_features
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import log_prediction
from failure_db import insert_failure_record, load_existing_failure_hashes
from logger import get_feature_hash
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
from logger import get_available_models
import json
from model.base_model import get_model, XGBoostWrapper
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE
NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
from config import get_class_groups
from collections import OrderedDict
# 변경
from config import get_class_ranges
# ✅ 기존 predict.py 맨 위 import 구역에 추가
from meta_learning import train_meta_learner, load_meta_learner
import safe_cleanup  # ✅ 오래된 로그 자동 정리


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


def ensemble_stacking(model_outputs, meta_model=None):
    import numpy as np

    X_stack = np.array(model_outputs)
    X_stack = X_stack.reshape(1, -1)

    if meta_model is not None:
        pred = meta_model.predict(X_stack)
        return int(pred[0])
    else:
        avg_probs = np.mean(model_outputs, axis=0)
        return int(np.argmax(avg_probs))

def class_to_expected_return(cls, recent_days=3):
    import pandas as pd
    import numpy as np
    from config import get_class_ranges

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
            return -0.01

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        df = df[df["predicted_class"].notna() & df["return"].notna()]
        df["predicted_class"] = df["predicted_class"].astype(int)

        centers_dynamic = df.groupby("predicted_class")["return"].mean().to_dict()

        # ✅ get_class_ranges 연동
        class_ranges = get_class_ranges()
        centers_default = [np.mean([low, high]) for (low, high) in class_ranges]

        if 0 <= cls < len(centers_default):
            if cls in centers_dynamic and np.isfinite(centers_dynamic[cls]):
                return centers_dynamic[cls]
            else:
                return centers_default[cls]

        print(f"[⚠️ 예상 수익률 계산 오류] 잘못된 클래스: {cls}")
        return centers_default[0]

    except Exception as e:
        print(f"[오류] class_to_expected_return 동적 매핑 실패 → {e}")

        # ✅ fallback
        class_ranges = get_class_ranges()
        centers_default = [np.mean([low, high]) for (low, high) in class_ranges]
        if 0 <= cls < len(centers_default):
            return centers_default[cls]
        return centers_default[0]


# ✅ 수정 요약:
# - failed_result(): label=-1 기본 포함
# - predict(): log_prediction() 호출 시 label 추가

def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    import numpy as np
    from datetime import datetime
    import pytz

    now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))
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
        # ✅ 실패 예측도 log_prediction 기록
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

    # ✅ 실패 DB 기록 추가 (feature_hash 포함)
    if X_input is not None and isinstance(X_input, np.ndarray):
        try:
            feature_hash = get_feature_hash(X_input)
            insert_failure_record(result, feature_hash, feature_vector=X_input.tolist(), label=label_val)
        except Exception as e:
            print(f"[failed_result insert_failure_record 오류] {e}")

    return result
def predict(symbol, strategy, source="일반", model_type=None):
    import numpy as np, pandas as pd, os, torch, json
    from sklearn.preprocessing import MinMaxScaler
    from window_optimizer import find_best_windows
    from logger import log_prediction, get_available_models
    from config import FEATURE_INPUT_SIZE
    from model_weight_loader import load_model_cached, get_class_return_range, class_to_expected_return
    from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
    from meta_learning import train_meta_learner, load_meta_learner, ensemble_stacking
    from data.utils import get_kline_by_strategy, compute_features
    from datetime import datetime
    import pytz
    from failure_db import insert_failure_record
    from logger import get_feature_hash

    os.makedirs("/persistent/logs", exist_ok=True)
    def now_kst(): return datetime.now(pytz.timezone("Asia/Seoul"))

    model_outputs_list, true_labels = [], []
    all_model_predictions = []

    try:
        window_list = find_best_windows(symbol, strategy)
        if not window_list:
            insert_failure_record(symbol, strategy, -1, -1, now_kst())
            return None

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 1:
            insert_failure_record(symbol, strategy, -1, -1, now_kst())
            return None

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < max(window_list) + 1:
            insert_failure_record(symbol, strategy, -1, -1, now_kst())
            return None

        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)
        input_size = feat_scaled.shape[1]
        if input_size < FEATURE_INPUT_SIZE:
            feat_scaled = np.pad(feat_scaled, ((0, 0), (0, FEATURE_INPUT_SIZE - input_size)), mode="constant")
        else:
            feat_scaled = feat_scaled[:, :FEATURE_INPUT_SIZE]

        models = get_available_models(symbol, strategy)
        if not models:
            print(f"[❌ 예측 실패] 모델 없음: {symbol}-{strategy}")
            insert_failure_record(symbol, strategy, -1, -1, now_kst())
            return None

        recent_freq = get_recent_class_frequencies(strategy)

        for window in window_list:
            if feat_scaled.shape[0] < window:
                continue
            X_input = feat_scaled[-window:]
            X = np.expand_dims(X_input, axis=0)

            for m in models:
                if f"_window{window}" not in m["pt_file"]:
                    continue

                model_path = os.path.join("/persistent/models", m["pt_file"])
                meta_path = model_path.replace(".pt", ".meta.json")

                if not os.path.exists(model_path) or not os.path.exists(meta_path):
                    print(f"[⚠️ 모델/메타 없음] {model_path}")
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
                num_classes = meta_info.get("num_classes", 21)
                group_id = m.get("group_id", 0)

                model = load_model_cached(model_path, m["model"], FEATURE_INPUT_SIZE, num_classes)
                if model is None:
                    print(f"[❌ 모델 로딩 실패] {model_path}")
                    continue

                with torch.no_grad():
                    logits = model(torch.tensor(X, dtype=torch.float32))
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

                adjusted_probs = adjust_probs_with_diversity(probs, recent_freq)
                final_class = int(np.argmax(adjusted_probs))
                model_outputs_list.append(adjusted_probs)

                all_model_predictions.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    "model_name": os.path.splitext(m["pt_file"])[0],
                    "model_type": m["model"],
                    "group_id": group_id,
                    "window": window,
                    "class": final_class,
                    "probs": adjusted_probs.tolist(),
                    "model_symbol": m["symbol"],
                    "entry_price": float(df.iloc[-1]["close"]),
                    "num_classes": num_classes
                })

        if not model_outputs_list:
            print(f"[❌ 예측 실패] 유효한 모델 없음")
            insert_failure_record(symbol, strategy, -1, -1, now_kst())
            return None

        meta_model = load_meta_learner()
        final_pred_class = ensemble_stacking(model_outputs_list, meta_model)

        cls_min, cls_max = get_class_return_range(final_pred_class)

        for pred in all_model_predictions:
            predicted_class = pred["class"]
            entry_price = pred["entry_price"]
            num_classes = pred["num_classes"]
            expected_return = class_to_expected_return(predicted_class, num_classes)
            target_price = entry_price * (1 + expected_return)
            is_main = (predicted_class == final_pred_class)

            success = cls_min <= expected_return <= cls_max if is_main else False

            log_prediction(
                symbol=pred["symbol"],
                strategy=pred["strategy"],
                direction="예측",
                entry_price=entry_price,
                target_price=target_price,
                model=pred["model_name"],
                success=success,
                reason="메타선택" if is_main else "미선택",
                rate=expected_return,
                return_value=expected_return,
                source=source,
                predicted_class=predicted_class,
                label=final_pred_class,
                group_id=pred["group_id"],
                model_symbol=pred["model_symbol"],
                model_name=pred["model_name"]
            )

            if not success:
                insert_failure_record(
                    symbol=pred["symbol"],
                    strategy=pred["strategy"],
                    predicted_class=predicted_class,
                    label=final_pred_class,
                    timestamp=now_kst()
                )

        return {
            "symbol": symbol,
            "strategy": strategy,
            "model": "meta",
            "class": final_pred_class,
            "expected_return": class_to_expected_return(final_pred_class, len(model_outputs_list[0])),
            "timestamp": now_kst().isoformat(),
            "reason": "메타 최종 선택",
            "source": source
        }

    except Exception as e:
        print(f"[predict 예외] {e}")
        insert_failure_record(symbol, strategy, -1, -1, now_kst())
        return None


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
    from config import get_class_ranges

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
            group_id = int(r.get("group_id", 0)) if str(r.get("group_id")).isdigit() else 0

            pred_class = int(float(r.get("predicted_class", -1))) if pd.notnull(r.get("predicted_class")) else -1
            label = int(float(r.get("label", -1))) if pd.notnull(r.get("label")) else -1
            r["label"] = label

            entry_price = float(r.get("entry_price", 0))
            if entry_price <= 0 or pred_class == -1:
                log_prediction(symbol, strategy, "예측실패", entry_price, entry_price, now_kst().isoformat(),
                               model, False, "entry_price 오류 또는 pred_class=-1", 0.0, 0.0, False, "평가",
                               predicted_class=pred_class, label=label, group_id=group_id)
                r.update({"status": "fail", "reason": "entry_price 오류 또는 pred_class=-1", "return": 0.0})
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

            # ✅ 그룹별 클래스 수익률 구간 반영
            class_ranges_for_group = get_class_ranges(group_id=group_id)
            success = False
            if 0 <= pred_class < len(class_ranges_for_group):
                cls_min, cls_max = class_ranges_for_group[pred_class]
                if cls_min <= gain <= cls_max:
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

