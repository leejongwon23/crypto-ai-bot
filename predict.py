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


DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))


def class_to_expected_return(cls, recent_days=3):
    import pandas as pd
    import numpy as np

    try:
        df = pd.read_csv("/persistent/prediction_log.csv", encoding="utf-8-sig")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        df = df[df["predicted_class"].notna() & df["return"].notna()]
        df["predicted_class"] = df["predicted_class"].astype(int)

        # ✅ 최근 n일간 클래스별 평균 수익률 계산
        centers_dynamic = df.groupby("predicted_class")["return"].mean().to_dict()

        # ✅ 기본 centers (fallback)
        centers_default = [
            -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
            -0.02, 0.0, 0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
            0.75, 1.50, 3.50
        ]

        # ✅ cls 범위 안전 체크 및 반환 로직 강화
        if isinstance(cls, int) and 0 <= cls < len(centers_default):
            if cls in centers_dynamic and np.isfinite(centers_dynamic[cls]):
                return centers_dynamic[cls]
            else:
                return centers_default[cls]

        print(f"[⚠️ 예상 수익률 계산 오류] 잘못된 클래스: {cls}")
        return 0.0

    except Exception as e:
        print(f"[오류] class_to_expected_return 동적 매핑 실패 → {e}")
        # ✅ 실패 시 기본 centers 반환
        centers_default = [
            -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
            -0.02, 0.0, 0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
            0.75, 1.50, 3.50
        ]
        if isinstance(cls, int) and 0 <= cls < len(centers_default):
            return centers_default[cls]
        return 0.0



# ✅ 수정 요약:
# - failed_result(): label=-1 기본 포함
# - predict(): log_prediction() 호출 시 label 추가

def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    pred_class_val = -1
    label_val = -1  # ✅ 실패시 label=-1 통일

    # ✅ 타입 강제 고정
    if not isinstance(pred_class_val, int):
        pred_class_val = -1
    if not isinstance(label_val, int):
        label_val = -1

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
    except:
        pass

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

    if X_input is not None:
        try:
            feature_hash = get_feature_hash(X_input)
            insert_failure_record(result, feature_hash, feature_vector=X_input, label=label_val)
        except:
            pass

    return result


def predict(symbol, strategy, source="일반", model_type=None):
    import os, json, torch, numpy as np, pandas as pd, datetime, pytz, sys
    from sklearn.preprocessing import MinMaxScaler
    from data.utils import get_kline_by_strategy, compute_features
    from model.base_model import get_model, XGBoostWrapper
    from model_weight_loader import get_model_weight
    from window_optimizer import find_best_window
    from logger import log_prediction, get_feature_hash
    from failure_db import insert_failure_record
    from config import NUM_CLASSES
    from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
    from logger import get_available_models
    from scipy.stats import entropy

    DEVICE = torch.device("cpu")
    MODEL_DIR = "/persistent/models"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

    try:
        max_retry = 2
        retry = 0

        while retry < max_retry:
            window = find_best_window(symbol, strategy)
            if not isinstance(window, int) or window <= 0:
                retry += 1
                continue

            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < window + 1:
                retry += 1
                continue

            feat = compute_features(symbol, df, strategy)
            if feat is None or feat.dropna().shape[0] < window + 1:
                retry += 1
                continue

            features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
            feat_scaled = MinMaxScaler().fit_transform(features_only)
            if feat_scaled.shape[0] < window:
                retry += 1
                continue

            X_input = feat_scaled[-window:]
            X = np.expand_dims(X_input, axis=0)
            input_size = X.shape[2]

            models = get_available_models()
            results, ensemble_probs, total_weight = [], None, 0.0

            for m in models:
                if m["symbol"] != symbol or m["strategy"] != strategy:
                    continue

                mt = m["model"]
                if model_type and mt != model_type:
                    continue

                model_path = os.path.join(MODEL_DIR, m["pt_file"])
                meta_path = model_path.replace(".pt", ".meta.json")

                if not os.path.exists(model_path) or not os.path.exists(meta_path):
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # ✅ input_size mismatch → window fallback 재시도
                if meta.get("input_size") != input_size:
                    print(f"[⚠️ input_size mismatch] {meta.get('input_size')} vs {input_size} → window fallback 재시도")
                    retry += 1
                    break  # while 재시도

                weight = get_model_weight(mt, strategy, symbol, input_size=input_size)
                if weight <= 0.0:
                    continue

                if mt == "xgboost":
                    try:
                        model = get_model(mt, input_size=input_size, model_path=model_path)
                        pred_class = int(model.predict(X)[0])
                    except Exception as e:
                        print(f"[XGBoost 예측 오류] {e}")
                        continue

                    expected_return = class_to_expected_return(pred_class)
                    label_val = pred_class

                    log_prediction(
                        symbol=symbol, strategy=strategy, direction=f"Class-{pred_class}",
                        entry_price=df["close"].iloc[-1],
                        target_price=df["close"].iloc[-1] * (1 + expected_return),
                        model=mt, success=True, reason="XGBoost 예측 완료",
                        rate=expected_return, timestamp=now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                        return_value=expected_return, volatility=True, source=source,
                        predicted_class=pred_class, label=label_val
                    )

                    results.append({
                        "symbol": symbol, "strategy": strategy, "model": mt,
                        "class": pred_class, "expected_return": expected_return,
                        "success": True, "predicted_class": pred_class,
                        "label": label_val, "confidence": 1.0
                    })
                    continue

                # ✅ torch 모델 predict
                model = get_model(mt, input_size, NUM_CLASSES).to(DEVICE)
                state = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state)
                model.eval()

                with torch.no_grad():
                    logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

                if ensemble_probs is None:
                    ensemble_probs = probs * weight
                else:
                    ensemble_probs += probs * weight
                total_weight += weight

                recent_freq = get_recent_class_frequencies(strategy=strategy)
                class_counts = meta.get("class_counts", {}) or {}
                adjusted_probs = adjust_probs_with_diversity(probs, recent_freq, class_counts)

                pred_class = int(adjusted_probs.argmax())
                expected_return = class_to_expected_return(pred_class)
                label_val = pred_class

                conf_score = 1 - entropy(probs) / np.log(len(probs))

                log_prediction(
                    symbol=symbol, strategy=strategy, direction=f"Class-{pred_class}",
                    entry_price=df["close"].iloc[-1],
                    target_price=df["close"].iloc[-1] * (1 + expected_return),
                    model=mt, success=True, reason=f"예측 완료 | confidence={conf_score:.4f}",
                    rate=expected_return, timestamp=now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                    return_value=expected_return, volatility=True, source=source,
                    predicted_class=pred_class, label=label_val
                )

                feature_hash = get_feature_hash(X_input)
                insert_failure_record({
                    "symbol": symbol, "strategy": strategy, "model": mt,
                    "class": pred_class, "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
                }, feature_hash, feature_vector=X_input, label=label_val)

                results.append({
                    "symbol": symbol, "strategy": strategy, "model": mt,
                    "class": pred_class, "expected_return": expected_return,
                    "success": True, "predicted_class": pred_class,
                    "label": label_val, "confidence": round(conf_score, 4)
                })
            else:
                # ✅ break 없이 끝났다면 while 루프 종료
                break

        # ✅ ensemble 최종 결과 추가
        if ensemble_probs is not None and total_weight > 0:
            ensemble_probs /= total_weight
            ensemble_class = int(ensemble_probs.argmax())
            ensemble_return = class_to_expected_return(ensemble_class)
            conf_score = 1 - entropy(ensemble_probs) / np.log(len(ensemble_probs))
            results.append({
                "symbol": symbol, "strategy": strategy, "model": "ensemble",
                "class": ensemble_class, "expected_return": ensemble_return,
                "success": True, "predicted_class": ensemble_class,
                "label": ensemble_class, "confidence": round(conf_score, 4)
            })

        if not results:
            return [failed_result(symbol, strategy, "unknown", "모델 예측 실패", source)]
        return results

    except Exception as e:
        print(f"[predict 예외] {e}")
        return [failed_result(symbol, strategy, "unknown", f"예외 발생: {e}", source)]



# 📄 predict.py 내부에 추가
import csv, datetime, pytz, os
import pandas as pd
from failure_db import ensure_failure_db, insert_failure_record
from logger import update_model_success

def evaluate_predictions(get_price_fn):
    import csv, os, datetime, pytz
    import pandas as pd
    from failure_db import ensure_failure_db
    from logger import update_model_success, log_prediction

    ensure_failure_db()

    PREDICTION_LOG = "/persistent/prediction_log.csv"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    date_str = now_kst().strftime("%Y-%m-%d")
    EVAL_RESULT = f"/persistent/logs/evaluation_{date_str}.csv"
    WRONG = f"/persistent/logs/wrong_{date_str}.csv"

    class_ranges = [
        (-0.99, -0.60), (-0.60, -0.30), (-0.30, -0.20), (-0.20, -0.15),
        (-0.15, -0.10), (-0.10, -0.07), (-0.07, -0.05), (-0.05, -0.03),
        (-0.03, -0.01), (-0.01, 0.01),
        (0.01, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 0.10),
        (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 0.60),
        (0.60, 1.00), (1.00, 2.00), (2.00, 5.00)
    ]

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

            try:
                pred_class = int(float(r.get("predicted_class", -1)))
            except:
                pred_class = -1

            # ✅ label 처리: 없는 경우 -1 fallback
            label_raw = r.get("label", -1)
            try:
                label = int(float(label_raw)) if pd.notnull(label_raw) else -1
            except:
                label = -1
            r["label"] = label

            # ✅ pred_class == -1 은 실패 기록 후 평가 제외
            if pred_class == -1:
                log_prediction(
                    symbol=symbol,
                    strategy=strategy,
                    direction="예측실패",
                    entry_price=0,
                    target_price=0,
                    timestamp=now_kst().isoformat(),
                    model=model,
                    success=False,
                    reason="pred_class=-1",
                    rate=0.0,
                    return_value=0.0,
                    volatility=False,
                    source="평가",
                    predicted_class=-1,
                    label=label
                )
                updated.append(r)
                continue

            try:
                entry_price = float(r.get("entry_price", 0))
            except:
                r.update({"status": "fail", "reason": "entry_price 오류", "return": 0.0})
                updated.append(r)
                continue

            if entry_price <= 0:
                r.update({"status": "fail", "reason": "entry_price 0이하", "return": 0.0})
                updated.append(r)
                continue

            timestamp = pd.to_datetime(r.get("timestamp"), utc=True).tz_convert("Asia/Seoul")
            deadline = timestamp + pd.Timedelta(hours=eval_horizon_map.get(strategy, 6))
            now = now_kst()
            if now < deadline:
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

            if 0 <= pred_class < len(class_ranges):
                cls_min, cls_max = class_ranges[pred_class]
                success = gain >= cls_min
            else:
                cls_min, cls_max = -999, 999
                success = False

            vol = str(r.get("volatility", "")).lower() in ["1", "true"]
            status = "v_success" if vol and success else \
                     "v_fail" if not success and vol else \
                     "success" if success else "fail"

            confidence = float(r.get("confidence", 0.0)) if "confidence" in r else 0.0

            r.update({
                "status": status,
                "reason": f"[cls={pred_class}] class_range=({cls_min:.3f}~{cls_max:.3f}), gain={gain:.3f}",
                "return": round(gain, 5),
                "confidence": confidence,
                "label": label  # ✅ 평가 결과에도 label 유지
            })

            r_clean = {str(k): v if v is not None else "" for k, v in r.items() if k is not None}

            update_model_success(symbol, strategy, model, success)
            evaluated.append(r_clean)

        except Exception as e:
            r.update({"status": "fail", "reason": f"예외: {e}", "return": 0.0})
            updated.append(r)

    updated += evaluated

    if updated:
        fieldnames = sorted({k for row in updated for k in row.keys() if k is not None})
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated)

    if evaluated:
        fieldnames = sorted({k for row in evaluated for k in row.keys() if k is not None})
        with open(EVAL_RESULT, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluated)

        failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
        if failed:
            fieldnames = sorted({k for row in failed for k in row.keys() if k is not None})
            with open(WRONG, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failed)

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

