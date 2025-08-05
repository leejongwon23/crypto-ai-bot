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
# 🔒 (예전 메타러너, 현재는 get_meta_prediction으로 대체됨)
# from meta_learning import train_meta_learner, load_meta_learner
import safe_cleanup  # ✅ 오래된 로그 자동 정리
import json
import torch.nn.functional as F



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
    import numpy as np, pandas as pd, os, torch
    from sklearn.preprocessing import MinMaxScaler
    from window_optimizer import find_best_windows
    from logger import log_prediction, get_available_models  # ✅ success_rate 기반 필터링 제거
    from config import FEATURE_INPUT_SIZE, get_class_return_range, class_to_expected_return
    from predict_trigger import get_recent_class_frequencies
    from meta_learning import get_meta_prediction
    from data.utils import get_kline_by_strategy, compute_features
    from datetime import datetime
    import pytz
    from failure_db import insert_failure_record, ensure_failure_db
    from predict import get_model_predictions
    from evo_meta_learner import get_best_strategy_by_failure_probability, predict_evo_meta

    ensure_failure_db()
    os.makedirs("/persistent/logs", exist_ok=True)
    def now_kst(): return datetime.now(pytz.timezone("Asia/Seoul"))

    # ✅ 1. 필수 입력 검증
    if not symbol or not strategy:
        insert_failure_record({"symbol": symbol or "None", "strategy": strategy or "None"},
                              "invalid_symbol_strategy", label=-1)
        return None

    log_strategy = strategy

    try:
        # ✅ 2. 최적 윈도우 탐색
        window_list = find_best_windows(symbol, strategy)
        if not window_list:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "window_list_none", label=-1)
            return None

        # ✅ 3. 데이터 로드
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 1:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "df_short", label=-1)
            return None

        # ✅ 4. 피처 생성
        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < max(window_list) + 1:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "feature_short", label=-1)
            return None

        # ✅ 5. 스케일링
        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)
        if feat_scaled.shape[1] < FEATURE_INPUT_SIZE:
            feat_scaled = np.pad(feat_scaled, ((0, 0), (0, FEATURE_INPUT_SIZE - feat_scaled.shape[1])), mode="constant")
        else:
            feat_scaled = feat_scaled[:, :FEATURE_INPUT_SIZE]

        # ✅ 6. 모델 로드 (필터 없이 전부 사용)
        models = get_available_models(symbol)
        if not models:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "no_models", label=-1)
            return None

        # ✅ 7. 최근 클래스 분포
        recent_freq = get_recent_class_frequencies(strategy)
        feature_tensor = torch.tensor(feat_scaled[-1], dtype=torch.float32)

        # ✅ 8. 개별 모델 예측 (전 모델 참여)
        model_outputs_list, all_model_predictions = get_model_predictions(
            symbol, strategy, models, df, feat_scaled, window_list, recent_freq
        )
        if not model_outputs_list:
            insert_failure_record({"symbol": symbol, "strategy": log_strategy}, "no_valid_model", label=-1)
            return None

        # ✅ 9. 진화형 실패확률 기반 전략 교체
        recommended_strategy = get_best_strategy_by_failure_probability(
            symbol=symbol, current_strategy=strategy,
            feature_tensor=feature_tensor, model_outputs=model_outputs_list
        )
        if recommended_strategy and recommended_strategy != strategy:
            print(f"[🔁 전략 교체됨] {strategy} → {recommended_strategy}")
            strategy = recommended_strategy

        # ✅ 10. 메타러너 최종 선택 (전 모델 기반)
        # success_stats는 참고용으로만 유지 (필터링 사용 안 함)
        success_stats = {}  # 품질 필터 제거
        meta_success_rate = {c: 0.5 for c in range(len(model_outputs_list[0]["probs"]))}

        final_pred_class = get_meta_prediction(
            [m["probs"] for m in model_outputs_list],
            feature_tensor,
            meta_info={"success_rate": meta_success_rate}
        )

        # ✅ 11. 진화형 메타러너 조건부 적용
        evo_model_path = "/persistent/models/evo_meta_learner.pt"
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

        # ✅ 12. 메타 결과 로깅
        cls_min, _ = get_class_return_range(final_pred_class)
        current_price = df.iloc[-1]["close"]
        evo_expected_return = class_to_expected_return(final_pred_class, len(model_outputs_list[0]["probs"]))
        actual_return_meta = (current_price / all_model_predictions[0]["entry_price"]) - 1
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
            entry_price=all_model_predictions[0]["entry_price"],
            target_price=all_model_predictions[0]["entry_price"] * (1 + evo_expected_return),
            model="meta",
            model_name="evo_meta_learner",
            predicted_class=final_pred_class,
            label=final_pred_class,
            note="진화형 메타 선택" if use_evo else "기본 메타 선택",
            success=meta_success_flag,
            reason=f"수익률도달:{meta_success_flag}",
            rate=evo_expected_return,
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
            "expected_return": evo_expected_return,
            "timestamp": now_kst().isoformat(),
            "reason": "진화형 메타 최종 선택" if use_evo else "기본 메타 최종 선택",
            "source": source
        }

    except Exception as e:
        insert_failure_record({"symbol": symbol or "None", "strategy": strategy or "None"}, "exception", label=-1)
        return None


# 📄 predict.py 내부에 추가
import csv, datetime, pytz, os
import pandas as pd
from failure_db import ensure_failure_db, insert_failure_record
from logger import update_model_success

def evaluate_predictions(get_price_fn):
    import csv, os, datetime, pytz
    import pandas as pd
    from collections import defaultdict
    from failure_db import ensure_failure_db, insert_failure_record, check_failure_exists
    from logger import update_model_success, log_prediction
    from config import get_class_return_range

    ensure_failure_db()

    PREDICTION_LOG = "/persistent/prediction_log.csv"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    date_str = now_kst().strftime("%Y-%m-%d")
    EVAL_RESULT = f"/persistent/logs/evaluation_{date_str}.csv"
    WRONG = f"/persistent/logs/wrong_{date_str}.csv"

    # 전략별 평가 마감시간
    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}

    updated, evaluated = [], []

    # CSV 읽기
    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            return
    except Exception as e:
        print(f"[오류] prediction_log.csv 읽기 실패 → {e}")
        return

    # 심볼+전략+타임스탬프별 그룹화
    grouped_preds = defaultdict(list)
    for r in rows:
        key = (r.get("symbol"), r.get("strategy"), r.get("timestamp"))
        grouped_preds[key].append(r)

    for key, preds in grouped_preds.items():
        for r in preds:
            try:
                # 이미 평가된 건 스킵
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
                    log_prediction(symbol, strategy, "예측실패", entry_price, entry_price, now_kst().isoformat(),
                                   model, False, reason, 0.0, 0.0, False, "평가",
                                   predicted_class=pred_class, label=label, group_id=group_id)
                    if not check_failure_exists(r):
                        insert_failure_record(r, f"{symbol}-{strategy}-{now_kst().isoformat()}",
                                              feature_vector=None, label=label)
                    updated.append(r)
                    continue

                # 평가 마감 시간
                timestamp = pd.to_datetime(r.get("timestamp"), utc=True).tz_convert("Asia/Seoul")
                eval_hours = eval_horizon_map.get(strategy, 6)
                deadline = timestamp + pd.Timedelta(hours=eval_hours)

                # 가격 데이터 로드
                df = get_price_fn(symbol, strategy)
                if df is None or "timestamp" not in df.columns:
                    r.update({"status": "fail", "reason": "가격 데이터 없음", "return": 0.0})
                    updated.append(r)
                    continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
                future_df = df[df["timestamp"] >= timestamp]
                if future_df.empty:
                    r.update({"status": "fail", "reason": "미래 데이터 없음", "return": 0.0})
                    updated.append(r)
                    continue

                # 실제 최대 상승률
                actual_max = future_df["high"].max()
                gain = (actual_max - entry_price) / (entry_price + 1e-6)

                # 클래스 수익률 범위
                cls_min, cls_max = get_class_return_range(label)
                reached_target = gain >= cls_min

                # 평가 시점 전 → 조기 성공 가능
                if now_kst() < deadline:
                    if reached_target:
                        status = "success"
                    else:
                        r.update({"status": "pending", "reason": "⏳ 평가 대기 중", "return": round(gain, 5)})
                        updated.append(r)
                        continue
                else:
                    # 평가 시점 도달 후 최종 판정
                    status = "success" if reached_target else "fail"

                # 변동성 전략 반영
                vol = str(r.get("volatility", "")).lower() in ["1", "true"]
                if vol:
                    status = "v_success" if status == "success" else "v_fail"

                r.update({
                    "status": status,
                    "reason": f"[label={label}] gain={gain:.3f} (cls_min={cls_min}, cls_max={cls_max})",
                    "return": round(gain, 5),
                    "group_id": group_id
                })

                # 로그 기록
                log_prediction(symbol, strategy, f"평가:{status}", entry_price,
                               entry_price * (1 + gain), now_kst().isoformat(), model,
                               status in ["success", "v_success"], r["reason"], gain, gain, vol, "평가",
                               predicted_class=pred_class, label=label, group_id=group_id)

                # 실패 시 모든 모델 실패 DB 기록
                if status in ["fail", "v_fail"] and not check_failure_exists(r):
                    insert_failure_record(r, f"{symbol}-{strategy}-{now_kst().isoformat()}",
                                          feature_vector=None, label=label)

                # 성공률 업데이트 (메타 모델은 성공률 기록)
                if model == "meta":
                    update_model_success(symbol, strategy, model, status in ["success", "v_success"])

                evaluated.append({str(k): (v if v is not None else "") for k, v in r.items()})

            except Exception as e:
                r.update({"status": "fail", "reason": f"예외: {e}", "return": 0.0})
                updated.append(r)

    # CSV 저장
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

import torch
import torch.nn.functional as F

import json
import os
from model.base_model import get_model
from model_weight_loader import load_model_cached

def get_model_predictions(symbol, strategy, models, df, feat_scaled, window_list, recent_freq):
    """
    ✅ [YOPO 전용]
    - 주어진 모델 리스트(models)로 예측 수행
    - 각 모델은 meta.json을 통해 정보 추출
    - 결과: model_outputs_list, all_model_predictions 반환
    """
    model_outputs_list = []
    all_model_predictions = []

    for model_info in models:
        model_path = model_info.get("model_path")
        meta_path = model_path.replace(".pt", ".meta.json")
        if not os.path.exists(meta_path):
            print(f"[⚠️ 메타파일 없음] {meta_path}")
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_type = meta.get("model", "lstm")
            group_id = meta.get("group_id", 0)
            input_size = meta.get("input_size", 11)
            num_classes = meta.get("num_classes", 21)
        except Exception as e:
            print(f"[⚠️ 메타파일 로딩 실패] {meta_path} → {e}")
            continue

        try:
            window = window_list[group_id]
            input_seq = feat_scaled[-window:]
            if input_seq.shape[0] < window:
                print(f"[⚠️ 데이터 부족] {symbol}-{strategy}-group{group_id}")
                continue

            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # (1, window, input_size)

            model = get_model(model_type, input_size=input_size, output_size=num_classes)
            model = load_model_cached(model_path, model)
            model.eval()

            with torch.no_grad():
                out = model(input_tensor)
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
            print(f"[❌ 모델 예측 실패] {model_path} → {e}")
            continue

    return model_outputs_list, all_model_predictions

