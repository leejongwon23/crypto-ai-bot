# diag_e2e.py
import os, traceback
import json
from datetime import datetime
import pytz

# ✅ 프로젝트 내부 모듈 import
from train import train_symbol_group_loop
from predict import predict
from evaluation import evaluate_predictions
from failure_db import insert_failure_record
from logger import (
    ensure_prediction_log_exists,
    get_available_models,
)

# --------------------------------
# Helper Functions
# --------------------------------

def _model_inventory():
    """현재 저장된 모델 파일 현황 확인"""
    try:
        models = get_available_models()
        return {"ok": True, "count": len(models), "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _prediction_log_status():
    """예측 로그 파일 생성 여부 확인"""
    try:
        ensure_prediction_log_exists()
        log_path = "/persistent/logs/prediction_log.csv"
        exists = os.path.exists(log_path)
        size = os.path.getsize(log_path) if exists else 0
        return {"ok": True, "exists": exists, "size": size, "path": log_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --------------------------------
# Main Run Function
# --------------------------------

def run(group=0, budget="mini", do_predict=True, do_evaluate=True, do_failure=False):
    """
    End-to-End 점검 실행:
    1) 그룹 학습
    2) 예측
    3) 평가
    4) 실패학습 (선택)
    """

    report = {
        "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "group": group,
        "budget": budget,
        "order_trace": [],
        "model_inventory": None,
        "prediction_log": None,
        "evaluate": None,
        "failure_learning": None,
    }

    try:
        # ✅ 1. 그룹 학습
        report["order_trace"].append("train:start")
        train_symbol_group_loop(group_id=group, budget=budget)
        report["order_trace"].append("train:done")

        # ✅ 2. 예측
        if do_predict:
            report["order_trace"].append("predict:start")
            # 예시: 심볼/전략을 직접 지정 → 그룹 단위로 실행하려면 수정 가능
            # 여기서는 BTC/단기만 샘플로 실행
            predict("BTCUSDT", "단기", source="diag")
            report["order_trace"].append("predict:done")

        # ✅ 3. 평가
        if do_evaluate:
            report["order_trace"].append("evaluate:start")
            eval_result = evaluate_predictions()
            report["evaluate"] = eval_result
            report["order_trace"].append("evaluate:done")

        # ✅ 4. 실패학습
        if do_failure:
            report["order_trace"].append("failure:start")
            try:
                insert_failure_record("BTCUSDT", "단기", {"mock": "test"})
                report["failure_learning"] = {"ok": True}
            except Exception as fe:
                report["failure_learning"] = {"ok": False, "error": str(fe)}
            report["order_trace"].append("failure:done")

        # ✅ 모델 현황 + 예측로그 상태
        report["model_inventory"] = _model_inventory()
        report["prediction_log"] = _prediction_log_status()

        report["ok"] = True
        return report

    except Exception as e:
        tb = traceback.format_exc()
        report["ok"] = False
        report["error"] = str(e)
        report["traceback"] = tb
        return report
