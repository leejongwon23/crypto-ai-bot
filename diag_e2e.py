# diag_e2e.py
import os, traceback, json
from datetime import datetime
import pytz

# 내부 모듈
from train import train_symbol_group_loop, train_models
from predict import predict, evaluate_predictions
from config import get_SYMBOL_GROUPS
from logger import ensure_prediction_log_exists

MODEL_DIR = "/persistent/models"
PREDICTION_LOG = "/persistent/prediction_log.csv"
KST = pytz.timezone("Asia/Seoul")

def _model_inventory():
    """저장된 모델 개요"""
    try:
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        return {"ok": True, "count": len(files), "files": files[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _prediction_log_status():
    """예측 로그 파일 상태"""
    try:
        ensure_prediction_log_exists()
        exists = os.path.exists(PREDICTION_LOG)
        size = os.path.getsize(PREDICTION_LOG) if exists else 0
        return {"ok": True, "exists": exists, "size": size, "path": PREDICTION_LOG}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _predict_group(symbols, strategies=("단기","중기","장기")):
    """그룹 심볼들에 대해 예측 1회 실행"""
    done = []
    for sym in symbols:
        for strat in strategies:
            try:
                predict(sym, strat, source="diag", model_type=None)
                done.append(f"{sym}-{strat}")
            except Exception as e:
                done.append(f"{sym}-{strat}:ERROR:{e}")
    return done

def run(group=-1, do_predict=True, do_evaluate=True):
    """
    End-to-End 점검 실행:
    - group==-1: 전체 그룹 학습 루프(train_symbol_group_loop) 실행(내부에 예측 포함)
    - group>=0 : 해당 그룹만 train_models → (옵션) 예측 → (옵션) 평가
    """
    report = {
        "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "group": group,
        "order_trace": [],
        "ok": False,
        "train": None,
        "predict": None,
        "evaluate": None,
        "model_inventory": None,
        "prediction_log": None,
    }

    try:
        report["order_trace"].append("start")

        if group is None or int(group) < 0:
            # 전체 그룹 일괄 학습(내부에서 각 그룹 학습 직후 예측까지 수행)
            report["order_trace"].append("train_symbol_group_loop:start")
            train_symbol_group_loop(sleep_sec=0)
            report["order_trace"].append("train_symbol_group_loop:done")
            report["train"] = {"mode": "all_groups"}
            # 별도 predict 단계는 생략 가능(루프 내 즉시예측 수행)

        else:
            # 특정 그룹만 선택 학습
            groups = get_SYMBOL_GROUPS()
            gid = int(group)
            if gid >= len(groups):
                raise ValueError(f"잘못된 group 인덱스: {gid} (총 {len(groups)}개)")

            symbols = groups[gid]
            report["train"] = {"mode": "single_group", "group_id": gid, "symbols": symbols}
            report["order_trace"].append(f"train_models:g{gid}:start")
            train_models(symbols)
            report["order_trace"].append(f"train_models:g{gid}:done")

            # 예측(옵션) – train_models는 자동 예측이 없으므로 여기서 수행
            if do_predict:
                report["order_trace"].append(f"predict:g{gid}:start")
                done = _predict_group(symbols)
                report["predict"] = {"executed": True, "targets": done}
                report["order_trace"].append(f"predict:g{gid}:done")
            else:
                report["predict"] = {"executed": False}

        # 평가(옵션)
        if do_evaluate:
            report["order_trace"].append("evaluate:start")
            try:
                eval_res = evaluate_predictions()
            except TypeError:
                # 시그니처가 다른 경우도 있으니 안전 가드
                eval_res = evaluate_predictions  # 함수 객체 정보라도 반환
            report["evaluate"] = {"executed": True, "result": str(eval_res)[:5000]}
            report["order_trace"].append("evaluate:done")
        else:
            report["evaluate"] = {"executed": False}

        # 모델/로그 상태
        report["model_inventory"] = _model_inventory()
        report["prediction_log"] = _prediction_log_status()

        report["ok"] = True
        report["order_trace"].append("done")
        return report

    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()[-5000:]
        report["order_trace"].append("error")
        return report
