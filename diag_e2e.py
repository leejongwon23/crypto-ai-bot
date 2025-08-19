import os, traceback, json
from datetime import datetime
import pytz

# 내부 모듈
from train import train_symbol_group_loop, train_models
from predict import predict, evaluate_predictions
from config import get_SYMBOL_GROUPS
from logger import (
    ensure_prediction_log_exists,
    get_model_success_rate,
    analyze_class_success,
)
from failure_db import load_existing_failure_hashes

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

def _failure_stats():
    """실패학습 DB 요약"""
    try:
        hashes = load_existing_failure_hashes()
        total = len(hashes)
        sample = hashes[-1] if hashes else None
        return {
            "ok": True,
            "total_failures": total,
            "last_failure_hash": sample,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _success_rate():
    """심볼/전략/모델별 성공률"""
    try:
        rates = {}
        for (symbol, strategy, model_type), val in get_model_success_rate().items():
            if symbol not in rates:
                rates[symbol] = {}
            if strategy not in rates[symbol]:
                rates[symbol][strategy] = {}
            rates[symbol][strategy][model_type] = round(val, 3)
        return {"ok": True, "rates": rates}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _group_status():
    """그룹별 학습 여부"""
    groups = get_SYMBOL_GROUPS()
    status = []
    try:
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        for gid, symbols in enumerate(groups):
            sym_status = {}
            for sym in symbols:
                sym_models = [f for f in files if f.startswith(sym)]
                sym_status[sym] = len(sym_models) > 0
            status.append({"group": gid, "symbols": sym_status})
        return {"ok": True, "groups": status}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run(group=-1, do_predict=True, do_evaluate=True):
    """
    End-to-End 점검 실행:
    - group==-1: 전체 그룹 학습 루프(train_symbol_group_loop) 실행
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
        "failure_stats": None,
        "success_rate": None,
        "group_status": None,
        "summary": None,  # ✅ 한글 요약 추가
    }

    try:
        report["order_trace"].append("start")

        if group is None or int(group) < 0:
            report["order_trace"].append("train_symbol_group_loop:start")
            train_symbol_group_loop(sleep_sec=0)
            report["order_trace"].append("train_symbol_group_loop:done")
            report["train"] = {"mode": "all_groups"}
        else:
            groups = get_SYMBOL_GROUPS()
            gid = int(group)
            if gid >= len(groups):
                raise ValueError(f"잘못된 group 인덱스: {gid} (총 {len(groups)}개)")
            symbols = groups[gid]
            report["train"] = {"mode": "single_group", "group_id": gid, "symbols": symbols}
            report["order_trace"].append(f"train_models:g{gid}:start")
            train_models(symbols)
            report["order_trace"].append(f"train_models:g{gid}:done")
            if do_predict:
                report["order_trace"].append(f"predict:g{gid}:start")
                done = _predict_group(symbols)
                report["predict"] = {"executed": True, "targets": done}
                report["order_trace"].append(f"predict:g{gid}:done")
            else:
                report["predict"] = {"executed": False}

        if do_evaluate:
            report["order_trace"].append("evaluate:start")
            try:
                eval_res = evaluate_predictions()
            except TypeError:
                eval_res = evaluate_predictions
            report["evaluate"] = {"executed": True, "result": str(eval_res)[:2000]}
            report["order_trace"].append("evaluate:done")
        else:
            report["evaluate"] = {"executed": False}

        # ✅ 추가 점검들
        report["model_inventory"] = _model_inventory()
        report["prediction_log"] = _prediction_log_status()
        report["failure_stats"] = _failure_stats()
        report["success_rate"] = _success_rate()
        report["group_status"] = _group_status()

        # ✅ 한글 요약
        report["summary"] = {
            "모델개수": report["model_inventory"].get("count"),
            "실패기록수": report["failure_stats"].get("total_failures"),
            "최근실패": report["failure_stats"].get("last_failure_hash"),
            "예측로그크기(bytes)": report["prediction_log"].get("size"),
        }

        report["ok"] = True
        report["order_trace"].append("done")
        return report

    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()[-5000:]
        report["order_trace"].append("error")
        return report
