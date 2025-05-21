# diagnostics_utils.py
import pandas as pd
from data.utils import SYMBOLS
from model_weight_loader import model_exists
from logger import (
    get_strategy_eval_count, get_actual_success_rate,
    get_strategy_fail_rate, print_prediction_stats
)

def check_model_presence():
    strategies = ["단기", "중기", "장기"]
    missing_models = []

    for symbol in SYMBOLS:
        for strategy in strategies:
            if not model_exists(symbol, strategy):
                missing_models.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    "status": "모델 없음"
                })

    return {
        "missing_count": len(missing_models),
        "missing_models": missing_models
    }

def check_strategy_diagnostics(threshold=0.7):
    strategies = ["단기", "중기", "장기"]
    results = []

    for strategy in strategies:
        eval_count = get_strategy_eval_count(strategy)
        success_rate = get_actual_success_rate(strategy, threshold=threshold)
        for symbol in SYMBOLS:
            fail_rate = get_strategy_fail_rate(symbol, strategy)
            results.append({
                "symbol": symbol,
                "strategy": strategy,
                "eval_count": eval_count,
                "success_rate": round(success_rate, 4),
                "fail_rate": round(fail_rate, 4)
            })

    return {
        "strategy_diagnostics": results,
        "total": len(results)
    }

def get_prediction_summary():
    return print_prediction_stats()
