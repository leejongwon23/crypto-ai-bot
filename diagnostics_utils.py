import pandas as pd
from logger import get_strategy_eval_count, get_actual_success_rate, get_strategy_fail_rate
from data.utils import SYMBOLS

def check_strategy_diagnostics(threshold=0.7):
    """
    전략별 평가 진행 상태, 성공률, 실패율 진단
    """
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
                "evaluated": eval_count,
                "success_rate": round(success_rate, 4),
                "fail_rate": round(fail_rate, 4)
            })

    return {
        "strategy_diagnostics": results,
        "total": len(results)
    }
