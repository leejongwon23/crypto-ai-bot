from model_weight_loader import model_exists
from data.utils import SYMBOLS

def check_model_presence():
    """
    모든 심볼-전략 조합에 대해 모델 파일 존재 여부 점검
    """
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
