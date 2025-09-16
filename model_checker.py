# model_checker.py (PATCHED)
import os
import time
from glob import glob

# 유틸 경로 불확실성 보호: data.utils 또는 루트 utils
try:
    from data.utils import SYMBOLS
except Exception:
    try:
        from utils import SYMBOLS
    except Exception:
        SYMBOLS = []  # 호출 측에서 비어있음을 확인하도록 함

# train 모듈의 공개 API 불확실성 보호:
# 우선 train_models을 사용하려 시도하고, 없으면 main_train_all(groups=...) 또는 train_symbol_group_loop로 폴백
try:
    from train import train_models  # preferred API (if exists)
except Exception:
    train_models = None

try:
    from train import main_train_all
except Exception:
    main_train_all = None

try:
    from train import train_symbol_group_loop
except Exception:
    train_symbol_group_loop = None

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
REQUIRED_MODELS = ["lstm", "cnn_lstm", "transformer"]
# 허용되는 weight 확장자 목록
_WEIGHT_EXTS = [".pt", ".ptz", ".safetensors"]


def _match_weight_files(symbol: str, strategy: str, model_type: str):
    """
    모델 파일 패턴 매칭. 여러 확장자(.pt, .ptz, .safetensors)를 확인.
    """
    out = []
    for ext in _WEIGHT_EXTS:
        pattern = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group*_cls*{ext}")
        out.extend(glob(pattern))
    return out


def model_exists(symbol: str, strategy: str, model_type: str) -> bool:
    """
    학습 산출물 및 메타(json) 존재 여부 확인.
    규칙: {symbol}_{strategy}_{model_type}_group{gid}_cls{n}.{ext}
    메타 파일은 동일 stem + ".meta.json"
    """
    weight_paths = _match_weight_files(symbol, strategy, model_type)
    for w in weight_paths:
        meta_path = None
        # 기본 메타 확장자 예측
        for meta_ext in (".meta.json", ".meta", ".json"):
            cand = w + meta_ext if not w.endswith(meta_ext) else w
            if os.path.exists(cand):
                meta_path = cand
                break
        # 일부 저장 방식에서는 stem을 바꿀 수 있으므로 대체 검사
        if meta_path is None:
            stem = os.path.splitext(w)[0]
            cand = stem + ".meta.json"
            if os.path.exists(cand):
                meta_path = cand
        if meta_path:
            return True
    return False


def _invoke_train_for_symbol(symbol: str):
    """
    안전한 학습 호출 래퍼:
    - train_models 존재 시 사용 (심볼 리스트 인수)
    - 없으면 main_train_all(groups=[[symbol]]) 시도
    - 그마저 없으면 train_symbol_group_loop([[symbol]]) 시도
    """
    if train_models is not None:
        try:
            print(f"⏳ 호출: train_models([{symbol!r}])")
            train_models([symbol])
            return True
        except Exception as e:
            print(f"[ERROR] train_models 호출 실패: {e}")

    if main_train_all is not None:
        try:
            print(f"⏳ 호출 폴백: main_train_all(groups=[[{symbol!r}]])")
            main_train_all(groups=[[symbol]])
            return True
        except Exception as e:
            print(f"[ERROR] main_train_all 호출 실패: {e}")

    if train_symbol_group_loop is not None:
        try:
            print(f"⏳ 호출 폴백2: train_symbol_group_loop([[{symbol!r}]])")
            train_symbol_group_loop([[symbol]])
            return True
        except Exception as e:
            print(f"[ERROR] train_symbol_group_loop 호출 실패: {e}")

    print(f"[FATAL] 학습을 호출할 수 있는 공개 API가 없습니다. train 모듈을 확인하세요.")
    return False


def check_and_train_models():
    """
    - 모든 심볼 × 전략 조합 점검
    - 누락된 (심볼, 전략)이 있으면 해당 심볼 전체를 학습하도록 시도
    """
    print("🔍 모델 존재 여부 점검 시작...")
    if not SYMBOLS:
        print("[WARN] SYMBOLS 목록이 비어있습니다. data/utils 또는 utils에서 SYMBOLS 정의를 확인하세요.")
        return

    missing_pairs = set()

    for symbol in SYMBOLS:
        for strategy in ["단기", "중기", "장기"]:
            # 세 모델 중 하나라도 없으면 학습 큐에 추가
            for model_type in REQUIRED_MODELS:
                try:
                    if not model_exists(symbol, strategy, model_type):
                        missing_pairs.add((symbol, strategy))
                        break  # 이 전략은 학습 대상 확정
                except Exception as e:
                    print(f"[WARN] model_exists 검사 중 예외: {symbol}-{strategy}-{model_type} → {e}")
                    missing_pairs.add((symbol, strategy))
                    break

    if not missing_pairs:
        print("✅ 모든 모델이 정상적으로 존재합니다.")
        return

    print(f"⚠️ 누락된 (심볼, 전략) 조합: {len(missing_pairs)}개 → 자동 학습 시도")

    # 심볼 단위로 묶어서 학습(같은 심볼 여러 전략이 빠졌을 수 있음)
    symbols_to_train = sorted({s for (s, _) in missing_pairs})
    for symbol in symbols_to_train:
        try:
            print(f"⏳ {symbol} 전체 전략 학습 시작")
            ok = _invoke_train_for_symbol(symbol)
            if not ok:
                print(f"[ERROR] {symbol} 학습 호출 실패. 수동 점검 필요.")
            time.sleep(1)  # 안전 간격
        except Exception as e:
            print(f"[오류] {symbol} 학습 중 예외: {e}")

    print("✅ 누락 모델 자동 학습 완료")


if __name__ == "__main__":
    check_and_train_models()
