import os
import json
import torch
from model.base_model import get_model  # ✅ 현재 모델 구조 반영 위해 import
from config import NUM_CLASSES

MODEL_DIR = "/persistent/models"


def fix_all_meta_json():
    from config import FEATURE_INPUT_SIZE
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta.json")]

    for file in files:
        path = os.path.join(MODEL_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[ERROR] {file} 읽기 실패: {e}")
            continue

        base = file.replace(".meta.json", "")
        parts = base.split("_")
        if len(parts) < 3:
            print(f"[SKIP] 잘못된 파일명 형식: {file}")
            continue

        symbol, strategy, model_type = parts[0], parts[1], parts[2]

        updated = False

        # ✅ 공백("")일 경우에도 파일명 기반으로 보정
        if not meta.get("symbol"):
            meta["symbol"] = symbol
            updated = True
        if not meta.get("strategy"):
            meta["strategy"] = strategy
            updated = True
        if not meta.get("model"):
            meta["model"] = model_type
            updated = True

        # ✅ input_size 확인 및 보정
        current_input_size = meta.get("input_size")
        try:
            model = get_model(model_type, input_size=FEATURE_INPUT_SIZE, output_size=NUM_CLASSES)
            sample_input = torch.randn(1, 20, FEATURE_INPUT_SIZE)
            output = model(sample_input) if not hasattr(model, "predict") else None
            expected_input_size = FEATURE_INPUT_SIZE

            # ✅ input_size 없거나 불일치 시 FEATURE_INPUT_SIZE 로 보정
            if current_input_size is None or current_input_size != expected_input_size:
                meta["input_size"] = expected_input_size
                updated = True
        except Exception as e:
            print(f"[⚠️ 모델 로드 실패] {file} → {e}")

        if updated:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[FIXED] {file} → 필드 보정 완료")
        else:
            print(f"[OK] {file} → 수정 불필요")

def check_meta_input_size():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta.json")]
    for file in files:
        path = os.path.join(MODEL_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "input_size" not in meta:
                print(f"[❌ 누락] {file} → input_size 없음")
            else:
                print(f"[✅ 확인됨] {file} → input_size = {meta['input_size']}")
        except Exception as e:
            print(f"[ERROR] {file} 읽기 실패: {e}")

if __name__ == "__main__":
    fix_all_meta_json()
