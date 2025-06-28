import os
import json
from model.base_model import get_model  # ✅ 추가

MODEL_DIR = "/persistent/models"

def fix_all_meta_json():
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
        if not meta.get("symbol"):
            meta["symbol"] = symbol
            updated = True
        if not meta.get("strategy"):
            meta["strategy"] = strategy
            updated = True
        if not meta.get("model"):
            meta["model"] = model_type
            updated = True

        # ✅ input_size: 실제 모델 구조에서 불러오기
        try:
            model = get_model(model_type)
            input_size = model.fc_logits.in_features if hasattr(model, "fc_logits") else 11
        except Exception as e:
            print(f"[ERROR] input_size 가져오기 실패: {e}")
            input_size = 11

        if meta.get("input_size") != input_size:
            meta["input_size"] = input_size
            updated = True

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
