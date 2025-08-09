# === logger.py 최종본 ===
import os
import csv
import json
import hashlib
import datetime
import pytz
import sqlite3
from typing import List, Dict, Any

# ----- 공용 경로 -----
PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")   # ✅ 경로 통일
AUDIT_LOG       = os.path.join(LOG_DIR, "prediction_audit.csv")
SUCCESS_DB      = os.path.join(LOG_DIR, "success.db")              # ✅ 성공률 DB 단일화

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()

# =========================
# 파일/DB 유틸
# =========================
def ensure_prediction_log_exists():
    """
    prediction_log.csv를 표준 헤더로 보장
    """
    header = [
        "timestamp","symbol","strategy","direction","entry_price","target_price",
        "model","model_name","group_id","success","reason","rate","return",
        "volatility","source","predicted_class","label","status","note","hash"
    ]
    if not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0:
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(header)

def get_db_connection():
    """
    성공률/집계용 단일 DB 커넥션
    """
    conn = sqlite3.connect(SUCCESS_DB)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS model_success (
        model_name TEXT PRIMARY KEY,
        success_count INTEGER NOT NULL DEFAULT 0,
        total_count   INTEGER NOT NULL DEFAULT 0,
        last_updated  TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS strategy_success (
        key TEXT PRIMARY KEY,        -- symbol|strategy|model
        success_count INTEGER NOT NULL DEFAULT 0,
        total_count   INTEGER NOT NULL DEFAULT 0,
        last_updated  TEXT
    )
    """)
    conn.commit()
    return conn

# =========================
# 해시/모델 검색
# =========================
def get_feature_hash(x_last) -> str:
    """
    마지막 타임스텝(또는 1D 벡터)을 라운딩하여 안정적 해시 생성
    """
    try:
        if hasattr(x_last, "tolist"):
            x_last = x_last.tolist()
        rounded = [round(float(v), 4) for v in x_last]
        return hashlib.sha1(",".join(map(str, rounded)).encode()).hexdigest()
    except Exception:
        return "invalid"

def get_available_models(symbol: str) -> List[Dict[str, Any]]:
    """
    주어진 symbol에 대해 사용 가능한 (pt + meta) 모델 목록을 리턴
    반환 예:
    [{"pt_file":"BTCUSDT_단기_lstm.pt","model_type":"lstm","strategy":"단기","group_id":0}, ...]
    """
    results = []
    metas = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta.json") and f.startswith(symbol + "_")]
    for mf in metas:
        meta_path = os.path.join(MODEL_DIR, mf)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        strategy = meta.get("strategy")
        model_type = meta.get("model")
        group_id = int(meta.get("group_id", 0))
        # 대응되는 pt 파일
        pt_file = mf.replace(".meta.json", ".pt")
        pt_path = os.path.join(MODEL_DIR, pt_file)
        if not (strategy and model_type and os.path.exists(pt_path)):
            continue

        results.append({
            "pt_file": pt_file,
            "model_type": model_type,
            "strategy": strategy,
            "group_id": group_id,
            "num_classes": meta.get("num_classes"),
            "input_size": meta.get("input_size"),
            "symbol": symbol,
        })

    # group_id, 모델타입 기준 정렬(안정적)
    results.sort(key=lambda r: (r.get("strategy",""), r.get("group_id", 0), r.get("model_type","")))
    return results

# =========================
# 로깅
# =========================
def log_prediction(
    *, symbol: str, strategy: str, direction: str,
    entry_price: float, target_price: float,
    timestamp: str = None,
    model: str = "unknown", model_name: str = "", success: bool = None,
    reason: str = "", rate: float = 0.0, return_value: float = 0.0,
    volatility: bool = False, source: str = "일반",
    predicted_class: int = -1, label: int = -1, group_id: int = 0, note: str = ""
):
    """
    예측/평가 공통 로그 함수. status는 규칙에 따라 자동/외부필드로 결정.
    """
    ensure_prediction_log_exists()

    # status 기본값 결정
    status = None
    if success is True:
        status = "success" if not volatility else "v_success"
    elif success is False:
        status = "fail" if not volatility else "v_fail"
    else:
        status = "pending"

    # 수익률 컬럼 통일
    rtn = return_value if return_value not in (None, "") else rate

    # 해시(중복 방지 힌트)
    base_hash_src = f"{symbol}|{strategy}|{direction}|{entry_price}|{target_price}|{model}|{predicted_class}|{label}"
    row_hash = hashlib.sha1(base_hash_src.encode()).hexdigest()

    row = {
        "timestamp": timestamp or now_kst(),
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction,
        "entry_price": entry_price,
        "target_price": target_price,
        "model": model,
        "model_name": model_name,
        "group_id": group_id,
        "success": success,
        "reason": reason,
        "rate": rate,
        "return": rtn,
        "volatility": bool(volatility),
        "source": source,
        "predicted_class": predicted_class,
        "label": label,
        "status": status,
        "note": note,
        "hash": row_hash
    }

    # 라인 추가
    with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        # 파일이 비어있으면 헤더부터
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    # 성공률 DB 갱신 (가능한 키만)
    try:
        update_model_success(symbol, strategy, model, status in ["success", "v_success"])
    except Exception:
        pass

def log_audit_prediction(symbol: str, strategy: str, result: str, status: str):
    """
    간단 감사 로그 (predict_trigger 등에서 사용)
    """
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp","symbol","strategy","result","status"])
            if f.tell() == 0:
                w.writeheader()
            w.writerow({
                "timestamp": now_kst(),
                "symbol": symbol,
                "strategy": strategy,
                "result": str(result),
                "status": status
            })
    except Exception as e:
        print(f"[log_audit_prediction 오류] {e}")

# =========================
# 성공률 집계
# =========================
def record_model_success(model_name: str, is_success: bool):
    """
    단일 모델에 대한 학습/검증 성공여부 기록
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT success_count, total_count FROM model_success WHERE model_name=?", (model_name,))
    row = cur.fetchone()
    if row:
        s, t = row
        s += 1 if is_success else 0
        t += 1
        cur.execute("UPDATE model_success SET success_count=?, total_count=?, last_updated=? WHERE model_name=?",
                    (s, t, now_kst(), model_name))
    else:
        cur.execute("INSERT INTO model_success(model_name, success_count, total_count, last_updated) VALUES (?,?,?,?)",
                    (model_name, 1 if is_success else 0, 1, now_kst()))
    conn.commit(); conn.close()

def update_model_success(symbol: str, strategy: str, model: str, is_success: bool):
    """
    전략/심볼/모델 조합에 대한 성공률 집계
    """
    key = f"{symbol}|{strategy}|{model}"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT success_count, total_count FROM strategy_success WHERE key=?", (key,))
    row = cur.fetchone()
    if row:
        s, t = row
        s += 1 if is_success else 0
        t += 1
        cur.execute("UPDATE strategy_success SET success_count=?, total_count=?, last_updated=? WHERE key=?",
                    (s, t, now_kst(), key))
    else:
        cur.execute("INSERT INTO strategy_success(key, success_count, total_count, last_updated) VALUES (?,?,?,?)",
                    (key, 1 if is_success else 0, 1, now_kst()))
    conn.commit(); conn.close()

def get_model_success_stats(model_name: str) -> Dict[str, float]:
    """
    모델명 기준 성공 통계
    """
    result = {"total": 0, "success": 0, "success_rate": 0.0}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT success_count, total_count FROM model_success WHERE model_name=?", (model_name,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return result
        s, t = row
        result["success"] = int(s)
        result["total"] = int(t)
        result["success_rate"] = (s / t) if t else 0.0
        return result
    except Exception:
        return result

# =========================
# 통계/집계 헬퍼
# =========================
def strategy_stats() -> Dict[str, Dict[str, int]]:
    """
    prediction_log 기반 간단 집계
    """
    ensure_prediction_log_exists()
    try:
        import pandas as pd
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty:
            return {}
        if "status" not in df.columns:
            return {}
        stats = {}
        for strat in df.get("strategy", pd.Series([], dtype=str)).dropna().unique():
            sub = df[df["strategy"] == strat]
            stats[strat] = {
                "success": int((sub["status"].isin(["success", "v_success"])).sum()),
                "fail":    int((sub["status"].isin(["fail", "v_fail"])).sum()),
                "pending": int((sub["status"] == "pending").sum())
            }
        return stats
    except Exception as e:
        print(f"[strategy_stats 오류] {e}")
        return {}

def get_strategy_eval_count(strategy: str) -> int:
    """
    특정 전략의 평가 완료(성공/실패) 개수
    """
    ensure_prediction_log_exists()
    try:
        import pandas as pd
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty or "status" not in df.columns:
            return 0
        sub = df[(df["strategy"] == strategy) & (df["status"].isin(["success","v_success","fail","v_fail"]))]
        return int(len(sub))
    except Exception:
        return 0
