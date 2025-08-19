# === failure_learn.py (신규: 실패학습 자동 재학습 루틴) ===
import os, json, time
from datetime import datetime, timedelta
import pandas as pd
import pytz

from config import get_class_groups, get_class_ranges
from train import train_one_model
import logger

PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

WRONG_CSV = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
STATE_JSON = os.path.join(LOG_DIR, "failure_learn_state.json")
SUMMARY_CSV = os.path.join(LOG_DIR, "failure_retrain_summary.csv")

KST = pytz.timezone("Asia/Seoul")

def _now_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

def _safe_read_csv(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

def _load_state():
    try:
        if os.path.exists(STATE_JSON):
            with open(STATE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"last_run_ts": None}

def _save_state(state: dict):
    try:
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _pick_targets(df: pd.DataFrame, lookback_days=7, max_targets=8):
    """
    잘 실패한 (symbol,strategy)를 점수화해서 상위 N개 선택.
      score = 최근 7일 실패수 * (최근일수 가중)
    """
    if df.empty: return []

    # 타임스탬프 정규화
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            df["timestamp"] = pd.NaT
    else:
        df["timestamp"] = pd.NaT

    now = datetime.now(KST)
    since = now - timedelta(days=lookback_days)
    dff = df.copy()

    # 실패 행만 사용 (success 컬럼/ status 컬럼 호환)
    if "success" in dff.columns:
        dff = dff[dff["success"].astype(str).str.lower().isin(["false", "0", "no", "n"])]
    elif "status" in dff.columns:
        dff = dff[dff["status"].astype(str).str.lower().isin(["fail", "v_fail"])]

    # 최근 가중치
    dff["recency_w"] = 1.0
    try:
        recent_mask = dff["timestamp"].notna() & (dff["timestamp"] >= since)
        dff.loc[recent_mask, "recency_w"] = 1.5
        very_recent = dff["timestamp"].notna() & (dff["timestamp"] >= now - timedelta(days=1))
        dff.loc[very_recent, "recency_w"] = 2.0
    except Exception:
        pass

    # (심볼,전략) 그룹 점수
    cols = [c for c in ["symbol","strategy"] if c in dff.columns]
    if len(cols) < 2:
        return []
    g = dff.groupby(cols)["recency_w"].sum().reset_index(name="score")
    g = g.sort_values("score", ascending=False)

    targets = []
    for _, r in g.head(max_targets).iterrows():
        s = str(r["symbol"]); t = str(r["strategy"])
        if s and t:
            targets.append((s, t, float(r["score"])))
    return targets

def _append_summary_row(row: dict):
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    try:
        import csv
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header: w.writeheader()
            w.writerow(row)
    except Exception:
        pass

def run_failure_training(max_targets: int = 8):
    """
    wrong_predictions.csv 기반으로 실패 많은 (symbol,strategy) 우선 재학습.
    - 최신 클래스 경계 기준으로 그룹별(train_one_model) 재학습
    - 요약을 logs/failure_retrain_summary.csv에 기록
    - 과도 실행 방지: 마지막 실행 후 최소 20분 쿨다운
    """
    state = _load_state()
    last_ts = state.get("last_run_ts")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if datetime.now(KST) - last_dt < timedelta(minutes=20):
                return {"ok": True, "skipped": True, "reason": "cooldown", "last_run_ts": last_ts}
        except Exception:
            pass

    df = _safe_read_csv(WRONG_CSV)
    if df.empty:
        return {"ok": True, "skipped": True, "reason": "no_wrong_data"}

    targets = _pick_targets(df, lookback_days=7, max_targets=max_targets)
    summary = {"ok": True, "skipped": False, "targets": [], "started_at": _now_str()}

    for (symbol, strategy, score) in targets:
        # 최신 클래스 경계/그룹 구하기
        try:
            class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
            if not class_ranges or len(class_ranges) < 2:
                logger.log_training_result(symbol, strategy, model="failure_retrain",
                                           note="경계<2 → 스킵", status="skipped")
                continue
            groups = get_class_groups(num_classes=len(class_ranges))
            max_gid = len(groups) - 1
        except Exception as e:
            logger.log_training_result(symbol, strategy, model="failure_retrain",
                                       note=f"경계계산실패:{e}", status="skipped")
            continue

        # 그룹별 소규모 재학습
        retrained = []
        for gid in range(max_gid + 1):
            try:
                train_one_model(symbol, strategy, group_id=gid, max_epochs=6)  # 가벼운 재학습
                retrained.append(gid)
            except Exception as e:
                logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                           note=f"예외:{e}", status="failed")

        summary["targets"].append({"symbol": symbol, "strategy": strategy,
                                   "score": score, "groups": retrained})

        # 요약 로그 한 줄
        _append_summary_row({
            "timestamp": _now_str(),
            "symbol": symbol,
            "strategy": strategy,
            "score": float(score),
            "groups": ",".join(map(str, retrained))
        })

        time.sleep(0.2)

    # 상태 저장(쿨다운 기준)
    _save_state({"last_run_ts": datetime.now(KST).isoformat()})

    return summary
