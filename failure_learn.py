# === failure_learn.py (MEM-SAFE REVISED: chunk merge of wrong logs + runtime lock + cooldown) ===
import os, json, time, glob
from datetime import datetime, timedelta
import pandas as pd
import pytz

from config import get_class_groups, get_class_ranges
from train import train_one_model
import logger

PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
LOCK_DIR = os.path.join(PERSIST_DIR, "locks")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)

# 루트 wrong (호환)
WRONG_CSV_ROOT = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
# 상태/요약
STATE_JSON = os.path.join(LOG_DIR, "failure_learn_state.json")
SUMMARY_CSV = os.path.join(LOG_DIR, "failure_retrain_summary.csv")

# 클린업 충돌 방지용 런타임 락
LOCK_PATH = os.getenv("SAFE_LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))

# ============ 설정 ============ #
KST = pytz.timezone("Asia/Seoul")
CSV_CHUNKSIZE = int(os.getenv("FAIL_LEARN_CHUNKSIZE", "50000"))  # 청크 크기
LOOKBACK_DAYS_DEFAULT = int(os.getenv("FAIL_LEARN_LOOKBACK_DAYS", "7"))
COOLDOWN_MINUTES = int(os.getenv("FAIL_LEARN_COOLDOWN_MIN", "20"))
# ============================= #

_now_kst = lambda: datetime.now(KST)
_now_str = lambda: _now_kst().strftime("%Y-%m-%d %H:%M:%S")

def _safe_read_csv(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
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

def _append_summary_row(row: dict):
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    try:
        import csv
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 메모리 안전: wrong_*.csv / root wrong를 "청크 + 기간필터"로 합치기
# ──────────────────────────────────────────────────────────────
def _iter_recent_rows_from_csv(path, cutoff_kst, chunksize=CSV_CHUNKSIZE):
    """
    지정 CSV에서 필요한 컬럼만 청크로 읽어, cutoff_kst 이후 데이터만 yield.
    """
    cols = ["timestamp", "symbol", "strategy", "status", "success"]
    try:
        for chunk in pd.read_csv(
            path,
            usecols=lambda c: c in cols,
            encoding="utf-8-sig",
            chunksize=int(chunksize),
            on_bad_lines="skip",
        ):
            if chunk.empty:
                continue
            # timestamp 파싱(UTC 가정→KST 변환; naive면 KST 부여)
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            if ts.notna().any():
                ts = ts.dt.tz_convert("Asia/Seoul")
            else:
                # 전부 NaT면 naive로 다시 시도
                ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
                try:
                    ts = ts.dt.tz_localize("Asia/Seoul")
                except Exception:
                    pass
            mask = ts >= cutoff_kst
            if not mask.any():
                continue
            sub = chunk.loc[mask].copy()
            sub["__ts"] = ts[mask]
            yield sub
    except Exception:
        # 파일 포맷 문제가 있어도 전체 실패로 만들지 않음
        df = _safe_read_csv(path)
        if df.empty:
            return
        # 가능한 한 최소 처리 후 반환
        yield df

def _load_recent_wrong(days=LOOKBACK_DAYS_DEFAULT) -> pd.DataFrame:
    """
    최근 N일치 logs/wrong_YYYY-MM-DD.csv + 루트 WRONG_CSV_ROOT를 메모리-세이프하게 합쳐서 반환.
    """
    cutoff = _now_kst() - timedelta(days=int(days))
    parts = []

    # 루트 호환 파일(있으면)
    if os.path.exists(WRONG_CSV_ROOT):
        for sub in _iter_recent_rows_from_csv(WRONG_CSV_ROOT, cutoff):
            parts.append(sub)

    # 최근 N일 wrong_*.csv들
    for path in glob.glob(os.path.join(LOG_DIR, "wrong_*.csv")):
        # 파일명이 오래돼도 내부에서 cutoff로 걸러짐
        for sub in _iter_recent_rows_from_csv(path, cutoff):
            parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=["timestamp","symbol","strategy","status","success","__ts"])

    # concat 시에도 메모리 폭증 방지: 열만 선택
    cols_union = ["timestamp","symbol","strategy","status","success","__ts"]
    df = pd.concat([p[cols_union] if all(c in p.columns for c in cols_union) else p for p in parts],
                   ignore_index=True)
    # 중복 제거(동일 ts/symbol/strategy/상태 조합)
    keep_cols = [c for c in ["__ts","symbol","strategy","status","success"] if c in df.columns]
    try:
        df = df.drop_duplicates(subset=keep_cols)
    except Exception:
        df = df.drop_duplicates()
    return df

# ──────────────────────────────────────────────────────────────
# 타깃 선정: 최근 실패 많은 (symbol,strategy)
# ──────────────────────────────────────────────────────────────
def _pick_targets(df: pd.DataFrame, lookback_days=LOOKBACK_DAYS_DEFAULT, max_targets=8):
    """
    실패 많은 (symbol,strategy)를 점수화해서 상위 N개 선택.
      score = 최근 7일 실패수(가중) 합
    """
    if df.empty:
        return []

    # 타임스탬프 컬럼 정규화
    if "__ts" in df.columns:
        ts_col = "__ts"
    else:
        ts_col = "timestamp"
        try:
            df[ts_col] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        except Exception:
            df[ts_col] = pd.NaT

    now = _now_kst()
    since = now - timedelta(days=int(lookback_days))

    # 실패 행만 사용 (status 또는 success 호환)
    dff = df.copy()
    if "status" in dff.columns:
        dff = dff[dff["status"].astype(str).str.lower().isin(["fail", "v_fail"])]
    elif "success" in dff.columns:
        dff = dff[dff["success"].astype(str).str.lower().isin(["false", "0", "no", "n"])]

    # 최근 가중치
    dff["recency_w"] = 1.0
    try:
        recent_mask = dff[ts_col].notna() & (dff[ts_col] >= since)
        dff.loc[recent_mask, "recency_w"] = 1.5
        very_recent = dff[ts_col].notna() & (dff[ts_col] >= now - timedelta(days=1))
        dff.loc[very_recent, "recency_w"] = 2.0
    except Exception:
        pass

    # (심볼,전략) 그룹 점수
    cols = [c for c in ["symbol","strategy"] if c in dff.columns]
    if len(cols) < 2:
        return []
    g = dff.groupby(cols, dropna=False)["recency_w"].sum().reset_index(name="score")
    g = g.sort_values("score", ascending=False)

    targets = []
    for _, r in g.head(int(max_targets)).iterrows():
        s = str(r["symbol"]); t = str(r["strategy"])
        if s and t:
            targets.append((s, t, float(r["score"])))
    return targets

# ──────────────────────────────────────────────────────────────
# 런타임 락
# ──────────────────────────────────────────────────────────────
def _touch_lock():
    try:
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(_now_str())
    except Exception:
        pass

def _release_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 메인: 실패 많은 (symbol,strategy) 우선 재학습
# ──────────────────────────────────────────────────────────────
def run_failure_training(max_targets: int = 8, lookback_days: int = LOOKBACK_DAYS_DEFAULT):
    """
    최근 실패(wrong_*.csv + root wrong) 기반으로 실패 많은 (symbol,strategy) 우선 재학습.
    - 최신 클래스 경계 기준으로 그룹별(train_one_model) 재학습
    - 요약을 logs/failure_retrain_summary.csv에 기록
    - 과도 실행 방지: 마지막 실행 후 COOLDOWN_MINUTES 분 쿨다운
    - 클린업 충돌 방지: 실행 동안 런타임 락 생성
    """
    # 쿨다운
    state = _load_state()
    last_ts = state.get("last_run_ts")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if _now_kst() - last_dt < timedelta(minutes=COOLDOWN_MINUTES):
                return {"ok": True, "skipped": True, "reason": "cooldown", "last_run_ts": last_ts}
        except Exception:
            pass

    # 실패 데이터 로드(메모리 세이프)
    df = _load_recent_wrong(days=lookback_days)
    if df.empty:
        return {"ok": True, "skipped": True, "reason": "no_wrong_data"}

    targets = _pick_targets(df, lookback_days=lookback_days, max_targets=max_targets)
    summary = {"ok": True, "skipped": False, "targets": [], "started_at": _now_str()}

    _touch_lock()
    try:
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
            _append_summary_row({
                "timestamp": _now_str(),
                "symbol": symbol,
                "strategy": strategy,
                "score": float(score),
                "groups": ",".join(map(str, retrained))
            })

            time.sleep(0.2)
    finally:
        _release_lock()

    # 쿨다운 저장
    _save_state({"last_run_ts": _now_kst().isoformat()})
    return summary
