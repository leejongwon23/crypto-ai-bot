# === failure_learn.py (v2025-09-17c: MEM-SAFE + mini-epochs + rollback + runtime lock + cooldown) ===
import os, json, time, glob, shutil, csv
from datetime import datetime, timedelta
import pandas as pd
import pytz

from config import get_class_groups, get_class_ranges
from train import train_one_model
import logger

# ── 디렉토리/파일 경로 ─────────────────────────────────────────
PERSIST_DIR = "/persistent"
LOG_DIR  = os.path.join(PERSIST_DIR, "logs")
LOCK_DIR = os.path.join(PERSIST_DIR, "locks")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)

WRONG_CSV_ROOT = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
STATE_JSON     = os.path.join(LOG_DIR, "failure_learn_state.json")
SUMMARY_CSV    = os.path.join(LOG_DIR, "failure_retrain_summary.csv")
LOCK_PATH      = os.getenv("SAFE_LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))

# ── 설정 (환경변수로 조정) ─────────────────────────────────────
KST = pytz.timezone("Asia/Seoul")
CSV_CHUNKSIZE        = int(os.getenv("FAIL_LEARN_CHUNKSIZE", "50000"))   # 청크 읽기 크기
LOOKBACK_DAYS_DEFAULT= int(os.getenv("FAIL_LEARN_LOOKBACK_DAYS", "7"))
COOLDOWN_MINUTES     = int(os.getenv("FAIL_LEARN_COOLDOWN_MIN", "20"))
# 이어학습/롤백 파라미터( failure_trainer 와 동일 키 )
MINI_EPOCHS          = max(1, min(3, int(os.getenv("FAIL_MINI_EPOCHS", "2"))))  # 1~3 미니에폭
ROLLBACK_ENABLE      = os.getenv("ROLLBACK_ON_DEGRADE", "1") == "1"
ROLLBACK_TOLERANCE   = float(os.getenv("ROLLBACK_TOLERANCE", "0.01"))           # 성능악화 허용오차
MAX_TARGETS          = int(os.getenv("FAIL_MAX_TARGETS", "8"))

# ── 모델 아티팩트 탐색/백업/복구 유틸 ───────────────────────────
MODEL_DIR   = "/persistent/models"
KNOWN_EXTS  = (".ptz", ".pt", ".safetensors")
MODEL_TYPES = ("lstm", "cnn_lstm", "transformer")

def _stem_wo_ext(p: str) -> str:
    base = p
    for e in KNOWN_EXTS:
        if base.endswith(e):
            return base[:-len(e)]
    return os.path.splitext(base)[0]

def _find_group_artifacts(symbol: str, strategy: str, group_id: int):
    items = []
    patt = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*_group{int(group_id)}_cls*")
    cands = []
    for e in KNOWN_EXTS:
        cands.extend(glob.glob(patt + e))
    # 별칭 디렉토리 경로도 탐색
    for mtype in MODEL_TYPES:
        for e in KNOWN_EXTS:
            p = os.path.join(MODEL_DIR, symbol, strategy, f"{mtype}{e}")
            if os.path.exists(p): cands.append(p)

    seen = set()
    for wpath in cands:
        stem = _stem_wo_ext(os.path.basename(wpath))
        if stem in seen: continue
        seen.add(stem)
        meta1 = os.path.join(MODEL_DIR, f"{stem}.meta.json")
        meta2 = None
        try:
            parts = stem.split("_")
            if len(parts) >= 3:
                mtype = parts[2]
                meta2 = os.path.join(MODEL_DIR, symbol, strategy, f"{mtype}.meta.json")
        except Exception:
            pass
        meta_path = meta1 if os.path.exists(meta1) else (meta2 if (meta2 and os.path.exists(meta2)) else None)
        items.append({"weight": wpath, "meta": meta_path})
    return items

def _read_meta_f1(meta_path: str):
    try:
        if not meta_path or not os.path.exists(meta_path): return None
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return float(m.get("metrics", {}).get("val_f1", None))
    except Exception:
        return None

BACKUP_DIR = os.path.join(PERSIST_DIR, "tmp", "failure_retrain_backups")
def _backup_group(symbol: str, strategy: str, group_id: int):
    try:
        ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"{symbol}_{strategy}_g{group_id}_{ts}")
        os.makedirs(dst, exist_ok=True)
        for it in _find_group_artifacts(symbol, strategy, group_id):
            for p in [it.get("weight"), it.get("meta")]:
                if p and os.path.exists(p):
                    shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
        return dst
    except Exception as e:
        print(f"[백업 실패] {symbol}-{strategy}-g{group_id} → {e}")
        return None

def _restore_from_backup(backup_dir: str) -> bool:
    try:
        if not backup_dir or not os.path.isdir(backup_dir): return False
        for fn in os.listdir(backup_dir):
            src = os.path.join(backup_dir, fn)
            dst = os.path.join(MODEL_DIR, fn)
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[복구 실패] {backup_dir} → {e}")
        return False

def _best_f1_for_group(symbol: str, strategy: str, group_id: int):
    f1s = []
    for it in _find_group_artifacts(symbol, strategy, group_id):
        f1 = _read_meta_f1(it.get("meta"))
        if f1 is not None: f1s.append(float(f1))
    return max(f1s) if f1s else None

# ── 공통 유틸 ──────────────────────────────────────────────────
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
    # 헤더를 고정 순서로( failure_trainer 와 호환 )
    base_fields = ["timestamp","symbol","strategy","score","group_id",
                   "before_f1","after_f1","delta","result","backup_dir"]
    for k in list(row.keys()):
        if k not in base_fields: base_fields.append(k)
    try:
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=base_fields)
            if write_header: w.writeheader()
            w.writerow({k: row.get(k,"") for k in base_fields})
    except Exception:
        pass

# ── 메모리-세이프 wrong 로그 머지 ───────────────────────────────
def _iter_recent_rows_from_csv(path, cutoff_kst, chunksize=CSV_CHUNKSIZE):
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
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            if ts.notna().any():
                ts = ts.dt.tz_convert("Asia/Seoul")
            else:
                ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
                try: ts = ts.dt.tz_localize("Asia/Seoul")
                except Exception: pass
            mask = ts >= cutoff_kst
            if not mask.any(): continue
            sub = chunk.loc[mask].copy()
            sub["__ts"] = ts[mask]
            yield sub
    except Exception:
        df = _safe_read_csv(path)
        if df.empty: return
        yield df

def _load_recent_wrong(days=LOOKBACK_DAYS_DEFAULT) -> pd.DataFrame:
    cutoff = _now_kst() - timedelta(days=int(days))
    parts = []
    if os.path.exists(WRONG_CSV_ROOT):
        for sub in _iter_recent_rows_from_csv(WRONG_CSV_ROOT, cutoff):
            parts.append(sub)
    for path in glob.glob(os.path.join(LOG_DIR, "wrong_*.csv")):
        for sub in _iter_recent_rows_from_csv(path, cutoff):
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["timestamp","symbol","strategy","status","success","__ts"])
    cols_union = ["timestamp","symbol","strategy","status","success","__ts"]
    df = pd.concat([p[cols_union] if all(c in p.columns for c in cols_union) else p for p in parts],
                   ignore_index=True)
    keep_cols = [c for c in ["__ts","symbol","strategy","status","success"] if c in df.columns]
    try: df = df.drop_duplicates(subset=keep_cols)
    except Exception: df = df.drop_duplicates()
    return df

# ── 타깃 선정(최근 실패 많은 (symbol,strategy)) ─────────────────
def _pick_targets(df: pd.DataFrame, lookback_days=LOOKBACK_DAYS_DEFAULT, max_targets=8):
    if df.empty: return []
    ts_col = "__ts" if "__ts" in df.columns else "timestamp"
    if ts_col == "timestamp":
        try:
            df[ts_col] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        except Exception:
            df[ts_col] = pd.NaT

    now = _now_kst()
    since = now - timedelta(days=int(lookback_days))

    dff = df.copy()
    if "status" in dff.columns:
        dff = dff[dff["status"].astype(str).str.lower().isin(["fail", "v_fail"])]
    elif "success" in dff.columns:
        dff = dff[dff["success"].astype(str).str.lower().isin(["false", "0", "no", "n"])]

    dff["recency_w"] = 1.0
    try:
        recent_mask = dff[ts_col].notna() & (dff[ts_col] >= since)
        dff.loc[recent_mask, "recency_w"] = 1.5
        very_recent = dff[ts_col].notna() & (dff[ts_col] >= now - timedelta(days=1))
        dff.loc[very_recent, "recency_w"] = 2.0
    except Exception:
        pass

    cols = [c for c in ["symbol","strategy"] if c in dff.columns]
    if len(cols) < 2: return []
    g = dff.groupby(cols, dropna=False)["recency_w"].sum().reset_index(name="score")
    g = g.sort_values("score", ascending=False)

    out = []
    for _, r in g.head(int(max_targets)).iterrows():
        s = str(r["symbol"]); t = str(r["strategy"])
        if s and t: out.append((s, t, float(r["score"])))
    return out

# ── 런타임 락 ───────────────────────────────────────────────────
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

# ── 메인: 실패 많은 (symbol,strategy) 우선 재학습(미니에폭+롤백) ──
def run_failure_training(max_targets: int = MAX_TARGETS, lookback_days: int = LOOKBACK_DAYS_DEFAULT):
    """
    최근 wrong_*.csv + root wrong 기반으로 실패 많은 (symbol,strategy)를 골라
    각 그룹별 미니에폭 이어학습을 수행하고, 성능 악화 시 자동 롤백한다.
    - 쿨다운: 마지막 실행 후 COOLDOWN_MINUTES 분 이내면 스킵
    - 실행 중 런타임 락 생성(클린업/동시 실행 충돌 방지)
    - 요약은 logs/failure_retrain_summary.csv 에 append
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

    # 실패 데이터 로드
    df = _load_recent_wrong(days=lookback_days)
    if df.empty:
        _save_state({"last_run_ts": _now_kst().isoformat()})
        return {"ok": True, "skipped": True, "reason": "no_wrong_data"}

    targets = _pick_targets(df, lookback_days=lookback_days, max_targets=max_targets)
    if not targets:
        _save_state({"last_run_ts": _now_kst().isoformat()})
        return {"ok": True, "skipped": True, "reason": "no_targets"}

    summary = {"ok": True, "skipped": False, "targets": [], "started_at": _now_str()}

    _touch_lock()
    try:
        for (symbol, strategy, score) in targets:
            # 최신 클래스 경계/그룹
            try:
                class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
                if not class_ranges or len(class_ranges) < 2:
                    logger.log_training_result(symbol, strategy, model="failure_retrain",
                                               note="경계<2 → 스킵", status="skipped")
                    # 요약만 기록
                    _append_summary_row({
                        "timestamp": _now_str(), "symbol": symbol, "strategy": strategy,
                        "score": float(score), "group_id": -1,
                        "before_f1": "", "after_f1": "", "delta": "", "result": "skip_bounds", "backup_dir": ""
                    })
                    continue
                groups = get_class_groups(num_classes=len(class_ranges))
                max_gid = len(groups) - 1
            except Exception as e:
                logger.log_training_result(symbol, strategy, model="failure_retrain",
                                           note=f"경계계산실패:{e}", status="skipped")
                continue

            retrained = []
            for gid in range(max_gid + 1):
                # ── 백업 & 기준 성능
                backup_dir = _backup_group(symbol, strategy, gid)
                before_f1  = _best_f1_for_group(symbol, strategy, gid)
                print(f"[failure_learn] {symbol}-{strategy}-g{gid} 이전 최고 F1 = {before_f1}")

                # ── 미니에폭 재학습
                try:
                    train_one_model(symbol, strategy, group_id=gid, max_epochs=MINI_EPOCHS)
                    retrained.append(gid)
                except Exception as ge:
                    logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                               note=f"예외:{ge}", status="failed")
                    _append_summary_row({
                        "timestamp": _now_str(), "symbol": symbol, "strategy": strategy,
                        "score": float(score), "group_id": gid,
                        "before_f1": before_f1, "after_f1": "", "delta": "",
                        "result": "train_error", "backup_dir": backup_dir or ""
                    })
                    continue

                # 파일/별칭 반영 대기
                time.sleep(0.5)

                # ── 성능 비교 및 롤백
                after_f1 = _best_f1_for_group(symbol, strategy, gid)
                delta = None if (before_f1 is None or after_f1 is None) else float(after_f1 - before_f1)
                result = "kept"
                if ROLLBACK_ENABLE and before_f1 is not None and after_f1 is not None:
                    if delta < -ROLLBACK_TOLERANCE:
                        ok = _restore_from_backup(backup_dir)
                        result = "rollback" if ok else "rollback_failed"
                        note = f"미니에폭 후 악화(delta={delta:.4f} < -tol={ROLLBACK_TOLERANCE}) → 롤백"
                        logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                                   note=note, status=("rolled_back" if ok else "failed"))
                        print(f"[ROLLBACK] {symbol}-{strategy}-g{gid}: {note} (ok={ok})")
                    else:
                        status = "improved" if (delta is not None and delta > 0) else "kept"
                        logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                                   note=f"미니에폭 완료 delta={delta}", status=status)
                else:
                    logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                               note="미니에폭 완료(비교 불가/롤백 비활성)", status="done")

                # ── 요약 기록
                _append_summary_row({
                    "timestamp": _now_str(),
                    "symbol": symbol,
                    "strategy": strategy,
                    "score": float(score),
                    "group_id": gid,
                    "before_f1": ("" if before_f1 is None else float(before_f1)),
                    "after_f1": ("" if after_f1 is None else float(after_f1)),
                    "delta": ("" if delta is None else float(delta)),
                    "result": result,
                    "backup_dir": backup_dir or ""
                })

                # 짧은 휴식(파일 flush 등)
                time.sleep(0.2)

            summary["targets"].append({
                "symbol": symbol, "strategy": strategy, "score": score, "groups": retrained
            })
    finally:
        _release_lock()

    # 쿨다운 저장
    _save_state({"last_run_ts": _now_kst().isoformat()})
    return summary

if __name__ == "__main__":
    out = run_failure_training()
    print(out)
