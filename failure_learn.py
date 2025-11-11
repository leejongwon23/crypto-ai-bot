# === failure_learn.py (v2025-11-11m: meta-aware targetting + lazy dirs + level/weight merge) ===
"""
failure_db 가 확장해서 남겨준
  - failure_level (recur/evo/noise)
  - train_weight
  - meta_choice
  - state_snapshot / meta_detail / picked_model
을 이용해서, 재학습 타깃을 더 정확히 고르는 버전.

핵심 변경
1) 모듈 import 시점에는 디렉터리 생성 안 함 → run 할 때 만들도록 조정
2) wrong_predictions.csv 를 읽을 때 새 컬럼들(failure_level, train_weight, meta_choice...)도 같이 들고와서
   가중치에 반영
3) recur 로 분류된 실패는 무조건 좀 더 세게, noise 는 0에 가깝게
4) 기존 백업/롤백/쿨다운/러ntime-lock 로직은 그대로 유지
"""

import os, json, time, glob, shutil, csv
from datetime import datetime, timedelta
import pandas as pd
import pytz

from config import get_class_groups, get_class_ranges
from train import train_one_model
import logger

# ── 경로/상수 정의 (생성은 나중에) ─────────────────────────────
PERSIST_DIR = "/persistent"
LOG_DIR  = os.path.join(PERSIST_DIR, "logs")
LOCK_DIR = os.path.join(PERSIST_DIR, "locks")

WRONG_CSV_ROOT = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
STATE_JSON     = os.path.join(LOG_DIR, "failure_learn_state.json")
SUMMARY_CSV    = os.path.join(LOG_DIR, "failure_retrain_summary.csv")
LOCK_PATH      = os.getenv("SAFE_LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))

# ── 설정 (환경변수로 조정) ─────────────────────────────────────
KST = pytz.timezone("Asia/Seoul")
CSV_CHUNKSIZE         = int(os.getenv("FAIL_LEARN_CHUNKSIZE", "50000"))
LOOKBACK_DAYS_DEFAULT = int(os.getenv("FAIL_LEARN_LOOKBACK_DAYS", "7"))
COOLDOWN_MINUTES      = int(os.getenv("FAIL_LEARN_COOLDOWN_MIN", "20"))
MINI_EPOCHS           = max(1, min(3, int(os.getenv("FAIL_MINI_EPOCHS", "2"))))
ROLLBACK_ENABLE       = os.getenv("ROLLBACK_ON_DEGRADE", "1") == "1"
ROLLBACK_TOLERANCE    = float(os.getenv("ROLLBACK_TOLERANCE", "0.01"))
MAX_TARGETS           = int(os.getenv("FAIL_MAX_TARGETS", "8"))

# 기존 시간가중
W_RECENT_DAY          = float(os.getenv("FAIL_WEIGHT_RECENT", "1.5"))
W_VERY_RECENT_DAY     = float(os.getenv("FAIL_WEIGHT_VERY_RECENT", "2.0"))
W_SHADOW_FAIL         = float(os.getenv("FAIL_WEIGHT_SHADOW", "1.3"))
W_NORMAL_FAIL         = float(os.getenv("FAIL_WEIGHT_NORMAL", "1.0"))

# failure_db 가 남긴 level/weight 추가 가중
W_LEVEL_RECUR         = float(os.getenv("FAIL_WEIGHT_LEVEL_RECUR", "2.0"))
W_LEVEL_EVO           = float(os.getenv("FAIL_WEIGHT_LEVEL_EVO", "1.2"))
W_LEVEL_NOISE         = float(os.getenv("FAIL_WEIGHT_LEVEL_NOISE", "0.0"))
W_TRAIN_WEIGHT_SCALE  = float(os.getenv("FAIL_WEIGHT_TRAIN_FIELD", "1.0"))

# ── 모델 아티팩트 탐색/백업/복구 유틸 ───────────────────────────
MODEL_DIR   = "/persistent/models"
KNOWN_EXTS  = (".ptz", ".pt", ".safetensors")
MODEL_TYPES = ("lstm", "cnn_lstm", "transformer")

def _ensure_runtime_dirs():
    """실행 시점에만 필요한 디렉터리 만든다."""
    for d in (LOG_DIR, LOCK_DIR, os.path.join(PERSIST_DIR, "tmp", "failure_retrain_backups")):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

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
    for mtype in MODEL_TYPES:
        for e in KNOWN_EXTS:
            p = os.path.join(MODEL_DIR, symbol, strategy, f"{mtype}{e}")
            if os.path.exists(p): cands.append(p)
    seen = set()
    for wpath in cands:
        stem = _stem_wo_ext(os.path.basename(wpath))
        if stem in seen: 
            continue
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
        if not meta_path or not os.path.exists(meta_path): 
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        for path in [
            ("metrics","val_f1"),
            ("metrics","best_val_f1"),
            ("val","f1"),
            ("f1",),
        ]:
            cur = m
            try:
                for k in path: 
                    cur = cur[k]
                if cur is not None: 
                    return float(cur)
            except Exception:
                continue
        return None
    except Exception:
        return None

BACKUP_DIR = os.path.join(PERSIST_DIR, "tmp", "failure_retrain_backups")

def _rel_path_under_model_dir(path: str) -> str:
    abs_p = os.path.abspath(path)
    abs_root = os.path.abspath(MODEL_DIR)
    if abs_p.startswith(abs_root):
        return os.path.relpath(abs_p, abs_root)
    return os.path.basename(path)

def _backup_group(symbol: str, strategy: str, group_id: int):
    try:
        ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"{symbol}_{strategy}_g{group_id}_{ts}")
        os.makedirs(dst, exist_ok=True)
        manifest = []
        copied = 0
        for it in _find_group_artifacts(symbol, strategy, group_id):
            for p in [it.get("weight"), it.get("meta")]:
                if p and os.path.exists(p):
                    rel = _rel_path_under_model_dir(p)
                    out_path = os.path.join(dst, rel)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    shutil.copy2(p, out_path)
                    manifest.append({"rel": rel})
                    copied += 1
        with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"items": manifest}, f, ensure_ascii=False, indent=2)
        return dst if copied > 0 else None
    except Exception as e:
        print(f"[백업 실패] {symbol}-{strategy}-g{group_id} → {e}")
        return None

def _restore_from_backup(backup_dir: str) -> bool:
    try:
        if not backup_dir or not os.path.isdir(backup_dir): 
            return False
        manifest_path = os.path.join(backup_dir, "manifest.json")
        items = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
                items = [it.get("rel") for it in doc.get("items", []) if it and it.get("rel")]
        else:
            for root, _, files in os.walk(backup_dir):
                for fn in files:
                    if fn == "manifest.json": 
                        continue
                    rel = os.path.relpath(os.path.join(root, fn), backup_dir)
                    items.append(rel)

        ok = True
        for rel in items:
            src = os.path.join(backup_dir, rel)
            dst = os.path.join(MODEL_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                ok = False
                print(f"[복구 실패] {src} → {dst} : {e}")
        return ok
    except Exception as e:
        print(f"[복구 실패] {backup_dir} → {e}")
        return False

def _best_f1_for_group(symbol: str, strategy: str, group_id: int):
    f1s = []
    for it in _find_group_artifacts(symbol, strategy, group_id):
        f1 = _read_meta_f1(it.get("meta"))
        if f1 is not None: 
            f1s.append(float(f1))
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

def _to_kst_aware(dt_str: str):
    if not dt_str:
        return None
    try:
        ts = pd.to_datetime(dt_str, errors="coerce", utc=True)
        if ts is not pd.NaT and pd.notnull(ts):
            return ts.tz_convert(KST).to_pydatetime()
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            return KST.localize(dt)
        return dt.astimezone(KST)
    except Exception:
        return None

def _save_state(state: dict):
    try:
        os.makedirs(os.path.dirname(STATE_JSON), exist_ok=True)
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_summary_row(row: dict):
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    base_fields = ["timestamp","symbol","strategy","score","group_id",
                   "before_f1","after_f1","delta","result","backup_dir"]
    for k in list(row.keys()):
        if k not in base_fields: 
            base_fields.append(k)
    try:
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=base_fields)
            if write_header: 
                w.writeheader()
            w.writerow({k: row.get(k,"") for k in base_fields})
    except Exception:
        pass

# ── 메모리-세이프 wrong 로그 머지 (failure_db 확장 컬럼 포함) ─────
def _iter_recent_rows_from_csv(path, cutoff_kst, chunksize=CSV_CHUNKSIZE):
    # failure_db.py 가 쓰는 확장 컬럼들까지 포함
    cols = [
        "timestamp","symbol","strategy","status","success","note",
        "failure_level","train_weight","meta_choice",
        "state_snapshot","meta_detail","picked_model",
    ]
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
        df = _safe_read_csv(path)
        if df.empty: 
            return
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
        return pd.DataFrame(columns=[
            "timestamp","symbol","strategy","status","success","note",
            "failure_level","train_weight","meta_choice",
            "state_snapshot","meta_detail","picked_model","__ts"
        ])
    cols_union = [
        "timestamp","symbol","strategy","status","success","note",
        "failure_level","train_weight","meta_choice",
        "state_snapshot","meta_detail","picked_model","__ts"
    ]
    df = pd.concat([p[cols_union] if all(c in p.columns for c in cols_union) else p for p in parts],
                   ignore_index=True)
    keep_cols = [c for c in ["__ts","symbol","strategy","status","success","note","failure_level","meta_choice"] if c in df.columns]
    try: 
        df = df.drop_duplicates(subset=keep_cols)
    except Exception: 
        df = df.drop_duplicates()
    return df

# ── 타깃 선정(레벨/학습가중 포함) ───────────────────────────────
def _pick_targets(df: pd.DataFrame, lookback_days=LOOKBACK_DAYS_DEFAULT, max_targets=8):
    if df.empty: 
        return []
    ts_col = "__ts" if "__ts" in df.columns else "timestamp"
    if ts_col == "timestamp":
        try:
            df[ts_col] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        except Exception:
            df[ts_col] = pd.NaT

    now = _now_kst()
    since = now - timedelta(days=int(lookback_days))

    dff = df.copy()

    # 실패 행만
    if "status" in dff.columns:
        dff = dff[dff["status"].astype(str).str.lower().isin(["fail", "v_fail", "shadow_fail"])]
    elif "success" in dff.columns:
        dff = dff[dff["success"].astype(str).str.lower().isin(["false", "0", "no", "n"])]

    if dff.empty:
        return []

    # 기본 가중
    dff["w"] = W_NORMAL_FAIL

    # 섀도우/강한 실패
    try:
        flags = (dff.get("status", "").astype(str).str.lower() + " " +
                 dff.get("note",   "").astype(str).str.lower())
        shadow_mask = flags.str.contains("shadow", na=False) | dff.get("status", "").astype(str).str.lower().eq("v_fail")
        dff.loc[shadow_mask, "w"] = dff.loc[shadow_mask, "w"] * W_SHADOW_FAIL
    except Exception:
        pass

    # 날짜 가중
    try:
        recent_mask = dff[ts_col].notna() & (dff[ts_col] >= since)
        dff.loc[recent_mask, "w"] = dff.loc[recent_mask, "w"] * W_RECENT_DAY
        very_recent = dff[ts_col].notna() & (dff[ts_col] >= now - timedelta(days=1))
        dff.loc[very_recent, "w"] = dff.loc[very_recent, "w"] * W_VERY_RECENT_DAY
    except Exception:
        pass

    # failure_db 의 level / train_weight 반영
    if "failure_level" in dff.columns:
        dff["failure_level"] = dff["failure_level"].fillna("").astype(str).str.lower()
        dff.loc[dff["failure_level"] == "recur", "w"] *= W_LEVEL_RECUR
        dff.loc[dff["failure_level"] == "evo",   "w"] *= W_LEVEL_EVO
        dff.loc[dff["failure_level"] == "noise", "w"] *= W_LEVEL_NOISE

    if "train_weight" in dff.columns:
        try:
            tw = pd.to_numeric(dff["train_weight"], errors="coerce").fillna(0.0)
            dff["w"] = dff["w"] * (1.0 + W_TRAIN_WEIGHT_SCALE * tw)
        except Exception:
            pass

    # meta_choice 가 rule_fallback 이나 low-confidence였던 것들은 조금 더
    if "meta_choice" in dff.columns:
        mc = dff["meta_choice"].astype(str)
        mc_mask = mc.str.contains("rule", na=False) | mc.str.contains("fallback", na=False)
        dff.loc[mc_mask, "w"] = dff.loc[mc_mask, "w"] * 1.3

    cols = [c for c in ["symbol","strategy"] if c in dff.columns]
    if len(cols) < 2: 
        return []
    g = dff.groupby(cols, dropna=False)["w"].sum().reset_index(name="score")
    g = g.sort_values("score", ascending=False)

    out = []
    for _, r in g.head(int(max_targets)).iterrows():
        s = str(r["symbol"]); t = str(r["strategy"])
        if s and t: 
            out.append((s, t, float(r["score"])))
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

# ── 메인 ────────────────────────────────────────────────────────
def run_failure_training(max_targets: int = MAX_TARGETS, lookback_days: int = LOOKBACK_DAYS_DEFAULT):
    _ensure_runtime_dirs()

    if os.path.exists(LOCK_PATH):
        return {"ok": True, "skipped": True, "reason": "global_lock"}

    state = _load_state()
    last_ts = state.get("last_run_ts")
    if last_ts:
        last_dt = _to_kst_aware(last_ts)
        if last_dt is not None:
            try:
                if (_now_kst() - last_dt) < timedelta(minutes=COOLDOWN_MINUTES):
                    return {"ok": True, "skipped": True, "reason": "cooldown", "last_run_ts": last_ts}
            except Exception:
                pass

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
            try:
                class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
                if not class_ranges or len(class_ranges) < 2:
                    logger.log_training_result(symbol, strategy, model="failure_retrain",
                                               note="경계<2 → 스킵", status="skipped")
                    _append_summary_row({
                        "timestamp": _now_str(), "symbol": symbol, "strategy": strategy,
                        "score": float(score), "group_id": -1,
                        "before_f1": "", "after_f1": "", "delta": "",
                        "result": "skip_bounds", "backup_dir": ""
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
                backup_dir = _backup_group(symbol, strategy, gid)
                before_f1  = _best_f1_for_group(symbol, strategy, gid)
                print(f"[failure_learn] {symbol}-{strategy}-g{gid} 이전 최고 F1 = {before_f1}")

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

                time.sleep(0.5)

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

                time.sleep(0.2)

            summary["targets"].append({
                "symbol": symbol, "strategy": strategy, "score": score, "groups": retrained
            })
    finally:
        _release_lock()

    _save_state({"last_run_ts": _now_kst().isoformat()})
    return summary

# --- backward-compatible alias ---
def run(limit: int | None = None,
        lookback_days: int | None = None,
        max_targets: int | None = None):
    if lookback_days is None:
        lookback_days = LOOKBACK_DAYS_DEFAULT
    if max_targets is None:
        max_targets = MAX_TARGETS
    return run_failure_training(max_targets=max_targets, lookback_days=lookback_days)

if __name__ == "__main__":
    out = run_failure_training()
    print(out)
