# === failure_trainer.py (v2025-10-26 안정완성: TZ-safe cooldown, robust backup/restore, shadow weighting, safer metrics, self-contained loader) ===
import os, csv, json, glob, shutil, time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import pytz

# 외부 의존
from train import train_one_model
from config import get_class_ranges, get_class_groups
import logger  # 안전 로그용

# ── 공통 경로/타임존 ───────────────────────────────────────────
KST = pytz.timezone("Asia/Seoul")
PERSIST_DIR = "/persistent"
LOG_DIR  = os.path.join(PERSIST_DIR, "logs")
LOCK_DIR = os.path.join(PERSIST_DIR, "locks")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)

STATE_JSON  = os.path.join(LOG_DIR, "failure_learn_state.json")        # 마지막 실행 시각 저장
SUMMARY_CSV = os.path.join(LOG_DIR, "failure_retrain_summary.csv")     # 재학습 요약 로그
BACKUP_DIR  = os.path.join(PERSIST_DIR, "tmp", "failure_retrain_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)
LOCK_PATH   = os.getenv("SAFE_LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))

# 실패 로그 소스(로더가 읽는 표준 + 히스토리)
WRONG_CSV_ROOT = os.path.join(PERSIST_DIR, "wrong_predictions.csv")

# ── 환경 파라미터 ─────────────────────────────────────────────
COOLDOWN_MIN         = int(os.getenv("FAIL_RETRAIN_COOLDOWN_MIN", "20"))
MINI_EPOCHS          = max(1, min(3, int(os.getenv("FAIL_MINI_EPOCHS", "2"))))   # 1~3
ROLLBACK_ENABLE      = os.getenv("ROLLBACK_ON_DEGRADE", "1") == "1"
ROLLBACK_TOLERANCE   = float(os.getenv("ROLLBACK_TOLERANCE", "0.01"))            # 성능악화 허용오차
MAX_TARGETS          = int(os.getenv("FAIL_MAX_TARGETS", "8"))
LOOKBACK_DAYS        = int(os.getenv("FAIL_LOOKBACK_DAYS", "7"))
CSV_CHUNKSIZE        = int(os.getenv("FAIL_LEARN_CHUNKSIZE", "50000"))

# 섀도우 실패 가중치(최근/섀도우 우선순위 강화)
W_RECENT_DAY         = float(os.getenv("FAIL_WEIGHT_RECENT", "1.5"))
W_VERY_RECENT_DAY    = float(os.getenv("FAIL_WEIGHT_VERY_RECENT", "2.0"))
W_SHADOW_FAIL        = float(os.getenv("FAIL_WEIGHT_SHADOW", "1.3"))   # 섀도우 실패 보정 가중
W_NORMAL_FAIL        = float(os.getenv("FAIL_WEIGHT_NORMAL", "1.0"))

# ── 모델/아티팩트 위치 ────────────────────────────────────────
MODEL_DIR   = "/persistent/models"
KNOWN_EXTS  = (".ptz", ".pt", ".safetensors")
MODEL_TYPES = ("lstm", "cnn_lstm", "transformer")

# ── 유틸 ──────────────────────────────────────────────────────
def _now_kst() -> datetime:
    return datetime.now(KST)

def _now_str() -> str:
    return _now_kst().strftime("%Y-%m-%d %H:%M:%S")

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
        os.makedirs(os.path.dirname(STATE_JSON), exist_ok=True)
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_summary_row(row: dict):
    write_header = not os.path.exists(SUMMARY_CSV)
    try:
        # 고정 헤더 + 추가 필드 보존
        base_fields = ["timestamp","symbol","strategy","score","group_id",
                       "before_f1","after_f1","delta","result","backup_dir"]
        for k in row.keys():
            if k not in base_fields:
                base_fields.append(k)
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=base_fields)
            if write_header: w.writeheader()
            w.writerow({k: row.get(k,"") for k in base_fields})
    except Exception:
        pass

# ── 실패 로그 로더(자급자족: wrong_predictions.csv + logs/wrong_*.csv) ──
def _iter_recent_rows_from_csv(path: str, cutoff_kst: datetime, chunksize: int = CSV_CHUNKSIZE):
    # 실패행 선별에 필요한 최소 컬럼
    cols = ["timestamp", "symbol", "strategy", "status", "success", "note"]
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
            # timestamp → KST aware 시리즈로 변환
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
        # 파싱 실패 시 통으로라도 읽기
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            if df.empty:
                return
            df["__ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
            yield df
        except Exception:
            return

def _load_recent_failures(days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    cutoff = _now_kst() - timedelta(days=int(days))
    parts: List[pd.DataFrame] = []

    if os.path.exists(WRONG_CSV_ROOT):
        for sub in _iter_recent_rows_from_csv(WRONG_CSV_ROOT, cutoff):
            parts.append(sub)

    for path in glob.glob(os.path.join(LOG_DIR, "wrong_*.csv")):
        for sub in _iter_recent_rows_from_csv(path, cutoff):
            parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=["timestamp","symbol","strategy","status","success","note","__ts"])

    cols_union = ["timestamp","symbol","strategy","status","success","note","__ts"]
    df = pd.concat([p[[c for c in cols_union if c in p.columns]] for p in parts], ignore_index=True)
    keep_cols = [c for c in ["__ts","symbol","strategy","status","success","note"] if c in df.columns]
    try:
        df = df.drop_duplicates(subset=keep_cols)
    except Exception:
        df = df.drop_duplicates()
    return df

def _coerce_bool_like(v: Any) -> Optional[bool]:
    s = str(v).strip().lower()
    if s in ("true","1","yes","y"):
        return True
    if s in ("false","0","no","n"):
        return False
    return None

# ── 모델/메타 탐색 & 백업/복구 ────────────────────────────────
def _stem_without_ext(p: str) -> str:
    base = p
    for e in KNOWN_EXTS:
        if base.endswith(e):
            return base[:-len(e)]
    return os.path.splitext(base)[0]

def _rel_under_model_dir(path: str) -> str:
    absp = os.path.abspath(path)
    absroot = os.path.abspath(MODEL_DIR)
    if absp.startswith(absroot):
        return os.path.relpath(absp, absroot)
    return os.path.basename(path)

def _find_group_artifacts(symbol: str, strategy: str, group_id: int):
    """
    해당 심볼/전략/그룹의 모델 가중치와 .meta.json 경로 수집.
    """
    items = []
    patt = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*_group{int(group_id)}_cls*")
    cands = []
    for e in KNOWN_EXTS:
        cands.extend(glob.glob(patt + e))
    # 디렉토리별 별칭 경로도 탐색
    for mtype in MODEL_TYPES:
        for e in KNOWN_EXTS:
            p = os.path.join(MODEL_DIR, symbol, strategy, f"{mtype}{e}")
            if os.path.exists(p):
                cands.append(p)

    seen = set()
    for wpath in cands:
        stem = _stem_without_ext(os.path.basename(wpath))
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
    """여러 키 후보에서 val_f1 탐색."""
    try:
        if not meta_path or not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        for k in [
            ("metrics","val_f1"),
            ("metrics","best_val_f1"),
            ("val","f1"),
            ("f1",),
        ]:
            cur = m
            try:
                for kk in k:
                    cur = cur[kk]
                if cur is not None:
                    return float(cur)
            except Exception:
                continue
        return None
    except Exception:
        return None

def _backup_group(symbol: str, strategy: str, group_id: int):
    """
    모델/메타를 MODEL_DIR 기준 상대경로로 백업(manifest 포함) → 원위치 복구 가능
    """
    try:
        ts = _now_kst().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"{symbol}_{strategy}_g{group_id}_{ts}")
        os.makedirs(dst, exist_ok=True)
        copied = 0
        manifest = []
        for it in _find_group_artifacts(symbol, strategy, group_id):
            for p in [it.get("weight"), it.get("meta")]:
                if p and os.path.exists(p):
                    rel = _rel_under_model_dir(p)
                    out = os.path.join(dst, rel)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    shutil.copy2(p, out)
                    manifest.append({"rel": rel})
                    copied += 1
        with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"items": manifest}, f, ensure_ascii=False, indent=2)
        return dst if copied > 0 else None
    except Exception as e:
        print(f"[백업 실패] {symbol}-{strategy}-g{group_id} → {e}")
        return None

def _restore_from_backup(backup_dir: str) -> bool:
    """
    manifest.json을 사용하여 MODEL_DIR 기준 원위치 복구.
    manifest 없으면 폴더 전체 평면 복구(하위 호환).
    """
    try:
        if not backup_dir or not os.path.isdir(backup_dir):
            return False
        manifest_path = os.path.join(backup_dir, "manifest.json")
        rels: List[str] = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
                rels = [it.get("rel") for it in doc.get("items", []) if it and it.get("rel")]
        else:
            for root, _, files in os.walk(backup_dir):
                for fn in files:
                    if fn == "manifest.json":
                        continue
                    rels.append(os.path.relpath(os.path.join(root, fn), backup_dir))

        ok = True
        for rel in rels:
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

# ── 타깃 스코어링(섀도우 실패 가중 포함) ───────────────────────
def _score_targets(df: pd.DataFrame, lookback_days=LOOKBACK_DAYS, max_targets=MAX_TARGETS):
    """
    실패 행들을 (symbol, strategy)로 묶고 점수화해 상위 N개만 반환.
    - 기본 가중치: 1.0
    - 최근(lookback) 범위면 W_RECENT_DAY, 1일 이내면 W_VERY_RECENT_DAY
    - 섀도우 실패(status/note에 'shadow' 또는 'v_fail' 포함, 혹은 is_shadow 플래그)면 W_SHADOW_FAIL 곱
    반환: [(symbol, strategy, score), ...]
    """
    if df is None or df.empty:
        return []

    ts_col = "__ts" if "__ts" in df.columns else "timestamp"
    now = _now_kst()
    since = now - timedelta(days=int(lookback_days))

    dff = df.copy()

    # 실패행 선별: status 우선, 없으면 success=False류
    if "status" in dff.columns:
        status_l = dff["status"].astype(str).str.lower()
        dff = dff[status_l.isin(["fail", "v_fail", "shadow_fail"])]
    elif "success" in dff.columns:
        succ = dff["success"].apply(_coerce_bool_like)
        dff = dff[succ == False]  # noqa: E712

    if dff.empty:
        return []

    # 기본 가중
    dff["w"] = W_NORMAL_FAIL

    # 섀도우 가중
    try:
        status = dff.get("status")
        note = dff.get("note")
        flags = (status.astype(str).str.lower() if status is not None else "") \
                + " " + (note.astype(str).str.lower() if note is not None else "")
        shadow_mask = flags.str.contains("shadow", na=False) | (status.astype(str).str.lower().isin(["v_fail","shadow_fail"]) if status is not None else False)
        dff.loc[shadow_mask, "w"] = dff.loc[shadow_mask, "w"] * W_SHADOW_FAIL
    except Exception:
        pass

    # 시점 가중
    try:
        if ts_col not in dff.columns:
            dff[ts_col] = pd.to_datetime(dff["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        recent_mask = dff[ts_col].notna() & (dff[ts_col] >= since)
        dff.loc[recent_mask, "w"] = dff.loc[recent_mask, "w"] * W_RECENT_DAY
        very_recent = dff[ts_col].notna() & (dff[ts_col] >= now - timedelta(days=1))
        dff.loc[very_recent, "w"] = dff.loc[very_recent, "w"] * W_VERY_RECENT_DAY
    except Exception:
        pass

    cols = [c for c in ["symbol","strategy","w"] if c in dff.columns]
    if len(cols) < 3:
        return []
    g = dff.groupby(["symbol","strategy"], dropna=False)["w"].sum().reset_index(name="score")
    g = g.sort_values("score", ascending=False)

    out = []
    for _, r in g.head(int(max_targets)).iterrows():
        s = str(r["symbol"]); t = str(r["strategy"])
        if s and t:
            out.append((s, t, float(r["score"])))
    return out

# ── 런타임 락(동시 실행 충돌 방지) ─────────────────────────────
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

# ── 메인 루틴 ─────────────────────────────────────────────────
def run_failure_training():
    """
    ✅ 실패샘플 기반 '이어학습'(미니에폭) + 성능악화 시 롤백.
    - 섀도우 실패를 점수에 포함(가중)해 우선순위 반영.
    - 과도 실행 방지: 마지막 실행 후 COOLDOWN_MIN 분 이내면 스킵.
    - 런타임 락으로 train/predict 충돌 방지.
    - 요약은 /persistent/logs/failure_retrain_summary.csv 에 적재.
    """
    # 쿨다운 체크 (TZ-aware 안전 비교)
    state = _load_state()
    last_ts = state.get("last_run_ts")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if last_dt.tzinfo is None:
                last_dt = KST.localize(last_dt)
            if _now_kst() - last_dt < timedelta(minutes=COOLDOWN_MIN):
                print(f"⏳ 실패학습 쿨다운 중({COOLDOWN_MIN}분). 이번 턴은 스킵.")
                return
        except Exception:
            pass

    # 실패 데이터 로드(자체 로더)
    df_fail = _load_recent_failures(days=LOOKBACK_DAYS)
    if df_fail.empty:
        print("✅ 실패 샘플 없음 → 실패학습 생략")
        _save_state({"last_run_ts": _now_kst().isoformat()})
        return

    targets = _score_targets(df_fail, lookback_days=LOOKBACK_DAYS, max_targets=MAX_TARGETS)
    if not targets:
        print("✅ 타깃 없음(스코어 0) → 실패학습 생략")
        _save_state({"last_run_ts": _now_kst().isoformat()})
        return

    print(f"🚨 실패학습 대상 {len(targets)}개:", targets)

    # 전역 락
    if os.path.exists(LOCK_PATH):
        print("🔒 전역 락 감지 → 안전상 이번 턴 스킵")
        _save_state({"last_run_ts": _now_kst().isoformat()})
        return

    _touch_lock()
    try:
        for symbol, strategy, score in targets:
            print(f"\n🚨 실패 학습 시작: {symbol}-{strategy} (score={score:.2f})")
            try:
                class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
                if not class_ranges or len(class_ranges) < 2:
                    logger.log_training_result(symbol, strategy, model="failure_retrain",
                                               note="경계<2 → 스킵", status="skipped")
                    print(f"⏭️ 경계<2 → 스킵: {symbol}-{strategy}")
                    _append_summary_row({
                        "timestamp": _now_str(), "symbol": symbol, "strategy": strategy,
                        "score": float(score), "group_id": -1,
                        "before_f1": "", "after_f1": "", "delta": "", "result": "skip_bounds", "backup_dir": ""
                    })
                    continue

                groups = get_class_groups(num_classes=len(class_ranges))
                max_gid = len(groups) - 1

                for gid in range(max_gid + 1):
                    # ── 백업 & 기준 성능
                    backup_dir = _backup_group(symbol, strategy, gid)
                    before_f1  = _best_f1_for_group(symbol, strategy, gid)
                    print(f"[INFO] g{gid} 이전 최고 F1 = {before_f1}")

                    # ── 미니에폭 재학습
                    try:
                        train_one_model(symbol, strategy, group_id=gid, max_epochs=MINI_EPOCHS)
                    except Exception as ge:
                        logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                                   note=f"예외:{ge}", status="failed")
                        print(f"[❌ 그룹 재학습 실패] {symbol}-{strategy}-g{gid} → {ge}")
                        _append_summary_row({
                            "timestamp": _now_str(), "symbol": symbol, "strategy": strategy,
                            "score": float(score), "group_id": gid, "before_f1": before_f1,
                            "after_f1": "", "delta": "", "result": "train_error", "backup_dir": backup_dir or ""
                        })
                        continue

                    # 파일 쓰기/링크 지연 대비 잠깐 대기
                    time.sleep(0.5)

                    # ── 성능 비교/로그/롤백
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
                            print(f"[ROLLBACK] g{gid}: {note} (ok={ok})")
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

            except Exception as e:
                logger.log_training_result(symbol, strategy, model="failure_retrain",
                                           note=f"예외:{e}", status="failed")
                print(f"[❌ 실패 학습 예외] {symbol}-{strategy} → {e}")
    finally:
        _release_lock()

    # 마지막 실행 시각 업데이트(쿨다운 기준)
    _save_state({"last_run_ts": _now_kst().isoformat()})

# 파일 맨 아래에
def retrain_failures(limit: int | None = None,
                     lookback_days: int | None = None,
                     max_targets: int | None = None):
    # 호환용 래퍼(파라미터는 현재 내부에서 직접 사용하지 않음)
    return run_failure_training()

if __name__ == "__main__":
    run_failure_training()
