# === failure_trainer.py (v2025-09-17b: 미니에폭 이어학습 + 성능저하 롤백 + 요약 로그) ===
import os, csv, json, glob, shutil, time
from datetime import datetime, timedelta
import pytz

from failure_db import load_failure_samples
from train import train_one_model
from config import get_class_ranges, get_class_groups
import logger  # 안전 로그용

KST = pytz.timezone("Asia/Seoul")
PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

STATE_JSON  = os.path.join(LOG_DIR, "failure_learn_state.json")        # 마지막 실행 시각 저장
SUMMARY_CSV = os.path.join(LOG_DIR, "failure_retrain_summary.csv")     # 재학습 요약 로그
BACKUP_DIR  = os.path.join(PERSIST_DIR, "tmp", "failure_retrain_backups")

# 환경 파라미터
COOLDOWN_MIN         = int(os.getenv("FAIL_RETRAIN_COOLDOWN_MIN", "20"))
MINI_EPOCHS          = max(1, min(3, int(os.getenv("FAIL_MINI_EPOCHS", "2"))))   # 1~3
ROLLBACK_ENABLE      = os.getenv("ROLLBACK_ON_DEGRADE", "1") == "1"
ROLLBACK_TOLERANCE   = float(os.getenv("ROLLBACK_TOLERANCE", "0.01"))            # 성능악화 허용오차
MAX_TARGETS          = int(os.getenv("FAIL_MAX_TARGETS", "8"))
LOOKBACK_DAYS        = int(os.getenv("FAIL_LOOKBACK_DAYS", "7"))

MODEL_DIR = "/persistent/models"
KNOWN_EXTS = (".ptz", ".pt", ".safetensors")
MODEL_TYPES = ("lstm", "cnn_lstm", "transformer")

def _now_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

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
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            # 헤더 컬럼 안정화
            base_fields = ["timestamp","symbol","strategy","score","group_id",
                           "before_f1","after_f1","delta","result","backup_dir"]
            for k in list(row.keys()):
                if k not in base_fields: base_fields.append(k)
            w = csv.DictWriter(f, fieldnames=base_fields)
            if write_header: w.writeheader()
            w.writerow({k: row.get(k,"") for k in base_fields})
    except Exception:
        pass

# ===== 모델/메타 탐색 & 백업/복원 유틸 =====
def _stem_without_ext(p: str) -> str:
    base = p
    for e in KNOWN_EXTS:
        if base.endswith(e):
            return base[:-len(e)]
    return os.path.splitext(base)[0]

def _find_group_artifacts(symbol: str, strategy: str, group_id: int):
    """해당 심볼/전략/그룹의 모델 가중치(.pt/.ptz/...)와 .meta.json 경로들을 수집."""
    items = []
    patt = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*_group{int(group_id)}_cls*")
    cands = []
    for e in KNOWN_EXTS:
        cands.extend(glob.glob(patt + e))
    # 디렉토리별 별칭 경로도 탐색
    for mtype in MODEL_TYPES:
        for e in KNOWN_EXTS:
            p = os.path.join(MODEL_DIR, symbol, strategy, f"{mtype}{e}")
            if os.path.exists(p): cands.append(p)

    seen = set()
    for wpath in cands:
        stem = _stem_without_ext(os.path.basename(wpath))
        if stem in seen: continue
        seen.add(stem)
        meta1 = os.path.join(MODEL_DIR, f"{stem}.meta.json")
        meta2 = os.path.join(MODEL_DIR, symbol, strategy, f"{stem.split('_')[2]}.meta.json") if "_" in stem else None
        meta_path = meta1 if os.path.exists(meta1) else (meta2 if (meta2 and os.path.exists(meta2)) else None)
        items.append({"weight": wpath, "meta": meta_path})
    return items

def _read_meta_f1(meta_path: str) -> float | None:
    try:
        if not meta_path or not os.path.exists(meta_path): return None
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return float(m.get("metrics", {}).get("val_f1", None))
    except Exception:
        return None

def _backup_group(symbol: str, strategy: str, group_id: int) -> str | None:
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
            # 심볼/전략 하위 별칭 파일이면 위치 다를 수 있음 → 그대로 루트에 복구, 별칭은 train 내 로직에서 다시 생성됨
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[복구 실패] {backup_dir} → {e}")
        return False

def _best_f1_for_group(symbol: str, strategy: str, group_id: int) -> float | None:
    f1s = []
    for it in _find_group_artifacts(symbol, strategy, group_id):
        f1 = _read_meta_f1(it.get("meta"))
        if f1 is not None: f1s.append(float(f1))
    return max(f1s) if f1s else None

# ===== 실패 타깃 스코어링 =====
def _score_targets(failure_data, lookback_days=LOOKBACK_DAYS, max_targets=MAX_TARGETS):
    """
    실패 샘플들을 (symbol,strategy)로 묶고 점수화해 상위 N개만 반환.
    - 점수 = 최근 N일 실패수 * 가중치(최근일수일수록 큼)
    - timestamp가 없으면 기본 가중치 1.0
    반환: [(symbol, strategy, score), ...]
    """
    from collections import defaultdict
    scores = defaultdict(float)
    now = datetime.now(KST)
    since = now - timedelta(days=lookback_days)

    for item in failure_data:
        s = str(item.get("symbol", "") or "")
        t = str(item.get("strategy", "") or "")
        if not s or not t: continue

        w = 1.0
        ts = item.get("timestamp")
        if ts:
            try:
                if isinstance(ts, str):
                    ts_dt = datetime.fromisoformat(ts.replace("Z","+00:00")) if "T" in ts else datetime.fromisoformat(ts)
                    ts_dt = (KST.localize(ts_dt) if ts_dt.tzinfo is None else ts_dt.astimezone(KST))
                else:
                    ts_dt = ts
                if ts_dt >= since:                 w = 1.5
                if ts_dt >= now - timedelta(days=1): w = 2.0
            except Exception:
                w = 1.0

        scores[(s, t)] += w

    ranked = sorted([(k[0], k[1], v) for k, v in scores.items()], key=lambda x: x[2], reverse=True)
    return ranked[:max_targets]

# ===== 메인 루틴 =====
def run_failure_training():
    """
    ✅ 실패샘플 기반 '이어학습' (미니에폭) + 성능악화 시 롤백.
    - 과도 실행 방지: 마지막 실행 후 COOLDOWN_MIN 분 이내면 스킵
    - 요약은 /persistent/logs/failure_retrain_summary.csv 에 적재
    """
    # 쿨다운 체크
    state = _load_state()
    last_ts = state.get("last_run_ts")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if datetime.now(KST) - last_dt < timedelta(minutes=COOLDOWN_MIN):
                print(f"⏳ 실패학습 쿨다운 중({COOLDOWN_MIN}분). 이번 턴은 스킵.")
                return
        except Exception:
            pass

    failure_data = load_failure_samples()
    if not failure_data:
        print("✅ 실패 샘플 없음 → 실패학습 생략")
        _save_state({"last_run_ts": datetime.now(KST).isoformat()})
        return

    targets = _score_targets(failure_data, lookback_days=LOOKBACK_DAYS, max_targets=MAX_TARGETS)
    if not targets:
        print("✅ 타깃 없음(스코어 0) → 실패학습 생략")
        _save_state({"last_run_ts": datetime.now(KST).isoformat()})
        return

    print(f"🚨 실패학습 대상 {len(targets)}개:", targets)

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

                after_f1 = _best_f1_for_group(symbol, strategy, gid)
                delta = None if (before_f1 is None or after_f1 is None) else float(after_f1 - before_f1)
                result = "kept"

                # ── 롤백 판단
                if ROLLBACK_ENABLE and before_f1 is not None and after_f1 is not None:
                    if delta < -ROLLBACK_TOLERANCE:
                        ok = _restore_from_backup(backup_dir)
                        result = "rollback" if ok else "rollback_failed"
                        note = f"미니에폭 후 악화(delta={delta:.4f} < -tol={ROLLBACK_TOLERANCE}) → 롤백"
                        logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                                   note=note, status=("rolled_back" if ok else "failed"))
                        print(f"[ROLLBACK] g{gid}: {note} (ok={ok})")
                    else:
                        # 개선/유지 시 성공 기록
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

    # 마지막 실행 시각 업데이트(쿨다운 기준)
    _save_state({"last_run_ts": datetime.now(KST).isoformat()})
