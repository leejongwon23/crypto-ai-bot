# === failure_trainer.py (업데이트: 타깃 선정 + 그룹별 재학습 + 쿨다운) ===
import os, csv, json
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

STATE_JSON  = os.path.join(LOG_DIR, "failure_learn_state.json")      # 마지막 실행 시각 저장
SUMMARY_CSV = os.path.join(LOG_DIR, "failure_retrain_summary.csv")   # 재학습 요약 로그

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
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_summary_row(row: dict):
    write_header = not os.path.exists(SUMMARY_CSV)
    try:
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass

def _score_targets(failure_data, lookback_days=7, max_targets=8):
    """
    실패 샘플들을 (symbol,strategy)로 묶고 점수화해 상위 N개만 반환.
    - 점수 = 최근 7일 실패수 * 가중치(최근일수일수록 큼)
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
        if not s or not t:
            continue

        # 최근 가중치
        w = 1.0
        ts = item.get("timestamp")
        if ts:
            try:
                # 다양한 포맷 호환
                if isinstance(ts, str):
                    ts_dt = datetime.fromisoformat(ts.replace("Z","+00:00")) if "T" in ts else datetime.fromisoformat(ts)
                    if ts_dt.tzinfo is None:
                        ts_dt = KST.localize(ts_dt)
                    else:
                        ts_dt = ts_dt.astimezone(KST)
                else:
                    ts_dt = ts
                if ts_dt >= since:   w = 1.5
                if ts_dt >= now - timedelta(days=1): w = 2.0
            except Exception:
                w = 1.0

        scores[(s, t)] += w

    # 점수 상위 정렬
    ranked = sorted([(k[0], k[1], v) for k, v in scores.items()],
                    key=lambda x: x[2], reverse=True)
    return ranked[:max_targets]

def run_failure_training():
    """
    ✅ 실패했던 예측 샘플들로 '우선순위 높은 (symbol,strategy)'를 골라
       각 그룹별로 가벼운 재학습을 수행.
    - 과도 실행 방지: 마지막 실행 후 20분 이내면 스킵
    - 요약은 /persistent/logs/failure_retrain_summary.csv 에 적재
    """
    # 쿨다운 체크(20분)
    state = _load_state()
    last_ts = state.get("last_run_ts")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if datetime.now(KST) - last_dt < timedelta(minutes=20):
                print("⏳ 실패학습 쿨다운 중(20분). 이번 턴은 스킵.")
                return
        except Exception:
            pass

    failure_data = load_failure_samples()
    if not failure_data:
        print("✅ 실패 샘플 없음 → 실패학습 생략")
        return

    # (symbol,strategy) 타깃 선정
    targets = _score_targets(failure_data, lookback_days=7, max_targets=8)
    if not targets:
        print("✅ 타깃 없음(스코어 0) → 실패학습 생략")
        return

    print(f"🚨 실패학습 대상 {len(targets)}개:", targets)

    for symbol, strategy, score in targets:
        print(f"\n🚨 실패 학습 시작: {symbol}-{strategy} (score={score:.2f})")
        try:
            # 최신 클래스 경계/그룹 파악
            class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
            if not class_ranges or len(class_ranges) < 2:
                logger.log_training_result(symbol, strategy, model="failure_retrain",
                                           note="경계<2 → 스킵", status="skipped")
                print(f"⏭️ 경계<2 → 스킵: {symbol}-{strategy}")
                continue

            groups = get_class_groups(num_classes=len(class_ranges))
            max_gid = len(groups) - 1

            retrained = []
            for gid in range(max_gid + 1):
                try:
                    # 가벼운 재학습(에폭 축소)
                    train_one_model(symbol, strategy, group_id=gid, max_epochs=6)
                    retrained.append(gid)
                except Exception as ge:
                    logger.log_training_result(symbol, strategy, model=f"failure_retrain_g{gid}",
                                               note=f"예외:{ge}", status="failed")
                    print(f"[❌ 그룹 재학습 실패] {symbol}-{strategy}-g{gid} → {ge}")

            _append_summary_row({
                "timestamp": _now_str(),
                "symbol": symbol,
                "strategy": strategy,
                "score": float(score),
                "groups": ",".join(map(str, retrained))
            })

        except Exception as e:
            logger.log_training_result(symbol, strategy, model="failure_retrain",
                                       note=f"예외:{e}", status="failed")
            print(f"[❌ 실패 학습 예외] {symbol}-{strategy} → {e}")

    # 마지막 실행 시각 업데이트(쿨다운 기준)
    _save_state({"last_run_ts": datetime.now(KST).isoformat()})
