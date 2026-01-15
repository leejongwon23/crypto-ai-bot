# recommend.py (PATCHED: MEM-SAFE FINAL — meta-only success rate + cached model list + recent-window volatility)
import os
import csv
import json
import traceback
import datetime
import pytz
import math
import pandas as pd

from predict import predict
# 🔐 예측 게이트(있으면 사용, 없으면 no-op)
try:
    from predict import open_predict_gate, close_predict_gate
except Exception:
    def open_predict_gate(*a, **k): return None
    def close_predict_gate(*a, **k): return None

# data.utils 경로 불확실성 보호: data.utils 또는 루트 utils
try:
    from data.utils import SYMBOLS, get_kline_by_strategy
except Exception:
    try:
        from utils import SYMBOLS, get_kline_by_strategy
    except Exception:
        SYMBOLS = []
        def get_kline_by_strategy(symbol, strategy):
            return None

# logger 함수 폴백 (안정성)
try:
    from logger import (
        ensure_prediction_log_exists,     # prediction_log 보장
        get_meta_success_rate,            # 메타(선택)만 성공률 집계 — 청크 기반
        get_strategy_eval_count,          # 메타+섀도우 평가 완료 건수 — 청크 기반
        get_PREDICTION_LOG_PATH           # 경로가 있으면 재사용
    )
except Exception:
    def ensure_prediction_log_exists():
        return None
    def get_meta_success_rate(strategy, min_samples=0):
        return 0.0
    def get_strategy_eval_count(strategy):
        return 0
    def get_PREDICTION_LOG_PATH():
        return os.path.join("/tmp/appdata", "prediction_log.csv")

# telegram bot 폴백
try:
    from telegram_bot import send_message
except Exception:
    def send_message(msg):
        print(f"[TELEGRAM MISSING] {msg}")

# === 공통 기본 경로 (render처럼 /persistent 못 쓰는 환경 대비) ===
BASE_DIR = os.getenv("APP_DATA_DIR", "/tmp/appdata")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/appdata/models")
os.makedirs(LOG_DIR, exist_ok=True)

# === 설정 (환경변수로도 조절 가능) ===
MIN_SUCCESS_RATE = float(os.getenv("RECO_MIN_SUCCESS_RATE", "0.65"))
MIN_SAMPLES      = int(os.getenv("RECO_MIN_SAMPLES", "10"))
VOL_RT_단기      = float(os.getenv("VOL_TH_SHORT",  "0.003"))
VOL_RT_중기      = float(os.getenv("VOL_TH_MID",    "0.005"))
VOL_RT_장기      = float(os.getenv("VOL_TH_LONG",   "0.008"))
VOL_LOOKBACK_MAX = int(os.getenv("VOL_LOOKBACK_MAX","120"))  # 변동성 계산 시 최근 N행만 사용

# 현재 KST 시각
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# 전략별 변동성 기준
STRATEGY_VOL = {"단기": VOL_RT_단기, "중기": VOL_RT_중기, "장기": VOL_RT_장기}

# 로그 경로 (★ /persistent 제거, 환경변수/기본경로로 통일)
AUDIT_LOG = os.path.join(LOG_DIR, "prediction_audit.csv")
FAILURE_LOG = os.path.join(LOG_DIR, "failure_count.csv")
try:
    PREDICTION_LOG = get_PREDICTION_LOG_PATH()
except Exception:
    PREDICTION_LOG = os.path.join(BASE_DIR, "prediction_log.csv")

# ──────────────────────────────────────────────────────────────
# 성공률 필터 (성공률 ≥65% + 최소 10회 평가 완료)
#   - 성공률: 메타(선택된) 예측만 집계
#   - 표본수: 메타+섀도우 모두 중 '성공/실패'로 평가 끝난 건수
# ──────────────────────────────────────────────────────────────
def check_prediction_filter(strategy, min_success_rate=MIN_SUCCESS_RATE, min_samples=MIN_SAMPLES):
    rate = float(get_meta_success_rate(strategy, min_samples=min_samples) or 0.0)
    n = int(get_strategy_eval_count(strategy) or 0)
    return (n >= min_samples) and (rate >= min_success_rate)

# ──────────────────────────────────────────────────────────────
# 텔레그램 메시지 포맷
# ──────────────────────────────────────────────────────────────
def format_message(data):
    def safe_float(value, default=0.0):
        try:
            if value is None or (isinstance(value, str) and not str(value).strip()):
                return default
            val = float(value)
            return val if not math.isnan(val) else default
        except:
            return default

    price = safe_float(data.get("price"), 0.0)
    direction = data.get("direction", "롱")
    strategy = data.get("strategy", "전략")
    symbol = data.get("symbol", "종목")
    success_rate = safe_float(data.get("success_rate"), 0.0)   # 메타 성공률만 반영
    rate = safe_float(data.get("rate"), 0.0)  # expected return (예: 0.125)
    reason = str(data.get("reason", "-")).strip()
    score = data.get("score", None)
    volatility = str(data.get("volatility", "False")).lower() in ["1", "true", "yes"]

    target = price * (1 + rate) if direction == "롱" else price * (1 - rate)
    stop_loss = price * (1 - 0.02) if direction == "롱" else price * (1 + 0.02)

    rate_pct = abs(rate) * 100
    success_rate_pct = success_rate * 100
    dir_str = "상승" if direction == "롱" else "하락"
    vol_tag = "⚡ " if volatility else ""

    message = (
        f"{vol_tag}{'📈' if direction == '롱' else '📉'} "
        f"[{strategy} 전략] {symbol} {direction} 추천\n"
        f"🎯 예상 수익률: {rate_pct:.2f}% ({dir_str} 예상)\n"
        f"💰 진입가: {price:.4f} USDT\n"
        f"🎯 목표가: {target:.4f} USDT\n"
        f"🛡 손절가: {stop_loss:.4f} USDT (-2.00%)\n\n"
        f"📊 최근 전략 성공률(메타): {success_rate_pct:.2f}%"
    )

    if isinstance(score, (float, int)) and not math.isnan(score):
        message += f"\n🏆 스코어: {score:.5f}"

    message += f"\n💡 추천 사유: {reason}\n\n🕒 (기준시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')} KST)"
    return message

# ──────────────────────────────────────────────────────────────
# 감사 로그 기록(원래 그대로 유지, 경로만 바뀜)
# ──────────────────────────────────────────────────────────────
def log_audit(symbol, strategy, result, status):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "strategy", "result", "status"])
            if f.tell() == 0:
                w.writeheader()
            w.writerow({
                "timestamp": now_kst().isoformat(),
                "symbol": symbol or "UNKNOWN",
                "strategy": strategy or "알수없음",
                "result": str(result),
                "status": status
            })
    except Exception as e:
        print(f"[log_audit 오류] {e}")

# ──────────────────────────────────────────────────────────────
# 실패 카운트 로드/저장 (원본 유지, 경로만 바뀜)
# ──────────────────────────────────────────────────────────────
def load_failure_count():
    if not os.path.exists(FAILURE_LOG):
        return {}
    try:
        with open(FAILURE_LOG, "r", encoding="utf-8-sig") as f:
            return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}
    except Exception as e:
        print(f"[load_failure_count 오류] {e}")
        return {}

def save_failure_count(fmap):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(FAILURE_LOG, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "strategy", "failures"])
            w.writeheader()
            for k, v in fmap.items():
                if "-" not in k:
                    continue
                s, strat = k.split("-", 1)
                w.writerow({"symbol": s, "strategy": strat, "failures": v})
    except Exception as e:
        print(f"[save_failure_count 오류] {e}")

# ──────────────────────────────────────────────────────────────
# 변동성 높은 심볼 추출 (최근 N행만 사용해 메모리/연산 절약)
# ──────────────────────────────────────────────────────────────
def _normalize_kline_result(k):
    """
    get_kline_by_strategy의 반환이 DataFrame 또는 tuple/list일 수 있음.
    DataFrame이면 그대로, tuple/list이면 첫 요소가 DataFrame인 경우 이를 반환.
    """
    if k is None:
        return None
    if isinstance(k, pd.DataFrame):
        return k
    if isinstance(k, (list, tuple)) and len(k) > 0:
        cand = k[0]
        if isinstance(cand, pd.DataFrame):
            return cand
    # 못 읽으면 None
    return None

def get_symbols_by_volatility(strategy):
    th = STRATEGY_VOL.get(strategy, VOL_RT_단기)
    result = []
    for symbol in SYMBOLS:
        try:
            raw = get_kline_by_strategy(symbol, strategy)
            df = _normalize_kline_result(raw)
            if df is None or len(df) < 60:
                continue
            # 최근 구간만 사용해 계산량 축소
            if VOL_LOOKBACK_MAX > 0 and len(df) > VOL_LOOKBACK_MAX:
                df = df.tail(VOL_LOOKBACK_MAX)
            # 안전: close 컬럼 존재 확인
            if "close" not in df.columns:
                continue
            r_std = df["close"].pct_change().rolling(20).std().iloc[-1]
            b_std = df["close"].pct_change().rolling(60).std().iloc[-1] if len(df) >= 60 else r_std
            if pd.isna(r_std):
                continue
            if r_std >= th and (r_std / (b_std + 1e-8)) >= 1.2:
                result.append({"symbol": symbol, "volatility": float(r_std)})
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return sorted(result, key=lambda x: -x["volatility"])

# ──────────────────────────────────────────────────────────────
# 내부 유틸: 최신 종가(진입가) 조회
# ──────────────────────────────────────────────────────────────
def _get_latest_price(symbol, strategy):
    try:
        raw = get_kline_by_strategy(symbol, strategy)
        df = _normalize_kline_result(raw)
        if df is None or len(df) == 0 or "close" not in df.columns:
            return 0.0
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0

# ──────────────────────────────────────────────────────────────
# ✅ 핵심 수정: 보류/실패 결과를 "0으로 채워서 추천"하지 않도록 차단
# ──────────────────────────────────────────────────────────────
def _is_actionable_prediction(res: dict) -> (bool, str):
    """
    ✅ 추천으로 보낼만한 '정상 예측'만 통과
    - expected_return 없거나 None이면: 보류/실패로 보고 제외
    - class 없으면 제외
    - reason/source에 보류/abstain/failed 류가 있으면 제외
    """
    if not isinstance(res, dict):
        return False, "not_dict"

    r_reason = str(res.get("reason", "") or "").strip().lower()
    r_source = str(res.get("source", "") or "").strip().lower()

    # 보류/실패 느낌 키워드 (predict.py의 soft_abstain/failed_result 계열 방어)
    bad_kw = ["abstain", "보류", "hold", "failed", "fail", "error", "locked", "closed", "timeout"]
    if any(k in r_reason for k in bad_kw) or any(k in r_source for k in bad_kw):
        return False, f"non_actionable_reason:{res.get('reason')}"

    if "expected_return" not in res or res.get("expected_return") is None:
        return False, "missing_expected_return"

    if "class" not in res or res.get("class") is None:
        return False, "missing_class"

    try:
        er = float(res.get("expected_return"))
        if math.isnan(er) or math.isinf(er):
            return False, "invalid_expected_return"
    except Exception:
        return False, "invalid_expected_return_cast"

    return True, "ok"

# ──────────────────────────────────────────────────────────────
# (신규) 모델 파일 인벤토리 캐시 — 하위 디렉토리까지 일괄 수집(일관화)
# ──────────────────────────────────────────────────────────────
def _build_model_index():
    """
    MODEL_DIR
      ├─ <files...>
      └─ <SYMBOL>/<STRATEGY>/<files...>
    를 모두 스캔하여 상대경로 세트로 반환.
    """
    model_dir = MODEL_DIR
    idx = set()
    try:
        if not os.path.isdir(model_dir):
            return idx
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".pt", ".ptz", ".safetensors", ".meta.json")):
                    rel = os.path.relpath(os.path.join(root, f), model_dir)
                    # 통일된 구분자 사용
                    idx.add(rel.replace("\\", "/"))
    except Exception as e:
        print(f"[warn] model index build failed: {e}")
    return idx

def _has_model_for(model_index, symbol, strategy):
    """
    1) 루트: {symbol}_{strategy}_*.{pt,ptz,safetensors}
    2) 트리:  {symbol}/{strategy}/*.{pt,ptz,safetensors}
    """
    pref_root = f"{symbol}_{strategy}_"
    pref_tree = f"{symbol}/{strategy}/"
    for p in model_index:
        if p.startswith(pref_root) and (p.endswith(".pt") or p.endswith(".ptz") or p.endswith(".safetensors")):
            return True
        if p.startswith(pref_tree) and (p.endswith(".pt") or p.endswith(".ptz") or p.endswith(".safetensors")):
            return True
    return False

# ──────────────────────────────────────────────────────────────
# (신규) 단일 심볼 예측 엔트리 — predict_trigger에서 사용
# ──────────────────────────────────────────────────────────────
def run_prediction(symbol, strategy, source="변동성", allow_send=True, _model_index=None):
    # 로그 파일 보장
    try:
        ensure_prediction_log_exists()
    except Exception as e:
        print(f"[경고] prediction_log 보장 실패: {e}")

    # 모델 존재 대략 체크(캐시 사용)
    try:
        model_index = _model_index if _model_index is not None else _build_model_index()
        if not _has_model_for(model_index, symbol, strategy):
            log_audit(symbol, strategy, None, "모델 없음")
            return None
    except Exception as e:
        print(f"[경고] 모델 체크 실패: {e}")

    try:
        res = predict(symbol, strategy, source=source)
        if isinstance(res, list):
            res = res[0] if res else None
        if not isinstance(res, dict):
            log_audit(symbol, strategy, None, "predict() 결과 없음/형식오류")
            return None

        ok, why = _is_actionable_prediction(res)
        if not ok:
            log_audit(symbol, strategy, res, f"추천 제외({why})")
            return None

        # 메시지용 필드 보강 — 메타 성공률만
        meta_rate = float(get_meta_success_rate(strategy, min_samples=MIN_SAMPLES) or 0.0)
        expected_ret = float(res.get("expected_return"))
        entry_price = _get_latest_price(symbol, strategy)
        direction = "롱" if expected_ret >= 0 else "숏"

        enriched = dict(res)
        enriched.update({
            "symbol": symbol,
            "strategy": strategy,
            "price": entry_price,
            "rate": expected_ret,
            "direction": direction,
            "success_rate": meta_rate,    # 0~1 (메타만)
            "volatility": True,           # 트리거 기반 호출이므로 신호 강조
        })

        # 필터 통과 시 텔레그램 (성공률/표본수 기준)
        if allow_send and check_prediction_filter(strategy):
            try:
                send_message(format_message(enriched))
            except Exception as e:
                print(f"[텔레그램 전송 실패] {e}")
        else:
            print(f"[알림 생략] {symbol}-{strategy} (필터 미통과 또는 전송 비활성)")

        return enriched

    except Exception as e:
        print(f"[ERROR] {symbol}-{strategy} run_prediction 실패: {e}")
        traceback.print_exc()
        log_audit(symbol, strategy, None, f"예측실패:{e}")
        return None

# ──────────────────────────────────────────────────────────────
# 예측 실행 루프 — predict() dict 기준 + 중복 log 제거
# ──────────────────────────────────────────────────────────────
def run_prediction_loop(strategy, symbols, source="일반", allow_prediction=True):
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼")
    results, fmap = [], load_failure_count()

    # 예측/로그 파일 보장
    try:
        ensure_prediction_log_exists()
    except Exception as e:
        print(f"[경고] prediction_log 보장 실패: {e}")

    # 모델 인벤토리 캐시 1회 생성 (하위 트리 포함)
    model_index = _build_model_index()

    for item in symbols:
        symbol = item.get("symbol") if isinstance(item, dict) else (item if isinstance(item, str) else None)
        if symbol is None:
            continue
        vol_val = float(item.get("volatility", 0.0)) if isinstance(item, dict) else 0.0

        if not allow_prediction:
            log_audit(symbol, strategy, "예측 생략", f"예측 차단됨 (source={source})")
            continue

        try:
            # 모델 존재 여부 캐시로 판정
            if not _has_model_for(model_index, symbol, strategy):
                log_audit(symbol, strategy, None, "모델 없음")
                continue

            # predict() 실행 (dict 기준 수용)
            res = predict(symbol, strategy, source=source)
            if isinstance(res, list):
                res = res[0] if res else None
            if not isinstance(res, dict):
                log_audit(symbol, strategy, None, "predict() 결과 없음/형식오류")
                continue

            ok, why = _is_actionable_prediction(res)
            if not ok:
                log_audit(symbol, strategy, res, f"추천 제외({why})")
                continue

            # 텔레그램 메시지용 필드 보강 — 메타 성공률만
            meta_rate = float(get_meta_success_rate(strategy, min_samples=MIN_SAMPLES) or 0.0)
            expected_ret = float(res.get("expected_return"))
            entry_price = _get_latest_price(symbol, strategy)
            direction = "롱" if expected_ret >= 0 else "숏"

            enriched = dict(res)
            enriched.update({
                "symbol": symbol,
                "strategy": strategy,
                "price": entry_price,
                "rate": expected_ret,
                "direction": direction,
                "success_rate": meta_rate,   # 0~1 (메타만)
                "volatility": (vol_val > 0),
            })

            results.append(enriched)
            fmap[f"{symbol}-{strategy}"] = 0  # 실패 카운터 리셋

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            traceback.print_exc()

    save_failure_count(fmap)
    return results

# ──────────────────────────────────────────────────────────────
# 메인 엔트리 — 배치 예측
# ──────────────────────────────────────────────────────────────
def main(strategy, symbols=None, force=False, allow_prediction=True):
    print(f"\n📋 [예측 시작] 전략: {strategy} | 시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    target_symbols = symbols if symbols is not None else get_symbols_by_volatility(strategy)
    if not target_symbols:
        print(f"[INFO] {strategy} 대상 심볼이 없습니다")
        return

    # 🔐 app.py 외 단독 실행 대비: 예측 구간을 게이트로 감싼다
    open_predict_gate(note=f"recommend_main_{strategy}")
    try:
        results = run_prediction_loop(strategy, target_symbols, source="배치", allow_prediction=allow_prediction)

        # 필터 통과했을 때만 텔레그램 발송 (성공률 65% + 최소 10회 평가 완료)
        if allow_prediction and check_prediction_filter(strategy):
            for r in results:
                try:
                    send_message(format_message(r))
                except Exception as e:
                    print(f"[텔레그램 전송 실패] {e}")
        else:
            print(f"[알림 생략] allow_prediction={allow_prediction} 또는 필터 미통과")
    finally:
        close_predict_gate(note=f"recommend_main_{strategy}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="단기", choices=["단기", "중기", "장기"])
    parser.add_argument("--allow_prediction", action="store_true", default=True)
    args = parser.parse_args()
    try:
        main(args.strategy, allow_prediction=args.allow_prediction)
    except Exception as e:
        print(f"[❌ 예측 실패] {e}")
        traceback.print_exc()
