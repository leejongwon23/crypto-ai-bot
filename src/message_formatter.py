import datetime
import pytz
import math

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

def _safe_float(v, default=0.0):
    try:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            return default
        x = float(v)
        return x if not math.isnan(x) else default
    except:
        return default

def format_message(data):
    # 기본 필드
    price      = _safe_float(data.get("price"), 0.0)
    direction  = data.get("direction", "롱")  # "롱" or "숏"
    strategy   = data.get("strategy", "전략")
    symbol     = data.get("symbol", "종목")
    reason     = str(data.get("reason", "-")).strip()
    score      = data.get("score", None)
    volatility = str(data.get("volatility", "False")).lower() in ["1","true","yes"]

    # 클래스 범위(신규) + 중앙값 계산
    # 우선순위: rate_min/rate_max → class_low/class_high → rate(단일)
    rmin = _safe_float(data.get("rate_min", data.get("class_low")), None)
    rmax = _safe_float(data.get("rate_max", data.get("class_high")), None)
    rmid = None
    if rmin is None and rmax is None:
        r = _safe_float(data.get("rate"), 0.0)   # 구버전 호환(단일 중앙값)
        rmin = rmax = r
    elif rmin is None:
        rmin = rmax
    elif rmax is None:
        rmax = rmin
    rmid = (rmin + rmax) / 2.0

    # 성공률: 횟수 기반(신규) → 비율(fallback)
    succ_n = data.get("success_successes")
    total_n = data.get("success_total")
    if isinstance(succ_n, str) and succ_n.isdigit(): succ_n = int(succ_n)
    if isinstance(total_n, str) and total_n.isdigit(): total_n = int(total_n)

    if isinstance(succ_n, int) and isinstance(total_n, int) and total_n > 0 and 0 <= succ_n <= total_n:
        succ_pct = (succ_n / total_n) * 100.0
        succ_text = f"{succ_n}/{total_n} ({succ_pct:.2f}%)"
    else:
        # 기존 success_rate(0.0~1.0) 폴백
        sr = _safe_float(data.get("success_rate"), 0.0)
        succ_text = f"{sr*100:.2f}%"

    # 목표가: 범위 표시(최소~최대)
    if direction == "롱":
        tgt_min = price * (1 + min(rmin, rmax))
        tgt_max = price * (1 + max(rmin, rmax))
        stop_loss = price * (1 - 0.02)
        dir_str = "상승"
        arrow = "📈"
    else:  # "숏"
        # 숏은 하락 목표: 더 크게 떨어진 값이 '최대 수익'이므로 (1 - rmax) 가 더 낮음
        lo = min(rmin, rmax); hi = max(rmin, rmax)
        tgt_min = price * (1 - lo)  # 최소 기대 하락
        tgt_max = price * (1 - hi)  # 최대 기대 하락(더 낮은 가격)
        stop_loss = price * (1 + 0.02)
        dir_str = "하락"
        arrow = "📉"

    vol_tag = "⚡ " if volatility else ""

    # 메시지 구성
    msg = (
        f"{vol_tag}{arrow} [{strategy} 전략] {symbol} {direction} 추천\n"
        f"🎯 예상 수익률 범위: {rmin*100:.2f}% ~ {rmax*100:.2f}% (중앙값 {rmid*100:.2f}%)\n"
        f"💰 진입가: {price:.4f} USDT\n"
        f"🎯 목표가(범위): {tgt_min:.4f} ~ {tgt_max:.4f} USDT\n"
        f"🛡 손절가: {stop_loss:.4f} USDT (-2.00%)\n\n"
        f"📊 신호 방향: {arrow} {dir_str}\n"
        f"✅ 최근 전략 성과: {succ_text}"
    )
    if isinstance(score, (float, int)) and not math.isnan(score):
        msg += f"\n🏆 스코어: {float(score):.5f}"
    msg += f"\n💡 추천 사유: {reason}\n\n🕒 (기준시각: {now_kst()} KST)"
    return msg
