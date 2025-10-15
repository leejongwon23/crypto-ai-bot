import datetime
import pytz
import math

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

def _is_nan_str(v):
    return isinstance(v, str) and v.strip().lower() in ["nan", "none", "null", ""]

def _safe_float(v, default=0.0, allow_none=False):
    try:
        if v is None or _is_nan_str(v):
            return None if allow_none else default
        x = float(v)
        if math.isnan(x):
            return None if allow_none else default
        return x
    except:
        return None if allow_none else default

def _pick_prob(d):
    cp = d.get("calib_prob")
    rp = d.get("raw_prob")
    p = _safe_float(cp, allow_none=True)
    if p is not None:
        return p, "calib"
    p = _safe_float(rp, allow_none=True)
    if p is not None:
        return p, "raw"
    return None, None

def _norm_direction_from_position(pos, fallback="롱"):
    s = (pos or "").strip().lower()
    if s in ["long", "롱", "buy", "상승"]:
        return "롱"
    if s in ["short", "숏", "sell", "하락"]:
        return "숏"
    return fallback

def _rate_range_and_mid(data):
    # 우선순위: (rg_lo, rg_hi) 표시는 따로. 계산은 rate_min/rate_max → class_low/class_high → rate 단일.
    rmin = _safe_float(data.get("rate_min", data.get("class_low")), allow_none=True)
    rmax = _safe_float(data.get("rate_max", data.get("class_high")), allow_none=True)
    if rmin is None and rmax is None:
        r = _safe_float(data.get("rate"), 0.0)
        rmin = rmax = r
    elif rmin is None:
        rmin = rmax
    elif rmax is None:
        rmax = rmin
    rmid = (rmin + rmax) / 2.0 if (rmin is not None and rmax is not None) else _safe_float(data.get("rate"), 0.0)
    return rmin, rmax, rmid

def _success_text(data):
    succ_n = data.get("success_successes")
    total_n = data.get("success_total")
    try:
        if isinstance(succ_n, str) and succ_n.isdigit(): succ_n = int(succ_n)
        if isinstance(total_n, str) and total_n.isdigit(): total_n = int(total_n)
    except:
        succ_n, total_n = None, None
    if isinstance(succ_n, int) and isinstance(total_n, int) and total_n > 0 and 0 <= succ_n <= total_n:
        succ_pct = (succ_n / total_n) * 100.0
        return f"{succ_n}/{total_n} ({succ_pct:.2f}%)"
    sr = _safe_float(data.get("success_rate"), 0.0)
    return f"{sr*100:.2f}%"

def format_message(data):
    # 기본 메타
    symbol     = data.get("symbol", "종목")
    strategy   = data.get("strategy", "전략")
    reason     = str(data.get("reason", "-")).strip()
    volatility = str(data.get("volatility", "False")).lower() in ["1","true","yes"]
    vol_tag = "⚡ " if volatility else ""

    # 포지션/방향
    position   = data.get("position")
    direction  = _norm_direction_from_position(position, fallback=data.get("direction", "롱"))

    # 가격
    entry_price = _safe_float(data.get("entry_price"), allow_none=True)
    price = _safe_float(data.get("price"), allow_none=True)
    base_price = entry_price if entry_price is not None else (price if price is not None else 0.0)

    # 클래스 및 구간 텍스트
    pred_class = data.get("predicted_class")
    try:
        if pred_class is not None and str(pred_class).isdigit():
            pred_class = str(int(pred_class))
        else:
            pred_class = "-" if pred_class is None else str(pred_class)
    except:
        pred_class = str(pred_class) if pred_class is not None else "-"

    class_text = data.get("class_return_text")
    if _is_nan_str(class_text):
        class_text = None

    # RG 범위
    rg_mu = _safe_float(data.get("rg_mu"), allow_none=True)
    rg_lo = _safe_float(data.get("rg_lo"), allow_none=True)
    rg_hi = _safe_float(data.get("rg_hi"), allow_none=True)
    if rg_lo is not None and rg_hi is not None and rg_mu is not None:
        rg_text = f"{rg_lo*100:.2f}% ~ {rg_hi*100:.2f}% (μ {rg_mu*100:.2f}%)"
    elif rg_lo is not None and rg_hi is not None:
        rg_text = f"{rg_lo*100:.2f}% ~ {rg_hi*100:.2f}%"
    elif rg_mu is not None:
        rg_text = f"μ {rg_mu*100:.2f}%"
    else:
        rg_text = None

    # 예상 수익률 범위
    rmin, rmax, rmid = _rate_range_and_mid(data)

    # 확률
    prob, prob_src = _pick_prob(data)
    prob_text = f"{prob*100:.2f}% ({prob_src})" if prob is not None else "-"

    # 메타 선택(있으면)
    meta_choice = data.get("meta_choice")
    if _is_nan_str(meta_choice):
        meta_choice = None

    # 목표가 범위 계산
    if direction == "롱":
        tgt_min = base_price * (1 + min(rmin, rmax))
        tgt_max = base_price * (1 + max(rmin, rmax))
        stop_loss = base_price * (1 - 0.02)
        arrow = "📈"; dir_str = "상승"
    else:
        lo = min(rmin, rmax); hi = max(rmin, rmax)
        tgt_min = base_price * (1 - lo)   # 최소 기대 하락
        tgt_max = base_price * (1 - hi)   # 최대 기대 하락
        stop_loss = base_price * (1 + 0.02)
        arrow = "📉"; dir_str = "하락"

    # 성공률 텍스트
    succ_text = _success_text(data)

    # 메시지
    lines = []
    lines.append(f"{vol_tag}{arrow} [{strategy} 전략] {symbol} {direction} 추천")
    if meta_choice:
        lines.append(f"🧠 메타 선택: {meta_choice}")
    lines.append(f"🎯 선택 클래스: {pred_class}")
    if class_text:
        lines.append(f"📦 클래스 구간: {class_text}")
    if rg_text:
        lines.append(f"🧭 RealityGuard: {rg_text}")
    lines.append(f"📈 예상 수익률: {rmin*100:.2f}% ~ {rmax*100:.2f}% (중앙값 {rmid*100:.2f}%)")
    lines.append(f"🔒 신호 확률: {prob_text}")
    lines.append(f"💰 진입가: {base_price:.4f} USDT")
    lines.append(f"🎯 목표가(범위): {tgt_min:.4f} ~ {tgt_max:.4f} USDT")
    lines.append(f"🛡 손절가: {stop_loss:.4f} USDT (-2.00%)")
    lines.append(f"📊 신호 방향: {arrow} {dir_str}")
    lines.append(f"✅ 최근 전략 성과: {succ_text}")
    score = data.get("score", None)
    if isinstance(score, (float, int)) and not math.isnan(float(score)):
        lines.append(f"🏆 스코어: {float(score):.5f}")
    lines.append(f"💡 추천 사유: {reason}")
    lines.append(f"🕒 기준시각: {now_kst()} KST")
    return "\n".join(lines)
