# === visualization.py — YOPO 학습로그 카드 UI (심볼×전략 1카드 출력) ===
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pytz
from matplotlib import font_manager

# ----------------------------------------
# 폰트 설정 (한글 + 이모지)
# ----------------------------------------
font_paths = [
    os.path.join("fonts", "NanumGothic-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
]
valid_fonts = []
for fp in font_paths:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        valid_fonts.append(font_manager.FontProperties(fname=fp).get_name())
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = valid_fonts or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------
# 기본 경로
# ----------------------------------------
TRAIN_LOG = "/persistent/train_log.csv"


# ----------------------------------------
# 유틸: HTML 카드 wrapper
# ----------------------------------------
def _wrap_card(html_inner):
    return f"""
<div style='width:98%;padding:18px;margin:12px;border:1px solid #ccc;border-radius:12px;
            background:#fafafa;font-size:15px;line-height:1.5'>
{html_inner}
</div>
"""


# ----------------------------------------
# MAIN: 심볼 + 전략 기반 카드 생성
# ----------------------------------------
def generate_card(symbol: str, strategy: str, df):
    """
    df = train_log 전체
    symbol & strategy 에 해당하는 마지막 학습 기록 1개 뽑아서 카드 생성
    """
    d = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
    if d.empty:
        return _wrap_card(f"<h3>❌ {symbol}-{strategy} 기록 없음</h3>")

    row = d.sort_values("timestamp").iloc[-1]

    # ----------------------------------------
    # ① 건강 상태
    # ----------------------------------------
    status = row.get("status", "")
    model_file = row.get("model", "")
    tstamp = row.get("timestamp", "")

    health = "🔴 오류"
    if status == "success":
        health = "🟢 정상"
    elif status in ("warn", "warning", "info"):
        health = "🟡 경고"

    part1 = f"""
<h2>📦 {symbol} · {strategy}</h2>
<b>건강 상태:</b> {health} (status={status})<br>
<b>모델:</b> {model_file}<br>
<b>마지막 학습:</b> {tstamp}<br>
"""

    # ----------------------------------------
    # ② 전체 성능
    # ----------------------------------------
    def _safe_float(v, default=0.0):
        try:
            if v is None or pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    acc = _safe_float(row.get("val_acc"), 0.0)
    f1 = _safe_float(row.get("val_f1"), 0.0)
    loss = _safe_float(row.get("val_loss"), 0.0)

    if f1 >= 0.6:
        comment = "👍 잘 맞추고 있어요."
    elif f1 >= 0.4:
        comment = "🙂 중간 정도 성능이에요."
    else:
        comment = "⚠️ 개선이 필요해요."

    part2 = f"""
<h3>🎯 전체 학습 성능</h3>
정확도: <b>{acc:.4f}</b><br>
F1 점수: <b>{f1:.4f}</b><br>
Loss: <b>{loss:.4f}</b><br>
👉 {comment}
"""

    # ----------------------------------------
    # ③ 데이터 / 라벨 상태
    #    (train.py 에서 기록한 usable_samples, masked_count, near_zero_count, near_zero_band 사용)
    # ----------------------------------------
    def _safe_int(v, default=0):
        try:
            if v is None or pd.isna(v):
                return default
            return int(v)
        except Exception:
            return default

    total = _safe_int(row.get("usable_samples") if "usable_samples" in row.index else None, 0)
    if total == 0:
        # usable_samples 가 없으면 class_counts 합으로 대체
        cc_raw = row.get("class_counts")
        if isinstance(cc_raw, str):
            try:
                cc_raw = json.loads(cc_raw)
            except Exception:
                cc_raw = []
        if cc_raw is None or (isinstance(cc_raw, float) and pd.isna(cc_raw)):
            cc_raw = []
        if isinstance(cc_raw, list):
            total = sum(_safe_int(x, 0) for x in cc_raw)

    masked = _safe_int(
        row.get("masked_count") if "masked_count" in row.index else row.get("label_masked"),
        0,
    )
    near_zero = _safe_int(
        row.get("near_zero_count") if "near_zero_count" in row.index else row.get("near_zero"),
        0,
    )
    nz_band = row.get("near_zero_band")
    if nz_band is None or (isinstance(nz_band, float) and pd.isna(nz_band)):
        nz_band = ""

    part3 = f"""
<h3>🧪 데이터 / 라벨 상태</h3>
전체 라벨(유효 샘플): {total}개<br>
마스킹 라벨: {masked}개<br>
0% 근처(near-zero): {near_zero}개 (구간: {nz_band})<br>
"""

    # ----------------------------------------
    # ④ 클래스 요약
    # ----------------------------------------
    class_ranges = row.get("class_ranges") or []
    if isinstance(class_ranges, str):
        try:
            class_ranges = json.loads(class_ranges)
        except Exception:
            class_ranges = []
    if class_ranges is None or (isinstance(class_ranges, float) and pd.isna(class_ranges)):
        class_ranges = []
    num_classes = len(class_ranges)

    class_counts = row.get("class_counts") or []
    if isinstance(class_counts, str):
        try:
            class_counts = json.loads(class_counts)
        except Exception:
            class_counts = []
    if class_counts is None or (isinstance(class_counts, float) and pd.isna(class_counts)):
        class_counts = []
    # 길이 맞추기
    if num_classes and len(class_counts) < num_classes:
        class_counts = class_counts + [0] * (num_classes - len(class_counts))

    participated = sum(1 for v in class_counts if v > 0)
    skipped = max(0, num_classes - participated)

    part4 = f"""
<h3>📊 클래스 요약</h3>
클래스 개수: {num_classes}개<br>
참여 클래스: {participated}/{num_classes}<br>
빠진 클래스: {skipped}개<br>
"""

    # ----------------------------------------
    # ⑤ per-class F1 + 상세 표
    #    (train.py 에서 per_class_f1 리스트를 그대로 기록해둔 컬럼 사용)
    # ----------------------------------------
    per_class_f1 = row.get("per_class_f1") or []
    if isinstance(per_class_f1, str):
        try:
            per_class_f1 = json.loads(per_class_f1)
        except Exception:
            per_class_f1 = []
    if per_class_f1 is None or (isinstance(per_class_f1, float) and pd.isna(per_class_f1)):
        per_class_f1 = []
    if num_classes and len(per_class_f1) < num_classes:
        per_class_f1 = per_class_f1 + [0.0] * (num_classes - len(per_class_f1))

    table_rows = ""
    for i in range(num_classes):
        try:
            lo, hi = class_ranges[i]
        except Exception:
            lo, hi = 0.0, 0.0
        cnt = class_counts[i] if i < len(class_counts) else 0
        participate = "✅" if cnt > 0 else "❌"

        f1_c = 0.0
        if i < len(per_class_f1):
            try:
                f1_c = float(per_class_f1[i])
            except Exception:
                f1_c = 0.0

        if cnt == 0:
            memo = "학습에 참여하지 않음"
        elif cnt <= 3:
            memo = "샘플 매우 적음"
        elif f1_c < 0.2:
            memo = "예측력이 매우 낮음"
        else:
            memo = ""

        table_rows += f"""
<tr>
<td>{i}</td>
<td>{lo:.4f} ~ {hi:.4f}</td>
<td>{cnt}</td>
<td>{participate}</td>
<td>{f1_c:.2f}</td>
<td>{memo}</td>
</tr>
"""

    part5 = f"""
<h3>📘 클래스 상세</h3>
<table border="1" cellspacing="0" cellpadding="4" style="width:98%;font-size:13px">
<tr>
<th>클래스</th>
<th>수익률 구간</th>
<th>데이터 수</th>
<th>참여 여부</th>
<th>F1</th>
<th>메모</th>
</tr>
{table_rows}
</table>
"""

    # ----------------------------------------
    # ⑥ 이상 징후 요약
    # ----------------------------------------
    alerts = []

    if class_counts:
        # 한 구간에 25% 이상 몰리면 경고
        if total > 0 and max(class_counts) > total * 0.25:
            alerts.append("특정 구간에 데이터가 많이 몰려 있음")

        non_zero = [c for c in class_counts if c > 0]
        if non_zero and min(non_zero) <= 2:
            alerts.append("샘플 매우 적은 클래스 존재(신뢰 낮음)")

    if not alerts:
        alerts.append("큰 이상 없음")

    part6 = f"""
<h3>⚠️ 이상 징후 요약</h3>
{"<br>".join(alerts)}
"""

    # === 완성 ===
    full = part1 + part2 + part3 + part4 + part5 + part6
    return _wrap_card(full)


# ----------------------------------------
# 전체 리포트: 심볼×전략 전체 카드 출력
# ----------------------------------------
def generate_visual_report():
    try:
        df = pd.read_csv(TRAIN_LOG)
    except Exception:
        return "<h3>train_log.csv 없음</h3>"

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    html = "<h1>📘 YOPO 학습 리포트</h1>"

    symbols = df["symbol"].dropna().unique().tolist()
    strategies = ["단기", "중기", "장기"]

    for s in symbols:
        for strat in strategies:
            html += generate_card(s, strat, df)

    return html

def generate_visuals_for_strategy(symbol: str, strategy: str) -> str:
    """
    app.py 에서 사용하는 단일 심볼·전략용 리포트.
    - train_log.csv 전체를 읽고
    - 해당 symbol, strategy 의 마지막 기록 1개를 카드로 보여준다.
    """
    try:
        df = pd.read_csv(TRAIN_LOG)
    except Exception:
        return "<h3>train_log.csv 없음</h3>"

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    html = f"<h1>📘 {symbol} · {strategy} 학습 리포트</h1>"
    html += generate_card(symbol, strategy, df)
    return html

