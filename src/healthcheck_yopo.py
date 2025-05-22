import os
import datetime
import pytz
import csv
import pandas as pd
from logger import get_min_gain
from model_weight_loader import model_exists
from data.utils import SYMBOLS
from predict import predict
from train import LOG_DIR

PRED_LOG = "/persistent/prediction_log.csv"
LAST_TRAIN_LOG = os.path.join(LOG_DIR, "train_log.csv")
STRATEGIES = ["단기", "중기", "장기"]
KST = pytz.timezone("Asia/Seoul")

def now_kst():
    return datetime.datetime.now(KST)

def parse_prediction_log():
    if not os.path.exists(PRED_LOG):
        return []
    try:
        df = pd.read_csv(PRED_LOG, encoding="utf-8-sig")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except:
        return []

def format_trend(conf_series):
    if len(conf_series) < 9:
        return "데이터 부족"
    avg_conf = conf_series.tail(3).mean()
    prev_conf = conf_series.tail(6).head(3).mean()
    pre_prev_conf = conf_series.tail(9).head(3).mean()
    trend = f"{pre_prev_conf:.2f} → {prev_conf:.2f} → {avg_conf:.2f}"
    if avg_conf < prev_conf and prev_conf < pre_prev_conf:
        return trend + " ⚠️ 하락 추세"
    elif avg_conf < prev_conf:
        return trend + " ⚠️ 감소 조짐"
    else:
        return trend + " ✅ 안정적"

def check_volatility_trigger_recent(df):
    try:
        recent = df[df["reason"].str.contains("반전|트리거", na=False)]
        if recent.empty:
            return "❌ 최근 트리거 예측 없음"
        recent_time = recent["timestamp"].max().tz_localize(None)
        elapsed = (now_kst() - recent_time).total_seconds() / 60
        if elapsed <= 120:
            return f"✅ 최근 트리거 예측 작동 (약 {int(elapsed)}분 전)"
        else:
            return f"⚠️ 트리거 예측 작동 이력 있지만 지연됨 (약 {int(elapsed)}분 전)"
    except:
        return "❌ 트리거 상태 확인 실패"

def generate_health_report():
    df = parse_prediction_log()
    if isinstance(df, list): return "❌ 예측 로그 없음"

    report_lines = ["========================= YOPO 상태 진단 (KST 기준) ========================="]

    for strategy in STRATEGIES:
        s_df = df[df["strategy"] == strategy]
        pred_df = s_df[s_df["status"].isin(["success", "fail", "pending", "failed"])]
        total = len(pred_df)
        success = len(pred_df[pred_df["status"] == "success"])
        fail = len(pred_df[pred_df["status"] == "fail"])
        pending = len(pred_df[pred_df["status"] == "pending"])
        failed = len(pred_df[pred_df["status"] == "failed"])

        avg_rate = round(pred_df["rate"].mean() * 100, 2) if not pred_df.empty else 0.0
        success_rate = round(success / total * 100, 1) if total else 0.0
        fail_rate = round(fail / total * 100, 1) if total else 0.0
        pending_rate = round(pending / total * 100, 1) if total else 0.0
        conf_trend = format_trend(pred_df["confidence"]) if not pred_df.empty else "데이터 부족"

        recent_pred_time = (
            s_df["timestamp"].max().astimezone(KST).strftime("%Y-%m-%d %H:%M")
            if not s_df.empty else "없음"
        )
        model_count = sum(1 for s in SYMBOLS if model_exists(s, strategy))

        train_time = "-"
        if os.path.exists(LAST_TRAIN_LOG):
            try:
                tdf = pd.read_csv(LAST_TRAIN_LOG, encoding="utf-8-sig")
                tdf = tdf[tdf["strategy"] == strategy]
                if not tdf.empty:
                    train_time = pd.to_datetime(tdf["timestamp"].max()).astimezone(KST).strftime("%Y-%m-%d %H:%M")
            except:
                pass

        summary = (
            "⚠️ 신뢰도 감소, 예측 안정성 점검 필요" if "하락" in conf_trend
            else "⚠️ 예측 지연 또는 없음" if recent_pred_time == "없음"
            else "✅ 전반적으로 안정"
        )

        report_lines += [
            f"\n📌 {strategy} 전략",
            f"- 모델 수             : {model_count}개",
            f"- 최근 예측 시각       : {recent_pred_time} {'✅ 정상 작동' if recent_pred_time != '없음' else '⚠️ 지연됨'}",
            f"- 최근 학습 시각       : {train_time} ✅ 정상 작동",
            f"- 최근 예측 건수       : {total}건 (성공: {success} / 실패: {fail} / 대기중: {pending} / 실패예측: {failed})",
            f"- 평균 수익률         : {avg_rate:.2f}%",
            f"- 평균 신뢰도         : {conf_trend}",
            f"- 성공률              : {success_rate:.1f}%",
            f"- 실패률              : {fail_rate:.1f}%",
            f"- 예측 대기 비율       : {pending_rate:.1f}%",
            f"- 재학습 상태         : 자동 트리거 정상 작동",
            f"- 상태 요약           : {summary}"
        ]

    # ✅ 트리거 상태 진단
    report_lines.append("\n============================================================================")
    report_lines.append("\n🧠 종합 진단:")

    for strategy in STRATEGIES:
        s_df = df[(df["strategy"] == strategy) & df["status"].isin(["success", "fail", "pending", "failed"])]
        if s_df.empty:
            report_lines.append(f"- [{strategy}] 예측 기록 없음")
        else:
            trend = format_trend(s_df["confidence"])
            if "하락" in trend:
                report_lines.append(f"- [{strategy}] 신뢰도 저하 및 예측 안정성 재점검 필요")
            else:
                report_lines.append(f"- [{strategy}] 안정적이나 지속 관찰 필요")

    trigger_status = check_volatility_trigger_recent(df)
    report_lines.append(f"- [변동성 예측] {trigger_status}")

    return "\n".join(report_lines)
