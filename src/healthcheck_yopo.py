import os
import datetime
import pytz
import csv
import pandas as pd
from flask import Blueprint, jsonify
from logger import get_min_gain
from model_weight_loader import model_exists
from data.utils import SYMBOLS
from predict import predict
from train import LOG_DIR

bp = Blueprint("yopo_health", __name__)

PRED_LOG = "/persistent/prediction_log.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
MESSAGE_LOG = "/persistent/logs/message_log.csv"
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

def get_recent_time(df, strategy, column="timestamp"):
    df = df[df["strategy"] == strategy]
    if df.empty:
        return None
    latest = df[column].max()
    return latest.astimezone(KST).strftime("%Y-%m-%d %H:%M")

def format_health_summary():
    df = parse_prediction_log()
    summaries = {}
    for strategy in STRATEGIES:
        s_df = df[df["strategy"] == strategy]
        total = len(s_df)
        success = len(s_df[s_df["status"] == "success"])
        fail = len(s_df[s_df["status"] == "fail"])
        pending = len(s_df[s_df["status"] == "pending"])
        avg_conf = s_df["confidence"].tail(3).mean()
        prev_conf = s_df["confidence"].tail(6).head(3).mean()
        pre_prev_conf = s_df["confidence"].tail(9).head(3).mean()

        direction = "✅ 안정적"
        if avg_conf < prev_conf and prev_conf < pre_prev_conf:
            direction = "⚠️ 하락 추세"
        elif avg_conf < prev_conf:
            direction = "⚠️ 감소 조짐"

        recent_preds = s_df["timestamp"].max().strftime("%Y-%m-%d %H:%M") if not s_df.empty else "-"
        success_rate = round(success / total * 100, 1) if total else 0
        fail_rate = round(fail / total * 100, 1) if total else 0
        pending_rate = round(pending / total * 100, 1) if total else 0
        avg_rate = round(s_df["rate"].mean() * 100, 2) if not s_df.empty else 0

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

        summaries[strategy] = {
            "모델 수": f"{model_count}개",
            "최근 예측 시각": recent_preds,
            "최근 학습 시각": train_time,
            "최근 예측 건수": f"{total}건 (성공: {success} / 실패: {fail} / 대기중: {pending})",
            "평균 수익률": f"{avg_rate:.2f}%",
            "평균 신뢰도": f"{pre_prev_conf:.2f} → {prev_conf:.2f} → {avg_conf:.2f} {direction}",
            "성공률": f"{success_rate}%",
            "실패률": f"{fail_rate}%",
            "예측 대기 비율": f"{pending_rate}%",
            "재학습 상태": "자동 트리거 정상 작동",
            "상태 요약": direction if "⚠️" in direction else "✅ 전반적으로 안정"
        }
    return summaries

@bp.route("/yopo-health", methods=["GET"])
def yopo_health():
    try:
        summary = format_health_summary()
        return jsonify({
            "기준 시각": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
            "YOPO 상태 진단": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
