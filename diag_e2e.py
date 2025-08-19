# diag_e2e.py (✅ 최종 안전본)

import os, json, traceback
from logger import get_available_models, export_recent_model_stats
from failure_db import load_existing_failure_hashes
from predict import evaluate_predictions
from flask import Response
import pandas as pd

LOG_DIR = "/persistent/logs"

def now_kst():
    import datetime, pytz
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()

def _prediction_metrics(cumulative=True):
    """✅ 누적 통계 로딩 (가볍게)"""
    stats_path = os.path.join(LOG_DIR, "recent_model_stats.csv")
    return safe_read_csv(stats_path)

def run(group=-1, view="json", cumulative=True):
    """✅ 안전한 점검용 실행: 절대 학습/예측/평가 안 함"""
    try:
        # 1. 모델 현황
        models = get_available_models()

        # 2. 최근 예측 통계 (누적)
        stats_df = _prediction_metrics(cumulative=cumulative)
        stats = stats_df.to_dict(orient="records") if not stats_df.empty else []

        # 3. 실패 예측 기록
        failures = load_existing_failure_hashes()
        failure_count = len(failures)

        # 4. 최종 결과
        result = {
            "time": now_kst().isoformat(),
            "models": models,
            "stats": stats,
            "failure_count": failure_count,
        }

        if view == "json":
            return Response(json.dumps(result, ensure_ascii=False, indent=2), mimetype="application/json")
        elif view == "html":
            html = f"<h2>✅ YOPO 점검 대시보드</h2>"
            html += f"<p>시간: {result['time']}</p>"
            html += f"<p>모델 수: {len(models)}</p>"
            html += f"<p>실패 기록 수: {failure_count}</p>"
            html += "<h3>최근 성능 통계</h3>"
            if stats:
                html += pd.DataFrame(stats).to_html(index=False)
            else:
                html += "<p>통계 없음</p>"
            return Response(html, mimetype="text/html")
        else:
            return result

    except Exception as e:
        return Response(json.dumps({
            "error": str(e),
            "trace": traceback.format_exc()
        }, ensure_ascii=False, indent=2), mimetype="application/json")
