# diag_e2e.py  (관우 · 종합점검 라우트 · 최종본)
import os, json, datetime, pytz, re, traceback
import pandas as pd
from flask import Response

PERSIST_DIR = "/persistent"
LOG_DIR     = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR   = os.path.join(PERSIST_DIR, "models")

PRED_LOG    = os.path.join(PERSIST_DIR, "prediction_log.csv")
TRAIN_LOG   = os.path.join(LOG_DIR, "train_log.csv")
AUDIT_LOG   = os.path.join(LOG_DIR, "evaluation_audit.csv")
WRONG_LOG   = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
EVAL_RESULT = os.path.join(LOG_DIR, "evaluation_result.csv")  # 있으면 사용

KST = pytz.timezone("Asia/Seoul")
now_kst = lambda: datetime.datetime.now(KST)

# --------------------------------------------------------------------
# 안전 유틸
# --------------------------------------------------------------------
def safe_read_csv(path, **kw):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", **kw)
    except Exception:
        return pd.DataFrame()

def _tz_series(s):
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("Asia/Seoul")
        else:
            ts = ts.dt.tz_convert("Asia/Seoul")
        return ts
    except Exception:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

def _pct(x):
    try:
        return round(100.0*float(x), 2)
    except Exception:
        return 0.0

def _fmt_pct(v):
    try: return f"{float(v):.2f}%"
    except: return "0.00%"

def _table_html(df, title=None, max_rows=200):
    try:
        if df is None or df.empty:
            return f"<p><i>{title or '표'}: 데이터 없음</i></p>"
        if len(df) > max_rows:
            df = df.head(max_rows).copy()
        return (f"<h4 style='margin:10px 0'>{title}</h4>" if title else "") + \
               df.to_html(index=False, border=1)
    except Exception as e:
        return f"<p style='color:red'>표 렌더 실패: {title or ''} → {e}</p>"

# --------------------------------------------------------------------
# 모델 스캔 (+ 메타 정합성 간단 점검)
# --------------------------------------------------------------------
def scan_models():
    rows = []
    try:
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    except Exception:
        files = []

    for pt in sorted(files):
        # 이름 규칙: {symbol}_{strategy}_{model}[_groupX]_clsY.pt
        m = re.match(r"^(?P<symbol>[^_]+)_(?P<strategy>[^_]+)_(?P<model>[^_]+)(?:_group(?P<gid>\d+))?(?:_cls(?P<nc>\d+))?\.pt$", pt)
        meta_path = os.path.join(MODEL_DIR, pt.replace(".pt", ".meta.json"))
        meta = safe_load_json(meta_path)
        rows.append({
            "pt_file": pt,
            "meta_file": os.path.basename(meta_path),
            "symbol": m.group("symbol") if m else meta.get("symbol",""),
            "strategy": m.group("strategy") if m else meta.get("strategy",""),
            "model": m.group("model") if m else meta.get("model",""),
            "group_id": int(m.group("gid")) if (m and m.group("gid")) else int(meta.get("group_id", 0) or 0),
            "num_classes": int(m.group("nc")) if (m and m.group("nc")) else int(meta.get("num_classes", meta.get("class_bins", 0)) or 0),
            "val_f1": float(((meta.get("metrics") or {}).get("val_f1", 0.0))),
            "timestamp": meta.get("timestamp") or meta.get("saved_at") or ""
        })
    return pd.DataFrame(rows)

def safe_load_json(path):
    try:
        if not os.path.exists(path): return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# --------------------------------------------------------------------
# 예측 로그 요약(전략/심볼/모델별 성공률, 최근 N건 디테일)
# --------------------------------------------------------------------
def summarize_predictions():
    df = safe_read_csv(PRED_LOG)
    if df.empty:
        return {
            "summary_by_strategy": pd.DataFrame(),
            "summary_by_symbol": pd.DataFrame(),
            "summary_by_model": pd.DataFrame(),
            "recent_rows": pd.DataFrame(),
            "problems": ["prediction_log.csv 없음 또는 비어 있음"]
        }

    # 컬럼 방어
    must = ["timestamp","symbol","strategy","model","predicted_class","label","status","return"]
    for c in must:
        if c not in df.columns:
            if c == "status" and "success" in df.columns:
                # 구버전 success(bool) → status 변환
                df["status"] = df["success"].map(lambda x: "success" if str(x).lower() in ["true","1","yes","y"] else "fail")
            elif c == "return" and "rate" in df.columns:
                df["return"] = df["rate"]
            else:
                df[c] = "" if c in ["symbol","strategy","model"] else 0

    # 타임존 정규화 + 최근 N
    df["timestamp"] = _tz_series(df["timestamp"])
    df = df[df["timestamp"].notna()]

    # 성공/실패만 필터 (pending은 별도 카운트)
    is_succ = df["status"].isin(["success","v_success"])
    is_fail = df["status"].isin(["fail","v_fail"])
    is_pend = df["status"].astype(str).str.contains("pending")

    # 전략별
    strat = df.assign(
        succ=is_succ.astype(int), fail=is_fail.astype(int), pend=is_pend.astype(int)
    ).groupby("strategy")[["succ","fail","pend"]].sum().reset_index()
    if not strat.empty:
        strat["success_rate"] = strat["succ"]/(strat["succ"]+strat["fail"]).replace(0, pd.NA)

    # 심볼별(상위 30)
    sym = df.assign(succ=is_succ.astype(int), fail=is_fail.astype(int)).groupby(["strategy","symbol"])[["succ","fail"]].sum().reset_index()
    if not sym.empty:
        sym["success_rate"] = sym["succ"]/(sym["succ"]+sym["fail"]).replace(0, pd.NA)
        sym = sym.sort_values(["strategy","success_rate"], ascending=[True, False]).head(30)

    # 모델별(메타 vs 단일·파일명)
    mdl = df.assign(succ=is_succ.astype(int), fail=is_fail.astype(int)).groupby(["strategy","model"])[["succ","fail"]].sum().reset_index()
    if not mdl.empty:
        mdl["success_rate"] = mdl["succ"]/(mdl["succ"]+mdl["fail"]).replace(0, pd.NA)
        mdl = mdl.sort_values(["strategy","success_rate"], ascending=[True, False])

    # 최근 20건: 새 컬럼 포함(regime/meta_choice/raw_prob/calib_prob/calib_ver)
    keep_cols = [
        "timestamp","symbol","strategy","model","model_name","group_id",
        "predicted_class","label","direction","rate","return","status",
        "regime","meta_choice","raw_prob","calib_prob","calib_ver","reason","source"
    ]
    recent = df[sorted(set(keep_cols) & set(df.columns))].sort_values("timestamp", ascending=False).head(20).copy()
    # 포맷
    if "return" in recent.columns:
        recent["return(%)"] = (pd.to_numeric(recent["return"], errors="coerce").fillna(0)*100).round(2)
    if "rate" in recent.columns:
        recent["rate(%)"] = (pd.to_numeric(recent["rate"], errors="coerce").fillna(0)*100).round(2)

    # 문제 감지 규칙
    problems = []
    if strat.empty or (strat["succ"]+strat["fail"]).sum() == 0:
        problems.append("예측/평가 기록이 없음")
    else:
        for _, r in strat.iterrows():
            tot = float(r["succ"]+r["fail"])
            if tot >= 20 and float(r["fail"])/max(1.0, tot) > 0.7:
                problems.append(f"{r['strategy']} 실패율 과다({int(r['fail'])}/{int(tot)})")

    # 오래된 pending 경고
    try:
        pend = df[df["status"].astype(str).str.contains("pending")]
        if not pend.empty:
            late = pend[pend["timestamp"] < (now_kst()-pd.Timedelta(hours=72))]
            if len(late) > 0:
                problems.append(f"평가 대기(pending) 72h 초과 {len(late)}건")
    except Exception:
        pass

    return {
        "summary_by_strategy": strat,
        "summary_by_symbol": sym,
        "summary_by_model": mdl,
        "recent_rows": recent,
        "problems": problems
    }

# --------------------------------------------------------------------
# 학습 로그 요약
# --------------------------------------------------------------------
def summarize_training():
    df = safe_read_csv(TRAIN_LOG)
    if df.empty:
        return {
            "last_train_time": "",
            "by_strategy": pd.DataFrame(),
            "failures": 0,
            "skipped": 0,
            "problems": ["train_log.csv 없음 또는 비어 있음"]
        }
    df["timestamp"] = _tz_series(df["timestamp"])
    df = df[df["timestamp"].notna()]

    # 상태 파싱(없으면 success)
    status = df.get("status")
    if status is None:
        df["status"] = "success"
    # 전략별 최근/실패
    bys = df.sort_values("timestamp").groupby("strategy").agg(
        last=("timestamp","last"),
        total=("timestamp","count"),
        failed=("status", lambda s: (s.astype(str)=="failed").sum()),
        skipped=("status", lambda s: (s.astype(str)=="skipped").sum())
    ).reset_index()
    last_train_time = df["timestamp"].max()
    failures = int((df["status"].astype(str)=="failed").sum())
    skipped  = int((df["status"].astype(str)=="skipped").sum())

    probs = []
    if failures > 0:
        probs.append(f"학습 실패 {failures}건")
    return {
        "last_train_time": str(last_train_time) if pd.notna(last_train_time) else "",
        "by_strategy": bys,
        "failures": failures,
        "skipped": skipped,
        "problems": probs
    }

# --------------------------------------------------------------------
# 실패학습(실패 DB/CSV) 요약
# --------------------------------------------------------------------
def summarize_failures():
    # wrong_predictions.csv 기반 요약(존재 시)
    df_wrong = safe_read_csv(WRONG_LOG)
    if not df_wrong.empty:
        # 전략/심볼/사유 top
        cols = [c for c in ["strategy","symbol","reason","label"] if c in df_wrong.columns]
        grp = df_wrong.groupby([c for c in cols if c != "label"]).size().reset_index(name="count") \
                      .sort_values("count", ascending=False).head(20)
    else:
        grp = pd.DataFrame()

    # failure_db가 있으면 더 풍부하지만, 의존 깨지지 않게 옵션 처리
    try:
        from failure_db import load_failure_samples
        samples = load_failure_samples(limit=200)
        df_db = pd.DataFrame(samples) if samples else pd.DataFrame()
    except Exception:
        df_db = pd.DataFrame()

    total = (len(df_wrong) if not df_wrong.empty else 0) + (len(df_db) if not df_db.empty else 0)
    return {
        "count": total,
        "top_reasons": grp,
        "db_samples": df_db.head(20) if not df_db.empty else pd.DataFrame()
    }

# --------------------------------------------------------------------
# 최종 run() — JSON/HTML 모두 지원
# --------------------------------------------------------------------
def run(view="json", **_ignored):
    try:
        models_df = scan_models()

        pred = summarize_predictions()
        train = summarize_training()
        fails = summarize_failures()

        # 종합 문제리스트
        problems = []
        problems += pred.get("problems", [])
        problems += train.get("problems", [])

        if models_df.empty:
            problems.append("모델(.pt) 파일 없음")
        else:
            # 전략별 모델 부재 점검
            if not any(models_df["strategy"].astype(str).str.contains("단기", na=False)):
                problems.append("단기 모델 없음")
            if not any(models_df["strategy"].astype(str).str.contains("중기", na=False)):
                problems.append("중기 모델 없음")
            if not any(models_df["strategy"].astype(str).str.contains("장기", na=False)):
                problems.append("장기 모델 없음")

        # ===================== JSON 뷰 =====================
        if str(view).lower() == "json":
            out = {
                "time": now_kst().isoformat(),
                "ok": len(problems) == 0,
                "problems": problems,
                "models": models_df.to_dict(orient="records"),
                "training": {
                    "last_train_time": train["last_train_time"],
                    "by_strategy": train["by_strategy"].to_dict(orient="records") if isinstance(train["by_strategy"], pd.DataFrame) else [],
                    "failures": train["failures"],
                    "skipped": train["skipped"],
                },
                "predictions": {
                    "by_strategy": pred["summary_by_strategy"].to_dict(orient="records") if isinstance(pred["summary_by_strategy"], pd.DataFrame) else [],
                    "by_symbol":   pred["summary_by_symbol"].to_dict(orient="records") if isinstance(pred["summary_by_symbol"], pd.DataFrame)   else [],
                    "by_model":    pred["summary_by_model"].to_dict(orient="records") if isinstance(pred["summary_by_model"], pd.DataFrame)    else [],
                    "recent":      pred["recent_rows"].to_dict(orient="records") if isinstance(pred["recent_rows"], pd.DataFrame) else [],
                },
                "fail_learn": {
                    "count": fails["count"],
                    "top_reasons": fails["top_reasons"].to_dict(orient="records") if isinstance(fails["top_reasons"], pd.DataFrame) else [],
                }
            }
            return out

        # ===================== HTML 뷰 =====================
        # 상단 요약
        header = f"""
        <div style="font-family:monospace;line-height:1.5">
        <h3>✅ YOPO 종합 점검 대시보드</h3>
        <div>시간: {now_kst().isoformat()}</div>
        <div>모델 수: {0 if models_df is None else len(models_df)}</div>
        <div>실패학습 기록 수: {fails['count']}</div>
        """

        if problems:
            header += "<div style='margin-top:8px;color:#a00'><b>⚠️ 감지된 문제</b><ul>"
            for p in problems:
                header += f"<li>{p}</li>"
            header += "</ul></div>"
        else:
            header += "<div style='margin-top:8px;color:#070'><b>🟢 이상 없음</b></div>"

        header += "<hr>"
        # 표 섹션
        html = header
        html += _table_html(models_df.sort_values(["strategy","symbol","model"]), "모델 현황(심볼/전략/모델/그룹/클래스/val_f1/시각)")
        html += _table_html(train["by_strategy"], "학습 요약(전략별 최근/실패/스킵)")
        html += _table_html(pred["summary_by_strategy"], "예측·평가 요약(전략별)")
        html += _table_html(pred["summary_by_symbol"], "예측 성공률 TOP(심볼·전략)")
        html += _table_html(pred["summary_by_model"], "예측 성공률(전략·모델)")
        html += _table_html(pred["recent_rows"], "최근 예측 20건(선택모델/보정확률/레짐 포함)")
        html += _table_html(fails["top_reasons"], "실패사유 TOP 20 (wrong_predictions.csv 기준)")
        if isinstance(fails["db_samples"], pd.DataFrame) and not fails["db_samples"].empty:
            html += _table_html(fails["db_samples"], "실패 DB 샘플(최대 20)")

        html += "</div>"
        return Response(html, mimetype="text/html")

    except Exception as e:
        err = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        # JSON이 기본
        return Response(json.dumps(err, ensure_ascii=False, indent=2), mimetype="application/json")
