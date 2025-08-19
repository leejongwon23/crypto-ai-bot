# === diag_e2e.py (FINAL, 예측/평가 상세 + 한글 HTML 뷰) ===
import os, re, traceback, math
from datetime import datetime, timedelta
import pytz
import pandas as pd

# 내부 모듈
from train import train_symbol_group_loop, train_models
from predict import predict, evaluate_predictions
from config import get_SYMBOL_GROUPS
from logger import ensure_prediction_log_exists

# 경로/상수
MODEL_DIR = "/persistent/models"
PREDICTION_LOG = "/persistent/prediction_log.csv"
WRONG_PREDICTIONS = "/persistent/wrong_predictions.csv"
KST = pytz.timezone("Asia/Seoul")

NEEDED_MODELS = {"lstm", "cnn_lstm", "transformer"}
STRATS = ["단기", "중기", "장기"]

# -------- 공통 유틸 --------
def _now_kst():
    return datetime.now(KST)

def _now_str():
    return _now_kst().strftime("%Y-%m-%d %H:%M:%S")

def _safe_read_csv(path, **kw):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", **kw)
    except Exception:
        return pd.DataFrame()

def _next_eval_eta_minutes(interval_min=30):
    """다음 평가(스케줄 30분)까지 남은 분"""
    now = _now_kst()
    minute = (now.minute // interval_min) * interval_min
    slot_start = now.replace(minute=minute, second=0, microsecond=0)
    if now >= slot_start + timedelta(minutes=interval_min):
        slot_start = slot_start + timedelta(minutes=interval_min)
    next_slot = slot_start + timedelta(minutes=interval_min)
    eta = (next_slot - now).total_seconds() / 60.0
    # 음수 방지/반올림
    return max(0, int(math.ceil(eta)))

# -------- 모델 인벤토리 --------
def _model_inventory_parsed():
    """
    모델 파일 파싱 → {심볼 → {전략 → {모델타입}}}
    파일명: SYMBOL_(단기|중기|장기)_(lstm|cnn_lstm|transformer)[_...].pt
    """
    info = {}
    try:
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    except Exception as e:
        return {"ok": False, "error": f"모델 폴더 접근 실패: {e}", "map": {}, "count": 0, "files": []}

    pat = re.compile(r"(.+?)_(단기|중기|장기)_(lstm|cnn_lstm|transformer)(?:_.*)?\.pt$")
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        sym, strat, mtype = m.groups()
        info.setdefault(sym, {}).setdefault(strat, set()).add(mtype)
    return {"ok": True, "map": info, "count": len(files), "files": files[:200]}

def _group_training_status(model_map):
    """
    그룹/심볼/전략별로 필수 모델(lstm/cnn_lstm/transformer) 보유 여부
    """
    groups = get_SYMBOL_GROUPS()
    out = []
    for gid, symbols in enumerate(groups):
        entry = {"group": gid, "symbols": {}}
        for sym in symbols:
            sym_info = model_map.get(sym, {})
            strat_ok = {}
            for strat in STRATS:
                have = sym_info.get(strat, set())
                strat_ok[strat] = NEEDED_MODELS.issubset(have)
            entry["symbols"][sym] = strat_ok
        out.append(entry)
    return out

# -------- 예측/평가 상세 --------
def _prediction_metrics():
    """
    prediction_log.csv에서 심볼×전략×모델별 성공률/건수 + 최신 예측 상세
    - status: success|fail|v_success|v_fail 만 평가집계에 사용
    - 최신 예측: 심볼×전략별 최신 1건에서 model/예측클래스/예측수익률/방향 추출
    """
    ensure_prediction_log_exists()
    df = _safe_read_csv(PREDICTION_LOG)
    if df.empty:
        return {
            "ok": False,
            "error": "prediction_log.csv 없음 또는 내용 없음",
            "by_symbol": {},
            "overall": {},
            "latest": {},
        }

    # 컬럼 보정
    for col in ["return", "rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "return" not in df.columns:
        df["return"] = df["rate"] if "rate" in df.columns else 0.0

    # ---- 평가 집계(성공/실패만) ----
    eval_mask = df["status"].isin(["success", "fail", "v_success", "v_fail"]) if "status" in df.columns else pd.Series([False]*len(df))
    dfe = df[eval_mask].copy()
    if not dfe.empty:
        dfe["ok_flag"] = dfe["status"].isin(["success","v_success"]).astype(int)
        total = len(dfe)
        succ = int(dfe["ok_flag"].sum())
        fail = int(total - succ)
        overall = {
            "total": total,
            "success": succ,
            "fail": fail,
            "success_rate": round(succ/total, 4) if total else 0.0,
            "avg_return": round(float(dfe["return"].mean()), 4),
        }
        by_symbol = {}
        cols = [c for c in ["symbol","strategy","model"] if c in dfe.columns]
        if cols:
            g = dfe.groupby(cols, dropna=False)["ok_flag"].agg(["count","sum"]).reset_index()
            for _, r in g.iterrows():
                sym = str(r.get("symbol","?"))
                strat = str(r.get("strategy","?"))
                model = str(r.get("model","?"))
                cnt = int(r["count"]); s = int(r["sum"]); f = cnt - s
                by_symbol.setdefault(sym, {}).setdefault(strat, {})[model] = {
                    "total": cnt, "success": s, "fail": f,
                    "success_rate": round(s/cnt, 4) if cnt else 0.0
                }
    else:
        overall = {"total": 0, "success": 0, "fail": 0, "success_rate": 0.0, "avg_return": 0.0}
        by_symbol = {}

    # ---- 최신 예측(심볼×전략별) ----
    latest = {}
    if "timestamp" in df.columns and {"symbol","strategy"}.issubset(df.columns):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df2 = df.sort_values("timestamp").dropna(subset=["symbol","strategy"])
            grp = df2.groupby(["symbol","strategy"], dropna=False)
            idx = grp["timestamp"].idxmax()
            rows = df2.loc[idx]
            for _, r in rows.iterrows():
                sym = str(r["symbol"]); strat = str(r["strategy"])
                latest.setdefault(sym, {})[strat] = {
                    "timestamp": str(r.get("timestamp","")),
                    "model": str(r.get("model","")),
                    "predicted_class": (int(r.get("predicted_class")) if pd.notna(r.get("predicted_class")) else None) if "predicted_class" in df.columns else None,
                    "top_k": str(r.get("top_k","")) if "top_k" in df.columns else None,
                    "direction": str(r.get("direction","")) if "direction" in df.columns else "",
                    "pred_return": float(r.get("return", 0.0)) if pd.notna(r.get("return", None)) else None,
                    "status": str(r.get("status","")) if "status" in df.columns else "",
                    "reason": str(r.get("reason","")) if "reason" in df.columns else "",
                }
        except Exception:
            pass

    return {"ok": True, "by_symbol": by_symbol, "overall": overall, "latest": latest}

def _failure_learning_status():
    """
    실패학습 지표: wrong_predictions.csv 요약
    - success 컬럼이 True/False로 있는 경우 집계
    """
    df = _safe_read_csv(WRONG_PREDICTIONS)
    if df.empty:
        return {"ok": False, "error": "wrong_predictions.csv 없음 또는 비어있음"}

    # 기본 집계
    total = len(df)
    if "success" in df.columns:
        succ = int((df["success"].astype(str) == "True").sum())
        fail = total - succ
    else:
        succ, fail = 0, total

    latest_ts = None
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            latest = df.sort_values("timestamp").tail(1)
            latest_ts = str(latest["timestamp"].iloc[0]) if not latest.empty else None
        except Exception:
            latest_ts = None

    # 실패 항목에서 어느 모델/전략/심볼이 많이 나오는지 TOP5
    top5 = []
    cols = [c for c in ["symbol","strategy","model"] if c in df.columns]
    if cols and "success" in df.columns:
        dff = df[df["success"].astype(str) == "False"]
        if not dff.empty:
            g = dff.groupby(cols).size().reset_index(name="count").sort_values("count", ascending=False).head(5)
            top5 = g.to_dict(orient="records")

    return {
        "ok": True,
        "total_rows": int(total),
        "success_rows": int(succ),
        "fail_rows": int(fail),
        "latest": latest_ts,
        "top_failed": top5
    }

# -------- HTML 렌더러(한글, 쉬운 버전) --------
def _render_html_kr(report):
    def pct(x):
        try:
            return f"{float(x)*100:.1f}%"
        except Exception:
            return "0.0%"

    overall = report.get("prediction_metrics", {}).get("overall", {}) or {}
    latest = report.get("prediction_metrics", {}).get("latest", {}) or {}
    eta_min = report.get("next_eval_eta_min", 30)

    # 1) 그룹/전략 모델 보유 현황 테이블
    groups_html = []
    for g in report.get("group_status", []):
        rows = []
        for sym, sdict in g.get("symbols", {}).items():
            tds = []
            for strat in STRATS:
                ok = sdict.get(strat, False)
                tds.append(f"<td style='text-align:center'>{'🟢' if ok else '🔴'}</td>")
            rows.append(f"<tr><td>{sym}</td>{''.join(tds)}</tr>")
        groups_html.append(f"""
        <div style="margin:8px 0">
          <b>그룹 #{g.get('group')}</b>
          <table border="1" cellspacing="0" cellpadding="6" style="margin-top:6px">
            <tr style="background:#f0f0f0"><th>심볼</th><th>단기</th><th>중기</th><th>장기</th></tr>
            {''.join(rows) if rows else '<tr><td colspan=4>심볼 없음</td></tr>'}
          </table>
        </div>""")

    # 2) 최신 예측 상세(모델/클래스/예측수익률)
    latest_rows = []
    for sym, strat_map in latest.items():
        for strat, rec in strat_map.items():
            pr = rec.get("pred_return", None)
            pr_s = f"{pr*100:.2f}%" if pr is not None else "-"
            latest_rows.append(
                f"<tr><td>{sym}</td><td>{strat}</td><td>{rec.get('model','')}</td>"
                f"<td>{rec.get('predicted_class','-')}</td><td>{rec.get('direction','')}</td>"
                f"<td>{pr_s}</td><td>{rec.get('status','')}</td><td>{rec.get('reason','')}</td></tr>"
            )
    latest_table = f"""
    <table border="1" cellspacing="0" cellpadding="6">
      <tr style="background:#f0f0f0"><th>심볼</th><th>전략</th><th>선택모델</th><th>예측클래스</th><th>방향</th><th>예측수익률</th><th>상태</th><th>메모</th></tr>
      {''.join(latest_rows) if latest_rows else '<tr><td colspan=8>최근 예측 없음</td></tr>'}
    </table>"""

    # 3) 평가 집계(심볼×전략×모델 성공률)
    bysym = report.get("prediction_metrics", {}).get("by_symbol", {}) or {}
    eval_rows = []
    for sym, strat_map in bysym.items():
        for strat, mdl_map in strat_map.items():
            for mdl, m in mdl_map.items():
                eval_rows.append(
                    f"<tr><td>{sym}</td><td>{strat}</td><td>{mdl}</td>"
                    f"<td>{m.get('total',0)}</td><td>{m.get('success',0)}</td><td>{m.get('fail',0)}</td>"
                    f"<td>{pct(m.get('success_rate',0.0))}</td></tr>"
                )
    eval_table = f"""
    <table border="1" cellspacing="0" cellpadding="6">
      <tr style="background:#f0f0f0"><th>심볼</th><th>전략</th><th>모델</th><th>총건수</th><th>성공</th><th>실패</th><th>성공률</th></tr>
      {''.join(eval_rows) if eval_rows else '<tr><td colspan=7>평가 데이터 없음</td></tr>'}
    </table>"""

    inv = report.get("model_inventory", {})
    fail = report.get("failure_learning", {})
    return f"""
    <div style="font-family:monospace; line-height:1.6">
      <h3>🧪 YOPO 종합 점검 보고서</h3>
      <div>생성 시각(KST): <b>{report.get('timestamp','')}</b> | 실행 모드: <b>{'전체 그룹' if (report.get('group',-1) in (-1,None)) else f'그룹 #{report.get("group")}'}</b></div>
      <div>다음 평가까지 예상 대기: <b>{eta_min}분</b></div>
      <hr>

      <h4>① 전체 요약</h4>
      <ul>
        <li>모델 파일 수: <b>{inv.get('count',0)}</b></li>
        <li>예측 로그: <b>{'존재' if report.get('prediction_log',{}).get('exists') else '없음'}</b> (크기: {report.get('prediction_log',{}).get('size',0)} bytes)</li>
        <li>평가 집계: 총 {overall.get('total',0)}건, 성공률 {pct(overall.get('success_rate',0.0))}, 평균수익률 {overall.get('avg_return',0.0):.4f}</li>
        <li>실패학습 로그: {('OK' if fail.get('ok') else '정보부족')} (행수 {fail.get('total_rows',0)}, 최근 {fail.get('latest','-')})</li>
      </ul>

      <h4>② 그룹/전략 모델 보유 상태</h4>
      <div>🟢: 단·중·장 3모델(lstm/cnn_lstm/transformer) 모두 존재, 🔴: 부족</div>
      {''.join(groups_html)}

      <h4>③ 최신 예측 상세</h4>
      {latest_table}

      <h4>④ 평가(심볼×전략×모델)</h4>
      {eval_table}

      <h4>⑤ 실행 순서(디버그)</h4>
      <div>{', '.join(report.get('order_trace',[]))}</div>

      <hr>
      <div>상태: <b style="color:{'#0a0' if report.get('ok') else '#a00'}">{'정상' if report.get('ok') else '오류'}</b></div>
      {f"<div style='color:#a00; white-space:pre-wrap; margin-top:8px'><b>오류:</b> {report.get('error','')}</div>" if not report.get('ok') else ''}
    </div>
    """

# -------- 외부 호출 엔트리 --------
def _predict_group(symbols, strategies=STRATS):
    done = []
    for sym in symbols:
        for strat in strategies:
            try:
                predict(sym, strat, source="diag", model_type=None)
                done.append(f"{sym}-{strat}")
            except Exception as e:
                done.append(f"{sym}-{strat}:ERROR:{e}")
    return done

def run(group=-1, do_predict=True, do_evaluate=True, view="json"):
    """
    End-to-End 점검 실행:
    - group==-1: 전체 그룹 학습 루프(train_symbol_group_loop) 실행(루프 내 즉시 예측 포함)
    - group>=0 : 해당 그룹만 train_models → (옵션) 예측 → (옵션) 평가
    반환: view="json" → dict, view="html" → HTML 문자열
    """
    report = {
        "timestamp": _now_str(),
        "group": group,
        "order_trace": [],
        "ok": False,
        "train": None,
        "predict": None,
        "evaluate": None,
        "model_inventory": None,
        "prediction_log": None,
        "prediction_metrics": None,
        "group_status": None,
        "failure_learning": None,
        "next_eval_eta_min": _next_eval_eta_minutes(30),
    }

    try:
        report["order_trace"].append("start")

        # === 학습/예측 ===
        if group is None or int(group) < 0:
            report["order_trace"].append("train_symbol_group_loop:start")
            train_symbol_group_loop(sleep_sec=0)
            report["order_trace"].append("train_symbol_group_loop:done")
            report["train"] = {"mode": "all_groups"}
        else:
            groups = get_SYMBOL_GROUPS()
            gid = int(group)
            if gid >= len(groups):
                raise ValueError(f"잘못된 group 인덱스: {gid} (총 {len(groups)}개)")
            symbols = groups[gid]
            report["train"] = {"mode": "single_group", "group_id": gid, "symbols": symbols}
            report["order_trace"].append(f"train_models:g{gid}:start")
            train_models(symbols)
            report["order_trace"].append(f"train_models:g{gid}:done")
            if do_predict:
                report["order_trace"].append(f"predict:g{gid}:start")
                done = _predict_group(symbols)
                report["predict"] = {"executed": True, "targets": done}
                report["order_trace"].append(f"predict:g{gid}:done")
            else:
                report["predict"] = {"executed": False}

        # === 평가 ===
        if do_evaluate:
            report["order_trace"].append("evaluate:start")
            try:
                eval_res = evaluate_predictions()
            except TypeError:
                eval_res = evaluate_predictions
            report["evaluate"] = {"executed": True, "result": str(eval_res)[:5000]}
            report["order_trace"].append("evaluate:done")
        else:
            report["evaluate"] = {"executed": False}

        # === 상태 수집 ===
        inv = _model_inventory_parsed()
        report["model_inventory"] = {"ok": inv.get("ok", False), "count": inv.get("count",0), "files": inv.get("files", [])}
        report["prediction_log"] = {
            "ok": True,
            "exists": os.path.exists(PREDICTION_LOG),
            "size": os.path.getsize(PREDICTION_LOG) if os.path.exists(PREDICTION_LOG) else 0,
            "path": PREDICTION_LOG,
        }
        report["prediction_metrics"] = _prediction_metrics()
        report["group_status"] = _group_training_status(inv.get("map", {}))
        report["failure_learning"] = _failure_learning_status()

        report["ok"] = True
        report["order_trace"].append("done")

        if view == "html":
            return _render_html_kr(report)
        return report

    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()[-5000:]
        report["order_trace"].append("error")
        if view == "html":
            return _render_html_kr(report)
        return report
