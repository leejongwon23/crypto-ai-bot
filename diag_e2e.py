# === diag_e2e.py (FINAL, 상세 한글 리포트 + pending 카운트 지원) ===
import os, re, traceback
from datetime import datetime
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

# -------- 내부 유틸 --------
def _now_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

def _safe_read_csv(path, **kw):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", **kw)
    except Exception:
        return pd.DataFrame()

def _model_inventory_parsed():
    """
    모델 파일을 파싱해서 (심볼 → 전략 → {모델타입}) 형태로 집계
    파일명 패턴: SYMBOL_단기|중기|장기_lstm|cnn_lstm|transformer[_...] .pt
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

def _prediction_metrics():
    """
    prediction_log.csv에서 심볼×전략×모델별 성공률/건수 계산
    - 평가완료: status ∈ {success, fail, v_success, v_fail}
    - 대기중:   status ∈ {pending, v_pending}
    """
    ensure_prediction_log_exists()
    df = _safe_read_csv(PREDICTION_LOG)
    if df.empty or "status" not in df.columns:
        return {
            "ok": False, "error": "예측 로그 부족(또는 status 컬럼 없음)",
            "by_symbol": {}, "overall": {}, "pending": {"overall":0, "by_symbol":{}}
        }

    # 숫자화
    if "return" in df.columns:
        df["return"] = pd.to_numeric(df["return"], errors="coerce")
    else:
        df["return"] = 0.0

    # --- 평가 완료 집계 ---
    is_done = df["status"].isin(["success","fail","v_success","v_fail"])
    done = df[is_done].copy()

    # 성공 플래그
    if not done.empty:
        done["ok_flag"] = done["status"].isin(["success","v_success"]).astype(int)
        total = len(done)
        succ = int(done["ok_flag"].sum())
        fail = int(total - succ)
        overall = {
            "total": total,
            "success": succ,
            "fail": fail,
            "success_rate": round(succ/total, 4) if total else 0.0,
            "avg_return": round(float(done["return"].mean() if "return" in done else 0.0), 4)
        }
        # 심볼×전략×모델
        cols = [c for c in ["symbol","strategy","model"] if c in done.columns]
        by_symbol = {}
        if cols:
            g = done.groupby(cols, dropna=False)["ok_flag"].agg(["count","sum"]).reset_index()
            for _, r in g.iterrows():
                sym = str(r.get("symbol","?")); strat = str(r.get("strategy","?")); model = str(r.get("model","?"))
                cnt = int(r["count"]); s = int(r["sum"]); f = int(cnt - s)
                by_symbol.setdefault(sym, {}).setdefault(strat, {})[model] = {
                    "total": cnt, "success": s, "fail": f, "success_rate": round(s/cnt, 4) if cnt else 0.0
                }
    else:
        overall = {"total":0,"success":0,"fail":0,"success_rate":0.0,"avg_return":0.0}
        by_symbol = {}

    # --- 대기(미평가) 집계 ---
    is_pending = df["status"].isin(["pending","v_pending"])
    pend = df[is_pending].copy()
    pending_overall = int(len(pend))
    pending_by_symbol = {}
    if not pend.empty:
        cols2 = [c for c in ["symbol","strategy"] if c in pend.columns]
        if cols2:
            g2 = pend.groupby(cols2, dropna=False)["status"].count().reset_index().rename(columns={"status":"count"})
            for _, r in g2.iterrows():
                sym = str(r.get("symbol","?")); strat = str(r.get("strategy","?"))
                pending_by_symbol.setdefault(sym, {})[strat] = int(r["count"])

    return {
        "ok": True,
        "by_symbol": by_symbol,
        "overall": overall,
        "pending": {"overall": pending_overall, "by_symbol": pending_by_symbol}
    }

def _group_training_status(model_map):
    """
    그룹/심볼/전략별로 필수 모델(lstm, cnn_lstm, transformer) 보유 여부 확인
    """
    groups = get_SYMBOL_GROUPS()
    need = {"lstm","cnn_lstm","transformer"}
    out = []
    for gid, symbols in enumerate(groups):
        entry = {"group": gid, "symbols": {}}
        for sym in symbols:
            sym_info = model_map.get(sym, {})
            strat_ok = {}
            for strat in ["단기","중기","장기"]:
                have = sym_info.get(strat, set())
                strat_ok[strat] = need.issubset(have)
            entry["symbols"][sym] = strat_ok
        out.append(entry)
    return out

def _failure_learning_status():
    """
    실패학습 관련 지표(로그 기반): wrong_predictions.csv 요약
    """
    df = _safe_read_csv(WRONG_PREDICTIONS)
    if df.empty:
        return {"ok": False, "error": "wrong_predictions.csv 없음 또는 비어있음"}

    if "success" in df.columns:
        total = len(df)
        succ = int((df["success"].astype(str) == "True").sum())
        fail = int(total - succ)
    else:
        total = len(df); succ = 0; fail = total

    latest_ts = None
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            latest = df.sort_values("timestamp").tail(1)
            latest_ts = str(latest["timestamp"].iloc[0]) if not latest.empty else None
        except Exception:
            latest_ts = None

    return {"ok": True, "total_rows": int(total), "success_rows": int(succ), "fail_rows": int(fail), "latest": latest_ts}

# -------- HTML 렌더러(한글) --------
def _render_html_kr(report, detail=False):
    def pct(x):
        try:
            return f"{float(x)*100:.1f}%"
        except Exception:
            return "0.0%"

    overall = report.get("prediction_metrics", {}).get("overall", {}) if report.get("prediction_metrics") else {}
    pending = report.get("prediction_metrics", {}).get("pending", {}) if report.get("prediction_metrics") else {}
    overall_total = overall.get("total", 0)
    overall_sr = pct(overall.get("success_rate", 0.0))
    overall_avg = overall.get("avg_return", 0.0)
    pending_total = pending.get("overall", 0)

    # 그룹별 표 (모델 보유 상태)
    groups_html = []
    grp = report.get("group_status", [])
    for g in grp:
        rows = []
        for sym, sdict in g.get("symbols", {}).items():
            td = []
            for strat in ["단기","중기","장기"]:
                good = sdict.get(strat, False)
                td.append(f"<td style='text-align:center'>{'🟢' if good else '🔴'}</td>")
            rows.append(f"<tr><td>{sym}</td>{''.join(td)}</tr>")
        table = f"""
        <div style="margin:10px 0">
          <b>그룹 #{g.get('group')}</b>
          <table border="1" cellspacing="0" cellpadding="6" style="margin-top:6px">
            <tr style="background:#f0f0f0"><th>심볼</th><th>단기(모델)</th><th>중기(모델)</th><th>장기(모델)</th></tr>
            {''.join(rows) if rows else '<tr><td colspan=4>심볼 없음</td></tr>'}
          </table>
        </div>"""
        groups_html.append(table)

    # 상세 성능 표 (옵션)
    detail_html = ""
    if detail:
        by_symbol = report.get("prediction_metrics", {}).get("by_symbol", {}) or {}
        pen_by = pending.get("by_symbol", {}) or {}

        # 심볼/전략/모델 테이블
        rows = []
        for sym in sorted(by_symbol.keys()):
            strat_map = by_symbol[sym]
            for strat in ["단기","중기","장기"]:
                model_map = strat_map.get(strat, {})
                # 대기건수
                pend_cnt = pen_by.get(sym, {}).get(strat, 0)
                if not model_map:
                    rows.append(f"<tr><td>{sym}</td><td>{strat}</td><td colspan=5>평가 완료 건 없음</td><td>{pend_cnt}</td></tr>")
                    continue
                for model in ["lstm","cnn_lstm","transformer"]:
                    m = model_map.get(model)
                    if m:
                        rows.append(
                            f"<tr><td>{sym}</td><td>{strat}</td><td>{model}</td>"
                            f"<td>{m['total']}</td><td>{m['success']}</td><td>{m['fail']}</td>"
                            f"<td>{pct(m['success_rate'])}</td><td>{pend_cnt}</td></tr>"
                        )
        if not rows:
            rows = ["<tr><td colspan=8>집계할 평가 완료 건이 없습니다.</td></tr>"]

        detail_html = f"""
        <h4>3) 상세 성능 (심볼×전략×모델, 평가완료 기준)</h4>
        <div>열 설명: 총건수/성공/실패/성공률은 평가완료 건만, '대기건수'는 pending+v_pending</div>
        <table border="1" cellspacing="0" cellpadding="6" style="margin-top:6px">
          <tr style="background:#f0f0f0">
            <th>심볼</th><th>전략</th><th>모델</th>
            <th>총건수</th><th>성공</th><th>실패</th><th>성공률</th><th>대기건수</th>
          </tr>
          {''.join(rows)}
        </table>
        """

    # 핵심 요약
    inv = report.get("model_inventory", {})
    fail = report.get("failure_learning", {})
    train = report.get("train", {})
    predict = report.get("predict", {})
    evaluate = report.get("evaluate", {})

    return f"""
    <div style="font-family:monospace; line-height:1.6">
      <h3>🧪 YOPO 종합 점검 보고서</h3>
      <div>생성 시각(KST): <b>{report.get('timestamp','')}</b> | 실행 모드: <b>{'전체 그룹' if (report.get('group',-1) in (-1, None)) else f'그룹 #{report.get('group')}'}</b></div>
      <hr>
      <h4>1) 전체 현황</h4>
      <ul>
        <li>모델 파일 수: <b>{inv.get('count',0)}</b></li>
        <li>예측 로그: <b>{'존재' if report.get('prediction_log',{}).get('exists') else '없음'}</b> (크기: {report.get('prediction_log',{}).get('size',0)} bytes)</li>
        <li>평가 집계(완료): 총 {overall_total}건, 성공률 {overall_sr}, 평균수익률 {overall_avg:.4f}</li>
        <li>예측 대기(미평가): 총 {pending_total}건</li>
        <li>실패학습 로그: {('OK' if fail.get('ok') else '정보부족')} (행수 {fail.get('total_rows',0)}, 최근 {fail.get('latest','-')})</li>
      </ul>

      <h4>2) 그룹/전략 학습 완결 상태</h4>
      <div>🟢: 단·중·장 3모델(lstm/cnn_lstm/transformer) 모두 존재 | 🔴: 부족</div>
      {''.join(groups_html)}

      {detail_html}

      <h4>{'4' if detail else '3'}) 실행 기록</h4>
      <ul>
        <li>학습: {train.get('mode','-')} {('→ 그룹#'+str(train.get('group_id')) if 'group_id' in train else '')}</li>
        <li>예측: {"실행" if predict.get("executed") else "미실행"} | 대상 수: {len(predict.get('targets',[])) if predict.get('executed') else 0}</li>
        <li>평가: {"실행" if evaluate.get("executed") else "미실행"}</li>
      </ul>

      <h4>{'5' if detail else '4'}) 내부 순서 추적(디버그)</h4>
      <div>{', '.join(report.get('order_trace',[]))}</div>

      <hr>
      <div>상태: <b style="color:{'#0a0' if report.get('ok') else '#a00'}">{'정상' if report.get('ok') else '오류'}</b></div>
      {f"<div style='color:#a00; white-space:pre-wrap; margin-top:8px'><b>오류:</b> {report.get('error','')}</div>" if not report.get('ok') else ''}
    </div>
    """

# -------- 외부 호출 엔트리 --------
def _predict_group(symbols, strategies=("단기","중기","장기")):
    done = []
    for sym in symbols:
        for strat in strategies:
            try:
                predict(sym, strat, source="diag", model_type=None)
                done.append(f"{sym}-{strat}")
            except Exception as e:
                done.append(f"{sym}-{strat}:ERROR:{e}")
    return done

def run(group=-1, do_predict=True, do_evaluate=True, view="json", detail=False):
    """
    End-to-End 점검 실행:
    - group==-1: 전체 그룹 학습 루프(train_symbol_group_loop) 실행(루프 내 즉시 예측 포함)
    - group>=0 : 해당 그룹만 train_models → (옵션) 예측 → (옵션) 평가
    반환:
      view="json"  → dict
      view="html"  → HTML 문자열(한글)
      detail=True  → HTML에 심볼×전략×모델 상세 표 + 대기건수 표시
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
            return _render_html_kr(report, detail=bool(detail))
        return report

    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()[-5000:]
        report["order_trace"].append("error")
        if view == "html":
            return _render_html_kr(report, detail=bool(detail))
        return report
