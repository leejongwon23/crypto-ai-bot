# === diag_e2e.py (Korean Easy Summary & Detail, JSON+HTML) ===
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

# -------- 기본 유틸 --------
def _now_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

def _safe_read_csv(path, **kw):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", **kw)
    except Exception:
        return pd.DataFrame()

def _pct(v, digits=1):
    try:
        return f"{float(v)*100:.{digits}f}%"
    except Exception:
        return "0.0%"

# -------- 모델/로그/메트릭 수집 --------
def _model_inventory_parsed():
    """
    모델 파일 → (심볼 → 전략 → {모델타입}) 집계
    파일명: SYMBOL_(단기|중기|장기)_(lstm|cnn_lstm|transformer)[_...] .pt
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
    status: success|fail|v_success|v_fail 만 카운트(평가 완료 건)
    """
    ensure_prediction_log_exists()
    df = _safe_read_csv(PREDICTION_LOG)
    if df.empty or "status" not in df.columns:
        return {"ok": False, "error": "예측 로그 부족(또는 status 컬럼 없음)", "by_symbol": {}, "overall": {}}

    df["return"] = pd.to_numeric(df.get("return", 0.0), errors="coerce").fillna(0.0)

    m = df["status"].isin(["success", "fail", "v_success", "v_fail"])
    df = df[m].copy()
    if df.empty:
        return {"ok": True, "by_symbol": {}, "overall": {"total": 0, "success": 0, "fail": 0, "success_rate": 0.0, "avg_return": 0.0}}

    df["ok_flag"] = df["status"].isin(["success","v_success"]).astype(int)

    # 전체 통계
    total = len(df)
    succ = int(df["ok_flag"].sum())
    fail = int(total - succ)
    overall = {
        "total": total,
        "success": succ,
        "fail": fail,
        "success_rate": round(succ/total, 4) if total else 0.0,
        "avg_return": round(float(df["return"].mean()), 4),
    }

    # 심볼×전략×모델 → 집계
    cols = [c for c in ["symbol","strategy","model"] if c in df.columns]
    by_symbol = {}
    if cols:
        g = df.groupby(cols, dropna=False)["ok_flag"].agg(["count","sum"]).reset_index()
        for _, r in g.iterrows():
            sym = str(r.get("symbol","?"))
            strat = str(r.get("strategy","?"))
            model = str(r.get("model","?"))
            cnt = int(r["count"]); s = int(r["sum"]); f = int(cnt - s)
            by_symbol.setdefault(sym, {}).setdefault(strat, {})[model] = {
                "total": cnt, "success": s, "fail": f, "success_rate": round(s/cnt, 4) if cnt else 0.0
            }

    return {"ok": True, "by_symbol": by_symbol, "overall": overall}

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
    실패학습 지표(로그 기반): wrong_predictions.csv 요약
    """
    df = _safe_read_csv(WRONG_PREDICTIONS)
    if df.empty:
        return {"ok": False, "error": "wrong_predictions.csv 없음 또는 비어있음"}

    total = len(df)
    succ = int((df.get("success", pd.Series(dtype=str)).astype(str) == "True").sum()) if "success" in df.columns else 0
    fail = int(total - succ)

    latest_ts = None
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            latest = df.sort_values("timestamp").tail(1)
            latest_ts = str(latest["timestamp"].iloc[0]) if not latest.empty else None
        except Exception:
            latest_ts = None

    return {"ok": True, "total_rows": int(total), "success_rows": int(succ), "fail_rows": int(fail), "latest": latest_ts}

# -------- 쉬운 한글 요약/상세 생성 --------
def _aggregate_symbol_strategy_rates(by_symbol):
    """
    모델별 성과를 심볼×전략 레벨로 합산(가중 평균)
    """
    agg = {}  # agg[sym][strat] = {"total":..,"success":..,"rate":..}
    for sym, strat_map in by_symbol.items():
        for strat, model_map in strat_map.items():
            tot = succ = 0
            for mstats in model_map.values():
                tot += int(mstats.get("total",0))
                succ += int(mstats.get("success",0))
            rate = (succ / tot) if tot else 0.0
            agg.setdefault(sym, {})[strat] = {"total": tot, "success": succ, "rate": round(rate,4)}
    return agg

def _build_kr_summary(report):
    """
    누구나 이해 가능한 초간단 요약(문장형)
    - 그룹/심볼/전략: 학습 준비(모델 3종 보유) 여부
    - 예측/평가 현황 핵심 지표
    """
    lines = []
    lines.append(f"생성 시각: {report.get('timestamp','-')}")
    inv = report.get("model_inventory", {})
    lines.append(f"모델 파일 수: {inv.get('count',0)}개")
    plog = report.get("prediction_log", {})
    lines.append(f"예측 로그: {'있음' if plog.get('exists') else '없음'} (크기 {plog.get('size',0)} bytes)")
    overall = (report.get("prediction_metrics", {}) or {}).get("overall", {})
    lines.append(f"평가 집계: 총 {overall.get('total',0)}건, 성공률 {_pct(overall.get('success_rate',0.0))}, 평균 수익률 {overall.get('avg_return',0.0):.4f}")

    # 그룹별 학습 준비 상태(3모델 보유 여부)
    lines.append("그룹/심볼 학습 준비(3모델 보유) 상태:")
    for g in report.get("group_status", []):
        lines.append(f" - 그룹 {g.get('group')}:")
        for sym, sdict in g.get("symbols", {}).items():
            flags = []
            for strat in ["단기","중기","장기"]:
                flags.append(("🟢" if sdict.get(strat) else "🔴") + strat)
            lines.append(f"   · {sym}: " + " / ".join(flags))

    # 심볼×전략 성과(가중 평균)
    by_symbol = (report.get("prediction_metrics", {}) or {}).get("by_symbol", {})
    agg = _aggregate_symbol_strategy_rates(by_symbol)
    if agg:
        lines.append("심볼별 전략 성과(평가 완료 기준):")
        for sym, strat_map in agg.items():
            parts = []
            for strat in ["단기","중기","장기"]:
                stats = strat_map.get(strat)
                if stats:
                    parts.append(f"{strat} {stats['total']}건 {_pct(stats['rate'])}")
            if parts:
                lines.append(f" - {sym}: " + " / ".join(parts))

    # 실패학습
    fl = report.get("failure_learning", {})
    if fl.get("ok"):
        lines.append(f"실패학습 로그: 총 {fl.get('total_rows',0)}건 (최근 기록: {fl.get('latest','-')})")
    else:
        lines.append("실패학습 로그: 정보 없음(선택 사항)")

    # 실행 결과
    state = "정상" if report.get("ok") else "오류"
    lines.append(f"종합 상태: {state}")
    if not report.get("ok"):
        lines.append(f"오류 내용: {report.get('error','')}")
    return "\n".join(lines)

def _build_kr_detail(report):
    """
    개발자/담당자도 이해 쉬운 상세 보고(문장형 + 핵심 숫자)
    - 실행 순서, 각 단계 결과, 누락·부족 항목 표시
    """
    L = []
    L.append(f"[기본 정보] 생성 시각(KST): {report.get('timestamp','-')}, 실행 모드: "
             + ("전체 그룹" if (report.get('group',-1) in (-1,None)) else f"그룹 #{report.get('group')}"))

    # 실행흐름
    flow = ", ".join(report.get("order_trace", []))
    L.append(f"[실행 순서] {flow if flow else '기록 없음'}")

    # 학습/예측/평가 단계
    tr = report.get("train", {}) or {}
    pr = report.get("predict", {}) or {}
    ev = report.get("evaluate", {}) or {}
    if tr:
        if tr.get("mode") == "all_groups":
            L.append("① 학습: 전체 그룹 루프 실행(그룹별 학습 후 즉시 예측 포함).")
        else:
            L.append(f"① 학습: 그룹 #{tr.get('group_id','?')} 대상 심볼 {tr.get('symbols',[])} 학습 완료.")
    L.append(f"② 예측: {'실행' if pr.get('executed') else '미실행'}"
             + (f" (대상 {len(pr.get('targets',[]))}건)" if pr.get('executed') else ""))
    L.append(f"③ 평가: {'실행' if ev.get('executed') else '미실행'}")

    # 모델 현황
    inv = report.get("model_inventory", {})
    if inv.get("ok"):
        L.append(f"[모델] 저장된 모델 파일 {inv.get('count',0)}개.")
    else:
        L.append(f"[모델] 조회 실패: {inv.get('error','-')}")

    # 예측 로그
    plog = report.get("prediction_log", {})
    L.append(f"[예측 로그] {'존재' if plog.get('exists') else '없음'} (크기 {plog.get('size',0)} bytes)")

    # 평가 결과 요약
    overall = (report.get("prediction_metrics", {}) or {}).get("overall", {})
    if overall:
        L.append(f"[평가 요약] 총 {overall.get('total',0)}건, 성공 {overall.get('success',0)}건, 실패 {overall.get('fail',0)}건, "
                 f"성공률 {_pct(overall.get('success_rate',0.0))}, 평균 수익률 {overall.get('avg_return',0.0):.4f}")

    # 그룹/심볼/전략 학습 준비(모델 3종 보유) 상태
    L.append("[학습 준비 상태] (🟢 준비됨=3모델 보유, 🔴 부족)")
    for g in report.get("group_status", []):
        L.append(f" - 그룹 {g.get('group')}:")
        for sym, sdict in g.get("symbols", {}).items():
            flags = []
            for strat in ["단기","중기","장기"]:
                flags.append(("🟢" if sdict.get(strat) else "🔴") + strat)
            L.append(f"   · {sym}: " + " / ".join(flags))

    # 심볼×전략 성과(가중 평균)
    by_symbol = (report.get("prediction_metrics", {}) or {}).get("by_symbol", {})
    agg = _aggregate_symbol_strategy_rates(by_symbol)
    if agg:
        L.append("[심볼×전략 성과(평가 완료 건 기준, 가중 평균)]")
        for sym, strat_map in agg.items():
            parts = []
            for strat in ["단기","중기","장기"]:
                s = strat_map.get(strat)
                if s:
                    parts.append(f"{strat} {s['total']}건({_pct(s['rate'])})")
            if parts:
                L.append(f" - {sym}: " + " / ".join(parts))

    # 실패학습
    fl = report.get("failure_learning", {})
    if fl.get("ok"):
        L.append(f"[실패학습] 총 {fl.get('total_rows',0)}건 수집(최근 {fl.get('latest','-')}).")
    else:
        L.append("[실패학습] 로그 없음(선택 구성).")

    # 최종 상태
    if report.get("ok"):
        L.append("[상태] 정상 작동.")
    else:
        L.append(f"[상태] 오류 발생: {report.get('error','-')}")

    return "\n".join(L)

# -------- HTML 렌더러(쉬운 한글) --------
def _render_html_kr(report):
    kr_summary = report.get("kr_summary","").replace("\n","<br>")
    kr_detail = report.get("kr_detail","").replace("\n","<br>")
    return f"""
    <div style="font-family:ui-monospace,Menlo,Consolas,monospace; line-height:1.6; font-size:15px">
      <h3>🧪 YOPO 종합 점검 보고서 (한글)</h3>
      <div style="margin:8px 0; color:#444">생성 시각: <b>{report.get('timestamp','')}</b></div>
      <hr>
      <h4>요약</h4>
      <div style="background:#f7f9ff; border:1px solid #cfe0ff; padding:10px; border-radius:8px">{kr_summary}</div>
      <h4 style="margin-top:18px">상세</h4>
      <div style="background:#fff8f3; border:1px solid #ffddc2; padding:10px; border-radius:8px">{kr_detail}</div>
      <hr>
      <div style="margin-top:6px">상태:
        <b style="color:{'#0a0' if report.get('ok') else '#a00'}">{'정상' if report.get('ok') else '오류'}</b>
      </div>
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

def run(group=-1, do_predict=True, do_evaluate=True, view="json"):
    """
    End-to-End 점검 실행:
    - group==-1: 전체 그룹 학습 루프(train_symbol_group_loop) 실행(루프 내 즉시 예측 포함)
    - group>=0 : 해당 그룹만 train_models → (옵션) 예측 → (옵션) 평가
    반환:
      - view="json": JSON(dict) + 'kr_summary', 'kr_detail' 포함
      - view="html": 한국어 요약/상세를 HTML로 렌더링한 문자열
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
        "kr_summary": "",
        "kr_detail": "",
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

        # === 쉬운 한글 요약/상세 채우기 ===
        report["kr_summary"] = _build_kr_summary(report)
        report["kr_detail"]  = _build_kr_detail(report)

        return _render_html_kr(report) if view == "html" else report

    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()[-5000:]
        report["order_trace"].append("error")
        # 실패해도 한글 텍스트는 채워준다
        report["kr_summary"] = _build_kr_summary(report)
        report["kr_detail"]  = _build_kr_detail(report)
        return _render_html_kr(report) if view == "html" else report
