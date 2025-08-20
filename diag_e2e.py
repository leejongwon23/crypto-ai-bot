# === diag_e2e.py (관우 v2.2-final: 메타선택 표시 + 문제진단 강화 + 가독성 향상 + 작동순서 뷰 + 모델별 최신 클래스/수익률) ===
import os, json, traceback, datetime, pytz, re
import pandas as pd
from collections import defaultdict, Counter

PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
TRAIN_LOG = os.path.join(LOG_DIR, "train_log.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")

KST = pytz.timezone("Asia/Seoul")
now_kst = lambda: pd.Timestamp.now(tz="Asia/Seoul")  # tz-aware

EVAL_HORIZON_HOURS = {"단기": 4, "중기": 24, "장기": 168}
STRATEGIES = ["단기", "중기", "장기"]
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]

# ===================== 유틸 =====================
def _safe_read_csv(path, **kwargs):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", **kwargs)
    except Exception:
        return pd.DataFrame()

def _to_kst(ts):
    try:
        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t): return None
        if t.tzinfo is None: t = t.tz_localize("Asia/Seoul")
        else: t = t.tz_convert("Asia/Seoul")
        return t
    except Exception:
        return None

def _pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except:
        return "0.0%"

def _fmt_ts(ts):
    return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "없음"

def _eval_deadline(pred_ts, strategy):
    if pred_ts is None: return None
    horizon_h = EVAL_HORIZON_HOURS.get(strategy, 6)
    return pred_ts + pd.Timedelta(hours=horizon_h)

def _grade_rate(rate):
    try: r = float(rate)
    except: r = 0.0
    if r >= 0.60: return "ok"
    if r >= 0.40: return "warn"
    return "err"

def _delay_badge(delay_min):
    if delay_min <= 0: return "ok"
    if delay_min <= 30: return "warn"
    return "err"

def _num(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def _list_models():
    out = []
    if not os.path.isdir(MODEL_DIR): return out
    for fn in os.listdir(MODEL_DIR):
        if not fn.endswith(".pt"): continue
        base = fn[:-3]
        m = re.match(r"^(?P<sym>[^_]+)_(?P<strat>단기|중기|장기)_(?P<model>lstm|cnn_lstm|transformer)(?:_group(?P<gid>\d+))?(?:_cls(?P<ncls>\d+))?$", base)
        if not m: continue
        d = m.groupdict()
        meta_path = os.path.join(MODEL_DIR, base + ".meta.json")
        val_f1, saved = None, None
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                val_f1 = float(meta.get("metrics", {}).get("val_f1", 0.0))
                saved_raw = meta.get("timestamp") or meta.get("saved_at")
                saved = _to_kst(saved_raw)
        except Exception:
            pass
        out.append({
            "pt_file": fn,
            "meta_file": os.path.basename(meta_path),
            "symbol": d["sym"],
            "strategy": d["strat"],
            "model": d["model"],
            "group_id": int(d["gid"]) if d.get("gid") else 0,
            "num_classes": int(d["ncls"]) if d.get("ncls") else None,
            "val_f1": val_f1,
            "timestamp": saved.isoformat() if saved else None
        })
    return out

def _summarize_fail_patterns(df_pred_sym):
    try:
        df = df_pred_sym.copy()
        if "status" in df.columns:
            df = df[df["status"].isin(["fail","v_fail"])]
        else:
            df = df[str(df.get("success","")).lower() == "false"]
        if df.empty: return []
        reasons = df.get("reason")
        if reasons is None: return []
        top = Counter([str(x).split("|")[0].strip() for x in reasons.dropna().tolist()]).most_common(3)
        return [f"{r} ({c})" for r,c in top]
    except Exception:
        return []

# ===================== 스냅샷 집계 =====================
def _build_snapshot(symbols_filter=None):
    df_pred = _safe_read_csv(PREDICTION_LOG)
    df_train = _safe_read_csv(TRAIN_LOG)
    _ = _safe_read_csv(AUDIT_LOG)

    # tz-normalize
    if "timestamp" in df_pred.columns:
        df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], errors="coerce")
        try:
            if getattr(df_pred["timestamp"].dt, "tz", None) is None:
                df_pred["timestamp"] = df_pred["timestamp"].dt.tz_localize("Asia/Seoul")
            else:
                df_pred["timestamp"] = df_pred["timestamp"].dt.tz_convert("Asia/Seoul")
        except Exception:
            pass
    else:
        df_pred["timestamp"] = pd.NaT

    if "timestamp" in df_train.columns:
        df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], errors="coerce")
        try:
            if getattr(df_train["timestamp"].dt, "tz", None) is None:
                df_train["timestamp"] = df_train["timestamp"].dt.tz_localize("Asia/Seoul")
            else:
                df_train["timestamp"] = df_train["timestamp"].dt.tz_convert("Asia/Seoul")
        except Exception:
            pass
    else:
        df_train["timestamp"] = pd.NaT

    models = _list_models()

    symbols = set([m["symbol"] for m in models])
    if "symbol" in df_pred.columns:
        symbols |= set(df_pred["symbol"].dropna().astype(str).tolist())
    symbols = sorted([s for s in symbols if s and s != "nan"])

    if symbols_filter:
        allow = set([s.strip() for s in symbols_filter.split(",") if s.strip()])
        symbols = [s for s in symbols if s in allow]

    model_index = defaultdict(list)
    for m in models:
        key = (m["symbol"], m["strategy"], m["model"])
        model_index[key].append(m)

    snapshot = {"time": now_kst().isoformat(), "symbols": []}

    for sym in symbols:
        sym_block = {"symbol": sym, "strategies": {}, "fail_summary": []}
        df_ps = df_pred[df_pred["symbol"] == sym] if "symbol" in df_pred.columns else pd.DataFrame()
        sym_block["fail_summary"] = _summarize_fail_patterns(df_ps)

        for strat in STRATEGIES:
            # 최근 학습
            if not df_train.empty:
                df_ts = df_train[(df_train["symbol"] == sym) & (df_train["strategy"] == strat)]
                last_train_ts = df_ts["timestamp"].max() if "timestamp" in df_ts.columns and not df_ts.empty else pd.NaT
            else:
                last_train_ts = pd.NaT

            # 해당 심볼/전략 예측 로그
            df_ss = df_ps[df_ps["strategy"] == strat] if not df_ps.empty else pd.DataFrame()

            def _stat_count(df, label):
                if df.empty or "status" not in df.columns: return 0
                return int((df["status"] == label).sum())

            if not df_ss.empty:
                if "status" in df_ss.columns:
                    df_ss["is_vol"] = df_ss["status"].astype(str).str.startswith("v_")
                else:
                    df_ss["is_vol"] = False
                try:
                    r_col = pd.to_numeric(df_ss.get("return", 0.0), errors="coerce").fillna(0.0)
                    df_ss["_return_val"] = r_col
                except Exception:
                    df_ss["_return_val"] = 0.0

            nvol = df_ss[~df_ss["is_vol"]] if not df_ss.empty else pd.DataFrame()
            vol  = df_ss[df_ss["is_vol"]]  if not df_ss.empty else pd.DataFrame()

            summary_n = {
                "succ": _stat_count(nvol, "success"),
                "fail": _stat_count(nvol, "fail"),
                "pending": _stat_count(nvol, "pending"),
                "failed": _stat_count(nvol, "failed"),
                "total": len(nvol),
                "avg_return": float(nvol["_return_val"].mean()) if not nvol.empty else 0.0
            }
            t = max(1, summary_n["total"]); summary_n["succ_rate"] = summary_n["succ"]/t

            summary_v = {
                "succ": _stat_count(vol, "v_success"),
                "fail": _stat_count(vol, "v_fail"),
                "pending": _stat_count(vol, "pending"),
                "failed": _stat_count(vol, "failed"),
                "total": len(vol),
                "avg_return": float(vol["_return_val"].mean()) if not vol.empty else 0.0
            }
            tv = max(1, summary_v["total"]); summary_v["succ_rate"] = summary_v["succ"]/tv

            # 모델별 (✅ 최신 클래스/수익률 포함)
            def _latest_for_model(df_model):
                if df_model.empty: 
                    return None, None
                dfm = df_model.copy()
                try:
                    dfm = dfm.sort_values("timestamp")
                except Exception:
                    pass
                last = dfm.iloc[-1]
                # 클래스 추정 컬럼 우선순위
                for key in ["predicted_class", "class", "pred_class", "label"]:
                    if key in dfm.columns:
                        val = last.get(key, None)
                        if pd.notna(val):
                            latest_cls = str(val)
                            break
                else:
                    latest_cls = "-"
                # 수익률 컬럼 우선순위
                latest_ret = None
                if "return" in dfm.columns:
                    latest_ret = _num(last.get("return"))
                elif "rate" in dfm.columns:
                    latest_ret = _num(last.get("rate"))
                return latest_cls, latest_ret

            models_detail = []
            for mt in MODEL_TYPES:
                lst = model_index.get((sym, strat, mt), [])
                latest_f1 = None
                if lst:
                    try:
                        lst_sorted = sorted(lst, key=lambda x: x.get("timestamp") or "", reverse=True)
                        latest_f1 = lst_sorted[0].get("val_f1", None)
                    except Exception:
                        pass
                df_model = df_ss[df_ss["model"].astype(str).str.contains(mt, na=False)] if not df_ss.empty else pd.DataFrame()
                latest_cls, latest_ret = _latest_for_model(df_model)
                md = {
                    "model": mt,
                    "val_f1": float(latest_f1) if latest_f1 is not None else None,
                    "succ": _stat_count(df_model, "success") + _stat_count(df_model, "v_success"),
                    "fail": _stat_count(df_model, "fail") + _stat_count(df_model, "v_fail"),
                    "total": int(len(df_model)),
                    "latest_class": latest_cls,
                    "latest_return": latest_ret
                }
                denom = max(1, md["total"]); md["succ_rate"] = md["succ"] / denom
                models_detail.append(md)

            # 평가 일정
            last_pred_ts = df_ss["timestamp"].max() if "timestamp" in df_ss.columns and not df_ss.empty else pd.NaT
            eval_due = _eval_deadline(last_pred_ts, strat) if pd.notna(last_pred_ts) else None

            last_eval_ts = None
            if not df_ss.empty:
                df_eval = df_ss.copy()
                try:
                    cond = (df_eval.get("source","") == "평가") | df_eval.get("direction","").astype(str).str.startswith("평가:")
                    df_eval = df_eval[cond]
                except Exception:
                    df_eval = pd.DataFrame()
                if not df_eval.empty:
                    last_eval_ts = df_eval["timestamp"].max()

            delayed_min = 0
            if eval_due is not None and last_eval_ts is not None:
                delayed_min = int(max(0, (last_eval_ts - eval_due) / pd.Timedelta(minutes=1)))
            elif eval_due is not None and last_eval_ts is None:
                now = now_kst()
                delayed_min = int(max(0, (now - eval_due) / pd.Timedelta(minutes=1))) if now > eval_due else 0

            # 메타 선택
            meta_choice_txt = "-"
            if not df_ss.empty:
                try:
                    meta_rows = df_ss[df_ss["model"].astype(str) == "meta"]
                except Exception:
                    meta_rows = pd.DataFrame()
                if not meta_rows.empty:
                    last_meta = meta_rows.sort_values("timestamp").iloc[-1]
                    mc = last_meta.get("meta_choice", None)
                    if pd.notna(mc) and str(mc).strip():
                        meta_choice_txt = str(mc)
                    else:
                        mn = last_meta.get("model_name", None)
                        if pd.notna(mn) and str(mn).strip():
                            meta_choice_txt = str(mn)
                        else:
                            nt = last_meta.get("note", None)
                            if pd.notna(nt) and str(nt).strip():
                                meta_choice_txt = str(nt)

            # 실패학습 반영률
            recent_fail = df_ss[df_ss["status"].isin(["fail","v_fail"])] if "status" in df_ss.columns else pd.DataFrame()
            recent_fail_n = int(len(recent_fail)); reflected = 0
            if recent_fail_n > 0 and "timestamp" in df_ss.columns:
                last_fail_time = recent_fail["timestamp"].max()
                after = df_ss[df_ss["timestamp"] > last_fail_time]
                reflected = int((after["status"].isin(["success","v_success"])).sum()) if "status" in after.columns else 0
            reflect_ratio = (reflected / max(1, recent_fail_n)) if recent_fail_n>0 else None

            # 전략 문제
            strat_problems = []
            total_models_for_strat = sum(len(model_index.get((sym, strat, mt), [])) for mt in MODEL_TYPES)
            if total_models_for_strat == 0: strat_problems.append("모델 파일 없음")
            if df_ss.empty: strat_problems.append("예측 기록 없음")
            if delayed_min > 0: strat_problems.append(f"평가 지연 {delayed_min}분")
            if pd.isna(last_train_ts): strat_problems.append("최근 학습 기록 없음")
            if eval_due is None and pd.isna(last_pred_ts):
                strat_problems.append("최근 예측 시각 없음(평가 예정 산출 불가)")

            sym_block["strategies"][strat] = {
                "last_train_time": last_train_ts.isoformat() if pd.notna(last_train_ts) else None,
                "prediction": {
                    "normal": {
                        "succ": summary_n["succ"], "fail": summary_n["fail"],
                        "pending": summary_n["pending"], "failed": summary_n["failed"],
                        "total": summary_n["total"],
                        "succ_rate": summary_n["succ_rate"],
                        "avg_return": summary_n["avg_return"],
                    },
                    "volatility": {
                        "succ": summary_v["succ"], "fail": summary_v["fail"],
                        "pending": summary_v["pending"], "failed": summary_v["failed"],
                        "total": summary_v["total"],
                        "succ_rate": summary_v["succ_rate"],
                        "avg_return": summary_v["avg_return"],
                    },
                    "by_model": models_detail,           # ← 모델별 누적/성공률 + 최신 클래스/수익률
                    "meta_choice": meta_choice_txt,      # ← 메타러너 선택 모델
                },
                "evaluation": {
                    "last_prediction_time": last_pred_ts.isoformat() if pd.notna(last_pred_ts) else None,
                    "due_time": eval_due.isoformat() if eval_due is not None else None,
                    "last_evaluated_time": last_eval_ts.isoformat() if last_eval_ts is not None else None,
                    "delay_min": delayed_min
                },
                "failure_learning": {
                    "recent_fail": recent_fail_n,
                    "reflected_count_after": reflected,
                    "reflect_ratio": reflect_ratio
                },
                "problems": strat_problems
            }
        snapshot["symbols"].append(sym_block)

    # 상단 요약
    total_normal_succ = total_normal_fail = 0
    total_vol_succ = total_vol_fail = 0
    problems = []
    for s in snapshot["symbols"]:
        for strat, blk in s["strategies"].items():
            n = blk["prediction"]["normal"]; v = blk["prediction"]["volatility"]
            total_normal_succ += n["succ"]; total_normal_fail += n["fail"]
            total_vol_succ += v["succ"]; total_vol_fail += v["fail"]
            if _grade_rate(n["succ_rate"]) == "err" and n["total"] >= 10:
                problems.append(f"{s['symbol']} {strat}: 일반 성공률 낮음({int(n['succ_rate']*100)}%)")
            if blk["evaluation"]["delay_min"] > 0:
                problems.append(f"{s['symbol']} {strat}: 평가 지연({blk['evaluation']['delay_min']}분)")
            for p in (blk.get("problems") or []):
                problems.append(f"{s['symbol']} {strat}: {p}")

    def _rate(a,b): denom = max(1,(a+b)); return a/denom
    snapshot["summary"] = {
        "normal_success_rate": _rate(total_normal_succ, total_normal_fail),
        "vol_success_rate": _rate(total_vol_succ, total_vol_fail),
        "symbols_count": len(snapshot["symbols"]),
        "models_count": len(_list_models()),
        "problems": problems
    }
    return snapshot

# ===================== HTML 렌더 =====================
def _render_html(snapshot):
    css = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', Arial, sans-serif; line-height:1.55; background:#f6f7fb; }
  .wrap { max-width: 1180px; margin: 20px auto; }
  .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; vertical-align:middle; }
  .ok { background:#e6ffed; color:#037a0d; border:1px solid #b7f5c0; }
  .warn { background:#fff7e6; color:#8a5b00; border:1px solid #ffe1a1; }
  .err { background:#ffecec; color:#a10000; border:1px solid #ffb3b3; }
  .card { border:1px solid #e2e8f0; border-radius:12px; padding:14px; margin:14px 0; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
  .subtle { color:#555; }
  table { border-collapse:collapse; width:100%; }
  th, td { border:1px solid #e5e7eb; padding:8px 10px; font-size:13px; text-align:center; }
  th { background:#f8fafc; }
  details { margin:8px 0; }
  summary { cursor:pointer; font-weight:600; outline:none; }
  .legend span { margin-right:8px; }
  .sticky-top { position: sticky; top: 0; background: #eef3ff; padding: 12px; border: 1px solid #ccd; z-index: 10; border-radius:12px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .index { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 0; }
  .index a { text-decoration:none; border:1px solid #e2e8f0; background:#fff; padding:6px 10px; border-radius:10px; font-size:13px; color:#333; }
  .kicker { color:#6b7280; font-size:12px; }
  .pill { border-radius:999px; padding:2px 8px; border:1px solid #e5e7eb; background:#fafafa; font-size:12px; }
  .row-title { font-weight:700; margin:6px 0; }
  .muted { color:#6b7280; }
  .small { font-size:12px; }
  .controls { display:flex; gap:8px; align-items:center; margin-top:8px; flex-wrap:wrap; }
  .btn { cursor:pointer; border:1px solid #d1d5db; background:#ffffff; padding:6px 10px; border-radius:8px; font-size:12px; }
  .view { display:none; }
  .view.active { display:block; }
  .step { font-weight:700; margin:6px 0 8px; }
  .hr { height:1px; background:#e5e7eb; margin:10px 0; }
</style>
"""
    sm = snapshot.get("summary", {})
    problems = sm.get("problems", []) or []
    status_class = "ok" if not problems else "err"
    status_text = "🟢 전체 정상" if not problems else f"🔴 문제 {len(problems)}건"

    idx_links = []
    for sym_item in snapshot.get("symbols", []):
        sym = sym_item.get("symbol","")
        if sym: idx_links.append(f"<a href='#{sym}'>{sym}</a>")
    idx_html = "<div class='index'>" + "".join(idx_links) + "</div>" if idx_links else ""

    header = f"""
<div class="sticky-top mono">
  <div><b>YOPO 통합 점검</b> <span class="kicker">— 시스템 상태를 한 눈에</span></div>
  <div class="small">생성시각 {snapshot.get('time','')}</div>
  <div style="margin-top:6px">
    <span class="badge {status_class}">{status_text}</span>
    <span class="pill">일반 성공률 {_pct(sm.get('normal_success_rate',0))}</span>
    <span class="pill">변동성 성공률 {_pct(sm.get('vol_success_rate',0))}</span>
    <span class="pill">심볼 {sm.get('symbols_count',0)}개</span>
    <span class="pill">모델 파일 {sm.get('models_count',0)}개</span>
  </div>
  <div class="legend" style="margin-top:6px">
    <span class="badge ok">성공률 양호 ≥60%</span>
    <span class="badge warn">보통 40~60%</span>
    <span class="badge err">주의 &lt;40% / 평가 지연</span>
  </div>
  <div class="controls">
    <button class="btn" onclick="switchView('symbol')">심볼 중심 보기</button>
    <button class="btn" onclick="switchView('flow')">작동순서 보기(전략→모델→심볼)</button>
    <button class="btn" onclick="toggleAll(true)">모두 펼치기</button>
    <button class="btn" onclick="toggleAll(false)">모두 접기</button>
  </div>
  {idx_html}
</div>
<script>
function toggleAll(open) {{
  document.querySelectorAll('details').forEach(d => d.open = open);
}}
function switchView(which) {{
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + which).classList.add('active');
}}
window.addEventListener('DOMContentLoaded', () => switchView('flow')); // 기본: 작동순서 보기
</script>
"""

    # ===== 심볼 중심 뷰 =====
    def render_symbol_centric():
        parts = []
        for sym_item in snapshot.get("symbols", []):
            sym = sym_item.get("symbol")
            fs = sym_item.get("fail_summary", []) or []
            fs_html = f"<div class='muted small'>최근 실패 패턴: {', '.join(fs)}</div>" if fs else ""
            sym_cards = []
            for strat, blk in (sym_item.get("strategies") or {}).items():
                n = blk["prediction"]["normal"]; v = blk["prediction"]["volatility"]
                by_model = blk["prediction"]["by_model"]; ev = blk["evaluation"]; fl = blk["failure_learning"]
                meta_choice = blk["prediction"].get("meta_choice", "-")
                n_cls, v_cls = _grade_rate(n["succ_rate"]), _grade_rate(v["succ_rate"])
                delay_cls = _delay_badge(ev.get("delay_min", 0))

                head = (f"<div class='row-title'>전략: <b>{strat}</b> &nbsp;"
                        f"<span class='muted small'>최근 학습 {_fmt_ts(_to_kst(blk['last_train_time']))}</span> &nbsp;"
                        f"<span class='badge warn'>메타 선택: {meta_choice}</span></div>")

                pred_table = (
                    "<table><tr><th>구분</th><th>성공</th><th>실패</th><th>대기</th><th>기록오류</th>"
                    "<th>총건수</th><th>성공률</th><th>평균수익</th></tr>"
                    f"<tr><td>일반</td><td>{n['succ']}</td><td>{n['fail']}</td><td>{n['pending']}</td><td>{n['failed']}</td>"
                    f"<td>{n['total']}</td><td>{_pct(n['succ_rate'])}</td><td>{_pct(n['avg_return'])}</td></tr>"
                    f"<tr><td>변동성</td><td>{v['succ']}</td><td>{v['fail']}</td><td>{v['pending']}</td><td>{v['failed']}</td>"
                    f"<td>{v['total']}</td><td>{_pct(v['succ_rate'])}</td><td>{_pct(v['avg_return'])}</td></tr></table>"
                )
                pred_header = (f"<div><span class='badge {n_cls}'>일반 {_pct(n['succ_rate'])}</span> "
                               f"<span class='badge {v_cls}'>변동성 {_pct(v['succ_rate'])}</span></div>")

                rows = []
                for md in by_model:
                    val_f1_val = md.get("val_f1", None)
                    val_f1_txt = f"{float(val_f1_val):.3f}" if (val_f1_val is not None) else "-"
                    last_cls = md.get("latest_class", "-")
                    last_ret = md.get("latest_return", None)
                    last_ret_txt = "-" if last_ret is None else _pct(last_ret)
                    rows.append("<tr>"
                                f"<td>{md.get('model','')}</td>"
                                f"<td>{val_f1_txt}</td>"
                                f"<td>{md.get('succ',0)}</td>"
                                f"<td>{md.get('fail',0)}</td>"
                                f"<td>{md.get('total',0)}</td>"
                                f"<td>{_pct(md.get('succ_rate',0.0))}</td>"
                                f"<td>{last_cls}</td>"
                                f"<td>{last_ret_txt}</td>"
                                "</tr>")
                model_details = ("<details class='card' style='margin-top:8px'><summary>모델별 상세</summary>"
                                 "<div style='margin-top:6px'>"
                                 "<table><tr><th>모델</th><th>최근 val_f1</th><th>성공</th><th>실패</th><th>총건수</th>"
                                 "<th>성공률</th><th>최근 클래스</th><th>최근 수익률</th></tr>"
                                 + "".join(rows) + "</table></div></details>")

                due = _fmt_ts(_to_kst(ev["due_time"]))
                lastp = _fmt_ts(_to_kst(ev["last_prediction_time"]))
                laste = _fmt_ts(_to_kst(ev["last_evaluated_time"]))
                delay = ev.get("delay_min", 0)
                eval_block = (f"<div class='card' style='margin-top:8px'><div class='step'>3) 평가</div>"
                              f"<div><span class='badge {delay_cls}'>지연 {delay}분</span></div>"
                              f"<div class='muted' style='margin-top:6px'>"
                              f"마지막 예측: {lastp} · 평가 예정: {due} · 최근 평가완료: {laste}</div></div>")

                rr = fl.get("reflect_ratio", None); rr_txt = "-" if rr is None else _pct(rr)
                fail_block = (f"<div class='card' style='margin-top:8px'>"
                              f"<div class='step'>🔁 실패학습</div>"
                              f"<div class='muted'>최근 실패 {fl['recent_fail']}건 / 이후반영 {fl['reflected_count_after']}건 / 반영률 {rr_txt}</div>"
                              f"</div>")

                strat_problems = blk.get("problems") or []
                prob_block = ""
                if strat_problems:
                    lis = "".join([f"<li>{p}</li>" for p in strat_problems])
                    prob_block = (f"<div class='card' style='margin-top:8px'><div class='step'>⚠️ 문제</div>"
                                  f"<ul style='margin:6px 0 0 18px'>{lis}</ul></div>")

                sym_cards.append(
                    "<div class='card'>"
                    f"{head}"
                    "<div class='step'>1) 학습</div>"
                    f"<div class='muted small'>최근 학습시각: {_fmt_ts(_to_kst(blk['last_train_time']))}</div>"
                    "<div class='hr'></div>"
                    "<div class='step'>2) 예측</div>"
                    f"{pred_header}{pred_table}"
                    f"{model_details}"
                    f"{eval_block}{fail_block}{prob_block}"
                    "</div>"
                )
            parts.append(f"<div class='card'><h2 id='{sym}'>📈 {sym}</h2>{fs_html}{''.join(sym_cards) if sym_cards else '<div class=\"muted\">전략 데이터가 없습니다.</div>'}</div>")
        return "<div id='view-symbol' class='view'>" + "".join(parts) + "</div>"

    # ===== 작동순서(전략 → 모델 → 심볼) 뷰 =====
    def render_flow_view():
        by_strategy = {st: [] for st in STRATEGIES}
        for sym_item in snapshot.get("symbols", []):
            for strat, blk in (sym_item.get("strategies") or {}).items():
                by_strategy[strat].append((sym_item.get("symbol"), blk))

        sections = []
        for strat in STRATEGIES:
            items = by_strategy.get(strat, [])
            # 상단: 전략 요약(모델별 합산)
            agg = {mt: {"succ":0,"fail":0,"total":0,"val_f1":None} for mt in MODEL_TYPES}
            rows = []
            for sym, blk in items:
                for md in blk["prediction"]["by_model"]:
                    mt = md["model"]; agg[mt]["succ"] += md["succ"]; agg[mt]["fail"] += md["fail"]; agg[mt]["total"] += md["total"]
                    if md.get("val_f1") is not None:
                        agg[mt]["val_f1"] = md["val_f1"]
            for mt in MODEL_TYPES:
                a = agg[mt]
                rate = (a["succ"]/max(1,a["total"])) if a["total"]>0 else 0.0
                rows.append("<tr>"
                            f"<td>{mt}</td>"
                            f"<td>{('-' if a['val_f1'] is None else f'{a['val_f1']:.3f}')}</td>"
                            f"<td>{a['succ']}</td><td>{a['fail']}</td><td>{a['total']}</td>"
                            f"<td>{_pct(rate)}</td>"
                            "</tr>")
            model_summary = ("<div class='card'><div class='row-title'>전략 요약 — 모델별</div>"
                             "<table><tr><th>모델</th><th>최근 val_f1(샘플)</th><th>성공</th><th>실패</th><th>총건수</th><th>성공률</th></tr>"
                             + "".join(rows) + "</table></div>")

            sym_cards = []
            for sym, blk in items:
                n = blk["prediction"]["normal"]; v = blk["prediction"]["volatility"]
                by_model = blk["prediction"]["by_model"]; ev = blk["evaluation"]; fl = blk["failure_learning"]
                meta_choice = blk["prediction"].get("meta_choice", "-")
                n_cls, v_cls = _grade_rate(n["succ_rate"]), _grade_rate(v["succ_rate"])
                delay_cls = _delay_badge(ev.get("delay_min", 0))

                head = (f"<div class='row-title'>{sym} · <span class='muted small'>최근 학습 {_fmt_ts(_to_kst(blk['last_train_time']))}</span> "
                        f" · <span class='badge warn'>메타 선택: {meta_choice}</span></div>")

                pred_table = (
                    "<table><tr><th>구분</th><th>성공</th><th>실패</th><th>대기</th><th>기록오류</th><th>총건수</th><th>성공률</th><th>평균수익</th></tr>"
                    f"<tr><td>일반</td><td>{n['succ']}</td><td>{n['fail']}</td><td>{n['pending']}</td><td>{n['failed']}</td>"
                    f"<td>{n['total']}</td><td>{_pct(n['succ_rate'])}</td><td>{_pct(n['avg_return'])}</td></tr>"
                    f"<tr><td>변동성</td><td>{v['succ']}</td><td>{v['fail']}</td><td>{v['pending']}</td><td>{v['failed']}</td>"
                    f"<td>{v['total']}</td><td>{_pct(v['succ_rate'])}</td><td>{_pct(v['avg_return'])}</td></tr></table>"
                )
                pred_header = (f"<div><span class='badge {n_cls}'>일반 {_pct(n['succ_rate'])}</span> "
                               f"<span class='badge {v_cls}'>변동성 {_pct(v['succ_rate'])}</span></div>")

                rows = []
                for md in by_model:
                    val_f1_val = md.get("val_f1", None)
                    val_f1_txt = f"{float(val_f1_val):.3f}" if (val_f1_val is not None) else "-"
                    last_cls = md.get("latest_class", "-")
                    last_ret = md.get("latest_return", None)
                    last_ret_txt = "-" if last_ret is None else _pct(last_ret)
                    rows.append("<tr>"
                                f"<td>{md.get('model','')}</td>"
                                f"<td>{val_f1_txt}</td>"
                                f"<td>{md.get('succ',0)}</td>"
                                f"<td>{md.get('fail',0)}</td>"
                                f"<td>{md.get('total',0)}</td>"
                                f"<td>{_pct(md.get('succ_rate',0.0))}</td>"
                                f"<td>{last_cls}</td>"
                                f"<td>{last_ret_txt}</td>"
                                "</tr>")
                model_table = ("<table><tr><th>모델</th><th>최근 val_f1</th><th>성공</th><th>실패</th><th>총건수</th>"
                               "<th>성공률</th><th>최근 클래스</th><th>최근 수익률</th></tr>"
                               + "".join(rows) + "</table>")

                due = _fmt_ts(_to_kst(ev["due_time"]))
                lastp = _fmt_ts(_to_kst(ev["last_prediction_time"]))
                laste = _fmt_ts(_to_kst(ev["last_evaluated_time"]))
                delay = ev.get("delay_min", 0)
                eval_block = (f"<div class='card' style='margin-top:8px'><div class='step'>3) 평가</div>"
                              f"<div><span class='badge {delay_cls}'>지연 {delay}분</span></div>"
                              f"<div class='muted' style='margin-top:6px'>마지막 예측: {lastp} · 평가 예정: {due} · 최근 평가완료: {laste}</div>"
                              f"</div>")

                rr = fl.get("reflect_ratio", None); rr_txt = "-" if rr is None else _pct(rr)
                fail_block = (f"<div class='card' style='margin-top:8px'><div class='step'>🔁 실패학습</div>"
                              f"<div class='muted'>최근 실패 {fl['recent_fail']}건 / 이후반영 {fl['reflected_count_after']}건 / 반영률 {rr_txt}</div></div>")

                strat_problems = blk.get("problems") or []
                prob_block = ""
                if strat_problems:
                    lis = "".join([f"<li>{p}</li>" for p in strat_problems])
                    prob_block = (f"<div class='card' style='margin-top:8px'><div class='step'>⚠️ 문제</div>"
                                  f"<ul style='margin:6px 0 0 18px'>{lis}</ul></div>")

                sym_cards.append("<div class='card'>"
                                 "<div class='step'>1) 학습</div>"
                                 f"{head}"
                                 "<div class='hr'></div>"
                                 "<div class='step'>2) 예측</div>"
                                 f"{pred_header}{pred_table}"
                                 "<details class='card' style='margin-top:8px'><summary>모델별 상세</summary>"
                                 f"{model_table}</details>"
                                 f"{eval_block}{fail_block}{prob_block}"
                                 "</div>")
            sections.append(f"<div class='card'><h2>🧭 전략: {strat}</h2>{model_summary}{''.join(sym_cards) if sym_cards else '<div class=\"muted\">데이터 없음</div>'}</div>")
        return "<div id='view-flow' class='view'>" + "".join(sections) + "</div>"

    html = f"<div class='wrap'>{css}{header}" + render_flow_view() + render_symbol_centric() + "</div>"
    return html

# ===================== 외부진입점 =====================
def run(group=-1, view="json", cumulative=True, symbols=None, **kwargs):
    """
    ✅ 절대 학습/예측/평가 실행 없음 — 점검 전용 스냅샷
    사용법:
      /diag/e2e?view=json
      /diag/e2e?view=html
      /diag/e2e?symbols=BTCUSDT,ETHUSDT
    """
    try:
        snapshot = _build_snapshot(symbols_filter=symbols)
        if view == "html":
            return _render_html(snapshot)
        else:
            return snapshot
    except Exception as e:
        err = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        if view == "html":
            return f"<pre>{json.dumps(err, ensure_ascii=False, indent=2)}</pre>"
        return err
