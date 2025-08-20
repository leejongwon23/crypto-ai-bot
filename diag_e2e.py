# === diag_e2e.py (관우 v2.2-final: 메타선택 표시 + 문제진단 강화 + 가독성 향상 + 작동순서 리스트 뷰 + 모델별 최신 클래스/수익률 + 아이콘) ===
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
# ⚠️ 출력(HTML)만 사진과 같은 불릿 리스트 스타일 + 아이콘. 데이터 로직은 그대로 유지.
def _render_html(snapshot):
    def _safe(s):
        try:
            return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        except:
            return str(s)

    def icon_train(last_train_iso):
        return "✅" if last_train_iso else "❌"

    def icon_ret(r):
        if r is None: return "⏺"
        try:
            r = float(r)
        except:
            return "⏺"
        if r > 1e-9: return "✅"
        if r < -1e-9: return "❌"
        return "⏺"

    def icon_delay(mins):
        return "⏰⚠️" if mins and mins>0 else "⏰✅"

    out = []
    out.append("<div style='font-family:-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Noto Sans KR, Arial; line-height:1.65; padding:8px 14px'>")
    out.append("<h2>📊 YOPO 운영 현황 (관우 한글 버전 예시)</h2>")
    out.append(f"<div style='color:#6b7280;font-size:12px'>🕒 생성시각: {_safe(snapshot.get('time',''))}</div>")

    # 1) 학습 현황
    out.append("<h3>1. 학습 현황</h3>")
    for strat in STRATEGIES:
        out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div>")
        out.append("<ul>")
        for sym_item in snapshot.get("symbols", []):
            sym = sym_item.get("symbol")
            blk = (sym_item.get("strategies") or {}).get(strat)
            if not blk:
                continue
            last_train = blk.get("last_train_time")
            icon = icon_train(last_train)
            out.append(f"<li>{_safe(sym)}")
            out.append("<ul>")
            out.append(f"<li>{icon} 최근 학습: {_safe(_fmt_ts(_to_kst(last_train)))}</li>")
            probs = blk.get("problems") or []
            if probs:
                out.append("<li>⚠️ 문제:")
                out.append("<ul>")
                for p in probs:
                    out.append(f"<li>{_safe(p)}</li>")
                out.append("</ul></li>")
            out.append("</ul></li>")
        out.append("</ul>")

    # 2) 예측 현황
    out.append("<h3 style='margin-top:14px'>2. 예측 현황</h3>")
    for strat in STRATEGIES:
        out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div>")
        out.append("<ul>")
        for sym_item in snapshot.get("symbols", []):
            sym = sym_item.get("symbol")
            blk = (sym_item.get("strategies") or {}).get(strat)
            if not blk:
                continue
            pred = blk.get("prediction") or {}
            out.append(f"<li>{_safe(sym)}")
            out.append("<ul>")
            out.append(f"<li>🎯 메타러너 선택: <b>{_safe(pred.get('meta_choice','-'))}</b></li>")
            for md in pred.get("by_model", []):
                last_cls = md.get("latest_class","-")
                last_ret = md.get("latest_return", None)
                last_ret_txt = "-" if last_ret is None else f"{last_ret:+.1%}"
                ir = icon_ret(last_ret)
                out.append(f"<li>{ir} {_safe(md.get('model','').upper())}: 클래스 {_safe(last_cls)} (수익률 {_safe(last_ret_txt)})</li>")
            out.append("</ul></li>")
        out.append("</ul>")

    # 3) 평가 현황
    out.append("<h3 style='margin-top:14px'>3. 평가 현황</h3>")
    for strat in STRATEGIES:
        out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div>")
        out.append("<ul>")
        for sym_item in snapshot.get("symbols", []):
            sym = sym_item.get("symbol")
            blk = (sym_item.get("strategies") or {}).get(strat)
            if not blk:
                continue
            ev = blk.get("evaluation") or {}
            delay_icon = icon_delay(int(ev.get("delay_min",0)))
            out.append(f"<li>{_safe(sym)}")
            out.append("<ul>")
            out.append(f"<li>🕒 마지막 예측: {_safe(_fmt_ts(_to_kst(ev.get('last_prediction_time'))))}</li>")
            out.append(f"<li>📅 평가 예정: {_safe(_fmt_ts(_to_kst(ev.get('due_time'))))}</li>")
            out.append(f"<li>🧪 최근 평가완료: {_safe(_fmt_ts(_to_kst(ev.get('last_evaluated_time'))))}</li>")
            out.append(f"<li>{delay_icon} 지연: {int(ev.get('delay_min',0))}분</li>")
            md_list = (blk.get("prediction") or {}).get("by_model", [])
            if md_list:
                out.append("<li>🧩 모델별:")
                out.append("<ul>")
                for md in md_list:
                    out.append(f"<li>{_safe(md.get('model','').upper())}</li>")
                out.append("</ul></li>")
            out.append("</ul></li>")
        out.append("</ul>")

    # 4) 실패 학습 현황
    out.append("<h3 style='margin-top:14px'>4. 실패 학습 현황</h3>")
    for strat in STRATEGIES:
        out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div>")
        out.append("<ul>")
        for sym_item in snapshot.get("symbols", []):
            sym = sym_item.get("symbol")
            blk = (sym_item.get("strategies") or {}).get(strat)
            if not blk:
                continue
            fl = blk.get("failure_learning") or {}
            rr = fl.get("reflect_ratio", None)
            rr_txt = "-" if rr is None else _pct(rr)
            out.append(f"<li>{_safe(sym)}")
            out.append("<ul>")
            out.append(f"<li>📉 최근 실패 {int(fl.get('recent_fail',0))}건</li>")
            out.append(f"<li>📈 이후 반영 {int(fl.get('reflected_count_after',0))}건</li>")
            out.append(f"<li>📘 반영률 {rr_txt}</li>")
            fs = sym_item.get("fail_summary") or []
            if fs:
                out.append("<li>🧾 최근 실패 패턴:")
                out.append("<ul>")
                for r in fs:
                    out.append(f"<li>{_safe(r)}</li>")
                out.append("</ul></li>")
            out.append("</ul></li>")
        out.append("</ul>")

    out.append("</div>")
    return "\n".join(out)

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
