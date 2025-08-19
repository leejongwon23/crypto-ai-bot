# === diag_e2e.py (관우 v2: 심볼/전략/모델 세분화 + 평가시간 표시, 대용량 안전) ===
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
now_kst = lambda: datetime.datetime.now(KST)

# 평가 지평(예측 1건의 평가마감 산출용)
EVAL_HORIZON_HOURS = {"단기": 4, "중기": 24, "장기": 168}
STRATEGIES = ["단기", "중기", "장기"]
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]

# -----------------------
# 유틸
# -----------------------
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

def _list_models():
    """models 폴더의 .pt / .meta.json 파싱 → 심볼/전략/모델/그룹/클래스/val_f1/저장시각"""
    out = []
    if not os.path.isdir(MODEL_DIR): return out
    for fn in os.listdir(MODEL_DIR):
        if not fn.endswith(".pt"): continue
        base = fn[:-3]
        # 예: BTCUSDT_단기_transformer_group2_cls5.pt
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
    """최근 실패 사유 Top3"""
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

# -----------------------
# 메인 집계
# -----------------------
def _build_snapshot(symbols_filter=None):
    # CSV 로딩(1회)
    df_pred = _safe_read_csv(PREDICTION_LOG)
    df_train = _safe_read_csv(TRAIN_LOG)
    df_audit = _safe_read_csv(AUDIT_LOG)

    # 타임스탬프 정규화
    if "timestamp" in df_pred.columns:
        df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], errors="coerce")
        try:
            # tz 보정
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

    # 심볼 목록: 모델에서 + 예측로그에서 수집
    symbols = set([m["symbol"] for m in models])
    if "symbol" in df_pred.columns:
        symbols |= set(df_pred["symbol"].dropna().astype(str).tolist())
    symbols = sorted([s for s in symbols if s and s != "nan"])

    # 필터가 지정되면 제한
    if symbols_filter:
        allow = set([s.strip() for s in symbols_filter.split(",") if s.strip()])
        symbols = [s for s in symbols if s in allow]

    # 모델 인덱스: (sym,strat,model)-> list
    model_index = defaultdict(list)
    for m in models:
        key = (m["symbol"], m["strategy"], m["model"])
        model_index[key].append(m)

    # 심볼 단위 요약 생성
    snapshot = {
        "time": now_kst().isoformat(),
        "symbols": []
    }

    # 빠른 접근 위해 미리 dtype 보정
    for col in ["status","return","model","strategy","symbol","success","reason","predicted_class"]:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col]

    for sym in symbols:
        sym_block = {"symbol": sym, "strategies": {}, "fail_summary": []}
        df_ps = df_pred[df_pred["symbol"] == sym] if "symbol" in df_pred.columns else pd.DataFrame()

        # 실패 패턴 요약(심볼 단위)
        sym_block["fail_summary"] = _summarize_fail_patterns(df_ps)

        for strat in STRATEGIES:
            # 학습 시각
            if not df_train.empty:
                df_ts = df_train[(df_train["symbol"] == sym) & (df_train["strategy"] == strat)]
                last_train_ts = df_ts["timestamp"].max() if "timestamp" in df_ts.columns and not df_ts.empty else pd.NaT
            else:
                last_train_ts = pd.NaT

            # 예측 집계
            df_ss = df_ps[df_ps["strategy"] == strat] if not df_ps.empty else pd.DataFrame()
            # 상태 라벨링
            def _stat_count(df, label):
                if df.empty or "status" not in df.columns: return 0
                return int((df["status"] == label).sum())

            # 변동성 여부 판단
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

            # 성공/실패/대기
            summary_n = {
                "succ": _stat_count(nvol, "success"),
                "fail": _stat_count(nvol, "fail"),
                "pending": _stat_count(nvol, "pending"),
                "failed": _stat_count(nvol, "failed"),
                "total": len(nvol),
                "avg_return": float(nvol["_return_val"].mean()) if not nvol.empty else 0.0
            }
            t = max(1, summary_n["total"])
            summary_n["succ_rate"] = summary_n["succ"]/t

            summary_v = {
                "succ": _stat_count(vol, "v_success"),
                "fail": _stat_count(vol, "v_fail"),
                "pending": _stat_count(vol, "pending"),
                "failed": _stat_count(vol, "failed"),
                "total": len(vol),
                "avg_return": float(vol["_return_val"].mean()) if not vol.empty else 0.0
            }
            tv = max(1, summary_v["total"])
            summary_v["succ_rate"] = summary_v["succ"]/tv

            # 모델별 세분화 (최근 val_f1 + 로그 기반 집계)
            models_detail = []
            for mt in MODEL_TYPES:
                lst = model_index.get((sym, strat, mt), [])
                latest_f1 = None
                if lst:
                    # 가장 최신 메타 기준
                    try:
                        lst_sorted = sorted(lst, key=lambda x: x.get("timestamp") or "", reverse=True)
                        latest_f1 = lst_sorted[0].get("val_f1", None)
                    except Exception:
                        pass

                # 로그에서 모델별 성공/실패
                df_model = df_ss[df_ss["model"].astype(str).str.contains(mt, na=False)]
                md = {
                    "model": mt,
                    "val_f1": float(latest_f1) if latest_f1 is not None else None,
                    "succ": _stat_count(df_model, "success") + _stat_count(df_model, "v_success"),
                    "fail": _stat_count(df_model, "fail") + _stat_count(df_model, "v_fail"),
                    "total": int(len(df_model))
                }
                denom = max(1, md["total"])
                md["succ_rate"] = md["succ"] / denom
                models_detail.append(md)

            # 평가 예정/완료(마지막 예측 기준)
            # 마지막 예측시각
            last_pred_ts = df_ss["timestamp"].max() if "timestamp" in df_ss.columns and not df_ss.empty else pd.NaT
            eval_due = _eval_deadline(last_pred_ts, strat) if pd.notna(last_pred_ts) else None

            # 마지막 “평가 기록” (source=="평가" 또는 direction startswith '평가:')
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

            # 지연 여부
            delayed_min = 0
            if eval_due is not None and last_eval_ts is not None:
                delayed_min = int(max(0, (last_eval_ts - eval_due).total_seconds() // 60))
            elif eval_due is not None and last_eval_ts is None:
                now = now_kst()
                delayed_min = int(max(0, (now - eval_due).total_seconds() // 60)) if now > eval_due else 0

            # 실패학습(해당 심볼·전략) 반영률 추정: 실패→이후 성공 여부로 근사 요약
            recent_fail = df_ss[df_ss["status"].isin(["fail","v_fail"])] if "status" in df_ss.columns else pd.DataFrame()
            recent_fail_n = int(len(recent_fail))
            # 근사치: 실패 이후 시각에 같은 전략으로 성공 기록이 있는지 비율
            reflected = 0
            if recent_fail_n > 0 and "timestamp" in df_ss.columns:
                last_fail_time = recent_fail["timestamp"].max()
                after = df_ss[df_ss["timestamp"] > last_fail_time]
                reflected = int((after["status"].isin(["success","v_success"])).sum()) if "status" in after.columns else 0
            reflect_ratio = (reflected / max(1, recent_fail_n)) if recent_fail_n>0 else None

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
                    "by_model": models_detail
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
                }
            }

        snapshot["symbols"].append(sym_block)

    # 시스템 요약
    total_normal_succ = total_normal_fail = 0
    total_vol_succ = total_vol_fail = 0
    for s in snapshot["symbols"]:
        for strat, blk in s["strategies"].items():
            n = blk["prediction"]["normal"]
            v = blk["prediction"]["volatility"]
            total_normal_succ += n["succ"]; total_normal_fail += n["fail"]
            total_vol_succ += v["succ"]; total_vol_fail += v["fail"]
    def _rate(a,b):
        denom = max(1,(a+b))
        return a/denom
    snapshot["summary"] = {
        "normal_success_rate": _rate(total_normal_succ, total_normal_fail),
        "vol_success_rate": _rate(total_vol_succ, total_vol_fail),
        "symbols_count": len(snapshot["symbols"]),
        "models_count": len(_list_models())
    }
    return snapshot

# -----------------------
# HTML 렌더
# -----------------------
def _render_html(snapshot):
    html = []
    html.append(f"<h2>✅ YOPO 통합 점검 리포트</h2>")
    html.append(f"<div>생성시각: {snapshot['time']}</div>")
    html.append("<hr>")
    # 시스템 요약
    sm = snapshot.get("summary", {})
    html.append(
        f"<div style='font-family:monospace;background:#f6f8ff;padding:8px;border:1px solid #ccd;'>"
        f"<b>📌 시스템 요약</b><br>"
        f"- 일반 성공률: {_pct(sm.get('normal_success_rate',0))} | 변동성 성공률: {_pct(sm.get('vol_success_rate',0))}<br>"
        f"- 심볼 수: {sm.get('symbols_count',0)} | 모델 파일 수: {sm.get('models_count',0)}"
        f"</div>"
    )
    # 심볼 섹션
    for sym_item in snapshot["symbols"]:
        sym = sym_item["symbol"]
        html.append(f"<h3 style='margin-top:18px'>📈 {sym}</h3>")
        # 실패 패턴
        fs = sym_item.get("fail_summary", []) or []
        if fs:
            html.append(f"<div style='color:#b00'>최근 실패 패턴: {', '.join(fs)}</div>")
        # 전략별 카드
        for strat, blk in sym_item["strategies"].items():
            n = blk["prediction"]["normal"]
            v = blk["prediction"]["volatility"]
            by_model = blk["prediction"]["by_model"]
            eval_blk = blk["evaluation"]
            fl = blk["failure_learning"]

            html.append("<div style='border:1px solid #ddd;padding:10px;margin:10px 0'>")
            html.append(f"<b>전략: {strat}</b> | 최근 학습: {_fmt_ts(_to_kst(blk['last_train_time']))}<br>")

            # 예측 요약(일반/변동성)
            html.append("<div style='margin-top:6px'><b>예측 요약</b></div>")
            html.append(
                "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:13px'>"
                "<tr style='background:#f0f0f0'><th>구분</th><th>성공</th><th>실패</th><th>대기</th><th>기록오류</th><th>총건수</th><th>성공률</th><th>평균수익</th></tr>"
                f"<tr><td>일반</td><td>{n['succ']}</td><td>{n['fail']}</td><td>{n['pending']}</td><td>{n['failed']}</td>"
                f"<td>{n['total']}</td><td>{_pct(n['succ_rate'])}</td><td>{_pct(n['avg_return'])}</td></tr>"
                f"<tr><td>변동성</td><td>{v['succ']}</td><td>{v['fail']}</td><td>{v['pending']}</td><td>{v['failed']}</td>"
                f"<td>{v['total']}</td><td>{_pct(v['succ_rate'])}</td><td>{_pct(v['avg_return'])}</td></tr>"
                "</table>"
            )

            # 모델별 상세
            html.append("<div style='margin-top:8px'><b>모델별 상세</b></div>")
            html.append(
                "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:13px'>"
                "<tr style='background:#f9f9f9'><th>모델</th><th>최근 val_f1</th><th>성공</th><th>실패</th><th>총건수</th><th>성공률</th></tr>"
            )
            for md in by_model:
                html.append(
                    f"<tr><td>{md['model']}</td>"
                    f"<td>{(f'{md['val_f1']:.3f}' if md['val_f1'] is not None else '-')}</td>"
                    f"<td>{md['succ']}</td><td>{md['fail']}</td><td>{md['total']}</td>"
                    f"<td>{_pct(md['succ_rate'])}</td></tr>"
                )
            html.append("</table>")

            # 평가 일정
            due = _fmt_ts(_to_kst(eval_blk["due_time"]))
            lastp = _fmt_ts(_to_kst(eval_blk["last_prediction_time"]))
            laste = _fmt_ts(_to_kst(eval_blk["last_evaluated_time"]))
            delay = eval_blk.get("delay_min", 0)
            dtxt = f"{delay}분 지연" if delay>0 else "지연 없음"
            html.append(
                f"<div style='margin-top:8px'><b>평가</b> — "
                f"마지막 예측: {lastp} / 평가 예정: {due} / 최근 평가완료: {laste} / {dtxt}</div>"
            )

            # 실패학습 반영
            rr = fl.get("reflect_ratio", None)
            rr_txt = "-" if rr is None else _pct(rr)
            html.append(
                f"<div><b>실패학습</b> — 최근 실패 {fl['recent_fail']}건 / 이후반영 {fl['reflected_count_after']}건 / 반영률 {rr_txt}</div>"
            )

            html.append("</div>")  # card
    return "\n".join(html)

# -----------------------
# 외부 진입점
# -----------------------
def run(group=-1, view="json", cumulative=True, symbols=None, **kwargs):
    """
    ✅ 절대 학습/예측/평가를 '실행'하지 않는 관우 점검 루트
    사용법:
      /diag/e2e?view=json
      /diag/e2e?view=html
      /diag/e2e?symbols=BTCUSDT,ETHUSDT   # 선택 출력(없으면 전체)
    """
    try:
        snapshot = _build_snapshot(symbols_filter=symbols)

        if view == "html":
            html = _render_html(snapshot)
            return html  # Flask에서 mimetype 지정
        else:
            return snapshot  # dict(JSON 직렬화)
    except Exception as e:
        err = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        if view == "html":
            return f"<pre>{json.dumps(err, ensure_ascii=False, indent=2)}</pre>"
        return err
