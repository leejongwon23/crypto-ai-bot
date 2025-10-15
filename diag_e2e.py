# === diag_e2e.py (관우 v2.7 — 모델별 예상수익률·확률·포지션 표기 추가) ===
import os, json, traceback, re
import pandas as pd
import pytz
from collections import defaultdict, Counter
from datetime import datetime

PERSIST_DIR   = "/persistent"
LOG_DIR       = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR     = os.path.join(PERSIST_DIR, "models")
PREDICTION_LOG= os.path.join(PERSIST_DIR, "prediction_log.csv")
TRAIN_LOG     = os.path.join(LOG_DIR, "train_log.csv")              # ← 원본 경로 유지
AUDIT_LOG     = os.path.join(LOG_DIR, "evaluation_audit.csv")

KST = pytz.timezone("Asia/Seoul")
now_kst = lambda: pd.Timestamp.now(tz="Asia/Seoul")

EVAL_HORIZON_HOURS = {"단기": 4, "중기": 24, "장기": 168}
STRATEGIES  = ["단기", "중기", "장기"]
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]

# (심볼 소스: data.utils 우선 → config → 유니온)
try:
    from data.utils import SYMBOLS as DATA_SYMBOLS
except Exception:
    DATA_SYMBOLS = None
try:
    from config import get_SYMBOLS
    CONFIG_SYMBOLS = get_SYMBOLS()
except Exception:
    CONFIG_SYMBOLS = None

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

def _normalize_ts_series_kst(s):
    if s is None:
        return pd.Series([], dtype="datetime64[ns, Asia/Seoul]")
    s2 = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s2.dt, "tz", None) is None:
            s2 = s2.dt.tz_localize("Asia/Seoul")
        else:
            s2 = s2.dt.tz_convert("Asia/Seoul")
        return s2
    except Exception:
        return s2.apply(_to_kst)

def _pct(x):
    try: return f"{float(x)*100:.1f}%"
    except: return "0.0%"

def _fmt_ts(ts): return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "없음"

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
    try: return float(x)
    except: return default

def _num_or_none(x):
    try:
        v = float(x)
        if pd.isna(v): return None
        return v
    except:
        return None

# ===================== 인벤토리 스캔(.pt + .meta.json) =====================
_NAME_RE = re.compile(
    r"^(?P<sym>[^_]+)_(?P<strat>단기|중기|장기)_(?P<model>lstm|cnn_lstm|transformer)"
    r"(?:_group(?P<gid>\d+))?(?:_cls(?P<ncls>\d+))?$"
)

def _parse_filename_base(fname):
    base = os.path.basename(fname)
    base = re.sub(r"\.(pt|meta\.json)$", "", base)
    m = _NAME_RE.match(base)
    if not m: return None
    d = m.groupdict()
    return {
        "symbol": d["sym"],
        "strategy": d["strat"],
        "model": d["model"],
        "group_id": int(d["gid"]) if d.get("gid") else 0,
        "num_classes": int(d["ncls"]) if d.get("ncls") else None,
        "base": base
    }

def _list_inventory():
    """
    models 디렉토리에서 *.pt 와 *.meta.json 모두 스캔해 합침.
    - key: (symbol, strategy, model)
    - has_pt / has_meta / val_f1 / saved_at(메타 timestamp) / group_id / num_classes
    """
    inv = {}
    if not os.path.isdir(MODEL_DIR): return inv

    # 1) 메타부터 스캔
    for fn in os.listdir(MODEL_DIR):
        if not fn.endswith(".meta.json"): continue
        meta_path = os.path.join(MODEL_DIR, fn)
        p = _parse_filename_base(fn)
        if not p: continue
        key = (p["symbol"], p["strategy"], p["model"])
        val = inv.get(key, {
            "symbol": p["symbol"], "strategy": p["strategy"], "model": p["model"],
            "group_id": p["group_id"], "num_classes": p["num_classes"],
            "has_pt": False, "has_meta": False,
            "pt_file": None, "meta_file": None,
            "val_f1": None, "saved_at": None
        })
        val["has_meta"] = True
        val["meta_file"] = fn
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {}) or {}
            if "val_f1" in metrics:
                val["val_f1"] = float(metrics.get("val_f1"))
            ts_raw = meta.get("timestamp") or meta.get("saved_at")
            ts_kst = _to_kst(ts_raw)
            val["saved_at"] = ts_kst.isoformat() if ts_kst else None
            val["num_classes"] = val["num_classes"] or meta.get("num_classes")
            if isinstance(val["num_classes"], str) and val["num_classes"].isdigit():
                val["num_classes"] = int(val["num_classes"])
        except Exception:
            pass
        inv[key] = val

    # 2) pt 스캔
    for fn in os.listdir(MODEL_DIR):
        if not fn.endswith(".pt"): continue
        p = _parse_filename_base(fn)
        if not p: continue
        key = (p["symbol"], p["strategy"], p["model"])
        val = inv.get(key, {
            "symbol": p["symbol"], "strategy": p["strategy"], "model": p["model"],
            "group_id": p["group_id"], "num_classes": p["num_classes"],
            "has_pt": False, "has_meta": False,
            "pt_file": None, "meta_file": None,
            "val_f1": None, "saved_at": None
        })
        val["has_pt"] = True
        val["pt_file"] = fn
        inv[key] = val

    return inv

# ===================== 보조 집계 =====================
def _summarize_fail_patterns(df_pred_sym):
    try:
        df = df_pred_sym.copy()
        if "status" in df.columns:
            df = df[df["status"].isin(["fail","v_fail"])]
        elif "success" in df.columns:
            df = df[df["success"].astype(str).str.lower().isin(["false","0","no"])]
        else:
            return []
        if df.empty: return []
        reasons = df.get("reason")
        if reasons is None: return []
        top = Counter([str(x).split("|")[0].strip() for x in reasons.dropna().tolist()]).most_common(3)
        return [f"{r} ({c})" for r,c in top]
    except Exception:
        return []

def _expected_tuples(symbols):
    return {(s, st, mt) for s in symbols for st in STRATEGIES for mt in MODEL_TYPES}

def _progress(inv_map, symbols):
    have = set([k for k in inv_map.keys()])  # (sym, strat, model)
    need = _expected_tuples(symbols)
    return {
        "expected": len(need),
        "have": len(need & have),
        "missing": sorted(list(need - have))
    }

# ===================== 스냅샷 집계 =====================
def _build_snapshot(symbols_filter=None):
    df_pred  = _safe_read_csv(PREDICTION_LOG)
    df_train = _safe_read_csv(TRAIN_LOG)
    _        = _safe_read_csv(AUDIT_LOG)   # (옵션)

    # 타임스템프 KST 통일
    if "timestamp" in df_pred.columns:
        df_pred["timestamp"] = _normalize_ts_series_kst(df_pred["timestamp"])
    else:
        df_pred["timestamp"] = pd.NaT
    if "timestamp" in df_train.columns:
        df_train["timestamp"] = _normalize_ts_series_kst(df_train["timestamp"])
    else:
        df_train["timestamp"] = pd.NaT

    # 인벤토리
    inv = _list_inventory()  # key=(sym,strat,model) → dict
    inv_keys = set(inv.keys())

    # 심볼 목록 결정
    if symbols_filter:
        symbols = [s.strip() for s in symbols_filter.split(",") if s.strip()]
    elif DATA_SYMBOLS:
        symbols = list(DATA_SYMBOLS)
    elif CONFIG_SYMBOLS:
        symbols = list(CONFIG_SYMBOLS)
    else:
        symbols = set([k[0] for k in inv_keys])
        if "symbol" in df_pred.columns:
            symbols |= set(df_pred["symbol"].dropna().astype(str).tolist())
        if "symbol" in df_train.columns:
            symbols |= set(df_train["symbol"].dropna().astype(str).tolist())
        symbols = sorted([s for s in symbols if s and s != "nan"])

    # 학습로그: 최근 f1/최근 학습시각
    train_last_map = {}
    train_f1_map   = {}
    if not df_train.empty:
        try:
            df_train = df_train.sort_values("timestamp")
            for (sym, st), g in df_train.groupby(["symbol","strategy"], dropna=False):
                ts = g["timestamp"].iloc[-1]
                train_last_map[(str(sym), str(st))] = ts
            if "model" in df_train.columns and "f1" in df_train.columns:
                for (sym, st, mdl), g in df_train.groupby(["symbol","strategy","model"], dropna=False):
                    try:
                        last_f1 = float(pd.to_numeric(g["f1"], errors="coerce").dropna().iloc[-1])
                        m = re.search(r"(lstm|cnn_lstm|transformer)", str(mdl))
                        if m:
                            mt = m.group(1)
                            train_f1_map[(str(sym), str(st), mt)] = last_f1
                    except Exception:
                        pass
        except Exception:
            pass

    snapshot = {
        "time": now_kst().isoformat(),
        "symbols": [],
        "progress": _progress(inv, symbols),
    }

    # === RealityGuard 표시 텍스트 생성기 ===
    def _rg_text_from_row(row):
        try:
            mu = row.get("rg_mu", None)
            lo = row.get("rg_lo", None)
            hi = row.get("rg_hi", None)
            if mu is None and lo is None and hi is None:
                return None  # RG 정보 없음
            def f(v):
                try: return float(v)
                except: return None
            mu, lo, hi = f(mu), f(lo), f(hi)
            if lo is not None and hi is not None and mu is not None:
                return f"{lo*100:.2f}% ~ {hi*100:.2f}% (μ {mu*100:.2f}%)"
            if lo is not None and hi is not None:
                return f"{lo*100:.2f}% ~ {hi*100:.2f}%"
            if mu is not None:
                return f"μ {mu*100:.2f}%"
            return None
        except Exception:
            return None

    # 심볼 단위 집계
    for sym in symbols:
        sym_block = {"symbol": sym, "strategies": {}, "fail_summary": []}
        df_ps = df_pred[df_pred["symbol"] == sym] if "symbol" in df_pred.columns else pd.DataFrame()
        sym_block["fail_summary"] = _summarize_fail_patterns(df_ps)

        for strat in STRATEGIES:
            last_train_ts = train_last_map.get((sym, strat), pd.NaT)
            df_ss = df_ps[df_ps["strategy"] == strat] if not df_ps.empty else pd.DataFrame()

            def _stat_count(df, label):
                if df.empty or "status" not in df.columns: return 0
                return int((df["status"] == label).sum())

            if not df_ss.empty:
                df_ss = df_ss.copy()
                if "status" in df_ss.columns:
                    df_ss["is_vol"] = df_ss["status"].astype(str).str.startswith("v_")
                elif "volatility" in df_ss.columns:
                    df_ss["is_vol"] = df_ss["volatility"].astype(str).str.lower().isin(["1","true"])
                else:
                    df_ss["is_vol"] = False
                ret_series = None
                for col in ["return","return_value","rate"]:
                    if col in df_ss.columns:
                        ret_series = pd.to_numeric(df_ss[col], errors="coerce")
                        break
                if ret_series is None:
                    ret_series = pd.Series(0.0, index=df_ss.index)
                df_ss["_return_val"] = ret_series.fillna(0.0)

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

            # 모델별 상세(인벤토리 + 예측로그)
            models_detail = []
            inventory_rows = []
            for mt in MODEL_TYPES:
                key = (sym, strat, mt)
                inv_item = inv.get(key)
                df_model = df_ss[df_ss["model"].astype(str).str.contains(mt, na=False)] if not df_ss.empty else pd.DataFrame()

                # 최신 클래스/수익률/확률/포지션/RealityGuard 텍스트
                def _latest_for_model(dfm):
                    if dfm.empty:
                        return {
                            "cls": "-",
                            "rate": None,
                            "rg_text": "",
                            "class_text": None,
                            "prob": None,
                            "prob_src": None,
                            "position": None
                        }
                    try: dfm = dfm.sort_values("timestamp")
                    except Exception: pass
                    last = dfm.iloc[-1]

                    # 클래스
                    latest_cls = "-"
                    for k in ["predicted_class","class","pred_class","label"]:
                        if k in dfm.columns:
                            v = last.get(k, None)
                            if pd.notna(v):
                                latest_cls = str(int(v)) if str(v).isdigit() else str(v)
                                break

                    # 예상 수익률
                    latest_rate = None
                    if "rate" in dfm.columns: latest_rate = _num_or_none(last.get("rate"))
                    elif "return" in dfm.columns: latest_rate = _num_or_none(last.get("return"))
                    elif "return_value" in dfm.columns: latest_rate = _num_or_none(last.get("return_value"))

                    # 구간 텍스트
                    class_text = None
                    if "class_return_text" in dfm.columns:
                        try:
                            txt = str(last.get("class_return_text","")).strip()
                            if txt: class_text = txt
                        except Exception:
                            pass

                    # 확률
                    prob = None; prob_src = None
                    cp = last.get("calib_prob", None)
                    rp = last.get("raw_prob", None)
                    if cp is not None and str(cp)!="nan":
                        prob = _num_or_none(cp); prob_src = "calib"
                    elif rp is not None and str(rp)!="nan":
                        prob = _num_or_none(rp); prob_src = "raw"

                    # 포지션
                    pos = last.get("position", None)
                    pos = str(pos) if pos is not None and str(pos).strip() not in ["nan","None",""] else None

                    # RealityGuard 우선 텍스트
                    rg_text = _rg_text_from_row(last)
                    return {
                        "cls": latest_cls,
                        "rate": latest_rate,
                        "rg_text": rg_text,
                        "class_text": class_text,
                        "prob": prob,
                        "prob_src": prob_src,
                        "position": pos
                    }

                latest = _latest_for_model(df_model)

                val_f1 = None
                if inv_item and inv_item.get("val_f1") is not None:
                    val_f1 = float(inv_item["val_f1"])
                elif (sym, strat, mt) in train_f1_map:
                    val_f1 = float(train_f1_map[(sym, strat, mt)])

                if inv_item is None:
                    status = "MISSING"
                else:
                    has_pt   = inv_item.get("has_pt", False)
                    has_meta = inv_item.get("has_meta", False)
                    if not has_meta and has_pt:
                        status = "ERROR_META"
                    elif df_model is not None and len(df_model) > 0:
                        status = "PREDICTED"
                    else:
                        status = "TRAINED_NO_PRED"

                md = {
                    "model": mt,
                    "status": status,
                    "val_f1": val_f1,
                    "succ": _stat_count(df_model, "success") + _stat_count(df_model, "v_success"),
                    "fail": _stat_count(df_model, "fail") + _stat_count(df_model, "v_fail"),
                    "total": int(len(df_model)),
                    "latest_class": latest["cls"],
                    "latest_return": latest["rate"],            # 숫자값
                    "latest_return_text": latest["rg_text"] or latest["class_text"],  # 표시용(우선 RG)
                    "latest_prob": latest["prob"],
                    "latest_prob_src": latest["prob_src"],
                    "latest_position": latest["position"],
                }
                denom = max(1, md["total"]); md["succ_rate"] = md["succ"] / denom
                models_detail.append(md)

                inv_row = {
                    "model": mt,
                    "has_pt": bool(inv_item and inv_item.get("has_pt")),
                    "has_meta": bool(inv_item and inv_item.get("has_meta")),
                    "val_f1": val_f1,
                    "saved_at": inv_item.get("saved_at") if inv_item else None,
                    "pt_file": inv_item.get("pt_file") if inv_item else None,
                    "meta_file": inv_item.get("meta_file") if inv_item else None,
                    "status": status
                }
                inventory_rows.append(inv_row)

            last_pred_ts_raw = df_ss["timestamp"].max() if "timestamp" in df_ss.columns and not df_ss.empty else pd.NaT
            last_pred_ts = _to_kst(last_pred_ts_raw)
            eval_due = _eval_deadline(last_pred_ts, strat) if last_pred_ts is not None else None

            last_eval_ts = None
            if not df_ss.empty:
                df_eval = df_ss.copy()
                try:
                    src_col = df_eval["source"].astype(str) if "source" in df_eval.columns else pd.Series("", index=df_eval.index)
                    dir_col = df_eval["direction"].astype(str) if "direction" in df_eval.columns else pd.Series("", index=df_eval.index)
                    cond = (src_col == "평가") | dir_col.str.startswith("평가:")
                    df_eval = df_eval[cond]
                except Exception:
                    df_eval = pd.DataFrame()
                if not df_eval.empty:
                    last_eval_ts = _to_kst(df_eval["timestamp"].max())

            delayed_min = 0
            if eval_due is not None and last_eval_ts is not None:
                delayed_min = int(max(0, (last_eval_ts - eval_due) / pd.Timedelta(minutes=1)))
            elif eval_due is not None and last_eval_ts is None:
                now = now_kst()
                delayed_min = int(max(0, (now - eval_due) / pd.Timedelta(minutes=1))) if now > eval_due else 0

            # 메타 선택 표시(note JSON에서 추출)
            meta_choice = "-"
            try:
                df_meta = df_ss[df_ss["model"] == "meta"]
                if not df_meta.empty and "note" in df_meta.columns:
                    last_note = str(df_meta.sort_values("timestamp").iloc[-1].get("note","") or "")
                    if last_note.strip().startswith("{"):
                        meta_choice = json.loads(last_note).get("meta_choice", meta_choice)
            except Exception:
                pass

            strat_problems = []
            strat_notes = []
            inv_count = sum(1 for mt in MODEL_TYPES if (sym, strat, mt) in inv)
            if inv_count == 0:
                strat_problems.append("모델 파일 없음")
            for mt in MODEL_TYPES:
                item = inv.get((sym, strat, mt))
                if item and item.get("has_pt") and not item.get("has_meta"):
                    strat_problems.append(f"{mt} 메타 누락")
            if df_ss.empty:
                if inv_count > 0 or pd.notna(last_train_ts):
                    strat_notes.append("예측 대기(훈련 완료)")
                else:
                    strat_problems.append("예측 기록 없음")
            if delayed_min > 0:
                strat_problems.append(f"평가 지연 {delayed_min}분")
            if pd.isna(last_train_ts):
                strat_notes.append("최근 학습 기록 없음")

            recent_fail = df_ss[df_ss["status"].isin(["fail","v_fail"])] if "status" in df_ss.columns else pd.DataFrame()
            recent_fail_n = int(len(recent_fail)); reflected = 0
            if recent_fail_n > 0 and "timestamp" in df_ss.columns:
                last_fail_time = _to_kst(recent_fail["timestamp"].max())
                after = df_ss[df_ss["timestamp"] > last_fail_time] if last_fail_time is not None else pd.DataFrame()
                reflected = int((after["status"].isin(["success","v_success"])).sum()) if "status" in after.columns else 0
            reflect_ratio = (reflected / max(1, recent_fail_n)) if recent_fail_n>0 else None

            # === 누적(일반+변동성) 집계 ===
            cum_succ    = summary_n["succ"] + summary_v["succ"]
            cum_fail    = summary_n["fail"] + summary_v["fail"]
            cum_pending = summary_n["pending"] + summary_v["pending"]
            cum_failed  = summary_n["failed"] + summary_v["failed"]
            cum_total   = (summary_n["total"] + summary_v["total"])
            denom_sf    = max(1, (cum_succ + cum_fail))  # 성공률 분모: 성공+실패
            cum_rate    = cum_succ / denom_sf

            sym_block["strategies"][strat] = {
                "last_train_time": last_train_ts.isoformat() if pd.notna(last_train_ts) else None,
                "inventory": {
                    "rows": inventory_rows,
                    "trained_models": inv_count,
                    "missing_models": [mt for mt in MODEL_TYPES if (sym, strat, mt) not in inv]
                },
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
                    "by_model": models_detail,
                    "meta_choice": meta_choice,
                    "cumulative": {                            # ★ NEW
                        "succ": cum_succ,
                        "fail": cum_fail,
                        "pending": cum_pending,
                        "failed": cum_failed,
                        "total": cum_total,
                        "succ_rate": cum_rate,                 # 성공/(성공+실패)
                        "sf_denominator": denom_sf             # (성공+실패)
                    }
                },
                "evaluation": {
                    "last_prediction_time": last_pred_ts.isoformat() if last_pred_ts is not None else None,
                    "due_time": eval_due.isoformat() if eval_due is not None else None,
                    "last_evaluated_time": last_eval_ts.isoformat() if last_eval_ts is not None else None,
                    "delay_min": delayed_min
                },
                "failure_learning": {
                    "recent_fail": recent_fail_n,
                    "reflected_count_after": reflected,
                    "reflect_ratio": reflect_ratio
                },
                "problems": strat_problems,
                "notes": strat_notes
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
        "models_count": len(inv),
        "problems": problems,
        "progress": snapshot["progress"]
    }
    return snapshot

# ===================== HTML 렌더 =====================
def _render_html(snapshot):
    def _safe(s):
        try: return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        except: return str(s)

    def icon_train(last_train_iso): return "✅" if last_train_iso else "❌"
    def icon_ret(r):
        if r is None: return "⏺"
        try: r = float(r)
        except: return "⏺"
        if r > 1e-9: return "✅"
        if r < -1e-9: return "❌"
        return "⏺"
    def icon_delay(mins): return "⏰⚠️" if mins and mins>0 else "⏰✅"

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
  ul { margin: 4px 0 6px 20px; }
</style>
"""
    sm = snapshot.get("summary", {})
    pr = sm.get("progress", {}) or {}
    problems = sm.get("problems", []) or []
    status_class = "ok" if not problems else "err"
    status_text  = "🟢 전체 정상" if not problems else f"🔴 문제 {len(problems)}건"
    idx_links = []
    for sym_item in snapshot.get("symbols", []):
        sym = sym_item.get("symbol","")
        if sym: idx_links.append(f"<a href='#{_safe(sym)}'>{_safe(sym)}</a>")
    idx_html = "<div class='index'>" + "".join(idx_links) + "</div>" if idx_links else ""

    header = f"""
<div class="sticky-top mono">
  <div><b>YOPO 통합 점검</b> <span class="kicker">— 시스템 상태를 한 눈에</span></div>
  <div class="small">생성시각 {snapshot.get('time','')}</div>
  <div style="margin-top:6px)">
    <span class="badge {status_class}">{status_text}</span>
    <span class="pill">일반 성공률 {_pct(sm.get('normal_success_rate',0))}</span>
    <span class="pill">변동성 성공률 {_pct(sm.get('vol_success_rate',0))}</span>
    <span class="pill">심볼 {sm.get('symbols_count',0)}개</span>
    <span class="pill">인벤토리 {sm.get('models_count',0)}건</span>
    <span class="pill">진행률 {pr.get('have',0)}/{pr.get('expected',0)}</span>
  </div>
  <div class="legend" style="margin-top:6px">
    <span class="badge ok">성공률 양호 ≥60%</span>
    <span class="badge warn">보통 40~60%</span>
    <span class="badge err">주의 &lt;40% / 평가 지연</span>
  </div>
  <div class="controls">
    <button class="btn" onclick="switchView('flow')">작동순서 리스트</button>
    <button class="btn" onclick="switchView('symbol')">심볼 카드</button>
    <button class="btn" onclick="toggleAll(true)">모두 펼치기</button>
    <button class="btn" onclick="toggleAll(false)">모두 접기</button>
  </div>
  {idx_html}
</div>
<script>
function toggleAll(open) {{ document.querySelectorAll('details').forEach(d => d.open = open); }}
function switchView(which) {{
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + which).classList.add('active');
}}
window.addEventListener('DOMContentLoaded', () => switchView('flow'));
</script>
"""

    # ===== (A) 심볼 카드 =====
    def render_symbol_centric():
        parts = []
        for sym_item in snapshot.get("symbols", []):
            sym = sym_item.get("symbol")
            fs = sym_item.get("fail_summary", []) or []
            fs_html = f"<div class='muted small'>최근 실패 패턴: {', '.join(map(_safe,fs))}</div>" if fs else ""
            sym_cards = []
            for strat, blk in (sym_item.get("strategies") or {}).items():
                n = blk["prediction"]["normal"]; v = blk["prediction"]["volatility"]
                cum = blk["prediction"]["cumulative"]  # ★ NEW
                by_model = blk["prediction"]["by_model"]; ev = blk["evaluation"]; fl = blk["failure_learning"]
                inv_rows = blk.get("inventory", {}).get("rows", [])
                notes = blk.get("notes", []) or []
                meta_choice = blk["prediction"].get("meta_choice", "-")
                n_cls, v_cls = _grade_rate(n["succ_rate"]), _grade_rate(v["succ_rate"])
                cum_cls = _grade_rate(cum.get("succ_rate",0.0))
                delay_cls = _delay_badge(ev.get("delay_min", 0))

                head = (f"<div class='row-title'>전략: <b>{_safe(strat)}</b> &nbsp;"
                        f"<span class='muted small'>최근 학습 {_safe(_fmt_ts(_to_kst(blk['last_train_time'])))}</span> &nbsp;"
                        f"<span class='badge warn'>🎯 메타 선택: {_safe(meta_choice)}</span></div>")

                pred_table = (
                    "<table><tr><th>구분</th><th>성공</th><th>실패</th><th>대기</th><th>기록오류</th>"
                    "<th>총건수</th><th>성공률</th><th>평균수익</th></tr>"
                    f"<tr><td>일반</td><td>{n['succ']}</td><td>{n['fail']}</td><td>{n['pending']}</td><td>{n['failed']}</td>"
                    f"<td>{n['total']}</td><td>{_pct(n['succ_rate'])}</td><td>{_pct(n['avg_return'])}</td></tr>"
                    f"<tr><td>변동성</td><td>{v['succ']}</td><td>{v['fail']}</td><td>{v['pending']}</td><td>{v['failed']}</td>"
                    f"<td>{v['total']}</td><td>{_pct(v['succ_rate'])}</td><td>{_pct(v['avg_return'])}</td></tr>"
                    f"<tr><td><b>누적</b></td><td><b>{cum['succ']}</b></td><td><b>{cum['fail']}</b></td>"
                    f"<td>{cum['pending']}</td><td>{cum['failed']}</td>"
                    f"<td><b>{cum['total']}</b></td>"
                    f"<td><b>{_pct(cum['succ_rate'])} ({cum['succ']}/{cum['sf_denominator']})</b></td>"
                    f"<td>-</td></tr>"
                    "</table>"
                )
                pred_header = (f"<div>"
                               f"<span class='badge {n_cls}'>일반 {_pct(n['succ_rate'])}</span> "
                               f"<span class='badge {v_cls}'>변동성 {_pct(v['succ_rate'])}</span> "
                               f"<span class='badge {cum_cls}'>누적 {_pct(cum['succ_rate'])} ({cum['succ']}/{cum['sf_denominator']})</span>"
                               f"</div>")

                # 모델별 상세(예측 요약) — 컬럼 확장
                rows = []
                for md in by_model:
                    val_f1_val = md.get("val_f1", None)
                    val_f1_txt = f"{float(val_f1_val):.3f}" if (val_f1_val is not None) else "-"
                    last_cls = md.get("latest_class", "-")
                    last_rate = md.get("latest_return", None)
                    rate_txt = "-" if last_rate is None else _pct(last_rate)
                    prob = md.get("latest_prob", None)
                    prob_txt = "-" if prob is None else _pct(prob)
                    pos = md.get("latest_position", None) or "-"
                    # 구간/RealityGuard 텍스트
                    rg_or_class = md.get("latest_return_text") or "-"
                    rows.append("<tr>"
                                f"<td>{_safe(md.get('model',''))}</td>"
                                f"<td>{_safe(md.get('status','-'))}</td>"
                                f"<td>{val_f1_txt}</td>"
                                f"<td>{md.get('succ',0)}</td>"
                                f"<td>{md.get('fail',0)}</td>"
                                f"<td>{md.get('total',0)}</td>"
                                f"<td>{_pct(md.get('succ_rate',0.0))}</td>"
                                f"<td>{_safe(last_cls)}</td>"
                                f"<td>{_safe(rg_or_class)}</td>"
                                f"<td>{_safe(rate_txt)}</td>"
                                f"<td>{_safe(prob_txt)}</td>"
                                f"<td>{_safe(pos)}</td>"
                                "</tr>")
                model_details = ("<details class='card' style='margin-top:8px'><summary>모델별 상세(예측/성능)</summary>"
                                 "<div style='margin-top:6px'>"
                                 "<table><tr><th>모델</th><th>상태</th><th>최근 val_f1</th>"
                                 "<th>성공</th><th>실패</th><th>총건수</th><th>성공률</th>"
                                 "<th>최근 클래스</th><th>구간(RG/클래스)</th><th>예상수익률</th><th>확률</th><th>포지션</th></tr>"
                                 + "".join(rows) + "</table></div></details>")

                # 파일 인벤토리
                inv_rows_html = []
                for r in inv_rows:
                    f1txt = "-" if r.get("val_f1") is None else f"{float(r['val_f1']):.3f}"
                    saved = _fmt_ts(_to_kst(r.get("saved_at")))
                    inv_rows_html.append("<tr>"
                        f"<td>{_safe(r['model'])}</td>"
                        f"<td>{'✅' if r['has_pt'] else '❌'}</td>"
                        f"<td>{'✅' if r['has_meta'] else '❌'}</td>"
                        f"<td>{f1txt}</td>"
                        f"<td>{_safe(saved)}</td>"
                        f"<td>{_safe(r.get('status','-'))}</td>"
                    "</tr>")
                inv_table = ("<details class='card' style='margin-top:8px'><summary>파일 인벤토리(PT/Meta)</summary>"
                             "<div style='margin-top:6px'>"
                             "<table><tr><th>모델</th><th>PT</th><th>Meta</th><th>최근 val_f1</th><th>저장시각</th><th>상태</th></tr>"
                             + "".join(inv_rows_html) + "</table></div></details>")

                due = _fmt_ts(_to_kst(ev["due_time"]))
                lastp = _fmt_ts(_to_kst(ev["last_prediction_time"]))
                laste = _fmt_ts(_to_kst(ev["last_evaluated_time"]))
                delay = ev.get("delay_min", 0)
                eval_block = (f"<div class='card' style='margin-top:8px'><div class='step'>3) 평가</div>"
                              f"<div><span class='badge {delay_cls}'>{icon_delay(delay)} 지연 {delay}분</span></div>"
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
                    lis = "".join([f"<li>{_safe(p)}</li>" for p in strat_problems])
                    prob_block = (f"<div class='card' style='margin-top:8px'><div class='step'>⚠️ 문제</div>"
                                  f"<ul style='margin:6px 0 0 18px'>{lis}</ul></div>")
                notes = blk.get("notes") or []
                note_block = ""
                if notes:
                    lis = "".join([f"<li>{_safe(p)}</li>" for p in notes])
                    note_block = (f"<div class='card' style='margin-top:8px'><div class='step'>📝 노트</div>"
                                  f"<ul style='margin:6px 0 0 18px'>{lis}</ul></div>")

                sym_cards.append(
                    "<div class='card'>"
                    f"{head}"
                    "<div class='step'>1) 학습</div>"
                    f"<div class='muted small'>{icon_train(blk['last_train_time'])} 최근 학습시각: {_safe(_fmt_ts(_to_kst(blk['last_train_time'])))}</div>"
                    "<div class='hr' style='height:1px;background:#e5e7eb;margin:10px 0'></div>"
                    "<div class='step'>2) 예측</div>"
                    f"{pred_header}{pred_table}"
                    f"{model_details}{inv_table}"
                    f"{eval_block}{fail_block}{note_block}{prob_block}"
                    "</div>"
                )
            body_html = ''.join(sym_cards) if sym_cards else '<div class="muted">전략 데이터가 없습니다.</div>'
            parts.append(f"<div class='card'><h2 id='{_safe(sym)}'>📈 {_safe(sym)}</h2>{fs_html}{body_html}</div>")
        return "<div id='view-symbol' class='view'>" + "".join(parts) + "</div>"

    # ===== (B) 작동순서 리스트 =====
    def render_flow_list():
        out = []
        out.append("<div class='card'><h2>📊 YOPO 운영 현황 (리스트)</h2>")
        out.append(f"<div class='muted small'>🕒 생성시각: {_safe(snapshot.get('time',''))}</div>")

        pr = snapshot.get("progress", {}) or {}
        if pr:
            miss_list = pr.get("missing", [])[:10]
            miss_txt = ", ".join([f"{a}/{b}/{c}" for (a,b,c) in miss_list]) if miss_list else "없음"
            out.append("<div class='muted small' style='margin:6px 0'>"
                       f"진행률: {pr.get('have',0)}/{pr.get('expected',0)}"
                       f" · 미싱(상위 10): { _safe(miss_txt) }</div>")

        # 1) 학습
        out.append("<h3 style='margin-top:8px'>1. 학습 현황</h3>")
        for strat in STRATEGIES:
            out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div><ul>")
            for sym_item in snapshot.get("symbols", []):
                sym = sym_item.get("symbol")
                blk = (sym_item.get("strategies") or {}).get(strat)
                if not blk: continue
                last_train = blk.get("last_train_time")
                out.append(f"<li>{_safe(sym)}<ul>")
                out.append(f"<li>{icon_train(last_train)} 최근 학습: {_safe(_fmt_ts(_to_kst(last_train)))}</li>")
                out.append("</ul></li>")
            out.append("</ul>")

        # 2) 예측
        out.append("<h3 style='margin-top:8px'>2. 예측 현황</h3>")
        for strat in STRATEGIES:
            out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div><ul>")
            for sym_item in snapshot.get("symbols", []):
                sym = sym_item.get("symbol")
                blk = (sym_item.get("strategies") or {}).get(strat)
                if not blk: continue
                pred = blk.get("prediction") or {}
                cum = pred.get("cumulative") or {"succ":0,"fail":0,"sf_denominator":1,"succ_rate":0.0}
                out.append(f"<li>{_safe(sym)}<ul>")
                out.append(f"<li>🎯 메타러너 선택: <b>{_safe(pred.get('meta_choice','-'))}</b></li>")
                for md in pred.get("by_model", []):
                    last_cls = md.get("latest_class","-")
                    rg_or_class = md.get("latest_return_text") or "-"
                    rate = md.get("latest_return", None)
                    rate_txt = "-" if rate is None else f"{rate:+.1%}"
                    prob = md.get("latest_prob", None)
                    prob_txt = "-" if prob is None else f"{prob:.1%}"
                    pos = md.get("latest_position", None) or "-"
                    out.append(f"<li>{icon_ret(rate)} {_safe(md.get('model','').upper())}: {_safe(md.get('status','-'))}, "
                               f"클래스 {_safe(last_cls)} · 구간 {_safe(rg_or_class)} · 예상 {rate_txt} · 확률 {prob_txt} · 포지션 {pos}</li>")
                out.append(f"<li>🧮 누적: 성공 {cum['succ']} / 실패 {cum['fail']} · 성공률 {_pct(cum['succ_rate'])} ({cum['succ']}/{cum['sf_denominator']})</li>")
                out.append("</ul></li>")
            out.append("</ul>")

        # 3) 평가
        out.append("<h3 style='margin-top:8px'>3. 평가 현황</h3>")
        for strat in STRATEGIES:
            out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div><ul>")
            for sym_item in snapshot.get("symbols", []):
                sym = sym_item.get("symbol")
                blk = (sym_item.get("strategies") or {}).get(strat)
                if not blk: continue
                ev = blk.get("evaluation") or {}
                delay = int(ev.get("delay_min",0))
                out.append(f"<li>{_safe(sym)}<ul>")
                out.append(f"<li>🕒 마지막 예측: {_safe(_fmt_ts(_to_kst(ev.get('last_prediction_time'))))}</li>")
                out.append(f"<li>📅 평가 예정: {_safe(_fmt_ts(_to_kst(ev.get('due_time'))))}</li>")
                out.append(f"<li>🧪 최근 평가완료: {_safe(_fmt_ts(_to_kst(ev.get('last_evaluated_time'))))}</li>")
                out.append(f"<li>{icon_delay(delay)} 지연: {delay}분</li>")
                out.append("</ul></li>")
            out.append("</ul>")

        # 4) 실패 학습
        out.append("<h3 style='margin-top:8px'>4. 실패 학습 현황</h3>")
        for strat in STRATEGIES:
            out.append(f"<div style='margin:6px 0 2px'><b>• 전략: {_safe(strat)}</b></div><ul>")
            for sym_item in snapshot.get("symbols", []):
                sym = sym_item.get("symbol")
                blk = (sym_item.get("strategies") or {}).get(strat)
                if not blk: continue
                fl = blk.get("failure_learning") or {}
                rr = fl.get("reflect_ratio", None)
                rr_txt = "-" if rr is None else _pct(rr)
                out.append(f"<li>{_safe(sym)}<ul>")
                out.append(f"<li>📉 최근 실패 {int(fl.get('recent_fail',0))}건</li>")
                out.append(f"<li>📈 이후 반영 {int(fl.get('reflected_count_after',0))}건</li>")
                out.append(f"<li>📘 반영률 {rr_txt}</li>")
                fs = sym_item.get("fail_summary") or []
                if fs:
                    out.append("<li>🧾 최근 실패 패턴:<ul>")
                    for r in fs:
                        out.append(f"<li>{_safe(r)}</li>")
                    out.append("</ul></li>")
                out.append("</ul></li>")
            out.append("</ul>")

        out.append("</div>")
        return "<div id='view-flow' class='view'>" + "".join(out) + "</div>"

    html = f"<div class='wrap'>{css}{header}" + render_flow_list() + render_symbol_centric() + "</div>"
    return html

# ===================== 외부 진입 =====================
def run(group=-1, view="json", cumulative=True, symbols=None, **kwargs):
    """
    ✅ 스냅샷만 생성 (학습/예측/평가 실행 없음)
    사용:
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
