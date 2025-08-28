# === train.py (STOP-friendly, cooperative cancel + heartbeat/watchdog) ===
import os
def _set_default_thread_env(n: str, v: int):
    if os.getenv(n) is None: os.environ[n] = str(v)
for _n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","TORCH_NUM_THREADS"):
    _set_default_thread_env(_n, int(os.getenv("CPU_THREAD_CAP","2")))

import json, time, tempfile, glob, shutil, gc, threading, traceback
from datetime import datetime
import pytz, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from model_io import convert_pt_to_ptz, save_model
try: torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","2")))
except: pass

_DISABLE_LIGHTNING = os.getenv("DISABLE_LIGHTNING","0")=="1"
_HAS_LIGHTNING=False
if not _DISABLE_LIGHTNING:
    try:
        import pytorch_lightning as pl
        _HAS_LIGHTNING=True
    except: _HAS_LIGHTNING=False

# ✅ 순서제어 래퍼 포함 임포트 (+ reset_group_order 추가)
from data.utils import (
    get_kline_by_strategy, compute_features, create_dataset, SYMBOL_GROUPS,
    should_train_symbol, mark_symbol_trained, ready_for_group_predict, mark_group_predicted,
    reset_group_order
)

from model.base_model import get_model
from feature_importance import compute_feature_importance, save_feature_importance  # 호환 유지
from failure_db import insert_failure_record, ensure_failure_db
import logger
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups, get_class_ranges, set_NUM_CLASSES, STRATEGY_CONFIG
from data_augmentation import balance_classes
from window_optimizer import find_best_window

# --- ssl_pretrain (옵션) ---
try:
    from ssl_pretrain import masked_reconstruction, get_ssl_ckpt_path
except:
    def masked_reconstruction(symbol,strategy,input_size): return None
    def get_ssl_ckpt_path(symbol:str,strategy:str)->str:
        base=os.getenv("SSL_CACHE_DIR","/persistent/ssl_models"); os.makedirs(base,exist_ok=True)
        return f"{base}/{symbol}_{strategy}_ssl.pt"

# --- evo meta learner (옵션) ---
try: from evo_meta_learner import train_evo_meta_loop
except:
    def train_evo_meta_loop(*a,**k): return None

def _safe_print(msg):
    try: print(msg, flush=True)
    except: pass

# ====== 🔔 글로벌 진행 하트비트/워치독 ======
_HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC","10"))          # 주기적 생존 로그
_STALL_WARN_SEC = int(os.getenv("STALL_WARN_SEC","60"))          # 진행지연 경고
_LAST_PROGRESS_TS = time.time()
_LAST_PROGRESS_TAG = "init"
_WATCHDOG_ABORT = threading.Event()

# ▶︎ BG 1회 기동 가드 & 실패DB 준비 플래그
_BG_STARTED = {"meta_fix": False, "failure_train": False, "evo_meta_train": False}
_FAILURE_DB_READY = False

def _progress(tag:str):
    """진행 시점 갱신 + 드문 로그(5s). 진행 발생 시 워치독 abort 래치 자동 해제."""
    global _LAST_PROGRESS_TS, _LAST_PROGRESS_TAG
    now = time.time()
    _LAST_PROGRESS_TS = now
    _LAST_PROGRESS_TAG = tag
    # ▶︎ 새 진행이 감지되면 abort 래치 해제 (핵심 수정)
    if _WATCHDOG_ABORT.is_set():
        _WATCHDOG_ABORT.clear()
        _safe_print(f"🟢 [WATCHDOG] abort cleared on progress → {tag}")
    if (now % 5.0) < 0.1:
        _safe_print(f"📌 progress: {tag}")

def _watchdog_loop(stop_event: threading.Event | None):
    """진행이 오래 멈추면 경고 로그(필요 시 abort flag 세팅)."""
    while True:
        if stop_event is not None and stop_event.is_set(): break
        now = time.time()
        since = now - _LAST_PROGRESS_TS
        if since > _STALL_WARN_SEC:
            _safe_print(f"🟡 [WATCHDOG] {since:.0f}s no progress at '{_LAST_PROGRESS_TAG}'")
            if since > _STALL_WARN_SEC * 2:
                # 래치 세트 (진행 재개 시 _progress에서 자동 해제됨)
                _WATCHDOG_ABORT.set()
                _safe_print("🔴 [WATCHDOG] abort set (hard stall)")
        time.sleep(5)

def _reset_watchdog(reason:str):
    """그룹/심볼 경계 등 안전 구간에서 명시적으로 abort 해제."""
    if _WATCHDOG_ABORT.is_set():
        _WATCHDOG_ABORT.clear()
        _safe_print(f"🟢 [WATCHDOG] abort cleared ({reason})")

def _try_auto_calibration(symbol,strategy,model_name):
    try: import calibration
    except Exception as e: _safe_print(f"[CALIB] skip ({e})"); return
    for fn_name in ("learn_and_save_from_checkpoint","learn_and_save"):
        try:
            fn=getattr(calibration,fn_name,None)
            if callable(fn):
                fn(symbol=symbol,strategy=strategy,model_name=model_name)
                _safe_print(f"[CALIB] {symbol}-{strategy}-{model_name} → {fn_name}")
                return
        except Exception as ce: _safe_print(f"[CALIB] {fn_name} err → {ce}")
    _safe_print("[CALIB] no API → skip")

try:
    _orig_log_training_result=logger.log_training_result
    def _wrapped_log_training_result(symbol,strategy,model="",accuracy=0.0,f1=0.0,loss=0.0,note="",source_exchange="BYBIT",status="success"):
        try: _orig_log_training_result(symbol,strategy,model,accuracy,f1,loss,note,source_exchange,status)
        finally:
            try: _try_auto_calibration(symbol,strategy,model or "")
            except Exception as e: _safe_print(f"[HOOK] calib err → {e}")
    logger.log_training_result=_wrapped_log_training_result
    _safe_print("[HOOK] log_training_result → calib hook on")
except Exception as _e: _safe_print(f"[HOOK] attach fail → {_e}")

def _maybe_run_failure_learn(background=True):
    def _job():
        try: import failure_learn
        except Exception as e: _safe_print(f"[FAIL-LEARN] skip ({e})"); return
        for name in ("mini_retrain","run_once","run"):
            try:
                fn=getattr(failure_learn,name,None)
                if callable(fn): fn(); _safe_print(f"[FAIL-LEARN] {name} done"); return
            except Exception as e: _safe_print(f"[FAIL-LEARN] {name} err → {e}")
        _safe_print("[FAIL-LEARN] no API]")
    (threading.Thread(target=_job,daemon=True).start() if background else _job())
try: _maybe_run_failure_learn(True)
except Exception as _e: _safe_print(f"[FAIL-LEARN] init err] {_e}")

NUM_CLASSES=get_NUM_CLASSES()
FEATURE_INPUT_SIZE=get_FEATURE_INPUT_SIZE()
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR="/persistent/models"; os.makedirs(MODEL_DIR,exist_ok=True)

_MAX_ROWS_FOR_TRAIN=int(os.getenv("TRAIN_MAX_ROWS","1200"))
_BATCH_SIZE=int(os.getenv("TRAIN_BATCH_SIZE","128"))
_NUM_WORKERS=int(os.getenv("TRAIN_NUM_WORKERS","0"))
_PIN_MEMORY=False; _PERSISTENT=False

now_kst=lambda: datetime.now(pytz.timezone("Asia/Seoul"))
training_in_progress={"단기":False,"중기":False,"장기":False}

# ===== ✅ 협조적 취소 유틸 =====
class _ControlledStop(Exception): ...
def _check_stop(ev: threading.Event | None, where:str=""):
    if _WATCHDOG_ABORT.is_set():
        _safe_print(f"[STOP] watchdog abort → {where}")
        raise _ControlledStop()
    if ev is not None and ev.is_set():
        _safe_print(f"[STOP] detected → {where}")
        raise _ControlledStop()

def _atomic_write(path:str,data,mode="wb"):
    d=os.path.dirname(path); os.makedirs(d,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=d,prefix=".tmp_",suffix=".swap")
    try:
        with os.fdopen(fd,mode) as f:
            if "b" in mode:
                f.write(data if isinstance(data,(bytes,bytearray)) else data.encode("utf-8"))
            else: f.write(data)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except: pass

def _log_skip(symbol,strategy,reason):
    logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=reason,status="skipped")
    insert_failure_record({"symbol":symbol,"strategy":strategy,"model":"all","predicted_class":-1,"success":False,"rate":0.0,"reason":reason},feature_vector=[])

def _log_fail(symbol,strategy,reason):
    logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=reason,status="failed")
    insert_failure_record({"symbol":symbol,"strategy":strategy,"model":"all","predicted_class":-1,"success":False,"rate":0.0,"reason":reason},feature_vector=[])

def _strategy_horizon_hours(s:str)->int: return {"단기":4,"중기":24,"장기":168}.get(s,24)

def _future_returns_by_timestamp(df:pd.DataFrame,horizon_hours:int)->np.ndarray:
    if df is None or len(df)==0 or "timestamp" not in df.columns: return np.zeros(0 if df is None else len(df),dtype=np.float32)
    ts=pd.to_datetime(df["timestamp"],errors="coerce")
    close=df["close"].astype(float).values
    high=(df["high"] if "high" in df.columns else df["close"]).astype(float).values
    ts = (ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts).dt.tz_convert("Asia/Seoul")
    out=np.zeros(len(df),dtype=np.float32); H=pd.Timedelta(hours=horizon_hours); j0=0
    for i in range(len(df)):
        t0=ts.iloc[i]; t1=t0+H; j=max(j0,i); mx=high[i]
        while j<len(df) and ts.iloc[j]<=t1:
            if high[j]>mx: mx=high[j]
            j+=1
        # 다음 루프에서 불필요한 재스캔 방지
        j0 = max(j-1, i)
        base=close[i] if close[i]>0 else (close[i]+1e-6)
        out[i]=float((mx-base)/(base+1e-12))
    return out

def _stem(p:str)->str: return os.path.splitext(p)[0]

def _save_model_and_meta(model:nn.Module,path_pt:str,meta:dict):
    stem=_stem(path_pt); weight=stem+".ptz"; save_model(weight, model.state_dict())
    _atomic_write(stem+".meta.json", json.dumps(meta,ensure_ascii=False,separators=(",",":")), mode="w")
    return weight, stem+".meta.json"

def _safe_alias(src:str,dst:str):
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    try:
        if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
    except: pass
    try:
        os.link(src,dst)
    except: shutil.copyfile(src,dst)

def _emit_aliases(model_path:str, meta_path:str, symbol:str, strategy:str, model_type:str):
    ext=os.path.splitext(model_path)[1]
    flat_pt=os.path.join(MODEL_DIR,f"{symbol}_{strategy}_{model_type}{ext}")
    _safe_alias(model_path,flat_pt); _safe_alias(meta_path,_stem(flat_pt)+".meta.json")
    dir_pt=os.path.join(MODEL_DIR,symbol,strategy,f"{model_type}{ext}")
    _safe_alias(model_path,dir_pt); _safe_alias(meta_path,_stem(dir_pt)+".meta.json")

def _archive_old_checkpoints(symbol:str,strategy:str,model_type:str,keep_n:int=1):
    patt_pt=os.path.join(MODEL_DIR,f"{symbol}_{strategy}_{model_type}_group*_cls*.pt")
    patt_ptz=os.path.join(MODEL_DIR,f"{symbol}_{strategy}_{model_type}_group*_cls*.ptz")
    paths=sorted(glob.glob(patt_pt)+glob.glob(patt_ptz), key=lambda p: os.path.getmtime(p), reverse=True)
    if not paths: return
    for p in paths[max(1,int(keep_n)):]:
        try:
            if p.endswith(".pt"):
                ptz=os.path.splitext(p)[0]+".ptz"
                if not os.path.exists(ptz): convert_pt_to_ptz(p, ptz)
                try: os.remove(p)
                except: pass
        except Exception as e: print(f"[ARCHIVE] {os.path.basename(p)} compress fail → {e}")

if _HAS_LIGHTNING:
    class LitSeqModel(pl.LightningModule):
        def __init__(self, base_model:nn.Module, lr:float=1e-3):
            super().__init__(); self.model=base_model; self.criterion=nn.CrossEntropyLoss(); self.lr=lr
        def forward(self,x): return self.model(x)
        def training_step(self,batch,idx):
            xb,yb=batch; logits=self(xb); return self.criterion(logits,yb)
        def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.lr)

    class _HeartbeatAndStop(pl.Callback):
        def __init__(self, ev: threading.Event | None):
            self.ev=ev
        def _hb(self, where:str, batch_idx:int|None=None):
            tag = where if batch_idx is None else f"{where}(b{batch_idx})"
            _progress(f"PL:{tag}")
            if self.ev is not None and self.ev.is_set():
                _safe_print(f"[STOP] PL callback → {tag}")
                raise _ControlledStop()
        def on_train_start(self, trainer, pl_module): self._hb("on_train_start")
        def on_train_epoch_start(self, trainer, pl_module): self._hb(f"on_train_epoch_start(ep{trainer.current_epoch})")
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): self._hb("on_train_batch_start", batch_idx)
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): self._hb("on_train_batch_end", batch_idx)
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): self._hb("on_val_batch_end", batch_idx)

# ⏱ TIMEOUT GUARD (+ heartbeat)
import multiprocessing as _mp
def _run_with_timeout(fn, args=(), kwargs=None, timeout_sec:float=120.0, stop_event: threading.Event | None = None, hb_tag:str|None=None, hb_interval:float=5.0):
    if kwargs is None: kwargs={}
    try:
        ctx=_mp.get_context("spawn")
        q=ctx.Queue(maxsize=1)
        def _worker(q, fn, args, kwargs):
            try:
                out=fn(*args, **kwargs)
                q.put(("ok", out))
            except Exception as e:
                q.put(("err", str(e)))
        p=ctx.Process(target=_worker, args=(q, fn, args, kwargs), daemon=True)
        p.start()
        deadline=time.time()+float(timeout_sec); last_hb=0.0
        while True:
            now=time.time()
            if hb_tag and (now-last_hb)>=hb_interval:
                _progress(hb_tag); last_hb=now
            if stop_event is not None and stop_event.is_set():
                try: p.terminate()
                except: pass
                return ("canceled", None)
            remaining=deadline-now
            if remaining<=0:
                try: p.terminate()
                except: pass
                return ("timeout", None)
            p.join(timeout=min(0.5, max(0.0, remaining)))
            if not p.is_alive(): break
        try:
            status, payload=q.get_nowait()
        except Exception:
            status, payload=("err","no-result")
        return (status, payload)
    except Exception as e:
        res=[None]; err=[None]; done=threading.Event()
        def _t():
            try:
                res[0]=fn(*args, **(kwargs or {}))
            except Exception as ex:
                err[0]=str(ex)
            finally:
                done.set()
        t=threading.Thread(target=_t, daemon=True); t.start()
        deadline=time.time()+float(timeout_sec); last_hb=0.0
        while True:
            now=time.time()
            if hb_tag and (now-last_hb)>=hb_interval:
                _progress(hb_tag); last_hb=now
            if done.wait(timeout=0.25): break
            if stop_event is not None and stop_event.is_set(): return ("canceled", None)
            if now>=deadline: return ("timeout", None)
        return ("ok", res[0]) if err[0] is None else ("err", err[0])

# 🧯 logger.log_class_ranges 타임아웃 래퍼
def _log_class_ranges_safe(symbol, strategy, group_id, class_ranges, note, stop_event: threading.Event | None = None):
    _LOGGER_TIMEOUT = float(os.getenv("LOGGER_TIMEOUT_SEC","10"))
    try:
        status, _ = _run_with_timeout(
            lambda: logger.log_class_ranges(symbol, strategy, group_id=group_id, class_ranges=class_ranges, note=note),
            args=(), kwargs={}, timeout_sec=_LOGGER_TIMEOUT, stop_event=stop_event,
            hb_tag="logger:wait", hb_interval=2.5
        )
        if status != "ok":
            _safe_print(f"[log_class_ranges skip] status={status}")
    except Exception as e:
        _safe_print(f"[log_class_ranges err] {e}")

def train_one_model(symbol, strategy, group_id=None, max_epochs=12, stop_event: threading.Event | None = None):
    global _FAILURE_DB_READY
    res={"symbol":symbol,"strategy":strategy,"group_id":int(group_id or 0),"models":[]}
    try:
        ensure_failure_db(); _FAILURE_DB_READY = True
        _safe_print(f"✅ [train_one_model] {symbol}-{strategy}-g{group_id}")
        _reset_watchdog("enter train_one_model")  # ▶︎ 안전 구간에서 한번 더 해제
        _progress(f"start:{symbol}-{strategy}-g{group_id}")

        _check_stop(stop_event,"before ssl_pretrain")
        try:
            ck=get_ssl_ckpt_path(symbol,strategy)
            if not os.path.exists(ck):
                _safe_print(f"[SSL] start masked_reconstruction → {ck}")
                _ssl_timeout=float(os.getenv("SSL_TIMEOUT_SEC","180"))
                status_ssl, _ = _run_with_timeout(
                    masked_reconstruction,
                    args=(symbol,strategy,FEATURE_INPUT_SIZE),
                    kwargs={}, timeout_sec=_ssl_timeout, stop_event=stop_event,
                    hb_tag="ssl:wait", hb_interval=5.0
                )
                if status_ssl != "ok":
                    _safe_print(f"[SSL] skip ({status_ssl})")
            else: _safe_print(f"[SSL] cache → {ck}")
        except Exception as e: _safe_print(f"[SSL] skip {e}")

        _check_stop(stop_event,"before data fetch")
        _progress("data_fetch")
        df=get_kline_by_strategy(symbol,strategy)
        if df is None or df.empty: _log_skip(symbol,strategy,"데이터 없음"); return res

        try: cfg=STRATEGY_CONFIG.get(strategy,{}) ; _limit=int(cfg.get("limit",300))
        except: _limit=300
        _min_required=max(60,int(_limit*0.90))
        _attrs=getattr(df,"attrs",{}) if df is not None else {}
        augment_needed=bool(_attrs.get("augment_needed", len(df)<_limit))
        enough_for_training=bool(_attrs.get("enough_for_training", len(df)>=_min_required))
        _safe_print(f"[DATA] {symbol}-{strategy} rows={len(df)} limit={_limit} min={_min_required} aug={augment_needed} enough_for_training={enough_for_training}")

        _check_stop(stop_event,"before compute_features")
        _progress("compute_features")
        _feat_timeout=float(os.getenv("FEATURE_TIMEOUT_SEC","120"))
        status, feat = _run_with_timeout(
            compute_features, args=(symbol,df,strategy), kwargs={},
            timeout_sec=_feat_timeout, stop_event=stop_event,
            hb_tag="feature:wait", hb_interval=3.0
        )
        if status != "ok" or feat is None or getattr(feat, "empty", True) or (hasattr(feat,"isnull") and feat.isnull().any().any()):
            reason = "피처 타임아웃" if status=="timeout" else ("피처 취소" if status=="canceled" else f"피처 실패({status})")
            _safe_print(f"[FEATURE] {reason} → 스킵")
            _log_skip(symbol,strategy, reason); return res
        _safe_print(f"[FEATURE] ok shape={getattr(feat,'shape',None)}"); _progress("feature_ok")

        _progress("class_ranges:get")
        try:
            class_ranges=get_class_ranges(symbol=symbol,strategy=strategy,group_id=group_id)
            _progress("class_ranges:ok")
        except Exception as e:
            _log_fail(symbol,strategy,"클래스 계산 실패"); return res

        num_classes=len(class_ranges); set_NUM_CLASSES(num_classes)
        if not class_ranges or len(class_ranges)<2:
            try:
                _log_class_ranges_safe(symbol,strategy,group_id=group_id,class_ranges=class_ranges or [],note="train_skip(<2 classes)", stop_event=stop_event)
                logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: g={group_id}, cls<2",status="skipped")
            except: pass
            return res

        _progress("after_class_ranges")
        _log_class_ranges_safe(symbol,strategy,group_id=group_id,class_ranges=class_ranges,note="train_one_model", stop_event=stop_event)
        try: _safe_print(f"[RANGES] {symbol}-{strategy}-g{group_id} → {class_ranges}")
        except Exception as e: _safe_print(f"[log_class_ranges err] {e}")

        H=_strategy_horizon_hours(strategy)
        _progress("future:calc")
        _fto=float(os.getenv("FUTURE_TIMEOUT_SEC","60"))
        status_fut, future = _run_with_timeout(
            _future_returns_by_timestamp, args=(df,), kwargs={"horizon_hours":H},
            timeout_sec=_fto, stop_event=stop_event, hb_tag="future:wait", hb_interval=5.0
        )
        if status_fut != "ok" or future is None or len(future)==0:
            reason = "미래수익 타임아웃" if status_fut=="timeout" else ("미래수익 취소" if status_fut=="canceled" else f"미래수익 실패({status_fut})")
            _safe_print(f"[FUTURE] {reason} → 스킵")
            _log_skip(symbol,strategy,reason); return res
        _progress("future:ok")

        try:
            fg=future[np.isfinite(future)]
            if fg.size>0:
                q=np.nanpercentile(fg,[0,25,50,75,90,95,99])
                _safe_print(f"[RET] {symbol}-{strategy}-g{group_id} min={q[0]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p95={q[5]:.4f} max={np.nanmax(fg):.4f}")
                try:
                    logger.log_return_distribution(symbol,strategy,group_id=group_id,horizon_hours=int(H),
                        summary={"min":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),"p75":float(q[3]),"p90":float(q[4]),"p95":float(q[5]),"p99":float(q[6]),"max":float(np.nanmax(fg)),"count":int(fg.size)},
                        note="train_one_model")
                except Exception as le: _safe_print(f"[log_return_distribution err] {le}")
        except Exception as e: _safe_print(f"[ret summary err] {e}")

        _check_stop(stop_event,"before labeling")
        _progress("labeling")
        labels=[]; lo0=class_ranges[0][0]; hi_last=class_ranges[-1][1]; clipped_low=clipped_high=unmatched=0
        for r in future:
            if not np.isfinite(r): r=lo0
            if r<lo0: labels.append(0); clipped_low+=1; continue
            if r>hi_last: labels.append(len(class_ranges)-1); clipped_high+=1; continue
            idx=None
            for i,(lo,hi) in enumerate(class_ranges):
                if lo<=r<=hi: idx=i; break
            if idx is None: idx=len(class_ranges)-1 if r>hi_last else 0; unmatched+=1
            labels.append(idx)
        if clipped_low or clipped_high or unmatched:
            _safe_print(f"[LABEL CLIP] low={clipped_low} high={clipped_high} unmatched={unmatched}")
        labels=np.array(labels,dtype=np.int64)

        features_only=feat.drop(columns=["timestamp","strategy"],errors="ignore")
        _check_stop(stop_event,"before scaler")
        feat_scaled=MinMaxScaler().fit_transform(features_only)

        if len(feat_scaled)>_MAX_ROWS_FOR_TRAIN or len(labels)>_MAX_ROWS_FOR_TRAIN:
            cut=min(_MAX_ROWS_FOR_TRAIN,len(feat_scaled),len(labels))
            feat_scaled=feat_scaled[-cut:]; labels=labels[-cut:]

        try: best_window=find_best_window(symbol,strategy,window_list=[20,40],group_id=group_id)
        except: best_window=40
        window=max(5,int(best_window)); window=min(window, max(6,len(feat_scaled)-1))

        _check_stop(stop_event,"before sequence build")
        _progress("seq_build")
        X,y=[],[]
        for i in range(len(feat_scaled)-window):
            if i % 128 == 0:
                _check_stop(stop_event,"seq build")
                _progress(f"seq_build@{i}")
            X.append(feat_scaled[i:i+window]); yi=i+window-1; y.append(labels[yi] if 0<=yi<len(labels) else 0)
        X,y=np.array(X,dtype=np.float32),np.array(y,dtype=np.int64)

        if len(X)<20:
            try:
                _cd_timeout=float(os.getenv("CREATE_DATASET_TIMEOUT_SEC","60"))
                status_ds, ds = _run_with_timeout(
                    create_dataset,
                    args=(feat.to_dict(orient="records"),),
                    kwargs={"window":window,"strategy":strategy,"input_size":FEATURE_INPUT_SIZE},
                    timeout_sec=_cd_timeout, stop_event=stop_event,
                    hb_tag="dataset:wait", hb_interval=3.0
                )
                if status_ds=="ok" and isinstance(ds,tuple) and len(ds)>=2:
                    X_fb,y_fb=ds[0],ds[1]
                elif status_ds=="ok":
                    X_fb,y_fb=ds
                else:
                    X_fb,y_fb=None,None
                if isinstance(X_fb,np.ndarray) and len(X_fb)>0:
                    X,y=X_fb.astype(np.float32),y_fb.astype(np.int64)
                    _safe_print("[DATASET] fallback create_dataset 사용")
            except Exception as e: _safe_print(f"[fallback dataset err] {e}")
        if len(X)<10: _log_skip(symbol,strategy,f"샘플 부족(rows={len(df)}, limit={_limit}, min={_min_required})"); return res

        try:
            if len(X)<200: X,y=balance_classes(X,y,num_classes=num_classes)
        except Exception as e: _safe_print(f"[balance err] {e}")

        for model_type in ["lstm","cnn_lstm","transformer"]:
            _check_stop(stop_event,f"before train {model_type}")
            _progress(f"train:{model_type}:prep")
            base=get_model(model_type,input_size=FEATURE_INPUT_SIZE,output_size=num_classes).to(DEVICE)
            val_len=max(1,int(len(X)*0.2));
            if len(X)-val_len<1: val_len=len(X)-1
            train_X,val_X=X[:-val_len],X[-val_len:]; train_y,val_y=y[:-val_len],y[-val_len:]
            train_loader=DataLoader(
                TensorDataset(torch.tensor(train_X),torch.tensor(train_y)),
                batch_size=_BATCH_SIZE,shuffle=True,
                num_workers=_NUM_WORKERS,pin_memory=_PIN_MEMORY,persistent_workers=_PERSISTENT
            )
            val_loader=DataLoader(
                TensorDataset(torch.tensor(val_X),torch.tensor(val_y)),
                batch_size=_BATCH_SIZE,
                num_workers=_NUM_WORKERS,pin_memory=_PIN_MEMORY,persistent_workers=_PERSISTENT
            )

            total_loss=0.0
            _safe_print(f"🟦 TRAIN begin → {symbol}-{strategy}-g{group_id} [{model_type}] (epochs={max_epochs}, train={len(train_X)}, val={len(val_X)})")

            if _HAS_LIGHTNING:
                lit=LitSeqModel(base,lr=1e-3)
                callbacks=[_HeartbeatAndStop(stop_event)] if stop_event is not None else [_HeartbeatAndStop(None)]
                trainer=pl.Trainer(max_epochs=max_epochs, accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
                                   devices=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
                                   enable_progress_bar=False, callbacks=callbacks)
                trainer.fit(lit,train_dataloaders=train_loader,val_dataloaders=val_loader)
                model=lit.model.to(DEVICE)
                _check_stop(stop_event,f"after PL train {model_type}")
            else:
                model=base; opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
                last_log_ts = time.time()
                for ep in range(max_epochs):
                    _check_stop(stop_event,f"epoch {ep} pre")
                    _progress(f"{model_type}:ep{ep}:start")
                    model.train()
                    for bi,(xb,yb) in enumerate(train_loader):
                        if bi % 16 == 0:
                            _check_stop(stop_event,f"epoch {ep} batch {bi}")
                            _progress(f"{model_type}:ep{ep}:b{bi}")
                        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
                        loss=crit(model(xb), yb)
                        if not torch.isfinite(loss): continue
                        opt.zero_grad(); loss.backward(); opt.step(); total_loss+=float(loss.item())
                    now=time.time()
                    if now-last_log_ts>2:
                        _safe_print(f"   ↳ {model_type} ep{ep+1}/{max_epochs} loss_sum={total_loss:.4f}")
                        last_log_ts=now
                    _progress(f"{model_type}:ep{ep}:end")

            _progress(f"eval:{model_type}")
            model.eval(); preds=[]; lbls=[]
            with torch.no_grad():
                for bi,(xb,yb) in enumerate(val_loader):
                    if bi % 32 == 0: _check_stop(stop_event,f"val batch {bi}"); _progress(f"val_b{bi}")
                    p=torch.argmax(model(xb.to(DEVICE)),dim=1).cpu().numpy()
                    preds.extend(p); lbls.extend(yb.numpy())
            acc=float(accuracy_score(lbls,preds)); f1=float(f1_score(lbls,preds,average="macro"))

            stem=os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group{int(group_id) if group_id is not None else 0}_cls{int(num_classes)}")
            meta={"symbol":symbol,"strategy":strategy,"model":model_type,"group_id":int(group_id) if group_id is not None else 0,"num_classes":int(num_classes),"input_size":int(FEATURE_INPUT_SIZE),"metrics":{"val_acc":acc,"val_f1":f1},"timestamp":now_kst().isoformat(),"model_name":os.path.basename(stem)+".ptz","window":int(window),"recent_cap":int(len(feat_scaled)),"engine":"lightning" if _HAS_LIGHTNING else "manual","data_flags":{"rows":int(len(df)),"limit":int(_limit),"min":int(_min_required),"augment_needed":bool(augment_needed),"enough_for_training":bool(enough_for_training)},"train_loss_sum":float(total_loss)}
            wpath,mpath=_save_model_and_meta(model, stem+".pt", meta)
            _archive_old_checkpoints(symbol,strategy,model_type,keep_n=1)
            _emit_aliases(wpath,mpath,symbol,strategy,model_type)

            logger.log_training_result(symbol, strategy, model=os.path.basename(wpath), accuracy=acc, f1=f1, loss=float(total_loss),
                note=(f"train_one_model(window={window}, cap={len(feat_scaled)}, engine={'lightning' if _HAS_LIGHTNING else 'manual'}, data_flags={{rows:{len(df)},limit:{_limit},min:{_min_required},aug:{int(augment_needed)},enough_for_training:{int(enough_for_training)}}})"),
                source_exchange="BYBIT", status="success")
            res["models"].append({"type":model_type,"acc":acc,"f1":f1,"loss_sum":float(total_loss),"pt":wpath,"meta":mpath})
            _safe_print(f"🟩 TRAIN done [{model_type}] acc={acc:.4f} f1={f1:.4f} → {os.path.basename(wpath)}")

            if torch.cuda.is_available(): torch.cuda.empty_cache()
        _progress("train_one_model:end")
        return res
    except _ControlledStop:
        _safe_print(f"[STOP] train_one_model canceled: {symbol}-{strategy}-g{group_id}")
        return res
    except Exception as e:
        _safe_print(f"[EXC] train_one_model {symbol}-{strategy}-g{group_id} → {e}\n{traceback.format_exc()}")
        _log_fail(symbol,strategy,str(e)); return res

def _prune_caches_and_gc():
    try:
        from cache import CacheManager as CM
        try: before=CM.stats()
        except: before=None
        pruned=CM.prune()
        try: after=CM.stats()
        except: after=None
        _safe_print(f"[CACHE] prune ok: before={before}, after={after}, pruned={pruned}")
    except Exception as e: _safe_print(f"[CACHE] skip ({e})")
    try:
        from safe_cleanup import trigger_light_cleanup
        trigger_light_cleanup()
    except: pass
    try:
        gc.collect()
    except: pass

def _rotate_groups_starting_with(groups, anchor_symbol="BTCUSDT"):
    norm=[list(g) for g in groups]; anchor=None
    for i,g in enumerate(norm):
        if anchor_symbol in g: anchor=i; break
    if anchor is not None and anchor!=0: norm=norm[anchor:]+norm[:anchor]
    if norm and anchor_symbol in norm[0]: norm[0]=[anchor_symbol]+[s for s in norm[0] if s!=anchor_symbol]
    return norm

# 🔎 콜드스타트 감지: 모델(.ptz) 하나라도 있으면 False
def _is_cold_start()->bool:
    try:
        any_flat = bool(glob.glob(os.path.join(MODEL_DIR, "*.ptz")))
        any_tree = bool(glob.glob(os.path.join(MODEL_DIR, "*", "*", "*.ptz")))
        return not (any_flat or any_tree)
    except Exception:
        return True

_PREDICT_TIMEOUT_SEC=float(os.getenv("PREDICT_TIMEOUT_SEC","30"))
def _safe_predict_with_timeout(predict_fn,symbol,strategy,source,model_type=None,timeout=_PREDICT_TIMEOUT_SEC, stop_event: threading.Event | None = None):
    err=[]; done=threading.Event()
    def _run():
        try:
            _safe_print(f"[PREDICT] start {symbol}-{strategy} ({source})")
            predict_fn(symbol, strategy, source=source, model_type=model_type)
            _safe_print(f"[PREDICT] done  {symbol}-{strategy}")
        except Exception as e:
            err.append(e)
        finally:
            done.set()
    t=threading.Thread(target=_run,daemon=True); t.start()
    deadline=time.time()+float(timeout)
    while True:
        if done.wait(timeout=0.25): break
        if stop_event is not None and stop_event.is_set():
            _safe_print(f"[STOP] predict canceled: {symbol}-{strategy}")
            return False
        if time.time()>=deadline:
            _safe_print(f"[TIMEOUT] predict {symbol}-{strategy} {timeout}s → skip"); return False
    if err:
        _safe_print(f"[PREDICT FAIL] {symbol}-{strategy}: {err[0]}"); return False
    return True

def _run_bg_if_not_stopped(name:str, fn, stop_event: threading.Event | None):
    """BG 작업은 최초 1회만 기동 + stop 요청시 미기동 + 실패DB 준비전엔 failure_train 미기동."""
    if stop_event is not None and stop_event.is_set():
        _safe_print(f"[SKIP:{name}] stop during reset"); return
    # 1회 기동 가드
    if _BG_STARTED.get(name, False):
        return
    # 실패학습은 DB 준비 후만
    if name=="failure_train" and not _FAILURE_DB_READY:
        _safe_print("[BG:failure_train] deferred (failure DB not ready yet)")
        return
    _BG_STARTED[name] = True
    th=threading.Thread(target=lambda: (fn()), daemon=True)
    th.start()
    _safe_print(f"[BG:{name}] started (daemon)")

# ⚠️ 변경: ignore_should 플래그 + 콜드스타트 1패스 강제학습
def train_models(symbol_list, stop_event: threading.Event | None = None, ignore_should: bool = False):
    strategies=["단기","중기","장기"]
    for symbol in symbol_list:
        if (not ignore_should) and (not should_train_symbol(symbol)):
            continue
        if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] train_models: early"); return
        trained_any=False
        for strategy in strategies:
            if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] train_models: early(strategy)"); return
            try:
                cr=get_class_ranges(symbol=symbol,strategy=strategy)
                if not cr: raise ValueError("빈 클래스 경계")
                num_classes=len(cr); groups=get_class_groups(num_classes=num_classes); max_gid=len(groups)-1
            except Exception as e: _log_fail(symbol,strategy,f"클래스 계산 실패: {e}"); continue
            for gid in range(max_gid+1):
                if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] train_models: early(group)"); return
                try:
                    gr=get_class_ranges(symbol=symbol,strategy=strategy,group_id=gid)
                    if not gr or len(gr)<2:
                        try:
                            _log_class_ranges_safe(symbol,strategy,group_id=gid,class_ranges=gr or [],note="train_skip(<2 classes)", stop_event=stop_event)
                            logger.log_training_result(symbol,strategy,model=f"group{gid}",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: group_id={gid}, cls<2",status="skipped")
                        except: pass
                        continue
                except Exception as e:
                    try:
                        logger.log_training_result(symbol,strategy,model=f"group{gid}",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: group_id={gid}, 경계계산실패 {e}",status="skipped")
                    except: pass
                    continue
                _reset_watchdog("enter symbol/group")   # ▶︎ 심볼 전환 시 한번 더 안전 해제
                _progress(f"train_models:{symbol}-{strategy}-g{gid}")
                res=train_one_model(symbol,strategy,group_id=gid, stop_event=stop_event)
                if res and isinstance(res,dict) and res.get("models"):
                    trained_any=True
                if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] train_models: after one model"); return
                time.sleep(0.1)
        if trained_any:
            mark_symbol_trained(symbol)

    try:
        import maintenance_fix_meta
        _run_bg_if_not_stopped("meta_fix", maintenance_fix_meta.fix_all_meta_json, stop_event)
    except Exception as e: _safe_print(f"[meta fix skip] {e}")
    try:
        import failure_trainer
        _run_bg_if_not_stopped("failure_train", failure_trainer.run_failure_training, stop_event)
    except Exception as e: _safe_print(f"[failure train skip] {e}")
    try:
        _run_bg_if_not_stopped("evo_meta_train", train_evo_meta_loop, stop_event)
    except Exception as e: _safe_print(f"[evo meta train skip] {e}")

def train_symbol_group_loop(sleep_sec:int=0, stop_event: threading.Event | None = None):
    threading.Thread(target=_watchdog_loop, args=(stop_event,), daemon=True).start()
    _reset_watchdog("loop start")  # ▶︎ 루프 시작시 초기화
    # ✅ 콜드스타트면 첫 패스만 should 체크 무시 + 그룹상태 리셋
    force_full_pass = _is_cold_start()
    if force_full_pass:
        _safe_print("🧪 cold start detected → first pass will ignore should_train_symbol()")
        try:
            reset_group_order(0)
            _safe_print("♻️ group order state reset (cold start)")
        except Exception as e:
            _safe_print(f"[group reset skip] {e}")

    while True:
        if stop_event is not None and stop_event.is_set():
            _safe_print("🛑 stop event set → exit main loop")
            break
        try:
            from predict import predict
            try:
                if hasattr(logger,"ensure_train_log_exists"): logger.ensure_train_log_exists()
            except: pass
            try:
                if hasattr(logger,"ensure_prediction_log_exists"): logger.ensure_prediction_log_exists()
            except: pass

            groups=[list(g) for g in SYMBOL_GROUPS]

            for idx, group in enumerate(groups):
                if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] group loop enter"); break
                _reset_watchdog(f"enter group {idx}")  # ▶︎ 그룹 경계에서도 초기화
                _safe_print(f"🚀 [group] {idx+1}/{len(groups)} → {group}")
                _progress(f"group{idx}:start")

                train_models(group, stop_event=stop_event, ignore_should=force_full_pass)
                if stop_event is not None and stop_event.is_set(): _safe_print("🛑 stop after train → exit"); break

                if ready_for_group_predict():
                    time.sleep(0.1)
                    _safe_print(f"[PREDICT] group {idx+1} begin")
                    for symbol in group:
                        if stop_event is not None and stop_event.is_set(): break
                        for strategy in ["단기","중기","장기"]:
                            if stop_event is not None and stop_event.is_set(): break
                            _safe_predict_with_timeout(predict, symbol, strategy, source="그룹직후", model_type=None, timeout=_PREDICT_TIMEOUT_SEC, stop_event=stop_event)
                    mark_group_predicted()
                    _safe_print(f"[PREDICT] group {idx+1} done")
                else:
                    _safe_print(f"[⏸ 대기] 그룹{idx} 일부 미학습 → 예측 보류")

                _prune_caches_and_gc()
                _progress(f"group{idx}:done")

                if sleep_sec>0:
                    for _ in range(sleep_sec):
                        if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] sleep break"); break
                        time.sleep(1)
                    if stop_event is not None and stop_event.is_set(): break

            _safe_print("✅ group pass done (loop will continue unless stopped)")
            # 콜드스타트 1회전 종료 후 정상 모드로 복귀
            if force_full_pass:
                force_full_pass = False
                _safe_print("🧪 cold start first pass completed → resume normal scheduling")
        except _ControlledStop:
            _safe_print("🛑 cooperative stop inside group loop")
            break
        except Exception as e:
            _safe_print(f"[group loop err] {e}\n{traceback.format_exc()}")

        _safe_print("💓 heartbeat: train loop alive")
        time.sleep(max(1, int(os.getenv("TRAIN_LOOP_IDLE_SEC","3"))))

_TRAIN_LOOP_THREAD: threading.Thread | None = None
_TRAIN_LOOP_STOP: threading.Event | None = None
_TRAIN_LOOP_LOCK=threading.Lock()

def start_train_loop(force_restart:bool=False, sleep_sec:int=0):
    global _TRAIN_LOOP_THREAD,_TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive():
            if not force_restart:
                _safe_print("ℹ️ start_train_loop: already running"); return False
            _safe_print("🛑 restarting..."); stop_train_loop(timeout=30)
        _TRAIN_LOOP_STOP=threading.Event()
        def _runner():
            try: train_symbol_group_loop(sleep_sec=sleep_sec, stop_event=_TRAIN_LOOP_STOP)
            finally: _safe_print("ℹ️ train loop thread exit")
        _TRAIN_LOOP_THREAD=threading.Thread(target=_runner,daemon=True); _TRAIN_LOOP_THREAD.start()
        _safe_print("✅ train loop started"); return True

def stop_train_loop(timeout:int|float|None=30):
    global _TRAIN_LOOP_THREAD,_TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is None or not _TRAIN_LOOP_THREAD.is_alive():
            _safe_print("ℹ️ no loop running"); return True
        if _TRAIN_LOOP_STOP is None:
            _safe_print("⚠️ no stop event"); return False
        _TRAIN_LOOP_STOP.set(); _TRAIN_LOOP_THREAD.join(timeout=timeout)
        if _TRAIN_LOOP_THREAD.is_alive():
            _safe_print("⚠️ stop timeout"); return False
        _TRAIN_LOOP_THREAD=None; _TRAIN_LOOP_STOP=None
        _safe_print("✅ loop stopped"); return True

def request_stop()->bool:
    global _TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_STOP is None: return True
        _TRAIN_LOOP_STOP.set(); return True

def is_loop_running()->bool:
    with _TRAIN_LOOP_LOCK:
        return bool(_TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive())

if __name__=="__main__":
    try: start_train_loop(force_restart=True, sleep_sec=0)
    except Exception as e: _safe_print(f"[MAIN] err: {e}")
