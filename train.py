# === train.py (STOP-friendly, cooperative cancel) ===
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

# ✅ 순서제어 래퍼 포함 임포트
from data.utils import (
    get_kline_by_strategy, compute_features, create_dataset, SYMBOL_GROUPS,
    should_train_symbol, mark_symbol_trained, ready_for_group_predict, mark_group_predicted
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
            if high[j]>mx: mx=high[j]; j+=1
        j0=max(j0,i); base=close[i] if close[i]>0 else (close[i]+1e-6)
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

    # ✅ 리셋 시 즉시 중단을 위한 콜백 (예외로 탈출)
    class _StopOnEvent(pl.Callback):
        def __init__(self, ev: threading.Event): self.ev=ev
        def _maybe_raise(self, where:str):
            if self.ev.is_set():
                _safe_print(f"[STOP] PL callback → {where}")
                raise _ControlledStop()
        def on_train_start(self, trainer, pl_module): self._maybe_raise("on_train_start")
        def on_train_epoch_start(self, trainer, pl_module): self._maybe_raise("on_train_epoch_start")
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): self._maybe_raise(f"on_train_batch_start(b{batch_idx})")
        def on_validation_start(self, trainer, pl_module): self._maybe_raise("on_validation_start")
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx): self._maybe_raise(f"on_val_batch_start(b{batch_idx})")

# ⏱ TIMEOUT GUARD: 안전 실행 헬퍼 (multiprocessing 우선, 실패 시 thread 대체)
import multiprocessing as _mp
def _run_with_timeout(fn, args=(), kwargs=None, timeout_sec:float=120.0, stop_event: threading.Event | None = None):
    """fn(*args, **kwargs)를 별도 프로세스에서 실행하고, timeout_sec 내 결과만 수집.
       ✅ stop_event 가 켜지면 즉시 terminate 하여 대기 차단."""
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
        deadline=time.time()+float(timeout_sec)
        # 🔁 짧은 간격으로 join하며 stop_event 감지
        while True:
            if stop_event is not None and stop_event.is_set():
                try: p.terminate()
                except: pass
                return ("canceled", None)
            remaining=deadline-time.time()
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
        # 최후 fallback: thread 기반 (stop_event 폴링)
        res=[None]; err=[None]; done=threading.Event()
        def _t():
            try:
                res[0]=fn(*args, **(kwargs or {}))
            except Exception as ex:
                err[0]=str(ex)
            finally:
                done.set()
        t=threading.Thread(target=_t, daemon=True); t.start()
        deadline=time.time()+float(timeout_sec)
        while True:
            if done.wait(timeout=0.25): break
            if stop_event is not None and stop_event.is_set(): return ("canceled", None)
            if time.time()>=deadline: return ("timeout", None)
        return ("ok", res[0]) if err[0] is None else ("err", err[0])

def train_one_model(symbol, strategy, group_id=None, max_epochs=12, stop_event: threading.Event | None = None):
    res={"symbol":symbol,"strategy":strategy,"group_id":int(group_id or 0),"models":[]}
    try:
        ensure_failure_db(); print(f"✅ [train_one_model] {symbol}-{strategy}-g{group_id}", flush=True)
        _check_stop(stop_event,"before ssl_pretrain")
        try:
            ck=get_ssl_ckpt_path(symbol,strategy)
            if not os.path.exists(ck): masked_reconstruction(symbol,strategy,FEATURE_INPUT_SIZE)
            else: print(f"[SSL] cache → {ck}", flush=True)
        except Exception as e: print(f"[SSL] skip {e}", flush=True)

        _check_stop(stop_event,"before data fetch")
        df=get_kline_by_strategy(symbol,strategy)
        if df is None or df.empty: _log_skip(symbol,strategy,"데이터 없음"); return res

        try: cfg=STRATEGY_CONFIG.get(strategy,{}) ; _limit=int(cfg.get("limit",300))
        except: _limit=300
        _min_required=max(60,int(_limit*0.90))
        _attrs=getattr(df,"attrs",{}) if df is not None else {}
        augment_needed=bool(_attrs.get("augment_needed", len(df)<_limit))
        enough_for_training=bool(_attrs.get("enough_for_training", len(df)>=_min_required))
        print(f"[DATA] {symbol}-{strategy} rows={len(df)} limit={_limit} min={_min_required} aug={augment_needed} enough={enough_for_training}", flush=True)

        _check_stop(stop_event,"before compute_features")

        # ⏱ TIMEOUT GUARD: compute_features 제한 실행 (stop_event 연동)
        _feat_timeout=float(os.getenv("FEATURE_TIMEOUT_SEC","120"))
        status, feat = _run_with_timeout(compute_features, args=(symbol,df,strategy), kwargs={}, timeout_sec=_feat_timeout, stop_event=stop_event)
        if status != "ok" or feat is None or getattr(feat, "empty", True) or (hasattr(feat,"isnull") and feat.isnull().any().any()):
            reason = "피처 타임아웃" if status=="timeout" else ("피처 취소" if status=="canceled" else f"피처 실패({status})")
            print(f"[FEATURE] {reason} → 스킵", flush=True)
            _log_skip(symbol,strategy, reason)
            return res

        try: class_ranges=get_class_ranges(symbol=symbol,strategy=strategy,group_id=group_id)
        except Exception as e: _log_fail(symbol,strategy,"클래스 계산 실패"); return res

        num_classes=len(class_ranges); set_NUM_CLASSES(num_classes)
        if not class_ranges or len(class_ranges)<2:
            try:
                logger.log_class_ranges(symbol,strategy,group_id=group_id,class_ranges=class_ranges or [],note="train_skip(<2 classes)")
                logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: g={group_id}, cls<2",status="skipped")
            except: pass
            return res
        try:
            logger.log_class_ranges(symbol,strategy,group_id=group_id,class_ranges=class_ranges,note="train_one_model")
            print(f"[RANGES] {symbol}-{strategy}-g{group_id} → {class_ranges}", flush=True)
        except Exception as e: print(f"[log_class_ranges err] {e}", flush=True)

        H=_strategy_horizon_hours(strategy)
        future=_future_returns_by_timestamp(df,horizon_hours=H)
        try:
            fg=future[np.isfinite(future)]
            if fg.size>0:
                q=np.nanpercentile(fg,[0,25,50,75,90,95,99])
                print(f"[RET] {symbol}-{strategy}-g{group_id} min={q[0]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p95={q[5]:.4f} max={np.nanmax(fg):.4f}", flush=True)
                try:
                    logger.log_return_distribution(symbol,strategy,group_id=group_id,horizon_hours=int(H),
                        summary={"min":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),"p75":float(q[3]),"p90":float(q[4]),"p95":float(q[5]),"p99":float(q[6]),"max":float(np.nanmax(fg)),"count":int(fg.size)},
                        note="train_one_model")
                except Exception as le: print(f"[log_return_distribution err] {le}", flush=True)
        except Exception as e: print(f"[ret summary err] {e}", flush=True)

        _check_stop(stop_event,"before labeling")
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
            print(f"[LABEL CLIP] low={clipped_low} high={clipped_high} unmatched={unmatched}", flush=True)
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
        X,y=[],[]
        for i in range(len(feat_scaled)-window):
            if i % 64 == 0: 
                try: _check_stop(stop_event,"seq build")
                except _ControlledStop: break
            X.append(feat_scaled[i:i+window]); yi=i+window-1; y.append(labels[yi] if 0<=yi<len(labels) else 0)
        X,y=np.array(X,dtype=np.float32),np.array(y,dtype=np.int64)

        if len(X)<20:
            try:
                # ⏱ TIMEOUT GUARD: create_dataset 제한 실행 (stop_event 연동)
                _cd_timeout=float(os.getenv("CREATE_DATASET_TIMEOUT_SEC","60"))
                status_ds, ds = _run_with_timeout(
                    create_dataset,
                    args=(feat.to_dict(orient="records"),),
                    kwargs={"window":window,"strategy":strategy,"input_size":FEATURE_INPUT_SIZE},
                    timeout_sec=_cd_timeout,
                    stop_event=stop_event
                )
                if status_ds=="ok" and isinstance(ds,tuple) and len(ds)>=2:
                    X_fb,y_fb=ds[0],ds[1]
                elif status_ds=="ok":
                    X_fb,y_fb=ds
                else:
                    X_fb,y_fb=None,None
                if isinstance(X_fb,np.ndarray) and len(X_fb)>0:
                    X,y=X_fb.astype(np.float32),y_fb.astype(np.int64)
            except Exception as e: print(f"[fallback dataset err] {e}", flush=True)
        if len(X)<10: _log_skip(symbol,strategy,f"샘플 부족(rows={len(df)}, limit={_limit}, min={_min_required})"); return res

        try:
            if len(X)<200: X,y=balance_classes(X,y,num_classes=num_classes)
        except Exception as e: print(f"[balance err] {e}", flush=True)

        for model_type in ["lstm","cnn_lstm","transformer"]:
            _check_stop(stop_event,f"before train {model_type}")
            base=get_model(model_type,input_size=FEATURE_INPUT_SIZE,output_size=num_classes).to(DEVICE)
            val_len=max(1,int(len(X)*0.2)); 
            if len(X)-val_len<1: val_len=len(X)-1
            train_X,val_X=X[:-val_len],X[-val_len:]; train_y,val_y=y[:-val_len],y[-val_len:]
            train_loader=DataLoader(TensorDataset(torch.tensor(train_X),torch.tensor(train_y)),batch_size=_BATCH_SIZE,shuffle=True,num_workers=_NUM_WORKERS,pin_memory=_PIN_MEMORY,persistent_workers=_PERSISTENT)
            val_loader=DataLoader(TensorDataset(torch.tensor(val_X),torch.tensor(val_y)),batch_size=_BATCH_SIZE,num_workers=_NUM_WORKERS,pin_memory=_PIN_MEMORY,persistent_workers=_PERSISTENT)

            total_loss=0.0
            if _HAS_LIGHTNING:
                lit=LitSeqModel(base,lr=1e-3)
                callbacks=[]
                if stop_event is not None: callbacks.append(_StopOnEvent(stop_event))
                trainer=pl.Trainer(max_epochs=max_epochs, accelerator=("gpu" if torch.cuda.is_available() else "cpu"), devices=1, enable_checkpointing=False, logger=False, enable_model_summary=False, enable_progress_bar=False, callbacks=callbacks)
                trainer.fit(lit,train_dataloaders=train_loader,val_dataloaders=val_loader)
                model=lit.model.to(DEVICE)
                _check_stop(stop_event,f"after PL train {model_type}")
            else:
                model=base; opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
                for ep in range(max_epochs):
                    _check_stop(stop_event,f"epoch {ep} pre")
                    model.train()
                    for bi,(xb,yb) in enumerate(train_loader):
                        if bi % 16 == 0:
                            _check_stop(stop_event,f"epoch {ep} batch {bi}")
                        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
                        loss=crit(model(xb), yb)
                        if not torch.isfinite(loss): continue
                        opt.zero_grad(); loss.backward(); opt.step(); total_loss+=float(loss.item())

            model.eval(); preds=[]; lbls=[]
            with torch.no_grad():
                for bi,(xb,yb) in enumerate(val_loader):
                    if bi % 32 == 0: _check_stop(stop_event,f"val batch {bi}")
                    p=torch.argmax(model(xb.to(DEVICE)),dim=1).cpu().numpy()
                    preds.extend(p); lbls.extend(yb.numpy())
            acc=float(accuracy_score(lbls,preds)); f1=float(f1_score(lbls,preds,average="macro"))

            stem=os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group{int(group_id) if group_id is not None else 0}_cls{int(num_classes)}")
            meta={"symbol":symbol,"strategy":strategy,"model":model_type,"group_id":int(group_id) if group_id is not None else 0,"num_classes":int(num_classes),"input_size":int(FEATURE_INPUT_SIZE),"metrics":{"val_acc":acc,"val_f1":f1},"timestamp":now_kst().isoformat(),"model_name":os.path.basename(stem)+".ptz","window":int(window),"recent_cap":int(len(feat_scaled)),"engine":"lightning" if _HAS_LIGHTNING else "manual","data_flags":{"rows":int(len(df)),"limit":int(_limit),"min":int(_min_required),"augment_needed":bool(augment_needed),"enough_for_training":bool(enough_for_training)},"train_loss_sum":float(total_loss)}
            wpath,mpath=_save_model_and_meta(model, stem+".pt", meta)
            _archive_old_checkpoints(symbol,strategy,model_type,keep_n=1)
            _emit_aliases(wpath,mpath,symbol,strategy,model_type)

            logger.log_training_result(symbol, strategy, model=os.path.basename(wpath), accuracy=acc, f1=f1, loss=float(total_loss),
                note=(f"train_one_model(window={window}, cap={len(feat_scaled)}, engine={'lightning' if _HAS_LIGHTNING else 'manual'}, data_flags={{rows:{len(df)},limit:{_limit},min:{_min_required},aug:{int(augment_needed)},enough:{int(enough_for_training)}}})"),
                source_exchange="BYBIT", status="success")
            res["models"].append({"type":model_type,"acc":acc,"f1":f1,"loss_sum":float(total_loss),"pt":wpath,"meta":mpath})

            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return res
    except _ControlledStop:
        _safe_print(f"[STOP] train_one_model canceled: {symbol}-{strategy}-g{group_id}")
        return res
    except Exception as e:
        _log_fail(symbol,strategy,str(e)); return res

def _prune_caches_and_gc():
    try:
        from cache import CacheManager as CM
        try: before=CM.stats()
        except: before=None
        pruned=CM.prune()
        try: after=CM.stats()
        except: after=None
        print(f"[CACHE] prune ok: before={before}, after={after}, pruned={pruned}", flush=True)
    except Exception as e: print(f"[CACHE] skip ({e})", flush=True)
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

_PREDICT_TIMEOUT_SEC=float(os.getenv("PREDICT_TIMEOUT_SEC","30"))
def _safe_predict_with_timeout(predict_fn,symbol,strategy,source,model_type=None,timeout=_PREDICT_TIMEOUT_SEC, stop_event: threading.Event | None = None):
    """예측을 백그라운드에서 실행하고, timeout 동안 250ms 폴링 + stop_event 를 감지하여 즉시 스킵."""
    err=[]; done=threading.Event()
    def _run():
        try: predict_fn(symbol, strategy, source=source, model_type=model_type)
        except Exception as e: err.append(e)
        finally: done.set()
    t=threading.Thread(target=_run,daemon=True); t.start()
    deadline=time.time()+float(timeout)
    while True:
        if done.wait(timeout=0.25): break
        if stop_event is not None and stop_event.is_set():
            _safe_print(f"[STOP] predict canceled: {symbol}-{strategy}")
            return False
        if time.time()>=deadline:
            print(f"[TIMEOUT] predict {symbol}-{strategy} {timeout}s → skip", flush=True); return False
    if err:
        print(f"[PREDICT FAIL] {symbol}-{strategy}: {err[0]}", flush=True); return False
    return True

def _run_bg_if_not_stopped(name:str, fn, stop_event: threading.Event | None):
    """리셋 중이면 스킵, 아니면 데몬 스레드로 비동기 실행하여 학습 루프 종료를 막지 않음"""
    if stop_event is not None and stop_event.is_set():
        print(f"[SKIP:{name}] stop during reset", flush=True); return
    th=threading.Thread(target=lambda: (fn()), daemon=True)
    th.start()
    print(f"[BG:{name}] started (daemon)", flush=True)

# ⚠️ 여기부터 변경: ignore_should 플래그 추가
def train_models(symbol_list, stop_event: threading.Event | None = None, ignore_should: bool = False):
    strategies=["단기","중기","장기"]
    for symbol in symbol_list:
        if (not ignore_should) and (not should_train_symbol(symbol)):
            continue
        if stop_event is not None and stop_event.is_set(): print("[STOP] train_models: early", flush=True); return
        trained_any=False
        for strategy in strategies:
            if stop_event is not None and stop_event.is_set(): print("[STOP] train_models: early(strategy)", flush=True); return
            try:
                cr=get_class_ranges(symbol=symbol,strategy=strategy)
                if not cr: raise ValueError("빈 클래스 경계")
                num_classes=len(cr); groups=get_class_groups(num_classes=num_classes); max_gid=len(groups)-1
            except Exception as e: _log_fail(symbol,strategy,f"클래스 계산 실패: {e}"); continue
            for gid in range(max_gid+1):
                if stop_event is not None and stop_event.is_set(): print("[STOP] train_models: early(group)", flush=True); return
                try:
                    gr=get_class_ranges(symbol=symbol,strategy=strategy,group_id=gid)
                    if not gr or len(gr)<2:
                        try:
                            logger.log_class_ranges(symbol,strategy,group_id=gid,class_ranges=gr or [],note="train_skip(<2 classes)")
                            logger.log_training_result(symbol,strategy,model=f"group{gid}",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: group_id={gid}, cls<2",status="skipped")
                        except: pass
                        continue
                except Exception as e:
                    try:
                        logger.log_training_result(symbol,strategy,model=f"group{gid}",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: group_id={gid}, 경계계산실패 {e}",status="skipped")
                    except: pass
                    continue
                res=train_one_model(symbol,strategy,group_id=gid, stop_event=stop_event)
                if res and isinstance(res,dict) and res.get("models"):
                    trained_any=True
                if stop_event is not None and stop_event.is_set(): print("[STOP] train_models: after one model", flush=True); return
                time.sleep(0.1)  # 더 빠른 중단 반응

        if trained_any:
            mark_symbol_trained(symbol)

    # ✅ 리셋 중이면 무거운 후처리 스킵, 아니면 비동기 실행(루프 종료 방해 금지)
    try:
        import maintenance_fix_meta
        _run_bg_if_not_stopped("meta_fix", maintenance_fix_meta.fix_all_meta_json, stop_event)
    except Exception as e: print(f"[meta fix skip] {e}", flush=True)
    try:
        import failure_trainer
        _run_bg_if_not_stopped("failure_train", failure_trainer.run_failure_training, stop_event)
    except Exception as e: print(f"[failure train skip] {e}", flush=True)
    try:
        _run_bg_if_not_stopped("evo_meta_train", train_evo_meta_loop, stop_event)
    except Exception as e: print(f"[evo meta train skip] {e}", flush=True)

def train_symbol_group_loop(sleep_sec:int=0, stop_event: threading.Event | None = None):
    try:
        from predict import predict
        try:
            if hasattr(logger,"ensure_train_log_exists"): logger.ensure_train_log_exists()
        except: pass
        try:
            if hasattr(logger,"ensure_prediction_log_exists"): logger.ensure_prediction_log_exists()
        except: pass

        # ✅ 그룹 순서/구성: SYMBOL_GROUPS 를 있는 그대로 사용 (정렬/회전 없음)
        groups=[list(g) for g in SYMBOL_GROUPS]

        for idx, group in enumerate(groups):
            if stop_event is not None and stop_event.is_set(): print("[STOP] group loop enter", flush=True); break
            print(f"🚀 [group] {idx+1}/{len(groups)} → {group}", flush=True)

            # ✅ 현재 그룹만 학습(should_train_symbol은 train_models 내부에서 필터)
            train_models(group, stop_event=stop_event, ignore_should=False)
            if stop_event is not None and stop_event.is_set(): print("🛑 stop after train → exit", flush=True); break

            # ✅ 그룹 전 심볼 학습 완료 시에만 예측 → 다음 그룹으로 이동
            if ready_for_group_predict():
                time.sleep(0.1)
                for symbol in group:
                    if stop_event is not None and stop_event.is_set(): break
                    for strategy in ["단기","중기","장기"]:
                        if stop_event is not None and stop_event.is_set(): break
                        _safe_predict_with_timeout(predict, symbol, strategy, source="그룹직후", model_type=None, timeout=_PREDICT_TIMEOUT_SEC, stop_event=stop_event)
                mark_group_predicted()
            else:
                print(f"[⏸ 대기] 그룹{idx} 일부 미학습 → 예측 보류")

            _prune_caches_and_gc()
            if sleep_sec>0:
                for _ in range(sleep_sec):
                    if stop_event is not None and stop_event.is_set(): print("[STOP] sleep break", flush=True); break
                    time.sleep(1)
                if stop_event is not None and stop_event.is_set(): break
        print("✅ group loop done", flush=True)
    except Exception as e: print(f"[group loop err] {e}", flush=True)

_TRAIN_LOOP_THREAD: threading.Thread | None = None
_TRAIN_LOOP_STOP: threading.Event | None = None
_TRAIN_LOOP_LOCK=threading.Lock()

def start_train_loop(force_restart:bool=False, sleep_sec:int=0):
    global _TRAIN_LOOP_THREAD,_TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is not None and _TRAIN_LOOP_THREAD.is_alive():
            if not force_restart:
                print("ℹ️ start_train_loop: already running", flush=True); return False
            print("🛑 restarting...", flush=True); stop_train_loop(timeout=30)
        _TRAIN_LOOP_STOP=threading.Event()
        def _runner():
            try: train_symbol_group_loop(sleep_sec=sleep_sec, stop_event=_TRAIN_LOOP_STOP)
            finally: print("ℹ️ train loop thread exit", flush=True)
        _TRAIN_LOOP_THREAD=threading.Thread(target=_runner,daemon=True); _TRAIN_LOOP_THREAD.start()
        print("✅ train loop started", flush=True); return True

def stop_train_loop(timeout:int|float|None=30):
    global _TRAIN_LOOP_THREAD,_TRAIN_LOOP_STOP
    with _TRAIN_LOOP_LOCK:
        if _TRAIN_LOOP_THREAD is None or not _TRAIN_LOOP_THREAD.is_alive():
            print("ℹ️ no loop running", flush=True); return True
        if _TRAIN_LOOP_STOP is None:
            print("⚠️ no stop event", flush=True); return False
        _TRAIN_LOOP_STOP.set(); _TRAIN_LOOP_THREAD.join(timeout=timeout)
        if _TRAIN_LOOP_THREAD.is_alive():
            print("⚠️ stop timeout", flush=True); return False
        _TRAIN_LOOP_THREAD=None; _TRAIN_LOOP_STOP=None
        print("✅ loop stopped", flush=True); return True

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
    except Exception as e: print(f"[MAIN] err: {e}", flush=True)
