# === train.py (STRICT: short->mid->long hard order, cooperative cancel, watchdog, robust predict barrier + SMOKE PREDICT) ===
import os
def _set_default_thread_env(n: str, v: int):
    if os.getenv(n) is None: os.environ[n] = str(v)
for _n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","TORCH_NUM_THREADS"):
    _set_default_thread_env(_n, int(os.getenv("CPU_THREAD_CAP","1")))  # ← default 1

import json, time, tempfile, glob, shutil, gc, threading, traceback, re, random
from datetime import datetime
import pytz, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# >>> ADD: deterministic seeds
def set_global_seed(s:int=20240101):
    os.environ["PYTHONHASHSEED"]=str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    except Exception:
        pass
set_global_seed(int(os.getenv("GLOBAL_SEED","20240101")))

from model_io import convert_pt_to_ptz, save_model
try: torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","1")))  # ← default 1
except: pass

_DISABLE_LIGHTNING = os.getenv("DISABLE_LIGHTNING","0")=="1"
_HAS_LIGHTNING=False
if not _DISABLE_LIGHTNING:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        _HAS_LIGHTNING=True
    except: _HAS_LIGHTNING=False

# ✅ 순서제어 래퍼 포함 임포트
from data.utils import (
    get_kline_by_strategy, compute_features, create_dataset, SYMBOL_GROUPS,
    should_train_symbol, mark_symbol_trained, ready_for_group_predict, mark_group_predicted,
    reset_group_order
)

from model.base_model import get_model
from feature_importance import compute_feature_importance, save_feature_importance
from failure_db import insert_failure_record, ensure_failure_db
import logger
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups, get_class_ranges, set_NUM_CLASSES, STRATEGY_CONFIG, get_QUALITY  # ← ADD get_QUALITY
from data_augmentation import balance_classes
from window_optimizer import find_best_window

# --- ssl_pretrain (옵션) ---
DISABLE_SSL = os.getenv("DISABLE_SSL","1")=="1"  # 기본 비활성(시간 단축)
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
    try:
        # 로그 폭주 방지: QUIET_PROGRESS=1이면 중요 수준만 출력
        if os.getenv("QUIET_PROGRESS","1")=="1":
            if not (isinstance(msg,str) and msg.startswith(("🟩","🟦","✅","🛑","🔴","⚠️","🚀","📌","🟡","🟢","ℹ️","[STOP]","[PREDICT]","[HALT]"))):
                return
        print(msg, flush=True)
    except: 
        pass

# ====== 🔔 하트비트/워치독 ======
_HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC","10"))
_STALL_WARN_SEC = int(os.getenv("STALL_WARN_SEC","60"))
_LAST_PROGRESS_TS = time.time()
_LAST_PROGRESS_TAG = "init"
_WATCHDOG_ABORT = threading.Event()

_BG_STARTED = {"meta_fix": False, "failure_train": False, "evo_meta_train": False}
_FAILURE_DB_READY = False

# === [NEW] 환경 가드 ===
_ENABLE_TRAIN_FAILURE_RECORD = os.getenv("ENABLE_TRAIN_FAILURE_RECORD","0")=="1"  # 기본 차단
_ENABLE_BG_FAILURE_TRAIN     = os.getenv("ENABLE_BG_FAILURE_TRAIN","0")=="1"      # 기본 차단
_ENABLE_EVO_META_BG          = os.getenv("ENABLE_EVO_META_BG","0")=="1"           # 기본 차단

def _progress(tag:str):
    global _LAST_PROGRESS_TS, _LAST_PROGRESS_TAG
    now = time.time()
    _LAST_PROGRESS_TS = now; _LAST_PROGRESS_TAG = tag
    if _WATCHDOG_ABORT.is_set():
        _WATCHDOG_ABORT.clear()
        _safe_print(f"🟢 [WATCHDOG] abort cleared → {tag}")
    # 조용모드에선 progress 스팸 억제
    if os.getenv("QUIET_PROGRESS","1")!="1" and (now % 5.0) < 0.1:
        _safe_print(f"📌 progress: {tag}")

def _watchdog_loop(stop_event: threading.Event | None):
    while True:
        if stop_event is not None and stop_event.is_set(): break
        now = time.time(); since = now - _LAST_PROGRESS_TS
        if since > _STALL_WARN_SEC:
            _safe_print(f"🟡 [WATCHDOG] {since:.0f}s no progress at '{_LAST_PROGRESS_TAG}'")
            if since > _STALL_WARN_SEC * 2:
                _WATCHDOG_ABORT.set()
                _safe_print("🔴 [WATCHDOG] abort set (hard stall)")
        time.sleep(5)

def _reset_watchdog(reason:str):
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
    if _is_cold_start():  # 콜드스타트 방지
        _safe_print("[FAIL-LEARN] cold start → skip")
        return
    if not _ENABLE_BG_FAILURE_TRAIN:
        _safe_print("[FAIL-LEARN] disabled by env → skip")
        return
    def _job():
        try: import failure_learn
        except Exception as e: _safe_print(f"[FAIL-LEARN] skip ({e})"); return
        for name in ("mini_retrain","run_once","run"):
            try:
                fn=getattr(failure_learn,name,None)
                if callable(fn): fn(); _safe_print(f"[FAIL-LEARN] {name} done"); return
            except Exception as e: _safe_print(f"[FAIL-LEARN] {name} err] → {e}")
        _safe_print("[FAIL-LEARN] no API]")
    (threading.Thread(target=_job,daemon=True).start() if background else _job())
try: _maybe_run_failure_learn(True)
except Exception as _e: _safe_print(f"[FAIL-LEARN] init err] {_e}")

# [ADD] failure DB를 모듈 로드시 보장 (예측이 선행되어도 안전)
try:
    ensure_failure_db(); _FAILURE_DB_READY = True
except Exception as _e:
    _safe_print(f"[FAILURE_DB] init err → {_e}")

NUM_CLASSES=get_NUM_CLASSES()
FEATURE_INPUT_SIZE=get_FEATURE_INPUT_SIZE()
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR="/persistent/models"; os.makedirs(MODEL_DIR,exist_ok=True)

_MAX_ROWS_FOR_TRAIN=int(os.getenv("TRAIN_MAX_ROWS","1200"))
_BATCH_SIZE=int(os.getenv("TRAIN_BATCH_SIZE","128"))
_NUM_WORKERS=int(os.getenv("TRAIN_NUM_WORKERS","0"))  # 기본 0 (멀티워커 금지)
_PIN_MEMORY=False; _PERSISTENT=False

# 전략별 에폭(기본: 단6/중8/장10)
def _epochs_for(strategy:str)->int:
    if strategy=="단기": return int(os.getenv("EPOCHS_SHORT","6"))
    if strategy=="중기": return int(os.getenv("EPOCHS_MID","8"))
    if strategy=="장기": return int(os.getenv("EPOCHS_LONG","10"))
    return 8

# === SMART TRAIN switches ===
SMART_TRAIN = os.getenv("SMART_TRAIN","1")=="1"
LABEL_SMOOTH = float(os.getenv("LABEL_SMOOTH","0.05"))   # 0.0~0.1 권장
GRAD_CLIP = float(os.getenv("GRAD_CLIP_NORM","1.0"))     # 0.0(해제)~2.0 권장

now_kst=lambda: datetime.now(pytz.timezone("Asia/Seoul"))

# ✅ 예측 게이트 폴백
try:
    from predict import open_predict_gate, close_predict_gate
except Exception:
    def open_predict_gate(*args, **kwargs): return None
    def close_predict_gate(*args, **kwargs): return None

# ===== 협조 취소 =====
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

def _fsync_dir(d: str):
    try:
        if not os.path.isdir(d): return
        fd = os.open(d, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except Exception: pass

def _disk_barrier(paths: list[str]):
    # 강제 플러시(가능한 경우)
    for p in paths:
        try:
            if os.path.isdir(p):
                _fsync_dir(p)
            elif os.path.exists(p):
                with open(p, "rb") as f:
                    os.fsync(f.fileno())
        except Exception: pass
    try:
        if hasattr(os, "sync"): os.sync()
    except Exception: pass

# === [NEW] 실패기록 가드 ===
def _maybe_insert_failure(payload:dict, feature_vector:list|None=None):
    try:
        if not _ENABLE_TRAIN_FAILURE_RECORD:
            # 평가/예측 체계 준비 전에는 차단
            if not ready_for_group_predict():
                return
        insert_failure_record(payload, feature_vector=(feature_vector or []))
    except Exception as e:
        _safe_print(f"[FAILREC skip] {e}")

def _log_skip(symbol,strategy,reason):
    logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=reason,status="skipped")
    _maybe_insert_failure({"symbol":symbol,"strategy":strategy,"model":"all","predicted_class":-1,"success":False,"rate":0.0,"reason":reason},feature_vector=[])

def _log_fail(symbol,strategy,reason):
    logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=reason,status="failed")
    _maybe_insert_failure({"symbol":symbol,"strategy":strategy,"model":"all","predicted_class":-1,"success":False,"rate":0.0,"reason":reason},feature_vector=[])

def _strategy_horizon_hours(s:str)->int: return {"단기":4,"중기":24,"장기":168}.get(s,24)

# ⬇️ 변경: 라벨 수익률을 모드로 선택 (기본 close→ signed ±)
def _future_returns_by_timestamp(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return np.zeros(0 if df is None else len(df), dtype=np.float32)

    mode = os.getenv("LABEL_RETURN_MODE", "close")  # "close" | "max" | "signed_extreme"
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    ts = (ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts).dt.tz_convert("Asia/Seoul")

    close = df["close"].astype(float).values
    high  = (df["high"] if "high" in df.columns else df["close"]).astype(float).values
    low   = (df["low"]  if "low"  in df.columns else df["close"]).astype(float).values

    out = np.zeros(len(df), dtype=np.float32)
    H = pd.Timedelta(hours=horizon_hours)
    j0 = 0

    for i in range(len(df)):
        t0 = ts.iloc[i]; t1 = t0 + H
        j = max(j0, i)
        mx = high[i]; mn = low[i]; j_last = i
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > mx: mx = high[j]
            if low[j]  < mn: mn = low[j]
            j_last = j
            j += 1
        j0 = max(j_last, i)

        base = close[i] if close[i] > 0 else (close[i] + 1e-6)

        if mode == "max":
            r = (mx - base) / (base + 1e-12)                 # 항상 ≥0 (옛 방식)
        elif mode == "signed_extreme":
            up = (mx - base) / (base + 1e-12)
            dn = (mn - base) / (base + 1e-12)                # 음수 가능
            r = up if abs(up) >= abs(dn) else dn             # 방향 보존 극값
        else:  # "close" (기본)
            fut = close[j_last]
            r = (fut - base) / (base + 1e-12)                # 종가-대-종가, ± 가능

        out[i] = float(r)

    return out

def _stem(p:str)->str: return os.path.splitext(p)[0]

def _save_model_and_meta(model:nn.Module,path_pt:str,meta:dict):
    stem=_stem(path_pt); weight=stem+".ptz"; save_model(weight, model.state_dict())
    meta_path = stem+".meta.json"
    _atomic_write(meta_path, json.dumps(meta,ensure_ascii=False,separators=(",",":")), mode="w")
    # 저장 직후 디스크 배리어
    _disk_barrier([weight, meta_path, MODEL_DIR])
    return weight, meta_path

# === 링크/별칭 생성기 ===
def _safe_alias(src:str,dst:str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except Exception:
        pass
    # 하드링크/심링크 실패시 복사로 폴백(신뢰성 우선)
    try:
        os.link(src, dst); mode="hardlink"
    except Exception:
        try:
            rel = os.path.relpath(src, os.path.dirname(dst))
            os.symlink(rel, dst); mode="symlink"
        except Exception:
            try:
                if os.getenv("ALIAS_COPY_FALLBACK","1")=="1":
                    shutil.copy2(src, dst); mode="copy"
                else:
                    _safe_print(f"[ALIAS] link failed → skip copy (dst={dst})"); return "skip"
            except Exception as e:
                _safe_print(f"[ALIAS] link/copy failed → {e} (dst={dst})"); return "skip"
    _disk_barrier([dst, os.path.dirname(dst)])
    return mode

def _emit_aliases(model_path:str, meta_path:str, symbol:str, strategy:str, model_type:str):
    ext=os.path.splitext(model_path)[1]
    if os.getenv("DISABLE_FLAT_ALIAS","0") != "1":
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

_KNOWN_EXTS = (".ptz", ".safetensors", ".pt")

def _has_any_model_for_symbol(symbol: str) -> bool:
    try:
        if any(glob.glob(os.path.join(MODEL_DIR, f"{symbol}_*{ext}")) for ext in _KNOWN_EXTS):
            return True
        if os.path.isdir(os.path.join(MODEL_DIR, symbol)):
            if any(glob.glob(os.path.join(MODEL_DIR, symbol, "*", f"*{ext}")) for ext in _KNOWN_EXTS):
                return True
    except Exception:
        pass
    return False

# === [ADD] 심볼+전략 단위 모델 존재 체크 (부분 예측에 필요) ===
def _has_model_for(symbol: str, strategy: str) -> bool:
    try:
        # 평면 형태
        if any(glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*{ext}")) for ext in _KNOWN_EXTS):
            return True
        # 디렉토리 형태
        d = os.path.join(MODEL_DIR, symbol, strategy)
        if os.path.isdir(d) and any(glob.glob(os.path.join(d, f"*{ext}")) for ext in _KNOWN_EXTS):
            return True
    except Exception:
        pass
    return False

# === [NEW] 예측 전 모델 가시화 보장 대기 ===
def _await_models_visible(symbols:list[str], timeout_sec:int=20, poll_sec:float=0.5) -> list[str]:
    """학습 직후 파일시스템 전파/동기화 지연 대비: 모델이 보일 때까지 대기."""
    deadline = time.time() + max(1, int(timeout_sec))
    remaining = set(symbols or [])
    last_report = 0.0
    while remaining and time.time() < deadline:
        ready = [s for s in list(remaining) if _has_any_model_for_symbol(s)]
        for s in ready:
            remaining.discard(s)
        now = time.time()
        if now - last_report > 2.5:
            _safe_print(f"[AWAIT] models visible check — ready:{sorted(set(symbols)-remaining)} pending:{sorted(remaining)}")
            last_report = now
        time.sleep(max(0.1, float(poll_sec)))
    return sorted(set(symbols) - remaining)

# ====== (★) 성능 임계치: 단기 통과가 반드시 선행되어야 다음 단계 진입 ======
EVAL_MIN_F1_SHORT = float(os.getenv("EVAL_MIN_F1_SHORT", "0.55"))
EVAL_MIN_F1_MID   = float(os.getenv("EVAL_MIN_F1_MID",   "0.50"))
EVAL_MIN_F1_LONG  = float(os.getenv("EVAL_MIN_F1_LONG",  "0.45"))
_SHORT_RETRY      = int(os.getenv("SHORT_STRATEGY_RETRY", "3"))

def _min_f1_for(strategy:str)->float:
    return EVAL_MIN_F1_SHORT if strategy=="단기" else (EVAL_MIN_F1_MID if strategy=="중기" else EVAL_MIN_F1_LONG)

if _HAS_LIGHTNING:
    class LitSeqModel(pl.LightningModule):
        def __init__(self, base_model:nn.Module, lr:float=1e-3, cls_w:torch.Tensor|None=None):
            super().__init__(); self.model=base_model; self.criterion=nn.CrossEntropyLoss(weight=cls_w); self.lr=lr
        def forward(self,x): return self.model(x)
        def training_step(self,batch,idx):
            xb,yb=batch; logits=self(xb); loss=self.criterion(logits,yb); return loss
        def validation_step(self,batch,idx):
            xb,yb=batch; logits=self(xb); loss=self.criterion(logits,yb)
            preds=torch.argmax(logits,dim=1)
            self.log("val_loss", loss, prog_bar=False)
            self.log("val_f1", pl.metrics.functional.f1.f1_score(preds, yb, num_classes=self.criterion.weight.shape[0] if self.criterion.weight is not None else None, average="macro"), prog_bar=False)
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
                out=fn(*args, **kwargs); q.put(("ok", out))
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

def train_one_model(symbol, strategy, group_id=None, max_epochs=None, stop_event: threading.Event | None = None):
    global _FAILURE_DB_READY
    # 동적 에폭 결정
    if max_epochs is None:
        max_epochs = _epochs_for(strategy)

    res={"symbol":symbol,"strategy":strategy,"group_id":int(group_id or 0),"models":[]}
    try:
        ensure_failure_db(); _FAILURE_DB_READY = True
        _safe_print(f"✅ [train_one_model] {symbol}-{strategy}-g{group_id}")
        _reset_watchdog("enter train_one_model")
        _progress(f"start:{symbol}-{strategy}-g{group_id}")

        _check_stop(stop_event,"before ssl_pretrain")
        try:
            if not DISABLE_SSL:
                ck=get_ssl_ckpt_path(symbol,strategy)
                if not os.path.exists(ck):
                    _safe_print(f"[SSL] start masked_reconstruction → {ck}")
                    _ssl_timeout=float(os.getenv("SSL_TIMEOUT_SEC","120"))
                    status_ssl, _ = _run_with_timeout(
                        masked_reconstruction,
                        args=(symbol,strategy,FEATURE_INPUT_SIZE),
                        kwargs={}, timeout_sec=_ssl_timeout, stop_event=stop_event,
                        hb_tag="ssl:wait", hb_interval=5.0
                    )
                    if status_ssl != "ok":
                        _safe_print(f"[SSL] skip ({status_ssl})")
                else: _safe_print(f"[SSL] cache → {ck}")
            else:
                _safe_print("[SSL] disabled → skip")
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

        # ---- 라벨링
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

        # ---- 특징행렬 (스케일링 없이 시퀀스 생성)
        features_only=feat.drop(columns=["timestamp","strategy"],errors="ignore")
        features_only = features_only.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # ← ADD: NaN/inf 가드
        try:
            feat_dim = int(getattr(features_only, "shape", [0, FEATURE_INPUT_SIZE])[1])
        except Exception:
            feat_dim = int(FEATURE_INPUT_SIZE)

        if len(features_only)>_MAX_ROWS_FOR_TRAIN or len(labels)>_MAX_ROWS_FOR_TRAIN:
            cut=min(_MAX_ROWS_FOR_TRAIN,len(features_only),len(labels))
            features_only=features_only.iloc[-cut:,:]
            labels=labels[-cut:]

        try:
            best_window=find_best_window(symbol,strategy,window_list=[20,40,60],group_id=group_id)  # ← window 후보 확장
        except: best_window=40
        window=max(5,int(best_window)); window=min(window, max(6,len(features_only)-1))

        _check_stop(stop_event,"before sequence build")
        _progress("seq_build")
        X_raw,y=[],[]
        fv=features_only.values.astype(np.float32)
        for i in range(len(fv)-window):
            if i % 128 == 0:
                _check_stop(stop_event,"seq build")
                _progress(f"seq_build@{i}")
            X_raw.append(fv[i:i+window])
            yi=i+window-1
            y.append(labels[yi] if 0<=yi<len(labels) else 0)
        X_raw=np.array(X_raw,dtype=np.float32); y=np.array(y,dtype=np.int64)

        # ===== 성능 0 방지: 단일 클래스/검증 불가 케이스 스킵 =====
        if len(X_raw) < 10:
            _log_skip(symbol,strategy,f"샘플 부족(rows={len(df)}, limit={_limit}, min={_min_required})"); return res
        uniq_all = np.unique(y)
        if len(uniq_all) < 2:
            _log_skip(symbol,strategy,"라벨 단일 클래스 → 학습/평가 스킵"); return res

        # ── 분할 (시간순: 마지막 20%를 검증으로)  ← CHANGED
        n = len(X_raw)
        cut = max(1, int(n * 0.2))
        train_idx = np.arange(0, n - cut)
        val_idx   = np.arange(n - cut, n)

        # ---- train-only fit scaler
        scaler = MinMaxScaler()
        Xtr_flat = X_raw[train_idx].reshape(-1, feat_dim)
        scaler.fit(Xtr_flat)
        train_X = scaler.transform(Xtr_flat).reshape(len(train_idx), window, feat_dim)
        val_X   = scaler.transform(X_raw[val_idx].reshape(-1, feat_dim)).reshape(len(val_idx), window, feat_dim)
        train_y, val_y = y[train_idx], y[val_idx]

        # ===== 성능 0 방지: 분할 후에도 단일 클래스면 스킵 =====
        if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            _log_skip(symbol,strategy,"분할 후 단일 클래스 → 학습/평가 스킵"); return res

        # 데이터 적을 땐 에폭 자동 축소
        if len(train_X) < 200:
            max_epochs = max(4, int(round(max_epochs * 0.7)))

        try:
            if len(train_X)<200: train_X,train_y=balance_classes(train_X,train_y,num_classes=len(class_ranges))
        except Exception as e: _safe_print(f"[balance err] {e}")

        # ===== 클래스 가중치: 등장한 클래스만 계산 → 전체 크기로 확장 =====
        present = np.unique(train_y)
        try:
            cw_present = compute_class_weight(class_weight="balanced", classes=present, y=train_y)
            w_full = np.ones(len(class_ranges), dtype=np.float32)
            for cls, wv in zip(present, cw_present):
                w_full[int(cls)] = float(wv)
        except Exception as e:
            _safe_print(f"[class_weight warn] {e}")
            w_full = np.ones(len(class_ranges), dtype=np.float32)
        w = torch.tensor(w_full, dtype=torch.float32, device=DEVICE)

        for model_type in ["lstm","cnn_lstm","transformer"]:
            _check_stop(stop_event,f"before train {model_type}")
            _progress(f"train:{model_type}:prep")
            base=get_model(model_type,input_size=feat_dim,output_size=len(class_ranges)).to(DEVICE)

            # DataLoaders
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
                lit=LitSeqModel(base,lr=1e-3,cls_w=w)
                callbacks=[_HeartbeatAndStop(stop_event)]
                ckpt_cb = ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1, filename=f"{symbol}-{strategy}-{model_type}-best")
                es_cb   = EarlyStopping(monitor="val_f1", mode="max", patience=int(os.getenv("EARLY_STOP_PATIENCE","5")), check_finite=True)
                callbacks += [ckpt_cb, es_cb]
                trainer=pl.Trainer(max_epochs=max_epochs, accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
                                   devices=1, enable_checkpointing=True, logger=False, enable_model_summary=False,
                                   enable_progress_bar=False, callbacks=callbacks,
                                   gradient_clip_val=GRAD_CLIP if SMART_TRAIN else 0.0)  # ← 추가
                trainer.fit(lit,train_dataloaders=train_loader,val_dataloaders=val_loader)
                model=base
                # ckpt 로드 (가능 시)
                if ckpt_cb.best_model_path and os.path.exists(ckpt_cb.best_model_path):
                    try:
                        state=torch.load(ckpt_cb.best_model_path, map_location="cpu")["state_dict"]
                        cleaned={k.replace("model.",""):v for k,v in state.items() if k.startswith("model.")}
                        model.load_state_dict(cleaned, strict=False)
                    except Exception as _e:
                        _safe_print(f"[CKPT load skip] {_e}")
                _check_stop(stop_event,f"after PL train {model_type}")
            else:
                # ====== 수동 학습 경로 (가중치 + 얼리스톱 + 베스트웨이트) ======
                model=base
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                # 손실: 라벨 스무딩 적용
                crit = nn.CrossEntropyLoss(weight=w, label_smoothing=(LABEL_SMOOTH if SMART_TRAIN else 0.0))

                # deterministic seed
                g = torch.Generator(); g.manual_seed(20240101)

                # WeightedRandomSampler (불균형 보정)
                base_ds = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
                if SMART_TRAIN:
                    cls_counts = np.bincount(train_y, minlength=len(class_ranges)).astype(np.float64)
                    inv = 1.0 / np.clip(cls_counts, 1.0, None)
                    sample_w = torch.tensor(inv[train_y], dtype=torch.double)
                    sampler = torch.utils.data.WeightedRandomSampler(sample_w, num_samples=len(train_y), replacement=True)
                    train_loader = DataLoader(base_ds, batch_size=_BATCH_SIZE, sampler=sampler,
                                              num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY, persistent_workers=_PERSISTENT)
                else:
                    train_loader = DataLoader(base_ds, batch_size=_BATCH_SIZE, shuffle=True, generator=g,
                                              num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY, persistent_workers=_PERSISTENT)

                # 스케줄러: plateau 시 LR ↓
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="max", factor=0.5, patience=2, min_lr=1e-5
                ) if SMART_TRAIN else None

                patience=int(os.getenv("EARLY_STOP_PATIENCE","5"))
                best_f1=-1.0; best_state=None; bad=0; total_loss=0.0
                last_log_ts=time.time()

                for ep in range(max_epochs):
                    _check_stop(stop_event,f"epoch {ep} pre")
                    _progress(f"{model_type}:ep{ep}:start")
                    model.train()
                    for bi,(xb,yb) in enumerate(train_loader):
                        if bi % 16 == 0:
                            _check_stop(stop_event,f"epoch {ep} batch {bi}")
                            _progress(f"{model_type}:ep{ep}:b{bi}")
                        xb,yb=xb.to(DEVICE), yb.to(DEVICE)
                        logits=model(xb); loss=crit(logits,yb)
                        if not torch.isfinite(loss): continue
                        opt.zero_grad(); loss.backward()
                        if SMART_TRAIN and GRAD_CLIP > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        opt.step(); total_loss += float(loss.item())

                    # ---- 평가 (val_f1)
                    model.eval(); preds=[]; lbls=[]
                    with torch.no_grad():
                        for xb,yb in val_loader:
                            p=torch.argmax(model(xb.to(DEVICE)),dim=1).cpu().numpy()
                            preds.extend(p); lbls.extend(yb.numpy())
                    cur_f1=float(f1_score(lbls,preds,average="macro"))

                    # 스케줄러 스텝
                    if scheduler is not None:
                        scheduler.step(cur_f1)

                    # early stop/bookkeep
                    improved = cur_f1 > best_f1 + 1e-6
                    if improved:
                        best_f1 = cur_f1
                        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                    if time.time()-last_log_ts>2:
                        _safe_print(f"   ↳ {model_type} ep{ep+1}/{max_epochs} val_f1={cur_f1:.4f} bad={bad}/{patience} loss_sum={total_loss:.4f}")
                        last_log_ts=time.time()
                    _progress(f"{model_type}:ep{ep}:end")
                    if bad >= patience:
                        _safe_print(f"🛑 early stop @ ep{ep+1} (best_f1={best_f1:.4f})")
                        break

                # load best weights before saving
                if best_state is not None:
                    model.load_state_dict(best_state)

            # ---- 최종 검증: acc, f1, val_loss 계산
            _progress(f"eval:{model_type}")
            model.eval(); preds=[]; lbls=[]; val_loss_sum=0.0; n_val=0
            crit_eval = nn.CrossEntropyLoss(weight=w)  # eval은 스무딩 없이
            with torch.no_grad():
                for bi,(xb,yb) in enumerate(val_loader):
                    if bi % 32 == 0: _check_stop(stop_event,f"val batch {bi}"); _progress(f"val_b{bi}")
                    logits=model(xb.to(DEVICE)); loss=crit_eval(logits, yb.to(DEVICE))
                    val_loss_sum += float(loss.item()) * xb.size(0); n_val += xb.size(0)
                    p=torch.argmax(logits,dim=1).cpu().numpy()
                    preds.extend(p); lbls.extend(yb.numpy())
            acc=float(accuracy_score(lbls,preds)); f1=float(f1_score(lbls,preds,average="macro"))
            val_loss = float(val_loss_sum / max(1,n_val))

            # === 품질 최소 게이트(전략별 임계와 전역 VAL_F1_MIN 중 더 높은 값) ← ADD
            min_gate = max(_min_f1_for(strategy), float(get_QUALITY().get("VAL_F1_MIN", 0.10)))

            stem=os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group{int(group_id) if group_id is not None else 0}_cls{int(len(class_ranges))}")

            # 🔸 meta에 class_ranges를 포함 + gate/passed 기록
            meta={
                "symbol":symbol,
                "strategy":strategy,
                "model":model_type,
                "group_id":int(group_id) if group_id is not None else 0,
                "num_classes":int(len(class_ranges)),
                "class_ranges": [[float(lo), float(hi)] for (lo,hi) in class_ranges],
                "input_size":int(feat_dim),
                "metrics":{"val_acc":acc,"val_f1":f1,"val_loss":val_loss},
                "timestamp":now_kst().isoformat(),
                "model_name":os.path.basename(stem)+".ptz",
                "window":int(window),
                "recent_cap":int(len(features_only)),
                "engine":"lightning" if _HAS_LIGHTNING else "manual",
                "data_flags":{"rows":int(len(df)),"limit":int(_limit),"min":int(_min_required),"augment_needed":bool(augment_needed),"enough_for_training":bool(enough_for_training)},
                "train_loss_sum":float(total_loss),
                "min_f1_gate": float(min_gate)  # ← ADD
            }

            wpath,mpath=_save_model_and_meta(model, stem+".pt", meta)
            _archive_old_checkpoints(symbol,strategy,model_type,keep_n=1)
            _emit_aliases(wpath,mpath,symbol,strategy,model_type)

            # 모델 가시화 보장(파일시스템 동기화 지연 방지)
            _disk_barrier([wpath, mpath, MODEL_DIR, os.path.join(MODEL_DIR, symbol), os.path.join(MODEL_DIR, symbol, strategy)])

            logger.log_training_result(
                symbol, strategy, model=os.path.basename(wpath), accuracy=acc, f1=f1, loss=val_loss,
                note=(f"train_one_model(window={window}, cap={len(features_only)}, engine={'lightning' if _HAS_LIGHTNING else 'manual'}, "
                      f"data_flags={{rows:{len(df)},limit:{_limit},min:{_min_required},aug:{int(augment_needed)},enough_for_training:{int(enough_for_training)}}})"),
                source_exchange="BYBIT", status="success"
            )

            # === (★) 개별 모델 통과 여부 기록 (전역게이트 반영) ← CHANGED
            passed = bool(f1 >= min_gate)
            meta.update({"passed": int(passed)})
            res["models"].append({
                "type":model_type,"acc":acc,"f1":f1,"val_loss":val_loss,
                "loss_sum":float(total_loss),"pt":wpath,"meta":mpath,"passed":passed
            })
            _safe_print(f"🟩 TRAIN done [{model_type}] acc={acc:.4f} f1={f1:.4f} val_loss={val_loss:.5f} → {os.path.basename(wpath)} (passed={int(passed)} gate={min_gate:.2f})")

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # === (★) 전략/그룹 레벨 ok: 임계치 통과 모델이 하나라도 있어야 True
        res["ok"] = any(m.get("passed") for m in res.get("models", []))
        _safe_print(f"[RESULT] {symbol}-{strategy}-g{group_id} ok={res['ok']}")
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

def _is_cold_start()->bool:
    try:
        any_flat = bool(glob.glob(os.path.join(MODEL_DIR, "*.ptz")))
        any_tree = bool(glob.glob(os.path.join(MODEL_DIR, "*", "*", "*.ptz")))
        return not (any_flat or any_tree)
    except Exception:
        return True

# === [NEW] 전역 예측 락 대기/정리 유틸 ===
_PREDICT_LOCK_FILE = "/persistent/run/predict_running.lock"
def _predict_lock_exists()->bool:
    try: return os.path.exists(_PREDICT_LOCK_FILE)
    except: return False

def _predict_lock_is_stale(stale_sec:int)->bool:
    try:
        if not os.path.exists(_PREDICT_LOCK_FILE): return False
        m = os.path.getmtime(_PREDICT_LOCK_FILE)
        return (time.time() - float(m)) > max(10, int(stale_sec))
    except: return False

def _clear_predict_lock(force:bool=False, stale_sec:int=120, tag:str=""):
    try:
        if not _predict_lock_exists(): return
        if force or _predict_lock_is_stale(stale_sec):
            os.remove(_PREDICT_LOCK_FILE)
            _safe_print(f"[LOCK] cleared predict lock (force={int(force)} stale>{stale_sec}s) {tag}")
    except Exception as e:
        _safe_print(f"[LOCK] clear fail → {e} {tag}")

def _wait_predict_lock_clear(timeout_sec:int=20, stale_sec:int=120, poll:float=0.25, tag:str=""):
    deadline = time.time() + max(1, int(timeout_sec))
    while _predict_lock_exists() and time.time() < deadline:
        if _predict_lock_is_stale(stale_sec):
            _clear_predict_lock(force=True, stale_sec=stale_sec, tag=tag)
            break
        time.sleep(max(0.05, float(poll)))
    if _predict_lock_exists():
        _clear_predict_lock(force=True, stale_sec=stale_sec, tag=f"{tag}|final")
    return not _predict_lock_exists()

# === 부분 예측 허용 스위치(ENV) ===
_PREDICT_PARTIAL_OK = os.getenv("PREDICT_PARTIAL_OK", "1") == "1"

# === [CHANGED] 예측은 **동기 실행** (스레드/타임아웃 제거)
_PREDICT_TIMEOUT_SEC=float(os.getenv("PREDICT_TIMEOUT_SEC","30"))
def _safe_predict_sync(predict_fn,symbol,strategy,source,model_type=None, stop_event: threading.Event | None = None):
    try:
        _safe_print(f"[PREDICT] start {symbol}-{strategy} ({source})")
        predict_fn(symbol, strategy, source=source, model_type=model_type)
        _safe_print(f"[PREDICT] done  {symbol}-{strategy}")
        return True
    except Exception as e:
        _safe_print(f"[PREDICT FAIL] {symbol}-{strategy}: {e}")
        return False

# === [ADD] predict_trigger가 사용할 (옵션) 타임아웃 래퍼 익스포트 ===
def _safe_predict_with_timeout(predict_fn, *, symbol, strategy, source="그룹직후", model_type=None, timeout=None):
    t = float(timeout or _PREDICT_TIMEOUT_SEC)
    status, _ = _run_with_timeout(
        lambda: predict_fn(symbol, strategy, source=source, model_type=model_type),
        args=(), kwargs={}, timeout_sec=t, stop_event=None,
        hb_tag="predict:wait", hb_interval=2.0
    )
    return status == "ok"

def _run_bg_if_not_stopped(name:str, fn, stop_event: threading.Event | None):
    if stop_event is not None and stop_event.is_set():
        _safe_print(f"[SKIP:{name}] stop during reset"); return
    if _BG_STARTED.get(name, False): return
    if name=="failure_train" and ( _is_cold_start() or not _ENABLE_BG_FAILURE_TRAIN ):
        _safe_print("[BG:failure_train] disabled or cold start → skip")
        return
    if name=="evo_meta_train" and ( _is_cold_start() or not _ENABLE_EVO_META_BG ):
        _safe_print("[BG:evo_meta_train] disabled or cold start → skip")
        return
    _BG_STARTED[name] = True
    th=threading.Thread(target=lambda: (fn()), daemon=True)
    th.start()
    _safe_print(f"[BG:{name}] started (daemon)")

# =========================
# 🔒 엄격 순서/완결 강제 설정
# =========================
_ENFORCE_FULL_STRATEGY = os.getenv("ENFORCE_FULL_STRATEGY","1")=="1"
_STRICT_HALT_ON_INCOMPLETE = os.getenv("STRICT_HALT_ON_INCOMPLETE","1")=="1"
_SYMBOL_RETRY_LIMIT = int(os.getenv("SYMBOL_RETRY_LIMIT","1"))
_REQUIRE_AT_LEAST_ONE_MODEL_PER_GROUP = os.getenv("REQUIRE_ONE_PER_GROUP","1")=="1"

def _train_full_symbol(symbol:str, stop_event: threading.Event | None = None) -> tuple[bool, dict]:
    """
    (핵심) 단기 → 중기 → 장기 순서 고정.
    '단기 성공(F1 임계치 통과)'이 선행되지 않으면 뒤 전략 **진입 자체 금지**.
    반환: (symbol_complete, detail)
    """
    strategies=["단기","중기","장기"]
    detail={}
    symbol_complete=True
    prev_strategy_ok = True

    for strategy in strategies:
        if stop_event is not None and stop_event.is_set(): return False, detail
        if not prev_strategy_ok:
            _safe_print(f"[ORDER-STOP] 이전 전략 미완료(성공 기준 미충족) → {symbol} {strategy} 스킵")
            detail[strategy] = {-1: False}
            symbol_complete = False
            break

        try:
            cr=get_class_ranges(symbol=symbol,strategy=strategy)
            if not cr:
                logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note="클래스 경계 없음",status="skipped")
                detail[strategy]={-1:False}
                symbol_complete=False
                prev_strategy_ok=False
                continue

            num_classes=len(cr); groups=get_class_groups(num_classes=num_classes); max_gid=len(groups)-1
            strat_complete=True; detail[strategy]={}

            for gid in range(max_gid+1):
                if stop_event is not None and stop_event.is_set(): return False, detail
                try:
                    gr=get_class_ranges(symbol=symbol,strategy=strategy,group_id=gid)
                except Exception:
                    gr=None
                if not gr or len(gr)<2:
                    try:
                        _log_class_ranges_safe(symbol,strategy,group_id=gid,class_ranges=gr or [],note="train_skip(<2 classes)", stop_event=stop_event)
                        logger.log_training_result(symbol,strategy,model=f"group{gid}",accuracy=0.0,f1=0.0,loss=0.0,note=f"스킵: group_id={gid}, cls<2",status="skipped")
                    except: pass
                    detail[strategy][gid]=False
                    strat_complete=False
                    continue

                _reset_watchdog("enter symbol/group")
                _progress(f"train_models:{symbol}-{strategy}-g{gid}")

                # === (★) 단기는 재시도 허용, 나머지는 1회
                attempts = (_SHORT_RETRY if strategy=="단기" else 1)
                ok_once = False
                for attempt in range(attempts):
                    res=train_one_model(symbol,strategy,group_id=gid, max_epochs=_epochs_for(strategy), stop_event=stop_event)
                    ok = bool(res and isinstance(res,dict) and res.get("ok") is True)
                    if ok:
                        ok_once = True
                        break
                    _safe_print(f"[RETRY] {symbol}-{strategy}-g{gid} attempt {attempt+1}/{attempts} failed(F1<th).")

                detail[strategy][gid]=ok_once
                if not ok_once and _REQUIRE_AT_LEAST_ONE_MODEL_PER_GROUP:
                    strat_complete=False
                if stop_event is not None and stop_event.is_set(): return False, detail
                time.sleep(0.05)

            if not strat_complete:
                symbol_complete=False
                prev_strategy_ok=False
            else:
                prev_strategy_ok=True

        except Exception as e:
            try:
                logger.log_training_result(symbol,strategy,model="all",accuracy=0.0,f1=0.0,loss=0.0,note=f"전략 실패: {e}",status="failed")
            except: pass
            detail[strategy]={-1:False}
            symbol_complete=False
            prev_strategy_ok=False

    return symbol_complete, detail

def train_models(symbol_list, stop_event: threading.Event | None = None, ignore_should: bool = False):
    """
    심볼당:
      - 단기 '성공(F1≥임계치)' 완료 → 중기 → 장기.
      - 중간에 미완료면 재시도 후에도 미완료면 **그 심볼 종료**.
    """
    completed_symbols=[]; partial_symbols=[]
    env_force = (os.getenv("TRAIN_FORCE_IGNORE_SHOULD","0") == "1")

    for symbol in symbol_list:
        if stop_event is not None and stop_event.is_set():
            _safe_print("[STOP] train_models: early"); break

        symbol_has_model = _has_any_model_for_symbol(symbol)
        local_ignore = ignore_should or env_force or (not symbol_has_model)
        if not local_ignore:
            if not should_train_symbol(symbol):
                _safe_print(f"[ORDER] skip {symbol} (should_train_symbol=False, models_exist={symbol_has_model})")
                continue
        else:
            why = "env" if env_force else ("no_model" if not symbol_has_model else "first_pass")
            _safe_print(f"[order-override] {symbol}: force train (reason={why})")

        trained_complete=False
        for attempt in range(max(1,_SYMBOL_RETRY_LIMIT)):
            if stop_event is not None and stop_event.is_set(): break
            complete, detail = _train_full_symbol(symbol, stop_event=stop_event)
            _safe_print(f"[ORDER] {symbol} attempt {attempt+1}/{_SYMBOL_RETRY_LIMIT} → complete={complete} detail={detail}")
            if complete:
                trained_complete=True
                break

        if trained_complete:
            completed_symbols.append(symbol)
            try: mark_symbol_trained(symbol)
            except Exception as e: _safe_print(f"[mark_symbol_trained err] {e}")
        else:
            partial_symbols.append(symbol)
            if _ENFORCE_FULL_STRATEGY and _STRICT_HALT_ON_INCOMPLETE:
                _safe_print(f"[HALT] {symbol} 미완료 → 그룹 진행 중단")
                break

    # BG 작업 트리거 (콜드스타트/ENV 가드)
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

    return completed_symbols, partial_symbols

# === [SMOKE] 후보 찾기 & 스모크 예측 유틸 ===
def _scan_symbols_from_model_dir() -> list[str]:
    syms=set()
    try:
        # flat files
        for p in glob.glob(os.path.join(MODEL_DIR, f"*_*_*.*")):
            b=os.path.basename(p)
            m=re.match(r"^([A-Z0-9]+)_[^_]+_", b)
            if m: syms.add(m.group(1))
        # tree dirs
        for d in glob.glob(os.path.join(MODEL_DIR, "*")):
            if os.path.isdir(d):
                syms.add(os.path.basename(d))
    except Exception: pass
    return sorted(syms)

def _pick_smoke_symbol(candidates:list[str]) -> str|None:
    cand = [s for s in candidates if _has_any_model_for_symbol(s)]
    if cand: return sorted(cand)[0]
    pool=_scan_symbols_from_model_dir()
    pool=[s for s in pool if _has_any_model_for_symbol(s)]
    return pool[0] if pool else None

def _run_smoke_predict(predict_fn, symbol: str):
    ok_any=False
    for strat in ["단기","중기","장기"]:
        if _has_model_for(symbol, strat):
            ok_any |= _safe_predict_sync(predict_fn, symbol, strat, source="그룹직후(스모크)")
    return ok_any

# === 그룹 예측 독점 플래그 (predict.py와 동일 경로/파일명) ===
_RUN_DIR = "/persistent/run"
_GROUP_ACTIVE = os.path.join(_RUN_DIR, "group_predict.active")

def _group_active_on(note:str=""):
    try:
        os.makedirs(_RUN_DIR, exist_ok=True)
        with open(_GROUP_ACTIVE, "w", encoding="utf-8") as f:
            f.write(note or "1")
            try: f.flush(); os.fsync(f.fileno())
            except Exception: pass
        _fsync_dir(_RUN_DIR)
    except Exception as e:
        _safe_print(f"[GROUP_ACTIVE on err] {e}")

def _group_active_off():
    try:
        if os.path.exists(_GROUP_ACTIVE):
            os.remove(_GROUP_ACTIVE)
            _fsync_dir(_RUN_DIR)
    except Exception as e:
        _safe_print(f"[GROUP_ACTIVE off err] {e}")

# === 그룹 루프 ===

def _get_group_stale_sec() -> int:
    """
    predict.py(2번)과 환경변수 합치:
      - 우선: PREDICT_LOCK_STALE_GROUP_SEC (기본 600)
      - 백워드: PREDICT_LOCK_STALE_TRAIN_SEC (설정돼 있으면 사용)
      - 최종 기본: 600
    """
    v1 = os.getenv("PREDICT_LOCK_STALE_GROUP_SEC")
    if v1 is not None: 
        try: return max(3, int(v1))
        except: pass
    v2 = os.getenv("PREDICT_LOCK_STALE_TRAIN_SEC")
    if v2 is not None:
        try: return max(3, int(v2))
        except: pass
    return 600

def train_symbol_group_loop(sleep_sec:int=0, stop_event: threading.Event | None = None):
    threading.Thread(target=_watchdog_loop, args=(stop_event,), daemon=True).start()
    _reset_watchdog("loop start")

    # 시작 시 혹시 남아있던 플래그 정리
    _group_active_off()
    # 루프 시작 즉시 예측 게이트 닫기(외부 조기예측 차단)
    try: close_predict_gate(note="loop_start")
    except Exception as e: _safe_print(f"[gate pre-close(err@start) {e}]")

    env_force_ignore = (os.getenv("TRAIN_FORCE_IGNORE_SHOULD","0") == "1")
    env_reset = (os.getenv("RESET_GROUP_ORDER_ON_START","0") == "1")

    force_full_pass = _is_cold_start() or env_force_ignore
    if force_full_pass or env_reset:
        _safe_print("🧪 start → force mode: "
                    f"ignore_should={force_full_pass} (env={env_force_ignore}), reset_group_order={env_reset or force_full_pass}")
        try:
            reset_group_order(0)
            _safe_print("♻️ group order state reset")
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
                _reset_watchdog(f"enter group {idx}")

                # 🔒 그룹 학습 전 전체 구간 보호: 플래그 ON + 게이트 닫기 + 낡은 예측 락 정리
                _group_active_on(note=f"group_{idx+1}_train")
                try: close_predict_gate(note=f"group_{idx+1}_train")
                except Exception as e: _safe_print(f"[gate pre-close err] {e}")
                _wait_predict_lock_clear(
                    timeout_sec=int(os.getenv("PREDICT_LOCK_WAIT_PREOPEN_SEC","15")),
                    stale_sec=_get_group_stale_sec(),
                    tag=f"group_{idx+1}:pre-train"
                )

                _safe_print(f"🚀 [group] {idx+1}/{len(groups)} → {group}")
                _progress(f"group{idx}:start")

                completed_syms, partial_syms = train_models(group, stop_event=stop_event, ignore_should=force_full_pass)
                if stop_event is not None and stop_event.is_set(): _safe_print("🛑 stop after train → exit"); break

                # === 그룹 완료 여부 판정 (부분 예측 금지)
                group_complete = set(completed_syms) >= set(group) and len(partial_syms) == 0
                if not group_complete:
                    _safe_print(f"[BLOCK] 그룹{idx+1} 미완료 → 예측/마킹 금지 "
                                f"(completed={sorted(completed_syms)}, partial={sorted(partial_syms)})")
                    # 그룹 보호 플래그 해제
                    _group_active_off()
                    # 정책상 중단이면 즉시 중단
                    if partial_syms and _ENFORCE_FULL_STRATEGY and _STRICT_HALT_ON_INCOMPLETE:
                        _safe_print(f"[HALT] 그룹 {idx+1}: 미완결 심볼 존재 → 그룹 루프 중단")
                        break
                    # 다음 그룹으로 넘어가되, 마킹은 하지 않음
                    _prune_caches_and_gc()
                    _progress(f"group{idx}:incomplete-skip-predict")
                    continue

                # === 예측 readiness 확인
                if not ready_for_group_predict():
                    _safe_print(f"[PREDICT-BLOCK] 그룹{idx+1} ready_for_group_predict()==False → 예측 보류 및 마킹 금지")
                    _group_active_off()
                    _prune_caches_and_gc()
                    _progress(f"group{idx}:ready_false")
                    continue

                # === 예측 대상은 '해당 그룹 전체 심볼'
                predict_candidates = list(group)

                # 🔐 모델 가시화 보장(파일시스템 동기화 지연 방지)
                await_sec_default = int(os.getenv("PREDICT_MODEL_AWAIT_SEC","60"))  # 기본 60초
                visible_syms = _await_models_visible(predict_candidates, timeout_sec=await_sec_default)
                predict_syms = sorted({s for s in visible_syms if _has_any_model_for_symbol(s)})

                _safe_print(f"[PREDICT-DECIDE] ready={bool(ready_for_group_predict())} "
                            f"group={group} completed={completed_syms} partial={partial_syms} "
                            f"visible_syms={predict_syms}")

                ran_any=False
                if predict_syms:
                    # 🧹 예측 시작 전: 남아있을 수 있는 전역 락 정리/대기
                    _wait_predict_lock_clear(
                        timeout_sec=int(os.getenv("PREDICT_LOCK_WAIT_PREOPEN_SEC","15")),
                        stale_sec=_get_group_stale_sec(),
                        tag=f"group_{idx+1}:pre-open"
                    )
                    try:
                        # 게이트 열기 (우리 예측 시작)
                        try: open_predict_gate(note=f"group_{idx+1}_start")
                        except Exception as e: _safe_print(f"[gate open err] {e}")
                        time.sleep(0.5)

                        _safe_print(f"[PREDICT] group {idx+1} begin")
                        for symbol in predict_syms:
                            if stop_event is not None and stop_event.is_set(): break
                            for strategy in ["단기","중기","장기"]:
                                if stop_event is not None and stop_event.is_set(): break
                                if not _has_model_for(symbol, strategy):
                                    _safe_print(f"[PREDICT-SKIP] {symbol}-{strategy}: 모델 없음(전략별)")
                                    continue
                                ran_any |= _safe_predict_sync(
                                    predict, symbol, strategy,
                                    source="그룹직후", model_type=None,
                                    stop_event=stop_event
                                )
                    finally:
                        try: close_predict_gate(note=f"group_{idx+1}_end(finalize)")
                        except Exception as e: _safe_print(f"[gate close err] {e}")
                        _wait_predict_lock_clear(
                            timeout_sec=int(os.getenv("PREDICT_LOCK_WAIT_POST_SEC","10")),
                            stale_sec=_get_group_stale_sec(),
                            tag=f"group_{idx+1}:post-close"
                        )
                        _safe_print(f"[PREDICT] group {idx+1} end")

                # ⛑ 스모크 폴백: 그룹 완료 상태에서 모델 가시성 지연일 때만 1건 보장
                if not ran_any:
                    cand_symbol = _pick_smoke_symbol(predict_candidates)
                    if cand_symbol:
                        try:
                            _safe_print(f"[SMOKE] no visible syms → fallback predict for {cand_symbol}")
                            _wait_predict_lock_clear(
                                timeout_sec=int(os.getenv("PREDICT_LOCK_WAIT_PREOPEN_SEC","15")),
                                stale_sec=_get_group_stale_sec(),
                                tag=f"group_{idx+1}:smoke-pre"
                            )
                            try: open_predict_gate(note=f"group_{idx+1}_smoke_start")
                            except Exception as e: _safe_print(f"[gate open err] {e}")
                            time.sleep(0.3)
                            try:
                                ran_any = _run_smoke_predict(predict, cand_symbol)
                            finally:
                                try: close_predict_gate(note=f"group_{idx+1}_smoke_end")
                                except Exception as e: _safe_print(f"[gate close err] {e}")
                                _wait_predict_lock_clear(
                                    timeout_sec=int(os.getenv("PREDICT_LOCK_WAIT_POST_SEC","10")),
                                    stale_sec=_get_group_stale_sec(),
                                    tag=f"group_{idx+1}:smoke-post"
                                )
                        finally:
                            pass

                # ✅ 전량 예측이 최소 1건 이상 수행된 경우에만 마킹
                if ran_any:
                    try: mark_group_predicted()
                    except Exception as e: _safe_print(f"[mark_group_predicted err] {e}")
                else:
                    _safe_print(f"[MARK-SKIP] group {idx+1}: 예측 수행 없음 → 마킹 생략")

                _prune_caches_and_gc()
                _progress(f"group{idx}:done")

                # 그룹 보호 플래그 해제(학습+예측 완료 또는 스킵)
                _group_active_off()

                if sleep_sec>0:
                    for _ in range(sleep_sec):
                        if stop_event is not None and stop_event.is_set(): _safe_print("[STOP] sleep break"); break
                        time.sleep(1)
                    if stop_event is not None and stop_event.is_set(): break

            _safe_print("✅ group pass done (loop will continue unless stopped)")
            if force_full_pass and not env_force_ignore:
                force_full_pass = False
                _safe_print("🧪 cold start first pass completed → resume normal scheduling")
        except _ControlledStop:
            _safe_print("🛑 cooperative stop inside group loop")
            break
        except Exception as e:
            _safe_print(f"[group loop err] {e}\n{traceback.format_exc()}")
        finally:
            # 루프 단위 안전 정리
            _group_active_off()

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
