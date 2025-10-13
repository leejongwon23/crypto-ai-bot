# === predict.py (YOPO v1.5 — 예측 정상화 완전판) ===
# 핵심 수정
# ① get_model_predictions(): 로더 실패시 다중 폴백( model_io → torch.load → state_dict 주입 )로 복구
# ② predict(): gate/group_active 스킵 사유를 콘솔에 즉시 출력 + 그룹 스테일 자동 정리
# ③ _group_active(): STALE 처리(만료시 자동 해제)

import os, sys, json, datetime, pytz, random, time, tempfile, shutil, csv, glob, inspect, threading
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
try:
    from data.utils import get_kline_by_strategy, compute_features
except Exception:
    from utils import get_kline_by_strategy, compute_features

__all__ = ["predict","is_predict_gate_open","open_predict_gate","close_predict_gate","run_evaluation_once","run_evaluation_loop"]

RUN_DIR="/persistent/run"; os.makedirs(RUN_DIR,exist_ok=True)
PREDICT_GATE=os.path.join(RUN_DIR,"predict_gate.json")
PREDICT_BLOCK="/persistent/predict.block"
GROUP_ACTIVE=os.path.join(RUN_DIR,"group_predict.active")
GROUP_ACTIVE_STALE_SEC=int(os.getenv("GROUP_ACTIVE_STALE_SEC","600"))  # 10분 기본

def _lock_path_for(symbol:str,strategy:str)->str:
    def _clean(s:str)->str:
        s=(s or "None"); return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s)
    return os.path.join(RUN_DIR,f"predict_{_clean(symbol)}_{_clean(strategy)}.lock")

PREDICT_LOCK_TTL=int(os.getenv("PREDICT_LOCK_TTL","600"))
PREDICT_LOCK_STALE_TRAIN_SEC=int(os.getenv("PREDICT_LOCK_STALE_TRAIN_SEC","600"))

def _now_kst(): return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def is_predict_gate_open():
    try:
        if os.getenv("FORCE_PREDICT_CLOSE","0")=="1": return False
        if os.path.exists(PREDICT_BLOCK): return False
        if os.path.exists(PREDICT_GATE):
            with open(PREDICT_GATE,"r",encoding="utf-8") as f: o=json.load(f)
            return bool(o.get("open",True))
        return True
    except Exception:
        return True

def _bypass_gate_for_source(source:str)->bool:
    s=str(source or "")
    if "그룹직후" in s: return True
    bl=os.getenv("PREDICT_GATE_BYPASS_SOURCES","")
    return any(t and t in s for t in [x.strip() for x in bl.split(",") if x.strip()])

def _group_active()->bool:
    try:
        if not os.path.exists(GROUP_ACTIVE): return False
        # 스테일 처리
        try:
            mtime=os.path.getmtime(GROUP_ACTIVE)
            if (time.time()-mtime)>max(60,GROUP_ACTIVE_STALE_SEC):
                os.remove(GROUP_ACTIVE)
                print(f"[group_active] stale file removed ({int(time.time()-mtime)}s old)")
                return False
        except Exception: pass
        return True
    except Exception: 
        return False

def open_predict_gate(note=""):
    try:
        with open(PREDICT_GATE,"w",encoding="utf-8") as f:
            json.dump({"open":True,"opened_at":_now_kst().isoformat(),"note":note},f,ensure_ascii=False)
            try: f.flush(); os.fsync(f.fileno())
            except Exception: pass
        if os.path.exists(PREDICT_BLOCK):
            try: os.remove(PREDICT_BLOCK)
            except Exception: pass
    except Exception: pass

def close_predict_gate(note=""):
    try:
        with open(PREDICT_GATE,"w",encoding="utf-8") as f:
            json.dump({"open":False,"closed_at":_now_kst().isoformat(),"note":note},f,ensure_ascii=False)
            try: f.flush(); os.fsync(f.fileno())
            except Exception: pass
        try:
            with open(PREDICT_BLOCK,"a") as bf:
                try: bf.flush(); os.fsync(bf.fileno())
                except Exception: pass
        except Exception: pass
    except Exception: pass

def _is_stale_lock(path:str,ttl_sec:int)->bool:
    try:
        if not os.path.exists(path): return False
        mtime=os.path.getmtime(path)
        return (time.time()-float(mtime))>max(30,int(ttl_sec))
    except Exception: return False

def _clear_stale_lock(path:str,ttl_sec:int,tag:str=""):
    try:
        if os.path.exists(path) and _is_stale_lock(path,ttl_sec):
            os.remove(path); print(f"[LOCK] stale predict lock removed ({os.path.basename(path)}) > {ttl_sec}s {tag}"); sys.stdout.flush()
    except Exception: pass

def _acquire_predict_lock(path:str):
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,"w") as f:
            f.write(json.dumps({"pid":os.getpid(),"ts":_now_kst().isoformat()},ensure_ascii=False))
            try: f.flush(); os.fsync(f.fileno())
            except Exception: pass
        return True
    except FileExistsError: return False
    except Exception: return False

def _release_predict_lock(path:str):
    try:
        if os.path.exists(path): os.remove(path)
    except Exception: pass

PREDICT_HEARTBEAT_SEC=int(os.getenv("PREDICT_HEARTBEAT_SEC","3"))

def _predict_hb_loop(stop_evt:threading.Event,tag:str,lock_path:str):
    last_note=""
    while not stop_evt.is_set():
        try:
            gate="open" if is_predict_gate_open() else "closed"
            lock=os.path.exists(lock_path)
            note=f"[HB] predict alive ({tag}) gate={gate} lock={'1' if lock else '0'} ts={_now_kst().strftime('%H:%M:%S')}"
            if note!=last_note:
                print(note); sys.stdout.flush(); last_note=note
        except Exception: pass
        stop_evt.wait(max(1,PREDICT_HEARTBEAT_SEC))

try:
    from window_optimizer import find_best_windows
except Exception:
    try:
        from window_optimizer import find_best_window
    except Exception:
        find_best_window=None
    def find_best_windows(symbol,strategy):
        try: best=int(find_best_window(symbol,strategy,window_list=[10,20,30,40,60],group_id=None)) if callable(find_best_window) else 60
        except Exception: best=60
        return [best,best,best]

try:
    from regime_detector import detect_regime
except Exception:
    def detect_regime(symbol,strategy,now=None): return "unknown"

try:
    from calibration import apply_calibration, get_calibration_version
except Exception:
    def apply_calibration(probs,*,symbol=None,strategy=None,regime=None,model_meta=None): return probs
    def get_calibration_version(): return "none"

STRICT_SAME_BOUNDS=os.getenv("STRICT_SAME_BOUNDS","0")=="1"

PREDICT_MODEL_LOADER_TTL=int(os.getenv("MODEL_LOADER_TTL","600"))
try:
    from model_weight_loader import load_model_cached as _raw_load_model
except Exception:
    _raw_load_model=None
if _raw_load_model is None:
    try:
        from model_io import load_model as _raw_load_model
    except Exception:
        _raw_load_model=None

def load_model_any(path,model=None,**kwargs):
    ttl=kwargs.pop("ttl_sec",PREDICT_MODEL_LOADER_TTL)
    if _raw_load_model is not None:
        try:
            sig=inspect.signature(_raw_load_model); params=list(sig.parameters.values())
            if len(params)>=2:
                try: return _raw_load_model(path,model,ttl_sec=ttl,**kwargs)
                except TypeError: return _raw_load_model(path,model,**kwargs)
            else: return _raw_load_model(path)
        except Exception: pass
    # 1차 실패 → torch.load 직접
    try:
        sd=torch.load(path,map_location="cpu")
        if isinstance(sd,dict) and model is not None:
            try:
                model.load_state_dict(sd,strict=False); model.eval(); return model
            except Exception: return sd
        return sd
    except Exception: 
        return None

from logger import log_prediction, update_model_success, PREDICTION_HEADERS, ensure_prediction_log_exists
from failure_db import insert_failure_record, ensure_failure_db
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
try:
    from model.base_model import get_model
except Exception:
    from base_model import get_model
from config import (get_NUM_CLASSES,get_FEATURE_INPUT_SIZE,get_class_groups,get_class_return_range,class_to_expected_return,get_CLASS_BIN,get_PUBLISH_RUNTIME)

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR="/persistent/models"
PREDICTION_LOG_PATH="/persistent/prediction_log.csv"
NUM_CLASSES=get_NUM_CLASSES()
FEATURE_INPUT_SIZE=get_FEATURE_INPUT_SIZE()

MIN_RET_THRESHOLD=float(os.getenv("PREDICT_MIN_RETURN","0.01"))
ABSTAIN_PROB_MIN=float(os.getenv("ABSTAIN_PROB_MIN","0.25"))
PREDICT_SOFT_ABORT=int(os.getenv("PREDICT_SOFT_ABORT","1"))

PREDICT_WINDOW_ENSEMBLE=os.getenv("PREDICT_WINDOW_ENSEMBLE","mean_var").lower()
ENSEMBLE_VAR_GAMMA=float(os.getenv("ENSEMBLE_VAR_GAMMA","1.0"))
ADJUST_WITH_DIVERSITY=os.getenv("ADJUST_WITH_DIVERSITY","0")=="1"

EXP_STATE="/persistent/logs/meta_explore_state.json"
EXP_EPS=float(os.getenv("EXPLORE_EPS_BASE","0.15"))
EXP_DEC_MIN=float(os.getenv("EXPLORE_DECAY_MIN","120"))
EXP_NEAR=float(os.getenv("EXPLORE_NEAR_GAP","0.07"))
EXP_GAMMA=float(os.getenv("EXPLORE_GAMMA","0.05"))

RG_ENABLE=os.getenv("RG_ENABLE","1")=="1"
RG_VOL_MULT=float(os.getenv("RG_VOL_MULT","3.0"))
RG_LOOKBACK_SHORT=int(os.getenv("RG_LOOKBACK_SHORT","48"))
RG_LOOKBACK_MID=int(os.getenv("RG_LOOKBACK_MID","96"))
RG_LOOKBACK_LONG=int(os.getenv("RG_LOOKBACK_LONG","336"))
RG_VOL_METHOD=os.getenv("RG_VOL_METHOD","std").lower()
RG_MIN_ABS_MID_FOR_VOLCHECK=float(os.getenv("RG_MIN_ABS_MID_FOR_VOLCHECK","0.004"))

def _norm_model_type(mt:str)->str:
    s=str(mt or "").lower()
    if "transformer" in s: return "TRANSFORMER"
    if "cnn" in s: return "CNN_LSTM"
    return "LSTM"

def _ranges_from_meta(meta):
    try:
        cr=meta.get("class_ranges",None)
        if isinstance(cr,list) and len(cr)>=2 and all(isinstance(x,(list,tuple)) and len(x)==2 for x in cr):
            return [(float(a),float(b)) for a,b in cr]
    except Exception: pass
    return None

def _class_range_by_meta_or_cfg(cls_id:int,meta,symbol:str,strategy:str):
    cr=_ranges_from_meta(meta) if isinstance(meta,dict) else None
    if STRICT_SAME_BOUNDS:
        if not (cr and 0<=int(cls_id)<len(cr)): raise RuntimeError("no_class_ranges_in_meta")
        return cr[int(cls_id)]
    return cr[int(cls_id)] if (cr and 0<=int(cls_id)<len(cr)) else get_class_return_range(int(cls_id),symbol,strategy)

def _class_min_meta_or_cfg(cls_id:int,meta,symbol:str,strategy:str)->float:
    lo,_=_class_range_by_meta_or_cfg(cls_id,meta,symbol,strategy); return float(lo)

def _expected_return_meta_or_cfg(cls_id:int,meta,symbol:str,strategy:str)->float:
    lo,hi=_class_range_by_meta_or_cfg(cls_id,meta,symbol,strategy); return (float(lo)+float(hi))/2.0

def _position_from_range(lo:float,hi:float)->str:
    try:
        lo=float(lo); hi=float(hi)
        if hi<=0 and lo<0: return "short"
        if lo>=0 and hi>0: return "long"
        return "neutral"
    except Exception: return "neutral"

def _meets_minret_with_hint(lo:float,hi:float,allow_long:bool,allow_short:bool,thr:float)->bool:
    try:
        lo=float(lo); hi=float(hi); thr=float(thr)
        long_ok=allow_long and (hi>0.0) and (hi>=thr)
        short_ok=allow_short and (lo<0.0) and ((-lo)>=thr)
        if allow_long and allow_short: return (hi>0.0 and hi>=thr) or (lo<0.0 and (-lo)>=thr)
        return long_ok or short_ok
    except Exception: return False

def _load_json(p,default):
    try:
        with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: return default

def _save_json(p,obj):
    try:
        os.makedirs(os.path.dirname(p),exist_ok=True)
        with open(p,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    except Exception: pass

def _bump_use(symbol,strategy,model_path,explored=False):
    st=_load_json(EXP_STATE,{})
    key=f"{symbol}|{strategy}"; st.setdefault(key,{})
    rec=st[key].setdefault(model_path,{"n":0,"n_explore":0,"last_explore_ts":0.0})
    rec["n"]+=1
    if explored:
        rec["n_explore"]+=1; rec["last_explore_ts"]=float(time.time())
    st[key][model_path]=rec; _save_json(EXP_STATE,st)

def _use_stat(symbol,strategy,model_path):
    st=_load_json(EXP_STATE,{}).get(f"{symbol}|{strategy}",{}).get(model_path,{"n":0,"n_explore":0,"last_explore_ts":0.0})
    return int(st.get("n",0)),int(st.get("n_explore",0)),float(st.get("last_explore_ts",0.0))

def _feature_hash(row):
    try:
        import hashlib
        if isinstance(row,torch.Tensor): arr=row.detach().cpu().flatten().numpy().astype(float)
        elif isinstance(row,np.ndarray): arr=row.flatten().astype(float)
        elif isinstance(row,(list,tuple)): arr=np.array(row,dtype=float).flatten()
        else: arr=np.array([float(row)],dtype=float)
        r=[round(float(x),2) for x in arr]
        return hashlib.sha1(",".join(map(str,r)).encode()).hexdigest()
    except Exception: return "hash_error"

_KNOWN_EXTS=(".pt",".ptz",".safetensors")
def _stem(fn):
    for e in _KNOWN_EXTS:
        if fn.endswith(e): return fn[:-len(e)]
    return os.path.splitext(fn)[0]

def _resolve_meta(weight_base):
    base=_stem(weight_base)
    cand=os.path.join(MODEL_DIR,f"{base}.meta.json")
    if os.path.exists(cand): return cand
    ms=sorted(glob.glob(os.path.join(MODEL_DIR,f"{base}_*.meta.json")))
    if ms: return ms[0]
    try:
        p=base.split("_")
        if len(p)>=3:
            sym,strat,mtype=p[0],p[1],p[2]
            cand2=os.path.join(MODEL_DIR,sym,strat,f"{mtype}.meta.json")
            if os.path.exists(cand2): return cand2
    except Exception: pass
    return None

def _glob_many(stem):
    out=[]
    for e in _KNOWN_EXTS: out.extend(glob.glob(f"{stem}{e}"))
    return out

# --- filename tags parser: *_groupX_clsY.*
def _parse_group_cls_from_filename(path:str):
    try:
        base=os.path.basename(_stem(path))
        parts=base.split("_")
        grp=None; cls=None
        for p in parts:
            if p.startswith("group") and p[5:].isdigit(): grp=int(p[5:])
            if p.startswith("cls") and p[3:].isdigit(): cls=int(p[3:])
        return grp,cls
    except Exception:
        return None,None

# === [NEW] 그룹 자동추론 및 보정 ===
def _infer_group_id(symbol: str, strategy: str) -> int:
    try:
        group_map = {
            "BTCUSDT": 0, "ETHUSDT": 1, "XRPUSDT": 2, "SOLUSDT": 3,
            "ADAUSDT": 4, "DOGEUSDT": 5, "DOTUSDT": 6, "MATICUSDT": 7
        }
        gid = group_map.get((symbol or "").upper(), None)
        if gid is None:
            gid = int(abs(hash(f"{symbol}|{strategy}")) % 8)
        return max(0,min(7,int(gid)))
    except Exception:
        return 0

# === [PATCHED] get_available_models — fallback + auto meta recovery ===
def get_available_models(symbol, strategy):
    try:
        if not os.path.isdir(MODEL_DIR):
            return []

        results = []
        search_patterns = [
            os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*"),
            os.path.join(MODEL_DIR, symbol, strategy, "*"),
            os.path.join(MODEL_DIR, symbol, f"{strategy}_*"),
        ]

        def _resolve_meta_abs(weight_abs):
            base = _stem(weight_abs)
            m1 = f"{base}.meta.json"
            if os.path.exists(m1):
                return m1
            cand = glob.glob(os.path.join(MODEL_DIR, f"**/{os.path.basename(base)}.meta.json"), recursive=True)
            return cand[0] if cand else None

        for pattern in search_patterns:
            for ext in _KNOWN_EXTS:
                for w in glob.glob(f"{pattern}{ext}", recursive=True):
                    if not os.path.isfile(w):
                        continue
                    meta_path = _resolve_meta_abs(w)
                    if not meta_path:
                        meta_tmp = {
                            "symbol": symbol,
                            "strategy": strategy,
                            "group_id": _infer_group_id(symbol, strategy),
                            "input_size": FEATURE_INPUT_SIZE,
                            "num_classes": NUM_CLASSES,
                            "created_at": time.time(),
                        }
                        meta_path = _stem(w) + ".meta.json"
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta_tmp, f, ensure_ascii=False, indent=2)

                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                    except Exception:
                        continue

                    fname_grp, _ = _parse_group_cls_from_filename(w)
                    gid = meta.get("group_id")
                    if fname_grp is not None:
                        gid = int(fname_grp)
                    elif gid is None or not str(gid).isdigit():
                        gid = _infer_group_id(symbol, strategy)
                    gid = max(0, min(7, int(gid)))
                    meta["group_id"] = gid

                    results.append({
                        "pt_file": os.path.relpath(w, MODEL_DIR),
                        "meta_path": meta_path,
                        "group_id": gid
                    })

        if not results:
            all_candidates = []
            for ext in _KNOWN_EXTS:
                all_candidates += glob.glob(os.path.join(MODEL_DIR, f"**/*{ext}"), recursive=True)
            if not all_candidates:
                return []
            all_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            top = all_candidates[:3]
            for w in top:
                meta_path = _resolve_meta_abs(w)
                if not meta_path:
                    meta_tmp = {
                        "symbol": symbol,
                        "strategy": strategy,
                        "group_id": _infer_group_id(symbol, strategy),
                        "input_size": FEATURE_INPUT_SIZE,
                        "num_classes": NUM_CLASSES,
                        "created_at": time.time(),
                    }
                    meta_path = _stem(w) + ".meta.json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta_tmp, f, ensure_ascii=False, indent=2)
                results.append({
                    "pt_file": os.path.relpath(w, MODEL_DIR),
                    "meta_path": meta_path,
                    "group_id": _infer_group_id(symbol, strategy)
                })

        seen = set()
        uniq = []
        for it in results:
            if it["pt_file"] not in seen:
                seen.add(it["pt_file"])
                uniq.append(it)
        uniq.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x["pt_file"])), reverse=True)

        if not uniq:
            print(f"[⚠️ 모델 탐색 실패] {symbol}-{strategy} → 빈 목록")
        else:
            print(f"[✅ 모델 탐색 성공] {symbol}-{strategy} → {len(uniq)}개 모델 발견")

        return uniq
    except Exception as e:
        print(f"[get_available_models 오류] {e}")
        return []

def failed_result(symbol,strategy,model_type="unknown",reason="",source="일반",X_input=None):
    t=_now_kst().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[predict] skip {symbol}-{strategy} :: {reason}")
    res={"symbol":symbol,"strategy":strategy,"success":False,"reason":reason,"model":str(model_type or "unknown"),"rate":0.0,"class":-1,"timestamp":t,"source":source,"predicted_class":-1,"label":-1}
    try:
        ensure_prediction_log_exists()
        log_prediction(symbol=symbol,strategy=strategy,direction="예측실패",entry_price=0,target_price=0,model=str(model_type or "unknown"),success=False,reason=reason,rate=0.0,timestamp=t,return_value=0.0,volatility=True,source=source,predicted_class=-1,label=-1,class_return_min=0.0,class_return_max=0.0,class_return_text="")
    except Exception as e: print(f"[failed_result log_prediction 오류] {e}")
    try:
        if X_input is not None: insert_failure_record(res,feature_vector=np.array(X_input).flatten().tolist(),context="prediction")
    except Exception as e: print(f"[failed_result insert_failure_record 오류] {e}")
    return res

def _soft_abstain(symbol,strategy,*,reason,meta_choice="abstain",regime="unknown",X_last=None,group_id=None,df=None,source="보류"):
    try:
        ensure_prediction_log_exists()
        cur=float((df["close"].iloc[-1] if df is not None and len(df) else 0.0))
        note={"reason":reason,"abstain_prob_min":float(ABSTAIN_PROB_MIN),"max_calib_prob":None,"meta_choice":meta_choice,"regime":regime}
        log_prediction(symbol=symbol,strategy=strategy,direction="예측보류",entry_price=cur,target_price=cur,model="meta",model_name=str(meta_choice),predicted_class=-1,label=-1,note=json.dumps(note,ensure_ascii=False),top_k=[],success=False,reason=reason,rate=0.0,return_value=0.0,source=source,group_id=group_id,feature_vector=(torch.tensor(X_last,dtype=torch.float32).numpy() if X_last is not None else None),regime=regime,meta_choice="abstain",raw_prob=None,calib_prob=None,calib_ver=get_calibration_version(),class_return_min=0.0,class_return_max=0.0,class_return_text="")
    except Exception as e: print(f"[soft_abstain 예외] {e}")
    print(f"[predict] abstain {symbol}-{strategy} :: {reason}")
    return {"symbol":symbol,"strategy":strategy,"model":"meta","class":-1,"expected_return":0.0,"class_return_min":0.0,"class_return_max":0.0,"class_return_text":"","position":"neutral","timestamp":_now_kst().isoformat(),"source":source,"regime":regime,"reason":reason,"success":False,"predicted_class":-1,"label":-1}

def _acquire_predict_lock_with_retry(path:str,max_wait_sec:int):
    deadline=time.time()+max(1,int(max_wait_sec))
    while time.time()<deadline:
        if _acquire_predict_lock(path): return True
        time.sleep(random.uniform(0.5,2.0))
    return False

def _prep_lock_for_source(source:str):
    src=str(source or "")
    if "그룹직후" in src:
        try: return int(os.getenv("PREDICT_LOCK_WAIT_GROUP_SEC","30"))
        except Exception: return 30
    return int(os.getenv("PREDICT_LOCK_WAIT_MAX_SEC","15"))

def _ema(arr:np.ndarray,span:int)->np.ndarray:
    if len(arr)==0: return arr
    s=pd.Series(arr,dtype=float); return s.ewm(span=span,adjust=False).mean().values

def _position_hint_from_market(df:pd.DataFrame)->dict:
    try:
        close=df["close"].astype(float).values
        if close.size<70: return {"allow_long":True,"allow_short":True,"ma_fast":None,"ma_slow":None,"slope":0.0}
        ma_fast=_ema(close,20); ma_slow=_ema(close,60)
        y=close[-30:]; x=np.arange(len(y))
        slope=float(np.polyfit(x,y,1)[0])/(np.mean(y)+1e-12)
        diff=float(ma_fast[-1]-ma_slow[-1])/(close[-1]+1e-12)
        strong_up=(diff>0.0015) and (slope>0)
        strong_dn=(diff<-0.0015) and (slope<0)
        if strong_up and not strong_dn:
            return {"allow_long":True,"allow_short":False,"ma_fast":float(ma_fast[-1]),"ma_slow":float(ma_slow[-1]),"slope":float(slope)}
        if strong_dn and not strong_up:
            return {"allow_long":False,"allow_short":True,"ma_fast":float(ma_fast[-1]),"ma_slow":float(ma_slow[-1]),"slope":float(slope)}
        return {"allow_long":True,"allow_short":True,"ma_fast":float(ma_fast[-1]),"ma_slow":float(ma_slow[-1]),"slope":float(slope)}
    except Exception:
        return {"allow_long":True,"allow_short":True,"ma_fast":None,"ma_slow":None,"slope":0.0}

def _recent_volatility(df:pd.DataFrame,strategy:str)->float:
    try:
        if df is None or df.empty or "close" not in df.columns: return 0.0
        close=df["close"].astype(float).values
        if close.size<10: return 0.0
        if strategy=="단기": lb=RG_LOOKBACK_SHORT
        elif strategy=="중기": lb=RG_LOOKBACK_MID
        else: lb=RG_LOOKBACK_LONG
        lb=max(10,min(int(lb),len(close)-1))
        ret=pd.Series(close).pct_change().dropna().values[-lb:]
        if ret.size==0: return 0.0
        if RG_VOL_METHOD=="mad": vol=float(np.mean(np.abs(ret-np.median(ret)))*1.2533)
        else: vol=float(np.std(ret))
        return max(0.0,vol)
    except Exception: return 0.0

def _reality_guard_check(df,strategy,hint,lo_sel,hi_sel,exp_mid)->tuple[bool,str]:
    try:
        pos=_position_from_range(lo_sel,hi_sel)
        if pos=="long" and not bool(hint.get("allow_long",True)): return True,"warn_position_conflict_long"
        if pos=="short" and not bool(hint.get("allow_short",True)): return True,"warn_position_conflict_short"
        if abs(exp_mid)>=RG_MIN_ABS_MID_FOR_VOLCHECK:
            vol=_recent_volatility(df,strategy)
            if vol>0:
                if abs(exp_mid)>RG_VOL_MULT*vol: return False,f"reality_guard_overclaim(vol={vol:.4f}, mid={exp_mid:.4f})"
        return True,"ok"
    except Exception as e: return True,f"rg_exception:{e}"

def _exit_guard_check(lo_sel:float,hi_sel:float,exp_ret:float)->tuple[bool,str]:
    try:
        bin_conf=get_CLASS_BIN(); pub_conf=get_PUBLISH_RUNTIME()
        max_width=float(bin_conf.get("max_width",0.03)); min_er=float(pub_conf.get("min_expected_return",0.01))
        width=float(hi_sel)-float(lo_sel)
        if width>(max_width*1.2+1e-12): return False,f"exit_guard_width(width={width:.4f}, max={max_width:.4f})"
        if abs(float(exp_ret))<(min_er*0.5): return False,f"exit_guard_min_expected_return(mid={float(exp_ret):.4f}, min={min_er:.4f})"
        return True,"ok"
    except Exception as e: return True,f"exit_guard_exception:{e}"

def predict(symbol,strategy,source="일반",model_type=None):
    # 그룹/게이트 체크 (이유 출력)
    if _group_active() and not _bypass_gate_for_source(source):
        print(f"[predict] blocked by group_active (source={source})")
        return failed_result(symbol or "None",strategy or "None",reason="group_predict_active",source=source,X_input=None)
    if not (_bypass_gate_for_source(source) or is_predict_gate_open()):
        print(f"[predict] gate closed (source={source})")
        return failed_result(symbol or "None",strategy or "None",reason="predict_gate_closed",source=source,X_input=None)

    lock_path=_lock_path_for(symbol or "None",strategy or "None")
    if "그룹직후" in str(source or ""): _clear_stale_lock(lock_path,PREDICT_LOCK_STALE_TRAIN_SEC,tag="(group)")
    else: _clear_stale_lock(lock_path,PREDICT_LOCK_TTL,tag="(normal)")
    lock_wait=_prep_lock_for_source(source)
    if not _acquire_predict_lock_with_retry(lock_path,lock_wait):
        print(f"[predict] lock timeout {symbol}-{strategy}")
        return failed_result(symbol or "None",strategy or "None",reason="predict_lock_timeout",source=source,X_input=None)

    _hb_stop=threading.Event(); _hb_tag=f"{symbol}-{strategy}"
    _hb_thread=threading.Thread(target=_predict_hb_loop,args=(_hb_stop,_hb_tag,lock_path),daemon=True); _hb_thread.start()
    try:
        try: ensure_prediction_log_exists()
        except Exception as _e: print(f"[헤더보장 실패] {_e}")
        try: from evo_meta_learner import predict_evo_meta
        except Exception: predict_evo_meta=None
        try: from meta_learning import get_meta_prediction
        except Exception:
            def get_meta_prediction(pl,ft,meta=None): return int(np.argmax(np.mean(np.array(pl),axis=0)))
        ensure_failure_db(); os.makedirs("/persistent/logs",exist_ok=True)
        if not symbol or not strategy: return failed_result(symbol or "None",strategy or "None",reason="invalid_symbol_strategy",source=source,X_input=None)
        regime=detect_regime(symbol,strategy,now=_now_kst()); _=get_calibration_version()
        print(f"[predict] start {symbol}-{strategy} regime={regime} source={source}"); sys.stdout.flush()
        windows=find_best_windows(symbol,strategy)
        if not windows:
            return _soft_abstain(symbol,strategy,reason="window_list_none",meta_choice="abstain",regime=regime,X_last=None,group_id=None,df=None) if PREDICT_SOFT_ABORT else failed_result(symbol,strategy,reason="window_list_none",source=source,X_input=None)
        df=get_kline_by_strategy(symbol,strategy)
        if df is None or len(df)<max(windows)+1:
            return _soft_abstain(symbol,strategy,reason="df_short",meta_choice="abstain",regime=regime,X_last=None,group_id=None,df=df) if PREDICT_SOFT_ABORT else failed_result(symbol,strategy,reason="df_short",source=source,X_input=None)
        feat=compute_features(symbol,df,strategy)
        if feat is None:
            return _soft_abstain(symbol,strategy,reason="feature_short",meta_choice="abstain",regime=regime,X_last=None,group_id=None,df=df) if PREDICT_SOFT_ABORT else failed_result(symbol,strategy,reason="feature_short",source=source,X_input=None)
        X=feat.drop(columns=["timestamp","strategy"],errors="ignore")
        X=MinMaxScaler().fit_transform(X); feat_dim=int(X.shape[1])

        if X.shape[0] < max(windows):
            return _soft_abstain(symbol,strategy,reason="insufficient_recent_rows",meta_choice="abstain",regime=regime,X_last=None,group_id=None,df=df) if PREDICT_SOFT_ABORT else failed_result(symbol,strategy,reason="insufficient_recent_rows",source=source,X_input=None)

        models=get_available_models(symbol,strategy)
        if not models:
            return _soft_abstain(symbol,strategy,reason="no_models",meta_choice="abstain",regime=regime,X_last=X[-1],group_id=None,df=df) if PREDICT_SOFT_ABORT else failed_result(symbol,strategy,reason="no_models",source=source,X_input=X[-1])
        rec_freq=get_recent_class_frequencies(strategy)
        feat_row=torch.tensor(X[-1],dtype=torch.float32)
        outs,all_preds=get_model_predictions(symbol,strategy,models,df,X,windows,rec_freq,regime=regime)
        if not outs:
            return _soft_abstain(symbol,strategy,reason="no_valid_model",meta_choice="abstain",regime=regime,X_last=X[-1],group_id=None,df=df) if PREDICT_SOFT_ABORT else failed_result(symbol,strategy,reason="no_valid_model",source=source,X_input=X[-1])
        hint=_position_hint_from_market(df); allow_long,allow_short=bool(hint["allow_long"]),bool(hint["allow_short"])
        final_cls=None; meta_choice="best_single"; chosen=None; used_minret=False

        if _glob_many(os.path.join(MODEL_DIR,"evo_meta_learner")):
            try:
                from evo_meta_learner import predict_evo_meta
                if callable(predict_evo_meta):
                    pred=int(predict_evo_meta(feat_row.unsqueeze(0),input_size=feat_dim))
                    cmin,cmax=_class_range_by_meta_or_cfg(pred,(chosen or {}).get("meta"),symbol,strategy)
                    if _meets_minret_with_hint(cmin,cmax,allow_long,allow_short,MIN_RET_THRESHOLD): final_cls=pred; meta_choice="evo_meta_learner"
            except Exception as e: print(f"[evo_meta 예외] {e}")

        def _maybe_adjust(probs,recent):
            if ADJUST_WITH_DIVERSITY: return adjust_probs_with_diversity(probs,recent,class_counts=None,alpha=0.10,beta=0.10)
            return np.asarray(probs,dtype=float)

        if final_cls is None:
            best_i,best_score,best_pred=-1,-1.0,None; scores=[]
            for i,m in enumerate(outs):
                adj=_maybe_adjust(m["calib_probs"],rec_freq)
                mask=np.zeros_like(adj,dtype=float)
                for ci in range(len(adj)):
                    try:
                        lo,hi=_class_range_by_meta_or_cfg(ci,m.get("meta"),symbol,strategy)
                        if _meets_minret_with_hint(lo,hi,allow_long,allow_short,MIN_RET_THRESHOLD): mask[ci]=1.0
                    except Exception: pass
                filt=adj*mask
                if filt.sum()>0:
                    filt=filt/filt.sum(); pred=int(np.argmax(filt)); p=float(filt[pred]); fused=True
                else:
                    pred=int(np.argmax(adj)); p=float(adj[pred]); fused=False
                score=p
                m.update({"adjusted_probs":adj,"filtered_probs":(filt if fused else None),
                          "candidate_pred":pred,"success_score":score,"filtered_used":fused})
                scores.append((i,score,pred))
                if score>best_score:
                    best_i,best_score,best_pred=i,score,pred; used_minret=fused
            if len(scores)>=2:
                ss=sorted(scores,key=lambda x:x[1],reverse=True); top1,top2=ss[0],ss[1]; gap=float(top1[1]-top2[1])
                st=_load_json(EXP_STATE,{}).get(f"{symbol}|{strategy}",{}); last=max([v.get("last_explore_ts",0.0) for v in st.values()],default=0.0) if st else 0.0
                minutes=(time.time()-last)/60.0 if last>0 else 1e9; eps=EXP_EPS*(0.5 if minutes<EXP_DEC_MIN else 1.0)
                if gap<=EXP_NEAR and random.random()<eps:
                    cands=[]
                    for i,base,_ in ss[:min(3,len(ss))]:
                        mp=outs[i].get("model_path",""); n,_ne,_ts=_use_stat(symbol,strategy,mp)
                        bonus=EXP_GAMMA/np.sqrt(1.0+float(n)); cands.append((i,base+bonus))
                    cands.sort(key=lambda x:x[1],reverse=True)
                    if cands and cands[0][0]!=top1[0]:
                        best_i=cands[0][0]; best_pred=outs[best_i]["candidate_pred"]; meta_choice="best_single_explore"
            final_cls=int(best_pred); chosen=outs[best_i]
            if meta_choice!="best_single_explore": meta_choice=os.path.basename(chosen["model_path"])
            try: _bump_use(symbol,strategy,chosen.get("model_path",""),explored=("best_single_explore" in meta_choice))
            except Exception: pass

        try:
            cmin_sel,cmax_sel=_class_range_by_meta_or_cfg(final_cls,(chosen or {}).get("meta"),symbol,strategy)
            if not _meets_minret_with_hint(cmin_sel,cmax_sel,allow_long,allow_short,MIN_RET_THRESHOLD):
                best_m,best_sc,best_cls=None,-1.0,None
                for m in outs:
                    adj=m.get("adjusted_probs",m["calib_probs"])
                    for ci in range(len(adj)):
                        try: lo,hi=_class_range_by_meta_or_cfg(ci,m.get("meta"),symbol,strategy)
                        except Exception: continue
                        if not _meets_minret_with_hint(lo,hi,allow_long,allow_short,MIN_RET_THRESHOLD): continue
                        sc=float(adj[ci])
                        if sc>best_sc: best_sc,best_m,best_cls=sc,m,int(ci)
                if best_cls is not None: final_cls,best_sc; chosen,used_minret=best_cls,best_m,True
        except Exception as e: print(f"[임계 가드 예외] {e}")

        try:
            lo_sel,hi_sel=_class_range_by_meta_or_cfg(final_cls,(chosen or {}).get("meta"),symbol,strategy)
            exp_ret=(float(lo_sel)+float(hi_sel))/2.0
            ok,why=_exit_guard_check(lo_sel,hi_sel,exp_ret)
            if not ok:
                return _soft_abstain(symbol,strategy,reason=why,meta_choice=str(meta_choice),regime=regime,X_last=X[-1],group_id=(chosen.get("group_id") if isinstance(chosen,dict) else None),df=df,source="보류")
        except Exception as e: print(f"[출구 가드 예외] {e}")

        try:
            lo_sel,hi_sel=_class_range_by_meta_or_cfg(final_cls,(chosen or {}).get("meta"),symbol,strategy)
            exp_ret=(float(lo_sel)+float(hi_sel))/2.0
            if RG_ENABLE:
                ok,why=_reality_guard_check(df,strategy,hint,lo_sel,hi_sel,exp_ret)
                if not ok:
                    return _soft_abstain(symbol,strategy,reason=why,meta_choice=str(meta_choice),regime=regime,X_last=X[-1],group_id=(chosen.get("group_id") if isinstance(chosen,dict) else None),df=df,source="보류")
        except Exception as e: print(f"[RealityGuard 예외] {e}")

        lo_sel,hi_sel=_class_range_by_meta_or_cfg(final_cls,(chosen or {}).get("meta"),symbol,strategy)
        exp_ret=(float(lo_sel)+float(hi_sel))/2.0
        pos_sel=_position_from_range(lo_sel,hi_sel)
        class_text=f"{float(lo_sel)*100:.2f}% ~ {float(hi_sel)*100:.2f}%"
        current=float(df.iloc[-1]["close"]); entry=current
        def _topk(p,k=3): return [int(i) for i in np.argsort(p)[::-1][:k]]
        topk=_topk((chosen or outs[0])["calib_probs"]) if (chosen or outs) else []
        raw_pred = float(np.nan_to_num((chosen or outs[0])["raw_probs"][final_cls],nan=0.0,posinf=0.0,neginf=0.0)) if (chosen or outs) else None
        calib_pred = float(np.nan_to_num((chosen or outs[0])["calib_probs"][final_cls],nan=0.0,posinf=0.0,neginf=0.0)) if (chosen or outs) else None

        note={"regime":regime,"meta_choice":meta_choice,"raw_prob_pred":raw_pred,"calib_prob_pred":calib_pred,"calib_ver":get_calibration_version(),"min_return_threshold":float(MIN_RET_THRESHOLD),"used_minret_filter":bool(used_minret),"explore_used":("best_single_explore" in str(meta_choice)),"class_range_lo":float(lo_sel),"class_range_hi":float(hi_sel),"expected_return_mid":float(exp_ret),"position":pos_sel,"hint_allow_long":allow_long,"hint_allow_short":allow_short,"hint_ma_fast":hint.get("ma_fast"),"hint_ma_slow":hint.get("ma_slow"),"hint_slope":hint.get("slope"),"reality_guard":{"enabled":bool(RG_ENABLE),"vol_mult":float(RG_VOL_MULT),"method":RG_VOL_METHOD}}
        ensure_prediction_log_exists()
        log_prediction(symbol=symbol,strategy=strategy,direction="예측",entry_price=entry,target_price=entry*(1+exp_ret),model="meta",model_name=("evo_meta_learner" if meta_choice=="evo_meta_learner" else str(meta_choice)),predicted_class=final_cls,label=final_cls,note=json.dumps(note,ensure_ascii=False),top_k=topk,success=False,reason="predicted",rate=float(exp_ret),return_value=0.0,source=("진화형" if meta_choice=="evo_meta_learner" else "기본"),group_id=(chosen.get("group_id") if isinstance(chosen,dict) else None),feature_vector=torch.tensor(X[-1],dtype=torch.float32).numpy(),regime=regime,meta_choice=meta_choice,raw_prob=raw_pred,calib_prob=calib_pred,calib_ver=get_calibration_version(),class_return_min=float(lo_sel),class_return_max=float(hi_sel),class_return_text=class_text)

        try:
            for m in outs:
                if chosen and m.get("model_path")==chosen.get("model_path"): continue
                adj=m.get("adjusted_probs",m["calib_probs"]); filt=m.get("filtered_probs",None)
                if filt is not None and np.sum(filt)>0:
                    pred_i=int(np.argmax(filt)); src=filt
                else:
                    mask=np.zeros_like(adj,dtype=float)
                    for ci in range(len(adj)):
                        try:
                            lo_i,hi_i=_class_range_by_meta_or_cfg(ci,m.get("meta"),symbol,strategy)
                            if _meets_minret_with_hint(lo_i,hi_i,allow_long,allow_short,MIN_RET_THRESHOLD): mask[ci]=1.0
                        except Exception: pass
                    adj2=adj*mask
                    if np.sum(adj2)==0: continue
                    adj2=adj2/np.sum(adj2); pred_i=int(np.argmax(adj2)); src=adj2
                lo_i,hi_i=_class_range_by_meta_or_cfg(pred_i,m.get("meta"),symbol,strategy)
                exp_i=(float(lo_i)+float(hi_i))/2.0; pos_i=_position_from_range(lo_i,hi_i)
                top_i=[int(i) for i in np.argsort(src)[::-1][:3]]
                class_text_i=f"{float(lo_i)*100:.2f}% ~ {float(hi_i)*100:.2f}%"
                raw_i=float(np.nan_to_num(m["raw_probs"][pred_i],nan=0.0,posinf=0.0,neginf=0.0))
                calib_i=float(np.nan_to_num(m["calib_probs"][pred_i],nan=0.0,posinf=0.0,neginf=0.0))
                note_s={"regime":regime,"shadow":True,"model_path":os.path.basename(m.get("model_path","")),"model_type":_norm_model_type(m.get("model_type","")),"val_f1":(None if m.get("val_f1") is None else float(m.get("val_f1"))),"calib_ver":get_calibration_version(),"min_return_threshold":float(MIN_RET_THRESHOLD),"class_range_lo":float(lo_i),"class_range_hi":float(hi_i),"expected_return_mid":float(exp_i),"position":pos_i,"hint_allow_long":allow_long,"hint_allow_short":allow_short}
                log_prediction(symbol=symbol,strategy=strategy,direction="예측(섀도우)",entry_price=entry,target_price=entry*(1+exp_i),model=_norm_model_type(m.get("model_type","")),model_name=os.path.basename(m.get("model_path","")),predicted_class=pred_i,label=pred_i,note=json.dumps(note_s,ensure_ascii=False),top_k=top_i,success=False,reason="shadow",rate=float(exp_i),return_value=0.0,source="섀도우",group_id=m.get("group_id",0),feature_vector=torch.tensor(X[-1],dtype=torch.float32).numpy(),regime=regime,meta_choice="shadow",raw_prob=raw_i,calib_prob=calib_i,calib_ver=get_calibration_version(),class_return_min=float(lo_i),class_return_max=float(hi_i),class_return_text=class_text_i)
        except Exception as e: print(f"[섀도우 로깅 예외] {e}")
        return {"symbol":symbol,"strategy":strategy,"model":"meta","class":final_cls,"expected_return":float(exp_ret),"class_return_min":float(lo_sel),"class_return_max":float(hi_sel),"class_return_text":class_text,"position":pos_sel,"timestamp":_now_kst().isoformat(),"source":source,"regime":regime,"reason":("진화형 메타 최종 선택" if meta_choice=="evo_meta_learner" else f"선택 모델: {meta_choice}")}
    finally:
        try: _hb_stop.set(); _hb_thread.join(timeout=2)
        except Exception: pass
        _release_predict_lock(lock_path)

def evaluate_predictions(get_price_fn):
    from failure_db import check_failure_exists
    ensure_failure_db(); ensure_prediction_log_exists()
    P=PREDICTION_LOG_PATH; now_local=lambda:_now_kst()
    date_str=now_local().strftime("%Y-%m-%d")
    LOG_DIR="/persistent/logs"; os.makedirs(LOG_DIR,exist_ok=True)
    EVAL=os.path.join(LOG_DIR,f"evaluation_{date_str}.csv")
    WRONG=os.path.join(LOG_DIR,f"wrong_{date_str}.csv")
    eval_h={"단기":4,"중기":24,"장기":168}
    tmp=None
    try:
        with open(P,"r",encoding="utf-8-sig",newline="") as f_in:
            rd=csv.DictReader(f_in)
            if rd.fieldnames is None:
                print("[오류] prediction_log.csv 헤더 없음"); return
            base=list(PREDICTION_HEADERS); extras=["status","return"]; fields=base+[c for c in extras if c not in base]
            dir_name=os.path.dirname(P) or "."
            fd,tmp=tempfile.mkstemp(prefix="predlog_",suffix=".csv",dir=dir_name,text=True); os.close(fd)
            with (open(tmp,"w",encoding="utf-8-sig",newline="") as f_tmp, open(EVAL,"w",encoding="utf-8-sig",newline="") as f_eval, open(WRONG,"w",encoding="utf-8-sig",newline="") as f_wrong):
                w_all=csv.DictWriter(f_tmp,fieldnames=fields); w_all.writeheader()
                eval_written=False; wrong_written=False
                for r in rd:
                    try:
                        if r.get("status") not in [None,"","pending","v_pending"]:
                            w_all.writerow({k:r.get(k,"") for k in fields}); continue
                        sym=r.get("symbol","UNKNOWN"); strat=r.get("strategy","알수없음"); model=r.get("model","unknown")
                        try: gid=int(float(r.get("group_id",0)))
                        except Exception: gid=0
                        def to_int(x,d):
                            try:
                                if x in [None,""]: return d
                                return int(float(x))
                            except Exception: return d
                        pred_cls=to_int(r.get("predicted_class",-1),-1)
                        label=to_int(r.get("label",-1),-1); r["label"]=label
                        try: entry=float(r.get("entry_price",0) or 0)
                        except Exception: entry=0.0
                        if entry<=0 or label==-1:
                            r.update({"status":"invalid","reason":"invalid_entry_or_label","return":0.0,"return_value":0.0})
                            if not check_failure_exists(r): insert_failure_record(r,feature_vector=None)
                            w_all.writerow({k:r.get(k,"") for k in fields})
                            if not wrong_written:
                                wrong_writer=csv.DictWriter(f_wrong,fieldnames=sorted(r.keys())); wrong_writer.writeheader(); wrong_written=True
                            wrong_writer.writerow({k:r.get(k,"") for k in r.keys()}); continue
                        ts=pd.to_datetime(r.get("timestamp"),errors="coerce")
                        if ts is None or pd.isna(ts):
                            r.update({"status":"invalid","reason":"timestamp_parse_error","return":0.0,"return_value":0.0})
                            w_all.writerow({k:r.get(k,"") for k in fields})
                            if not wrong_written:
                                wrong_writer=csv.DictWriter(f_wrong,fieldnames=sorted(r.keys())); wrong_writer.writeheader(); wrong_written=True
                            wrong_writer.writerow({k:r.get(k,"") for k in r.keys()}); continue
                        if ts.tzinfo is None: ts=ts.tz_localize("Asia/Seoul")
                        else: ts=ts.tz_convert("Asia/Seoul")
                        hours={"단기":4,"중기":24,"장기":168}.get(strat,6); deadline=ts+pd.Timedelta(hours=hours)
                        dfp=get_price_fn(sym,strat)
                        if dfp is None or "timestamp" not in dfp.columns:
                            r.update({"status":"invalid","reason":"no_price_data","return":0.0,"return_value":0.0})
                            w_all.writerow({k:r.get(k,"") for k in fields})
                            if not wrong_written:
                                wrong_writer=csv.DictWriter(f_wrong,fieldnames=sorted(r.keys())); wrong_writer.writeheader(); wrong_written=True
                            wrong_writer.writerow({k:r.get(k,"") for k in r.keys()}); continue
                        dfp=dfp.copy()
                        dfp["timestamp"]=pd.to_datetime(dfp["timestamp"],errors="coerce").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                        fut=dfp.loc[(dfp["timestamp"]>=ts)&(dfp["timestamp"]<=deadline)]
                        if fut.empty:
                            if _now_kst()<deadline:
                                r.update({"status":"pending","reason":"⏳ 평가 대기 중(마감 전 데이터 없음)","return":0.0,"return_value":0.0})
                                w_all.writerow({k:r.get(k,"") for k in fields}); continue
                            else:
                                r.update({"status":"invalid","reason":"no_data_until_deadline","return":0.0,"return_value":0.0})
                                w_all.writerow({k:r.get(k,"") for k in fields})
                                if not wrong_written:
                                    wrong_writer=csv.DictWriter(f_wrong,fieldnames=sorted(r.keys())); wrong_writer.writeheader(); wrong_written=True
                                wrong_writer.writerow({k:r.get(k,"") for k in r.keys()}); continue
                        actual_max=float(fut["high"].max()); gain=(actual_max-entry)/(entry+1e-12)
                        if pred_cls>=0:
                            try: cmin,cmax=get_class_return_range(pred_cls,sym,strat)
                            except Exception: cmin,cmax=(0.0,0.0)
                        else: cmin,cmax=(0.0,0.0)
                        reached=gain>=cmin
                        if _now_kst()<deadline and reached:
                            status="v_success" if str(r.get("volatility","")).strip().lower() in ["1","true"] else "success"
                            r.update({"status":status,"reason":f"[조기성공 pred_class={pred_cls}] gain={gain:.3f} (cls_min={cmin}, cls_max={cmax})","return":round(gain,5),"return_value":round(gain,5),"group_id":gid})
                            log_prediction(symbol=sym,strategy=strat,direction=f"평가:{status}",entry_price=entry,target_price=entry*(1+gain),timestamp=_now_kst().isoformat(),model=model,predicted_class=pred_cls,success=True,reason=r["reason"],rate=gain,return_value=gain,volatility=(status=="v_success"),source="평가",label=label,group_id=gid)
                            if model=="meta": update_model_success(sym,strat,model,True)
                            w_all.writerow({k:r.get(k,"") for k in fields})
                            if not eval_written:
                                eval_writer=csv.DictWriter(f_eval,fieldnames=sorted(r.keys())); eval_writer.writeheader(); eval_written=True
                            eval_writer.writerow({k:r.get(k,"") for k in r.keys()}); continue
                        if _now_kst()<deadline and not reached:
                            r.update({"status":"pending","reason":"⏳ 평가 대기 중","return":round(gain,5),"return_value":round(gain,5)})
                            w_all.writerow({k:r.get(k,"") for k in fields}); continue
                        status="success" if reached else "fail"
                        if str(r.get("volatility","")).strip().lower() in ["1","true"]: status="v_success" if status=="success" else "v_fail"
                        r.update({"status":status,"reason":f"[pred_class={pred_cls}] gain={gain:.3f} (cls_min={cmin}, cls_max={cmax})","return":round(gain,5),"return_value":round(gain,5),"group_id":gid})
                        log_prediction(symbol=sym,strategy=strat,direction=f"평가:{status}",entry_price=entry,target_price=entry*(1+gain),timestamp=_now_kst().isoformat(),model=model,predicted_class=pred_cls,success=(status in ["success","v_success"]),reason=r["reason"],rate=gain,return_value=gain,volatility=("v_" in status),source="평가",label=label,group_id=gid)
                        if status in ["fail","v_fail"]:
                            if not check_failure_exists(r): insert_failure_record(r,feature_vector=None)
                        if model=="meta": update_model_success(sym,strat,model,status in ["success","v_success"])
                        w_all.writerow({k:r.get(k,"") for k in fields})
                        if not eval_written:
                            eval_writer=csv.DictWriter(f_eval,fieldnames=sorted(r.keys())); eval_writer.writeheader(); eval_written=True
                        eval_writer.writerow({k:r.get(k,"") for k in r.keys()})
                        if status in ["fail","v_fail"]:
                            if not wrong_written:
                                wrong_writer=csv.DictWriter(f_wrong,fieldnames=sorted(r.keys())); wrong_writer.writeheader(); wrong_written=True
                            wrong_writer.writerow({k:r.get(k,"") for k in r.keys()})
                    except Exception as e:
                        r.update({"status":"invalid","reason":f"exception:{e}","return":0.0,"return_value":0.0})
                        w_all.writerow({k:r.get(k,"") for k in fields})
                        if not wrong_written:
                            wrong_writer=csv.DictWriter(f_wrong,fieldnames=sorted(r.keys())); wrong_writer.writeheader(); wrong_written=True
                        wrong_writer.writerow({k:r.get(k,"") for k in r.keys()})
            shutil.move(tmp,P); print("[✅ 평가 완료] 스트리밍 재작성 성공")
    except FileNotFoundError:
        print(f"[정보] {P} 없음 → 평가 스킵")
    except Exception as e:
        try:
            if tmp and os.path.exists(tmp): os.remove(tmp)
        except Exception: pass
        print(f"[오류] evaluate_predictions 스트리밍 실패 → {e}")

def _combine_windows(calib_stack:np.ndarray,raw_stack:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    eps=1e-12
    mean_c=calib_stack.mean(axis=0); mean_r=raw_stack.mean(axis=0)
    if PREDICT_WINDOW_ENSEMBLE=="mean":
        cc=mean_c; rr=mean_r
    else:
        var_c=calib_stack.var(axis=0); var_r=raw_stack.var(axis=0)
        cc=mean_c/(1.0+ENSEMBLE_VAR_GAMMA*var_c); rr=mean_r/(1.0+ENSEMBLE_VAR_GAMMA*var_r)
        if PREDICT_WINDOW_ENSEMBLE=="mean_var":
            cc=0.5*mean_c+0.5*cc; rr=0.5*mean_r+0.5*rr
    cc=np.nan_to_num(cc,nan=0.0,posinf=0.0,neginf=0.0); rr=np.nan_to_num(rr,nan=0.0,posinf=0.0,neginf=0.0)
    cc=(cc+1e-9); rr=(rr+1e-9)
    cc=cc/(cc.sum()+1e-12); rr=rr/(rr.sum()+1e-12)
    return cc.astype(float),rr.astype(float)

def get_model_predictions(symbol,strategy,models,df,feat_scaled,window_list,recent_freq,regime="unknown"):
    outs=[]; allpreds=[]
    for info in models:
        try:
            pt=info.get("pt_file"); meta_path=info.get("meta_path")
            if not pt or not meta_path: continue
            model_path=os.path.join(MODEL_DIR,pt)
            if os.path.exists(model_path) is False:
                try:
                    p=_stem(pt).split("_")
                    if len(p)>=3:
                        sym,strat,mtype=p[0],p[1],p[2]
                        for e in _KNOWN_EXTS:
                            alt=os.path.join(MODEL_DIR,sym,strat,f"{mtype}{e}")
                            if os.path.exists(alt): model_path=alt; break
                except Exception: pass
            with open(meta_path,"r",encoding="utf-8") as mf: meta=json.load(mf)

            # 파일명 group 태그가 있으면 meta 보정
            fname_grp,_fname_cls=_parse_group_cls_from_filename(model_path)
            if fname_grp is not None:
                meta["group_id"]=int(fname_grp)
            elif "group_id" not in meta or not str(meta.get("group_id")).isdigit():
                meta["group_id"]=_infer_group_id(symbol,strategy)

            if STRICT_SAME_BOUNDS:
                cr_meta=_ranges_from_meta(meta)
                if not (cr_meta and len(cr_meta)>=2):
                    print(f"[STRICT] no class_ranges in meta → {os.path.basename(model_path)} (skip)")
                    continue

            mtype_raw=meta.get("model","lstm"); mtype=_norm_model_type(mtype_raw); gid=meta.get("group_id",0)
            inp_size=int(meta.get("input_size",feat_scaled.shape[1]))
            cr_meta=_ranges_from_meta(meta)
            num_cls=int(meta.get("num_classes",(len(cr_meta) if cr_meta else NUM_CLASSES)))

            preds_c_list,preds_r_list=[],[]; used_windows=[]
            for win in list(dict.fromkeys([int(w) for w in window_list if int(w)>0])):
                if feat_scaled.shape[0]<win:
                    preds_c_list=[]; preds_r_list=[]; used_windows=[]; break
                seq=feat_scaled[-win:]
                if seq.shape[1]<inp_size: seq=np.pad(seq,((0,0),(0,inp_size-seq.shape[1])),mode="constant")
                elif seq.shape[1]>inp_size: seq=seq[:,:inp_size]
                x=torch.tensor(seq,dtype=torch.float32).unsqueeze(0)

                # === 로딩 파이프라인 (다중 폴백) ===
                model=get_model(mtype_raw,input_size=inp_size,output_size=num_cls)
                loaded=load_model_any(model_path,model,ttl_sec=PREDICT_MODEL_LOADER_TTL)
                if loaded is None:
                    # 2차: torch.load 후 주입 시도
                    try:
                        obj=torch.load(model_path,map_location="cpu")
                        if isinstance(obj,dict):
                            try: model.load_state_dict(obj,strict=False); loaded=model
                            except Exception: loaded=None
                        else:
                            loaded=obj
                    except Exception:
                        loaded=None
                if isinstance(loaded,dict) and model is not None:
                    try: model.load_state_dict(loaded,strict=False); model.eval()
                    except Exception: 
                        print(f"[모델 주입 실패] {model_path}")
                        preds_c_list=[]; preds_r_list=[]; break
                elif loaded is None:
                    print(f"[⚠️ 모델 로딩 실패] {model_path}")
                    preds_c_list=[]; preds_r_list=[]; break
                else:
                    if hasattr(loaded,"eval"): model=loaded  # 모듈 자체 로드
                model.to(DEVICE); model.eval()

                with torch.no_grad():
                    logits=model(x.to(DEVICE))
                    logits=torch.nan_to_num(logits)
                    probs=F.softmax(logits,dim=1)
                    probs=torch.nan_to_num(probs).squeeze().cpu().numpy()
                probs=np.nan_to_num(probs,nan=0.0,posinf=0.0,neginf=0.0)+1e-9
                probs=probs/(probs.sum()+1e-12)
                cprobs=apply_calibration(probs,symbol=symbol,strategy=strategy,regime=regime,model_meta=meta).astype(float)
                cprobs=np.nan_to_num(cprobs,nan=0.0,posinf=0.0,neginf=0.0)+1e-9
                cprobs=cprobs/(cprobs.sum()+1e-12)
                preds_c_list.append(cprobs); preds_r_list.append(probs); used_windows.append(int(win))

            if not preds_c_list: 
                print(f"[모델 스킵] 예측값 없음 → {os.path.basename(model_path)}")
                continue

            calib_stack=np.vstack(preds_c_list); raw_stack=np.vstack(preds_r_list)
            comb_c,comb_r=_combine_windows(calib_stack,raw_stack)

            outs.append({"raw_probs":comb_r,"calib_probs":comb_c,"predicted_class":int(np.argmax(comb_c)),"group_id":gid,"model_type":mtype,"model_path":model_path,"val_f1":None,"symbol":symbol,"strategy":strategy,"meta":meta,"window_ensemble":{"mode":PREDICT_WINDOW_ENSEMBLE,"gamma":ENSEMBLE_VAR_GAMMA,"wins":used_windows}})
            entry_price=df["close"].iloc[-1]
            allpreds.append({"class":int(np.argmax(comb_c)),"probs":comb_c,"entry_price":float(entry_price),"num_classes":num_cls,"group_id":gid,"model_name":mtype,"model_symbol":symbol,"symbol":symbol,"strategy":strategy})
        except Exception as e:
            print(f"[❌ 모델 예측 실패] {info} → {e}"); continue
    return outs,allpreds

def _get_price_df_for_eval(symbol,strategy): return get_kline_by_strategy(symbol,strategy)
def run_evaluation_once(): evaluate_predictions(_get_price_df_for_eval)
def run_evaluation_loop(interval_minutes=None):
    try: iv=int(os.getenv("EVAL_INTERVAL_MIN","30")) if interval_minutes is None else int(interval_minutes)
    except Exception: iv=30
    iv=max(1,iv); print(f"[EVAL_LOOP] 시작 — {iv}분 주기")
    while True:
        try: run_evaluation_once()
        except Exception as e: print(f"[EVAL_LOOP] evaluate_predictions 예외 → {e}")
        time.sleep(iv*60)

if __name__=="__main__":
    res=predict("BTCUSDT","단기",source="테스트"); print(res)
    try:
        df=pd.read_csv(PREDICTION_LOG_PATH,encoding="utf-8-sig")
        print("[✅ prediction_log.csv 상위 20줄 출력]"); print(df.head(20))
    except Exception as e:
        print(f"[오류] prediction_log.csv 로드 실패] {e}")
    if str(os.getenv("EVAL_LOOP","0")).strip()=="1": run_evaluation_loop()
