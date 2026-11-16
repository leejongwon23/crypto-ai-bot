/**
 * Empirical direction / trajectory research engine (non-trading).
 * Only timestamped user-submitted V1.7 observations and complete, CLOSED
 * Bybit-labelled 1m paths can supply labels. This is not exchange-authenticated.
 * No fabricated future candles, probabilities, position fills or guaranteed wins.
 */
import {kstMs, floorMinute} from './price-path-recovery.mjs';
const TF=[15,30,60];
const clip=n=>Math.max(-4,Math.min(4,n));
const n=(s,re)=>{const m=re.exec(s);return m?Number(m[1]):NaN;};
const ok=x=>Number.isFinite(x);
const rd=(x,d=4)=>Math.round(x*10**d)/10**d;
const minSample=35,minTrain=20,minTest=10;
const section=(raw,sym)=>{
 const i=raw.indexOf('========== '+sym+' ('+sym+'USDT) ==========');
 if(i<0)return '';
 const j=sym==='BTC'?raw.indexOf('========== ETH (ETHUSDT) ==========',i+10):-1;
 return raw.slice(i,j>=0?j:undefined);
};
const pt=(s,tf,key)=>n(s,new RegExp('^'+tf+':[^\\n]*\\b'+key+'=([-\\d.]+)','m'));
function change(s){
 const m=/^price_change_5m\/15m\/30m\/1h\/4h:\s*([-\d.]+)%\s*\/\s*([-\d.]+)%\s*\/\s*([-\d.]+)%\s*\/\s*([-\d.]+)%/m.exec(s);
 return m?m.slice(1).map(Number):null;
}
function cvdImbalance(s,tf){
 const re=new RegExp('^cvd_net_snapshot\\('+tf+'m\\):[^\\n]*\\bbuy:\\s*([\\d.]+)\\s*\\|\\s*sell:\\s*([\\d.]+)','m');
 const m=re.exec(s);if(!m)return NaN;
 const buy=Number(m[1]),sell=Number(m[2]);
 return buy+sell>0?(buy-sell)/(buy+sell):NaN;
}
export function extractObservation(raw,symbol){
 const s=section(raw,symbol);if(!s)return null;
 const t=/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(s)?.[1];
 if(!t)return null;
 const price=n(s,/^lastPrice:\s*([\d.]+)/m),atr=n(s,/^15m:\s*atr14=([\d.]+)/m);
 const ch=change(s),oi=n(s,/^oi_change_5m\/15m\/30m\/1h\/4h:[^\n]*?\/\s*([-\d.]+)%\s*\/\s*[-\d.]+%$/m);
 const vwap=n(s,/^vwap_24h:[^\n]*vwap_session_6h:\s*([\d.]+)/m);
 if(!(price>0&&atr>0)||!ch||!ok(vwap)||!ok(oi))return null;
 const atrPct=100*atr/price;
 const f=[
  ch[0]/atrPct,ch[1]/atrPct,ch[3]/atrPct,
  (price-pt(s,'5m','ema26'))/atr,(price-pt(s,'15m','ema26'))/atr,
  (price-vwap)/atr,
  cvdImbalance(s,5)/0.25,cvdImbalance(s,15)/0.25,cvdImbalance(s,30)/0.25,
  oi/atrPct
 ];
 if(!f.every(ok))return null;
 return {symbol,atMs:kstMs(t),price,atr,features:f.map(clip),source:'USER_SUBMITTED_MARKET_SNAPSHOT'};
}
export function extractClosedMinuteCandles(raw,symbol){
 const s=section(raw,symbol),anchor=s.indexOf('[10B] BYBIT 1M GAP EVIDENCE (AUTO)');
 if(anchor<0)return [];
 const p=s.slice(anchor),h=new RegExp('^source:\\s*Bybit linear '+symbol+'USDT\\s*\\|\\s*coverage:\\s*COMPLETE\\s*\\|\\s*expected:\\s*(\\d+)\\s*\\|\\s*received:\\s*(\\d+)','m').exec(p);
 const range=/^minute_range:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*~\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*KST/m.exec(p);
 const hdr=/^minute_OHLC_KST:\s*timestamp,open,high,low,close\s*$/m.exec(p);
 const at=/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(s)?.[1];
 if(!h||!range||!hdr||!at)return [];
 const expected=Number(h[1]);
 if(expected<2||expected>1500||expected!==Number(h[2]))return [];
 const rows=[],pat=/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),([\d.]+),([\d.]+),([\d.]+),([\d.]+)$/;
 for(const line of p.slice(hdr.index+hdr[0].length).trimStart().split(/\r?\n/)){
  const m=pat.exec(line.trim());if(!m)break;
  const t=kstMs(m[1]),a=m.slice(2).map(Number),[open,high,low,close]=a;
  if(t%60000!==0||a.some(x=>!ok(x)||x<=0)||low>Math.min(open,close)||high<Math.max(open,close)||low>high)return [];
  if(rows.length&&t!==rows.at(-1).t+60000)return [];
  rows.push({t,open,high,low,close});
 }
 if(rows.length!==expected||rows[0].t!==kstMs(range[1])||rows.at(-1).t!==kstMs(range[2]))return [];
 // The first boundary may include pre-collection prices, last is live/incomplete.
 const firstPartial=/^first_minute_partial:\s*true/m.test(p);
 const cutoff=floorMinute(kstMs(at));
 return rows.filter((r,i)=>r.t<cutoff&&(!firstPartial||i>0));
}
const quantile=(xs,q)=>{
 if(!xs.length)return null;
 const a=[...xs].sort((x,y)=>x-y),pos=(a.length-1)*q,i=Math.floor(pos);
 return rd(a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(pos-i));
};
function closedPath(obs,bars,h){
 const start=floorMinute(obs.atMs)+60000;
 const path=[];for(let i=0;i<h;i++){const c=bars.get(start+i*60000);if(!c)return null;path.push(c);}
 return path;
}
function makeLabeled(rows,bars,h){
 const out=[];let latestAccepted=-Infinity;
 for(const o of rows.sort((a,b)=>a.atMs-b.atMs)){
  if(!o||!ok(o.atMs)||o.atMs-latestAccepted<h*60000||!(o.price>0&&o.atr>0)||!Array.isArray(o.features)||o.features.length!==10||!o.features.every(ok))continue;
  const path=closedPath(o,bars,h);if(!path)continue;
  const p=path.at(-1),hi=Math.max(...path.map(x=>x.high)),lo=Math.min(...path.map(x=>x.low));
  out.push({...o,targetEndMs:path.at(-1).t+60000,moveR:(p.close-o.price)/o.atr,
   highR:(hi-o.price)/o.atr,lowR:(lo-o.price)/o.atr,path});
  latestAccepted=o.atMs;
 }
 return out;
}
const dist=(a,b)=>Math.sqrt(a.reduce((sum,v,i)=>sum+(v-b[i])**2,0)/a.length);
function neighbors(train,q,k=20){
 return train.map(x=>({x,d:dist(x.features,q.features)})).sort((a,b)=>a.d-b.d||b.x.atMs-a.x.atMs).slice(0,k).map(z=>z.x);
}
const probUp=xs=>(xs.filter(x=>x.moveR>0).length+1)/(xs.length+2); // Beta(1,1) smoothed empirical frequency.
function validateForward(rows){
 if(rows.length<minSample)return {status:'INSUFFICIENT_INDEPENDENT_SAMPLES',samples:rows.length,minRequired:minSample};
 const held=rows.slice(minTrain),scores=[];
 for(const q of held){
  const past=rows.filter(x=>x.targetEndMs<=q.atMs&&x.atMs<q.atMs);
  if(past.length<minTrain)continue;
  const n=neighbors(past,q);
  if(n.length<minTrain)continue;
  const up=q.moveR>0?1:0;
  const p=probUp(n),base=probUp(past);
  scores.push({p,base,up});
 }
 if(scores.length<minTest)return {status:'INSUFFICIENT_WALK_FORWARD',samples:rows.length,tested:scores.length,minRequired:minTest};
 const brier=xs=>xs.reduce((a,x)=>a+(x.p-x.up)**2,0)/xs.length;
 const modelBrier=brier(scores),baseBrier=brier(scores.map(x=>({...x,p:x.base})));
 return {status:modelBrier<baseBrier?'OUT_OF_SAMPLE_EDGE_OBSERVED':'NO_OUT_OF_SAMPLE_EDGE',
  samples:rows.length,tested:scores.length,modelBrier:rd(modelBrier),historicalBaseBrier:rd(baseBrier),
  note:'Chronological walk-forward comparison with prior-only labels, not proof of future profit. Unverified exchange provenance.'};
}
function pathComparison(path,original,candidate){
 const delta=(candidate.entry-original.price)/original.atr;
 const s=(candidate.stop-original.price)/original.atr;
 const tp1=(candidate.tp1-original.price)/original.atr;
 const tp2=(candidate.tp2-original.price)/original.atr;
 const side=candidate.side,short=side==='SHORT';
 let entered=false,reachedTP1=false;
 for(const c of path){
  const lo=(c.low-original.price)/original.atr,hi=(c.high-original.price)/original.atr;
  const hit=x=>lo<=x&&hi>=x;
  const stop=short?hi>=s:lo<=s;
  const target1=short?lo<=tp1:hi>=tp1;
  const target2=short?lo<=tp2:hi>=tp2;
  if(!entered){
   if(!hit(delta))continue;
   if(stop||target1)return 'AMBIGUOUS_ENTRY_MINUTE';
   entered=true;continue;
  }
  if(stop&&target1)return 'AMBIGUOUS_STOP_TARGET';
  if(stop)return reachedTP1?'STOP_AFTER_TP1':'STOP_BEFORE_TP1';
  if(target2)return 'TP2_BEFORE_STOP';
  if(target1)reachedTP1=true;
 }
 return !entered?'NO_ENTRY':reachedTP1?'TP1_ONLY':'ENTERED_UNRESOLVED';
}
function pathFrequencies(nearest,candidate,q){
 const result={n:nearest.length,source:'HISTORIC_1M_USER_SUBMITTED_NOT_EXCHANGE_AUTHENTICATED',outcomes:{}};
 for(const x of nearest){
  // Same normalized ATR-relative plan replayed in genuinely prior future paths.
  const virtual={side:candidate.side,
   entry:x.price+(candidate.entry-q.price)/q.atr*x.atr,
   stop:x.price+(candidate.stop-q.price)/q.atr*x.atr,
   tp1:x.price+(candidate.tp1-q.price)/q.atr*x.atr,
   tp2:x.price+(candidate.tp2-q.price)/q.atr*x.atr};
  const k=pathComparison(x.path,x,virtual);
  result.outcomes[k]=(result.outcomes[k]||0)+1;
 }
 return result;
}
export function buildEmpiricalForecast({raw,symbol,history=[],candles=[],maxHorizon=60}){
 const query=extractObservation(raw,symbol);
 if(!query)return {symbol,status:'SOURCE_FEATURES_INCOMPLETE'};
 const bars=new Map(candles.filter(x=>ok(x.t)&&x.t<floorMinute(query.atMs)).map(x=>[x.t,x]));
 const observed=history.filter(x=>x.symbol===symbol&&x.atMs<query.atMs&&x.atMs>=query.atMs-14*86400000&&x.features?.length===10);
 const horizons={},candidates={};
 for(const h of TF.filter(x=>x<=maxHorizon)){
  const eligible=observed.filter(x=>x.atMs+h*60000<=query.atMs);
  const rows=makeLabeled(eligible,bars,h);
  const validated=validateForward(rows);
  if(rows.length<minSample||!['OUT_OF_SAMPLE_EDGE_OBSERVED','NO_OUT_OF_SAMPLE_EDGE'].includes(validated.status)){
   horizons[h]={...validated,measuredDirection:null};continue;
  }
  const n=neighbors(rows.filter(x=>x.targetEndMs<=query.atMs),query);
  const moves=n.map(x=>x.moveR);
  horizons[h]={...validated,measuredDirection:validated.status==='OUT_OF_SAMPLE_EDGE_OBSERVED'?(probUp(n)>0.5?'UP':'DOWN'):'UNRELIABLE',
   neighborCount:n.length,
   neighborUpCount:n.filter(x=>x.moveR>0).length,
   empiricalUpFrequency:rd(n.filter(x=>x.moveR>0).length/n.length),
   distributionAtr:{q10:quantile(moves,.1),median:quantile(moves,.5),q90:quantile(moves,.9)},
   projectedCloseReference:{q10:rd(query.price+quantile(moves,.1)*query.atr),median:rd(query.price+quantile(moves,.5)*query.atr),q90:rd(query.price+quantile(moves,.9)*query.atr)},
   historicalHighLowAtr:{highMedian:quantile(n.map(x=>x.highR),.5),lowMedian:quantile(n.map(x=>x.lowR),.5)},
   notice:'Empirical analogue frequencies and conditional range, NOT guaranteed path or calibrated exchange trade prediction.'};
  if(h===60)candidates[60]={rows:n,features:query};
 }
 return {symbol,atMs:query.atMs,spot:query.price,atr15:query.atr,status:'EMPIRICAL_RESEARCH',
  source:'USER_SUBMITTED_BYBIT_LABELLED_SNAPSHOT_AND_1M_ONLY',horizons,
  studyCandidate(candidate,h=60){
   const ready=candidates[h];
   if(!ready)return {status:'INSUFFICIENT_VALIDATED_HISTORY'};
   return {status:'HISTORICAL_PATH_FREQUENCIES',...pathFrequencies(ready.rows,candidate,query)};
  }};
}
// JSON-safe projection for server responses; runtime evaluator is internal only.
export function publicForecast(forecast,candidates=[]){
 const {symbol,atMs,spot,atr15,status,source,horizons}=forecast;
 const out={symbol,atMs,spot,atr15,status,source,horizons,candidates:[]};
 for(const c of candidates){
  const r=forecast.studyCandidate?.(c,60);
  out.candidates.push({side:c.side,entry:c.entry,stop:c.stop,tp1:c.tp1,tp2:c.tp2,
   sourceAnchor:c.entryAnchor,distanceAtr:c.priceDistanceAtr,evidence:r});
 }
 return out;
}
