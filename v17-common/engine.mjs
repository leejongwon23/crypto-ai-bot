import crypto from 'node:crypto';
const RX_HEADER=/^==========\s*(BTC|ETH)\s*\((BTCUSDT|ETHUSDT)\)\s*==========\s*$/gm;
const FINISHED=new Set(['TP3_DONE','STOPPED','CANCELLED','EARLY_EXIT','REVERSE_EXIT']);
const q=(x,n=2)=>Number(Number(x).toFixed(n));
const found=(text,re)=>{const m=text.match(re); return m?m[1]:null};
const val=(text,re)=>{const t=found(text,re);return t==null?null:Number(t)};
export function sha(x){return crypto.createHash('sha256').update(typeof x==='string'?x:stable(x)).digest('hex')}
export function stable(x){if(Array.isArray(x))return `[${x.map(stable).join(',')}]`;if(x&&typeof x==='object')return `{${Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+stable(x[k])).join(',')}}`;return JSON.stringify(x)}
function parseOne(source){
 const symbol=found(source,/^==========\s*(BTC|ETH)\s*\((BTCUSDT|ETHUSDT)\)/m)?.concat('USDT');
 const timestamp=found(source,/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m);
 const last=val(source,/^lastPrice:\s*([\d.]+)/m);
 const mark=val(source,/^lastPrice:[^\n]*?\bmark:\s*([\d.]+)/m);
 const e={},macd={},cvd={};for(const tf of ['1m','5m','15m','30m','1h','4h','1D']){
  const m=source.match(new RegExp('^'+tf+':\\s*n=\\d+[^\\n]*$', 'm'));
  e[tf]=m?found(m[0],/\balign=(BULL|BEAR)/):null;
  macd[tf]=m?val(m[0],/\bhist=([-+\d.]+)/):null;
 }
 const cm=source.match(/^cvd_net_5m\/15m\/30m\/1h:\s*([-+\d.]+)\s*\/\s*([-+\d.]+)\s*\/\s*([-+\d.]+)\s*\/\s*([-+\d.]+)/m);
 if(cm) ['5m','15m','30m','1h'].forEach((k,i)=>cvd[k]=Number(cm[i+1]));
 const vwap24=val(source,/^vwap_24h:\s*([\d.]+)/m),vwap6=val(source,/^vwap_24h:[^\n]*?vwap_session_6h:\s*([\d.]+)/m);
 const am=source.match(/^15m:\s*atr14=([\d.]+)/m), atr15=am?Number(am[1]):null;
 const swings={},v24={}; for(const tf of ['15m','1h','4h','1D']){
  const m=source.match(new RegExp('^'+tf+':\\s*atr14=[^\\n]*?swingHigh=([\\d.]+)\\s*\\|\\s*swingLow=([\\d.]+)','m'));
  if(m)swings[tf]={high:Number(m[1]),low:Number(m[2])};
 }
 for(const [key,re] of Object.entries({high:/^prev_day_high\/low:\s*([\d.]+)/m,low:/^prev_day_high\/low:\s*[\d.]+\s*\/\s*([\d.]+)/m}))v24[key]=val(source,re);
 const sec=source.split('[10] PRICE PATH SINCE LAST RUN (AUTO)')[1]?.split('[7] HISTORICAL FLOW SNAPSHOT (AUTO)')[0]||null;
 let path=null;
 if(sec){
  const bounds=sec.match(/^path_high\/low:\s*([\d.]+)\s*@\s*([\d-]+ [\d:]+)\s*\/\s*([\d.]+)\s*@\s*([\d-]+ [\d:]+)/m);
  const ends=sec.match(/^start_price:\s*([\d.]+)\s*@\s*([\d-]+ [\d:]+)\s*\|\s*end_price:\s*([\d.]+)\s*@\s*([\d-]+ [\d:]+)/m);
  if(bounds&&ends)path={high:Number(bounds[1]),highAt:bounds[2],low:Number(bounds[3]),lowAt:bounds[4],startPrice:Number(ends[1]),startAt:ends[2],endPrice:Number(ends[3]),endAt:ends[4],bars:val(sec,/^lookback_min:[^\n]*?bars:\s*(\d+)/m)};
 }
 const warnings=[]; if(!timestamp||last==null)warnings.push('MISSING_REQUIRED_PRICE_OR_TIME');if(!path)warnings.push('MISSING_PRICE_PATH');if(!cm)warnings.push('MISSING_CVD');
 if(path&&(path.low>path.high||path.endPrice<path.low||path.endPrice>path.high))warnings.push('INVALID_PATH_RANGE');
 if(path&&Math.abs(path.endPrice-last)/last>0.0005)warnings.push('PATH_PRICE_MISMATCH');
 return {symbol,timestamp,last,mark,ema:e,macd,cvd,vwap24,vwap6,atr15,swings,prevDay:v24,path,warnings};
}
export function parse(raw){
 if(typeof raw!=='string'||raw.length>450000)throw Error('raw snapshot must be text <=450000 characters');
 RX_HEADER.lastIndex=0;const m=[...raw.matchAll(RX_HEADER)];if(m.length!==2||!m.some(x=>x[2]==='BTCUSDT')||!m.some(x=>x[2]==='ETHUSDT'))throw Error('require exactly one BTC and one ETH complete snapshot');
 const parts=m.map((x,i)=>parseOne(raw.slice(x.index,i===m.length-1?undefined:m[i+1].index)));
 for(const p of parts)if(!p.symbol||!p.timestamp||!Number.isFinite(p.last)||p.last<=0)throw Error('invalid market record');
 return parts.sort((a,b)=>a.symbol.localeCompare(b.symbol));
}
export function metrics(s){
 const e=s.direction==='SHORT'?Number(s.entryLow):Number(s.entryHigh), sl=Number(s.stop); if(!(e>0&&sl>0))throw Error('invalid prices');
 const targets=['tp1','tp2','tp3'].map(k=>Number(s[k]));
 if(s.direction==='SHORT' && !(sl>Number(s.entryHigh)&&Number(s.entryLow)>targets[0]&&targets[0]>targets[1]&&targets[1]>targets[2]))throw Error('short price ordering violation');
 if(s.direction==='LONG'&&!(sl<Number(s.entryLow)&&Number(s.entryHigh)<targets[0]&&targets[0]<targets[1]&&targets[1]<targets[2]))throw Error('long price ordering violation');
 const risk=Math.abs(sl-e);return {referenceEntry:e,stopPct:q(-risk/e*100),tp1Pct:q(Math.abs(targets[0]-e)/e*100),tp2Pct:q(Math.abs(targets[1]-e)/e*100),tp3Pct:q(Math.abs(targets[2]-e)/e*100),rr2:q(Math.abs(targets[1]-e)/risk)};
}
export function verifySignal(s){
 if(!s||!s.id||!['BTCUSDT','ETHUSDT'].includes(s.symbol)||!['SHORT','LONG'].includes(s.direction)||!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(s.issuedAt))throw Error('invalid original signal');
 if(!Number.isFinite(Number(s.entryLow))||!Number.isFinite(Number(s.entryHigh))||Number(s.entryLow)>Number(s.entryHigh))throw Error('invalid entry');metrics(s);return true;
}
const epoch=s=>Date.parse(s.replace(' ','T')+'+09:00');
export function auditSignal(m,s){
 verifySignal(s);
 let state=s.status||'PENDING';const old=state;
 const facts={entry:false,stop:false,tp1:false,tp2:false,tp3:false};const p=m.path;
 if(FINISHED.has(state))return {id:s.id,state,stateBefore:old,confidence:'LOCKED',facts,metrics:metrics(s),warnings:[]};
 const warnings=[...m.warnings];
 if(!p) return {id:s.id,state:'UNVERIFIED',stateBefore:old,confidence:'UNVERIFIED',facts,metrics:metrics(s),warnings};
 const overlap=Number(p.low)<=Number(s.entryHigh)&&Number(p.high)>=Number(s.entryLow);
 facts.entry=overlap;facts.stop=s.direction==='SHORT'?p.high>=s.stop:p.low<=s.stop;
 for(const k of ['tp1','tp2','tp3'])facts[k]=s.direction==='SHORT'?p.low<=s[k]:p.high>=s[k];
 if(epoch(p.endAt)<epoch(s.issuedAt))return {id:s.id,state,stateBefore:old,confidence:'PRE_ISSUE',facts:{},metrics:metrics(s),warnings:['OBSERVATION_PRECEDES_ISSUANCE']};
 const anchor=s.verifiedUntil||s.issuedAt;
 if(epoch(p.startAt)>epoch(anchor)+65000)warnings.push('MISSING_CHRONOLOGICAL_PATH');
 if(epoch(p.startAt)<epoch(s.issuedAt))warnings.push('ISSUANCE_WITHIN_AGGREGATE_SEGMENT');
 if(warnings.length)return {id:s.id,state:'UNVERIFIED',stateBefore:old,confidence:'UNVERIFIED',facts,metrics:metrics(s),warnings};
 if(state==='PENDING'){
  if(!facts.entry)return {id:s.id,state,stateBefore:old,confidence:'VERIFIED_SEGMENT',facts,metrics:metrics(s),warnings};
  if(facts.stop||facts.tp1)return {id:s.id,state:'UNVERIFIED',stateBefore:old,confidence:'EVENT_ORDER_UNKNOWN',facts,metrics:metrics(s),warnings:['ENTRY_AND_EXIT_OR_TARGET_IN_SAME_AGGREGATE']};
  state='ENTERED';
 } else if(['ENTERED','TP1','TP2'].includes(state)){
  if(facts.stop && facts.tp1)return {id:s.id,state:'UNVERIFIED',stateBefore:old,confidence:'EVENT_ORDER_UNKNOWN',facts,metrics:metrics(s),warnings:['STOP_AND_TARGET_IN_SAME_AGGREGATE']};
  if(facts.stop)state='STOPPED';
  else if(facts.tp3)state='TP3_DONE';
  else if(facts.tp2)state='TP2';
  else if(facts.tp1 && state==='ENTERED')state='TP1';
 }
 return {id:s.id,state,stateBefore:old,confidence:'VERIFIED_SEGMENT',facts,metrics:metrics(s),warnings};
}
function evidence(m,side){
 const bias=side==='LONG'?'BULL':'BEAR',flowSign=side==='LONG'?1:-1;
 const intervals=['1D','4h','1h','30m','15m','5m'];
 const t=Object.fromEntries(intervals.map(tf=>[tf,m.ema[tf]===bias]));
 const c=Object.fromEntries(['5m','15m','30m','1h'].map(tf=>[tf, m.cvd[tf]==null?null:m.cvd[tf]*flowSign>0]));
 const vwap=m.vwap24==null||m.vwap6==null?null:(side==='LONG'?m.last>m.vwap24&&m.last>m.vwap6:m.last<m.vwap24&&m.last<m.vwap6);
 return {side,trendSupport:t,cvdSupport:c,vwapSupport:vwap,conflicts:intervals.filter(tf=>m.ema[tf]&&m.ema[tf]!==bias),currentPrice:m.last};
}
function candidate(m,side){
 const ev=evidence(m,side);const why=[];
 if(!['1D','4h','15m','30m','5m'].every(tf=>ev.trendSupport[tf]))why.push('TIMEFRAME_ALIGNMENT_NOT_MET');
 if(!['5m','15m'].every(tf=>ev.cvdSupport[tf]))why.push('SHORT_TERM_CVD_MISMATCH');
 if(!ev.vwapSupport)why.push('VWAP_POSITION_CONFLICT');
 if(!m.path||m.warnings.length)why.push('DATA_INTEGRITY_NOT_VERIFIED');
 const l=m.last,ss=m.swings; if(!ss['15m']||!ss['1h']||!m.prevDay.high||!m.prevDay.low||!m.vwap24||!m.vwap6)why.push('STRUCTURE_INCOMPLETE');
 if(why.length)return {side,decision:'NO_NEW',evidence:ev,why};
 const stop=side==='SHORT'?Math.max(m.path.high,m.vwap24,m.vwap6,ss['15m'].high)+0.01:Math.min(m.path.low,m.vwap24,m.vwap6,ss['15m'].low)-0.01;
 const ordered=[ss['15m'].low,ss['1h'].low,m.prevDay.low,ss['4h']?.low,ss['1D']?.low].filter(Number.isFinite);
 const filtered=side==='SHORT'?ordered.filter(v=>v<l).sort((a,b)=>b-a):[ss['15m'].high,ss['1h'].high,m.prevDay.high,ss['4h']?.high,ss['1D']?.high].filter(v=>Number.isFinite(v)&&v>l).sort((a,b)=>a-b);
 const uniq=[...new Set(filtered)];if(uniq.length<3)return {side,decision:'NO_NEW',evidence:ev,why:['NOT_ENOUGH_STRUCTURAL_TARGETS']};
 const s={direction:side,entryLow:l,entryHigh:l,stop:q(stop),tp1:uniq[0],tp2:uniq[1],tp3:uniq[2]};
 let risk;try{risk=metrics(s)}catch{return {side,decision:'NO_NEW',evidence:ev,why:['PRICE_ORDER_INVALID']}}
 if(risk.rr2<2)return {side,decision:'NO_NEW',evidence:ev,why:['TP2_RR_BELOW_2'],reference:{entry:l,stop:q(stop),tp2:uniq[1],rr2:risk.rr2}};
 return {side,decision:'QUALIFIED_COMPANION',evidence:ev,why:['ALL_SPECIFIED_COMPANION_GATES_MET'],plan:{...s,...risk}};
}
export function analyze(raw,ledger=[]){
 const markets=parse(raw),details={};
 const returns=[];
 for(const m of markets){
  const current=ledger.filter(x=>x.symbol===m.symbol&&!FINISHED.has(x.status));
  const sides={LONG:candidate(m,'LONG'),SHORT:candidate(m,'SHORT')};
  const recommendation=current.length?'MANAGE_EXISTING':(sides.LONG.decision==='QUALIFIED_COMPANION'&&sides.SHORT.decision!=='QUALIFIED_COMPANION'?'LONG':sides.SHORT.decision==='QUALIFIED_COMPANION'&&sides.LONG.decision!=='QUALIFIED_COMPANION'?'SHORT':'NO_NEW');
  details[m.symbol]={timestamp:m.timestamp,last:m.last,trend:{mid:m.ema['1D']===m.ema['4h']?(m.ema['1D']==='BEAR'?'SHORT':'LONG'):'MIXED',short:m.ema['5m']===m.ema['15m']?(m.ema['5m']==='BEAR'?'SHORT':'LONG'):'MIXED'},fourDirections:sides,officialSignals:current.map(s=>auditSignal(m,s)),recommendation,selectedPlan:recommendation==='LONG'||recommendation==='SHORT'?sides[recommendation].plan:null};
 }
 const [btc,eth]=markets;let coupling={status:'UNVERIFIED'};
 if(btc.path&&eth.path&&Math.abs(epoch(btc.path.startAt)-epoch(eth.path.startAt))<=120000&&Math.abs(epoch(btc.path.endAt)-epoch(eth.path.endAt))<=120000){
  const a=q((btc.path.endPrice/btc.path.startPrice-1)*100),b=q((eth.path.endPrice/eth.path.startPrice-1)*100);
  coupling={status:'OBSERVED',btcPct:a,ethPct:b,ethMinusBtcPctPoint:q(b-a),sameDirection:Math.sign(a)===Math.sign(b)};
 }
 return {engine:'V1.7_SHARED_VERDICT_V1',sourcePromptSha256:'dd53a208898124cb815433caca4348b95347d2bd06a49f7cda4b6cc57bcf8e52',policy:'DETERMINISTIC_COMPANION_NOT_V17_FULL_EQUIVALENCE',results:details,coupling,disclaimer:'No proof of profitability or exact equivalence to discretionary V1.7; no exchange order executed.'};
}