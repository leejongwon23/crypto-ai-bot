// Public market-data recovery for V1.7. No exchange account credentials, orders,
// policy edits or guesswork. Recovery is proof only when every 1m interval exists.
const OFFSET=9*3600000;
export const kstMs=s=>{
 if(typeof s!=='string'||!/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(s))throw Error('INVALID_KST_TIME');
 const n=Date.parse(s.replace(' ','T')+'+09:00');
 if(!Number.isFinite(n))throw Error('INVALID_KST_TIME');
 return n;
};
export const kstTime=n=>{
 const d=new Date(n+OFFSET);
 return d.toISOString().slice(0,19).replace('T',' ');
};
export const floorMinute=t=>Math.floor(t/60000)*60000;
export function rawMarketWindows(raw){
 const out={};
 for(const symbol of ['BTC','ETH']){
  const begin=raw.indexOf('========== '+symbol+' ('+symbol+'USDT) ==========');
  if(begin<0)throw Error('MISSING_MARKET_'+symbol);
  const end=symbol==='BTC'?raw.indexOf('========== ETH (ETHUSDT) ==========',begin+1):raw.length;
  const part=raw.slice(begin,end<0?undefined:end);
  const stamp=/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(part);
  const path=/^start_price:\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*\|\s*end_price:\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(part);
  if(!stamp||!path)throw Error('MISSING_PRICE_PATH_'+symbol);
  out[symbol]={timestamp:stamp[1],pathStart:path[2],pathEnd:path[4]};
 }
 return out;
}
export function parseBybitCandles(body,symbol){
 if(body?.retCode!==0||!Array.isArray(body?.result?.list))throw Error('BYBIT_RESPONSE_INVALID_'+symbol);
 const candles=body.result.list.map(a=>{
  if(!Array.isArray(a)||a.length<5)throw Error('BYBIT_CANDLE_SHAPE_INVALID');
  const [start,open,high,low,close]=a.slice(0,5).map(Number);
  if(![start,open,high,low,close].every(Number.isFinite)||start%60000!==0||
   ![open,high,low,close].every(x=>x>0)||low>Math.min(open,close)||high<Math.max(open,close))
   throw Error('BYBIT_CANDLE_VALUES_INVALID');
  return {start,open,high,low,close};
 });
 candles.sort((a,b)=>a.start-b.start);
 for(let i=1;i<candles.length;i++)if(candles[i].start===candles[i-1].start)
  throw Error('BYBIT_DUPLICATE_MINUTE');
 return candles;
}
export function verifyCandleCoverage(candles,fromMs,toMs){
 const start=floorMinute(fromMs),end=floorMinute(toMs);
 if(!Array.isArray(candles)||!candles.length||end<start)return false;
 const map=new Map(candles.map(x=>[x.start,x]));
 for(let t=start;t<=end;t+=60000)if(!map.has(t))return false;
 return true;
}
// Both hosts are official Bybit mainnet REST endpoints. Never replace exchange
// OHLC with another venue's prices when resolving an original Bybit signal.
export const BYBIT_PUBLIC_ENDPOINTS=Object.freeze([
 'https://api.bybit.com','https://api.bytick.com'
]);
export async function fetchBybitCandles({symbol,fromMs,toMs,fetcher=globalThis.fetch,timeoutMs=2500,maxMinutes=1000}){
 if(!['BTC','ETH'].includes(symbol))throw Error('INVALID_RECOVERY_SYMBOL');
 if(toMs<fromMs||toMs-fromMs>maxMinutes*60000)throw Error('RECOVERY_WINDOW_OUT_OF_RANGE');
 const start=floorMinute(fromMs),end=floorMinute(toMs),count=(end-start)/60000+1;
 if(count>1000)throw Error('RECOVERY_TOO_MANY_MINUTES');
 const failures=[];
 for(const base of BYBIT_PUBLIC_ENDPOINTS){
  const url=new URL('/v5/market/kline',base);
  url.searchParams.set('category','linear');url.searchParams.set('symbol',symbol+'USDT');
  url.searchParams.set('interval','1');url.searchParams.set('start',String(start));
  url.searchParams.set('end',String(end+60000-1));url.searchParams.set('limit',String(count));
  const ac=new AbortController();
  const timeout=setTimeout(()=>ac.abort(),timeoutMs);
  try{
   const result=await fetcher(url.toString(),{signal:ac.signal,headers:{accept:'application/json'}});
   if(!result.ok)throw Error('BYBIT_HTTP_'+result.status);
   const data=await result.json();
   const candles=parseBybitCandles(data,symbol);
   if(!verifyCandleCoverage(candles,fromMs,toMs))throw Error('BYBIT_INCOMPLETE_1M_COVERAGE');
   return candles.filter(c=>c.start>=start&&c.start<=end);
  }catch(e){
   failures.push(new URL(base).hostname+':'+String(e?.message||e));
  }finally{clearTimeout(timeout);}
 }
 throw Error('BYBIT_PUBLIC_1M_ALL_ENDPOINTS_FAILED: '+failures.join(' | '));
}

// Compute only mechanical crossings. Intraminute stop/TP conflicts remain unresolved.
export function auditMinuteCandles(signal,candles,{initialStatus=null,afterMs=null}={}){
 const o=signal.original||signal;
 const side=o.direction,live=initialStatus||signal.status;
 if(!['LONG','SHORT'].includes(side))throw Error('INVALID_SIGNAL_DIRECTION');
 if(!['PENDING','NEW_ENTRY','ENTERED','TP1','TP2'].includes(live))throw Error('INVALID_AUDIT_START_STATE');
 let status=live;
 const events=[];
 const checked=candles.filter(c=>afterMs===null||c.start>floorMinute(afterMs));
 let ambiguous=null;
 const crossed=(c,level,which)=>side==='SHORT'?(which==='stop'?c.high>=level:c.low<=level):(which==='stop'?c.low<=level:c.high>=level);
 const entry=(c)=>c.high>=o.entryLow&&c.low<=o.entryHigh;
 for(const c of checked){
  if(['STOPPED','TP3_DONE'].includes(status))break;
  const wasPending=status==='PENDING';
  if(wasPending){
   if(!entry(c))continue;
   // Full-minute OHLC cannot prove whether an exit came before first entry.
   if(crossed(c,o.stop,'stop')||crossed(c,o.tp1,'target')){
    ambiguous={minute:kstTime(c.start),reason:'ENTRY_AND_EXIT_SAME_1M',high:c.high,low:c.low};
    break;
   }
   events.push({kind:'ENTERED',minute:kstTime(c.start),high:c.high,low:c.low,level:o.entryLow});
   status='ENTERED';continue;
  }
  const stop=crossed(c,o.stop,'stop');
  const t1=crossed(c,o.tp1,'target');
  const t2=crossed(c,o.tp2,'target');
  const t3=crossed(c,o.tp3,'target');
  if(stop&&t1){ambiguous={minute:kstTime(c.start),reason:'STOP_AND_TARGET_SAME_1M',high:c.high,low:c.low};break;}
  const kind=stop?'STOPPED':t3?'TP3_DONE':t2?'TP2':t1?'TP1':null;
  if(!kind||kind===status||(status==='TP2'&&kind==='TP1'))continue;
  events.push({kind,minute:kstTime(c.start),high:c.high,low:c.low,level:kind==='STOPPED'?o.stop:o[kind==='TP3_DONE'?'tp3':kind.toLowerCase()]});
  status=kind;
 }
 return {status,events,ambiguous,minutesChecked:checked.length};
}
export async function recoverMarketGaps(raw,previousSnapshot,{fetcher=globalThis.fetch,timeoutMs=2500,maxMinutes=1000}={}){
 const market=rawMarketWindows(raw),result={BTC:{},ETH:{}};
 await Promise.all(['BTC','ETH'].map(async symbol=>{
  const prev=previousSnapshot?.[symbol==='BTC'?'btcTime':'ethTime'];
  const pathStart=market[symbol].pathStart;
  const gapSeconds=prev?Math.round((kstMs(pathStart)-kstMs(prev))/1000):0;
  if(gapSeconds<=60){result[symbol]={recovered:false,needed:false,gapSeconds};return;}
  try{
   const bars=await fetchBybitCandles({symbol,fromMs:kstMs(prev),toMs:kstMs(pathStart),fetcher,timeoutMs,maxMinutes});
   result[symbol]={recovered:true,needed:true,gapSeconds,source:'BYBIT_LINEAR_PUBLIC_1M',
    coveredFrom:kstTime(bars[0].start),coveredThrough:kstTime(bars[bars.length-1].start),
    barCount:bars.length,bars};
  }catch(e){
   result[symbol]={recovered:false,needed:true,gapSeconds,reason:String(e.message||e).slice(0,160)};
  }
 }));
 return result;
}


// Fetch full, closed minute bars only for symbols with live formal signals.
// A missing 1m candle invalidates the whole recovery; no interpolation.
// Revisit the earliest unresolved priced path for every active signal, not only
// signals explicitly marked UNVERIFIED. Historical evidence is immutable and
// cannot be cleared by a later contiguous snapshot.
export function earliestUnknownGapOrigins(ledger,evidenceRows){
 const live=new Set(['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED']);
 const origins={};
 for(const signal of ledger){
  if(!live.has(signal.status))continue;
  const sym=signal.symbol==='BTCUSDT'?'BTC':signal.symbol==='ETHUSDT'?'ETH':null;
  if(!sym)continue;
  for(const row of evidenceRows){
   const e=row?.evidence?.perSymbol?.[sym]||row?.perSymbol?.[sym];
   if(!e||!(e.gapSeconds>60)||!e.priorOfficialAt||
     !(e.couldNotVerifySignals||[]).includes(signal.id))continue;
   if(!origins[signal.id]||kstMs(e.priorOfficialAt)<kstMs(origins[signal.id]))
    origins[signal.id]=e.priorOfficialAt;
  }
 }
 return origins;
}

export async function recoverActiveSignalWindows(raw,previousSnapshot,ledger,{
 fetcher=globalThis.fetch,timeoutMs=2500,maxMinutes=1000,unverifiedOrigins={}
}={}){
 const m=rawMarketWindows(raw),result={BTC:{needed:false},ETH:{needed:false}};
 await Promise.all(['BTC','ETH'].map(async symbol=>{
  const active=ledger.filter(s=>s.symbol===symbol+'USDT'&&
   ['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(s.status));
  if(!active.length)return;
  const prev=previousSnapshot?.[symbol==='BTC'?'btcTime':'ethTime'];
  const candidate=[...(prev?[kstMs(prev)]:[])];
  for(const signal of active){
   const back=unverifiedOrigins[signal.id];
   // Rewind even when an old ENTERED/TP1/TP2 remained in the ledger after a failed recovery.
   if(back)candidate.push(kstMs(back));
   if(!prev)candidate.push(kstMs(signal.verifiedUntil||signal.original?.issuedAt||m[symbol].pathStart));
  }
  const fromMs=Math.min(...candidate);
  const atMs=kstMs(m[symbol].timestamp);
  const untilMs=floorMinute(atMs)-60000; // Never treat a still-open candle as complete.
  if(untilMs<floorMinute(fromMs)){
   result[symbol]={needed:true,recovered:false,reason:'NO_CLOSED_1M_AFTER_PREVIOUS_OBSERVATION'};
   return;
  }
  try{
   const bars=await fetchBybitCandles({symbol,fromMs,toMs:untilMs,fetcher,timeoutMs,maxMinutes});
   const rawGap=prev?Math.max(0,Math.round((kstMs(m[symbol].pathStart)-kstMs(prev))/1000)):0;
   // Complete OHLC can disprove a crossing in a previously unobserved interval.
   const bridgeCovered=Boolean(prev&&rawGap>60&&verifyCandleCoverage(bars,kstMs(prev),kstMs(m[symbol].pathStart)));
   result[symbol]={needed:true,recovered:true,source:'BYBIT_LINEAR_PUBLIC_1M',
    barCount:bars.length,coveredFrom:kstTime(bars[0].start),
    coveredThrough:kstTime(bars[bars.length-1].start),
    coveredFromMs:bars[0].start,coveredThroughMs:bars[bars.length-1].start,
    bridgeCovered,rawGapSeconds:rawGap,bars};
  }catch(e){
   result[symbol]={needed:true,recovered:false,reason:String(e.message||e).slice(0,180)};
  }
 }));
 return result;
}
