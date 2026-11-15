// Standalone early-reversal screen; it does NOT publish a formal trade.
// Original V1.7 policy and legacy trend-following SHORT decisions are unchanged.
const n=(t,r)=>{const m=r.exec(t);return m?Number(m[1]):null;};
const str=(t,r)=>r.exec(t)?.[1]??null;
const pos=x=>typeof x==='number'&&Number.isFinite(x)&&x>0;
const section=(raw,sym)=>{
 const start=raw.indexOf('========== '+sym+' ('+sym+'USDT) ==========');
 if(start<0)return '';
 const end=sym==='BTC'?raw.indexOf('========== ETH (ETHUSDT) ==========',start+1):-1;
 return raw.slice(start,end<0?undefined:end);
};
const pair=(t,r)=>{const m=r.exec(t);return m?[Number(m[1]),Number(m[2])]:[null,null];};
const candidateTargets=(sources,price,long)=>{
 const ordered=sources.filter(x=>pos(x.value)&&(long?x.value>price:x.value<price))
  .sort((a,b)=>long?a.value-b.value:b.value-a.value);
 const out=[];
 for(const item of ordered)if(!out.length||Math.abs(item.value-out.at(-1).value)>price*0.000001)out.push(item);
 return out.slice(0,4);
};
function screenOne(t,symbol){
 const price=n(t,/^lastPrice:\s*([\d.]+)/m),at=str(t,/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m);
 const cvdQuality=str(t,/^cvd_fetch_fallbacks:[^\n]*\bcvd_reliability:\s*(HIGH|MEDIUM|LOW)/m)||str(t,/^cvd_reliability:\s*(HIGH|MEDIUM|LOW)/m);
 const cvd5=n(t,/^cvd_net_snapshot\(5m\):\s*([-\d.]+)/m),cvd15=n(t,/^cvd_net_snapshot\(15m\):\s*([-\d.]+)/m);
 const hist1=n(t,/^1m:\s*n=\d+[^\n]*\bhist=([-\d.]+)/m),hist5=n(t,/^5m:\s*n=\d+[^\n]*\bhist=([-\d.]+)/m);
 const ema5=str(t,/^5m:\s*n=\d+[^\n]*\balign=(BULL|BEAR)/m);
 const ema15=str(t,/^15m:\s*n=\d+[^\n]*\balign=(BULL|BEAR)/m);
 const ema4h=str(t,/^4h:\s*n=\d+[^\n]*\balign=(BULL|BEAR)/m);
 const ema1d=str(t,/^1D:\s*n=\d+[^\n]*\balign=(BULL|BEAR)/m);
 const vwap=n(t,/^vwap_24h:\s*([\d.]+)/m),atr=n(t,/^15m:\s*atr14=([\d.]+)/m);
 const [pathHigh,pathLow]=pair(t,/^path_high\/low:\s*([\d.]+)\s*@[^\n]*?\/\s*([\d.]+)\s*@/m);
 const [sessionHigh,sessionLow]=pair(t,/^session_6h_high\/low:\s*([\d.]+)\s*\/\s*([\d.]+)/m);
 const [prevHigh,prevLow]=pair(t,/^prev_day_high\/low:\s*([\d.]+)\s*\/\s*([\d.]+)/m);
 const [swingHigh,swingLow]=pair(t,/^15m:\s*atr14=[^\n]*\bswingHigh=([\d.]+)\s*\|\s*swingLow=([\d.]+)/m);
 const order=str(t,/^path_range_pct:[^\n]*\border:\s*(LOW_FIRST|HIGH_FIRST|UNKNOWN)/m);
 const complete=[price,at,cvd5,cvd15,hist1,hist5,ema5,ema15,ema4h,ema1d,vwap,atr,pathHigh,pathLow,order].every(x=>x!==null&&x!==undefined);
 const result={symbol:symbol+'USDT',snapshotAt:at,price,source:'USER_PASTED_SNAPSHOT',
  countertrendLong:ema4h==='BEAR'||ema1d==='BEAR',countertrendShort:ema4h==='BULL'||ema1d==='BULL'};
 for(const direction of ['LONG','SHORT']){
  const long=direction==='LONG';
  const structural=long?order==='LOW_FIRST'&&pos(pathLow)&&price>pathLow:order==='HIGH_FIRST'&&pos(pathHigh)&&price<pathHigh;
  const flow=cvdQuality==='HIGH'&&(long?cvd5>0&&cvd15>0:cvd5<0&&cvd15<0);
  // Deliberately no 30m, 1h, 4h or 1D trend alignment requirement.
  const momentum=long?hist1>0&&(hist5>0||ema5==='BULL'):hist1<0&&(hist5<0||ema5==='BEAR');
  const vwapReclaimed=long?pos(vwap)&&pathLow<vwap&&price>vwap:pos(vwap)&&pathHigh>vwap&&price<vwap;
  const stopRef=long?pathLow:pathHigh,risk=Math.abs(price-stopRef);
  const targets=candidateTargets(long?[
   {anchor:'PATH_HIGH',value:pathHigh},{anchor:'SESSION_6H_HIGH',value:sessionHigh},
   {anchor:'SWING_HIGH_15M',value:swingHigh},{anchor:'PREV_DAY_HIGH',value:prevHigh}
  ]:[
   {anchor:'PATH_LOW',value:pathLow},{anchor:'SESSION_6H_LOW',value:sessionLow},
   {anchor:'SWING_LOW_15M',value:swingLow},{anchor:'PREV_DAY_LOW',value:prevLow}
  ],price,long);
  const rr2=pos(risk)&&targets.length>=2?Math.abs(targets[1].value-price)/risk:null;
  const checks={sourceComplete:complete,chronologicalReversal:structural,flowReliable:cvdQuality==='HIGH',
   flowAligned:flow,momentumShift:momentum,vwapReclaimed,
   structuralStopAvailable:long?pos(stopRef)&&stopRef<price:pos(stopRef)&&stopRef>price,
   structuralTargetsAvailable:targets.length>=2,referenceRR2AtLeast2:rr2!==null&&rr2>=2};
  const ready=Object.values(checks).every(Boolean);
  result[long?'long':'short']={direction,status:ready?'REVIEW_REQUIRED':complete?'WATCH':'INSUFFICIENT_DATA',
   checks,referenceEntry:price,stopStructureReference:pos(stopRef)?stopRef:null,stopNotChosen:true,
   targetReferences:targets,referenceRR2:rr2===null?null:Math.round(rr2*100)/100,
   observed:{cvd5,cvd15,cvdQuality,hist1,hist5,ema5,ema15,ema4h,ema1d,vwap24h:vwap,pathOrder:order,atr15m:atr},
   failedChecks:Object.entries(checks).filter(([,v])=>!v).map(([k])=>k),
   warning:'Screen only: V1.7 full analysis, risk and target viability, coupling and publication validation remain mandatory.'};
 }
 return result;
}
export function screenReversalEntries(raw){
 return Object.fromEntries(['BTC','ETH'].map(sym=>[sym,screenOne(section(raw,sym),sym)]));
}
const ACTIONS=new Set(['ISSUE_NOW','ISSUE_PENDING','HOLD']);
const REASONS=new Set(['STRUCTURE','STOP','TARGET','RR','RETEST','EXECUTION_TIMING','COUPLING','EXPOSURE','CONFLICTING_FLOW','DATA_QUALITY']);
export function validateReversalReview(analysis,screen){
 for(const sym of ['BTC','ETH']){
  for(const [side,key] of [['LONG','long'],['SHORT','short']]){
   if(screen[sym]?.[key]?.status!=='REVIEW_REQUIRED')continue;
   const r=analysis?.reversalReview?.[sym]?.[side];
   if(!r||!ACTIONS.has(r.action)||typeof r.evidence!=='string'||!r.evidence.trim()||
      typeof r.counterEvidence!=='string'||!r.counterEvidence.trim())
    throw Error('REVERSAL_REVIEW_REQUIRED_'+sym+'_'+side);
   if(r.action==='HOLD'){
    if(!REASONS.has(r.blocker)||typeof r.rejectionEvidence!=='string'||!r.rejectionEvidence.trim())
     throw Error('REVERSAL_HOLD_NEEDS_STRUCTURAL_OR_RISK_REJECTION_'+sym+'_'+side);
   }else{
    const signals=(analysis.newSignals||[]).filter(s=>s.symbol===sym+'USDT'&&s.direction===side&&s.tradeType==='추세 반전형');
    if(signals.length!==1||signals[0].status!==(r.action==='ISSUE_NOW'?'NEW_ENTRY':'PENDING'))
     throw Error('REVERSAL_FORMAL_SIGNAL_MISSING_'+sym+'_'+side);
   }
  }
 }
 return true;
}
