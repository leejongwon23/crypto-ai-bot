// V1.7 clause 21 execution audit. This is NOT a market prediction model.
// Each pending trade is compared with current and nearer plans using observed
// source anchors and independently recalculated risk/target metrics.
const n=x=>typeof x==='number'&&Number.isFinite(x);
const has=x=>typeof x==='string'&&x.trim().length>0;
const close=(a,b,rel=0.000001)=>Math.abs(a-b)<=Math.max(0.000001,Math.abs(b)*rel);
const num=(text,re)=>{const m=re.exec(text);return m?Number(m[1]):null;};
const symName=(sym)=>sym==='BTC'?'BTCUSDT':'ETHUSDT';
const dist=(p,l,h)=>p<l?l-p:p>h?p-h:0;

export function buildAutoSwitchContext(raw,ledger=[],snapshot){
 const btc=raw.indexOf('========== BTC (BTCUSDT) =========='),eth=raw.indexOf('========== ETH (ETHUSDT) ==========');
 const result={};
 for(const symbol of ['BTC','ETH']){
  const frag=symbol==='BTC'?raw.slice(btc,eth):raw.slice(eth);
  const price=symbol==='BTC'?snapshot.btcPrice:snapshot.ethPrice;
  const snapshotAt=symbol==='BTC'?snapshot.btcTime:snapshot.ethTime;
  const atr15m=num(frag,/^15m:\s*atr14=([\d.]+)/m);
  const anchors={SPOT:price};
  const add=(key,re)=>{const v=num(frag,re);if(n(v)&&v>0)anchors[key]=v;};
  add('EMA_1m_12',/^1m:\s*n=\d+[^\n]*\bema12=([\d.]+)/m);
  add('EMA_1m_26',/^1m:\s*n=\d+[^\n]*\bema26=([\d.]+)/m);
  add('EMA_5m_12',/^5m:\s*n=\d+[^\n]*\bema12=([\d.]+)/m);
  add('EMA_5m_26',/^5m:\s*n=\d+[^\n]*\bema26=([\d.]+)/m);
  add('EMA_15m_12',/^15m:\s*n=\d+[^\n]*\bema12=([\d.]+)/m);
  add('EMA_15m_26',/^15m:\s*n=\d+[^\n]*\bema26=([\d.]+)/m);
  add('VWAP_6h',/^vwap_24h:[^\n]*\bvwap_session_6h:\s*([\d.]+)/m);
  add('VWAP_24h',/^vwap_24h:\s*([\d.]+)/m);
  add('PATH_HIGH',/^path_high\/low:\s*([\d.]+)/m);
  add('PATH_LOW',/^path_high\/low:\s*[\d.]+[^\n]*\/\s*([\d.]+)/m);
  add('SWING_HIGH_15m',/^15m:\s*atr14=[^\n]*\bswingHigh=([\d.]+)/m);
  add('SWING_LOW_15m',/^15m:\s*atr14=[^\n]*\bswingLow=([\d.]+)/m);
  add('PREV_DAY_HIGH',/^prev_day_high\/low:\s*([\d.]+)/m);
  add('PREV_DAY_LOW',/^prev_day_high\/low:\s*[\d.]+\s*\/\s*([\d.]+)/m);
  const pending=ledger.filter(x=>x.symbol===symName(symbol)&&x.status==='PENDING').map(x=>{
   const o=x.original||x,d=dist(price,o.entryLow,o.entryHigh);
   return {id:x.id,direction:o.direction,entryLow:o.entryLow,entryHigh:o.entryHigh,
    distance:d,distancePct:d/price*100,distanceAtr:n(atr15m)&&atr15m>0?d/atr15m:null};
  });
  result[symbol]={snapshotAt,price,atr15m:n(atr15m)&&atr15m>0?atr15m:null,anchors,pending};
 }
 return result;
}

function validateCandidate(c,market,old,kind){
 if(!c||!['PLAN','NO_PLAN'].includes(c.kind))throw Error('SWITCH_'+kind+'_MISSING_QUANTITATIVE_CANDIDATE_'+old.id);
 const anchor=market.anchors[c.anchor];
 if(!has(c.anchor)||!n(anchor)||!n(c.anchorPrice)||!close(c.anchorPrice,anchor))throw Error('SWITCH_'+kind+'_SOURCE_ANCHOR_NOT_IN_CURRENT_SNAPSHOT_'+old.id);
 if(kind==='NEAR'&&c.anchor==='SPOT')throw Error('SWITCH_NEAR_REQUIRES_OBSERVED_STRUCTURE_'+old.id);
 if(!has(c.evidence)||!has(c.counterEvidence))throw Error('SWITCH_'+kind+'_POSITIVE_AND_NEGATIVE_EVIDENCE_REQUIRED_'+old.id);
 if(c.kind==='NO_PLAN'){
  if(!['ENTRY_STRUCTURE','STOP','TP1','TP2','TP3','PRICE_PATH','COUPLING'].includes(c.missingLeg))
   throw Error('SWITCH_'+kind+'_MISSING_STRUCTURAL_LEG_REQUIRED_'+old.id);
  if(!Array.isArray(c.blockers)||!c.blockers.some(x=>['STRUCTURE','STOP','TARGET','RR','RETEST','PRICE_PATH','COUPLING'].includes(x)))
   throw Error('SWITCH_'+kind+'_GENERIC_CVD_HOLD_FORBIDDEN_'+old.id);
  return null;
 }
 const p=c.plan;
 if(!p||!['LONG','SHORT'].includes(p.direction))throw Error('SWITCH_'+kind+'_PLAN_DIRECTION_REQUIRED_'+old.id);
 if(p.direction!==old.direction)throw Error('SWITCH_'+kind+'_MUST_MATCH_OLD_DIRECTION_'+old.id);
 for(const v of ['entryLow','entryHigh','stop','tp1','tp2','tp3','rr2','stopPct','tp2Pct'])
  if(!n(p[v])||(p[v]<=0&&v!=='stopPct'))throw Error('SWITCH_'+kind+'_PLAN_NUMBER_MISSING_'+v+'_'+old.id);
 if(p.entryLow>p.entryHigh)throw Error('SWITCH_'+kind+'_ENTRY_RANGE_REVERSED_'+old.id);
 if(p.direction==='LONG'){
  if(!(p.stop<p.entryLow&&p.entryHigh<p.tp1&&p.tp1<p.tp2&&p.tp2<p.tp3))throw Error('SWITCH_'+kind+'_LONG_GEOMETRY_INVALID_'+old.id);
 }else if(!(p.stop>p.entryHigh&&p.entryLow>p.tp1&&p.tp1>p.tp2&&p.tp2>p.tp3))
  throw Error('SWITCH_'+kind+'_SHORT_GEOMETRY_INVALID_'+old.id);
 const ref=p.direction==='LONG'?p.entryHigh:p.entryLow;
 const risk=Math.abs(p.stop-ref),reward=Math.abs(p.tp2-ref),rr=reward/risk;
 if(!n(rr)||!close(p.rr2,rr,0.0005)||!close(p.stopPct,risk/ref*100,0.0005)||!close(p.tp2Pct,reward/ref*100,0.0005))
  throw Error('SWITCH_'+kind+'_METRICS_NOT_RECALCULATED_'+old.id);
 const gap=dist(market.price,p.entryLow,p.entryHigh);
 if(kind==='IMMEDIATE'&&gap>Math.max(0.0000001,market.price*0.000001))
  throw Error('SWITCH_IMMEDIATE_NOT_AT_CURRENT_PRICE_'+old.id);
 if(kind==='NEAR'){
  if(gap<=market.price*0.000001)throw Error('SWITCH_NEAR_MUST_BE_PENDING_NOT_CURRENT_'+old.id);
  if(!(gap<old.distance))throw Error('SWITCH_NEAR_NOT_CLOSER_THAN_OLD_'+old.id);
  if(!(dist(anchor,p.entryLow,p.entryHigh)<=market.atr15m))
   throw Error('SWITCH_NEAR_ANCHOR_TOO_FAR_FROM_PLAN_'+old.id);
 }
 if(!has(c.stopEvidence)||!has(c.tp1Evidence)||!has(c.tp2Evidence)||!has(c.tp3Evidence)||!has(c.retestEvidence))
  throw Error('SWITCH_'+kind+'_SOURCE_AND_RETEST_EVIDENCE_REQUIRED_'+old.id);
 if(!['OBSERVED','PROJECTED','HISTORICAL'].includes(c.tp2Kind)||!['OBSERVED','PROJECTED','HISTORICAL'].includes(c.tp3Kind))
  throw Error('SWITCH_'+kind+'_TARGET_SOURCE_TYPE_REQUIRED_'+old.id);
 return {rr2:rr,distance:gap};
}

export function validateAutoSwitchAudit(analysis,raw,ledger,snapshot,news=[],events=[]){
 const context=buildAutoSwitchContext(raw,ledger,snapshot);
 for(const symbol of ['BTC','ETH']){
  const market=context[symbol];
  // Legacy sparse fixtures lacking ATR are not retroactively assigned numeric
  // levels: the full existing history/price-event checks still run unchanged.
  if(!market.pending.length||!market.atr15m||Object.keys(market.anchors).length<=1)continue;
  const audit=analysis.autoSwitchAudit?.[symbol];
  if(!audit||audit.snapshotAt!==market.snapshotAt||!n(audit.price)||!close(audit.price,market.price)||!Array.isArray(audit.reviews))
   throw Error('AUTO_SWITCH_AUDIT_REQUIRED_'+symbol);
  if(audit.reviews.length!==market.pending.length)throw Error('AUTO_SWITCH_REVIEW_EVERY_PENDING_'+symbol);
  const checked=new Set();
  for(const old of market.pending){
   const review=audit.reviews.find(x=>x.id===old.id);
   if(!review||checked.has(old.id))throw Error('AUTO_SWITCH_PENDING_ID_MISSING_OR_DUPLICATE_'+old.id);
   checked.add(old.id);
   if(!n(review.distanceAtr)||!n(review.distancePct)||!close(review.distanceAtr,old.distanceAtr,0.0005)||!close(review.distancePct,old.distancePct,0.0005))
    throw Error('AUTO_SWITCH_OLD_DISTANCE_METRIC_MISMATCH_'+old.id);
   validateCandidate(review.immediate,market,old,'IMMEDIATE');
   validateCandidate(review.near,market,old,'NEAR');
   for(const key of ['farEvidence','oppositeEvidence','keepEvidence','executionTimingEvidence','candidateComparison','couplingEvidence'])
    if(!has(review[key]))throw Error('AUTO_SWITCH_BIAS_COMPARISON_MISSING_'+key+'_'+old.id);
   if(!['KEEP','REDESIGN','CANCEL','QUARANTINE'].includes(review.selection))throw Error('AUTO_SWITCH_SELECTION_INVALID_'+old.id);
   const change=events.find(e=>e.id===old.id&&['REDESIGNED_CANCELLED','CANCELLED','UNVERIFIED'].includes(e.kind));
   if(review.selection==='KEEP'&&change)throw Error('AUTO_SWITCH_KEEP_BUT_TERMINATED_'+old.id);
   if(review.selection==='REDESIGN'){
    if(!change||change.kind!=='REDESIGNED_CANCELLED'||!news.some(s=>s.symbol===symName(symbol)))
     throw Error('AUTO_SWITCH_REDESIGN_REQUIRES_ATOMIC_EVENT_AND_NEW_ID_'+old.id);
    // Far or opposite-side replacements remain valid under unchanged policy 21.
   }
   if(review.selection==='CANCEL'&&(!change||change.kind!=='CANCELLED'))throw Error('AUTO_SWITCH_CANCEL_REQUIRES_EVENT_'+old.id);
   if(review.selection==='QUARANTINE'&&(!change||change.kind!=='UNVERIFIED'))throw Error('AUTO_SWITCH_UNKNOWN_REQUIRES_UNVERIFIED_EVENT_'+old.id);
   if(review.selection==='KEEP'){
    for(const key of ['immediate','near']){
     const candidate=review[key];
     if(candidate.kind==='PLAN'){
      const contrary=market.anchors[candidate.rejectionAnchor];
      if(!has(candidate.rejectionEvidence)||!Array.isArray(candidate.rejectionFactors)||!candidate.rejectionFactors.some(x=>['STRUCTURE','RR','STOP','TARGET','RETEST','TIME','PRICE_PATH','COUPLING'].includes(x))||
         !n(contrary)||!n(candidate.rejectionAnchorPrice)||!close(candidate.rejectionAnchorPrice,contrary))
       throw Error('AUTO_SWITCH_KEEP_REJECTED_PLAN_NEEDS_CAUSAL_COMPARISON_'+key+'_'+old.id);
     }
    }
   }
  }
 }
 return context;
}
