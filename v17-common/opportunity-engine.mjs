/** V1.7 additive opportunity search.
 * Returns non-binding, source-anchored alternative plans: it NEVER places orders,
 * overwrites a signal, or claims a candidate is profitable. No hindsight.
 * Long-term losing trades are not excluded from the candidate set.
 */
const round = n => Math.round((n + Number.EPSILON) * 1e4) / 1e4;
const num = (s, re) => {const m = s.match(re);return m ? Number(m[1]) : null;};
const positive = n => Number.isFinite(n) && n > 0;
function symbolSection(raw, sym) {
  const re = /========== (BTC|ETH) \(\1USDT\) ==========/g;
  const positions=[...String(raw).matchAll(re)];
  const idx=positions.findIndex(x=>x[1]===sym);
  if(idx<0)return '';
  return raw.slice(positions[idx].index+(positions[idx][0].length),idx+1<positions.length?positions[idx+1].index:undefined);
}
function timeframe(text, tf) {
  const line = text.match(new RegExp('^'+tf.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+':\\s*n=\\d+[^\\n]*$', 'm'))?.[0]||'';
  return {ema12:num(line,/\bema12=([-\d.]+)/),ema26:num(line,/\bema26=([-\d.]+)/),hist:num(line,/\bhist=([-\d.]+)/),align:line.match(/\balign=(BULL|BEAR)/)?.[1]||null};
}
function parseMarket(text,symbol){
  const p = num(text,/^lastPrice:\s*([\d.]+)/m),atr=num(text,/^15m:\s*atr14=([\d.]+)/m);
  const dated = text.match(/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m)?.[1]||null;
  const path=text.match(/^path_high\/low:\s*([\d.]+)\s*@[^\n/]*\/\s*([\d.]+)\s*@/m);
  const session=text.match(/^session_6h_high\/low:\s*([\d.]+)\s*\/\s*([\d.]+)/m);
  const prev=text.match(/^prev_day_high\/low:\s*([\d.]+)\s*\/\s*([\d.]+)/m);
  const swing=text.match(/^15m:\s*atr14=[^\n]*?swingHigh=([\d.]+)\s*\|\s*swingLow=([\d.]+)/m);
  const quality=text.match(/^cvd_fetch_fallbacks:\s*\d+[^\n]*?cvd_reliability:\s*(HIGH|MEDIUM|LOW)/m)?.[1]||'UNKNOWN';
  return {symbol,at:dated,price:p,atr,
    ema1:timeframe(text,'1m'),ema5:timeframe(text,'5m'),ema15:timeframe(text,'15m'),ema30:timeframe(text,'30m'),ema4h:timeframe(text,'4h'),ema1d:timeframe(text,'1D'),
    priceChange15:num(text,/^price_change_5m\/15m\/30m\/1h\/4h:\s*[-\d.]+%\s*\/\s*([-\d.]+)%/m),
    candle15:text.match(/^candle_structure_5m\/15m:[^\n]*\/\s*(UP|DOWN|MIXED)/m)?.[1]||null,
    vwap6:num(text,/^vwap_24h:[^\n]*?vwap_session_6h:\s*([\d.]+)/m),vwap24:num(text,/^vwap_24h:\s*([\d.]+)/m),
    pathHigh:path?Number(path[1]):null,pathLow:path?Number(path[2]):null,
    sessionHigh:session?Number(session[1]):null,sessionLow:session?Number(session[2]):null,
    prevHigh:prev?Number(prev[1]):null,prevLow:prev?Number(prev[2]):null,
    swingHigh:swing?Number(swing[1]):null,swingLow:swing?Number(swing[2]):null,
    cvd5:num(text,/^cvd_net_snapshot\(5m\):\s*([-\d.]+)/m),
    cvd15:num(text,/^cvd_net_snapshot\(15m\):\s*([-\d.]+)/m),
    cvd30:num(text,/^cvd_net_snapshot\(30m\):\s*([-\d.]+)/m),
    cvdQuality:quality};
}
function anchors(m,side){
  const long=side==='LONG', spot=m.price;
  const all=[
    {method:'IMMEDIATE',anchor:'SPOT',entry:spot},
    {method:'NEAR_EMA',anchor:'EMA_5M_12',entry:m.ema5.ema12},
    {method:'NEAR_EMA',anchor:'EMA_5M_26',entry:m.ema5.ema26},
    {method:'NEAR_EMA',anchor:'EMA_15M_12',entry:m.ema15.ema12},
    {method:'NEAR_EMA',anchor:'EMA_15M_26',entry:m.ema15.ema26},
    {method:'VWAP_RETEST',anchor:'VWAP_6H',entry:m.vwap6},
    {method:'VWAP_RETEST',anchor:'VWAP_24H',entry:m.vwap24},
    {method:'PRICE_RETEST',anchor:long?'PATH_LOW':'PATH_HIGH',entry:long?m.pathLow:m.pathHigh},
    {method:'PRICE_RETEST',anchor:long?'SESSION_6H_LOW':'SESSION_6H_HIGH',entry:long?m.sessionLow:m.sessionHigh},
    {method:'BREAKOUT',anchor:long?'PATH_HIGH':'PATH_LOW',entry:long?m.pathHigh:m.pathLow}
  ].filter(x=>positive(x.entry));
  const dedupe=new Set();return all.filter(x=>{const k=side+':'+round(x.entry);if(dedupe.has(k))return false;dedupe.add(k);return true;});
}
const stopAnchors=(m,side)=>side==='LONG'?
 [{name:'PATH_LOW',v:m.pathLow},{name:'SESSION_6H_LOW',v:m.sessionLow},{name:'SWING_15M_LOW',v:m.swingLow}]:
 [{name:'PATH_HIGH',v:m.pathHigh},{name:'SESSION_6H_HIGH',v:m.sessionHigh},{name:'SWING_15M_HIGH',v:m.swingHigh}];
const targetAnchors=(m,side)=>side==='LONG'?
 [{name:'PATH_HIGH',v:m.pathHigh},{name:'VWAP_6H',v:m.vwap6},{name:'EMA_15M_26',v:m.ema15.ema26},{name:'SESSION_6H_HIGH',v:m.sessionHigh},{name:'SWING_15M_HIGH',v:m.swingHigh},{name:'PREV_DAY_HIGH',v:m.prevHigh}]:
 [{name:'PATH_LOW',v:m.pathLow},{name:'VWAP_6H',v:m.vwap6},{name:'EMA_15M_26',v:m.ema15.ema26},{name:'SESSION_6H_LOW',v:m.sessionLow},{name:'SWING_15M_LOW',v:m.swingLow},{name:'PREV_DAY_LOW',v:m.prevLow}];
// Evidence is observable context, not a trained probability model, hard gate,
// or independent CVD+taker vote. No score/weights, no claim of profitability.
function directionEvidence(m,side){
 const trend=side==='LONG'?'BULL':'BEAR',opposite=side==='LONG'?'BEAR':'BULL',sgn=side==='LONG'?1:-1;
 const structural = [m.ema5.align,m.ema15.align,m.ema30.align];
 const flow=[m.cvd5,m.cvd15,m.cvd30];
 const priceMomentum=Number.isFinite(m.priceChange15)?Math.sign(m.priceChange15*sgn):null;
 const structure15=m.candle15==='UP'?1:m.candle15==='DOWN'?-1:null;
 const priceStructure=structure15===null?null:structure15*sgn;
 const nearPriceSupports=structural.filter(x=>x===trend).length;
 const nearPriceOpposes=structural.filter(x=>x===opposite).length;
 const reliableFlow=m.cvdQuality==='HIGH'||m.cvdQuality==='MEDIUM';
 const alignedFlow=reliableFlow&&flow.every(Number.isFinite)&&flow.every(x=>x*sgn>0);
 const opposingFlow=reliableFlow&&flow.every(Number.isFinite)&&flow.every(x=>x*sgn<0);
 const priceAligned=nearPriceSupports===3&&priceMomentum===1&&priceStructure!==-1;
 const priceOpposed=nearPriceOpposes===3&&priceMomentum===-1&&priceStructure!==1;
 let regime='MIXED';
 if(priceAligned&&alignedFlow)regime='PRICE_AND_FLOW_ALIGNED';
 else if(priceOpposed&&opposingFlow)regime='PRICE_AND_FLOW_OPPOSE';
 else if(!reliableFlow||priceMomentum===null)regime='DATA_LIMITED';
 return {regime,source:'EMA_5_15_30+CVD_5_15_30+OBSERVED_15M_PRICE_CHANGE',
   price15mChangePct:m.priceChange15,price15mStructure:m.candle15,
   ema5:m.ema5.align,ema15:m.ema15.align,ema30:m.ema30.align,
   macro4h:m.ema4h.align,macro1d:m.ema1d.align,
   cvd5:m.cvd5,cvd15:m.cvd15,cvd30:m.cvd30,cvdQuality:m.cvdQuality,
   caution:'Describes present evidence only. Not a forecast, success probability, or mandatory veto. Countertrend reversals require separate structure review.'};
}
function optionsFor(m,side){
 const long=side==='LONG',sign=long?1:-1;
 const ranked=[];
 const evidence=directionEvidence(m,side);
 for(const a of anchors(m,side)){
   const distAtr=round(Math.abs(a.entry-m.price)/m.atr);
   const origin=a.method==='IMMEDIATE'?'NOW':((long?a.entry<m.price:a.entry>m.price)?'PULLBACK': 'BREAKOUT_OR_LATE');
   const targets=targetAnchors(m,side).filter(t=>positive(t.v)&&sign*(t.v-a.entry)>0).sort((x,y)=>sign*(x.v-y.v));
   const dedup=[];for(const target of targets){if(!dedup.some(t=>Math.abs(t.v-target.v)<m.atr*0.05))dedup.push(target)}
   const eligibleStops=stopAnchors(m,side).filter(t=>positive(t.v)&&sign*(a.entry-t.v)>0);
   for(const k of eligibleStops){
     // 0.35 or 0.75 ATR are parallel *sensitivity scenarios*, NOT proven optimal settings.
     for(const cushionAtr of [0.35,0.75]){
       const stop=round(k.v-sign*m.atr*cushionAtr),risk=Math.abs(a.entry-stop);
       if(!positive(risk)||dedup.length<2)continue;
       const tp1=dedup[0],tp2=dedup[1];
       const rr2=round(Math.abs(tp2.v-a.entry)/risk);
       const tp1R=round(Math.abs(tp1.v-a.entry)/risk);
       const flow=[m.cvd5,m.cvd15,m.cvd30].filter(Number.isFinite);
       const favorable=flow.filter(v=>sign*v>0).length,unfavorable=flow.filter(v=>sign*v<0).length;
       ranked.push({method:a.method,origin,entryAnchor:a.anchor,side,entry:round(a.entry),priceDistanceAtr:distAtr,
         stopAnchor:k.name,stop,cushionAtr,stopRisk:round(risk),tp1:round(tp1.v),tp1Anchor:tp1.name,
         tp2:round(tp2.v),tp2Anchor:tp2.name,rr2,tp1R,meetsArithmeticRR2:rr2>=2,
         tp1AtLeastOneR:tp1R>=1,marketDirectionEvidence:evidence.regime,
         cvdQuality:m.cvdQuality,flowAlignedWindows:favorable,flowOpposingWindows:unfavorable,
         aboveBelowSpot:origin, warning:'Counterfactual candidate only. Target is not a price forecast; stop cushion is not calibrated; pending first touch not confirmed.'});
     }
   }
 }
 // Do NOT rank the highest mathematical RR ahead of a barely useful TP1.
 // Both RR2 and a first target worth >=1 initial risk are transparency checks,
 // not calibrated performance filters: retain every alternative, including counterflow trades.
 // Source-grounded proximity wins before oversized hypothetical reward.
 ranked.sort((a,b)=>Number(b.meetsArithmeticRR2)-Number(a.meetsArithmeticRR2)
     ||Number(b.tp1AtLeastOneR)-Number(a.tp1AtLeastOneR)
     ||a.priceDistanceAtr-b.priceDistanceAtr ||b.flowAlignedWindows-a.flowAlignedWindows
     ||b.tp1R-a.tp1R ||b.rr2-a.rr2 ||a.cushionAtr-b.cushionAtr);
 const viable=ranked.filter(x=>x.meetsArithmeticRR2);
 const chosen=[],groups=new Set();
 for(const x of ranked){const group=x.method+(x.origin==='NOW'?':NOW':'');if(!groups.has(group)){chosen.push(x);groups.add(group);}}
 for(const x of ranked){if(chosen.length>=8)break;if(!chosen.includes(x))chosen.push(x);}
 return {direction:side,directionalEvidence:evidence,generated:ranked.length,arithmeticViable:viable.length,
   nowCandidates:ranked.filter(x=>x.origin==='NOW').length,
   nearbyPending:viable.filter(x=>x.origin==='PULLBACK'&&x.priceDistanceAtr<=1).length,
   availableMethods:[...new Set(ranked.map(x=>x.method))],bestAlternatives:chosen.slice(0,8),
   caveat:'Evidence is contemporaneous only, NOT a proven direction predictor. 1R TP1 screening is descriptive and has NOT been validated out of sample. No automatic signals or historical account fills.'};
}
export function buildOpportunityContext(raw){
 const result={version:'V17_OPPORTUNITY_EXPLORER_V1',policy:'PRESERVE_ISSUED_SIGNALS_AND_ORIGINAL_V17',
    method:'Search NOW/EMA_PULLBACK/VWAP_RETEST/PRICE_RETEST/BREAKOUT for BOTH directions before choosing HOLD.',symbols:{}};
 for(const sym of ['BTC','ETH']){
   const market=parseMarket(symbolSection(raw,sym),sym);
   if(!positive(market.price)||!positive(market.atr)){
     result.symbols[sym]={status:'UNVERIFIED',reason:'MISSING_SPOT_OR_ATR'};continue;
   }
   result.symbols[sym]={status:'SCREENED',at:market.at,price:market.price,atr15:market.atr,
     trends:{h4:market.ema4h.align,d1:market.ema1d.align},
     cvd:{quality:market.cvdQuality,net5:market.cvd5,net15:market.cvd15,net30:market.cvd30},
     LONG:optionsFor(market,'LONG'),SHORT:optionsFor(market,'SHORT')};
 }
 return result;
}
