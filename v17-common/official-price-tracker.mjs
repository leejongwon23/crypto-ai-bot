import {auditMinuteCandles,kstMs,kstTime,floorMinute,verifyCandleCoverage,rawMarketWindows} from './price-path-recovery.mjs';
const ACTIVE=new Set(['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED']);
const DONE=new Set(['STOPPED','TP3_DONE']);
const has=x=>typeof x==='string'&&x.trim().length>0;
const endOfMinute=t=>kstTime(kstMs(t)+60000);
const initial=s=>s.status==='UNVERIFIED'?s.unverifiedFrom:s.status;
function eventEvidence(s,proof,change){
 return 'Public Bybit USDT-linear '+s.symbol+' 1m OHLC (all minutes verified), '
  +proof.coveredFrom+' through '+proof.coveredThrough
  +'; observed minute '+change.minute+' high '+change.high+' low '+change.low
  +'; crossed original '+change.kind+' level '+change.level
  +'; exact tick timestamp and actual account fill not asserted.';
}
/**
 * Pure planner; never writes. The server must serialize the resulting events
 * under a session lock, validate with checkEvent(), and store them append-only.
 * A quote-level historic fill is never synthesized from minute bars.
 */
export function planVerifiedMechanicalEvents({ledger,history,currentAtBySymbol,unverifiedOrigins={}}){
 const result={events:[],notes:[],proofs:{}};
 const market=['BTC','ETH'];
 for(const sym of market){
  const proof=history[sym];
  const signals=ledger.filter(s=>s.symbol===sym+'USDT'&&ACTIVE.has(s.status));
  if(!signals.length)continue;
  if(!proof?.recovered){result.notes.push(sym+':HISTORICAL_DATA_NOT_RECOVERED');continue;}
  const now=currentAtBySymbol[sym];
  if(!has(now))throw Error('SNAPSHOT_TIME_REQUIRED_'+sym);
  result.proofs[sym]={recovered:true,verified:true,bridgeCovered:proof.bridgeCovered===true,
   source:proof.source,coveredFrom:proof.coveredFrom,coveredThrough:proof.coveredThrough,
   barCount:proof.barCount};
  for(const signal of signals){
   const state=initial(signal);
   const historyFrom=signal.status==='UNVERIFIED'?
    unverifiedOrigins[signal.id]:signal.verifiedUntil||signal.original?.issuedAt;
   if(!has(historyFrom)||!ACTIVE.has(state)){
    result.notes.push(signal.id+':PRIOR_STATE_OR_ORIGIN_UNKNOWN');continue;
   }
   const fromMs=kstMs(historyFrom);
   if(proof.coveredFromMs>floorMinute(fromMs)){
    result.notes.push(signal.id+':INCOMPLETE_FROM_LAST_CONFIRMED_TIME');continue;
   }
   const first=proof.bars.find(c=>c.start===floorMinute(fromMs));
   if(!first){result.notes.push(signal.id+':MISSING_START_MINUTE');continue;}
   // If the first minute was already in progress at the last official check,
   // an observed SL/TP there might have preceded that check. Never backdate it.
   const o=signal.original||signal;
   const stopHit=o.direction==='SHORT'?first.high>=o.stop:first.low<=o.stop;
   const tpHit=o.direction==='SHORT'?first.low<=o.tp1:first.high>=o.tp1;
   const entryHit=first.high>=o.entryLow&&first.low<=o.entryHigh;
   if((state==='PENDING'&&entryHit)||(state!=='PENDING'&&(stopHit||tpHit))){
    result.notes.push(signal.id+':FIRST_PARTIAL_MINUTE_CONFLICT');continue;
   }
   const verdict=auditMinuteCandles({...signal,status:state},proof.bars,{initialStatus:state,afterMs:fromMs});
   if(signal.status==='UNVERIFIED'){
    if(verdict.ambiguous){
     result.notes.push(signal.id+':RECOVERED_MINUTE_ORDER_AMBIGUOUS');continue;
    }
    const last=verdict.events.at(-1);
    // Recovered OHLC only proves source-level crossings, not exact tick fills.
    const evidence='Recovered original '+signal.id+' from '+state+
     ' using every minute of '+proof.source+' '+proof.coveredFrom+
     ' ... '+proof.coveredThrough+'; '+verdict.events.map(x=>x.kind+'@'+x.minute).join(', ');
    result.events.push({
     id:signal.id,kind:verdict.status,at:now,
     reason:'Verified historical gap resolution: replayed source 1m candles against the immutable original plan.',
     evidence:evidence||'Recovered complete historical minute coverage without entry/stop/target crossings.',
     gapResolutionEvidence:'Complete minute-by-minute exchange history. Previous UNVERIFIED state restored only on unambiguous ordered price evidence. No guessed orders.',
     ...(verdict.status!=='PENDING'?{pricePathEvidence:last?eventEvidence(signal,proof,last):'Verified no mechanical exit within recovered 1m history; original '+state+' status restored.'}:{}),
     recoveryAudit:{priorState:state,observations:verdict.events,coverage:[proof.coveredFrom,proof.coveredThrough],source:proof.source},
    });
    continue;
   }
   for(const change of verdict.events){
    const evt={
     id:signal.id,kind:change.kind,at:endOfMinute(change.minute),
     reason:'Automatic mechanical V1.7 price-level crossing verified from Bybit public 1m OHLC; no AI opinion or further trigger.',
     evidence:eventEvidence(signal,proof,change),
     pricePathEvidence:eventEvidence(signal,proof,change),
     observedMinute:change.minute,source:proof.source,
    };
    if(kstMs(evt.at)>kstMs(now))throw Error('FUTURE_AUTO_EVENT_FORBIDDEN');
    result.events.push(evt);
   }
   if(verdict.ambiguous){
    result.events.push({
     id:signal.id,kind:'UNVERIFIED',at:now,
     reason:'The same minute contains competing entry/stop/TP price levels without reliable intraminute order.',
     evidence:'Public Bybit 1m OHLC '+proof.coveredFrom+' ... '+proof.coveredThrough+
      ': '+verdict.ambiguous.minute+', high='+verdict.ambiguous.high+', low='+verdict.ambiguous.low,
     uncertaintyScope:verdict.ambiguous.reason,
     priorConfirmedState:verdict.status,
     marketReassessment:'Independent current market LONG/SHORT assessment continues; earlier fill/exit order is not invented.',
    });
   }
  }
 }
 result.events.sort((a,b)=>kstMs(a.at)-kstMs(b.at));
 return result;
}
export function validateRecoveredProof(raw,previousSnapshot,proofs){
 const market=rawMarketWindows(raw);
 const result={};
 for(const symbol of ['BTC','ETH']){
  const p=proofs?.[symbol];
  const prev=previousSnapshot?.[symbol==='BTC'?'btcTime':'ethTime'];
  if(!p?.recovered||!prev){result[symbol]={recovered:false};continue;}
  const from=kstMs(prev),start=kstMs(market[symbol].pathStart);
  const bridge=p.bridgeCovered&&p.coveredFrom<=kstTime(floorMinute(from))
   &&p.coveredThrough>=kstTime(floorMinute(start));
  result[symbol]=bridge?{recovered:true,source:p.source,coveredFrom:p.coveredFrom,coveredThrough:p.coveredThrough}
   :{recovered:false};
 }
 return result;
}
