import {kstMs,kstTime,rawMarketWindows} from './price-path-recovery.mjs';
const live=new Set(['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED']);
const tpState=new Set(['STOPPED','TP3_DONE']);
const time=x=>kstMs(x);
export function parsePreparedSegments(raw){
 const windows=rawMarketWindows(raw),output={};
 for(const symbol of ['BTC','ETH']){
  const begin=raw.indexOf('========== '+symbol+' ('+symbol+'USDT) ==========');
  const end=symbol==='BTC'?raw.indexOf('========== ETH (ETHUSDT) ==========',begin+1):raw.length;
  const part=raw.slice(begin,end<0?undefined:end);
  const bounds=/^path_high\/low:\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*\/\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(part);
  if(!bounds)continue;
  const high=Number(bounds[1]),low=Number(bounds[3]),start=windows[symbol].pathStart,endAt=windows[symbol].pathEnd;
  if(!Number.isFinite(high)||!Number.isFinite(low)||!(0<low&&low<=high))continue;
  if(time(endAt)<time(start)||time(bounds[2])<time(start)-60000||time(bounds[2])>time(endAt)+60000||
   time(bounds[4])<time(start)-60000||time(bounds[4])>time(endAt)+60000)continue;
  output[symbol]={start,end:endAt,high,low,highAt:bounds[2],lowAt:bounds[4],observedAt:windows[symbol].timestamp};
 }
 return output;
}
// Uses only verbatim server-persisted snapshots, never fabricated intermediate bars.
// Contiguity tolerance is one minute because the client's paths use 1m buckets.
export function assemblePreparedChains(rawTexts,previousSnapshot,latestRaw){
 const result={BTC:{recovered:false},ETH:{recovered:false}};
 const current=rawMarketWindows(latestRaw);
 const data=rawTexts.map(x=>{try{return parsePreparedSegments(x);}catch{return {};}}).concat(parsePreparedSegments(latestRaw));
 for(const sym of ['BTC','ETH']){
  const prev=previousSnapshot?.[sym==='BTC'?'btcTime':'ethTime'];
  if(!prev)continue;
  const max=time(current[sym].timestamp),earliest=time(prev);
  const spans=data.map(x=>x[sym]).filter(x=>x&&time(x.end)>earliest&&time(x.start)<=max)
   .sort((a,b)=>time(a.start)-time(b.start)||time(a.end)-time(b.end));
  let through=earliest;
  const chosen=[];
  for(const seg of spans){
   if(time(seg.start)>through+60000)break;
   if(time(seg.end)<=through)continue;
   chosen.push(seg);
   through=Math.max(through,time(seg.end));
   if(through+60000>=max)break;
  }
  if(through+60000<max||!chosen.length)continue;
  const pathStart=time(current[sym].pathStart);
  result[sym]={recovered:true,source:'USER_PREPARED_SNAPSHOT_PATH_CHAIN',segments:chosen,
   bridgeCovered:through+60000>=pathStart,barCount:chosen.length,
   coveredFrom:prev,coveredThrough:kstTime(through),coveredFromMs:earliest,coveredThroughMs:through,
   currentAt:current[sym].timestamp,
   note:'Source is user-supplied automatic snapshots preserved verbatim, not independently fetched Bybit candles.'};
 }
 return result;
}
export function auditPreparedSegments(signal,segments,{from,initialState}={}){
 const o=signal.original||signal;
 const side=o.direction;
 const crossed=(seg,level,stop=false)=>side==='SHORT'?(stop?seg.high>=level:seg.low<=level):(stop?seg.low<=level:seg.high>=level);
 let state=initialState||signal.status;
 const events=[];
 let ambiguous=null;
 let prev=time(from);
 for(const seg of segments){
  if(tpState.has(state))break;
  const exitStop=crossed(seg,o.stop,true),exitT1=crossed(seg,o.tp1),exitT2=crossed(seg,o.tp2),exitT3=crossed(seg,o.tp3);
  const entry=seg.high>=o.entryLow&&seg.low<=o.entryHigh;
  if(state==='PENDING'){
   if(!entry)continue;
   // A crossing seen inside an overlapping span may predate the confirmed state.
   if(time(seg.start)<prev-60000||exitStop||exitT1){
    ambiguous={segment:seg,reason:'ENTRY_TIMING_OR_EXIT_ORDER_AMBIGUOUS'};break;
   }
   events.push({kind:'ENTERED',observedAt:seg.highAt,at:seg.end,high:seg.high,low:seg.low,source:seg});
   state='ENTERED';prev=time(seg.end);
   continue;
  }
  if(exitStop&&exitT1){ambiguous={segment:seg,reason:'STOP_AND_TP_ORDER_AMBIGUOUS'};break;}
  const next=exitStop?'STOPPED':exitT3?'TP3_DONE':exitT2?'TP2':exitT1?'TP1':null;
  if(!next||next===state||(state==='TP2'&&next==='TP1'))continue;
  if(time(seg.start)<prev-60000){
   ambiguous={segment:seg,reason:'PARTIAL_OVERLAP_POSSIBLE_OLD_CROSSING'};break;
  }
  events.push({kind:next,observedAt:next==='STOPPED'?(side==='SHORT'?seg.highAt:seg.lowAt):(side==='SHORT'?seg.lowAt:seg.highAt),
   at:seg.end,high:seg.high,low:seg.low,source:seg});
  state=next;prev=time(seg.end);
 }
 return {status:state,events,ambiguous};
}
export function planPreparedEvents({ledger,history,currentAtBySymbol,unverifiedOrigins={},auditStarts={}}){
 const result={events:[],notes:[],proofs:{}};
 for(const sym of ['BTC','ETH']){
  const proof=history[sym],signals=ledger.filter(x=>x.symbol===sym+'USDT'&&live.has(x.status));
  if(!signals.length)continue;
  if(!proof?.recovered){result.notes.push(sym+':PREPARED_CHAIN_INCOMPLETE');continue;}
  result.proofs[sym]={recovered:true,source:proof.source,bridgeCovered:proof.bridgeCovered,
   coveredFrom:proof.coveredFrom,coveredThrough:proof.coveredThrough,barCount:proof.barCount};
  for(const s of signals){
   const prior=s.status==='UNVERIFIED'?s.unverifiedFrom:s.status;
   const from=s.status==='UNVERIFIED'?unverifiedOrigins[s.id]:auditStarts[s.id]||s.verifiedUntil||s.original?.issuedAt;
   if(!from||!prior){result.notes.push(s.id+':NO_TRUSTWORTHY_BASELINE');continue;}
   if(time(proof.coveredFrom)>time(from)){
    result.notes.push(s.id+':PREPARED_CHAIN_NOT_FROM_CONFIRMED_STATE');continue;
   }
   const a=auditPreparedSegments(s,proof.segments,{from,initialState:prior});
   if(s.status==='UNVERIFIED'){
    if(a.ambiguous){result.notes.push(s.id+':PREPARED_CHAIN_AMBIGUOUS');continue;}
    const summary=a.events.map(e=>e.kind+'@'+e.observedAt).join(',');
    result.events.push({id:s.id,kind:a.status,at:currentAtBySymbol[sym],
      reason:'Reconciled previously unknown mechanical position path using server-persisted user snapshots.',
      evidence:'Source: verbatim prepared snapshots, '+proof.coveredFrom+' to '+proof.coveredThrough+'; '+(summary||'no levels crossed'),
      gapResolutionEvidence:'Complete overlapping client price-path coverage; no fabricated tick entry or exit time.',
      ...(a.status!=='PENDING'?{pricePathEvidence:'Prepared original path segments '+summary+'; no unsupported event order.'}:{}),
      recoveredPathEvents:a.events.map(e=>({kind:e.kind,observedAt:e.observedAt,high:e.high,low:e.low}))});
    continue;
   }
   for(const x of a.events){
    result.events.push({id:s.id,kind:x.kind,at:x.at,
      reason:'Mechanical V1.7 price touch from complete server-persisted original market snapshots.',
      evidence:'Client-supplied price path '+x.source.start+' to '+x.source.end+' H='+x.high+' L='+x.low,
      pricePathEvidence:'Source saved BTC/ETH auto-snapshot path; '+x.kind+' '+x.observedAt+' (minute granularity).',
      source:proof.source,observedMinute:x.observedAt});
   }
   if(a.ambiguous)result.notes.push(s.id+':PREPARED_CHAIN_AMBIGUOUS');
  }
 }
 result.events.sort((a,b)=>time(a.at)-time(b.at));
 return result;
}
