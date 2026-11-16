// Read-only audit. Never manufactures market signals, account fills or realized PnL.
// Intended for chronological verification of existing official V1.7 records.
const TERMINAL = new Set(['STOPPED','TP3_DONE','EARLY_EXIT','REVERSE_EXIT','CANCELLED','REDESIGNED_CANCELLED','USER_CLOSED']);
const PAIR = new Set(['BTCUSDT','ETHUSDT']);
const ms = s => Date.parse(String(s).replace(' ', 'T') + '+09:00');
const validNumber = n => typeof n === 'number' && Number.isFinite(n);
const pct = (a,b) => b ? (a / b * 100) : null;

export function summarizeOfficialOutcomes(ledger) {
  if (!Array.isArray(ledger)) throw Error('LEDGER_ARRAY_REQUIRED');
  const statusCounts = {};
  const seen = new Set();
  for (const row of ledger) {
    if (!row?.id || seen.has(row.id)) throw Error('DUPLICATE_OR_MISSING_ID');
    seen.add(row.id);
    statusCounts[row.status] = (statusCounts[row.status] || 0) + 1;
  }
  const stopped = statusCounts.STOPPED || 0;
  const tp3 = statusCounts.TP3_DONE || 0;
  const knownDirectionalTerminal = stopped + tp3;
  const censored = ledger.length - knownDirectionalTerminal;
  return {
    registeredSignals:ledger.length,
    statusCounts,
    verifiedStopCount:stopped,
    verifiedTP3Count:tp3,
    stoppedAmongStopOrTP3Pct:knownDirectionalTerminal ? pct(stopped,knownDirectionalTerminal) : null,
    censoredOrOtherCount:censored,
    interpretation:'STOPPED/(STOPPED+TP3_DONE) is a descriptive selected-outcome ratio, NOT user-account loss rate, full strategy win rate, or estimated predictive accuracy.',
    performanceVerified:false
  };
}

// Requires minute candle timestamps denoting minute STARTS, not candle end or arbitrary snapshots.
// A first partial issuance minute is never retrospectively interpreted as a fill.
export function replayOriginalSignal(signal,minuteCandles) {
  const o = signal?.original || signal;
  if (!o || !PAIR.has(o.symbol) || !['LONG','SHORT'].includes(o.direction)) throw Error('INVALID_SIGNAL');
  if (!Number.isFinite(ms(o.issuedAt))) throw Error('INVALID_ISSUE_TIME');
  for (const k of ['entryLow','entryHigh','stop','tp1','tp2','tp3']) if (!validNumber(o[k])) throw Error('INVALID_PRICE_'+k);
  if (!(o.entryLow <= o.entryHigh)) throw Error('ENTRY_REVERSED');
  if (o.direction === 'LONG' ? !(o.stop<o.entryLow && o.entryHigh<o.tp1 && o.tp1<o.tp2 && o.tp2<o.tp3)
    : !(o.stop>o.entryHigh && o.entryLow>o.tp1 && o.tp1>o.tp2 && o.tp2>o.tp3)) throw Error('PRICE_GEOMETRY');
  if (!Array.isArray(minuteCandles)) throw Error('CANDLES_REQUIRED');
  let prev= -Infinity;
  const bars=minuteCandles.map(c=>{
    const t=ms(c.at);
    if (!Number.isFinite(t) || t<=prev || ![c.open,c.high,c.low,c.close].every(validNumber) || c.low>Math.min(c.open,c.close) || c.high<Math.max(c.open,c.close) || c.low>c.high) throw Error('INVALID_OR_OUT_OF_ORDER_CANDLE');
    prev=t;return {...c,t};
  });
  const issued=ms(o.issuedAt);
  const issueMinute=Math.floor(issued/60000)*60000;
  const long=o.direction==='LONG';
  const levelHit=(c,k)=>long ? c.high>=o[k] : c.low<=o[k];
  const stopHit=c=>long ? c.low<=o.stop : c.high>=o.stop;
  const entryHit=c=>c.low<=o.entryHigh && c.high>=o.entryLow;
  let state=o.status==='NEW_ENTRY'?'ENTERED':'PENDING';
  let maxTP=0;
  let entryAt=null;
  const events=[];
  for (const c of bars) {
    if(c.t<issueMinute) continue;
    if(c.t===issueMinute){
      // Even a known high/low on this minute may predate the issuance tick.
      if((state==='PENDING'&&entryHit(c)) || (state==='ENTERED'&&(stopHit(c)||levelHit(c,'tp1'))))
        return {status:'UNVERIFIED',reason:'ISSUE_MINUTE_PARTIAL_ORDER',events,entryAt:null,firstTargetReached:false,terminal:false};
      continue;
    }
    if(state==='PENDING'){
      if(!entryHit(c)) continue;
      if(stopHit(c)||levelHit(c,'tp1'))
        return {status:'UNVERIFIED',reason:'ENTRY_AND_BOUNDARY_SAME_MINUTE',events,entryAt:null,firstTargetReached:false,terminal:false};
      state='ENTERED';entryAt=c.at;events.push({kind:'ENTERED',at:c.at});continue;
    }
    const sl=stopHit(c);
    const hit=[1,2,3].filter(k=>levelHit(c,'tp'+k));
    if(sl && hit.length) return {status:'UNVERIFIED',reason:'STOP_AND_TARGET_SAME_MINUTE',events,entryAt,firstTargetReached:maxTP>=1,terminal:false};
    if(sl){events.push({kind:'STOPPED',at:c.at});return {status:'STOPPED',events,entryAt,firstTargetReached:maxTP>=1,terminal:true};}
    if(hit.length){
      const peak=Math.max(...hit);
      for(let k=maxTP+1;k<=peak;k++)events.push({kind:'TP'+k,at:c.at});
      maxTP=peak;
      if(peak===3)return {status:'TP3_DONE',events,entryAt,firstTargetReached:true,terminal:true};
    }
  }
  return {status:maxTP ? 'TP'+maxTP : state,events,entryAt,firstTargetReached:maxTP>=1,terminal:false};
}
