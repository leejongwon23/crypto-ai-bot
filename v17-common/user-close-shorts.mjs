// User-directed administrative closure of old analytical SHORT signals.
// This never attests to an exchange fill, stop-loss, TP, actual account exit or PnL.
export const USER_CLOSE_SHORTS_KIND='V17_USER_CLOSE_ALL_EXISTING_SHORT_SIGNALS';
export const USER_CLOSE_SHORTS_CONFIRM='USER_DIRECTS_CLOSE_ALL_EXISTING_BTC_ETH_SHORT_SIGNALS';
export const USER_CLOSED='USER_CLOSED';
export const closeableShorts=ledger=>ledger.filter(row=>row?.original?.direction==='SHORT'&&
 ['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(row.status)&&
 ['BTCUSDT','ETHUSDT'].includes(row.symbol));
export function planUserCloseShorts({ledger,signalIds,at}={}){
 if(!Array.isArray(ledger)||!Array.isArray(signalIds)||signalIds.length===0||new Set(signalIds).size!==signalIds.length)
  throw Error('USER_SHORT_CLOSE_EXACT_IDS_REQUIRED');
 if(typeof at!=='string'||!/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(at))throw Error('USER_SHORT_CLOSE_TIME_INVALID');
 const eligible=closeableShorts(ledger);
 const expected=new Set(eligible.map(s=>s.id));
 if(signalIds.length!==eligible.length||signalIds.some(id=>!expected.has(id)))
  throw Error('USER_SHORT_CLOSE_MUST_INCLUDE_ALL_AND_ONLY_ACTIVE_SHORT_IDS');
 return eligible.map(s=>({
  id:s.id,kind:USER_CLOSED,at,
  reason:'Explicit user instruction to end tracking all existing BTC/ETH SHORT signals.',
  evidence:'Direct user instruction (Korean): 기존숏전부종료처리하고 수정 저장해',
  provenance:'USER_DIRECTED_ADMINISTRATIVE_TRACKING_CLOSURE',
  explicitUserInstruction:'CLOSE_ALL_EXISTING_SHORT_SIGNALS',
  noTradeExecutionClaim:true,
  accountPositionUnverified:true,
  historicalOutcomeUnverified:true,
  priorAnalyticalStatus:s.status,
  priorConfirmedState:s.status==='UNVERIFIED'?s.unverifiedFrom||'UNKNOWN':s.status,
  originalSignalPreserved:true,
  noPnLClaim:true,
 }));
}
