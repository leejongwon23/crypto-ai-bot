import {ledgerHash} from './v17-core.mjs';

// A deliberately narrow, user-authorized administrative closure. Never call
// this a chronologically *verified* STOP-before-TP3 result.
export const TWO_USER_TP3_IDS=Object.freeze([
 'V17-BTC-SHORT-20261114-083322-RETEST68310',
 'V17-ETH-SHORT-20261114-072532-NEAR1933'
]);
export const ATTEST_CONFIRM='USER_CONFIRMS_BOTH_TP3_SUCCESSES_AND_NO_EARLIER_STOPS';
const fail=s=>{throw Error(s);};
function dayLow(raw,sym){
 const match=raw.match(new RegExp('========== '+sym+' \\('+sym+'USDT\\) ==========([\\s\\S]*?)(?=========== (?:BTC|ETH) \\(|$)'));
 if(!match)fail('ATTEST_SNAPSHOT_SECTION_MISSING_'+sym);
 const low=match[1].match(/^24h high\/low:\s*[\d.]+\s*\/\s*([\d.]+)/m);
 if(!low)fail('ATTEST_SNAPSHOT_LOW_MISSING_'+sym);
 const n=Number(low[1]);if(!Number.isFinite(n)||n<=0)fail('ATTEST_SNAPSHOT_LOW_INVALID_'+sym);
 return n;
}
export function planUserAttestedTp3({ledger,sourceRaw,sourceSnapshotSha256,expectedLedgerSha256,signalIds,confirmation,at}){
 if(confirmation!==ATTEST_CONFIRM)fail('EXPLICIT_USER_TP3_CONFIRMATION_REQUIRED');
 if(!/^[a-f0-9]{64}$/.test(expectedLedgerSha256||'')||
    ledgerHash(ledger)!==expectedLedgerSha256)fail('ATTEST_LEDGER_CHANGED');
 if(!/^[a-f0-9]{64}$/.test(sourceSnapshotSha256||''))fail('ATTEST_SOURCE_SHA_REQUIRED');
 if(!/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(at||''))fail('ATTEST_TIME_REQUIRED');
 if(!Array.isArray(signalIds)||signalIds.length!==2||new Set(signalIds).size!==2||
    !TWO_USER_TP3_IDS.every(id=>signalIds.includes(id)))fail('ATTEST_ONLY_TWO_ORIGINAL_IDS');
 if(typeof sourceRaw!=='string'||!sourceRaw)fail('ATTEST_SAVED_SNAPSHOT_REQUIRED');
 return TWO_USER_TP3_IDS.map(id=>{
  const row=ledger.find(x=>x.id===id);
  if(!row)fail('ATTEST_ORIGINAL_ID_MISSING_'+id);
  if(!['ENTERED','TP1','TP2'].includes(row.status))fail('ATTEST_ONLY_KNOWN_ENTERED_OPEN_'+id);
  if(row.original?.direction!=='SHORT'||row.original?.id!==id)fail('ATTEST_ORIGINAL_IMMUTABLE_MISMATCH_'+id);
  const sym=row.symbol==='BTCUSDT'?'BTC':row.symbol==='ETHUSDT'?'ETH':null;
  if(!sym)fail('ATTEST_SYMBOL_MISMATCH_'+id);
  const low=dayLow(sourceRaw,sym);
  if(!(low<row.original.tp3))fail('ATTEST_LOW_DID_NOT_CROSS_TP3_'+id);
  return {id,symbol:row.symbol,kind:'TP3_DONE',at,
   source:'USER_ATTESTATION_WITH_RECORDED_24H_LOW',
   reason:'User expressly confirms TP3 success and that the original stop did not occur first; terminalized by explicit user choice.',
   evidence:sym+' saved snapshot 24h low '+low+' below original TP3 '+row.original.tp3+
     '; original stop '+row.original.stop+' not independently checked in missing interval.',
   pricePathEvidence:'Saved 24h low corroborates TP3 level touch, not chronological stop-first order.',
   provenance:'USER_ATTESTED_NOT_CHRONOLOGICALLY_VERIFIED',
   userConfirmedSuccessful:true,
   independentStopOrderVerified:false,
   realizedPnlVerified:false,
   exitTimeKnown:false,
   eventAtMeaning:'TIME_USER_CONFIRMATION_RECORDED_NOT_ACTUAL_TARGET_TOUCH',
   referenceSnapshotSha256:sourceSnapshotSha256,
   originalTp3:row.original.tp3,originalStop:row.original.stop,observedRollingLow:low};
 });
}
