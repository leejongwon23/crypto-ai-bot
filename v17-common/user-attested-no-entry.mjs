// User-reported exchange-account execution state is NOT market price-touch proof.
// Do not change signal.status, original values, or historical UNVERIFIED events.
export const USER_NO_ENTRY_KIND = 'V17_USER_ATTESTED_NO_ENTRY_TWO_ONLY';
export const USER_NO_ENTRY_CONFIRM = 'USER_CONFIRMS_BTC_AND_ETH_NO_ACCOUNT_ENTRY';
export const USER_NO_ENTRY_IDS = Object.freeze([
 'V17-BTC-SHORT-20261114-125202-RETEST67230',
 'V17-ETH-SHORT-20261114-125216-RETEST19055'
]);
const fail = m => { throw Error(m); };
export function planUserNoEntryAttestation({session,kind,confirmation,signalIds,ledger,at}){
 if(session !== 'OFFICIAL_V17' || kind !== USER_NO_ENTRY_KIND)fail('USER_NO_ENTRY_SCOPE_FORBIDDEN');
 if(confirmation !== USER_NO_ENTRY_CONFIRM)fail('USER_NO_ENTRY_CONFIRMATION_REQUIRED');
 if(!Array.isArray(signalIds) || signalIds.length !== 2 || new Set(signalIds).size !== 2 ||
    !USER_NO_ENTRY_IDS.every(id => signalIds.includes(id)))fail('USER_NO_ENTRY_EXACT_ORIGINAL_IDS_REQUIRED');
 if(!/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(at||''))fail('USER_NO_ENTRY_TIME_INVALID');
 return USER_NO_ENTRY_IDS.map(id => {
  const row = ledger.find(r=>r.id===id);
  if(!row || row.original?.id!==id || row.original?.status!=='PENDING' ||
     row.original?.direction!=='SHORT' || row.status!=='UNVERIFIED' || row.unverifiedFrom!=='PENDING')
   fail('USER_NO_ENTRY_UNEXPECTED_SIGNAL_STATE_'+id);
  return {
   id, kind:'USER_ACCOUNT_ENTRY_NOT_EXECUTED', at,
   report:'NOT_ENTERED', source:'EXPLICIT_USER_ACCOUNT_ATTESTATION',
   historicalMarketTouch:'UNVERIFIED', priorLedgerStatus:row.status,
   originalStatus:row.original.status,
   reason:'User explicitly confirmed no BTC/ETH account position entry; previous missing exchange price path remains unverified.',
   doesNotModifyFormalStatus:true
  };
 });
}
