import test from 'node:test';
import assert from 'node:assert/strict';
import {parsePreparedSegments,assemblePreparedChains,auditPreparedSegments,planPreparedEvents} from './prepared-path-chain.mjs';
import {checkEvent} from './v17-core.mjs';
const k='2026-11-13 ';
function record(symbol,at,start,low,high,highAt,lowAt){
 const p=symbol==='ETH'?1950:69300;
 return '========== '+symbol+' ('+symbol+'USDT) ==========\n'
  +'timestamp_kst: '+k+at+'\n[1] REALTIME PRICE (AUTO)\nlastPrice: '+p+'\n'
  +'[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: '+p+' @ '+k+start
  +' | end_price: '+p+' @ '+k+at+'\npath_high/low: '+high+' @ '+k+highAt
  +' / '+low+' @ '+k+lowAt+'\n[7] HISTORICAL FLOW SNAPSHOT (AUTO)\n';
}
function raw(at,start,ethLow=1940,ethHigh=1950){
 return record('BTC',at,start,69000,69400,at,start)+'\n'+record('ETH',at,start,ethLow,ethHigh,at,start);
}
const previous={btcTime:k+'19:04:20',ethTime:k+'19:04:20'};
const a=raw('19:07:20','19:04:20');
const b=raw('19:10:20','19:07:20');
const c=raw('19:13:20','19:10:20');
const old={id:'ETH-HISTORY',symbol:'ETHUSDT',status:'ENTERED',verifiedUntil:k+'19:04:20',
 original:{direction:'SHORT',issuedAt:k+'19:01:00',entryLow:1944.8,entryHigh:1948.2,stop:1969,tp1:1933,tp2:1894.1,tp3:1882.5}};
test('parse each saved BTC ETH segment without confusing symbols',()=>{
 const p=parsePreparedSegments(a);
 assert.equal(p.ETH.start,k+'19:04:20');assert.equal(p.BTC.high,69400);
});
test('overlapping saved snapshots close earlier official path gap',()=>{
 const q=assemblePreparedChains([a,b],previous,c);
 assert(q.ETH.recovered);assert(q.ETH.bridgeCovered);assert.equal(q.ETH.segments.length,3);
 assert.equal(q.ETH.source,'USER_PREPARED_SNAPSHOT_PATH_CHAIN');
});
test('gaps bigger than 1m stay uncovered, even if latest quote is current',()=>{
 const q=assemblePreparedChains([a],previous,c);
 assert.equal(q.ETH.recovered,false);
});
test('no SL/TP in continuous path preserves entered state',()=>{
 const q=assemblePreparedChains([a,b],previous,c);
 const x=auditPreparedSegments(old,q.ETH.segments,{from:previous.ethTime});
 assert.equal(x.status,'ENTERED');assert.deepEqual(x.events,[]);assert.equal(x.ambiguous,null);
});
test('PENDING crossed once becomes entered by proof of original path',()=>{
 const p={...old,status:'PENDING'};
 const q=assemblePreparedChains([raw('19:07:20','19:04:20',1941,1942),raw('19:10:20','19:07:20',1941,1946)],previous,raw('19:13:20','19:10:20',1940,1947));
 const x=planPreparedEvents({ledger:[p],history:q,auditStarts:{'ETH-HISTORY':previous.ethTime},
 currentAtBySymbol:{ETH:k+'19:13:20',BTC:k+'19:13:20'}});
 assert.equal(x.events.length,1);assert.equal(x.events[0].kind,'ENTERED');assert.doesNotThrow(()=>checkEvent(x.events[0],[p]));
});
test('stop and profit within same segment are not classified as either',()=>{
 const seg={start:k+'19:04:20',end:k+'19:07:20',high:1971,low:1931,highAt:k+'19:06:20',lowAt:k+'19:05:20'};
 const x=auditPreparedSegments(old,[seg],{from:previous.ethTime});
 assert.equal(x.events.length,0);assert.equal(x.ambiguous.reason,'STOP_AND_TP_ORDER_AMBIGUOUS');
});
test('already UNVERIFIED is restored only after a full historical chain',()=>{
 const unknown={...old,status:'UNVERIFIED',unverifiedFrom:'ENTERED',verifiedUntil:k+'19:16:00'};
 const q=assemblePreparedChains([a,b],previous,c);
 const x=planPreparedEvents({ledger:[unknown],history:q,unverifiedOrigins:{'ETH-HISTORY':previous.ethTime},
 currentAtBySymbol:{ETH:k+'19:19:20',BTC:k+'19:19:20'}});
 assert.equal(x.events[0].kind,'ENTERED');assert.match(x.events[0].gapResolutionEvidence,/Complete/);
 assert.doesNotThrow(()=>checkEvent(x.events[0],[unknown]));
});
test('never record guessed entry when interval starts before last confirmed state',()=>{
 const p={...old,status:'PENDING'};
 const q=assemblePreparedChains([a,b],previous,c);
 const x=planPreparedEvents({ledger:[p],history:q,auditStarts:{'ETH-HISTORY':k+'19:11:00'},
 currentAtBySymbol:{ETH:k+'19:13:20',BTC:k+'19:13:20'}});
 assert.equal(x.events.length,0);
});
