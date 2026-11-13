import test from 'node:test';
import assert from 'node:assert/strict';
import {kstMs,kstTime} from './price-path-recovery.mjs';
import {planVerifiedMechanicalEvents,validateRecoveredProof} from './official-price-tracker.mjs';
import {checkEvent} from './v17-core.mjs';
const stamp=(t)=>kstMs('2026-11-13 '+t);
const b=(time,low=1940,high=1950)=>({start:stamp(time),open:(low+high)/2,high,low,close:(low+high)/2});
const old={id:'ETH-TEST',symbol:'ETHUSDT',status:'ENTERED',verifiedUntil:'2026-11-13 19:04:20',
 original:{id:'ETH-TEST',symbol:'ETHUSDT',direction:'SHORT',issuedAt:'2026-11-13 19:01:00',entryLow:1944.8,entryHigh:1948.2,stop:1969,tp1:1933,tp2:1894.1,tp3:1882.5}};
const currentAtBySymbol={ETH:'2026-11-13 19:16:29',BTC:'2026-11-13 19:16:19'};
const makeProof=(bars)=>({ETH:{recovered:true,source:'BYBIT_LINEAR_PUBLIC_1M',barCount:bars.length,coveredFrom:kstTime(bars[0].start),coveredThrough:kstTime(bars.at(-1).start),coveredFromMs:bars[0].start,coveredThroughMs:bars.at(-1).start,bridgeCovered:true,bars}});
test('no signal means no automatic events and no network dependence',()=>{
 const out=planVerifiedMechanicalEvents({ledger:[],history:{},currentAtBySymbol});
 assert.deepEqual(out.events,[]);
});
test('entered short remains unchanged when every recovered candle is between SL and TP',()=>{
 const o=planVerifiedMechanicalEvents({ledger:[old],history:makeProof([b('19:04:00',1940,1950),b('19:05:00',1940,1955),b('19:06:00',1942,1956)]),currentAtBySymbol});
 assert.deepEqual(o.events,[]);assert.equal(o.proofs.ETH.recovered,true);
});
test('pending entry at 19:05 becomes ENTERED at minute-end 19:06',()=>{
 const p={...old,status:'PENDING',verifiedUntil:'2026-11-13 19:04:20'};
 const x=planVerifiedMechanicalEvents({ledger:[p],history:makeProof([b('19:04:00',1938,1940),b('19:05:00',1938,1946)]),currentAtBySymbol});
 assert.equal(x.events.length,1);assert.equal(x.events[0].kind,'ENTERED');
 assert.equal(x.events[0].at,'2026-11-13 19:06:00');
 assert.doesNotThrow(()=>checkEvent(x.events[0],[p]));
});
test('entered stop on a later minute is STOPPED and valid by original validator',()=>{
 const x=planVerifiedMechanicalEvents({ledger:[old],history:makeProof([b('19:04:00',1940,1950),b('19:05:00',1942,1971)]),currentAtBySymbol});
 assert.equal(x.events[0].kind,'STOPPED');assert.doesNotThrow(()=>checkEvent(x.events[0],[old]));
});
test('unknown old ENTERED is restored with complete proof of no exit',()=>{
 const u={...old,status:'UNVERIFIED',unverifiedFrom:'ENTERED',verifiedUntil:'2026-11-13 19:16:00'};
 const x=planVerifiedMechanicalEvents({ledger:[u],unverifiedOrigins:{'ETH-TEST':'2026-11-13 19:04:20'},
 history:makeProof([b('19:04:00',1940,1950),b('19:05:00',1942,1955),b('19:06:00',1948,1956)]),currentAtBySymbol});
 assert.equal(x.events.length,1);assert.equal(x.events[0].kind,'ENTERED');
 assert.match(x.events[0].gapResolutionEvidence,/Complete minute/);
 assert.doesNotThrow(()=>checkEvent(x.events[0],[u]));
});
test('unknown old ENTERED with observed stop resolves STOPPED',()=>{
 const u={...old,status:'UNVERIFIED',unverifiedFrom:'ENTERED',verifiedUntil:'2026-11-13 19:16:00'};
 const x=planVerifiedMechanicalEvents({ledger:[u],unverifiedOrigins:{'ETH-TEST':'2026-11-13 19:04:20'},
 history:makeProof([b('19:04:00',1940,1950),b('19:05:00',1942,1971)]),currentAtBySymbol});
 assert.equal(x.events[0].kind,'STOPPED');assert.doesNotThrow(()=>checkEvent(x.events[0],[u]));
});
test('ambiguous stop+target minute leaves recovered unknown unresolved',()=>{
 const u={...old,status:'UNVERIFIED',unverifiedFrom:'ENTERED',verifiedUntil:'2026-11-13 19:16:00'};
 const x=planVerifiedMechanicalEvents({ledger:[u],unverifiedOrigins:{'ETH-TEST':'2026-11-13 19:04:20'},
 history:makeProof([b('19:04:00',1940,1950),b('19:05:00',1931,1971)]),currentAtBySymbol});
 assert.deepEqual(x.events,[]);assert(x.notes.some(t=>t.includes('AMBIGUOUS')));
});
test('not enough minutes at the beginning must not infer healthy position',()=>{
 const u={...old,status:'UNVERIFIED',unverifiedFrom:'ENTERED',verifiedUntil:'2026-11-13 19:16:00'};
 const x=planVerifiedMechanicalEvents({ledger:[u],unverifiedOrigins:{'ETH-TEST':'2026-11-13 19:04:20'},
 history:makeProof([b('19:05:00',1940,1950),b('19:06:00',1940,1955)]),currentAtBySymbol});
 assert.equal(x.events.length,0);assert(x.notes.some(t=>t.includes('INCOMPLETE')));
});
test('first partial minute with stop touch is not backdated',()=>{
 const u={...old,status:'UNVERIFIED',unverifiedFrom:'ENTERED',verifiedUntil:'2026-11-13 19:16:00'};
 const x=planVerifiedMechanicalEvents({ledger:[u],unverifiedOrigins:{'ETH-TEST':'2026-11-13 19:04:20'},
 history:makeProof([b('19:04:00',1940,1971),b('19:05:00',1940,1955)]),currentAtBySymbol});
 assert.equal(x.events.length,0);assert(x.notes.some(t=>t.includes('FIRST_PARTIAL')));
});
