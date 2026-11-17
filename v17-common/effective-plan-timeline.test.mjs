import test from 'node:test';
import assert from 'node:assert/strict';
import {planForSpan} from './effective-plan-timeline.mjs';
import {auditMinuteCandles,kstMs} from './price-path-recovery.mjs';
import {planVerifiedMechanicalEvents} from './official-price-tracker.mjs';
import {auditPreparedSegments} from './prepared-path-chain.mjs';
const at=t=>'2026-11-17 '+t,ms=t=>kstMs(at(t));
const base={id:'SYNTHETIC',symbol:'ETHUSDT',status:'ENTERED',verifiedUntil:at('06:35:19'),
 original:{id:'SYNTHETIC',symbol:'ETHUSDT',issuedAt:at('06:00:00'),direction:'LONG',entryLow:100,entryHigh:100,stop:90,tp1:110,tp2:120,tp3:130}};
const p1={stop:95,tp1:110,tp2:120,tp3:130},p2={stop:97,tp1:110,tp2:125,tp3:135};
const single={...base,version:1,effectivePlan:p1,planEffectiveAt:at('06:36:00'),managementRevisions:[{at:at('06:36:00'),plan:p1}]};
const bar=(t,low,high=99)=>({start:ms(t),open:high,high,low,close:high});
test('post-revision low94 hits effective STOP95 while preserving original STOP90',()=>{
 const input=JSON.stringify(single),r=auditMinuteCandles(single,[bar('06:37:00',94)],{afterMs:ms('06:36:00')});
 assert.equal(r.status,'STOPPED');assert.equal(r.events[0].level,95);assert.equal(JSON.stringify(single),input);
});
test('earlier low94 does not use a future STOP95 retroactively',()=>{
 const r=auditMinuteCandles(single,[bar('06:34:00',94)],{afterMs:ms('06:33:00')});
 assert.equal(r.status,'ENTERED');assert.equal(r.events.length,0);
});
test('multiple revisions select the plan at each observation, not the latest plan for all history',()=>{
 const s={...single,version:2,effectivePlan:p2,planEffectiveAt:at('06:38:00'),managementRevisions:[...single.managementRevisions,{at:at('06:38:00'),plan:p2}]};
 const r=auditMinuteCandles(s,[bar('06:37:00',96),bar('06:39:00',96)],{afterMs:ms('06:36:00')});
 assert.equal(r.events.length,1);assert.equal(r.events[0].minute,at('06:39:00'));assert.equal(r.events[0].level,97);
});
test('intraminute revision with potential crossing remains ambiguous',()=>{
 const s={...single,planEffectiveAt:at('06:36:30'),managementRevisions:[{at:at('06:36:30'),plan:p1}]};
 const r=auditMinuteCandles(s,[bar('06:36:00',94)],{afterMs:ms('06:35:00')});
 assert.equal(r.events.length,0);assert.equal(r.ambiguous.reason,'MANAGEMENT_REVISION_INSIDE_1M_WITH_CROSSING');
});
test('missing multi-version history cannot silently apply last plan',()=>{
 const s={...single,version:2,managementRevisions:undefined};
 assert.throws(()=>planForSpan(s,ms('06:37:00'),ms('06:38:00')),/INCOMPLETE/);
 const r=auditMinuteCandles(s,[bar('06:37:00',94)],{afterMs:ms('06:36:00')});
 assert.equal(r.ambiguous.reason,'REVISION_HISTORY_INCOMPLETE');
});
test('reversed history and latest-plan mismatch reject',()=>{
 const s={...single,version:2,managementRevisions:[{at:at('06:38:00'),plan:p2},{at:at('06:36:00'),plan:p1}]};
 assert.throws(()=>planForSpan(s,ms('06:37:00'),ms('06:38:00')),/ORDER/);
 assert.throws(()=>planForSpan({...single,effectivePlan:p2},ms('06:37:00'),ms('06:38:00')),/MISMATCH/);
});
test('SHORT mirrored protective stop is applied prospectively',()=>{
 const original={...base.original,direction:'SHORT',stop:110,tp1:90,tp2:80,tp3:70};
 const plan={stop:105,tp1:90,tp2:80,tp3:70};
 const s={...base,original,version:1,effectivePlan:plan,planEffectiveAt:at('06:36:00'),managementRevisions:[{at:at('06:36:00'),plan}]};
 const r=auditMinuteCandles(s,[bar('06:37:00',101,106)],{afterMs:ms('06:36:00')});
 assert.equal(r.status,'STOPPED');assert.equal(r.events[0].level,105);
});
test('automatic event planner receives effective history without backdating first partial minute',()=>{
 const s={...single,verifiedUntil:at('06:36:20')};
 const bars=[bar('06:36:00',96),bar('06:37:00',94)];
 const proof={recovered:true,source:'USER_SUBMITTED_BYBIT_1M',coveredFrom:at('06:36:00'),coveredThrough:at('06:37:00'),coveredFromMs:ms('06:36:00'),barCount:2,bars};
 const r=planVerifiedMechanicalEvents({ledger:[s],history:{ETH:proof},currentAtBySymbol:{ETH:at('06:38:00')}});
 assert.equal(r.events[0].kind,'STOPPED');assert.equal(r.events[0].at,at('06:38:00'));
 const partial={...proof,bars:[bar('06:36:00',94),bar('06:37:00',96)]};
 const ambiguous=planVerifiedMechanicalEvents({ledger:[s],history:{ETH:partial},currentAtBySymbol:{ETH:at('06:38:00')}});
 assert.equal(ambiguous.events.length,0);assert.ok(ambiguous.notes.some(x=>x.includes('FIRST_PARTIAL')));
});
test('prepared aggregate path also uses effective plan; overlapping revision never inferred',()=>{
 const seg={start:at('06:37:00'),end:at('06:37:59'),low:94,high:99,lowAt:at('06:37:00'),highAt:at('06:37:59')};
 const a=auditPreparedSegments(single,[seg],{from:at('06:36:20'),initialState:'ENTERED'});
 assert.equal(a.status,'STOPPED');
 const b=auditPreparedSegments(single,[{...seg,start:at('06:35:00')}],{from:at('06:34:20'),initialState:'ENTERED'});
 assert.equal(b.events.length,0);assert.equal(b.ambiguous.reason,'MANAGEMENT_REVISION_INSIDE_AGGREGATE_PATH');
});
test('no revision preserves original events and same-minute STOP/TP ambiguity',()=>{
 const r=auditMinuteCandles(base,[bar('06:37:00',89)],{afterMs:ms('06:36:00')});
 assert.equal(r.events[0].level,90);
 const a=auditMinuteCandles(base,[bar('06:37:00',89,111)],{afterMs:ms('06:36:00')});
 assert.equal(a.ambiguous.reason,'STOP_AND_TARGET_SAME_1M');
});
