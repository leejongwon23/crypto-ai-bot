import test from 'node:test';
import assert from 'node:assert/strict';
import {summarizeOfficialOutcomes,replayOriginalSignal} from './forensic-outcome-audit.mjs';
const base={id:'test',symbol:'BTCUSDT',direction:'SHORT',issuedAt:'2026-11-16 08:00:12',status:'PENDING',entryLow:100,entryHigh:102,stop:105,tp1:98,tp2:96,tp3:94};
const c=(t,open,high,low,close)=>({at:'2026-11-16 08:'+t+':00',open,high,low,close});
test('26-row style outcome report does not call 7/3 a verified trading win rate',()=>{
  const l=[...Array.from({length:7},(_,i)=>({id:'stop'+i,status:'STOPPED'})),...Array.from({length:3},(_,i)=>({id:'tp'+i,status:'TP3_DONE'})),{id:'early',status:'EARLY_EXIT'}];
  const o=summarizeOfficialOutcomes(l);assert.equal(o.stoppedAmongStopOrTP3Pct,70);assert.equal(o.censoredOrOtherCount,1);assert.equal(o.performanceVerified,false);
});
test('entry after issuance followed by stop before TP1 is loss evidence, not TP success',()=>{
  const r=replayOriginalSignal(base,[c('00',99,99,97,98),c('01',99,101,99,100),c('02',100,106,100,105)]);
  assert.equal(r.status,'STOPPED');assert.deepEqual(r.events.map(x=>x.kind),['ENTERED','STOPPED']);assert.equal(r.firstTargetReached,false);
});
test('TP3 requires all targets chronologically after entry',()=>{
  const r=replayOriginalSignal(base,[c('01',103,103,101,101),c('02',101,101,97,97),c('03',97,97,95,95),c('04',95,95,93,93)]);
  assert.equal(r.status,'TP3_DONE');assert.deepEqual(r.events.map(x=>x.kind),['ENTERED','TP1','TP2','TP3']);
});
test('ambiguous stop and target candle cannot be counted as a winner or loser',()=>{
  const r=replayOriginalSignal(base,[c('01',103,103,101,101),c('02',101,106,97,101)]);
  assert.equal(r.status,'UNVERIFIED');assert.equal(r.reason,'STOP_AND_TARGET_SAME_MINUTE');
});
test('issuance minute crossing is not automatically backfilled as a fill',()=>{
  const r=replayOriginalSignal(base,[c('00',101,103,99,101),c('01',101,103,101,102)]);
  assert.equal(r.status,'UNVERIFIED');assert.equal(r.reason,'ISSUE_MINUTE_PARTIAL_ORDER');
});
test('signal price invalidity, bar order, and duplicate ledger ids are rejected',()=>{
  assert.throws(()=>replayOriginalSignal({...base,stop:95},[]),/PRICE_GEOMETRY/);
  assert.throws(()=>replayOriginalSignal(base,[c('02',100,101,99,100),c('01',100,101,99,100)]),/INVALID_OR_OUT_OF_ORDER_CANDLE/);
  assert.throws(()=>summarizeOfficialOutcomes([{id:'a',status:'STOPPED'},{id:'a',status:'TP3_DONE'}]),/DUPLICATE_OR_MISSING_ID/);
});
