import test from 'node:test';import assert from 'node:assert/strict';
import {sha} from './quant-structure.mjs';
import {modelExposure,planModelTransition,reviewResearchPath,freezeResearchForecasts,observedCounterexamples} from './quant-lifecycle.mjs';
import {compareIntegrity} from './research-control.mjs';
const t=Date.parse('2026-11-17T05:00:00Z');
const plan={entryLow:100,entryHigh:100,stop:95,tp1:110,tp2:120,tp3:130};
const f={direction:'LONG',frozenAt:t,plan};
const bar=(i,open,low,high,close=open)=>({t:t+i*60000,end:t+(i+1)*60000,intervalMs:60000,open,low,high,close});
test('paper forecast uses only first price touch after actual freeze',()=>{
 const r=reviewResearchPath(f,[bar(-1,100,94,131),bar(0,100,99,101),bar(1,100,94,101)]);
 assert.equal(r.status,'STOPPED');assert.equal(r.entryAt,t);assert.equal(r.grossR,-1);assert.equal(r.accountFill,false);assert.equal(r.netR,null);
});
test('entry and exit within one minute stays unknown in both directions',()=>{
 assert.equal(reviewResearchPath(f,[bar(0,100,94,131)]).status,'UNVERIFIED');
 const short={direction:'SHORT',frozenAt:t,plan:{entryLow:100,entryHigh:100,stop:105,tp1:90,tp2:80,tp3:70}};
 assert.equal(reviewResearchPath(short,[bar(0,100,69,106)]).status,'UNVERIFIED');
});
test('TP1 and TP2 accumulate, STOP/TP3 collision never chooses profitable ordering',()=>{
 const base=[bar(0,100,99,101),bar(1,101,100,112),bar(2,112,110,122)];
 assert.equal(reviewResearchPath(f,[...base,bar(3,120,94,131)]).status,'UNVERIFIED');
 const r=reviewResearchPath(f,[...base,bar(3,121,120,131)]);assert.equal(r.status,'TP3_COMPLETE');assert.equal(r.tp1,true);assert.equal(r.tp2,true);assert.equal(r.costsVerified,false);
});
test('data gap and adverse stop gap never become a favorable assumed fill',()=>{
 assert.equal(reviewResearchPath(f,[bar(1,100,99,101)]).reason,'POST_FREEZE_PATH_GAP');
 const r=reviewResearchPath(f,[bar(0,100,99,101),bar(1,90,89,96)]);assert.equal(r.grossR,-2);
});
test('stop before first touch invalidates a pending research setup',()=>{assert.equal(reviewResearchPath(f,[bar(0,94,93,94)]).status,'NOT_ENTERED_INVALIDATED');});
test('unsupported plans abstain independently for all four directions',()=>{
 const c={sourceSha256:'a'.repeat(64),symbols:{}};for(const symbol of ['BTC','ETH'])c.symbols[symbol]={asOf:'2026-11-17 14:00:00',structure:{regime:'MIXED'},directions:{}};
 const result=freezeResearchForecasts(c,{frozenAt:t});assert.equal(result.length,4);assert.ok(result.every(x=>x.status==='ABSTAIN'&&x.officialSignal===false));
});
test('newly frozen setups use supplied structure, never a fabricated price or score',()=>{
 const c={sourceSha256:'b'.repeat(64),symbols:{}};for(const symbol of ['BTC','ETH'])c.symbols[symbol]={asOf:'2026-11-17 14:00:00',structure:{regime:'TREND_UP',structuralEvents:{higherHighs:true,higherLows:true}},directions:{LONG:{A:{...plan,kind:'ARITHMETIC_CANDIDATE_NOT_FORMAL',rr2:4}}}};
 const r=freezeResearchForecasts(c,{frozenAt:t});assert.equal(r.filter(x=>x.status==='FROZEN_PENDING').length,2);assert.ok(r.every(x=>x.frozenAt===t));assert.deepEqual(r[0].plan,c.symbols.BTC.directions.LONG.A);
 c.symbols.BTC.structure.structuralEvents.invalidatedUp=true;assert.equal(freezeResearchForecasts(c,{frozenAt:t})[0].status,'ABSTAIN');
 assert.ok(freezeResearchForecasts(c,{frozenAt:t+120000}).every(x=>x.reason==='SOURCE_NOT_CURRENT_AT_ACTUAL_FREEZE'));
});
const model={id:'SYNTHETIC_TEST_MODEL',family:'ORDERED_STRUCTURE_CONTINUATION',logic:{all:['confirmedHigherHighs','confirmedHigherLows'],invalidates:['breakLastConfirmedLow']},validationHoldoutId:'first',rollout:{stage:'SHADOW',since:10}};
const evidence={modelSha256:sha({family:model.family,logic:model.logic}),ledgerInvariant:true,dataComplete:true,policyCompatible:true,performance:{eligible:true},phase:'SHADOW',phaseStartedAt:10,phaseEndedAt:20,holdoutId:'second',rollbackRehearsed:true};
test('shadow cannot influence formal analysis, canary is confined to its explicit asset',()=>{
 assert.equal(modelExposure(model,'BTC').useForFormalAnalysis,false);
 const next=planModelTransition(model,evidence);assert.equal(next.model.rollout.stage,'CANARY');assert.equal(modelExposure(next.model,'BTC').useForFormalAnalysis,true);assert.equal(modelExposure(next.model,'ETH').useForFormalAnalysis,false);
 assert.equal(model.rollout.stage,'SHADOW');
});
test('missing proof, reused holdout and quarantine cannot auto-promote',()=>{
 for(const e of [{...evidence,holdoutId:'first'},{...evidence,performance:{eligible:false}},{...evidence,ledgerInvariant:false}])assert.equal(planModelTransition(model,e).changed,false);
 assert.equal(planModelTransition({...model,rollout:{stage:'QUARANTINED',since:10}},evidence).changed,false);
});
test('ledger checkpoint tolerates new entries but rejects edited originals or removed events',()=>{
 const before={originals:[{idSha256:'a',sha256:'old'}],events:['e1','e2']};
 assert.equal(compareIntegrity(before,{originals:[...before.originals,{idSha256:'b',sha256:'new'}],events:['e1','e2','e3']}).intact,true);
 assert.equal(compareIntegrity(before,{originals:[{idSha256:'a',sha256:'changed'}],events:['e1','e2']}).intact,false);
 assert.equal(compareIntegrity(before,{originals:before.originals,events:['e2']}).intact,false);
});
test('observed contrary structure can trigger research before a STOP, never causal certainty',()=>{
 const forecast={...f,symbol:'BTC',sourceSha256:'old',anchors:{lastLow:{price:95,knownAt:t-60000}}};
 const obs={sourceSha256:'new',receivedAt:t+60000,symbols:{BTC:{asOf:'2026-11-17 14:01:00',lastPrice:99,structuralEvents:{lowerHighs:true,lowerLows:true}}}};
 const r=observedCounterexamples(forecast,[obs]);assert.equal(r.length,1);assert.equal(r[0].kind,'OBSERVED_OPPOSING_STRUCTURE_REVIEW');assert.equal(r[0].facts.breakLastConfirmedLow,false);assert.match(r[0].causalAttribution,/UNCONFIRMED/);
 assert.equal(observedCounterexamples(forecast,[{...obs,receivedAt:t-1}]).length,0);
});
