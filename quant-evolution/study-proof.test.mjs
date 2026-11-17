import test from 'node:test';import assert from 'node:assert/strict';
import {netResearchReturn,verifyStudyFiles} from './study-proof.mjs';
const f={direction:'LONG',plan:{entryLow:100,entryHigh:100}},o={status:'TP3_COMPLETE',entryAt:10,labelEnd:20,exitPrice:110};
const costs={sourceStatus:'DOCUMENTED_ASSUMPTIONS',entryFeeRate:.001,exitFeeRate:.001,adverseEntrySlippageBps:10,adverseExitSlippageBps:10,notional:1,fundingCoverageFrom:0,fundingCoverageThrough:30,fundingCoverageComplete:true,funding:[]};
test('net study return actually subtracts documented fees, adverse slippage and signed funding',()=>{
 const a=netResearchReturn(f,o,costs),b=netResearchReturn(f,o,{...costs,funding:[{at:15,rate:.001,markPrice:105}]});assert.ok(a<.1);assert.ok(b<a);
 assert.throws(()=>netResearchReturn(f,o,{...costs,fundingCoverageThrough:15}),/FUNDING/);
 assert.throws(()=>netResearchReturn(f,o,{...costs,fundingCoverageFrom:undefined}),/COMPLETE/);
});
test('unknown minute order or absent costs cannot acquire a net performance number',()=>{assert.throws(()=>netResearchReturn(f,{...o,status:'UNVERIFIED'},costs));assert.throws(()=>netResearchReturn(f,o,{}));});
test('a green boolean and arbitrary matching text files cannot pass actual study proof',async()=>{
 const result=await verifyStudyFiles({protocol:{status:'LOCKED'},cases:[],candidate:{unitTests:true,integrationTests:true,shadowVerified:true,rollbackRehearsed:true},proofs:{}},{read:async()=>'{"passed":true}'});assert.equal(result.eligible,false);assert.ok(result.reasons.some(x=>x.includes('TAP')||x.includes('UNIT_PROOF')));
});
