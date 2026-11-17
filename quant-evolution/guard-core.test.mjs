import test from 'node:test';import assert from 'node:assert/strict';
import {guardedRollout,inspectDeployment,buildRollbackTree} from './guard-core.mjs';
import {mkdtemp,writeFile,mkdir,rm} from 'node:fs/promises';import {tmpdir} from 'node:os';import {join} from 'node:path';import {execFileSync} from 'node:child_process';
const expected={commit:'candidate',policySha256:'policy',registrySha256:'registry'};
const good={status:'ready',buildCommit:'candidate',policySha256:'policy',quantEngine:{registrySha256:'registry',automaticOrders:false}};
const intact={intact:true,originalsIntact:true,eventsIntact:true};
test('guard requires live commit, exact policy and model bytes, no orders, intact ledger',()=>{
 assert.equal(inspectDeployment({health:good,expected,integrity:intact}).passed,true);
 for(const health of [{...good,buildCommit:'old'},{...good,policySha256:'changed'},{...good,quantEngine:{registrySha256:'bad',automaticOrders:false}},{...good,quantEngine:{registrySha256:'registry',automaticOrders:true}}])assert.equal(inspectDeployment({health,expected,integrity:intact}).passed,false);
});
test('three actual consecutive matching checks finish a rollout without rollback',async()=>{
 let calls=0;const r=await guardedRollout({expected,checkpoint:'cp',readHealth:async()=>++calls<2?{...good,buildCommit:'old'}:good,verifyIntegrity:async()=>intact,wait:async()=>{},rollback:async()=>{throw Error('must not rollback');}});
 assert.equal(r.status,'VERIFIED_LIVE');assert.equal(calls,4);
});
test('policy mismatch triggers rollback and reports successful verification only after callback proof',async()=>{
 let calls=0;const r=await guardedRollout({expected,readHealth:async()=>({...good,policySha256:'wrong'}),verifyIntegrity:async()=>intact,wait:async()=>{},rollback:async()=>{calls++;return {verified:true,commit:'restored'};}});
 assert.equal(calls,1);assert.equal(r.status,'ROLLED_BACK');
});
test('concurrent source collision or failed restored health remains explicitly unverified',async()=>{
 const r=await guardedRollout({expected,readHealth:async()=>good,verifyIntegrity:async()=>({intact:false}),wait:async()=>{},rollback:async()=>({verified:false,reason:'CONCURRENT_SOURCE_CHANGE_NO_OVERWRITE'})});assert.equal(r.status,'ROLLBACK_UNVERIFIED');
});
test('controlled rehearsal exercises the same rollback path without corrupting trading logic',async()=>{
 const r=await guardedRollout({expected,drill:true,readHealth:async()=>good,verifyIntegrity:async()=>intact,wait:async()=>{},rollback:async()=>({verified:true})});assert.equal(r.status,'ROLLED_BACK');assert.ok(r.failure.reasons.includes('CONTROLLED_ROLLBACK_REHEARSAL'));
});
test('deployment timeout does not masquerade as live',async()=>{
 const r=await guardedRollout({expected,attempts:2,readHealth:async()=>{throw Error('offline');},verifyIntegrity:async()=>intact,wait:async()=>{},rollback:async()=>({verified:false})});assert.equal(r.status,'ROLLBACK_UNVERIFIED');
});
test('real Git rollback restores source while preserving every consumed holdout',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'v20-rollback-test-')),git=(...args)=>execFileSync('git',args,{cwd:dir,encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
 try{
  git('init','-q');git('config','user.name','Synthetic Test');git('config','user.email','synthetic@example.invalid');await mkdir(join(dir,'quant-evolution'));
  await writeFile(join(dir,'engine.txt'),'verified source');await writeFile(join(dir,'quant-evolution/holdout-consumption.json'),'["spent-1"]');git('add','.');git('commit','-qm','verified');const verifiedCommit=git('rev-parse','HEAD');
  await writeFile(join(dir,'engine.txt'),'failed source');await writeFile(join(dir,'quant-evolution/holdout-consumption.json'),'["spent-1","spent-2"]');git('add','.');git('commit','-qm','failed');const failedCommit=git('rev-parse','HEAD');
  const tree=buildRollbackTree(git,{verifiedCommit,failedCommit});assert.equal(git('show',tree+':engine.txt'),'verified source');assert.deepEqual(JSON.parse(git('show',tree+':quant-evolution/holdout-consumption.json')),['spent-1','spent-2']);
 }finally{await rm(dir,{recursive:true,force:true});}
});
