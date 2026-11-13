import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {prepare,submissionContract,validateOfficialSubmission,ledgerHash,VERSION,CLAUSES} from './v17-core.mjs';
import {applyCachedPolicyResponse} from './cached-policy-response.mjs';
const raw=readFileSync(new URL('./fixture-private.txt',import.meta.url),'utf8');
const initial=prepare({raw,session:'OFFICIAL_V17',ledger:[]});
const env=analysis=>({raw,session:'OFFICIAL_V17',policySha256:initial.policy.sha256,snapshotSha256:initial.snapshot.sha256,beforeLedgerSha256:initial.beforeLedgerSha256,decisionKey:initial.decisionKey,analysis});
function completeUnverifiedHold(){
 const o=structuredClone(submissionContract().skeleton);
 o.reportMarkdown='합성 회귀 시험: BTC/ETH 신규 진입 미발행, 과거 계좌 체결 미확인';
 for(const d of Object.values(o.directions)){
  d.marketEvidence='UNVERIFIED: test fixture cannot support an actual trade recommendation';
  d.counterEvidence='UNVERIFIED: no live exchange validation';
  for(const v of Object.values(d.alternatives)){v.judgement='HOLD';v.evidence='UNVERIFIED: isolated synthetic regression fixture';}
 }
 o.coupling={status:'UNVERIFIED',evidence:'No verified matched live trajectories'};
 for(const c of Object.values(o.checks)){c.status='UNVERIFIED';c.evidence='Synthetic fixture, not an official discretionary trade review';}
 for(const c of Object.values(o.clauses)){c.status='UNVERIFIED';c.evidence='Synthetic fixture, no live-trading confidence claim';}
 o.audit20={rules:'Original V1.7 and immutable ledger unaffected',bias:'None inferred from synthetic case',evidence:'No live position claimed'};
 o.conclusions={BTC:{decision:'HOLD',evidence:'Synthetic fixture; no entry'},ETH:{decision:'HOLD',evidence:'Synthetic fixture; no entry'}};
 return o;
}
test('full typed contract is available with current V1.7 and no signal mutation',()=>{
 const c=initial.requirements.submissionContract;
 assert.equal(c.version,'V17_STRUCTURED_SUBMISSION_CONTRACT_V1');
 assert.ok(VERSION.includes('1.0.3_AUDITED_AUTO_SWITCH'));
 assert.equal(Object.keys(c.skeleton.directions).length,4);
 assert.equal(Object.keys(c.skeleton.checks).length,16);
 assert.deepEqual(Object.keys(c.skeleton.clauses),CLAUSES);
 for(const x of Object.values(c.skeleton.directions)){assert.deepEqual(Object.keys(x.alternatives),['A','B','C','D','E']);}
 assert.equal(c.skeleton.reportMarkdown,null);
 assert.deepEqual(c.skeleton.newSignals,[]);
 assert.deepEqual(c.skeleton.events,[]);
 assert.equal(c.skeleton.autoSwitchAudit.BTC.price,null);
 assert.equal(initial.ledger.length,0);
});
test('correct official report key is required, alias stays rejected',()=>{
 assert.throws(()=>validateOfficialSubmission(env({report:'not valid'}),[]),/COMPLETE_USER_REPORT_REQUIRED/);
 assert.throws(()=>validateOfficialSubmission(env({userReport:'not valid'}),[]),/COMPLETE_USER_REPORT_REQUIRED/);
 assert.throws(()=>validateOfficialSubmission(env({reportMarkdown:'test-only'}),[]),/FOUR_DIRECTIONS_NOT_ANALYZED/);
});
test('exact raw checksum and unchanged ledger/key are still mandatory',()=>{
 const complete=completeUnverifiedHold();
 assert.throws(()=>validateOfficialSubmission({...env(complete),raw:raw+' '},[]),/SNAPSHOT_SHA_MISMATCH/);
 assert.throws(()=>validateOfficialSubmission({...env(complete),beforeLedgerSha256:'WRONG'},[]),/LEDGER_HAS_CHANGED_FETCH_AGAIN/);
 assert.throws(()=>validateOfficialSubmission({...env(complete),decisionKey:'WRONG'},[]),/DECISION_KEY_INVALID/);
 assert.equal(ledgerHash([]),initial.beforeLedgerSha256);
});
test('complete evidence-shaped HOLD remains accepted without inventing a signal',()=>{
 const validated=validateOfficialSubmission(env(completeUnverifiedHold()),[]);
 assert.equal(validated.snapshot.sha256,initial.snapshot.sha256);
 assert.equal(validated.policySha256,initial.policy.sha256);
 assert.deepEqual(validated.newSignals,[]);
 assert.deepEqual(validated.events,[]);
 assert.equal(initial.beforeLedgerSha256,prepare({raw,ledger:[]}).beforeLedgerSha256);
 assert.equal(initial.decisionKey,prepare({raw,ledger:[]}).decisionKey);
});
test('cached policy response preserves the contract and all hashes',()=>{
 const c=applyCachedPolicyResponse(initial,initial.policy.sha256);
 assert.equal(c.policy.text,undefined);
 assert.deepEqual(c.requirements.submissionContract,initial.requirements.submissionContract);
 assert.equal(c.snapshot.sha256,initial.snapshot.sha256);
 assert.equal(c.decisionKey,initial.decisionKey);
 assert.deepEqual(c.ledger,initial.ledger);
});
test('saved replay never reopens an already persisted decision',()=>{
 const saved={id:'synthetic'};
 const r=prepare({raw,ledger:[],saved});
 assert.equal(r.status,'OFFICIAL_REPLAY');
 assert.strictEqual(r.officialDecision,saved);
});
