import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {POLICY_TEXT,POLICY_SHA256,POLICY_BYTES,POLICY_LINES} from './policy-v18.mjs';
import {OFFICIAL_POLICY,VERSION,CLAUSES,prepare,validateOfficialSubmission,validateSignal,submissionContract,validateSnapshot} from './v18-core.mjs';
const raw=readFileSync(new URL('./fixture-private.txt',import.meta.url),'utf8');
const snapshot=validateSnapshot(raw);
function model(){
 const a=structuredClone(submissionContract().skeleton);
 a.reportMarkdown='V1.8 synthetic regression only: no trading signal';
 for(const d of Object.values(a.directions)){
  d.marketEvidence='Source fixture only, insufficient independently verified trading structure';d.counterEvidence='Source fixture opposing evidence present';
  for(const z of Object.values(d.alternatives)){z.judgement='HOLD';z.evidence='Unverified actual structural entry stop and TP2 for synthetic fixture';}
 }
 a.coupling={status:'UNVERIFIED',evidence:'Fixture lacks verified matched-period movement'};
 for(const z of Object.values(a.checks)){z.status='UNVERIFIED';z.evidence='Fixture-only review';}
 for(const z of Object.values(a.clauses)){z.status='UNVERIFIED';z.evidence='Fixture-only review';}
 a.audit20={rules:'미검증',bias:'미검증',evidence:'No retroactive signal claim'};
 a.conclusions={BTC:{decision:'HOLD',evidence:'Unverified fixture'},ETH:{decision:'HOLD',evidence:'Unverified fixture'}};
 for(const sym of ['BTC','ETH'])a.dynamicManagement[sym]={asOf:sym==='BTC'?snapshot.btcTime:snapshot.ethTime,action:'HOLD',marketEvidence:'Fixture price and structure',counterEvidence:'Fixture competing directions',candidateComparisonEvidence:'Immediate and near and far alternatives reviewed but no verified structure',combinedRiskStatus:'UNVERIFIED',combinedRiskEvidence:'No account quantity',signalReviews:[]};
 return a;
}
const first=prepare({raw,ledger:[]});
const env=a=>({raw,session:'OFFICIAL_V17',policySha256:first.policy.sha256,snapshotSha256:first.snapshot.sha256,beforeLedgerSha256:first.beforeLedgerSha256,decisionKey:first.decisionKey,analysis:a});
test('exact full V1.8 user policy with sections 24-26',()=>{
 assert.equal(POLICY_SHA256,'083eac6ac099f57ddad80896a7744c8e6a95daba58504266118e248f8b32b2b7');assert.equal(OFFICIAL_POLICY.name,'MASTER PROMPT V1.8 FINAL');
 assert.equal(POLICY_BYTES,74490);assert.equal(POLICY_LINES,1101);assert.equal(CLAUSES.length,30);
 assert(POLICY_TEXT.includes('# 24.'));assert(POLICY_TEXT.includes('# 25.'));assert(POLICY_TEXT.includes('# 26.'));assert(VERSION.startsWith('V18'));
});
test('existing OFFICIAL_V17 ledger session survives V1.8 update',()=>{
 assert.equal(first.session,'OFFICIAL_V17');assert.equal(first.ledger.length,0);assert.equal(first.policy.sha256,POLICY_SHA256);
 assert.equal(first.requirements.requiredClauses.length,30);
});
test('complete V1.8 independent review can hold without fabricating signals',()=>assert.equal(validateOfficialSubmission(env(model()),[]).analysis.conclusions.BTC.decision,'HOLD'));
test('dynamic management is required by server validator',()=>{
 const a=model();delete a.dynamicManagement;assert.throws(()=>validateOfficialSubmission(env(a),[]),/V18_DYNAMIC_MANAGEMENT_REQUIRED/);
});
test('stale snapshot management is rejected',()=>{
 const a=model();a.dynamicManagement.ETH.asOf='2026-11-16 01:00:00';assert.throws(()=>validateOfficialSubmission(env(a),[]),/V18_REVIEW_MUST_USE_CURRENT_DATA_ETH/);
});
test('V1.7 as a *new* signal is not allowed under V1.8',()=>{
 const p=snapshot.btcPrice;
 const s={id:'BTC-SHORT-TEST',issuedAt:snapshot.btcTime,dataAt:snapshot.btcTime,symbol:'BTCUSDT',direction:'SHORT',tradeType:'지지·저항 반응형',exchange:'Bybit',product:'linear perpetual',invalidation:'before-touch invalidation',supportingEvidence:'real resistance',opposingEvidence:'real buying',issuedVersion:'V1.7',comparisonEvidence:'A-E compared',couplingEvidence:'relative BTC ETH price',entryLow:p+100,entryHigh:p+200,stop:p+300,tp1:p-100,tp2:p-400,tp3:p-600,status:'PENDING'};
 assert.throws(()=>validateSignal(s,snapshot),/SOURCE_POLICY_VERSION_MISMATCH/);
});
