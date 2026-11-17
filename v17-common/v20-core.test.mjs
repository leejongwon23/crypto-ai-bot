import test from 'node:test';import assert from 'node:assert/strict';
import {OFFICIAL_POLICY,CLAUSES,prepare,validateOfficialSubmission,validateSignal,quantEngineStatus} from './v20-core.mjs';
const raw=`========== BTC (BTCUSDT) ==========\ntimestamp_kst: 2026-11-17 09:10:00\nlastPrice: 100\n========== ETH (ETHUSDT) ==========\ntimestamp_kst: 2026-11-17 09:10:00\nlastPrice: 200\n`;
function fixture(){const p=prepare({raw}),ev='SYNTHETIC TEST: raw interval data absent; inference explicitly unverified',a=p.requirements.submissionContract.skeleton;
 a.reportMarkdown='SYNTHETIC TEST UNVERIFIED';for(const d of Object.values(a.directions)){d.marketEvidence=ev;d.counterEvidence=ev;for(const x of Object.values(d.alternatives)){x.judgement='HOLD';x.evidence=ev;}}
 a.coupling={status:'UNVERIFIED',evidence:ev};for(const xs of [a.checks,a.clauses])for(const k in xs)xs[k]={status:'UNVERIFIED',evidence:ev};a.audit20={rules:ev,bias:ev,evidence:ev};
 a.quantAudit={contextSha256:p.quantContext.sha256,researchProposals:[]};
 for(const sym of ['BTC','ETH']){const at=sym==='BTC'?p.snapshot.btcTime:p.snapshot.ethTime;
  a.conclusions[sym]={decision:'HOLD',evidence:ev};a.marketOutlook[sym]={asOf:at,bias:'UNVERIFIED',currentSituation:ev,expectedPath:ev,alternative:ev};
  a.dynamicManagement[sym]={asOf:at,action:'HOLD',marketEvidence:ev,counterEvidence:ev,candidateComparisonEvidence:ev,combinedRiskStatus:'NOT_APPLICABLE',combinedRiskEvidence:ev,signalReviews:[]};
  a.regimeAudit[sym]={asOf:at,higherRegime:'UNVERIFIED',executionRegime:'UNVERIFIED',status:'UNVERIFIED',dataEvidence:ev,counterEvidence:ev,sourceLimitations:ev,directionReviews:{}};
  for(const dir of ['LONG','SHORT'])a.regimeAudit[sym].directionReviews[dir]={tactic:'NO_QUALIFIED_TACTIC',selection:'HOLD',entryEvidence:ev,stopEvidence:ev,targetEvidence:ev,exitRiskEvidence:ev};
  a.quantAudit[sym]={asOf:at,intentEvidenceLevel:'UNVERIFIED',participantIdentityKnown:false,directions:{}};
  for(const k of ['regimeMechanism','flowImpactMechanism','noiseRisk','invalidationRisk','executionCostRisk','preStopErrorReview','competingHypotheses','modelRevisionDecision','sourceLimitations'])a.quantAudit[sym][k]=ev;
  for(const dir of ['LONG','SHORT'])a.quantAudit[sym].directions[dir]={mathematicalBasis:ev,counterexample:ev,entryStopTargetBasis:ev,costAndCapacityLimitations:ev};
 }
 return {raw,session:'OFFICIAL_V17',policySha256:p.policy.sha256,snapshotSha256:p.snapshot.sha256,beforeLedgerSha256:p.beforeLedgerSha256,decisionKey:p.decisionKey,analysis:a};
}
test('actual active policy is V2.0 with complete original safeguards and mathematical 0..43 audit',()=>{assert.equal(OFFICIAL_POLICY.name,'MASTER PROMPT V2.0 FINAL');assert.equal(CLAUSES.length,47);const p=prepare({raw});for(const n of [0,7,8,14,20,27,32,36,37,38,39,43])assert.match(p.policy.text,new RegExp('# '+n+'\\.'));assert.ok(p.policy.bytes>96602);assert.equal(quantEngineStatus().weightedScores,false);});
test('preparation actually computes mathematical context and proposals without issuing trades',()=>{const p=prepare({raw});assert.equal(p.session,'OFFICIAL_V17');assert.ok(p.quantContext.sha256);assert.deepEqual(p.researchProposals,[]);assert.equal(p.status,'NEEDS_AI_ANALYSIS');});
test('full unverified no-data analysis can be stored without fabricating candles or trades',()=>{const v=validateOfficialSubmission(fixture(),[]);assert.deepEqual(v.newSignals,[]);assert.deepEqual(v.events,[]);assert.equal(v.quantContext.symbols.BTC.dataStatus.closedBars,0);});
test('mathematical audit cannot be omitted or use another source hash',()=>{const x=fixture();delete x.analysis.quantAudit;assert.throws(()=>validateOfficialSubmission(x,[]),/QUANT_CONTEXT/);const y=fixture();y.analysis.quantAudit.contextSha256='invented';assert.throws(()=>validateOfficialSubmission(y,[]),/QUANT_CONTEXT/);});
test('invisible participant identity and missing opposing-direction evidence reject',()=>{const x=fixture();x.analysis.quantAudit.ETH.participantIdentityKnown=true;assert.throws(()=>validateOfficialSubmission(x,[]),/PARTICIPANT/);const y=fixture();delete y.analysis.quantAudit.BTC.directions.SHORT;assert.throws(()=>validateOfficialSubmission(y,[]),/BOTH_DIRECTIONS/);});
