import {readFile} from 'node:fs/promises';
import {resolve,relative} from 'node:path';
import {sha,validateBars} from '../v17-common/quant-structure.mjs';
import {reviewResearchPath} from '../v17-common/quant-lifecycle.mjs';

export function netResearchReturn(forecast,outcome,costs){
 if(!['STOPPED','TP3_COMPLETE'].includes(outcome.status)||!Number.isFinite(outcome.exitPrice))throw Error('ORDERED_TERMINAL_PATH_REQUIRED');
 if(costs?.sourceStatus!=='DOCUMENTED_ASSUMPTIONS'||![costs.entryFeeRate,costs.exitFeeRate,costs.adverseEntrySlippageBps,costs.adverseExitSlippageBps,costs.notional,costs.fundingCoverageFrom,costs.fundingCoverageThrough].every(Number.isFinite)||costs.entryFeeRate<0||costs.exitFeeRate<0||costs.adverseEntrySlippageBps<0||costs.adverseExitSlippageBps<0||!(costs.notional>0)||!Array.isArray(costs.funding))throw Error('COMPLETE_DOCUMENTED_COST_MODEL_REQUIRED');
 const sign=forecast.direction==='LONG'?1:-1,entry=(sign===1?forecast.plan.entryHigh:forecast.plan.entryLow)*(1+sign*costs.adverseEntrySlippageBps/10000),exit=outcome.exitPrice*(1-sign*costs.adverseExitSlippageBps/10000);
 if(!(entry>0&&exit>0)||costs.fundingCoverageFrom>outcome.entryAt||costs.fundingCoverageThrough<outcome.labelEnd||costs.fundingCoverageComplete!==true)throw Error('FUNDING_COVERAGE_UNVERIFIED');
 let funding=0;for(const f of costs.funding){if(![f.at,f.rate,f.markPrice].every(Number.isFinite)||!(f.markPrice>0))throw Error('FUNDING_EVENT_INVALID');if(f.at>=outcome.entryAt&&f.at<outcome.labelEnd)funding+=sign*f.rate*f.markPrice/entry;}
 return sign*(exit-entry)/entry-costs.entryFeeRate-costs.exitFeeRate*exit/entry-funding;
}
export async function verifyStudyFiles(input,{read=readFile,cwd=process.cwd()}={}){
 const reasons=[],check=(ok,s)=>{if(!ok)reasons.push(s);};
 const loadPinned=async ref=>{
  if(!ref||typeof ref.path!=='string'||!ref.path.startsWith('quant-evolution/'))throw Error('PROOF_PATH_OUTSIDE_RESEARCH_DIRECTORY');
  const p=resolve(cwd,ref.path);if(relative(resolve(cwd,'quant-evolution'),p).startsWith('..'))throw Error('PROOF_PATH_ESCAPE');
  const bytes=await read(p,'utf8');if(sha(bytes)!==ref.sha256)throw Error('PINNED_PROOF_SHA_MISMATCH');return bytes;
 };
 try{const protocol=JSON.parse(await loadPinned(input.protocolFile));check(sha(protocol)===sha(input.protocol),'SEPARATE_LOCKED_PROTOCOL_MISMATCH');}catch(e){reasons.push('LOCKED_PROTOCOL_FILE_'+e.message);}
 const replayed=[];
 for(const c of input.cases??[]){
  try{
   const raw=await loadPinned({path:c.rawSourcePath,sha256:c.rawSourceSha256});
   const path=JSON.parse(await loadPinned(c.pathProof));check(['BYBIT_LINEAR_PUBLIC_1M','USER_SUBMITTED_BYBIT_1M'].includes(path.source),'MARKET_PATH_PROVENANCE_MISSING');
   const bars=validateBars(path.bars,c.labelEnd).bars;
   const costs=JSON.parse(await loadPinned(c.costProof));check(sha(costs)===input.protocol?.spec?.costModelSha256,'COST_ASSUMPTIONS_NOT_PRELOCKED');
   const nets={};
   for(const [kind,field]of [['candidate','candidateNet'],['baseline','baselineNet'],['opposite','oppositeNet']]){
    const forecast=c.frozenForecasts?.[kind];if(!forecast||forecast.frozenAt!==c.decisionAt||forecast.sourceSha256!==sha(raw))throw Error('FORECAST_NOT_BOUND_TO_FROZEN_SOURCE_AND_TIME');
    if(kind==='baseline'&&forecast.policyVersion!=='V1.9')throw Error('V19_FROZEN_BASELINE_REQUIRED');
    if(kind==='candidate'&&forecast.modelSha256!==sha({family:input.candidate?.family,logic:input.candidate?.logic}))throw Error('CANDIDATE_MODEL_PROOF_MISMATCH');
    const outcome=reviewResearchPath(forecast,bars);nets[field]=netResearchReturn(forecast,outcome,costs);
    if(!Number.isFinite(c[field])||Math.abs(c[field]-nets[field])>1e-12)throw Error('CLAIMED_NET_DIFFERS_FROM_REAL_PATH_COST_REPLAY');
   }
   check(c.abstainNet===0,'ABSTENTION_CANNOT_HAVE_INVENTED_PROFIT');
   replayed.push({...c,...nets,costsVerified:true,pathOrderVerified:true});
  }catch(e){reasons.push('CASE_REPLAY_'+e.message);}
 }
 for(const key of ['unit','integration']){
  try{const text=await loadPinned(input.proofs?.[key]);const tests=Number(text.match(/^# tests (\d+)\s*$/m)?.[1]),failed=Number(text.match(/^# fail (\d+)\s*$/m)?.[1]),skipped=Number(text.match(/^# skipped (\d+)\s*$/m)?.[1]);check(tests>0&&failed===0&&skipped===0,'ACTUAL_'+key.toUpperCase()+'_TAP_NOT_PASSING');}catch(e){reasons.push(key.toUpperCase()+'_PROOF_'+e.message);}
 }
 for(const key of ['shadow','rollback','ledger','drawdown']){
  try{
   const proof=JSON.parse(await loadPinned(input.proofs?.[key]));
   if(key==='shadow')check(proof.prospective===true&&proof.completedCases>0&&proof.modelSha256===sha({family:input.candidate?.family,logic:input.candidate?.logic}),'ACTUAL_PROSPECTIVE_SHADOW_PROOF_REQUIRED');
   if(key==='rollback')check(proof.drill===true&&proof.status==='ROLLED_BACK'&&proof.rollback?.verified===true&&proof.rollback?.integrity?.intact===true,'ACTUAL_LIVE_ROLLBACK_PROOF_REQUIRED');
   if(key==='ledger')check(proof.intact===true&&proof.originalsIntact===true&&proof.eventsIntact===true,'ACTUAL_LEDGER_PREFIX_PROOF_REQUIRED');
   if(key==='drawdown'){
    const curve=JSON.parse(await loadPinned(proof.source));let candidatePeak=-Infinity,baselinePeak=-Infinity,candidateDD=0,baselineDD=0,previous=-Infinity;
    for(const row of curve.equityPath??[]){if(![row.t,row.candidate,row.baseline].every(Number.isFinite)||row.t<=previous)throw Error('EQUITY_PATH_ORDER_OR_VALUES_INVALID');previous=row.t;candidatePeak=Math.max(candidatePeak,row.candidate);baselinePeak=Math.max(baselinePeak,row.baseline);candidateDD=Math.max(candidateDD,candidatePeak-row.candidate);baselineDD=Math.max(baselineDD,baselinePeak-row.baseline);}
    check(proof.pathOrderVerified===true&&curve.basis==='SOURCE_REPLAYED_MARK_TO_MARKET'&&curve.equityPath?.length>1&&candidateDD===input.candidate?.drawdown&&baselineDD===input.candidate?.baselineDrawdown,'MARK_TO_MARKET_DRAWDOWN_SOURCE_REQUIRED');
   }
  }catch(e){reasons.push(key.toUpperCase()+'_PROOF_'+e.message);}
 }
 return {eligible:reasons.length===0,reasons,cases:replayed,accountPerformance:false};
}
