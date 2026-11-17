import {sha} from './quant-structure.mjs';
export const MODEL_FAMILIES=['ORDERED_STRUCTURE_CONTINUATION','BOUNDARY_REJECTION','FLOW_PRICE_ABSORPTION','STRUCTURAL_TRANSITION'];
const mechanisms={
 ORDERED_STRUCTURE_CONTINUATION:{all:['confirmedHigherHighs','confirmedHigherLows'],invalidates:['breakLastConfirmedLow']},
 BOUNDARY_REJECTION:{all:['observedFrozenBoundaryExcursion','observedReturnInsideBoundary'],invalidates:['acceptanceBeyondBoundary']},
 FLOW_PRICE_ABSORPTION:{all:['aggressorFlowPriceDisagreement','observedPassiveResponse'],invalidates:['priceFollowsAggressorInstead']},
 STRUCTURAL_TRANSITION:{all:['priorStructureInvalidated','newOpposingStructureObserved'],invalidates:['priorStructureRestored']}
};
const ATOMS=new Set(['confirmedHigherHighs','confirmedHigherLows','confirmedLowerHighs','confirmedLowerLows','breakLastConfirmedLow','breakLastConfirmedHigh','observedFrozenBoundaryExcursion','observedReturnInsideBoundary','acceptanceBeyondBoundary','aggressorFlowPriceDisagreement','observedPassiveResponse','priceFollowsAggressorInstead','priorStructureInvalidated','newOpposingStructureObserved','priorStructureRestored']);
export function supportedLogic(model){return MODEL_FAMILIES.includes(model?.family)&&model.logic&&Array.isArray(model.logic.all)&&model.logic.all.length>=2&&Array.isArray(model.logic.invalidates)&&model.logic.invalidates.length>=1&&[...model.logic.all,...model.logic.invalidates].every(x=>ATOMS.has(x))&&new Set([...model.logic.all,...model.logic.invalidates]).size===model.logic.all.length+model.logic.invalidates.length;}
export function synthesizeMechanismRevisions(model,counterexamples){
 if(!supportedLogic(model))throw Error('UNSUPPORTED_PARENT_MODEL');
 const existing=new Set([...model.logic.all,...model.logic.invalidates]),out=[];
 for(const atom of ATOMS){
  if(existing.has(atom)||!counterexamples.some(c=>c.facts?.[atom]===true&&c.frozenSourceSha256&&c.errorObservedAt>=c.forecastAt))continue;
  const child={family:model.family,logic:{all:[...model.logic.all],invalidates:[...model.logic.invalidates,atom]}};
  out.push({id:sha(child).slice(0,20),parentModelId:model.id,mechanism:child,mechanismSha256:sha(child),change:{kind:'ADD_OBSERVED_STRUCTURAL_INVALIDATOR',atom},
   sourceCounterexamples:counterexamples.filter(c=>c.facts?.[atom]===true).map(c=>({frozenSourceSha256:c.frozenSourceSha256,forecastAt:c.forecastAt,errorObservedAt:c.errorObservedAt})),
   status:'RESEARCH_ONLY',deployEligible:false,requiredTest:'Matched independent good/bad cases and opposite-direction controls; never promote from a single error.'});
 }
 return out;
}
export function generateResearchProposals(context,prior=[]){
 const seen=new Set(prior.map(x=>x.mechanismSha256+x.symbol));const result=[];
 for(const symbol of ['BTC','ETH']){const s=context.symbols[symbol];const families=[];
  if(s.structure.pivots?.length)families.push('ORDERED_STRUCTURE_CONTINUATION','BOUNDARY_REJECTION','STRUCTURAL_TRANSITION');
  if(s.flow.hypotheses?.length)families.push('FLOW_PRICE_ABSORPTION');
  for(const family of families){const mechanism={family,logic:mechanisms[family]},id=sha(mechanism);if(seen.has(id+symbol))continue;seen.add(id+symbol);
   result.push({id:symbol+'-'+id.slice(0,16),symbol,mechanism,mechanismSha256:id,status:'RESEARCH_ONLY',sourceSha256:context.sourceSha256,createdAsOf:s.asOf,
    improvementKind:'COMPETING_MARKET_MECHANISM_NOT_PARAMETER_OR_WEIGHT_CHANGE',
    requiredTest:'Frozen chronological matched cases, real costs, dependence-aware confidence interval, independent holdout and opposite-side controls',
    unknowns:s.dataStatus,trigger:'LATEST_OBSERVATION_NOT_ONLY_STOP',deployEligible:false});
  }
  const errors=s.modelErrorReview.filter(x=>x.intervalBreached===true&&x.frozenSourceSha256&&x.forecastAt<s.asOf);
  for(const model of context.activeModels?.[symbol]??[]){
   const matching=errors.filter(x=>x.modelId===model.id);if(!model.logic||!matching.length)continue;
   const facts={confirmedHigherHighs:s.structure.structuralEvents?.higherHighs,confirmedHigherLows:s.structure.structuralEvents?.higherLows,confirmedLowerHighs:s.structure.structuralEvents?.lowerHighs,confirmedLowerLows:s.structure.structuralEvents?.lowerLows,breakLastConfirmedLow:s.structure.structuralEvents?.invalidatedUp,breakLastConfirmedHigh:s.structure.structuralEvents?.invalidatedDown,aggressorFlowPriceDisagreement:!!s.flow.hypotheses?.length};
   result.push(...synthesizeMechanismRevisions(model,matching.map(x=>({facts,frozenSourceSha256:x.frozenSourceSha256,forecastAt:Date.parse(x.forecastAt.replace(' ','T')+'+09:00'),errorObservedAt:Date.parse(s.asOf.replace(' ','T')+'+09:00')}))).map(x=>({...x,symbol})));
  }
 }return result;
}
export function evaluateSymbolicModel(model,facts){
 if(!MODEL_FAMILIES.includes(model.family))throw Error('UNKNOWN_MATHEMATICAL_MODEL_FAMILY');
 if(!supportedLogic(model))throw Error('MODEL_MECHANISM_NOT_SUPPORTED_BY_RUNTIME');const expected=model.logic;
 if(expected.invalidates.some(k=>facts[k]===true))return {status:'FALSIFIED'};
 const missing=[...expected.all,...expected.invalidates].filter(k=>typeof facts[k]!=='boolean');
 if(missing.length)return {status:'UNIDENTIFIED',missing};
 if(expected.invalidates.some(k=>facts[k]))return {status:'FALSIFIED'};
 return {status:expected.all.every(k=>facts[k])?'SUPPORTED_CONDITIONAL_HYPOTHESIS':'NOT_SUPPORTED',tradeApproval:false};
}
export function blockMeanInterval(values,blockLength,alpha,repetitions=1000){
 if(values.length<2||!values.every(Number.isFinite)||!Number.isInteger(blockLength)||blockLength<1||blockLength>values.length||!(alpha>0&&alpha<1)||!Number.isInteger(repetitions)||repetitions<100)throw Error('DEPENDENCE_AWARE_PROTOCOL_REQUIRED');
 // Reproducible resampling, not a trade score or a tunable directional weight.
 let state=0x9e3779b9;const rand=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
 const means=[];for(let k=0;k<repetitions;k++){const sample=[];while(sample.length<values.length){const start=Math.floor(rand()*(values.length-blockLength+1));for(let j=0;j<blockLength&&sample.length<values.length;j++)sample.push(values[start+j]);}means.push(sample.reduce((a,b)=>a+b,0)/sample.length);}
 means.sort((a,b)=>a-b);return {lower:means[Math.floor(alpha/2*(repetitions-1))],upper:means[Math.ceil((1-alpha/2)*(repetitions-1))],n:values.length,blockLength,alpha,method:'MOVING_BLOCK_BOOTSTRAP'};
}
export function evaluatePromotion({protocol,cases,candidate,usedHoldouts=[]}){
 const reasons=[];const check=(ok,why)=>{if(!ok)reasons.push(why);};
 check(protocol?.status==='LOCKED','PROTOCOL_NOT_LOCKED');check(protocol?.sha256===sha(protocol?.spec??{}),'PROTOCOL_HASH_MISMATCH');
 const p=protocol?.spec??{};check(Number.isFinite(p.lockedAt)&&Number.isFinite(p.holdoutStart)&&Number.isFinite(p.holdoutEnd)&&p.lockedAt<p.holdoutStart&&p.holdoutStart<p.holdoutEnd,'PRELOCKED_UNTOUCHED_HOLDOUT_REQUIRED');
 check(Number.isFinite(p.alpha)&&p.alpha>0&&p.alpha<1&&Number.isInteger(p.blockLength)&&p.blockLength>0&&Number.isInteger(p.testFamilySize)&&p.testFamilySize>0,'STATISTICAL_PROTOCOL_MISSING');
 check(Number.isInteger(p.minCasesPerStratum)&&p.minCasesPerStratum>=2&&Number.isInteger(p.minimumIndependentBlocks)&&p.minimumIndependentBlocks>=2&&typeof p.sampleAdequacyJustification==='string'&&p.sampleAdequacyJustification.trim().length>0,'PRELOCKED_SAMPLE_ADEQUACY_REQUIRED');
 check(Number.isFinite(p.maximumDrawdown)&&p.maximumDrawdown>=0,'PRELOCKED_LOSS_LIMIT_REQUIRED');
 check(!usedHoldouts.includes(p.holdoutId)&&typeof p.holdoutId==='string','HOLDOUT_ALREADY_USED_OR_MISSING');
 check(supportedLogic(candidate),'UNSUPPORTED_SYMBOLIC_MODEL');
 check(Array.isArray(cases)&&cases.length>0,'NO_REAL_CASES');
 const holdout=[],seen=new Set();
 for(const c of cases??[]){
  const key=String(c.symbol)+':'+String(c.decisionAt)+':'+String(c.rawSourceSha256);check(!seen.has(key),'DUPLICATE_RESEARCH_CASE');seen.add(key);
  check(c.synthetic===false&&/^[a-f0-9]{64}$/.test(c.rawSourceSha256??''),'CASE_PROVENANCE_MISSING');
  check([c.featureStart,c.featureEnd,c.decisionAt,c.labelEnd,c.trainLabelEnd].every(Number.isFinite)&&c.featureStart<=c.featureEnd&&c.featureEnd<=c.decisionAt&&c.decisionAt<=c.labelEnd&&Number.isFinite(p.embargoMs)&&c.trainLabelEnd+p.embargoMs<c.featureStart,'TIME_LEAKAGE_OR_PURGE_FAILURE');
  check(c.costsVerified===true&&c.pathOrderVerified===true&&c.positionOverlapVerified===true,'COST_PATH_OR_OVERLAP_UNVERIFIED');
  check([c.candidateNet,c.baselineNet,c.oppositeNet,c.abstainNet].every(Number.isFinite),'PAIRED_NET_OUTCOMES_MISSING');
  if(c.decisionAt>=p.holdoutStart&&c.labelEnd<p.holdoutEnd)holdout.push(c);
 }
 check(holdout.length>1,'INSUFFICIENT_INDEPENDENT_HOLDOUT');
 for(const symbol of ['BTC','ETH']){const ordered=holdout.filter(c=>c.symbol===symbol).sort((a,b)=>a.featureStart-b.featureStart);for(let i=1;i<ordered.length;i++)check(ordered[i-1].labelEnd+(p.embargoMs??Infinity)<ordered[i].featureStart,'OVERLAPPING_HOLDOUT_INFORMATION_SPANS');}
 const strata=[];for(const symbol of ['BTC','ETH'])for(const regime of p.requiredRegimes??[]){const rows=holdout.filter(c=>c.symbol===symbol&&c.regime===regime);
  rows.sort((a,b)=>a.decisionAt-b.decisionAt);
  if(rows.length<2||rows.length<(p.minCasesPerStratum??Infinity)||Math.floor(rows.length/(p.blockLength??Infinity))<(p.minimumIndependentBlocks??Infinity)){reasons.push('UNVERIFIED_STRATUM_'+symbol+'_'+regime);continue;}
  if(!(p.alpha>0&&p.alpha<1&&Number.isInteger(p.testFamilySize)&&p.testFamilySize>0))continue;
  for(const control of ['baselineNet','oppositeNet','abstainNet']){const differences=rows.map(c=>c.candidateNet-c[control]);if(!differences.every(Number.isFinite)){reasons.push('INVALID_PAIRED_COST_OUTCOME');continue;}const alpha=p.alpha/p.testFamilySize,repetitions=Math.max(1000,Math.ceil(20/alpha));if(repetitions>200000){reasons.push('MONTE_CARLO_TAIL_RESOLUTION_INSUFFICIENT');continue;}const ci=blockMeanInterval(differences,p.blockLength,alpha,repetitions);strata.push({symbol,regime,control,...ci});check(ci.lower>0,'NET_IMPROVEMENT_NOT_ESTABLISHED_'+symbol+'_'+regime+'_'+control);}
 }
 check(Array.isArray(p.requiredRegimes)&&p.requiredRegimes.length>0,'REGIME_COVERAGE_NOT_LOCKED');
 check(Number.isInteger(p.registeredModels)&&p.registeredModels>0&&p.testFamilySize>=p.registeredModels*(p.requiredRegimes?.length??0)*2*3,'MULTIPLE_SEARCH_FAMILY_UNDERCOUNTED');
 check(Number.isFinite(candidate?.drawdown)&&Number.isFinite(candidate?.baselineDrawdown)&&candidate.drawdown<=candidate.baselineDrawdown,'DRAWDOWN_NOT_NONINFERIOR');
 check(Number.isFinite(candidate?.drawdown)&&candidate.drawdown<=p.maximumDrawdown,'LOCKED_LOSS_LIMIT_EXCEEDED_OR_UNVERIFIED');
 check(candidate?.unitTests===true&&candidate?.integrationTests===true&&candidate?.ledgerInvariant===true&&candidate?.shadowVerified===true&&candidate?.rollbackRehearsed===true,'ENGINE_SHADOW_ROLLBACK_PROOF_MISSING');
 return {eligible:reasons.length===0,reasons,strata,holdoutId:p.holdoutId??null,ordersEnabled:false,rule:'NO_WINRATE_ONLY_PROMOTION_NO_WEIGHTS_NO_SCORES'};
}
