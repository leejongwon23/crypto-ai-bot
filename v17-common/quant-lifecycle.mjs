import {sha} from './quant-structure.mjs';
import {supportedLogic} from './quant-evolution.mjs';

export const STAGES=['SHADOW','CANARY','LIVE','QUARANTINED'];
export function modelExposure(model,symbol){
 const stage=model.rollout?.stage??'SHADOW';
 return {stage,useForFormalAnalysis:stage==='LIVE'||(stage==='CANARY'&&model.rollout?.symbols?.includes(symbol)),exchangeOrdersEnabled:false};
}
export function planModelTransition(model,evidence){
 const stage=model.rollout?.stage??'SHADOW',reasons=[];
 if(!supportedLogic(model)||!STAGES.includes(stage))throw Error('ROLLOUT_MODEL_INVALID');
 const fail=x=>reasons.push(x);
 if(!evidence||evidence.modelSha256!==sha({family:model.family,logic:model.logic}))fail('MODEL_PROOF_BINDING_MISMATCH');
 if(evidence?.ledgerInvariant!==true||evidence?.dataComplete!==true||evidence?.policyCompatible!==true)fail('OPERATIONAL_INVARIANT_NOT_VERIFIED');
 if(evidence?.performance?.eligible!==true)fail('INDEPENDENT_PERFORMANCE_NOT_VERIFIED');
 if(!['SHADOW','CANARY'].includes(stage))fail('NO_AUTOMATIC_TRANSITION_FROM_'+stage);
 if(evidence?.phase!==stage||!(evidence?.phaseStartedAt>=model.rollout?.since)||!(evidence?.phaseEndedAt>evidence?.phaseStartedAt))fail('PROSPECTIVE_PHASE_PROOF_MISSING');
 if(!evidence?.holdoutId||evidence?.holdoutId===model.validationHoldoutId||(model.rollout?.usedPhaseHoldouts??[]).includes(evidence?.holdoutId))fail('PHASE_HOLDOUT_REUSE');
 if(evidence?.rollbackRehearsed!==true)fail('ROLLBACK_REHEARSAL_MISSING');
 if(reasons.length)return {changed:false,model,reasons};
 const next=stage==='SHADOW'?'CANARY':'LIVE';
 return {changed:true,reasons:[],model:{...model,rollout:{stage:next,since:evidence.phaseEndedAt,symbols:next==='CANARY'?['BTC']:['BTC','ETH'],usedPhaseHoldouts:[...(model.rollout?.usedPhaseHoldouts??[]),evidence.holdoutId],proofSha256:sha(evidence)}}};
}

export function reviewResearchPath(forecast,bars){
 const p=forecast.plan,sign=forecast.direction==='LONG'?1:-1;
 if(!p||![p.entryLow,p.entryHigh,p.stop,p.tp1,p.tp2,p.tp3].every(Number.isFinite)||p.entryLow>p.entryHigh)throw Error('RESEARCH_PLAN_INVALID');
 const entry=sign===1?p.entryHigh:p.entryLow;
 if(!(sign*(entry-p.stop)>0&&sign*(p.tp1-entry)>0&&sign*(p.tp2-p.tp1)>0&&sign*(p.tp3-p.tp2)>0))throw Error('RESEARCH_PLAN_ORDER_INVALID');
 const first=Math.ceil(forecast.frozenAt/60000)*60000;
 const rows=bars.filter(b=>b.intervalMs===60000&&b.t>=first).sort((a,b)=>a.t-b.t);
 let entered=false,tp1=false,tp2=false,previous=first,entryAt=null;
 for(const b of rows){
  if(b.t!==previous)return {status:'UNVERIFIED',reason:'POST_FREEZE_PATH_GAP',entered,entryAt};
  previous=b.end;
  const touchesEntry=b.low<=p.entryHigh&&b.high>=p.entryLow;
  const stop=sign===1?b.low<=p.stop:b.high>=p.stop;
  const t1=sign===1?b.high>=p.tp1:b.low<=p.tp1,t2=sign===1?b.high>=p.tp2:b.low<=p.tp2,t3=sign===1?b.high>=p.tp3:b.low<=p.tp3;
  if(!entered){
   // A price already beyond the stop before a pending first touch invalidates
   // the research setup. Never backfill an entry before the freeze timestamp.
   if(!touchesEntry&&stop)return {status:'NOT_ENTERED_INVALIDATED',reason:'STOP_BEFORE_FIRST_TOUCH',entered:false};
   if(!touchesEntry)continue;
   if(stop||t1)return {status:'UNVERIFIED',reason:'ENTRY_EXIT_ORDER_WITHIN_SAME_MINUTE',entered:false};
   entered=true;entryAt=b.t;
  }
  if(stop&&(t1&&!tp1||t2&&!tp2||t3))return {status:'UNVERIFIED',reason:'STOP_TARGET_ORDER_WITHIN_SAME_MINUTE',entered,entryAt};
  if(stop){const fill=sign===1?Math.min(p.stop,b.open):Math.max(p.stop,b.open);return {status:'STOPPED',entered,entryAt,labelEnd:b.end,exitPrice:fill,tp1,tp2,grossR:sign*(fill-entry)/Math.abs(entry-p.stop),costsVerified:false,netR:null,accountFill:false};}
  tp1=tp1||t1;tp2=tp2||t2;
  if(t3)return {status:'TP3_COMPLETE',entered,entryAt,labelEnd:b.end,exitPrice:p.tp3,tp1,tp2,grossR:sign*(p.tp3-entry)/Math.abs(entry-p.stop),costsVerified:false,netR:null,accountFill:false};
 }
 return {status:entered?'OPEN':'WAITING_FIRST_TOUCH',entered,entryAt,tp1,tp2,costsVerified:false,netR:null,accountFill:false};
}

export function freezeResearchForecasts(context,{frozenAt,sourceReceivedAt=frozenAt}={}){
 if(!Number.isFinite(frozenAt))throw Error('RESEARCH_FREEZE_CLOCK_REQUIRED');
 const out=[];
 for(const symbol of ['BTC','ETH']){
  const s=context.symbols[symbol],events=s.structure.structuralEvents??{};
  for(const direction of ['LONG','SHORT']){
   const up=direction==='LONG',supported=up?events.higherHighs&&events.higherLows&&!events.invalidatedUp:events.lowerHighs&&events.lowerLows&&!events.invalidatedDown;
   const logic={all:up?['confirmedHigherHighs','confirmedHigherLows']:['confirmedLowerHighs','confirmedLowerLows'],invalidates:[up?'breakLastConfirmedLow':'breakLastConfirmedHigh']};
   const mechanism={family:'ORDERED_STRUCTURE_CONTINUATION',logic},plan=s.directions?.[direction]?.A;
   const row={symbol,direction,regime:s.structure.regime??'UNVERIFIED',mechanism,modelSha256:sha(mechanism),sourceSha256:context.sourceSha256,sourceAsOf:s.asOf,frozenAt,sourceReceivedAt,mode:'PROSPECTIVE_RESEARCH_ONLY',officialSignal:false,exchangeOrdersEnabled:false};
   const sourceTime=Date.parse(String(s.asOf).replace(' ','T')+'+09:00');
   const reason=!Number.isFinite(sourceTime)||sourceTime>frozenAt||sourceTime+60000<frozenAt?'SOURCE_NOT_CURRENT_AT_ACTUAL_FREEZE':!supported?'MATHEMATICAL_STRUCTURE_NOT_SUPPORTED':plan?.kind!=='ARITHMETIC_CANDIDATE_NOT_FORMAL'?'STRUCTURAL_PLAN_UNAVAILABLE':plan.rr2<2?'TP2_RR_BELOW_EXISTING_POLICY':null;
   out.push({...row,id:sha({symbol,direction,sourceSha256:context.sourceSha256,modelSha256:row.modelSha256}),status:reason?'ABSTAIN':'FROZEN_PENDING',reason,plan:reason?null:plan});
  }
 }
 return out;
}
