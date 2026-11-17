import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {evaluatePromotion} from '../v17-common/quant-evolution.mjs';
import {sha} from '../v17-common/quant-structure.mjs';
import {planModelTransition} from '../v17-common/quant-lifecycle.mjs';
import {verifyStudyFiles} from './study-proof.mjs';
const out='quant-evolution/output';await mkdir(out,{recursive:true});
const git=(...args)=>execFileSync('git',args,{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
let actualCi=null;try{actualCi=JSON.parse(await readFile(out+'/ci-proof-manifest.json','utf8'));if(actualCi.workflowRun!==process.env.GITHUB_RUN_ID||actualCi.sourceCommit!==git('rev-parse','HEAD'))actualCi=null;}catch{}
let registry=JSON.parse(await readFile('v17-common/quant-model-registry.json','utf8')),journal=[];
try{journal=JSON.parse(await readFile('quant-evolution/holdout-consumption.json','utf8'));}catch{}
let files=[];try{await readFile('quant-evolution/frozen-study.json');files.push('quant-evolution/frozen-study.json');}catch{}
try{files.push(...(await readdir('quant-evolution/phase-studies')).filter(x=>x.endsWith('.json')).sort().map(x=>'quant-evolution/phase-studies/'+x));}catch{}
const reports=[];
for(const file of files){
 const input=JSON.parse(await readFile(file,'utf8')),consumed=[...new Set([...registry.usedHoldouts,...journal.map(x=>x.holdoutId)])],id=input.protocol?.spec?.holdoutId;
 if(input.protocol?.status==='LOCKED'&&typeof id==='string'&&!consumed.includes(id)&&input.cases?.length){
  journal.push({holdoutId:id,protocolSha256:input.protocol.sha256,studySha256:sha(input),openedAt:new Date().toISOString()});await writeFile('quant-evolution/holdout-consumption.json',JSON.stringify(journal,null,2)+'\n');
  // Persist spent evidence before looking at outcomes. A failed job must not
  // silently make the same holdout fresh for the next automatic attempt.
  if(process.env.GITHUB_ACTIONS==='true'){
   const remote=git('ls-remote','origin','refs/heads/v17-canonical-engine').split(/\s+/)[0];if(remote!==git('rev-parse','HEAD'))throw Error('CONCURRENT_SOURCE_CHANGE_HOLDOUT_NOT_OPENED');
   git('add','quant-evolution/holdout-consumption.json');git('-c','user.name=btc-eth-mathematical-research','-c','user.email=btc-eth-research@users.noreply.github.com','commit','-m','Permanently consume frozen independent holdout before evaluation [skip render]');git('push','origin','HEAD:v17-canonical-engine');
  }
 }
 if(actualCi)input.proofs={...input.proofs,unit:actualCi.unit,integration:actualCi.integration};
 const proof=await verifyStudyFiles(input);const result=evaluatePromotion({...input,cases:proof.cases,usedHoldouts:consumed});result.reasons.push(...proof.reasons);result.eligible=result.eligible&&proof.eligible;
 if(!actualCi){result.eligible=false;result.reasons.push('ACTUAL_CURRENT_CI_PROOFS_REQUIRED');}
 try{
  const committed=Number(execFileSync('git',['log','-1','--format=%ct','--',input.protocolFile?.path??'quant-evolution/locked-protocol.json'],{encoding:'utf8'}).trim())*1000;
  const committedBytes=execFileSync('git',['show','HEAD:'+input.protocolFile.path],{encoding:'utf8'});if(sha(committedBytes)!==input.protocolFile.sha256)throw Error('UNCOMMITTED_PROTOCOL_CHANGE');
  if(!(committed>0&&committed<input.protocol.spec.holdoutStart)){result.eligible=false;result.reasons.push('PROTOCOL_NOT_COMMITTED_BEFORE_HOLDOUT');}
 }catch{result.eligible=false;result.reasons.push('PROTOCOL_COMMIT_ATTESTATION_UNAVAILABLE');}
 if(result.eligible){
  if(input.candidate.sourceRegistrySha256!==sha(registry))throw Error('MODEL_REGISTRY_CHANGED_REBASE_RESEARCH');
  let model;const old=registry.models.find(m=>m.id===input.candidate.id);
  if(input.phase){
   if(!old)throw Error('PHASE_MODEL_NOT_REGISTERED');
   if(sha({family:old.family,logic:old.logic})!==sha({family:input.candidate.family,logic:input.candidate.logic}))throw Error('PHASE_MECHANISM_CHANGED_WITHOUT_NEW_MODEL');
   const next=planModelTransition(old,{...input.phase,performance:result,modelSha256:sha({family:old.family,logic:old.logic}),holdoutId:id});
   if(!next.changed){result.eligible=false;result.reasons.push(...next.reasons);}else model=next.model;
  }else if(old){result.eligible=false;result.reasons.push('EXISTING_MODEL_REQUIRES_NEW_PROSPECTIVE_PHASE');}
  else model={id:input.candidate.id,family:input.candidate.family,logic:input.candidate.logic,version:registry.version+1,validationProtocolSha256:input.protocol.sha256,validationHoldoutId:id,rollout:{stage:'SHADOW',since:Date.now(),symbols:[],usedPhaseHoldouts:[]}};
  if(model){registry={...registry,version:registry.version+1,status:'EVIDENCE_GATED_STAGED_MODELS',models:[...registry.models.filter(x=>x.id!==model.id),model],usedHoldouts:[...registry.usedHoldouts,id],ordersEnabled:false};await writeFile('v17-common/quant-model-registry.json',JSON.stringify(registry,null,2)+'\n');}
 }
 reports.push({file,...result});
}
if(!reports.length)reports.push({eligible:false,reasons:['FROZEN_INDEPENDENT_STUDY_NOT_AVAILABLE'],ordersEnabled:false});
await writeFile(out+'/promotion.json',JSON.stringify({reports,registryVersion:registry.version,ordersEnabled:false},null,2)+'\n');for(const r of reports)console.log(JSON.stringify(r));
