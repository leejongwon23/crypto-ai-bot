import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {evaluatePromotion} from '../v17-common/quant-evolution.mjs';
import {sha} from '../v17-common/quant-structure.mjs';
const out='quant-evolution/output';await mkdir(out,{recursive:true});
let input;
try{input=JSON.parse(await readFile('quant-evolution/frozen-study.json','utf8'));}
catch{const report={eligible:false,reasons:['FROZEN_INDEPENDENT_STUDY_NOT_AVAILABLE'],ordersEnabled:false};await writeFile(out+'/promotion.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));process.exit(0);}
const registry=JSON.parse(await readFile('v17-common/quant-model-registry.json','utf8'));
let journal=[];try{journal=JSON.parse(await readFile('quant-evolution/holdout-consumption.json','utf8'));}catch{}
const consumed=[...new Set([...registry.usedHoldouts,...journal.map(x=>x.holdoutId)])];
const id=input.protocol?.spec?.holdoutId;
if(input.protocol?.status==='LOCKED'&&typeof id==='string'&&!consumed.includes(id)&&input.cases?.length){
 journal.push({holdoutId:id,protocolSha256:input.protocol.sha256,studySha256:sha(input),openedAt:new Date().toISOString()});
 await writeFile('quant-evolution/holdout-consumption.json',JSON.stringify(journal,null,2)+'\n');
}
const result=evaluatePromotion({...input,usedHoldouts:consumed});
for(const c of input.cases??[]){let valid=false;try{valid=sha(await readFile(c.rawSourcePath,'utf8'))===c.rawSourceSha256;}catch{}if(!valid){result.eligible=false;result.reasons.push('RAW_CASE_SOURCE_BYTES_UNVERIFIED');}}
// Operational proof must be backed by files pinned in the frozen protocol, not prose alone.
for(const proof of ['unit','integration','shadow','rollback','ledger']){
 const ref=input.proofs?.[proof];let valid=false;
 try{valid=!!ref&&sha(await readFile(ref.path,'utf8'))===ref.sha256;}catch{}
 if(!valid){result.eligible=false;result.reasons.push('PINNED_PROOF_MISSING_'+proof);}
}
await writeFile(out+'/promotion.json',JSON.stringify(result,null,2)+'\n');
if(result.eligible){
 if(input.candidate.sourceRegistrySha256!==sha(registry))throw Error('MODEL_REGISTRY_CHANGED_REBASE_RESEARCH');
 const model={id:input.candidate.id,family:input.candidate.family,logic:input.candidate.logic,version:registry.version+1,validationProtocolSha256:input.protocol.sha256};
 const next={...registry,version:registry.version+1,status:'VALIDATED_SYMBOLIC_MODELS',models:[...registry.models.filter(x=>x.id!==model.id),model],usedHoldouts:[...registry.usedHoldouts,result.holdoutId],ordersEnabled:false};
 await writeFile('v17-common/quant-model-registry.json',JSON.stringify(next,null,2)+'\n');
}
console.log(JSON.stringify(result));
