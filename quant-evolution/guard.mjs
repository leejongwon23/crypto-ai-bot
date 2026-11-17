import {execFileSync} from 'node:child_process';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {sha} from '../v17-common/quant-structure.mjs';
import {guardedRollout,inspectDeployment,buildRollbackTree} from './guard-core.mjs';
import {control,liveHealth} from './control-client.mjs';
const git=(...args)=>execFileSync('git',args,{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
const expectedCommit=process.env.EXPECTED_DEPLOY_COMMIT||git('rev-parse','HEAD');
if(!/^[a-f0-9]{40}$/.test(expectedCommit))throw Error('DEPLOY_COMMIT_REQUIRED');
const branch='refs/heads/v17-canonical-engine',out='quant-evolution/output';await mkdir(out,{recursive:true});
function manifest(commit){
 return {commit,policySha256:sha(execFileSync('git',['show',commit+':v17-common/MASTER_PROMPT_V2.0_FINAL.txt'],{encoding:'utf8'})),registrySha256:sha(JSON.parse(git('show',commit+':v17-common/quant-model-registry.json')))};
}
const wait=()=>new Promise(r=>setTimeout(r,10000));
let checkpoint=null,bootstrap=false;
try{checkpoint=(await control({action:'CHECKPOINT'})).checkpoint;}
catch(e){
 // This single bootstrap is needed when upgrading from the pre-controller
 // release. The user-directed release is separately checked by official read.
 const parent=git('rev-parse',expectedCommit+'^1');
 if(parent!=='c5095a6326b4a6a8a9fc652a059d9043583ab2d3')throw e;
 bootstrap=true;
 for(let n=0;n<90;n++){try{const h=await liveHealth();if(h.buildCommit===expectedCommit){checkpoint=(await control({action:'CHECKPOINT'})).checkpoint;break;}}catch{}await wait();}
 if(!checkpoint)throw Error('RESEARCH_CONTROL_BOOTSTRAP_FAILED');
}
let drill=false;try{const f=JSON.parse(git('show',expectedCommit+':v17-common/deployment-drill.json'));drill=f.kind==='CONTROLLED_ROLLBACK_REHEARSAL'&&f.expectedFailure===true;}catch{}
const expected=manifest(expectedCommit);
const result=await guardedRollout({expected,checkpoint,readHealth:liveHealth,verifyIntegrity:id=>control({action:'VERIFY',checkpoint:id}),wait,drill,forcedFailure:!!process.env.ENGINE_TEST_STATUS&&process.env.ENGINE_TEST_STATUS!=='success',rollback:async()=>{
 if(git('ls-remote','origin',branch).split(/\s+/)[0]!==expectedCommit)return {verified:false,reason:'CONCURRENT_SOURCE_CHANGE_NO_OVERWRITE'};
 let known;
 try{git('fetch','origin','v20-verified-production');known=git('rev-parse','FETCH_HEAD');}
 catch{try{known=git('rev-parse','refs/remotes/origin/v20-verified-production');}catch{const parent=git('rev-parse',expectedCommit+'^1');if(parent!=='c5095a6326b4a6a8a9fc652a059d9043583ab2d3')throw Error('NO_PREVIOUS_VERIFIED_ROLLBACK_TARGET');known=parent;}}
 if(known===expectedCommit)return {verified:false,reason:'NO_PRIOR_DISTINCT_VERIFIED_RELEASE'};
 const tree=buildRollbackTree(git,{verifiedCommit:known,failedCommit:expectedCommit});
 const rollbackCommit=git('-c','user.name=btc-eth-deployment-guard','-c','user.email=btc-eth-research@users.noreply.github.com','commit-tree',tree,'-p',expectedCommit,'-m','Automatically restore verified source after failed V2 deployment checks');
 git('push','origin',rollbackCommit+':'+branch);
 const target=manifest(rollbackCommit);
 for(let i=0;i<90;i++){
  try{const health=await liveHealth(),integrity=await control({action:'VERIFY',checkpoint});if(inspectDeployment({health,expected:target,integrity}).passed)return {verified:true,commit:rollbackCommit,restoredTree:tree,health,integrity,ordersEnabled:false};}catch{}
  await wait();
 }
 return {verified:false,commit:rollbackCommit,reason:'ROLLBACK_LIVE_VERIFICATION_TIMEOUT'};
}});
if(result.status==='VERIFIED_LIVE'){
 const live=git('ls-remote','origin',branch).split(/\s+/)[0];
 if(live!==expectedCommit)throw Error('CONCURRENT_RELEASE_DO_NOT_MARK_VERIFIED');
 // The ref moves only forward in the production history, without rewriting it.
 git('push','origin',expectedCommit+':refs/heads/v20-verified-production');
}
const report={...result,expected,checkpoint,bootstrap,drill,checkedAt:new Date().toISOString(),ordersEnabled:false};
await writeFile(out+'/deployment-guard.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:result.status,commit:expectedCommit,rollbackCommit:result.rollback?.commit,bootstrap,drill}));
if(result.status!=='VERIFIED_LIVE'&&!(drill&&result.status==='ROLLED_BACK'))process.exitCode=1;
