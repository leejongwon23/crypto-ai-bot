export function inspectDeployment({health,expected,integrity,forcedFailure=false,drill=false}){
 const reasons=[];
 if(forcedFailure)reasons.push('SOURCE_TESTS_FAILED');
 if(!health||health.status!=='ready')reasons.push('SERVICE_NOT_READY');
 if(health?.buildCommit!==expected.commit)reasons.push('DEPLOYED_COMMIT_MISMATCH');
 if(health?.policySha256!==expected.policySha256)reasons.push('DEPLOYED_POLICY_MISMATCH');
 if(health?.quantEngine?.registrySha256!==expected.registrySha256)reasons.push('DEPLOYED_MODEL_REGISTRY_MISMATCH');
 if(health?.quantEngine?.automaticOrders!==false)reasons.push('UNAUTHORIZED_EXCHANGE_ORDER_CAPABILITY');
 if(integrity?.intact!==true)reasons.push('OFFICIAL_ORIGINAL_OR_EVENT_PREFIX_CHANGED');
 if(drill)reasons.push('CONTROLLED_ROLLBACK_REHEARSAL');
 return {passed:reasons.length===0,reasons};
}
export async function guardedRollout({expected,checkpoint,readHealth,verifyIntegrity,rollback,wait,attempts=90,drill=false,forcedFailure=false}){
 let last=null,consecutive=0;
 for(let i=0;i<attempts;i++){
  try{
   const health=await readHealth();
   if(health.buildCommit===expected.commit||forcedFailure){
    const integrity=await verifyIntegrity(checkpoint),inspection=inspectDeployment({health,expected,integrity,drill,forcedFailure});last={health,integrity,...inspection};
    if(inspection.passed){if(++consecutive>=3)return {status:'VERIFIED_LIVE',...last};}
    else if(health.buildCommit===expected.commit||forcedFailure)break;
   }else consecutive=0;
  }catch(e){last={error:String(e.message||e)};consecutive=0;}
  await wait();
 }
 const result=await rollback({expected,checkpoint,failure:last??{reasons:['DEPLOYMENT_TIMEOUT']}});
 return {status:result.verified?'ROLLED_BACK':'ROLLBACK_UNVERIFIED',failure:last,rollback:result};
}
