const ORIGIN='https://btc-eth-v17-canonical-api.onrender.com';
let cached=null,expires=0;
export async function researchToken(){
 if(cached&&Date.now()<expires)return cached;
 const tokenUrl=process.env.ACTIONS_ID_TOKEN_REQUEST_URL,requestToken=process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN;
 if(!tokenUrl||!requestToken)throw Error('GITHUB_OIDC_IDENTITY_REQUIRED');
 const u=new URL(tokenUrl);u.searchParams.set('audience',ORIGIN);
 const r=await fetch(u,{headers:{Authorization:'Bearer '+requestToken},signal:AbortSignal.timeout(15000)});
 if(!r.ok)throw Error('GITHUB_OIDC_ISSUANCE_FAILED');
 cached=(await r.json()).value;if(typeof cached!=='string')throw Error('GITHUB_OIDC_TOKEN_MISSING');expires=Date.now()+180000;return cached;
}
export async function control(body){
 const r=await fetch(ORIGIN+'/research/control',{method:'POST',headers:{Authorization:'Bearer '+await researchToken(),'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(45000)});
 if(!r.ok)throw Error('RESEARCH_CONTROL_HTTP_'+r.status);return r.json();
}
export async function liveHealth(){const r=await fetch(ORIGIN+'/health',{signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('ENGINE_HEALTH_HTTP_'+r.status);return r.json();}
