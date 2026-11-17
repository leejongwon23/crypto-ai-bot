import {createPublicKey,verify} from 'node:crypto';
import {buildQuantContext,sha} from './quant-structure.mjs';

const ISSUER='https://token.actions.githubusercontent.com';
const JWKS=ISSUER+'/.well-known/jwks';
const REPO='leejongwon23/crypto-ai-bot';
const REPO_ID='978519065',OWNER_ID='209888234';
const REFS=new Set(['refs/heads/main','refs/heads/v17-canonical-engine']);
const WORKFLOW='.github/workflows/v20-mathematical-evolution.yml';
let keys=null,expires=0;

// This identity grants only this read-only, market-only export. It is never
// accepted by /api, OAuth, the ledger writer, or any exchange-order interface.
export async function verifyResearchIdentity(token,{audience,now=Date.now(),fetcher=fetch}={}){
 if(typeof token!=='string'||token.length>20000)throw Error('RESEARCH_IDENTITY_INVALID');
 const parts=token.split('.');if(parts.length!==3||parts.some(x=>!x||!/^[A-Za-z0-9_-]+$/.test(x)))throw Error('RESEARCH_IDENTITY_INVALID');
 let header,p;try{header=JSON.parse(Buffer.from(parts[0],'base64url'));p=JSON.parse(Buffer.from(parts[1],'base64url'));}catch{throw Error('RESEARCH_IDENTITY_INVALID');}
 if(header.alg!=='RS256'||typeof header.kid!=='string'||header.jku||header.jwk||header.x5u)throw Error('RESEARCH_ALGORITHM_INVALID');
 const seconds=Math.floor(now/1000);
 if(!audience||p.iss!==ISSUER||p.aud!==audience||![p.exp,p.iat,p.nbf].every(Number.isFinite)||p.exp<=seconds||p.nbf>seconds+30||p.iat>seconds+30||p.exp-p.iat>600||seconds-p.iat>600)throw Error('RESEARCH_TOKEN_SCOPE_OR_TIME_INVALID');
 if(p.repository!==REPO||p.repository_id!==REPO_ID||p.repository_owner_id!==OWNER_ID||!REFS.has(p.ref)||p.ref_type!=='branch'||!['schedule','workflow_dispatch','push'].includes(p.event_name)||p.workflow_ref!==REPO+'/'+WORKFLOW+'@'+p.ref)throw Error('RESEARCH_WORKFLOW_FORBIDDEN');
 // Pinned numeric repository and owner IDs protect against recycled names.
 if(!keys||expires<now){const r=await fetcher(JWKS,{signal:AbortSignal.timeout(5000),redirect:'error'});if(!r.ok)throw Error('RESEARCH_KEYS_UNAVAILABLE');const body=await r.json();if(!Array.isArray(body.keys)||body.keys.length>20)throw Error('RESEARCH_KEYS_INVALID');keys=body.keys;expires=now+300000;}
 const key=keys.find(x=>x.kid===header.kid&&x.kty==='RSA'&&x.use==='sig'&&(x.alg===undefined||x.alg==='RS256'));
 if(!key||!verify('RSA-SHA256',Buffer.from(parts[0]+'.'+parts[1]),createPublicKey({key,format:'jwk'}),Buffer.from(parts[2],'base64url')))throw Error('RESEARCH_SIGNATURE_INVALID');
 return {repository:REPO,ref:p.ref,runId:String(p.run_id??'')};
}

export function marketOnlyRecord(row){
 const context=buildQuantContext(row.raw,[]),symbols={};
 for(const symbol of ['BTC','ETH']){const s=context.symbols[symbol];if(!s.asOf||!Number.isFinite(Date.parse(s.asOf.replace(' ','T')+'+09:00')))throw Error('RESEARCH_SOURCE_TIME_MISSING');symbols[symbol]={asOf:s.asOf,identity:s.identity,provenance:s.provenance,dataStatus:s.dataStatus,structure:s.structure,flow:s.flow,timeframes:s.timeframes,modelErrorReview:[]};}
 const record={sourceSha256:sha(row.raw),observedAt:new Date(row.created_at).toISOString(),version:context.version,symbols,source:'PRESERVED_USER_SUPPLIED_MARKET_OBSERVATIONS',independentExchangeAuthentication:false};
 return {...record,recordSha256:sha(record)};
}

export async function readMarketResearchPage(db,{beforeAt,beforeSha,limit=25}={}){
 if(!Number.isInteger(limit)||limit<1||limit>50)throw Error('RESEARCH_PAGE_LIMIT');
 if(beforeAt!==undefined&&(!Number.isFinite(Date.parse(beforeAt))||!/^[a-f0-9]{64}$/.test(beforeSha??'')))throw Error('RESEARCH_CURSOR_INVALID');
 // No signals, events, account information, free-form raw text or AI prose leave
// the server. Original snapshots remain in the existing append-preserved DB.
 const r=await db.query(`WITH sources AS (
 SELECT snapshot_sha,raw,created_at FROM v17_ai_prepared_snapshots WHERE session='OFFICIAL_V17'
 UNION ALL SELECT snapshot_sha,raw,created_at FROM v20_ai_analysis_history WHERE session='OFFICIAL_V17'
 ), unique_sources AS (SELECT DISTINCT ON(snapshot_sha) snapshot_sha,raw,created_at FROM sources ORDER BY snapshot_sha,created_at)
 SELECT snapshot_sha,raw,created_at,created_at::text AS cursor_at FROM unique_sources
 WHERE ($1::timestamptz IS NULL OR (created_at,snapshot_sha)<($1::timestamptz,$2::text))
 ORDER BY created_at DESC,snapshot_sha DESC LIMIT $3`,[beforeAt??null,beforeSha??null,limit]);
 const records=[],rejected=[];
 for(const row of r.rows){try{if(sha(row.raw)!==row.snapshot_sha)throw Error('SOURCE_SHA_MISMATCH');records.push(marketOnlyRecord(row));}catch{rejected.push({sourceSha256:row.snapshot_sha,reason:'MALFORMED_OR_UNVERIFIABLE_MARKET_SOURCE'});}}
 const last=r.rows.at(-1);
 return {schema:'V20_MARKET_RESEARCH_FEED_1',records,rejected,next:r.rows.length===limit?{beforeAt:last.cursor_at,beforeSha:last.snapshot_sha}:null,ordersEnabled:false};
}
