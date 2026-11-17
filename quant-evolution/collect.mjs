import {mkdir,writeFile} from 'node:fs/promises';
import {generateResearchProposals} from '../v17-common/quant-evolution.mjs';
import {sha} from '../v17-common/quant-structure.mjs';
const origin='https://btc-eth-v17-canonical-api.onrender.com';
const out='quant-evolution/output';await mkdir(out,{recursive:true});
// GitHub issues a short-lived identity, not an API master key or DB password.
const tokenUrl=process.env.ACTIONS_ID_TOKEN_REQUEST_URL,requestToken=process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN;
if(!tokenUrl||!requestToken)throw Error('GITHUB_OIDC_IDENTITY_REQUIRED');
const u=new URL(tokenUrl);u.searchParams.set('audience',origin);
const issued=await fetch(u,{headers:{Authorization:'Bearer '+requestToken},signal:AbortSignal.timeout(15000)});
if(!issued.ok)throw Error('GITHUB_OIDC_ISSUANCE_FAILED');
const token=(await issued.json()).value;if(typeof token!=='string')throw Error('GITHUB_OIDC_TOKEN_MISSING');
const records=[],rejected=[];let cursor=null,complete=false;
for(let page=0;page<20;page++){
 const url=new URL('/research/market',origin);url.searchParams.set('limit','25');if(cursor)for(const [k,v]of Object.entries(cursor))url.searchParams.set(k,v);
 const r=await fetch(url,{headers:{Authorization:'Bearer '+token},signal:AbortSignal.timeout(45000)});
 if(!r.ok)throw Error('RESEARCH_FEED_HTTP_'+r.status);
 const data=await r.json();if(data.schema!=='V20_MARKET_RESEARCH_FEED_1'||!Array.isArray(data.records))throw Error('RESEARCH_FEED_SCHEMA_INVALID');
 for(const row of data.records){const {recordSha256,...body}=row;if(sha(body)!==recordSha256)throw Error('RESEARCH_RECORD_SHA_MISMATCH');records.push(row);}
 rejected.push(...data.rejected);cursor=data.next;if(!cursor){complete=true;break;}
}
const unique=[...new Map(records.map(r=>[r.sourceSha256,r])).values()].sort((a,b)=>a.observedAt.localeCompare(b.observedAt));
const proposals=[];for(const row of unique){proposals.push(...generateResearchProposals({...row,sourceSha256:row.sourceSha256},proposals));}
await writeFile(out+'/market-observations.json',JSON.stringify(unique,null,2)+'\n');
await writeFile(out+'/mechanism-proposals.json',JSON.stringify(proposals,null,2)+'\n');
const report={connected:true,records:unique.length,proposals:proposals.length,rejected:rejected.length,complete,next:cursor,datasetSha256:sha(unique),rawUserTextExported:false,independentExchangeAuthentication:false,frozenStudyCreated:false,marketPerformanceVerified:false,ordersEnabled:false};
await writeFile(out+'/collection.json',JSON.stringify({...report,rejectedSources:rejected},null,2)+'\n');console.log(JSON.stringify(report));
