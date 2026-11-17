import test from 'node:test';
import assert from 'node:assert/strict';
import {generateKeyPairSync,sign} from 'node:crypto';
import {verifyResearchIdentity,marketOnlyRecord,readMarketResearchPage} from './research-feed.mjs';
import {sha} from './quant-structure.mjs';

const origin='https://btc-eth-v17-canonical-api.onrender.com',now=Date.parse('2026-11-17T03:00:00Z');
const {privateKey,publicKey}=generateKeyPairSync('rsa',{modulusLength:2048});
const jwk={...publicKey.export({format:'jwk'}),kid:'test-key',use:'sig',alg:'RS256'};
const claims={iss:'https://token.actions.githubusercontent.com',aud:origin,exp:now/1000+300,iat:now/1000,nbf:now/1000,repository:'leejongwon23/crypto-ai-bot',repository_id:'978519065',repository_owner_id:'209888234',ref:'refs/heads/main',ref_type:'branch',event_name:'push',workflow_ref:'leejongwon23/crypto-ai-bot/.github/workflows/v20-mathematical-evolution.yml@refs/heads/main',run_id:'123'};
const encode=x=>Buffer.from(JSON.stringify(x)).toString('base64url');
function jwt(p=claims,h={alg:'RS256',kid:'test-key'}){const s=encode(h)+'.'+encode(p);return s+'.'+sign('RSA-SHA256',Buffer.from(s),privateKey).toString('base64url');}
const options={audience:origin,now,fetcher:async url=>{assert.equal(url,'https://token.actions.githubusercontent.com/.well-known/jwks');return {ok:true,json:async()=>({keys:[jwk]})};}};
test('real RSA signature accepts only the pinned repo and research workflow',async()=>{assert.equal((await verifyResearchIdentity(jwt(),options)).runId,'123');});
test('cross-repository, fork, alternate workflow and PR identities reject',async()=>{
 for(const change of [{repository_id:'999'},{repository_owner_id:'999'},{repository:'evil/crypto-ai-bot'},{ref:'refs/heads/feature'},{event_name:'pull_request'},{workflow_ref:'leejongwon23/crypto-ai-bot/.github/workflows/other.yml@refs/heads/main'}])await assert.rejects(verifyResearchIdentity(jwt({...claims,...change}),options),/FORBIDDEN/);
});
test('expired tokens, wrong audience, algorithm substitution and altered signatures reject',async()=>{
 for(const change of [{exp:now/1000-1},{aud:'https://other.example'},{nbf:now/1000+100},{iat:now/1000-1000}])await assert.rejects(verifyResearchIdentity(jwt({...claims,...change}),options));
 await assert.rejects(verifyResearchIdentity(jwt(claims,{alg:'none',kid:'test-key'}),options));
 const t=jwt(),parts=t.split('.');parts[1]=encode({...claims,run_id:'tampered'});await assert.rejects(verifyResearchIdentity(parts.join('.'),options),/SIGNATURE/);
});
const raw=`========== BTC (BTCUSDT) ==========\ntimestamp_kst: 2026-11-17 09:10:00\nlastPrice: 100\n========== ETH (ETHUSDT) ==========\ntimestamp_kst: 2026-11-17 09:10:00\nlastPrice: 200\nPRIVATE_ACCOUNT_SECRET: should-never-leave\nmy_position: 25 ETH\n`;
const row={raw,snapshot_sha:sha(raw),created_at:new Date(now),cursor_at:'2026-11-17 03:00:00.000123+00'};
test('market export strips user prose, account data, plans and signal identifiers',()=>{
 const r=marketOnlyRecord({...row,official:{signals:[{id:'private-signal'}]}}),text=JSON.stringify(r);
 assert.ok(!text.includes('should-never-leave')&&!text.includes('private-signal')&&!text.includes('my_position'));
 assert.ok(!('raw'in r)&&!('directions'in r.symbols.BTC));assert.equal(r.symbols.BTC.dataStatus.closedBars,0);
 const {recordSha256,...body}=r;assert.equal(sha(body),recordSha256);assert.equal(r.independentExchangeAuthentication,false);
});
test('read-only SQL uses fixed official session and parameters; cursor and page are bounded',async()=>{
 let calls=0;const db={query:async(sql,args)=>{calls++;assert.doesNotMatch(sql,/\b(UPDATE|DELETE|INSERT|ALTER)\b/);assert.deepEqual(args,[null,null,1]);return {rows:[row]};}};
 const p=await readMarketResearchPage(db,{limit:1});assert.equal(calls,1);assert.equal(p.records.length,1);assert.equal(p.next.beforeSha,sha(raw));assert.equal(p.next.beforeAt,row.cursor_at);assert.equal(p.ordersEnabled,false);
 await assert.rejects(readMarketResearchPage(db,{limit:100}));await assert.rejects(readMarketResearchPage(db,{beforeAt:'invalid',beforeSha:'x'}));
});
test('changed stored raw source is quarantined rather than treated as valid evidence',async()=>{
 const p=await readMarketResearchPage({query:async()=>({rows:[{...row,raw:raw+'tampered'}]})});assert.equal(p.records.length,0);assert.equal(p.rejected.length,1);
});
