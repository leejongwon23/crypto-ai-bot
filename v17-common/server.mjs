import http from 'node:http';
import crypto from 'node:crypto';
import {makeOAuth} from './oauth.mjs';
import {OFFICIAL_POLICY,VERSION,prepare,sourcePolicy,validateSnapshot,validateOfficialSubmission,ledgerHash,sessionId,digest,checkEvent} from './v17-core.mjs';

const PORT=Number(process.env.PORT||3000);
const MODE=process.env.DEPLOYMENT_MODE||'local';
const MASTER_SECRET=process.env.SIGNAL_API_KEY||'';
const DB_URL=process.env.DATABASE_URL||'';
const PUBLIC_ORIGIN=process.env.PUBLIC_ORIGIN||'https://btc-eth-v17-canonical-api.onrender.com';
const MAX_BODY=600000;
let pool;let oauth;
const sql=`
CREATE TABLE IF NOT EXISTS v17_ai_sessions (name TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v17_ai_signals (session TEXT NOT NULL, id TEXT NOT NULL, symbol TEXT NOT NULL, original JSONB NOT NULL, status TEXT NOT NULL, verified_until TEXT, plan_version INT NOT NULL DEFAULT 0, PRIMARY KEY(session,id));
CREATE TABLE IF NOT EXISTS v17_ai_events (session TEXT NOT NULL, seq BIGSERIAL PRIMARY KEY, id TEXT NOT NULL, event JSONB NOT NULL, at_kst TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v17_ai_analyses (session TEXT NOT NULL, snapshot_sha TEXT NOT NULL, before_sha TEXT NOT NULL, after_sha TEXT NOT NULL, policy_sha TEXT NOT NULL, raw TEXT NOT NULL, official JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(session,snapshot_sha,before_sha));
CREATE INDEX IF NOT EXISTS v17_ai_analyses_after ON v17_ai_analyses (session,snapshot_sha,after_sha);
`;
const json=(res,code,payload,headers={})=>{res.writeHead(code,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff',...headers});res.end(JSON.stringify(payload));};
const responseError=(res,code,m)=>json(res,code,{error:String(m)});
async function readJSON(req){let size=0,b=[];for await(const chunk of req){size+=chunk.length;if(size>MAX_BODY)throw Error('REQUEST_TOO_LARGE');b.push(chunk);}const s=Buffer.concat(b).toString('utf8');return s?JSON.parse(s):{};}
async function database(){
 if(pool)return pool;
 if(!DB_URL)throw Error('PERSISTENT_DATABASE_REQUIRED');
 const {Pool}=await import('pg');
 const p=new Pool({connectionString:DB_URL,ssl:DB_URL.includes('localhost')?false:{rejectUnauthorized:false},connectionTimeoutMillis:10000,max:3});
 try{await p.query(sql);}catch(e){await p.end();throw e;}
 pool=p;return p;
}
async function getOAuth(){if(oauth)return oauth;const db=await database();const x=makeOAuth({origin:PUBLIC_ORIGIN,pool:db,ownerSecret:MASTER_SECRET});await x.init();oauth=x;return oauth;}
const KEY_CHECK=x=>{
 if(!MASTER_SECRET||typeof x!=='string'||x.length!==MASTER_SECRET.length)return false;
 return crypto.timingSafeEqual(Buffer.from(x),Buffer.from(MASTER_SECRET));
};
function isAuthorized(req){const header=String(req.headers.authorization||'');const provided=header.replace(/^Bearer\s+/i,'')||String(req.headers['x-api-key']||'');return KEY_CHECK(provided);}
export const rowToLedger=r=>({id:r.id,symbol:r.symbol,original:r.original,status:r.status,verifiedUntil:r.verified_until,version:r.plan_version,unverifiedFrom:r.unverified_from??undefined,...(r.effective_plan?{effectivePlan:r.effective_plan,planEffectiveAt:r.plan_effective_at}:{})});
async function readLedger(db,session){
 const r=await db.query(`SELECT s.id,s.symbol,s.original,s.status,s.verified_until,s.plan_version,
 (SELECT e.event->>'priorConfirmedState' FROM v17_ai_events e
  WHERE e.session=s.session AND e.id=s.id AND e.event->>'kind'='UNVERIFIED'
  ORDER BY e.seq DESC LIMIT 1) AS unverified_from,
 (SELECT e.event->'plan' FROM v17_ai_events e
  WHERE e.session=s.session AND e.id=s.id AND e.event->>'kind'='REVISION'
  ORDER BY e.seq DESC LIMIT 1) AS effective_plan,
 (SELECT e.at_kst FROM v17_ai_events e
  WHERE e.session=s.session AND e.id=s.id AND e.event->>'kind'='REVISION'
  ORDER BY e.seq DESC LIMIT 1) AS plan_effective_at
 FROM v17_ai_signals s WHERE s.session=$1 ORDER BY s.id`,[session]);
 return r.rows.map(rowToLedger);
}
// Recover previously detected gaps from the immutable analysis history, even if
// the latest *new* time window is continuous. A later candle never repairs an
// unobserved older interval by itself. No schema or historical rows are changed.
async function carriedUncertainty(db,session,ledger){
 const ids=new Set(ledger.filter(s=>['PENDING','NEW_ENTRY','ENTERED','TP1','TP2'].includes(s.status)).map(s=>s.id));
 const result={BTC:[],ETH:[]};
 if(!ids.size)return result;
 const r=await db.query("SELECT official->'evidenceQuality' AS evidence FROM v17_ai_analyses WHERE session=$1 AND official->'evidenceQuality' IS NOT NULL ORDER BY created_at DESC LIMIT 1000",[session]);
 for(const row of r.rows){
  for(const sym of ['BTC','ETH']){
   const q=row.evidence?.perSymbol?.[sym]||{};
   for(const id of [...(q.couldNotVerifySignals||[]),...(q.ambiguousSignals||[]),...(q.carriedUncertaintyIds||[])]){
    if(ids.has(id)&&!result[sym].includes(id))result[sym].push(id);
   }
  }
 }
 return result;
}
async function selectOfficial(db,session,snapshotSha,stateSha){
 const q=await db.query('SELECT official FROM v17_ai_analyses WHERE session=$1 AND snapshot_sha=$2 AND (before_sha=$3 OR after_sha=$3) ORDER BY created_at ASC LIMIT 1',[session,snapshotSha,stateSha]);
 return q.rows[0]?.official||null;
}
async function latestOfficialSnapshot(db,session){
 const r=await db.query("SELECT official->'snapshot' AS snapshot FROM v17_ai_analyses WHERE session=$1 ORDER BY created_at DESC LIMIT 1",[session]);
 return r.rows[0]?.snapshot||null;
}
async function beginAnalysis(body){
 const raw=body.raw,session=sessionId(body.session||'OFFICIAL_V17');
 const snapshot=validateSnapshot(raw);const db=await database();
 const ledger=await readLedger(db,session);
 const saved=await selectOfficial(db,session,snapshot.sha256,ledgerHash(ledger));
 const previousSnapshot=await latestOfficialSnapshot(db,session);
 const carry=await carriedUncertainty(db,session,ledger);
 return prepare({raw,session,ledger,saved,previousSnapshot,carriedUncertaintyBySymbol:carry});
}
async function publishAnalysis(body){
 const raw=body.raw,session=sessionId(body.session||'OFFICIAL_V17');const shot=validateSnapshot(raw);
 const db=await database(),client=await db.connect();
 try{
  await client.query('BEGIN');
  await client.query('INSERT INTO v17_ai_sessions(name) VALUES($1) ON CONFLICT DO NOTHING',[session]);
  await client.query('SELECT name FROM v17_ai_sessions WHERE name=$1 FOR UPDATE',[session]);
  const ledger=await readLedger(client,session);
  const before=ledgerHash(ledger);
  // At most one official result per snapshot and current ledger state.
  const prior=await selectOfficial(client,session,shot.sha256,before);
  if(prior){await client.query('COMMIT');return {status:'OFFICIAL_REPLAY',officialDecision:prior,sameState:true,notReanalyzed:true};}
  const previousSnapshot=await latestOfficialSnapshot(client,session);
  const carry=await carriedUncertainty(client,session,ledger);
  const verified=validateOfficialSubmission(body,ledger,previousSnapshot,carry);
  const newSignals=verified.newSignals,events=verified.events;
  for(const s of newSignals){
   await client.query('INSERT INTO v17_ai_signals(session,id,symbol,original,status,verified_until,plan_version) VALUES($1,$2,$3,$4,$5,$6,0)',[session,s.id,s.symbol,JSON.stringify(s),s.status,s.issuedAt]);
  }
  for(const event of events){
   // Stored events are append-only. A revision never overwrites the original plan.
   await client.query('INSERT INTO v17_ai_events(session,id,event,at_kst) VALUES($1,$2,$3,$4)',[session,event.id,JSON.stringify(event),event.at]);
   if(event.kind==='REVISION')await client.query('UPDATE v17_ai_signals SET plan_version=plan_version+1,verified_until=$3 WHERE session=$1 AND id=$2',[session,event.id,event.at]);
   else await client.query('UPDATE v17_ai_signals SET status=$3,verified_until=$4 WHERE session=$1 AND id=$2',[session,event.id,event.kind,event.at]);
  }
  const after=ledgerHash(await readLedger(client,session));
  const official={
   engine:VERSION,
   status:'OFFICIAL_STORED',
   origin:'CHATGPT_AI_APPLIED_FULL_VERBATIM_V17_PROMPT',
   notAutomatedServerTradeModel:true,
   policy:OFFICIAL_POLICY,
   snapshot:shot,
   evidenceQuality:verified.evidenceQuality,
   session,
   beforeLedgerSha256:before,
   afterLedgerSha256:after,
   decisionKey:verified.decisionKey,
   reportMarkdown:verified.analysis.reportMarkdown,
   directions:verified.analysis.directions,
   clauses:verified.analysis.clauses,
   coupling:verified.analysis.coupling,
   checks:verified.analysis.checks,
   audit20:verified.analysis.audit20,
   conclusions:verified.analysis.conclusions,
   historicalUncertainty:verified.analysis.historicalUncertainty||{},
   newSignals,
   events,
   caveat:'The same snapshot and same ledger state replays this official result. AI analytical correctness and complete semantic compliance require independent review; no exchange orders are placed.'
  };
  await client.query('INSERT INTO v17_ai_analyses(session,snapshot_sha,before_sha,after_sha,policy_sha,raw,official) VALUES($1,$2,$3,$4,$5,$6,$7)',[session,shot.sha256,before,after,OFFICIAL_POLICY.sha256,raw,JSON.stringify(official)]);
  await client.query('COMMIT');return {status:'OFFICIAL_STORED',officialDecision:official,sameState:true};
 }catch(e){try{await client.query('ROLLBACK');}catch{}throw e;}finally{client.release();}
}
async function ledgerView(args){const session=sessionId(args.session||'OFFICIAL_V17'),db=await database(),ledger=await readLedger(db,session);return {session,policy:OFFICIAL_POLICY,ledgerSha256:ledgerHash(ledger),signals:ledger,events:(await db.query('SELECT id,event,at_kst FROM v17_ai_events WHERE session=$1 ORDER BY seq DESC LIMIT 500',[session])).rows};}
// Read-only audit trail. The official signal/event ledger is never rewritten.
async function decisionHistory(args={}){
 const session=sessionId(args.session||'OFFICIAL_V17'),db=await database();
 const limit=Math.max(1,Math.min(100,Number.isSafeInteger(args.limit)?args.limit:30));
 const r=await db.query("SELECT created_at, official->'snapshot' AS snapshot, official->'conclusions' AS conclusions, official->'directions' AS directions, official->'newSignals' AS new_signals, official->'events' AS events FROM v17_ai_analyses WHERE session=$1 ORDER BY created_at DESC LIMIT $2",[session,limit]);
 return {session,limit,officialHistory:r.rows.map(x=>({createdAt:x.created_at,snapshot:x.snapshot,conclusions:x.conclusions,
  directions:args.includeDirections===true?x.directions:undefined,newSignals:x.new_signals,events:x.events})),
  warning:'Decisions were made as of each snapshot. A later favorable price move does not retroactively prove an earlier candidate had been eligible.'};
}
async function showPolicy(){return {engine:VERSION,policy:sourcePolicy(),instruction:'Use every source section 0–23, appendices A and B and final integrated execution instruction. No substitutions.'};}
const requiredSchema={type:'object',properties:{raw:{type:'string',description:'Verbatim BTC and ETH snapshots, as pasted by the user'},session:{type:'string',description:'Use shared OFFICIAL_V17 for a single official signal ledger. Separate arbitrary sessions must not import old signals.'}},required:['raw']};
const tools=[
 {name:'run_official_btc_eth_snapshot',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'FIRST STEP (MANDATORY). Fetch ENTIRE unchanged V1.7 MASTER PROMPT, official ledger and evidenceQuality, including carried historical unknown periods. If OFFICIAL_REPLAY show officialDecision.reportMarkdown verbatim. Analyze current LONG/SHORT and A-E independently of unknown old trade outcome; preserve and report historicalUnknownIds separately, with no guessed fills or flat positions. Old pending can be quarantined with documented UNVERIFIED event before prospective replacement if it cannot be resolved. AUTO-SWITCH: on each latest snapshot compare existing KEEP, old pending redesign (terminate old then mint complete new ID), same-direction ADD_ON, opposite-side PENDING switch candidate, verified EARLY_EXIT or REVERSE_EXIT, and prospective REVISION. Same-direction and opposite pending can coexist with a proven entered trade when evidence and metadata meet policy; never claim both sides entered. Publish via full original V1.7 analysis. Never bypass risk/exposure checks.',inputSchema:requiredSchema},
 {name:'get_original_v17_full_prompt',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'Fetch EXACT full 902-line user-supplied V1.7 source, CRLF and original punctuation preserved. Must be the sole authoritative strategy; no new indicator thresholds, no removed functionality.',inputSchema:{type:'object',properties:{},additionalProperties:false}},
 {name:'publish_official_v17_analysis',securitySchemes:[{type:'oauth2',scopes:['signals:write']}],description:'SECOND STEP after NEEDS_AI_ANALYSIS. Existing confirmed price-touch and stop/TP event checks still mandatory. IMPORTANT GAP FIX: unknown historical signal outcome does NOT force current BTC/ETH conclusion=UNVERIFIED or block fresh independent market analysis. Include analysis.historicalUncertainty.{BTC|ETH}={unresolvedIds:[...],evidence:string} for each affected symbol; do not claim old position closed or account flat. To replace old PENDING with unresolved gap, include an old-ID UNVERIFIED quarantine event (uncertaintyScope,priorConfirmedState,marketReassessment, at current snapshot time) in same atomic submission, and new formal plan historicalExposureUnknownIds, historicalExposureRiskDisclosure, independentCurrentMarketEvidence. New plan is an analytical signal, NOT exchange account position certainty. An old ENTERED can coexist with separately justified same-direction ADD_ON or opposite PENDING switch plan; new signal must identify parentSignalId, relationToExisting SAME_DIRECTION_ADD_ON with addOnRiskEvidence and existingPositionRiskEvidence, OR OPPOSITE_PENDING with transitionOnTouchEvidence and existingPositionRiskEvidence. Old proven ENTERED cannot be silently replaced or counted flat; any opposite ENTERED event needs all opposed positions to have verified exit events first. REVERSE_EXIT/EARLY_EXIT requires strongCounterEvidence, structureEvidence, flowEvidence, contextEvidence, and outcomeUnverified=true if no verified entry/exit prices; actual PnL requires verifiedEntryPrice and entryPriceEvidence matching calculated PnL. REVISION event requires observedPrice=current snapshot, new plan stop/TP1/TP2/TP3, immutable previous achieved milestones and tighter effective stop; revised plan is read from ledger on following runs. Never retroactively cancel or claim TP/SL in gaps. Four directions A-E, original V1.7 unchanged, original stop/TP unchanged. Supply exact hashes and full 16-step+clause audit.',inputSchema:{type:'object',properties:{raw:{type:'string'},session:{type:'string'},policySha256:{type:'string'},snapshotSha256:{type:'string'},beforeLedgerSha256:{type:'string'},decisionKey:{type:'string'},analysis:{type:'object',description:'Full V1.7 decision: reportMarkdown; directions BTC_LONG/BTC_SHORT/ETH_LONG/ETH_SHORT each marketEvidence,counterEvidence,alternatives A-E; coupling; checks 1..16; clauses 0..23,A,B,FINAL; audit20; conclusions BTC/ETH {decision=LONG|SHORT|HOLD|MANAGE|UNVERIFIED,evidence}; historicalUncertainty (only when affected) {BTC|ETH:{unresolvedIds:[old IDs],evidence:string}}; newSignals[]; events[] with prior PENDING UNVERIFIED quarantine if a gap blocks historical state. Keep current decisions independent of unknown outcomes. Use parentSignalId and relationToExisting for add-on or opposite pending. Event ordering is chronological and exit must precede opposed ENTERED; revisions require observedPrice matching the current snapshot.'}},required:['raw','policySha256','snapshotSha256','beforeLedgerSha256','decisionKey','analysis']}},
 {name:'read_official_btc_eth_ledger',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'Retrieve NEW V1.7 source-exact official signal ledger and chronological immutable events for this session. Previous noncompliant legacy companion signals remain isolated, not silently imported.',inputSchema:{type:'object',properties:{session:{type:'string'}},required:[]}},
 {name:'read_official_v17_decision_history',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'Read-only chronological OFFICIAL_V17 AI decision history for missed-opportunity review. Never convert hypothetical candidates to historical trades or alter immutable signal outcomes.',inputSchema:{type:'object',properties:{session:{type:'string'},limit:{type:'integer',minimum:1,maximum:100},includeDirections:{type:'boolean'}},required:[]}},
 {name:'register_formal_btc_eth_signal',securitySchemes:[{type:'oauth2',scopes:['signals:write']}],description:'COMPATIBILITY path for previously installed plugins: use signal={kind:V17_OFFICIAL_SUBMISSION,raw,policySha256,snapshotSha256,beforeLedgerSha256,decisionKey,analysis} to publish complete original V1.7 AI analysis. Standalone signal writing is forbidden.',inputSchema:{type:'object',properties:{session:{type:'string'},signal:{type:'object'}},required:['signal']}}
];
async function mcp(req,res,principal){
 const b=await readJSON(req),id=b.id;
 if(b.method==='notifications/initialized'){res.writeHead(202);return res.end();}
 if(b.method==='initialize')return json(res,200,{jsonrpc:'2.0',id,result:{protocolVersion:'2025-03-26',capabilities:{tools:{listChanged:true}},serverInfo:{name:'v17-verbatim-official-registry',version:'1.0.0'}}});
 if(b.method==='ping')return json(res,200,{jsonrpc:'2.0',id,result:{}});
 if(b.method==='tools/list')return json(res,200,{jsonrpc:'2.0',id,result:{tools}});
 if(b.method!=='tools/call')return json(res,200,{jsonrpc:'2.0',id,error:{code:-32601,message:'Unknown method'}});
 const n=b.params?.name,args=b.params?.arguments||{};
 const granted=String(principal?.scope||'').split(/\s+/);
 if((n==='publish_official_v17_analysis'||n==='register_formal_btc_eth_signal')&&!granted.includes('signals:write'))return json(res,200,{jsonrpc:'2.0',id,result:{content:[{type:'text',text:'OAUTH_SCOPE_SIGNALS_WRITE_REQUIRED'}],isError:true}});
 if(!granted.includes('signals:read'))return json(res,200,{jsonrpc:'2.0',id,result:{content:[{type:'text',text:'OAUTH_SCOPE_SIGNALS_READ_REQUIRED'}],isError:true}});
 try{
  let result;
  if(n==='run_official_btc_eth_snapshot')result=await beginAnalysis(args);
  else if(n==='get_original_v17_full_prompt')result=await showPolicy();
  else if(n==='publish_official_v17_analysis')result=await publishAnalysis(args);
  else if(n==='read_official_btc_eth_ledger')result=await ledgerView(args);
  else if(n==='read_official_v17_decision_history')result=await decisionHistory(args);
  else if(n==='register_formal_btc_eth_signal'){
   // Backward-compatible path for existing installed ChatGPT plugins whose MCP tool list is cached.
   // Standalone signal creation remains forbidden: only a complete source-verified AI analysis may be published.
   if(args.signal?.kind!=='V17_OFFICIAL_SUBMISSION')throw Error('Standalone signal writing is forbidden. Supply signal.kind=V17_OFFICIAL_SUBMISSION with the complete V1.7 analysis envelope.');
   const payload={...args.signal,session:args.session||args.signal.session||'OFFICIAL_V17'};
   delete payload.kind;
   result=await publishAnalysis(payload);
  }
  else return json(res,200,{jsonrpc:'2.0',id,error:{code:-32601,message:'Unknown tool'}});
  return json(res,200,{jsonrpc:'2.0',id,result:{content:[{type:'text',text:JSON.stringify(result)}],isError:false}});
 }catch(e){return json(res,200,{jsonrpc:'2.0',id,result:{content:[{type:'text',text:String(e.message||e)}],isError:true}});}
}
const webPage=`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V1.7 공식 분석 서버</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:680px;margin:8vh auto;padding:20px;background:#0b1220;color:#eee}code{overflow-wrap:anywhere}a{color:#93c5fd}</style></head><body><h2>BTC·ETH MASTER PROMPT V1.7 공식 분석 서버</h2><p>이 서버는 원본 V1.7 전체를 그대로 전달하고, AI가 수행한 공식 분석을 최초 1회 저장해 다른 대화창에서 조회합니다.</p><p><strong>주의:</strong> 서버 자체는 매매 방향을 임의 계산하지 않습니다. ChatGPT 플러그인에서 자료를 전달하고 원문 기준 분석을 완료한 뒤 저장해야 공식 결과가 생성됩니다.</p><p>연결 주소: <code>/mcp</code></p><p>원본 SHA-256: <code>${OFFICIAL_POLICY.sha256}</code></p></body></html>`;
const server=http.createServer(async(req,res)=>{
 try{
  if(req.url==='/health'&&req.method==='GET')return json(res,200,{status:!DB_URL||!MASTER_SECRET?'not_ready':'ready',engine:VERSION,policySha256:OFFICIAL_POLICY.sha256,policyBytes:OFFICIAL_POLICY.bytes,mode:'AI_VERBATIM_POLICY_OFFICIAL_CACHE',dbConfigured:!!DB_URL,authenticationConfigured:!!MASTER_SECRET});
  if(req.url==='/'&&req.method==='GET'){res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store','Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'"});return res.end(webPage);}
  if(req.url.startsWith('/.well-known/')||req.url.startsWith('/oauth/')){const o=await getOAuth();const handled=await o.handle(req,res);if(handled!==false)return;}
  if(req.url==='/mcp'&&req.method==='POST'){
   const o=await getOAuth();const who=await o.authenticate(req);
   if(!who)return json(res,401,{error:'invalid_token'},{'WWW-Authenticate':o.challenge});
   // OAuth scopes enforced on mutating calls.
   if(String(req.headers['content-type']||'').includes('application/json')===false)return responseError(res,415,'JSON_REQUIRED');
   return await mcp(req,res,who);
  }
  if(req.url==='/mcp'&&req.method==='GET'){res.writeHead(405,{'Allow':'POST'});return res.end();}
  if(req.url.startsWith('/api/')){
   if(!isAuthorized(req))return responseError(res,401,'UNAUTHORIZED');
   if(req.method==='POST'&&['/api/prepare','/api/run'].includes(req.url))return json(res,200,await beginAnalysis(await readJSON(req)));
   if(req.method==='POST'&&req.url==='/api/publish')return json(res,200,await publishAnalysis(await readJSON(req)));
   if(req.method==='GET'&&req.url==='/api/ledger'){return json(res,200,await ledgerView({}));}
   if(req.method==='GET'&&req.url==='/api/policy')return json(res,200,await showPolicy());
   if(req.method==='GET'&&req.url==='/api/history')return json(res,200,await decisionHistory({}));
  }
  return responseError(res,404,'NOT_FOUND');
 }catch(e){return responseError(res,400,e.message||'BAD_REQUEST');}
});
if(process.env.V17_NO_LISTEN!=='1'){
 try{
  if(MODE==='cloud'&&(!DB_URL||!MASTER_SECRET))throw Error('CLOUD_REQUIRES_DATABASE_AND_API_SECRET');
  await getOAuth();server.listen(PORT,'0.0.0.0',()=>process.stderr.write('v17-verbatim-listening:'+PORT+'\n'));
 }catch(e){process.stderr.write('v17-startup-failed:'+e.message+'\n');process.exit(1);}
}
export {server,beginAnalysis,publishAnalysis,ledgerView,readLedger,decisionHistory,carriedUncertainty,tools};
