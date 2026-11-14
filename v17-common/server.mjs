import http from 'node:http';
import crypto from 'node:crypto';
import {makeOAuth} from './oauth.mjs';
import {applyCachedPolicyResponse} from './cached-policy-response.mjs';
import {buildSignalView,SIGNAL_TABLE_OUTPUT_RULE} from './signal-table-view.mjs';
import {storePreparedSnapshot,recoverOfficialSubmissionRaw} from './prepared-snapshot.mjs';
import {rawMarketWindows,recoverActiveSignalWindows,earliestUnknownGapOrigins} from './price-path-recovery.mjs';
import {planVerifiedMechanicalEvents,validateRecoveredProof} from './official-price-tracker.mjs';
import {assemblePreparedChains,planPreparedEvents} from './prepared-path-chain.mjs';
import {OFFICIAL_POLICY,VERSION,prepare,sourcePolicy,validateSnapshot,validateOfficialSubmission,ledgerHash,sessionId,digest,checkEvent} from './v17-core.mjs';
import {ATTEST_CONFIRM,TWO_USER_TP3_IDS,planUserAttestedTp3} from './user-attested-tp3.mjs';
import {USER_NO_ENTRY_KIND,USER_NO_ENTRY_CONFIRM,USER_NO_ENTRY_IDS,planUserNoEntryAttestation} from './user-attested-no-entry.mjs';

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
CREATE TABLE IF NOT EXISTS v17_ai_prepared_snapshots (
 session TEXT NOT NULL,
 snapshot_sha TEXT NOT NULL,
 raw TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 PRIMARY KEY(session,snapshot_sha)
);
CREATE TABLE IF NOT EXISTS v17_ai_gap_proofs (
 session TEXT NOT NULL,
 snapshot_sha TEXT NOT NULL,
 proof JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 PRIMARY KEY(session,snapshot_sha)
);
CREATE TABLE IF NOT EXISTS v17_ai_reset_archives (
 reset_id TEXT PRIMARY KEY,
 session TEXT NOT NULL,
 before_ledger_sha TEXT NOT NULL,
 signals JSONB NOT NULL,
 events JSONB NOT NULL,
 analyses JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
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
export const rowToLedger=r=>({id:r.id,symbol:r.symbol,original:r.original,status:r.status,verifiedUntil:r.verified_until,version:r.plan_version,userAccountEntry:r.user_account_entry??undefined,userAccountEntryAt:r.user_account_entry_at??undefined,unverifiedFrom:r.unverified_from??undefined,...(r.effective_plan?{effectivePlan:r.effective_plan,planEffectiveAt:r.plan_effective_at}:{})});
async function readLedger(db,session){
 const r=await db.query(`SELECT s.id,s.symbol,s.original,s.status,s.verified_until,s.plan_version,
 (SELECT e.event->>'priorConfirmedState' FROM v17_ai_events e
  WHERE e.session=s.session AND e.id=s.id AND e.event->>'kind'='UNVERIFIED'
  ORDER BY e.seq DESC LIMIT 1) AS unverified_from,
 (SELECT e.event->>'report' FROM v17_ai_events e
  WHERE e.session=s.session AND e.id=s.id AND e.event->>'kind'='USER_ACCOUNT_ENTRY_NOT_EXECUTED'
  ORDER BY e.seq DESC LIMIT 1) AS user_account_entry,
 (SELECT e.at_kst FROM v17_ai_events e
  WHERE e.session=s.session AND e.id=s.id AND e.event->>'kind'='USER_ACCOUNT_ENTRY_NOT_EXECUTED'
  ORDER BY e.seq DESC LIMIT 1) AS user_account_entry_at,
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
 // Verified gap-resolution events supersede older historical warnings.
 const recent=await db.query("SELECT DISTINCT ON(id) id,event FROM v17_ai_events WHERE session=$1 AND ((event->>'kind')='UNVERIFIED' OR event ? 'gapResolutionEvidence') ORDER BY id,seq DESC",[session]);
 const resolved=new Set(recent.rows.filter(x=>typeof x.event?.gapResolutionEvidence==='string'&&x.event.gapResolutionEvidence.length>0).map(x=>x.id));
 const r=await db.query("SELECT official->'evidenceQuality' AS evidence FROM v17_ai_analyses WHERE session=$1 AND official->'evidenceQuality' IS NOT NULL ORDER BY created_at DESC LIMIT 1000",[session]);
 for(const row of r.rows){
  for(const sym of ['BTC','ETH']){
   const q=row.evidence?.perSymbol?.[sym]||{};
   for(const id of [...(q.couldNotVerifySignals||[]),...(q.ambiguousSignals||[]),...(q.carriedUncertaintyIds||[])]){
    if(ids.has(id)&&!resolved.has(id)&&!result[sym].includes(id))result[sym].push(id);
   }
  }
 }
 return result;
}
// Resolve the first known gap origin for an already quarantined signal. A later
// complete candle cannot silently erase an old unresolved interval.
async function unverifiedOrigins(db,session,ledger){
 const wanted=ledger.filter(s=>['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(s.status));
 if(!wanted.length)return {};
 const r=await db.query("SELECT official->'evidenceQuality' AS evidence FROM v17_ai_analyses WHERE session=$1 AND official->'evidenceQuality' IS NOT NULL ORDER BY created_at DESC LIMIT 1000",[session]);
 // The failed 11:02→11:31 history must be retried even if the status still
 // says ENTERED. A new verified snapshot does not erase missing past minutes.
 return earliestUnknownGapOrigins(wanted,r.rows);
}
async function readVerifiedGapProof(db,session,snapshotSha){
 const r=await db.query('SELECT proof FROM v17_ai_gap_proofs WHERE session=$1 AND snapshot_sha=$2',[session,snapshotSha]);
 return r.rows[0]?.proof||{};
}
async function storeVerifiedGapProof(db,session,snapshotSha,proof){
 await db.query('INSERT INTO v17_ai_gap_proofs(session,snapshot_sha,proof) VALUES($1,$2,$3) ON CONFLICT(session,snapshot_sha) DO UPDATE SET proof=EXCLUDED.proof',[session,snapshotSha,JSON.stringify(proof)]);
}
// The only automatic transitions are verifiable mechanical level crossings.
// Discretionary entries, directions, revisions and early exits still belong to
// the AI and must pass the original V1.7 publication validator.
async function reconcileVerifiedPricePath(db,session,raw,snapshot,previousSnapshot){
 const at={BTC:snapshot.btcTime,ETH:snapshot.ethTime};
 const ledger=await readLedger(db,session);
 if(!ledger.some(s=>['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(s.status)))
  return {proof:{},events:[],notes:['NO_ACTIVE_SIGNAL'],backendRecoveryMs:0};
 const begin=Date.now(),origins=await unverifiedOrigins(db,session,ledger);
 const needIntermediate=Object.keys(origins).length>0||ledger.some(x=>x.status==='UNVERIFIED')||
  ['BTC','ETH'].some(sym=>{
   const prev=previousSnapshot?.[sym==='BTC'?'btcTime':'ethTime'];
   const path=rawMarketWindows(raw)[sym].pathStart;
   return prev&&(Date.parse(path.replace(' ','T')+'+09:00')-Date.parse(prev.replace(' ','T')+'+09:00')>60000);
  });
 let preparedRaws=[];
 if(needIntermediate){
  const q=await db.query("SELECT raw FROM v17_ai_prepared_snapshots WHERE session=$1 AND created_at>NOW()-INTERVAL '24 hours' ORDER BY created_at DESC LIMIT 160",[session]);
  preparedRaws=q.rows.map(x=>x.raw);
 }
 const base={...previousSnapshot};
 for(const s of ledger.filter(x=>origins[x.id])){
  const origin=origins[s.id],field=s.symbol==='BTCUSDT'?'btcTime':'ethTime';
  if(origin&&(!base[field]||origin<base[field]))base[field]=origin;
 }
 const prepared=assemblePreparedChains(preparedRaws,base,raw);
 // A prior stored official analysis already checked the path through its own timestamp.
 // Do not rescan earlier PENDING history or omit recent entry crosses.
 const auditStarts=Object.fromEntries(ledger.map(sig=>{
  const sym=sig.symbol==='BTCUSDT'?'BTC':'ETH';
  const previous=previousSnapshot?.[sym==='BTC'?'btcTime':'ethTime'];
  const from=sig.verifiedUntil||sig.original?.issuedAt;
  const ordinary=previous&&from?(previous>from?previous:from):previous||from;
  // A prior missing interval belongs to this exact original signal, and
  // must be replayed before any later TP/STOP event may be finalized.
  return [sig.id,origins[sig.id]||ordinary];
 }));
 let planned=planPreparedEvents({ledger,history:prepared,currentAtBySymbol:at,unverifiedOrigins:origins,auditStarts});
 const needsRemote=ledger.some(s=>['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(s.status)&&
  !prepared[s.symbol==='BTCUSDT'?'BTC':'ETH']?.recovered);
 let remote={BTC:{needed:false},ETH:{needed:false}};
 if(needsRemote){
  remote=await recoverActiveSignalWindows(raw,previousSnapshot,ledger,{unverifiedOrigins:origins});
  const remotePlan=planVerifiedMechanicalEvents({ledger,history:remote,currentAtBySymbol:at,unverifiedOrigins:origins,auditStarts});
  const preparedSymbols=new Set(['BTC','ETH'].filter(sym=>prepared[sym]?.recovered));
  for(const e of remotePlan.events){
   const row=ledger.find(x=>x.id===e.id);
   if(row&&!preparedSymbols.has(row.symbol==='BTCUSDT'?'BTC':'ETH'))planned.events.push(e);
  }
  planned.events.sort((x,y)=>Date.parse(x.at.replace(' ','T')+'+09:00')-Date.parse(y.at.replace(' ','T')+'+09:00'));
  planned.notes.push(...remotePlan.notes);
 }
 const sourceProofs=Object.fromEntries(['BTC','ETH'].map(sym=>[sym,prepared[sym]?.recovered?planned.proofs[sym]:{
  ...(remote[sym]?.recovered?{recovered:true,source:remote[sym].source,bridgeCovered:remote[sym].bridgeCovered,
    coveredFrom:remote[sym].coveredFrom,coveredThrough:remote[sym].coveredThrough,barCount:remote[sym].barCount}:{recovered:false})
 }]));
 const verified=validateRecoveredProof(raw,previousSnapshot,sourceProofs);
 const auditSummary=Object.fromEntries(['BTC','ETH'].map(sym=>[sym,{
  ...verified[sym],
  needed:ledger.some(s=>s.symbol===sym+'USDT'&&['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(s.status)),
  status:sourceProofs[sym]?.recovered?(sourceProofs[sym].source==='USER_PREPARED_SNAPSHOT_PATH_CHAIN'?'VERIFIED_PREPARED_CHAIN':'VERIFIED_SOURCE_1M'):'RECOVERY_UNAVAILABLE',
  reason:sourceProofs[sym]?.recovered?undefined:remote[sym]?.reason||'NO_CONTIGUOUS_PREPARED_PATH',
  coveredThrough:sourceProofs[sym]?.coveredThrough,
  barCount:sourceProofs[sym]?.barCount,
 }]));
 // No DB mutation on an unavailable, incomplete, or ambiguous source.
 if(!planned.events.length){
  return {proof:auditSummary,events:[],notes:planned.notes,backendRecoveryMs:Date.now()-begin};
 }
 const client=await db.connect();
 try{
  await client.query('BEGIN');
  await client.query('INSERT INTO v17_ai_sessions(name) VALUES($1) ON CONFLICT DO NOTHING',[session]);
  await client.query('SELECT name FROM v17_ai_sessions WHERE name=$1 FOR UPDATE',[session]);
  const locked=await readLedger(client,session);
  if(ledgerHash(locked)!==ledgerHash(ledger)){
   await client.query('ROLLBACK');
   return {proof:{},events:[],notes:['CONCURRENT_LEDGER_CHANGE_RETRY_PREPARE'],backendRecoveryMs:Date.now()-begin};
  }
  const staged=locked.map(x=>({...x}));
  for(const evt of planned.events){
   checkEvent(evt,staged);
   await client.query('INSERT INTO v17_ai_events(session,id,event,at_kst) VALUES($1,$2,$3,$4)',[session,evt.id,JSON.stringify(evt),evt.at]);
   await client.query('UPDATE v17_ai_signals SET status=$3,verified_until=$4 WHERE session=$1 AND id=$2',[session,evt.id,evt.kind,evt.at]);
   const row=staged.find(x=>x.id===evt.id);
   row.status=evt.kind;row.verifiedUntil=evt.at;
  }
  await client.query('COMMIT');
  return {proof:auditSummary,events:planned.events,notes:planned.notes,backendRecoveryMs:Date.now()-begin};
 }catch(err){
  try{await client.query('ROLLBACK');}catch{}
  // Never mutate the official ledger on unexpected validation/database errors.
  return {proof:{},events:[],notes:['AUTO_TRACKER_ROLLBACK:'+String(err.message||err).slice(0,160)],backendRecoveryMs:Date.now()-begin};
 }finally{client.release();}
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
 const startMs=Date.now();
 const raw=body.raw,session=sessionId(body.session||'OFFICIAL_V17');
 const snapshot=validateSnapshot(raw);const db=await database();
 // The pasted source is permanent and byte-exact. Exchange OHLC is supplementary proof.
 await storePreparedSnapshot(db,session,snapshot.sha256,raw);
 const oldOfficial=await latestOfficialSnapshot(db,session);
 const outOfOrder=Boolean(oldOfficial&&(snapshot.btcTime<oldOfficial.btcTime||snapshot.ethTime<oldOfficial.ethTime));
 let tracked={proof:{},events:[],notes:outOfOrder?['OUT_OF_ORDER_NO_AUTO_MUTATION']:[],backendRecoveryMs:0};
 if(!outOfOrder){
  try{tracked=await reconcileVerifiedPricePath(db,session,raw,snapshot,oldOfficial);}
  catch(e){tracked={proof:{},events:[],notes:['RECOVERY_ERROR_NO_LEDGER_MUTATION:'+String(e.message||e).slice(0,160)],backendRecoveryMs:0};}
 }
 // Concurrent independent reads preserve original official registry behavior.
 const [ledger,previousSnapshot]=await Promise.all([
  readLedger(db,session),latestOfficialSnapshot(db,session)
 ]);
 const [saved,carry]=await Promise.all([
  selectOfficial(db,session,snapshot.sha256,ledgerHash(ledger)),
  carriedUncertainty(db,session,ledger)
 ]);
 await storeVerifiedGapProof(db,session,snapshot.sha256,tracked.proof);
 const full=prepare({raw,session,ledger,saved,previousSnapshot,carriedUncertaintyBySymbol:carry,verifiedGapRecoveryBySymbol:tracked.proof});
 const delivered={...applyCachedPolicyResponse(full,body.cachedPolicySha256),
  signalView:buildSignalView({ledger,snapshot,changes:tracked.events}),
  userOutputRule:SIGNAL_TABLE_OUTPUT_RULE,
  preparedRaw:{persisted:true,snapshotSha256:snapshot.sha256,submitWithoutRaw:true},
  automaticTracking:{events:tracked.events.map(e=>({id:e.id,kind:e.kind,at:e.at,source:e.source||'VERIFIED_GAP_RECOVERY'})),
   sourceBySymbol:Object.fromEntries(['BTC','ETH'].map(sym=>[sym,{status:tracked.proof[sym]?.status||'NOT_NEEDED',coveredThrough:tracked.proof[sym]?.coveredThrough,reason:tracked.proof[sym]?.reason}])),
   recoveryMs:tracked.backendRecoveryMs,notes:tracked.notes.slice(0,8)}};
 if(body.debugTimings!==true)return delivered;
 return {...delivered,serverTimings:{
  backendMs:Date.now()-startMs,recoveryMs:tracked.backendRecoveryMs,
  fullBytes:Buffer.byteLength(JSON.stringify(full),'utf8'),
  deliveredBytes:Buffer.byteLength(JSON.stringify(delivered),'utf8'),
  policyTextOmitted:delivered.policyCache?.used===true,
 }};
}
async function publishAnalysis(body){
 const session=sessionId(body.session||'OFFICIAL_V17');
 const db=await database(),client=await db.connect();
 try{
  await client.query('BEGIN');
  await client.query('INSERT INTO v17_ai_sessions(name) VALUES($1) ON CONFLICT DO NOTHING',[session]);
  await client.query('SELECT name FROM v17_ai_sessions WHERE name=$1 FOR UPDATE',[session]);
  // The SHA256 reference is scoped to the official session and resolves to
  // the exact text first received by prepare; no AI reconstruction is needed.
  const raw=await recoverOfficialSubmissionRaw(client,session,body);
  const shot=validateSnapshot(raw);
  const ledger=await readLedger(client,session);
  const before=ledgerHash(ledger);
  // At most one official result per snapshot and current ledger state.
  const prior=await selectOfficial(client,session,shot.sha256,before);
  if(prior){await client.query('COMMIT');return {status:'OFFICIAL_REPLAY',officialDecision:{...prior,signalView:prior.signalView||buildSignalView({ledger,snapshot:shot}),userOutputRule:SIGNAL_TABLE_OUTPUT_RULE},sameState:true,notReanalyzed:true};}
  const previousSnapshot=await latestOfficialSnapshot(client,session);
  const carry=await carriedUncertainty(client,session,ledger);
  const verifiedGapRecoveryBySymbol=await readVerifiedGapProof(client,session,shot.sha256);
  const verified=validateOfficialSubmission({...body,raw},ledger,previousSnapshot,carry,verifiedGapRecoveryBySymbol);
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
  const afterLedger=await readLedger(client,session);
  const after=ledgerHash(afterLedger);
  const signalView=buildSignalView({ledger:afterLedger,snapshot:shot,changes:[...newSignals,...events]});
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
   signalView,
   userOutputRule:SIGNAL_TABLE_OUTPUT_RULE,
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
// User account execution status is stored separately from exchange price-touch verification.
export async function attestUserNoAccountEntry(args={}){
 const session=sessionId(args.session||'OFFICIAL_V17');
 if(session!=='OFFICIAL_V17'||args.kind!==USER_NO_ENTRY_KIND)throw Error('USER_NO_ENTRY_SCOPE_FORBIDDEN');
 if(args.confirmation!==USER_NO_ENTRY_CONFIRM)throw Error('USER_NO_ENTRY_CONFIRMATION_REQUIRED');
 if(!Array.isArray(args.signalIds)||args.signalIds.length!==2||new Set(args.signalIds).size!==2||
    !USER_NO_ENTRY_IDS.every(id=>args.signalIds.includes(id)))throw Error('USER_NO_ENTRY_EXACT_ORIGINAL_IDS_REQUIRED');
 if(typeof args.expectedLedgerSha256!=='string'||!/^[a-f0-9]{64}$/.test(args.expectedLedgerSha256))
  throw Error('USER_NO_ENTRY_LEDGER_SHA_REQUIRED');
 const db=await database(),client=await db.connect();
 try{
  await client.query('BEGIN');
  await client.query('SELECT name FROM v17_ai_sessions WHERE name=$1 FOR UPDATE',[session]);
  const beforeRows=await readLedger(client,session),before=ledgerHash(beforeRows);
  if(args.expectedLedgerSha256!==before)throw Error('USER_NO_ENTRY_LEDGER_CHANGED_RETRY');
  if(USER_NO_ENTRY_IDS.every(id=>{
   const row=beforeRows.find(s=>s.id===id);
   return row?.status==='UNVERIFIED'&&row?.unverifiedFrom==='PENDING'&&row.userAccountEntry==='NOT_ENTERED';
  })){
   const snapshot=await latestOfficialSnapshot(client,session);
   await client.query('COMMIT');
   return {status:'OFFICIAL_USER_NO_ENTRY_REPLAY',session,ids:USER_NO_ENTRY_IDS,
    afterLedgerSha256:before,accountEntry:'NOT_ENTERED',historicalMarketTouch:'UNVERIFIED',
    signalView:buildSignalView({ledger:beforeRows,snapshot})};
  }
  if(USER_NO_ENTRY_IDS.some(id=>beforeRows.find(s=>s.id===id)?.userAccountEntry==='NOT_ENTERED'))
   throw Error('USER_NO_ENTRY_PARTIAL_CONFLICT_REQUIRES_REVIEW');
  const time=await client.query("SELECT to_char(NOW() AT TIME ZONE 'Asia/Seoul','YYYY-MM-DD HH24:MI:SS') AS at");
  const events=planUserNoEntryAttestation({session,kind:args.kind,confirmation:args.confirmation,
   signalIds:args.signalIds,ledger:beforeRows,at:time.rows[0].at});
  for(const e of events){
   await client.query('INSERT INTO v17_ai_events(session,id,event,at_kst) VALUES($1,$2,$3,$4)',
    [session,e.id,JSON.stringify(e),e.at]);
  }
  const afterRows=await readLedger(client,session);
  if(ledgerHash(afterRows)!==before)throw Error('USER_NO_ENTRY_FORMAL_LEDGER_MUTATED_ROLLBACK');
  if(!USER_NO_ENTRY_IDS.every(id=>afterRows.find(s=>s.id===id)?.userAccountEntry==='NOT_ENTERED'))
   throw Error('USER_NO_ENTRY_PERSISTENCE_CHECK_FAILED');
  const snapshot=await latestOfficialSnapshot(client,session);
  await client.query('COMMIT');
  return {status:'OFFICIAL_USER_NO_ENTRY_STORED',session,ids:USER_NO_ENTRY_IDS,
   afterLedgerSha256:before,accountEntry:'NOT_ENTERED',historicalMarketTouch:'UNVERIFIED',
   formalSignalStatusUnchanged:true,events,signalView:buildSignalView({ledger:afterRows,snapshot})};
 }catch(e){try{await client.query('ROLLBACK');}catch{}throw e;}finally{client.release();}
}

// Narrow, explicit user-confirmed closure of the two November 14 originals.
// This does NOT weaken checkEvent() or verified chronological tracking.
export async function closeTwoUserAttestedTp3(args={}){
 const session=sessionId(args.session||'OFFICIAL_V17');
 if(session!=='OFFICIAL_V17'||args.kind!=='V17_USER_ATTESTED_TP3_TWO_ONLY')
  throw Error('ATTEST_SCOPE_FORBIDDEN');
 if(args.confirmation!==ATTEST_CONFIRM)throw Error('USER_TP3_CONFIRMATION_REQUIRED');
 const sha='9932522f124bfb0c5350f9e7acc1058d6472a5fbd5a6e6f7ed639d42dba8a2dc';
 if(args.sourceSnapshotSha256!==sha)throw Error('ATTEST_USE_EXACT_SAVED_NOV14_SNAPSHOT');
 const db=await database(),client=await db.connect();
 try{
  await client.query('BEGIN');
  await client.query('SELECT name FROM v17_ai_sessions WHERE name=$1 FOR UPDATE',[session]);
  const ledger=await readLedger(client,session),before=ledgerHash(ledger);
  const already=ledger.filter(x=>TWO_USER_TP3_IDS.includes(x.id)&&x.status==='TP3_DONE');
  if(already.length===2){
   const events=await client.query(
    "SELECT id,event FROM v17_ai_events WHERE session=$1 AND id=ANY($2::text[]) AND event->>'provenance'='USER_ATTESTED_NOT_CHRONOLOGICALLY_VERIFIED' ORDER BY seq DESC",
    [session,TWO_USER_TP3_IDS]);
   if(new Set(events.rows.map(x=>x.id)).size===2){
    await client.query('COMMIT');
    return {status:'OFFICIAL_USER_ATTESTED_TP3_REPLAY',session,ids:TWO_USER_TP3_IDS,
     afterLedgerSha256:before,events:events.rows.slice(0,2).map(x=>x.event),
     source:'USER_ATTESTED_NOT_CHRONOLOGICALLY_VERIFIED',
     signalView:buildSignalView({ledger,snapshot:await latestOfficialSnapshot(client,session)})};
   }
  }
  const saved=await client.query(
   'SELECT raw FROM v17_ai_prepared_snapshots WHERE session=$1 AND snapshot_sha=$2 LIMIT 1',
   [session,sha]);
  const fallback=saved.rows.length?saved:await client.query(
   'SELECT raw FROM v17_ai_analyses WHERE session=$1 AND snapshot_sha=$2 ORDER BY created_at DESC LIMIT 1',
   [session,sha]);
  if(fallback.rows.length!==1)throw Error('ATTEST_IMMUTABLE_SOURCE_SNAPSHOT_NOT_FOUND');
  const now=await client.query("SELECT to_char(NOW() AT TIME ZONE 'Asia/Seoul','YYYY-MM-DD HH24:MI:SS') AS at");
  const events=planUserAttestedTp3({
   ledger,sourceRaw:fallback.rows[0].raw,sourceSnapshotSha256:sha,
   expectedLedgerSha256:args.expectedLedgerSha256,
   signalIds:args.signalIds,confirmation:args.confirmation,at:now.rows[0].at
  });
  if(args.expectedLedgerSha256!==before)throw Error('ATTEST_LEDGER_CHANGED_RETRY');
  for(const event of events){
   await client.query('INSERT INTO v17_ai_events(session,id,event,at_kst) VALUES($1,$2,$3,$4)',
    [session,event.id,JSON.stringify(event),event.at]);
   const update=await client.query(
    "UPDATE v17_ai_signals SET status='TP3_DONE', verified_until=$3 WHERE session=$1 AND id=$2 AND status IN ('ENTERED','TP1','TP2')",
    [session,event.id,event.at]);
   if(update.rowCount!==1)throw Error('ATTEST_CONCURRENT_SIGNAL_CHANGE_ROLLBACK');
  }
  const after=await readLedger(client,session);
  if(events.some(x=>after.find(s=>s.id===x.id)?.status!=='TP3_DONE'))
   throw Error('ATTEST_CLOSURE_NOT_PERSISTED');
  await client.query('COMMIT');
  return {status:'OFFICIAL_USER_ATTESTED_TP3_STORED',session,ids:TWO_USER_TP3_IDS,
   beforeLedgerSha256:before,afterLedgerSha256:ledgerHash(after),events,
   source:'USER_ATTESTED_NOT_CHRONOLOGICALLY_VERIFIED',independentStopOrderVerified:false,
   signalView:buildSignalView({ledger:after,snapshot:await latestOfficialSnapshot(db,session),changes:events})};
 }catch(e){try{await client.query('ROLLBACK')}catch{}throw e;}
 finally{client.release();}
}

// Administrative one-time full reset of the default official ledger only.
// Preserve an immutable database backup before deleting live signals, events and
// snapshot analyses, so prior snapshots cannot rehydrate the old state.
export async function resetOfficialLedger(args={}) {
 if(args.session!=='OFFICIAL_V17')throw Error('RESET_DEFAULT_OFFICIAL_SESSION_ONLY');
 const session=sessionId(args.session);
 if(args.kind!=='V17_RESET_ALL_OFFICIAL')throw Error('RESET_KIND_REQUIRED');
 if(args.confirmation!=='RESET_ALL_OFFICIAL_V17_2026_11_13')throw Error('RESET_EXPLICIT_CONFIRMATION_REQUIRED');
 if(!/^[0-9a-f]{64}$/.test(args.expectedLedgerSha256||''))throw Error('RESET_EXPECTED_LEDGER_SHA_REQUIRED');
 const db=await database(),client=await db.connect();
 try {
  await client.query('BEGIN');
  await client.query('INSERT INTO v17_ai_sessions(name) VALUES($1) ON CONFLICT DO NOTHING',[session]);
  await client.query('SELECT name FROM v17_ai_sessions WHERE name=$1 FOR UPDATE',[session]);
  const ledger=await readLedger(client,session);
  const before=ledgerHash(ledger);
  if(before!==args.expectedLedgerSha256)throw Error('RESET_LEDGER_CHANGED_ABORT');
  const backup=await client.query(String.raw`SELECT
   (SELECT COALESCE(jsonb_agg(to_jsonb(s) ORDER BY s.id),'[]'::jsonb) FROM v17_ai_signals s WHERE s.session=$1) AS signals,
   (SELECT COALESCE(jsonb_agg(to_jsonb(e) ORDER BY e.seq),'[]'::jsonb) FROM v17_ai_events e WHERE e.session=$1) AS events,
   (SELECT COALESCE(jsonb_agg(to_jsonb(a) ORDER BY a.created_at),'[]'::jsonb) FROM v17_ai_analyses a WHERE a.session=$1) AS analyses`,[session]);
  const copy=backup.rows[0];
  const resetId='V17_RESET_20261113_'+before.slice(0,16);
  await client.query('INSERT INTO v17_ai_reset_archives(reset_id,session,before_ledger_sha,signals,events,analyses) VALUES($1,$2,$3,$4,$5,$6)',
   [resetId,session,before,JSON.stringify(copy.signals),JSON.stringify(copy.events),JSON.stringify(copy.analyses)]);
  const analyses=await client.query('DELETE FROM v17_ai_analyses WHERE session=$1',[session]);
  const events=await client.query('DELETE FROM v17_ai_events WHERE session=$1',[session]);
  const signals=await client.query('DELETE FROM v17_ai_signals WHERE session=$1',[session]);
  const after=await readLedger(client,session);
  const remain=await client.query(String.raw`SELECT
   (SELECT COUNT(*)::integer FROM v17_ai_signals WHERE session=$1) AS signals,
   (SELECT COUNT(*)::integer FROM v17_ai_events WHERE session=$1) AS events,
   (SELECT COUNT(*)::integer FROM v17_ai_analyses WHERE session=$1) AS analyses`,[session]);
  if(after.length||Object.values(remain.rows[0]).some(x=>x!==0))throw Error('RESET_VERIFICATION_NONEMPTY_ROLLBACK');
  await client.query('COMMIT');
  return {status:'OFFICIAL_FULL_RESET',session,archiveId:resetId,archivedCounts:{signals:copy.signals.length,events:copy.events.length,analyses:copy.analyses.length},
   deletedCounts:{signals:signals.rowCount,events:events.rowCount,analyses:analyses.rowCount},
   beforeLedgerSha256:before,afterLedgerSha256:ledgerHash(after),remaining:remain.rows[0],sourcePolicySha256:OFFICIAL_POLICY.sha256};
 }catch(e){try{await client.query('ROLLBACK')}catch{}throw e}finally{client.release()}
}
async function ledgerView(args){const session=sessionId(args.session||'OFFICIAL_V17'),db=await database();
 const [ledger,snapshot]=await Promise.all([readLedger(db,session),latestOfficialSnapshot(db,session)]);
 return {session,policy:OFFICIAL_POLICY,ledgerSha256:ledgerHash(ledger),signals:ledger,
  signalView:buildSignalView({ledger,snapshot}),userOutputRule:SIGNAL_TABLE_OUTPUT_RULE,
  events:(await db.query('SELECT id,event,at_kst FROM v17_ai_events WHERE session=$1 ORDER BY seq DESC LIMIT 500',[session])).rows};}
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
const requiredSchema={type:'object',properties:{raw:{type:'string',description:'Verbatim BTC and ETH snapshots, as pasted by the user'},session:{type:'string',description:'Use shared OFFICIAL_V17 for a single official signal ledger. Separate arbitrary sessions must not import old signals.'},cachedPolicySha256:{type:'string',description:'OPTIONAL SPEED OPTIMIZATION. Supply exact 64-character V1.7 SHA-256 ONLY when the complete unchanged 902-line original prompt has ALREADY been read in this same conversation. The server then omits only the repeated verbatim policy text; active ledger, path evidence, mandatory four directions, full 16 checks/27 clauses and official publication remain unchanged. Never invent a hash. Wrong hashes safely return full policy.'},debugTimings:{type:'boolean',description:'Optional read-only server timing diagnostic. Does not affect trade decisions or ledger.'}},required:['raw']};
const tools=[
 {name:'run_official_btc_eth_snapshot',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'FIRST STEP (MANDATORY). Fetch entire unchanged V1.7 source, official ledger and evidenceQuality. For faster repeated analyses in the SAME conversation only, pass cachedPolicySha256 equal to the exactly previously read full 902-line prompt hash: this omits solely duplicate policy text while retaining full ledger/evidence and ALL original rules, decisions, A-E, 16 checks, 27 clauses and publication validation. If no full prompt has been read in this conversation omit cachedPolicySha256 and receive ENTIRE verbatim source. If the cached hash mismatches the server sends the full source. The initial run response already includes full ledger, so do NOT make a redundant read_official_btc_eth_ledger call unless troubleshooting the ledger. No server-side trade-model shortcuts. If any PENDING signal exists, use the returned autoSwitchContext from THIS snapshot and provide numeric analysis.autoSwitchAudit for each pending ID: recompute current-price and nearer-source-anchored candidate entry/SL/TP1-3 and TP2 RR, or document the missing structural leg with an observed source anchor. CVD LOW alone is not a sufficient HOLD rejection. Include carried historical unknown periods. If OFFICIAL_REPLAY show officialDecision.reportMarkdown verbatim. Analyze current LONG/SHORT and A-E independently of unknown old trade outcome; preserve and report historicalUnknownIds separately, with no guessed fills or flat positions. Old pending can be quarantined with documented UNVERIFIED event before prospective replacement if it cannot be resolved. AUTO-SWITCH: on each latest snapshot compare existing KEEP, old pending redesign (terminate old then mint complete new ID), same-direction ADD_ON, opposite-side PENDING switch candidate, verified EARLY_EXIT or REVERSE_EXIT, and prospective REVISION. Same-direction and opposite pending can coexist with a proven entered trade when evidence and metadata meet policy; never claim both sides entered. Publish via full original V1.7 analysis. Never bypass risk/exposure checks.',inputSchema:requiredSchema},
 {name:'get_original_v17_full_prompt',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'Fetch EXACT full 902-line user-supplied V1.7 source, CRLF and original punctuation preserved. Must be the sole authoritative strategy; no new indicator thresholds, no removed functionality.',inputSchema:{type:'object',properties:{},additionalProperties:false}},
 {name:'publish_official_v17_analysis',securitySchemes:[{type:'oauth2',scopes:['signals:write']}],description:'SECOND STEP after NEEDS_AI_ANALYSIS. Existing confirmed price-touch and stop/TP event checks still mandatory. IMPORTANT GAP FIX: unknown historical signal outcome does NOT force current BTC/ETH conclusion=UNVERIFIED or block fresh independent market analysis. Include analysis.historicalUncertainty.{BTC|ETH}={unresolvedIds:[...],evidence:string} for each affected symbol; do not claim old position closed or account flat. To replace old PENDING with unresolved gap, include an old-ID UNVERIFIED quarantine event (uncertaintyScope,priorConfirmedState,marketReassessment, at current snapshot time) in same atomic submission, and new formal plan historicalExposureUnknownIds, historicalExposureRiskDisclosure, independentCurrentMarketEvidence. New plan is an analytical signal, NOT exchange account position certainty. An old ENTERED can coexist with separately justified same-direction ADD_ON or opposite PENDING switch plan; new signal must identify parentSignalId, relationToExisting SAME_DIRECTION_ADD_ON with addOnRiskEvidence and existingPositionRiskEvidence, OR OPPOSITE_PENDING with transitionOnTouchEvidence and existingPositionRiskEvidence. Old proven ENTERED cannot be silently replaced or counted flat; any opposite ENTERED event needs all opposed positions to have verified exit events first. REVERSE_EXIT/EARLY_EXIT requires strongCounterEvidence, structureEvidence, flowEvidence, contextEvidence, and outcomeUnverified=true if no verified entry/exit prices; actual PnL requires verifiedEntryPrice and entryPriceEvidence matching calculated PnL. REVISION event requires observedPrice=current snapshot, new plan stop/TP1/TP2/TP3, immutable previous achieved milestones and tighter effective stop; revised plan is read from ledger on following runs. Never retroactively cancel or claim TP/SL in gaps. Four directions A-E, original V1.7 unchanged, original stop/TP unchanged. For live PENDING with current ATR/EMA structure, supply analysis.autoSwitchAudit.BTC/ETH as described in run response requirements, with each old ID, observed distance in ATR/percent, numeric immediate+near alternatives, source anchors, stop/TP/RR and quantified KEEP rejection or atomic REDESIGN. Prose-only HOLD or LOW CVD-only justification is rejected. Supply exact hashes and full 16-step+clause audit.',inputSchema:{type:'object',properties:{raw:{type:'string'},session:{type:'string'},policySha256:{type:'string'},snapshotSha256:{type:'string'},beforeLedgerSha256:{type:'string'},decisionKey:{type:'string'},analysis:{type:'object',description:'Full V1.7 decision: reportMarkdown; directions BTC_LONG/BTC_SHORT/ETH_LONG/ETH_SHORT each marketEvidence,counterEvidence,alternatives A-E; coupling; checks 1..16; clauses 0..23,A,B,FINAL; audit20; conclusions BTC/ETH {decision=LONG|SHORT|HOLD|MANAGE|UNVERIFIED,evidence}; historicalUncertainty (only when affected) {BTC|ETH:{unresolvedIds:[old IDs],evidence:string}}; newSignals[]; events[] with prior PENDING UNVERIFIED quarantine if a gap blocks historical state. Keep current decisions independent of unknown outcomes. Use parentSignalId and relationToExisting for add-on or opposite pending. Event ordering is chronological and exit must precede opposed ENTERED; revisions require observedPrice matching the current snapshot.'}},required:['raw','policySha256','snapshotSha256','beforeLedgerSha256','decisionKey','analysis']}},
 {name:'read_official_btc_eth_ledger',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'Retrieve NEW V1.7 source-exact official signal ledger and chronological immutable events for this session. Previous noncompliant legacy companion signals remain isolated, not silently imported.',inputSchema:{type:'object',properties:{session:{type:'string'}},required:[]}},
 {name:'read_official_v17_decision_history',securitySchemes:[{type:'oauth2',scopes:['signals:read']}],description:'Read-only chronological OFFICIAL_V17 AI decision history for missed-opportunity review. Never convert hypothetical candidates to historical trades or alter immutable signal outcomes.',inputSchema:{type:'object',properties:{session:{type:'string'},limit:{type:'integer',minimum:1,maximum:100},includeDirections:{type:'boolean'}},required:[]}},
 {name:'register_formal_btc_eth_signal',securitySchemes:[{type:'oauth2',scopes:['signals:write']}],description:'COMPATIBILITY path for installed plugins: call run_official_btc_eth_snapshot ONCE with exact user raw. Then submit signal={kind:V17_OFFICIAL_SUBMISSION,policySha256,snapshotSha256,beforeLedgerSha256,decisionKey,analysis} WITHOUT raw; server recovers the identical prepared raw securely by session plus SHA. Older raw submissions remain supported. Standalone signal writing is forbidden. Include autoSwitchAudit for live pending entries. For explicit user-confirmed account nonentry only, use signal.kind=V17_USER_ATTESTED_NO_ENTRY_TWO_ONLY with confirmation=USER_CONFIRMS_BTC_AND_ETH_NO_ACCOUNT_ENTRY, both exact 2026-11-14 signal IDs and expectedLedgerSha256 from ledger view; historical price-touch remains unverified.',inputSchema:{type:'object',properties:{session:{type:'string'},signal:{type:'object'}},required:['signal']}}
];
// This is a user-facing output preference only; core V1.7 analysis,
// validator, safety audits, and immutable signal state are unchanged.
for(const t of tools){
 if(['run_official_btc_eth_snapshot','publish_official_v17_analysis','register_formal_btc_eth_signal','read_official_btc_eth_ledger'].includes(t.name)){
  const display='USER DISPLAY CONTRACT FOR ALL CHATS: after official analysis is stored, show ONLY officialDecision.signalView.markdown (or the top-level signalView.markdown). It includes every nonterminal BTC/ETH original signal including UNVERIFIED, entry-relative stop/TP1-3 % and no unconfirmed PnL. Append signalView.changeNotices ONLY on an actual auto-entry, cancellation, addition, redesign, switch, or exit. Do not show commentary, market narrative, checks, or reportMarkdown in the final answer. Perform all original V1.7 calculations and validations internally before rendering the table. ';
  t.description=display+t.description.replace('If OFFICIAL_REPLAY show officialDecision.reportMarkdown verbatim.','If OFFICIAL_REPLAY show officialDecision.signalView.markdown only.');
 }
}
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
   if(args.signal?.kind==='V17_USER_ATTESTED_TP3_TWO_ONLY'){
    result=await closeTwoUserAttestedTp3({...args.signal,session:args.session||args.signal.session});
   }else if(args.signal?.kind==='V17_RESET_ALL_OFFICIAL'){
    result=await resetOfficialLedger({...args.signal,session:args.session||args.signal.session});
   }else if(args.signal?.kind===USER_NO_ENTRY_KIND){
    result=await attestUserNoAccountEntry({...args.signal,session:args.session||args.signal.session});
   }else{
   if(args.signal?.kind!=='V17_OFFICIAL_SUBMISSION')throw Error('Standalone signal writing is forbidden. Supply signal.kind=V17_OFFICIAL_SUBMISSION with the complete V1.7 analysis envelope.');
   const payload={...args.signal,session:args.session||args.signal.session||'OFFICIAL_V17'};
   delete payload.kind;
   result=await publishAnalysis(payload);
   }
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
