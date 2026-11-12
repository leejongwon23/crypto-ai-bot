import http from 'node:http';
import crypto from 'node:crypto';
import {analyze,parse,sha,stable,verifySignal} from './engine.mjs';

const MODE=process.env.DEPLOYMENT_MODE||'local';
const SECRET=process.env.SIGNAL_API_KEY||'';
const DB_URL=process.env.DATABASE_URL||'';
const PORT=Number(process.env.PORT||3000);
const MAX=500000;
let pool;
const liveMemory=new Map();
function guarded(){if(MODE==='cloud'&&(!SECRET||!DB_URL))throw Error('production requires API key and durable database')}
function isAuthorized(req){
 if(!SECRET)return MODE!=='cloud';
 const key=(req.headers.authorization||'').replace(/^Bearer\s+/i,'')||(req.headers['x-api-key']||'');
 if(typeof key!=='string'||key.length!==SECRET.length)return false;
 return crypto.timingSafeEqual(Buffer.from(key),Buffer.from(SECRET));
}
function json(res,code,body,headers={}){res.writeHead(code,{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff',...headers});res.end(JSON.stringify(body));}
function fail(res,code,message){json(res,code,{error:message});}
async function readJSON(req){let size=0,parts=[];for await(const b of req){size+=b.length;if(size>MAX)throw Error('request exceeds 500KB');parts.push(b);}const s=Buffer.concat(parts).toString('utf8');return s?JSON.parse(s):{};}
const DB_SCHEMA=`CREATE TABLE IF NOT EXISTS v17_sessions (name TEXT PRIMARY KEY, updated_at TIMESTAMPTZ DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v17_signals (session TEXT NOT NULL,id TEXT NOT NULL,original JSONB NOT NULL,status TEXT NOT NULL,verified_until TEXT,PRIMARY KEY(session,id));
CREATE TABLE IF NOT EXISTS v17_decisions (session TEXT NOT NULL,raw_sha TEXT NOT NULL,before_sha TEXT NOT NULL,after_sha TEXT NOT NULL,report JSONB NOT NULL,created_at TIMESTAMPTZ DEFAULT NOW(),PRIMARY KEY(session,raw_sha,before_sha));`;
async function database(){
 if(pool)return pool;
 if(!DB_URL){if(MODE==='cloud')throw Error('database not configured');return null;}
 const {Pool}=await import('pg');
 pool=new Pool({connectionString:DB_URL,ssl:DB_URL.includes('localhost')?false:{rejectUnauthorized:false},max:3,connectionTimeoutMillis:10000});
 await pool.query(DB_SCHEMA);
 return pool;
}
function standardSession(x){if(typeof x!=='string')return 'OFFICIAL_V17';if(!/^[A-Z0-9_\-]{1,80}$/i.test(x))throw Error('invalid session identifier');return x;}
async function getLedger(session){
 const db=await database();if(!db)return liveMemory.get(session)||[];
 const r=await db.query('SELECT id,original,status,verified_until FROM v17_signals WHERE session=$1 ORDER BY id',[session]);
 return r.rows.map(v=>({...v.original,status:v.status,verifiedUntil:v.verified_until}));
}
async function addSignal(payload){
 const session=standardSession(payload.session||'OFFICIAL_V17');const source=payload.signal;
 verifySignal(source);const s={...source};delete s.status;delete s.verifiedUntil;
 if(!['PENDING','ENTERED','TP1','TP2','STOPPED','TP3_DONE','UNKNOWN'].includes(source.status||'PENDING'))throw Error('invalid formal state');
 const db=await database();
 if(!db){const l=liveMemory.get(session)||[];const other=l.find(x=>x.id===source.id);if(other&&stable({...other,status:undefined,verifiedUntil:undefined})!==stable({...source,status:undefined,verifiedUntil:undefined}))throw Error('FORMAL_ORIGINAL_LOCKED');if(!other){l.push(source);liveMemory.set(session,l);}return {ok:true,id:source.id,session};}
 const client=await db.connect();try{
  await client.query('BEGIN');await client.query('INSERT INTO v17_sessions(name) VALUES($1) ON CONFLICT DO NOTHING',[session]);await client.query('SELECT name FROM v17_sessions WHERE name=$1 FOR UPDATE',[session]);
  const old=await client.query('SELECT original FROM v17_signals WHERE session=$1 AND id=$2',[session,source.id]);
  if(old.rows.length&&stable(old.rows[0].original)!==stable(s))throw Error('FORMAL_ORIGINAL_LOCKED');
  if(!old.rows.length)await client.query('INSERT INTO v17_signals(session,id,original,status,verified_until) VALUES($1,$2,$3,$4,$5)',[session,source.id,s,source.status||'PENDING',source.verifiedUntil||source.issuedAt]);
  await client.query('COMMIT');return {ok:true,id:source.id,session};
 }catch(e){await client.query('ROLLBACK');throw e;}finally{client.release();}
}
async function compute(payload){
 const raw=payload.raw;if(typeof raw!=='string')throw Error('raw snapshot required');const markets=parse(raw);const session=standardSession(payload.session||'OFFICIAL_V17');
 const db=await database();if(!db){const ledger=await getLedger(session);return {session,rawSha:sha(raw),official:analyze(raw,ledger),ledger:ledger.map(x=>({id:x.id,status:x.status})),consistency:'LOCAL_ONLY_EPHEMERAL'};}
 const client=await db.connect();try{
  await client.query('BEGIN');await client.query('INSERT INTO v17_sessions(name) VALUES($1) ON CONFLICT DO NOTHING',[session]);await client.query('SELECT name FROM v17_sessions WHERE name=$1 FOR UPDATE',[session]);
  const r=await client.query('SELECT original,status,verified_until FROM v17_signals WHERE session=$1 ORDER BY id',[session]);
  const ledger=r.rows.map(x=>({...x.original,status:x.status,verifiedUntil:x.verified_until}));
  const before=sha(ledger),rawSha=sha(raw);
  const prior=await client.query('SELECT report FROM v17_decisions WHERE session=$1 AND raw_sha=$2 AND (before_sha=$3 OR after_sha=$3) ORDER BY created_at ASC LIMIT 1',[session,rawSha,before]);
  if(prior.rows.length){await client.query('COMMIT');return {...prior.rows[0].report,replayed:true};}
  const result=analyze(raw,ledger);const changed=[];
  for(const [symbol,summary] of Object.entries(result.results))for(const audit of summary.officialSignals){
   const old=ledger.find(x=>x.id===audit.id);if(!old)continue;
   if(audit.confidence!=='VERIFIED_SEGMENT'||audit.state===old.status)continue;
   const end=markets.find(x=>x.symbol===symbol).path.endAt;
   // Do not overwrite a newer committed state with older or ambiguous input.
   if(old.verifiedUntil&&Date.parse(end.replace(' ','T')+'+09:00')<Date.parse(old.verifiedUntil.replace(' ','T')+'+09:00'))continue;
   await client.query('UPDATE v17_signals SET status=$3,verified_until=$4 WHERE session=$1 AND id=$2',[session,audit.id,audit.state,end]);changed.push({id:audit.id,from:old.status,to:audit.state,observedUntil:end});
  }
  const afterRows=await client.query('SELECT original,status,verified_until FROM v17_signals WHERE session=$1 ORDER BY id',[session]);
  const after=sha(afterRows.rows.map(x=>({...x.original,status:x.status,verifiedUntil:x.verified_until})));
  const response={session,rawSha,beforeLedgerSha:before,afterLedgerSha:after,sourcePromptSha256:'dd53a208898124cb815433caca4348b95347d2bd06a49f7cda4b6cc57bcf8e52',official:result,changed,ledger:ledger.map(x=>({id:x.id,stateAtStart:x.status})),verified:'DETERMINISTIC_COMPANION_CONSISTENCY_ONLY',replayed:false};
  await client.query('INSERT INTO v17_decisions(session,raw_sha,before_sha,after_sha,report) VALUES($1,$2,$3,$4,$5)',[session,rawSha,before,after,response]);
  await client.query('COMMIT');return response;
 }catch(e){await client.query('ROLLBACK');throw e;}finally{client.release();}
}
const tools=[
 {name:'run_official_btc_eth_snapshot',description:'Use this canonical shared V1.7 companion engine to evaluate both pasted BTC/ETH snapshots and retrieve the one common deterministic verdict with formal ledger, chronological price-touch audits, calculated SL and TP percentages. NOT guaranteed complete discretionary V1.7 semantic equivalence or profitable trading.',inputSchema:{type:'object',properties:{raw:{type:'string',description:'Complete BTC and ETH raw snapshots pasted by the user'},session:{type:'string',description:'Shared official ledger identifier, default OFFICIAL_V17'}},required:['raw']}},
 {name:'read_official_btc_eth_ledger',description:'Get the original immutable BTC/ETH formal signal ledger for the given canonical session.',inputSchema:{type:'object',properties:{session:{type:'string'}},required:[]}},
 {name:'register_formal_btc_eth_signal',description:'Explicitly register an authorized original formal LONG/SHORT signal. Never guess old signals, prices, or fills.',inputSchema:{type:'object',properties:{session:{type:'string'},signal:{type:'object'}},required:['signal']}}
];
async function mcp(req,res){
 const body=await readJSON(req);if(body.method==='notifications/initialized'){res.writeHead(202);res.end();return;}
 const id=body.id;
 if(body.method==='initialize')return json(res,200,{jsonrpc:'2.0',id,result:{protocolVersion:'2025-03-26',capabilities:{tools:{listChanged:false}},serverInfo:{name:'v17-canonical-engine',version:'1.0.0'}}});
 if(body.method==='ping')return json(res,200,{jsonrpc:'2.0',id,result:{}});
 if(body.method==='tools/list')return json(res,200,{jsonrpc:'2.0',id,result:{tools}});
 if(body.method==='tools/call'){
  if(!isAuthorized(req))return json(res,200,{jsonrpc:'2.0',id,result:{isError:true,content:[{type:'text',text:'Unauthorized: configure secret credential in connection'}]}});
  const name=body.params?.name,args=body.params?.arguments||{};
  try{
   let output;if(name==='run_official_btc_eth_snapshot')output=await compute(args);
   else if(name==='read_official_btc_eth_ledger')output=await getLedger(standardSession(args.session||'OFFICIAL_V17'));
   else if(name==='register_formal_btc_eth_signal')output=await addSignal(args);
   else return json(res,200,{jsonrpc:'2.0',id,error:{code:-32601,message:'Unknown tool'}});
   return json(res,200,{jsonrpc:'2.0',id,result:{content:[{type:'text',text:JSON.stringify(output)}],isError:false}});
  }catch(e){return json(res,200,{jsonrpc:'2.0',id,result:{content:[{type:'text',text:e.message}],isError:true}});}
 }
 return json(res,200,{jsonrpc:'2.0',id,error:{code:-32601,message:'Method not found'}});
}
const PAGE=`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC·ETH V1.7 공통 판정 엔진</title><style>body{font-family:system-ui,sans-serif;max-width:950px;margin:25px auto;padding:0 15px;background:#0f172a;color:#f8fafc}textarea,input,button{box-sizing:border-box;width:100%;padding:12px;border-radius:9px;border:1px solid #475569;background:#1e293b;color:white;font-size:15px}textarea{height:200px}button{margin-top:12px;background:#2563eb;font-weight:700;cursor:pointer}label{display:block;margin-top:15px;margin-bottom:5px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;padding:15px;border:1px solid #334155;border-radius:9px}small{color:#cbd5e1}</style></head><body><h2>BTC·ETH V1.7 공통 판정</h2><p>BTC·ETH 데이터를 그대로 붙여넣으면 같은 공식 원장과 정량 판정 결과를 확인합니다.</p><small>이 도구는 V1.7을 수정하지 않는 별도 정량 보조판정입니다. 주관적 V1.7 전체와의 동치 또는 수익성을 보장하지 않습니다. 거래소 주문을 실행하지 않습니다.</small><label>연결 암호</label><input type="password" id="key" autocomplete="off" placeholder="본인 전용 서버 암호"/><label>공식 원장</label><input id="session" value="OFFICIAL_V17"/><label>BTC·ETH 원본 데이터</label><textarea id="raw" placeholder="========== BTC (BTCUSDT) =========="></textarea><button onclick="run()">공식 분석 실행</button><pre id="result">진입가·손절·목표 및 분석 결과가 이곳에 표시됩니다.</pre><script>async function run(){const o=document.getElementById('result');o.textContent='분석 중...';try{const r=await fetch('/api/run',{method:'POST',headers:{'content-type':'application/json','authorization':'Bearer '+document.getElementById('key').value},body:JSON.stringify({raw:document.getElementById('raw').value,session:document.getElementById('session').value})});const j=await r.json();if(!r.ok)throw Error(j.error||r.status);if(!j.official){o.textContent=JSON.stringify(j,null,2);return}let s='';for(const [k,v] of Object.entries(j.official.results)){s+='\\n'+k+' | $'+v.last+' | '+v.recommendation+'\\n';for(const sig of v.officialSignals)s+=sig.id+' '+sig.state+' | SL '+sig.metrics.stopPct+'% | TP1 '+sig.metrics.tp1Pct+'% / TP2 '+sig.metrics.tp2Pct+'% / TP3 '+sig.metrics.tp3Pct+'%\\n';if(v.selectedPlan)s+='진입 '+v.selectedPlan.entryLow+' | 손절 '+v.selectedPlan.stop+' | TP1 '+v.selectedPlan.tp1+' | TP2 '+v.selectedPlan.tp2+' | TP3 '+v.selectedPlan.tp3+'\\n';}o.textContent=s+'\\n\\n참고: '+j.official.policy;}catch(e){o.textContent='오류: '+e.message}}</script></body></html>`;
const server=http.createServer(async(req,res)=>{
 try{
  if(req.url==='/health'&&req.method==='GET')return json(res,200,{status:!SECRET||!DB_URL?'not_ready':'ready',engine:'V1.7_SHARED_VERDICT_V1',db:!!DB_URL,authentication:!!SECRET});
  if(req.url==='/'&&req.method==='GET'){res.writeHead(200,{'content-type':'text/html; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff','content-security-policy':"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"});return res.end(PAGE);}
  if(req.url==='/mcp'&&req.method==='POST')return await mcp(req,res);
  if(req.url==='/mcp'&&req.method==='GET'){res.writeHead(405,{'allow':'POST'});return res.end();}
  if(req.url.startsWith('/api/')){
   if(!isAuthorized(req))return fail(res,401,'invalid authentication');
   if(req.method==='POST'&&req.url==='/api/run')return json(res,200,await compute(await readJSON(req)));
   if(req.method==='POST'&&req.url==='/api/signal')return json(res,200,await addSignal(await readJSON(req)));
   if(req.method==='GET'&&req.url.startsWith('/api/ledger')){const session=new URL(req.url,'http://localhost').searchParams.get('session')||'OFFICIAL_V17';return json(res,200,{session,signals:await getLedger(standardSession(session))});}
  }
  return fail(res,404,'not found');
 }catch(e){return fail(res,400,e.message||'invalid request');}
});
if(process.env.V17_NO_LISTEN!=='1'){
 try{guarded();await database();server.listen(PORT,'0.0.0.0',()=>process.stderr.write('v17-server-listening:'+PORT+'\n'));}
 catch(e){process.stderr.write('startup-failed:'+e.message+'\n');process.exit(1);}
}
export {server,compute,addSignal,getLedger};