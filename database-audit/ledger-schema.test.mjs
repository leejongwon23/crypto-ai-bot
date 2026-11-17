import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
process.env.V17_NO_LISTEN='1';
const {readLedger}=await import('../v17-common/server.mjs');
const require=createRequire(new URL('../v17-common/server.mjs',import.meta.url)),{Client}=require('pg');
const connectionString=process.env.RESEARCH_TEST_DATABASE_URL;
if(!connectionString?.startsWith('postgresql://research:research_only@localhost:5432/research_test'))throw Error('DISPOSABLE_LOCAL_DB_ONLY');
async function connection(){const c=new Client({connectionString});await c.connect();return c;}
test('actual source DDL and revision aggregation SQL execute on disposable PostgreSQL',async()=>{
 const c=await connection();
 try{
 const source=await readFile('v17-common/server.mjs','utf8');
 const ddl=source.match(/const sql=\x60([\s\S]*?)\x60;/)?.[1];assert.ok(ddl);
 await c.query(ddl);await c.query('BEGIN');
 const original={id:'SYNTHETIC_SQL_ONLY',direction:'LONG',entryLow:100,entryHigh:100,stop:90,tp1:110,tp2:120,tp3:130,issuedAt:'2026-11-17 06:00:00'};
 const plan={stop:95,tp1:110,tp2:120,tp3:130},revision={kind:'REVISION',at:'2026-11-17 06:36:00',plan};
 await c.query("INSERT INTO v17_ai_sessions(name) VALUES('AUDIT_SYNTHETIC')");
 await c.query('INSERT INTO v17_ai_signals(session,id,symbol,original,status,plan_version) VALUES($1,$2,$3,$4,$5,$6)',['AUDIT_SYNTHETIC',original.id,'ETHUSDT',original,'ENTERED',1]);
 await c.query('INSERT INTO v17_ai_events(session,id,event,at_kst) VALUES($1,$2,$3,$4)',['AUDIT_SYNTHETIC',original.id,revision,revision.at]);
 const rows=await readLedger(c,'AUDIT_SYNTHETIC');
 assert.deepEqual(rows[0].original,original);assert.deepEqual(rows[0].managementRevisions,[revision]);
 assert.deepEqual(rows[0].effectivePlan,plan);assert.equal(rows[0].version,1);
 }finally{await c.query('ROLLBACK');await c.end();}
});
test('separate research schema rejects UPDATE DELETE TRUNCATE on persisted evidence',async()=>{
 const c=await connection();
 try{
 await c.query(await readFile('research-v2/schema.sql','utf8'));
 await c.query("INSERT INTO btc_eth_research_v2.evidence(digest,event_time,known_at,captured_at,exchange,category,symbol,source_kind,source_uri,vintage_status,payload) VALUES($1,NOW(),NOW(),NOW(),'Bybit','linear','ETHUSDT','SYNTHETIC_TEST','urn:test','UNVERIFIED','{}')",['a'.repeat(64)]);
 for(const sql of ["UPDATE btc_eth_research_v2.evidence SET payload='{}'","DELETE FROM btc_eth_research_v2.evidence","TRUNCATE btc_eth_research_v2.evidence"])
  await assert.rejects(c.query(sql),/APPEND_ONLY_RESEARCH_EVIDENCE/);
 const rows=await c.query('SELECT count(*)::int AS n FROM btc_eth_research_v2.evidence');assert.equal(rows.rows[0].n,1);
 }finally{await c.end();}
});
