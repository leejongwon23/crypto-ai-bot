import test from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';import {createRequire} from 'node:module';
process.env.V17_NO_LISTEN='1';
const require=createRequire(new URL('../v17-common/server.mjs',import.meta.url));const {Client}=require('pg');
test('V2 schema preserves original signals/events/analyses and accepts versioned same-snapshot research',async()=>{
 const url=process.env.QUANT_TEST_DATABASE_URL;if(!url?.startsWith('postgresql://quant:quant_test@localhost:5432/quant_test'))throw Error('DISPOSABLE_TEST_DATABASE_ONLY');
 const db=new Client({connectionString:url});await db.connect();
 try{
  const source=await readFile('v17-common/server.mjs','utf8'),ddl=source.match(/const sql=\x60([\s\S]*?)\x60;/)[1];await db.query(ddl);
  await db.query("INSERT INTO v17_ai_sessions(name) VALUES('SYNTHETIC')");
  await db.query("INSERT INTO v17_ai_signals(session,id,symbol,original,status) VALUES('SYNTHETIC','OLD','ETHUSDT','{\"stop\":90}','STOPPED')");
  await db.query("INSERT INTO v17_ai_events(session,id,event,at_kst) VALUES('SYNTHETIC','OLD','{\"kind\":\"STOPPED\"}','2026-11-17 06:44:00')");
  await db.query("INSERT INTO v17_ai_analyses(session,snapshot_sha,before_sha,after_sha,policy_sha,raw,official) VALUES('SYNTHETIC','same','same','same','V19','original','{}')");
  const before=await db.query("SELECT (SELECT jsonb_agg(s) FROM v17_ai_signals s)::text AS signals,(SELECT jsonb_agg(e) FROM v17_ai_events e)::text AS events,(SELECT jsonb_agg(a) FROM v17_ai_analyses a)::text AS analyses");
  await db.query("INSERT INTO v20_ai_analyses(session,snapshot_sha,before_sha,after_sha,policy_sha,raw,official) VALUES('SYNTHETIC','same','same','same','V20','new','{}')");
  const after=await db.query("SELECT (SELECT jsonb_agg(s) FROM v17_ai_signals s)::text AS signals,(SELECT jsonb_agg(e) FROM v17_ai_events e)::text AS events,(SELECT jsonb_agg(a) FROM v17_ai_analyses a)::text AS analyses");
  assert.deepEqual(after.rows,before.rows);
  const both=await db.query("SELECT policy_sha FROM v20_ai_analysis_history ORDER BY policy_sha");assert.deepEqual(both.rows,[{policy_sha:'V19'},{policy_sha:'V20'}]);
 }finally{await db.end();}
});
