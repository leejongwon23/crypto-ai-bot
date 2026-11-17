import test from 'node:test';
import assert from 'node:assert/strict';
process.env.V17_NO_LISTEN='1';
const {readLedger}=await import('./server.mjs');
test('ledger reader carries complete ordered revision events without changing original',async()=>{
 const original={stop:90},plan={stop:95,tp1:110,tp2:120,tp3:130},event={kind:'REVISION',at:'2026-11-17 06:36:00',plan};
 let query;
 const db={query:async(sql,args)=>{query={sql,args};return {rows:[{id:'S',symbol:'ETHUSDT',original,status:'ENTERED',verified_until:'2026-11-17 06:36:00',plan_version:1,effective_plan:plan,plan_effective_at:event.at,plan_revisions:[event]}]};}};
 const rows=await readLedger(db,'TEST_ONLY');
 assert.match(query.sql,/jsonb_agg\(e\.event ORDER BY e\.seq\)/);
 assert.deepEqual(query.args,['TEST_ONLY']);assert.equal(rows[0].original,original);
 assert.deepEqual(rows[0].managementRevisions,[event]);assert.equal(rows[0].original.stop,90);
});
