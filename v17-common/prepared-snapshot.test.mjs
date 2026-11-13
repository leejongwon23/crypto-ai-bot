import test from 'node:test';
import assert from 'node:assert/strict';
import {storePreparedSnapshot,recoverOfficialSubmissionRaw} from './prepared-snapshot.mjs';

function mockDb() {
 const snapshots=new Map();let sqlCount=0;
 return {snapshots,get queryCount(){return sqlCount},async query(sql,params) {
  sqlCount++;
  if(sql.startsWith('INSERT INTO v17_ai_prepared_snapshots')) {
   const [session,sha,raw]=params;const key=session+'#'+sha;
   if(!snapshots.has(key))snapshots.set(key,raw);
   return {rows:[],rowCount:1};
  }
  if(sql.startsWith('SELECT raw FROM v17_ai_prepared_snapshots')) {
   const [session,sha]=params,key=session+'#'+sha;
   return {rows:snapshots.has(key)?[{raw:snapshots.get(key)}]:[],rowCount:snapshots.has(key)?1:0};
  }
  throw Error('TEST_UNKNOWN_SQL:'+sql);
 }};
}
const shaA='a'.repeat(64),shaB='b'.repeat(64);
const raw='BTC (BTCUSDT)\r\nlastPrice: 68837.00  \n\nETH (ETHUSDT)\ntimestamp_kst: 2026-11-13 16:32:59\nlastPrice: 1938.52';

test('server stores and restores the EXACT original input including CRLF and whitespace',async()=>{
 const db=mockDb();await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,raw);
 assert.equal(await recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{snapshotSha256:shaA}),raw);
});

test('same digest and session are idempotent; alternative sessions stay isolated',async()=>{
 const db=mockDb();await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,raw);
 await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,raw);
 assert.equal(db.snapshots.size,1);
 await assert.rejects(recoverOfficialSubmissionRaw(db,'OTHER_SESSION',{snapshotSha256:shaA}),/PREPARED_SNAPSHOT_NOT_FOUND/);
 await storePreparedSnapshot(db,'OTHER_SESSION',shaA,'different snapshot text');
 assert.equal(await recoverOfficialSubmissionRaw(db,'OTHER_SESSION',{snapshotSha256:shaA}),'different snapshot text');
 assert.equal(await recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{snapshotSha256:shaA}),raw);
});

test('unknown digest does not silently use the previous snapshot',async()=>{
 const db=mockDb();await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,raw);
 await assert.rejects(recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{snapshotSha256:shaB}),/PREPARED_SNAPSHOT_NOT_FOUND/);
});

test('missing or malformed digests cannot resolve a raw',async()=>{
 const db=mockDb();
 await assert.rejects(recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{}),/PREPARED_SNAPSHOT_SHA_REQUIRED/);
 await assert.rejects(recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{snapshotSha256:'bad'}),/PREPARED_SNAPSHOT_SHA_REQUIRED/);
 await assert.rejects(storePreparedSnapshot(db,'OFFICIAL_V17','not-a-sha',raw),/PREPARED_SNAPSHOT_SHA_REQUIRED/);
});

test('legacy submit-with-raw remains backwards compatible',async()=>{
 const db=mockDb();
 assert.equal(await recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{raw,snapshotSha256:shaA}),raw);
 await assert.rejects(recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{raw:''}),/SUBMISSION_RAW_INVALID/);
});

test('stored snapshot is not overwritten by later same-sha prepare',async()=>{
 const db=mockDb();await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,raw);
 await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,'DIFFERENT');
 assert.equal(await recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{snapshotSha256:shaA}),raw);
});

test('a legacy client that mistakenly retypes raw cannot override the byte-exact prepared source',async()=>{
 const db=mockDb();await storePreparedSnapshot(db,'OFFICIAL_V17',shaA,raw);
 const mistakenlyRetyped=raw+'\\nextra accidental whitespace';
 assert.equal(await recoverOfficialSubmissionRaw(db,'OFFICIAL_V17',{raw:mistakenlyRetyped,snapshotSha256:shaA}),raw);
});
