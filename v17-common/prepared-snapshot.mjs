/**
 * Transport-only raw snapshot preservation for V1.7 official submissions.
 * Snapshot strategy, validator and immutable official signal ledger are unchanged.
 * Once a snapshot has been prepared, the write step can reference its digest
 * without reconstructing/retyping the original input.
 */
const SHA256=/^[0-9a-f]{64}$/;

export async function storePreparedSnapshot(db,session,snapshotSha256,raw) {
 if(typeof session!=='string'||!session)throw Error('PREPARED_SESSION_REQUIRED');
 if(!SHA256.test(snapshotSha256||''))throw Error('PREPARED_SNAPSHOT_SHA_REQUIRED');
 if(typeof raw!=='string'||!raw)throw Error('PREPARED_RAW_REQUIRED');
 await db.query(
  'INSERT INTO v17_ai_prepared_snapshots(session,snapshot_sha,raw) VALUES($1,$2,$3) ON CONFLICT(session,snapshot_sha) DO NOTHING',
  [session,snapshotSha256,raw]
 );
}

export async function recoverOfficialSubmissionRaw(db,session,submission) {
 const sha=submission.snapshotSha256;
 // Always use the byte-exact server copy if a prepared SHA is available.
 // In particular, ignore a mistakenly retyped 'raw' from legacy clients.
 // The original V1.7 validator still checks SHA, ledger and decision keys.
 if(SHA256.test(sha||'')) {
  const result=await db.query(
   'SELECT raw FROM v17_ai_prepared_snapshots WHERE session=$1 AND snapshot_sha=$2',
   [session,sha]
  );
  if(result.rows.length===1&&typeof result.rows[0].raw==='string')
   return result.rows[0].raw;
 }
 // Before this feature was deployed, older clients could publish exact raw
 // without first persisting it; retain that backwards-compatible fallback.
 if(Object.prototype.hasOwnProperty.call(submission,'raw')&&submission.raw!=null) {
  if(typeof submission.raw!=='string'||!submission.raw)throw Error('SUBMISSION_RAW_INVALID');
  return submission.raw;
 }
 if(!SHA256.test(sha||''))throw Error('PREPARED_SNAPSHOT_SHA_REQUIRED');
 throw Error('PREPARED_SNAPSHOT_NOT_FOUND_REPREPARE_REQUIRED');
}
