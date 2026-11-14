import test from 'node:test';
import assert from 'node:assert/strict';
import {ledgerHash} from './v17-core.mjs';
import {ATTEST_CONFIRM,TWO_USER_TP3_IDS,planUserAttestedTp3} from './user-attested-tp3.mjs';

const rows=()=>[
 {id:TWO_USER_TP3_IDS[0],symbol:'BTCUSDT',status:'ENTERED',verifiedUntil:'2026-11-14 09:00:28',
  original:{id:TWO_USER_TP3_IDS[0],issuedAt:'2026-11-14 08:33:22',
   direction:'SHORT',entryLow:68310,entryHigh:68355,stop:68575,tp1:67813,tp2:67468.4,tp3:67100}},
 {id:TWO_USER_TP3_IDS[1],symbol:'ETHUSDT',status:'ENTERED',verifiedUntil:'2026-11-14 09:00:34',
  original:{id:TWO_USER_TP3_IDS[1],issuedAt:'2026-11-14 07:25:32',
   direction:'SHORT',entryLow:1933,entryHigh:1937,stop:1946,tp1:1914.28,tp2:1894.1,tp3:1880}}
];
const raw='========== BTC (BTCUSDT) ==========\n24h high/low: 70187 / 66871\n\n' +
          '========== ETH (ETHUSDT) ==========\n24h high/low: 1988.67 / 1875.78\n';
const args=()=>{const ledger=rows();return {
 ledger,sourceRaw:raw,sourceSnapshotSha256:'a'.repeat(64),
 expectedLedgerSha256:ledgerHash(ledger),
 signalIds:[...TWO_USER_TP3_IDS],confirmation:ATTEST_CONFIRM,at:'2026-11-14 13:33:00'
};};
test('exact original two signals close as user-attested TP3, not exchange-proven wins',()=>{
 const p=planUserAttestedTp3(args());
 assert.deepEqual(p.map(x=>x.kind),['TP3_DONE','TP3_DONE']);
 assert.deepEqual(p.map(x=>x.observedRollingLow),[66871,1875.78]);
 assert(p.every(x=>x.userConfirmedSuccessful&&!x.independentStopOrderVerified&&!x.realizedPnlVerified));
 assert(p.every(x=>x.provenance==='USER_ATTESTED_NOT_CHRONOLOGICALLY_VERIFIED'));
});
test('does not allow mutation for old completed signal or non-entered original',()=>{
 const a=args();a.ledger[0].status='TP3_DONE';a.expectedLedgerSha256=ledgerHash(a.ledger);
 assert.throws(()=>planUserAttestedTp3(a),/ONLY_KNOWN_ENTERED_OPEN/);
 const b=args();b.ledger[1].status='PENDING';b.expectedLedgerSha256=ledgerHash(b.ledger);
 assert.throws(()=>planUserAttestedTp3(b),/ONLY_KNOWN_ENTERED_OPEN/);
});
test('requires both exact IDs, explicit stop-order user attestation, and unchanged ledger',()=>{
 const a=args();a.signalIds[0]='WRONG';assert.throws(()=>planUserAttestedTp3(a),/ONLY_TWO_ORIGINAL_IDS/);
 const b=args();b.confirmation='YES';assert.throws(()=>planUserAttestedTp3(b),/EXPLICIT_USER_TP3_CONFIRMATION/);
 const c=args();c.expectedLedgerSha256='b'.repeat(64);assert.throws(()=>planUserAttestedTp3(c),/LEDGER_CHANGED/);
});
test('low above target or missing price source never produces a success',()=>{
 const a=args();a.sourceRaw=a.sourceRaw.replace('1875.78','1881');
 assert.throws(()=>planUserAttestedTp3(a),/LOW_DID_NOT_CROSS_TP3/);
 const b=args();b.sourceRaw=b.sourceRaw.replace('24h high/low: 70187 / 66871','broken');
 assert.throws(()=>planUserAttestedTp3(b),/LOW_MISSING/);
});
test('the administrative attestation does not change original entry stop or TP levels',()=>{
 const a=args();const before=JSON.stringify(a.ledger);
 planUserAttestedTp3(a);
 assert.equal(JSON.stringify(a.ledger),before);
});
