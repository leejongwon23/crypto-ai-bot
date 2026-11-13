import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
process.env.V17_NO_LISTEN='1';
const {resetOfficialLedger}=await import('./server.mjs');

test('server syntax is valid',()=>{
 const r=spawnSync(process.execPath,['--check',new URL('./server.mjs',import.meta.url).pathname],{encoding:'utf8'});
 assert.equal(r.status,0,r.stderr);
});
test('reset rejects missing or alternate sessions before database access',async()=>{
 await assert.rejects(resetOfficialLedger({kind:'V17_RESET_ALL_OFFICIAL',confirmation:'RESET_ALL_OFFICIAL_V17_2026_11_13',expectedLedgerSha256:'a'.repeat(64)}),/RESET_DEFAULT_OFFICIAL_SESSION_ONLY/);
 await assert.rejects(resetOfficialLedger({session:'OFFICIAL_V17_CLEAN_20261113_1611',kind:'V17_RESET_ALL_OFFICIAL',confirmation:'RESET_ALL_OFFICIAL_V17_2026_11_13',expectedLedgerSha256:'a'.repeat(64)}),/RESET_DEFAULT_OFFICIAL_SESSION_ONLY/);
});
test('reset rejects missing exact operation and explicit confirmation',async()=>{
 await assert.rejects(resetOfficialLedger({session:'OFFICIAL_V17',confirmation:'RESET_ALL_OFFICIAL_V17_2026_11_13',expectedLedgerSha256:'a'.repeat(64)}),/RESET_KIND_REQUIRED/);
 await assert.rejects(resetOfficialLedger({session:'OFFICIAL_V17',kind:'V17_RESET_ALL_OFFICIAL',confirmation:'WRONG',expectedLedgerSha256:'a'.repeat(64)}),/RESET_EXPLICIT_CONFIRMATION_REQUIRED/);
});
test('reset rejects missing and malformed expected ledger fingerprint',async()=>{
 const base={session:'OFFICIAL_V17',kind:'V17_RESET_ALL_OFFICIAL',confirmation:'RESET_ALL_OFFICIAL_V17_2026_11_13'};
 await assert.rejects(resetOfficialLedger(base),/RESET_EXPECTED_LEDGER_SHA_REQUIRED/);
 await assert.rejects(resetOfficialLedger({...base,expectedLedgerSha256:'0000'}),/RESET_EXPECTED_LEDGER_SHA_REQUIRED/);
});
