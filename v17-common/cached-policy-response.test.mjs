import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCachedPolicyResponse } from './cached-policy-response.mjs';
import {POLICY_TEXT,POLICY_SHA256,POLICY_BYTES,POLICY_LINES} from './policy-v17.mjs';
const original = {
  status:'NEEDS_AI_ANALYSIS',
  policy:{sha256:POLICY_SHA256,bytes:POLICY_BYTES,lines:POLICY_LINES,text:POLICY_TEXT},
  ledger:[{id:'ETH-PENDING',status:'PENDING',original:{entryLow:2014,entryHigh:2015.5}}],
  snapshot:{btcPrice:71133.6,ethPrice:2013.07},
  evidenceQuality:{warnings:['ETH:OLD_GAP']},
  beforeLedgerSha256:'ledger-hash',decisionKey:'decision-hash',
};
test('default response remains strictly identical without opt-in',()=>{
  assert.strictEqual(applyCachedPolicyResponse(original),original);
  assert.strictEqual(applyCachedPolicyResponse(original,''),original);
});
test('valid cached hash skips only verbatim policy text',()=>{
  const got=applyCachedPolicyResponse(original,POLICY_SHA256);
  assert.equal(got.policy.text,undefined);
  assert.equal(got.policy.sha256,POLICY_SHA256);
  assert.equal(got.policyCache.used,true);
  for(const k of ['ledger','snapshot','evidenceQuality','beforeLedgerSha256','decisionKey'])
    assert.deepEqual(got[k],original[k]);
  assert.strictEqual(got.ledger,original.ledger);
});
test('wrong or outdated hash safely returns full V1.7 policy',()=>{
  for(const sha of ['old','a'.repeat(64),null]) {
    const got=applyCachedPolicyResponse(original,sha);
    assert.equal(got.policy.text,POLICY_TEXT);
    if(sha)assert.equal(got.policyCache.used,false);
  }
});
test('OFFICIAL_REPLAY remains original response even when cached hash set',()=>{
  const replay={status:'OFFICIAL_REPLAY',officialDecision:{reportMarkdown:'official'}};
  assert.strictEqual(applyCachedPolicyResponse(replay,POLICY_SHA256),replay);
});
test('policy source, ledger and signal fields stay immutable',()=>{
  const policyTextBefore=original.policy.text;
  const got=applyCachedPolicyResponse(original,POLICY_SHA256);
  assert.equal(original.policy.text,policyTextBefore);
  assert.equal(got.ledger[0].original.entryLow,2014);
  assert.equal(original.policy.sha256,POLICY_SHA256);
});
test('JSON response shrinks and hashes remain identical, using real 902-line policy',()=>{
  const normal=JSON.stringify(original);
  const cached=JSON.stringify(applyCachedPolicyResponse(original,POLICY_SHA256));
  assert.equal(POLICY_LINES,902);
  assert.ok(normal.length>cached.length+20000,{normalBytes:normal.length,cachedBytes:cached.length});
  assert.equal(JSON.parse(cached).policy.sha256,POLICY_SHA256);
});
