import test from 'node:test';
import assert from 'node:assert/strict';
import {tools} from './server.mjs';
import {POLICY_SHA256} from './policy-v17.mjs';
import {applyCachedPolicyResponse} from './cached-policy-response.mjs';
test('MCP tool publishes optional cache and timing flags without changing required input',()=>{
 const run=tools.find(x=>x.name==='run_official_btc_eth_snapshot');
 assert.ok(run,'MCP snapshot tool must remain available');
 assert.deepEqual(run.inputSchema.required,['raw']);
 assert.equal(run.inputSchema.properties.cachedPolicySha256.type,'string');
 assert.equal(run.inputSchema.properties.debugTimings.type,'boolean');
 assert.ok(run.description.includes('SAME conversation'));
 const publish=tools.find(x=>x.name==='publish_official_v17_analysis');
 assert.ok(publish);
 const oldCompat=tools.find(x=>x.name==='register_formal_btc_eth_signal');
 assert.ok(oldCompat);
});
test('cache is opt-in, never bypasses the policy digest and never alters official signal fields',()=>{
 const source={status:'NEEDS_AI_ANALYSIS',
 policy:{sha256:POLICY_SHA256,lines:902,text:'V1.7 body'},
 ledger:[{id:'BTC-KEEP',status:'ENTERED',original:{direction:'SHORT',entryLow:71265,stop:71480}}],
 snapshot:{btcPrice:71133.60,ethPrice:2013.07},checks:{},clauses:{},
 requirements:{requiredDirections:['BTC_LONG','BTC_SHORT','ETH_LONG','ETH_SHORT'],requiredSteps:[1,2,3],requiredClauses:['0','1','FINAL_EXECUTION']},
 decisionKey:'original',beforeLedgerSha256:'v17ledger'};
 const baseline=applyCachedPolicyResponse(source,null);
 const valid=applyCachedPolicyResponse(source,POLICY_SHA256);
 assert.strictEqual(baseline,source);
 assert.equal(valid.policy.text,undefined);
 assert.deepEqual(valid.ledger,source.ledger);
 assert.deepEqual(valid.requirements,source.requirements);
 assert.deepEqual(valid.snapshot,source.snapshot);
 assert.equal(valid.decisionKey,'original');
 assert.strictEqual(source.policy.text,'V1.7 body');
});
