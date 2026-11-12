import test from 'node:test';
import assert from 'node:assert/strict';
import {assessSnapshotEvidence,prepare,validateOfficialSubmission,ledgerHash,digest} from './v17-core.mjs';
import {POLICY_SHA256} from './policy-v17.mjs';
const t='2026-11-13 05:15:35';
function snap({ethStart='2026-11-13 03:43:54',ethHigh=2084.59,btcFailure=true}={}){
 const btc=`========== BTC (BTCUSDT) ==========\ntimestamp_kst: 2026-11-13 05:15:27\n[1] REALTIME PRICE (AUTO)\nlastPrice: 71571.70\n24h high/low: 74783.50 / 71205.30\n[2] OHLCV + EMA/MACD (AUTO)\n[2B] ATR/VOL + SWING LEVELS (AUTO)\n[3] OPEN INTEREST (AUTO)\n[5] ORDERBOOK + CVD SNAPSHOT (AUTO)\n[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO): ${btcFailure?'FAIL':'OK'}\n${btcFailure?'error: Bybit:Too many visits. Exceeded the API Rate Limit.':''}\n[8] SENTIMENT (AUTO - Fear&Greed only)\n[CVD] SERIES METRICS (AUTO, exact time-window aggTrades)\ncvd_reliability: HIGH\n[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: 71853.20 @ 2026-11-13 03:43:47 | end_price: 71571.70 @ 2026-11-13 05:15:27\n[7] HISTORICAL FLOW SNAPSHOT (AUTO)\n`;
 const eth=`========== ETH (ETHUSDT) ==========\ntimestamp_kst: ${t}\n[1] REALTIME PRICE (AUTO)\nlastPrice: 2018.99\n24h high/low: ${ethHigh} / 1982.36\n[2] OHLCV + EMA/MACD (AUTO)\n[2B] ATR/VOL + SWING LEVELS (AUTO)\n[3] OPEN INTEREST (AUTO)\n[5] ORDERBOOK + CVD SNAPSHOT (AUTO)\n[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO)\n[8] SENTIMENT (AUTO - Fear&Greed only)\n[CVD] SERIES METRICS (AUTO, exact time-window aggTrades)\ncvd_reliability: HIGH\n[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: 2018.26 @ ${ethStart} | end_price: 2018.99 @ ${t}\n[7] HISTORICAL FLOW SNAPSHOT (AUTO)\n`;
 return btc+eth;
}
const previous={btcTime:'2026-11-12 22:20:56',ethTime:'2026-11-12 22:21:10'};
const old={id:'ET-OLD',symbol:'ETHUSDT',status:'PENDING',original:{direction:'SHORT',entryLow:2045.2,entryHigh:2045.7,issuedAt:'2026-11-12 20:26:28'},verifiedUntil:'2026-11-12 20:26:28'};
test('gap from 22:21 to 03:43 correctly exposes uncertain ETH PENDING status',()=>{
 const q=assessSnapshotEvidence(snap(),{previousSnapshot:previous,ledger:[old]});
 assert(q.perSymbol.ETH.gapSeconds>18000);
 assert(q.perSymbol.ETH.flags.includes('PRICE_PATH_GAP'));
 assert(q.perSymbol.ETH.flags.includes('EXISTING_SIGNAL_STATE_UNVERIFIED'));
 assert.deepEqual(q.perSymbol.ETH.couldNotVerifySignals,['ET-OLD']);
});
test('API rate limit on BTC 5B is detected, not mistaken for complete data',()=>{
 const q=assessSnapshotEvidence(snap(),{previousSnapshot:previous,ledger:[old]});
 assert.equal(q.perSymbol.BTC.rateLimitVWAP,true);
 assert(q.warnings.includes('BTC:VWAP_TAKER_RATE_LIMIT'));
});
test('gap is not misclassified as pending touch if the inclusive 24h high disproves entry',()=>{
 const q=assessSnapshotEvidence(snap({ethHigh:2030}),{previousSnapshot:previous,ledger:[old]});
 assert(q.perSymbol.ETH.flags.includes('PRICE_PATH_GAP'));
 assert.deepEqual(q.perSymbol.ETH.couldNotVerifySignals,[]);
});
test('no gap when incoming path overlaps last official snapshot',()=>{
 const q=assessSnapshotEvidence(snap({ethStart:'2026-11-12 22:21:10'}),{previousSnapshot:previous,ledger:[old]});
 assert(!q.perSymbol.ETH.flags.includes('PRICE_PATH_GAP'));
});
test('missing core market sections does not silently pass completeness',()=>{
 const short=snap().replace('[CVD] SERIES METRICS (AUTO, exact time-window aggTrades)','[CVD REMOVED]');
 const q=assessSnapshotEvidence(short,{});
 assert(q.perSymbol.BTC.missingSections.includes('[CVD]'));
});
const sides=['BTC_LONG','BTC_SHORT','ETH_LONG','ETH_SHORT'];
const alts={}; for(const d of sides) alts[d]={marketEvidence:'data',counterEvidence:'counter',alternatives:Object.fromEntries('ABCDE'.split('').map(k=>[k,{judgement:'HOLD',evidence:'still under independent review'}]))};
const checks=Object.fromEntries(Array.from({length:16},(_,i)=>[String(i+1),{status:'CONFIRMED',evidence:'checked'}]));
const keys=[...Array.from({length:24},(_,i)=>String(i)),'APPENDIX_A','APPENDIX_B','FINAL_EXECUTION'];
const clauses=Object.fromEntries(keys.map(k=>[k,{status:'CONFIRMED',evidence:'checked'}]));
const analysis={reportMarkdown:'BTC HOLD / ETH UNVERIFIED',directions:alts,coupling:{status:'MIXED',evidence:'time alignment checked'},checks,clauses,audit20:{rules:'test',bias:'test',evidence:'test'},conclusions:{BTC:{decision:'HOLD',evidence:'incomplete'},ETH:{decision:'HOLD',evidence:'incorrect confident outcome'}},newSignals:[],events:[]};
function args(){const raw=snap(),s=prepare({raw,ledger:[old],previousSnapshot:previous});return {raw,session:'OFFICIAL_V17',policySha256:POLICY_SHA256,snapshotSha256:s.snapshot.sha256,beforeLedgerSha256:ledgerHash([old]),decisionKey:s.decisionKey,analysis:structuredClone(analysis)};}
test('historical uncertainty must be disclosed, even when current market outcome is HOLD',()=>{
 assert.throws(()=>validateOfficialSubmission(args(),[old],previous),/HISTORICAL_UNCERTAINTY_DISCLOSURE_REQUIRED_ETH/);
});
test('current analysis can be HOLD without falsely confirming old PENDING during gap',()=>{
 const a=args();a.analysis.historicalUncertainty={ETH:{unresolvedIds:['ET-OLD'],evidence:'old pending entry/outcome cannot be verified in missing path'}};
 const b=validateOfficialSubmission(a,[old],previous);
 assert.equal(b.analysis.conclusions.ETH.decision,'HOLD');
 assert.equal(b.evidenceQuality.perSymbol.ETH.couldNotVerifySignals[0],'ET-OLD');
 assert.equal(b.newSignals.length,0);
});
test('original V1.7 SHA unchanged after deterministic audit patch',()=>{
 assert.equal(POLICY_SHA256,'5fbd801f1378df7f584b248aadfaf98ae7e38d0bca782e3e09625bafe39e66ad');
});
