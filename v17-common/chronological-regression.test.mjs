import test from 'node:test';
import assert from 'node:assert/strict';
import {assessSnapshotEvidence,prepare,validateOfficialSubmission,ledgerHash} from './v17-core.mjs';
import {POLICY_SHA256} from './policy-v17.mjs';

const old={id:'ETH-ORIGINAL-PENDING',symbol:'ETHUSDT',status:'PENDING',original:{direction:'SHORT',entryLow:2045.20,entryHigh:2045.70,stop:2049.40,tp1:2038.34,tp2:2033.97,tp3:2003.22,issuedAt:'2026-11-12 20:26:28'},verifiedUntil:'2026-11-12 20:26:28'};
const t='2026-11-12 22:25:10';
const prev={btcTime:'2026-11-12 22:20:56',ethTime:'2026-11-12 22:20:56'};
function raw({ethStart='2026-11-12 22:20:56',ethHigh=2046,ethLow=2043,ethHigh24=2084.59,ethLast=2043.75}={}){
 const b=`========== BTC (BTCUSDT) ==========\ntimestamp_kst: ${t}\n[1] REALTIME PRICE (AUTO)\nlastPrice: 72571.90\n24h high/low: 74783.50 / 72197.00\n[2] OHLCV + EMA/MACD (AUTO)\n[2B] ATR/VOL + SWING LEVELS (AUTO)\n[3] OPEN INTEREST (AUTO)\n[5] ORDERBOOK + CVD SNAPSHOT (AUTO)\n[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO)\n[8] SENTIMENT (AUTO)\n[CVD] SERIES METRICS (AUTO)\ncvd_reliability: HIGH\n[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: 72550 @ 2026-11-12 22:20:56 | end_price: 72571.90 @ ${t}\npath_high/low: 72600 @ 2026-11-12 22:24:00 / 72500 @ 2026-11-12 22:22:00\n[7] HISTORICAL FLOW SNAPSHOT (AUTO)\n`;
 const e=`========== ETH (ETHUSDT) ==========\ntimestamp_kst: ${t}\n[1] REALTIME PRICE (AUTO)\nlastPrice: ${ethLast}\n24h high/low: ${ethHigh24} / 1982.36\n[2] OHLCV + EMA/MACD (AUTO)\n[2B] ATR/VOL + SWING LEVELS (AUTO)\n[3] OPEN INTEREST (AUTO)\n[5] ORDERBOOK + CVD SNAPSHOT (AUTO)\n[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO)\n[8] SENTIMENT (AUTO)\n[CVD] SERIES METRICS (AUTO)\ncvd_reliability: HIGH\n[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: 2044 @ ${ethStart} | end_price: ${ethLast} @ ${t}\npath_high/low: ${ethHigh} @ 2026-11-12 22:24:00 / ${ethLow} @ 2026-11-12 22:22:00\n[7] HISTORICAL FLOW SNAPSHOT (AUTO)\n`;
 return b+e;
}
const sides=['BTC_LONG','BTC_SHORT','ETH_LONG','ETH_SHORT'];
const directions=Object.fromEntries(sides.map(d=>[d,{marketEvidence:'market data and structure',counterEvidence:'counterevidence examined',alternatives:Object.fromEntries('ABCDE'.split('').map(x=>[x,{judgement:'HOLD',evidence:'analysed against original V1.7'}]))}]));
const checks=Object.fromEntries(Array.from({length:16},(_,i)=>[String(i+1),{status:'CONFIRMED',evidence:'reviewed'}]));
const clauses=Object.fromEntries([...Array.from({length:24},(_,i)=>String(i)),'APPENDIX_A','APPENDIX_B','FINAL_EXECUTION'].map(x=>[x,{status:'CONFIRMED',evidence:'reviewed'}]));
const ana=()=>({reportMarkdown:'Current official report with prices and status; prior historical outcome 미확인',directions,coupling:{status:'MIXED',evidence:'independent BTC ETH price relationship checked'},checks,clauses,audit20:{rules:'check original signal ledger',bias:'considered bullish and bearish',evidence:'price path analysed'},conclusions:{BTC:{decision:'HOLD',evidence:'quality unproven'},ETH:{decision:'SHORT',evidence:'existing signal checked'}},newSignals:[],events:[]});
function submission(input,decision){const prepared=prepare({raw:input,ledger:[old],previousSnapshot:prev});return {raw:input,session:'OFFICIAL_V17',policySha256:POLICY_SHA256,snapshotSha256:prepared.snapshot.sha256,beforeLedgerSha256:ledgerHash([old]),decisionKey:prepared.decisionKey,analysis:decision};}

test('continuous post-issue short entry touch is reported as required',()=>{
 const a=assessSnapshotEvidence(raw(),{ledger:[old],previousSnapshot:prev});
 assert.deepEqual(a.perSymbol.ETH.requiredEntrySignals,[old.id]);
 assert(a.warnings.includes('ETH:PENDING_ENTRY_TOUCH_MUST_BE_REGISTERED'));
});
test('official HOLD without required ENTERED event is rejected',()=>{
 const inp=raw();assert.throws(()=>validateOfficialSubmission(submission(inp,ana()),[old],prev),/CONFIRMED_PENDING_TOUCH_EVENT_MISSING/);
});
test('ENTERED event with price path evidence passes original immutable plan audit',()=>{
 const inp=raw();const a=ana();a.analysisReason='';a.events=[{id:old.id,kind:'ENTERED',at:t,reason:'Observed continuous price-zone crossing',evidence:'Path high2046 exceeds entryLow2045.20 after issue',pricePathEvidence:'22:20:56 to 22:25:10 path 2043..2046 entered without stop/TP'}];
 const v=validateOfficialSubmission(submission(inp,a),[old],prev);
 assert.equal(v.events.length,1);assert.equal(old.status,'PENDING');
});
test('no touched entry in continuous path does not force event',()=>{
 const inp=raw({ethHigh:2044.5});const a=assessSnapshotEvidence(inp,{ledger:[old],previousSnapshot:prev});
 assert.deepEqual(a.perSymbol.ETH.requiredEntrySignals,[]);
 assert.equal(validateOfficialSubmission(submission(inp,ana()),[old],prev).events.length,0);
});
test('entry+stop overlap in one aggregated path is ambiguous, not auto-success',()=>{
 const inp=raw({ethHigh:2050});const a=assessSnapshotEvidence(inp,{ledger:[old],previousSnapshot:prev});
 assert.deepEqual(a.perSymbol.ETH.ambiguousSignals,[old.id]);
 assert.deepEqual(a.perSymbol.ETH.requiredEntrySignals,[]);
 assert.throws(()=>validateOfficialSubmission(submission(inp,ana()),[old],prev),/ENTRY_EXIT_ORDER_REQUIRES_UNVERIFIED/);
 const unverified=ana();unverified.conclusions.ETH.decision='UNVERIFIED';unverified.historicalUncertainty={ETH:{unresolvedIds:[old.id],evidence:'same interval entry and stop crossing cannot be ordered'}};
 assert.equal(validateOfficialSubmission(submission(inp,unverified),[old],prev).events.length,0);
});
test('unknown history is disclosed and can be quarantined before new current signal',()=>{
 const inp=raw({ethStart:'2026-11-12 22:24:00',ethHigh:2043,ethLow:2041,ethHigh24:2084.59,ethLast:2041.7});
 const a=ana();a.conclusions.ETH.decision='SHORT';
 a.historicalUncertainty={ETH:{unresolvedIds:[old.id],evidence:'unobserved previous minutes may have contained entry/stop/TP; actual outcome unknown'}};
 const current={id:'ETH-NEW-DURING-GAP',issuedAt:'2026-11-12 22:25:20',dataAt:t,symbol:'ETHUSDT',direction:'SHORT',tradeType:'지지·저항 반응형',exchange:'Bybit',product:'USDT perpetual',entryLow:2041,entryHigh:2042,stop:2050,tp1:2035,tp2:2020,tp3:2000,status:'NEW_ENTRY',invalidation:'prior to first entry touch new structure invalidated',supportingEvidence:'new observed market structure',opposingEvidence:'opposing flow',issuedVersion:'V1.7',comparisonEvidence:'A/B/C/D/E reviewed',couplingEvidence:'BTC ETH independently examined',historicalExposureUnknownIds:[old.id],historicalExposureRiskDisclosure:'Old historical position may already be entered and not closed; no exchange account flatness is verified.',independentCurrentMarketEvidence:'Current direction derived from new snapshot; original historical outcome is not assumed.'};
 a.newSignals=[current];
 assert.throws(()=>validateOfficialSubmission(submission(inp,a),[old],prev),/OLD_SIGNAL_MUST_BE_ACCOUNTED_FOR_BEFORE_NEW_PLAN/);
 a.events=[{id:old.id,kind:'UNVERIFIED',at:t,reason:'Historical PENDING outcome cannot be settled across a real gap',evidence:'22:20:56 to 22:24:00 source gap, 24h high includes trade zone, exact touch/exit chronology unavailable',priorConfirmedState:'PENDING',uncertaintyScope:'ENTRY_OR_STOP_OR_TARGET_DURING_MISSING_WINDOW',marketReassessment:'New market data assessed independently; original trade not deemed closed.'}];
 const out=validateOfficialSubmission(submission(inp,a),[old],prev);
 assert.equal(out.newSignals.length,1);assert.equal(out.events[0].kind,'UNVERIFIED');assert.equal(old.status,'PENDING');
 const changed=structuredClone(a);changed.events[0].at='2026-11-12 22:24:55';
 assert.throws(()=>validateOfficialSubmission(submission(inp,changed),[old],prev),/QUARANTINE_TIME_MUST_EQUAL_SNAPSHOT/);
});
test('out-of-order historical snapshot cannot silently rewrite ledger',()=>{
 const inp=raw();const future={btcTime:'2026-11-12 22:50:00',ethTime:'2026-11-12 22:50:00'};
 assert.throws(()=>validateOfficialSubmission(submission(inp,ana()),[old],future),/OUT_OF_ORDER_SNAPSHOT/);
});
test('do not infer entry from 24h high alone when interval path is below entry',()=>{
 const inp=raw({ethHigh:2042.8,ethLow:2040,ethHigh24:2084.59});const a=assessSnapshotEvidence(inp,{ledger:[old],previousSnapshot:prev});
 assert.deepEqual(a.perSymbol.ETH.requiredEntrySignals,[]);
 assert.deepEqual(a.perSymbol.ETH.ambiguousSignals,[]);
});

const entered={...old,status:'ENTERED',verifiedUntil:'2026-11-12 22:20:56'};
function enteredArgs(inp,a){const pr=prepare({raw:inp,ledger:[entered],previousSnapshot:prev});return {raw:inp,session:'OFFICIAL_V17',policySha256:POLICY_SHA256,snapshotSha256:pr.snapshot.sha256,beforeLedgerSha256:ledgerHash([entered]),decisionKey:pr.decisionKey,analysis:a};}
test('continuous entered SHORT stop touch is detected independently from AI wording',()=>{
 const inp=raw({ethHigh:2050,ethLow:2040});const q=assessSnapshotEvidence(inp,{ledger:[entered],previousSnapshot:prev});
 assert.deepEqual(q.perSymbol.ETH.requiredExitSignals,[{id:entered.id,kind:'STOPPED'}]);
 assert.throws(()=>validateOfficialSubmission(enteredArgs(inp,ana()),[entered],prev),/CONFIRMED_EXIT_OR_TP_EVENT_MISSING/);
});
test('continuous entered SHORT TP1 touch must be reported',()=>{
 const inp=raw({ethHigh:2048,ethLow:2037});const q=assessSnapshotEvidence(inp,{ledger:[entered],previousSnapshot:prev});
 assert.deepEqual(q.perSymbol.ETH.requiredExitSignals,[{id:entered.id,kind:'TP1'}]);
 assert.throws(()=>validateOfficialSubmission(enteredArgs(inp,ana()),[entered],prev),/CONFIRMED_EXIT_OR_TP_EVENT_MISSING/);
});
test('continuous entered SHORT TP3 touch requires terminal completion or ambiguity',()=>{
 const inp=raw({ethHigh:2048,ethLow:2001});const q=assessSnapshotEvidence(inp,{ledger:[entered],previousSnapshot:prev});
 assert.deepEqual(q.perSymbol.ETH.requiredExitSignals,[{id:entered.id,kind:'TP3_DONE'}]);
 assert.throws(()=>validateOfficialSubmission(enteredArgs(inp,ana()),[entered],prev),/CONFIRMED_EXIT_OR_TP_EVENT_MISSING/);
});
test('both stop and TP touched: chronology ambiguous, no fabricated completion',()=>{
 const inp=raw({ethHigh:2051,ethLow:2000});const q=assessSnapshotEvidence(inp,{ledger:[entered],previousSnapshot:prev});
 assert.deepEqual(q.perSymbol.ETH.requiredExitSignals,[]);
 assert.deepEqual(q.perSymbol.ETH.ambiguousSignals,[entered.id]);
 assert.throws(()=>validateOfficialSubmission(enteredArgs(inp,ana()),[entered],prev),/ENTRY_EXIT_ORDER_REQUIRES_UNVERIFIED/);
});
