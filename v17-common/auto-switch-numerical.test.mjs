import test from 'node:test';
import assert from 'node:assert/strict';
import {buildAutoSwitchContext,validateAutoSwitchAudit} from './auto-switch-audit.mjs';

const raw=`========== BTC (BTCUSDT) ==========
timestamp_kst: 2026-11-13 12:25:23
[1] REALTIME PRICE (AUTO)
lastPrice: 69204.30
[2] OHLCV + EMA/MACD (AUTO)
1m: n=200 close=69223.50 ema12=69435.71 ema26=69688.65 align=BEAR
5m: n=200 close=69223.50 ema12=69876.81 ema26=70272.99 align=BEAR
15m: n=200 close=69223.50 ema12=70379.47 ema26=70844.36 align=BEAR
[2B] ATR/VOL + SWING LEVELS (AUTO)
15m: atr14=425.957 | swingHigh=72205.40 | swingLow=68893.90
[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO)
vwap_24h: 71629.59 | vwap_session_6h: 70398.21
prev_day_high/low: 74194.30 / 71072.50
[10] PRICE PATH SINCE LAST RUN (AUTO)
path_high/low: 69946.10 @ 2026-11-13 12:13:00 / 68893.90 @ 2026-11-13 12:23:00
========== ETH (ETHUSDT) ==========
timestamp_kst: 2026-11-13 12:25:40
[1] REALTIME PRICE (AUTO)
lastPrice: 1957.36
[2] OHLCV + EMA/MACD (AUTO)
1m: n=200 close=1957.48 ema12=1965.19 ema26=1973.87 align=BEAR
5m: n=200 close=1957.48 ema12=1979.46 ema26=1990.75 align=BEAR
15m: n=200 close=1957.48 ema12=1992.73 ema26=2002.36 align=BEAR
[2B] ATR/VOL + SWING LEVELS (AUTO)
15m: atr14=14.0779 | swingHigh=2027.62 | swingLow=1944.10
[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO)
vwap_24h: 2008.12 | vwap_session_6h: 1990.83
prev_day_high/low: 2065.58 / 1982.36
[10] PRICE PATH SINCE LAST RUN (AUTO)
path_high/low: 1982.89 @ 2026-11-13 12:15:00 / 1944.10 @ 2026-11-13 12:23:00`;
const snapshot={btcTime:'2026-11-13 12:25:23',ethTime:'2026-11-13 12:25:40',btcPrice:69204.30,ethPrice:1957.36};
const ledger=[
 {id:'BTC-PENDING',symbol:'BTCUSDT',status:'PENDING',original:{direction:'SHORT',entryLow:71265,entryHigh:71285}},
 {id:'ETH-PENDING',symbol:'ETHUSDT',status:'PENDING',original:{direction:'SHORT',entryLow:2003.5,entryHigh:2005.5}}
];
const round=n=>Number(n.toFixed(5));
const plan=(entryLow,entryHigh,stop,tp1,tp2,tp3)=>{
 const risk=stop-entryLow,reward=entryLow-tp2;
 return {direction:'SHORT',entryLow,entryHigh,stop,tp1,tp2,tp3,rr2:round(reward/risk),stopPct:round(risk/entryLow*100),tp2Pct:round(reward/entryLow*100)};
};
function proposal(anchor,anchorPrice,p,rejectionAnchor,rejectionAnchorPrice){
 return {kind:'PLAN',anchor,anchorPrice,plan:p,
  evidence:'Price structure with observed EMA or VWAP at snapshot time',
  counterEvidence:'Rejection can occur without pre-touch retracement or because stop is too close',
  stopEvidence:'Stop compared to observed EMA or recent high',tp1Evidence:'Latest observed support',
  tp2Evidence:'Measured previous low',tp3Evidence:'Explicit projected extension not observed',
  retestEvidence:'Retest probability unknown and prior reaction alone not proof',tp2Kind:'OBSERVED',tp3Kind:'PROJECTED',
  rejectionEvidence:'Current observed price path does not confirm another retest; structural stop and target risks remain',
  rejectionFactors:['RETEST','STRUCTURE'],rejectionAnchor,rejectionAnchorPrice};
}
function good(){
 const ctx=buildAutoSwitchContext(raw,ledger,snapshot);
 const base=(sym,id,immediate,near)=>({id,
  distanceAtr:ctx[sym].pending[0].distanceAtr,distancePct:ctx[sym].pending[0].distancePct,
  immediate,near,selection:'KEEP',farEvidence:'Distant option requires separate future price event',
  oppositeEvidence:'Independent LONG is unsupported by down structure',
  keepEvidence:'Original signal not invalidated, new alternative not demonstrated superior',
  executionTimingEvidence:'Future issue only; cannot use retrospective price touch',
  candidateComparison:'Immediate/near/original fully compared for risk and target quality',
  couplingEvidence:'BTC and ETH decline together, but no direction copied'});
 const BTCi=proposal('SPOT',69204.3,plan(69204.3,69204.3,69688.65,69050,68893.9,68000),'PATH_HIGH',69946.1);
 const BTCn=proposal('EMA_1m_12',69435.71,plan(69425,69445,69700,69050,68893.9,68000),'SPOT',69204.3);
 const ETHi=proposal('SPOT',1957.36,plan(1957.36,1957.36,1979.46,1950,1944.1,1930),'PATH_HIGH',1982.89);
 const ETHn=proposal('VWAP_6h',1990.83,plan(1989,1992,2005.83,1973.18,1944.1,1930),'SPOT',1957.36);
 return {autoSwitchAudit:{
  BTC:{snapshotAt:snapshot.btcTime,price:snapshot.btcPrice,reviews:[base('BTC','BTC-PENDING',BTCi,BTCn)]},
  ETH:{snapshotAt:snapshot.ethTime,price:snapshot.ethPrice,reviews:[base('ETH','ETH-PENDING',ETHi,ETHn)]}
 }};
}
test('uses actual 12:25 prices and ATR instead of frozen original entry',()=>{
 const x=buildAutoSwitchContext(raw,ledger,snapshot);
 assert(Math.abs(x.BTC.pending[0].distance-2060.7)<1e-5);
 assert(Math.abs(x.ETH.pending[0].distance-46.14)<1e-5);
 assert.equal(x.ETH.anchors.VWAP_6h,1990.83);
 assert.equal(x.BTC.anchors.EMA_1m_12,69435.71);
});
test('existing PENDING with market structure cannot submit prose-only HOLD',()=>assert.throws(()=>validateAutoSwitchAudit({},raw,ledger,snapshot),/AUTO_SWITCH_AUDIT_REQUIRED/));
test('full immediate and nearer numerical comparisons allow evidence-based KEEP',()=>assert.doesNotThrow(()=>validateAutoSwitchAudit(good(),raw,ledger,snapshot)));
test('do not accept CVD LOW alone as rejection',()=>{
 const a=good();a.autoSwitchAudit.BTC.reviews[0].near.rejectionFactors=['CVD'];
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/AUTO_SWITCH_KEEP_REJECTED_PLAN/);
});
test('wrong old distance ATR is rejected',()=>{
 const a=good();a.autoSwitchAudit.ETH.reviews[0].distanceAtr=1;
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/AUTO_SWITCH_OLD_DISTANCE_METRIC_MISMATCH/);
});
test('near must be closer than original signal and source anchored',()=>{
 const a=good();a.autoSwitchAudit.BTC.reviews[0].near.plan=plan(71800,71820,72000,71700,71500,71300);
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/SWITCH_NEAR_NOT_CLOSER|SWITCH_NEAR_ANCHOR_TOO_FAR/);
});
test('server recomputes stop percent, TP2 percent and R:R',()=>{
 const a=good();a.autoSwitchAudit.ETH.reviews[0].near.plan.rr2=8;
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/SWITCH_NEAR_METRICS_NOT_RECALCULATED/);
});
test('invented entry anchor cannot pass current raw evidence check',()=>{
 const a=good();a.autoSwitchAudit.BTC.reviews[0].near.anchor='HIDDEN_LIQUIDITY';
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/SWITCH_NEAR_SOURCE_ANCHOR_NOT_IN_CURRENT_SNAPSHOT/);
});
test('redesign without atomic old terminal event and new ID rejected',()=>{
 const a=good();a.autoSwitchAudit.ETH.reviews[0].selection='REDESIGN';
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/AUTO_SWITCH_REDESIGN_REQUIRES_ATOMIC_EVENT/);
});
test('source-anchored NO_PLAN can acknowledge a missing TP2, not just CVD',()=>{
 const a=good(),s=a.autoSwitchAudit.BTC.reviews[0].near;
 s.kind='NO_PLAN';delete s.plan;s.missingLeg='TP2';s.blockers=['TARGET','STRUCTURE'];
 assert.doesNotThrow(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot));
 s.blockers=['CVD'];assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/GENERIC_CVD_HOLD_FORBIDDEN/);
});
test('no pending trade keeps legacy 4-direction workflow compatible',()=>assert.doesNotThrow(()=>validateAutoSwitchAudit({},raw,[],snapshot)));
test('no stale snapshot timestamp passes as a fresh auto-switch audit',()=>{
 const a=good();a.autoSwitchAudit.ETH.snapshotAt='2026-11-13 11:53:55';
 assert.throws(()=>validateAutoSwitchAudit(a,raw,ledger,snapshot),/AUTO_SWITCH_AUDIT_REQUIRED_ETH/);
});
