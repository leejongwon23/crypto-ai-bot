import test from 'node:test';
import assert from 'node:assert/strict';
import {screenReversalEntries,validateReversalReview} from './reversal-entry-audit.mjs';
import {POLICY_SHA256} from './policy-v17.mjs';

// Pure synthetic regression fixtures; never executed against a live trade account.
function part(sym,price,low,high,order,cvd5,cvd15,hist1,hist5,vwap){
 const top='2026-11-15 14:08:00',bottom='2026-11-15 14:16:00';
 const highAt=order==='HIGH_FIRST'?top:bottom;
 const lowAt=order==='HIGH_FIRST'?bottom:top;
 return [
 '========== '+sym+' ('+sym+'USDT) ==========',
 'timestamp_kst: 2026-11-15 14:18:40',
 '[1] REALTIME PRICE (AUTO)',
 'lastPrice: '+price+' | mark: '+price+' | index: '+price,
 '[2] OHLCV + EMA/MACD (AUTO)',
 '1m: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist='+hist1+' align=BEAR',
 '5m: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist='+hist5+' align=BULL',
 '15m: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist=1 align=BULL',
 '30m: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist=1 align=BEAR',
 '1h: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist=1 align=BEAR',
 '4h: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist=-1 align=BEAR',
 '1D: n=200 close='+price+' ema12=1 ema26=2 macd=1 sig=1 hist=-1 align=BEAR',
 '[2B] ATR/VOL + SWING LEVELS (AUTO)',
 '15m: atr14=510 | vol60(ret%)=1 | swingHigh='+(sym==='BTC'?66178.6:1859.16)+' | swingLow='+(sym==='BTC'?61899.2:1709.06),
 '[5B] VWAP + KEY LEVELS + TAKER FLOW (AUTO)',
 'vwap_24h: '+vwap+' | vwap_session_6h: '+vwap,
 'prev_day_high/low: '+(sym==='BTC'?'68423.60 / 63269.90':'1936.86 / 1726.89'),
 'session_6h_high/low: '+(sym==='BTC'?'65565.00 / 61899.20':'1833.43 / 1709.06'),
 '[CVD] SERIES METRICS (AUTO, exact time-window aggTrades)',
 'cvd_net_snapshot(5m): '+cvd5+' | buy: 20 | sell: 10',
 'cvd_net_snapshot(15m): '+cvd15+' | buy: 20 | sell: 10',
 'cvd_fetch_fallbacks: 0 | cvd_reliability: HIGH | cvd_state_neutralize: FALSE',
 '[10] PRICE PATH SINCE LAST RUN (AUTO)',
 'start_price: '+price+' @ 2026-11-15 14:01:40 | end_price: '+price+' @ 2026-11-15 14:18:40',
 'path_high/low: '+high+' @ '+highAt+' / '+low+' @ '+lowAt,
 'path_range_pct: 1% | net_change_pct: 0.1% | order: '+order,
 '[7] HISTORICAL FLOW SNAPSHOT (AUTO)',
 'price_change_5m/15m/30m/1h/4h: 0% / 1% / 2% / 3% / 4%',
 ].join('\n')+'\n';
}
const source=()=>part('BTC',64911.6,64700,64980.1,'LOW_FIRST',24,15,17,-36,64725.26)
 +'\n'+part('ETH',1809.43,1807.47,1822.16,'HIGH_FIRST',-797,-1039,-0.88,-2.75,1809.51);
test('legacy original V1.7 policy hash preserved',()=>assert.equal(POLICY_SHA256,'5fbd801f1378df7f584b248aadfaf98ae7e38d0bca782e3e09625bafe39e66ad'));
test('bottom LONG must be examined with bearish higher timeframe',()=>{
 const s=screenReversalEntries(source()).BTC.long;
 assert.equal(s.status,'REVIEW_REQUIRED');assert.equal(s.observed.ema4h,'BEAR');assert(s.referenceRR2>=2);
 assert.equal(s.stopNotChosen,true);
});
test('top SHORT must be examined before 5m EMA turns bearish',()=>{
 const s=screenReversalEntries(source()).ETH.short;
 assert.equal(s.status,'REVIEW_REQUIRED');assert.equal(s.observed.ema5,'BULL');
});
test('invalid or same-minute turning chronology cannot issue screen',()=>{
 const bad=source().replace('64980.1 @ 2026-11-15 14:16:00','64980.1 @ 2026-11-15 14:08:00');
 assert.equal(screenReversalEntries(bad).BTC.long.status,'WATCH');
});
test('unreliable CVD cannot produce automatic high confidence screen',()=>{
 const bad=source().replace('cvd_reliability: HIGH','cvd_reliability: LOW');
 assert.equal(screenReversalEntries(bad).BTC.long.status,'WATCH');
});
test('bare HOLD when reversal qualifies must be rejected',()=>{
 assert.throws(()=>validateReversalReview({newSignals:[]},screenReversalEntries(source())),/REVERSAL_REVIEW_REQUIRED_BTC_LONG/);
 const a={newSignals:[],reversalReview:{
  BTC:{LONG:{action:'HOLD',evidence:'observed bounce',counterEvidence:'higher trend down',blocker:'HIGHER_TF_DOWN',rejectionEvidence:'4h bearish'}},
  ETH:{SHORT:{action:'HOLD',evidence:'sell observed',counterEvidence:'retest uncertain',blocker:'RETEST',rejectionEvidence:'retest unverified'}}
 }};
 assert.throws(()=>validateReversalReview(a,screenReversalEntries(source())),/REVERSAL_HOLD_NEEDS_STRUCTURAL_OR_RISK_REJECTION_BTC_LONG/);
 a.reversalReview.BTC.LONG.blocker='STOP';
 assert.equal(validateReversalReview(a,screenReversalEntries(source())),true);
});
test('screen never manufactures an executable or formal signal',()=>{
 const q=screenReversalEntries(source()).BTC.long;
 assert.equal(q.stopNotChosen,true);
 assert.throws(()=>validateReversalReview({newSignals:[],reversalReview:{
  BTC:{LONG:{action:'ISSUE_NOW',evidence:'bounce',counterEvidence:'risk'}},
  ETH:{SHORT:{action:'HOLD',blocker:'TARGET',evidence:'sell',counterEvidence:'risk',rejectionEvidence:'no viable nearby TP2'}}
 }},screenReversalEntries(source())),/REVERSAL_FORMAL_SIGNAL_MISSING_BTC_LONG/);
});
