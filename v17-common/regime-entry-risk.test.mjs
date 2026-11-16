import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateRangeStopContext} from './regime-entry-risk.mjs';
const eth=`========== ETH (ETHUSDT) ==========
timestamp_kst: 2026-11-17 06:35:19
lastPrice: 1823.63 | mark: 1823.63 | index: 1824.58
5m: n=200 close=1823.63 ema12=1821.55 ema26=1820.44 macd=1.12 sig=0.59 hist=0.53 align=BULL
15m: n=200 close=1823.63 ema12=1820.60 ema26=1819.17 macd=1.43 sig=1.28 hist=0.15 align=BULL
30m: n=200 close=1823.63 ema12=1820.76 ema26=1818.20 macd=2.56 sig=2.11 hist=0.46 align=BULL
1h: n=200 close=1823.63 ema12=1819.65 ema26=1818.63 macd=1.02 sig=-0.44 hist=1.46 align=BULL
vwap_24h: 1819.37 | vwap_session_6h: 1821.04
session_6h_high/low: 1836.88 / 1809.22
`;
const btc=`========== BTC (BTCUSDT) ==========
timestamp_kst: 2026-11-17 06:35:13
lastPrice: 65169.70 | mark: 65175.01
5m: n=200 close=65169.70 ema12=65091.22 ema26=65031.22 align=BULL
15m: n=200 close=65169.70 ema12=65021.78 ema26=64890.73 align=BULL
30m: n=200 close=65169.70 ema12=64968.48 ema26=64826.03 align=BULL
1h: n=200 close=65169.70 ema12=64866.81 ema26=64847.88 align=BULL
vwap_24h: 64724.68 | vwap_session_6h: 64830.10
session_6h_high/low: 65293.70 / 64200.00
`;
const sig=(direction,stop,tactic='RANGE_BOUNDARY',symbol='ETHUSDT')=>({direction,stop,entryTactic:tactic,symbol});
test('Original 2026-11-17 06:35 ETH SHORT narrow stop is blocked',()=>{
 const r=evaluateRangeStopContext({raw:btc+'\\n'+eth,signal:sig('SHORT',1828.91)});
 assert.equal(r.allowed,false);assert.equal(r.reason,'RANGE_SHORT_STOP_INSIDE_UNCLEARED_6H_HIGH');
 assert.equal(r.observed.high,1836.88);
});
test('Wider ETH stop passes only this context gate; RR2 must still be checked separately',()=>{
 const r=evaluateRangeStopContext({raw:eth,signal:sig('SHORT',1839.5)});
 assert.equal(r.allowed,true);
 assert.ok((1823.63-1811.19)/(1839.5-1823.63)<2);
});
test('Not a blanket short veto when trend alignment reverses',()=>{
 const r=evaluateRangeStopContext({raw:eth.replaceAll('align=BULL','align=BEAR'),signal:sig('SHORT',1828.91)});
 assert.equal(r.allowed,true);
});
test('Trend continuation not affected',()=>assert.equal(evaluateRangeStopContext({raw:eth,signal:sig('SHORT',1828.91,'TREND_CONTINUATION')}).allowed,true));
test('Missing VWAP evidence cannot approve countertrend range stop',()=>{
 const r=evaluateRangeStopContext({raw:eth.replace('vwap_session_6h: 1821.04','vwap_session_6h: N/A'),signal:sig('SHORT',1828.91)});
 assert.equal(r.reason,'RANGE_COUNTERTREND_CONTEXT_MISSING');
});
test('Mirrored BTC long stop inside 6h lower structure blocked in bearish context',()=>{
 const bearish=btc.replaceAll('align=BULL','align=BEAR').replace('lastPrice: 65169.70','lastPrice: 64600.00');
 const r=evaluateRangeStopContext({raw:bearish,signal:sig('LONG',64600,'RANGE_BOUNDARY','BTCUSDT')});
 assert.equal(r.reason,'RANGE_LONG_STOP_INSIDE_UNCLEARED_6H_LOW');
});
