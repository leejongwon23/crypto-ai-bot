import {test} from 'node:test';import assert from 'node:assert/strict';
import {buildOpportunityContext} from '../opportunity-engine.mjs';
const one=(sym,p,atr,cvd)=>`========== ${sym} (${sym}USDT) ==========
timestamp_kst: 2026-11-16 04:42:29
lastPrice: ${p} | mark: ${p}
1m: n=200 close=${p} ema12=${p+1} ema26=${p-1} hist=2 align=BULL
5m: n=200 close=${p} ema12=${p-4} ema26=${p-6} hist=2 align=BULL
15m: n=200 close=${p} ema12=${p-3} ema26=${p-5} hist=3 align=BEAR
30m: n=200 close=${p} ema12=${p-3} ema26=${p-5} hist=1 align=BEAR
4h: n=200 close=${p} ema12=${p+30} ema26=${p+50} hist=-2 align=BEAR
1D: n=200 close=${p} ema12=${p+80} ema26=${p+100} hist=-9 align=BEAR
15m: atr14=${atr} | swingHigh=${p+4*atr} | swingLow=${p-4*atr}
vwap_24h: ${p-atr*2} | vwap_session_6h: ${p-atr*.5}
session_6h_high/low: ${p+2.5*atr} / ${p-2.5*atr}
prev_day_high/low: ${p+6*atr} / ${p-6*atr}
cvd_net_snapshot(5m): ${cvd}
cvd_net_snapshot(15m): ${cvd*2}
cvd_net_snapshot(30m): ${cvd*3}
cvd_fetch_fallbacks: 0 | cvd_reliability: HIGH | cvd_state_neutralize: FALSE
path_high/low: ${p+atr} @ 2026-11-16 04:10:00 / ${p-atr} @ 2026-11-16 04:04:00
`;
const raw=one('BTC',64927.9,279.714,-77)+one('ETH',1824.22,10.1636,92);
test('both symbols, both LONG and SHORT searched',()=>{const r=buildOpportunityContext(raw);assert.equal(r.symbols.BTC.status,'SCREENED');assert.equal(r.symbols.ETH.status,'SCREENED');for(const s of ['BTC','ETH'])for(const d of ['LONG','SHORT'])assert.ok(r.symbols[s][d].generated>0)});
test('opposing flow NEVER removes an entire direction',()=>{const r=buildOpportunityContext(raw);assert.ok(r.symbols.BTC.LONG.generated>0);assert.ok(r.symbols.ETH.SHORT.generated>0)});
test('screen ranks arithmetic candidate without inventing success probabilities',()=>{const r=buildOpportunityContext(raw);for(const s of ['BTC','ETH'])for(const d of ['LONG','SHORT']){const o=r.symbols[s][d];assert.ok(o.bestAlternatives.every(x=>Number.isFinite(x.rr2)));assert.ok(o.bestAlternatives.every(x=>!('winRate' in x)))}});
test('explores at least one near and one now option',()=>{const r=buildOpportunityContext(raw);for(const s of ['BTC','ETH'])for(const d of ['LONG','SHORT']){const o=r.symbols[s][d];assert.ok(o.nowCandidates>0);assert.ok(o.bestAlternatives.length>0)}});
test('candidate anchors never manufacture price levels beyond given structures',()=>{const r=buildOpportunityContext(raw);for(const s of ['BTC','ETH'])for(const d of ['LONG','SHORT'])for(const a of r.symbols[s][d].bestAlternatives)assert.ok(a.tp2Anchor&&a.stopAnchor&&a.entryAnchor)});
test('ranking does not collapse after changing CVD sign',()=>{const x=buildOpportunityContext(one('BTC',64927.9,279.714,77)+one('ETH',1824.22,10.1636,-92));assert.ok(x.symbols.BTC.SHORT.generated>0);assert.ok(x.symbols.ETH.LONG.generated>0)});
test('missing ATR remains explicitly unverified',()=>{const x=buildOpportunityContext(one('BTC',64927.9,279.714,77)+one('ETH',1824.22,10.1636,-92).replace('atr14=10.1636','atr14=notavailable'));assert.equal(x.symbols.ETH.status,'UNVERIFIED')});
test('no past signal or stop mutation',()=>{const x=buildOpportunityContext(raw);assert.equal(x.policy,'PRESERVE_ISSUED_SIGNALS_AND_ORIGINAL_V17');assert.ok(!('newSignals' in x)&&!('events' in x))});
