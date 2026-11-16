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

test('directional evidence reads price and CVD separately without fabricating win odds',()=>{
 const a=one('BTC',64927.9,279.714,77)
  .replace('15m: n=200 close=64927.9 ema12=64924.9 ema26=64922.9 hist=3 align=BEAR', '15m: n=200 close=64927.9 ema12=64924.9 ema26=64922.9 hist=3 align=BULL')
  .replace('30m: n=200 close=64927.9 ema12=64924.9 ema26=64922.9 hist=1 align=BEAR', '30m: n=200 close=64927.9 ema12=64924.9 ema26=64922.9 hist=1 align=BULL')
  +'price_change_5m/15m/30m/1h/4h: 0.02% / 0.15% / 0.31% / 0.12% / -0.9%\n'
  +'candle_structure_5m/15m: MIXED(HH5/HL5/LH5/LL5) / UP(HH5/HL6/LH3/LL2)\n';
 const r=buildOpportunityContext(a+one('ETH',1824.22,10.1636,-92));
 assert.equal(r.symbols.BTC.LONG.directionalEvidence.regime,'PRICE_AND_FLOW_ALIGNED');
 assert.equal(r.symbols.BTC.SHORT.directionalEvidence.regime,'PRICE_AND_FLOW_OPPOSE');
 assert.ok(!('winRate' in r.symbols.BTC.LONG.directionalEvidence));
});
test('first target reward vs stop risk is measured and visible, not hidden by TP2 RR',()=>{
 const a=buildOpportunityContext(raw).symbols.BTC.SHORT;
 for(const x of a.bestAlternatives){
  assert.ok(Number.isFinite(x.tp1R));
  assert.equal(x.tp1AtLeastOneR,x.tp1R>=1);
 }
});
test('opposing data remains reviewable with a directional evidence warning, never auto-pruned',()=>{
 const x=buildOpportunityContext(raw);
 assert.ok(x.symbols.BTC.LONG.bestAlternatives.length>0);
 assert.ok(x.symbols.BTC.SHORT.bestAlternatives.length>0);
 assert.equal(x.symbols.BTC.LONG.directionalEvidence.regime,'DATA_LIMITED');
});


const realistic10am= (sym,spot,atr,ema5,ema5b,ema15,ema15b,pathHigh,pathLow,sessionHigh,sessionLow,prevLow,swLow,vwap6,vwap24,cvd5,cvd15,cvd30)=>\`========== \${sym} (\${sym}USDT) ==========
timestamp_kst: 2026-11-16 10:03:09
lastPrice: \${spot} | mark: \${spot}
1m: n=200 close=\${spot} ema12=\${spot+1} ema26=\${spot+2} hist=-2 align=BEAR
5m: n=200 close=\${spot} ema12=\${ema5} ema26=\${ema5b} hist=-2 align=BEAR
15m: n=200 close=\${spot} ema12=\${ema15} ema26=\${ema15b} hist=-2 align=BEAR
30m: n=200 close=\${spot} ema12=\${ema15} ema26=\${ema15b} hist=-2 align=BULL
4h: n=200 close=\${spot} ema12=\${spot+30} ema26=\${spot+50} hist=-2 align=BEAR
1D: n=200 close=\${spot} ema12=\${spot+80} ema26=\${spot+100} hist=-2 align=BEAR
15m: atr14=\${atr} | swingHigh=\${sessionHigh} | swingLow=\${swLow}
vwap_24h: \${vwap24} | vwap_session_6h: \${vwap6}
session_6h_high/low: \${sessionHigh} / \${sessionLow}
prev_day_high/low: \${sessionHigh} / \${prevLow}
cvd_net_snapshot(5m): \${cvd5}
cvd_net_snapshot(15m): \${cvd15}
cvd_net_snapshot(30m): \${cvd30}
cvd_fetch_fallbacks: 0 | cvd_reliability: HIGH | cvd_state_neutralize: FALSE
path_high/low: \${pathHigh} @ 2026-11-16 09:04:00 / \${pathLow} @ 2026-11-16 09:43:00
price_change_5m/15m/30m/1h/4h: -0.2% / -0.3% / -0.5% / -0.9% / -1.0%
candle_structure_5m/15m: DOWN(HH2/HL2/LH10/LL10) / MIXED(HH2/HL4/LH6/LL4)
\`;
const case10am=realistic10am('BTC',64806.30,285.657,64946.87,65015.36,64978.69,64977.62,65371.5,64630.3,65485,64630.3,61899.2,64234.6,65048.45,64411.8,-0.7,2.95,-174.27)+
 realistic10am('ETH',1809.99,10.325,1818.06,1821.84,1820.13,1820.89,1837.99,1805.44,1842.5,1805.44,1709.06,1793,1824.47,1801.29,28.50,122.10,-745.21);
test('10:03 ETH does not promote 1837.99 SHORT retest requiring 2.71 ATR over nearby entry exploration',()=>{
 const x=buildOpportunityContext(case10am).symbols.ETH.SHORT;
 assert.equal(x.bestAlternatives[0].entryAccess.accessTier,'NOW');
 assert.ok(x.nearbyAlternatives.some(a=>a.entry===1818.06&&a.priceDistanceAtr<1));
 const far=x.distantContingencies.find(a=>a.entry===1837.99);
 assert.ok(far);
 assert.ok(far.entryAccess.distanceAtr>2.7);
 assert.equal(far.entryAccess.pathToEntry,'UP');
 assert.equal(far.entryAccess.preEntryCounterDirection,'LONG');
 assert.equal(far.entryAccess.momentumSupportsPath,false);
 assert.ok(x.primaryCandidate===null || x.primaryCandidate.priceDistanceAtr<=1);
 assert.equal(x.remoteScenarioRule.includes('CONDITIONAL'),true);
});
test('10:03 BTC source-anchored near retest precedes old remote swing-high retest',()=>{
 const x=buildOpportunityContext(case10am).symbols.BTC.SHORT;
 assert.ok(x.nearbyAlternatives.some(a=>a.entry===65048.45&&a.priceDistanceAtr<1));
 assert.ok(x.distantContingencies.some(a=>a.entry===65371.5&&a.priceDistanceAtr>1));
 assert.ok(x.bestAlternatives[0].priceDistanceAtr<=1);
});
test('near opportunity output is never silently replaced with far RR even when no near RR2 qualifies',()=>{
 const x=buildOpportunityContext(case10am);
 for(const s of ['BTC','ETH'])for(const d of ['LONG','SHORT']){
  const a=x.symbols[s][d];
  assert.ok(a.nearbyAlternatives.length>0);
  assert.ok(a.bestAlternatives.every(y=>y.entryAccess&&y.entryAccess.retestArrivalConfirmed===false));
  assert.ok(a.primaryCandidate===null||(a.primaryCandidate.meetsArithmeticRR2&&a.primaryCandidate.tp1AtLeastOneR&&a.primaryCandidate.priceDistanceAtr<=1));
 }
});
test('opposite direction remains independently available, not automatically labeled profitable',()=>{
 const x=buildOpportunityContext(case10am);
 assert.equal(x.symbols.ETH.LONG.directionalEvidence.price15mChangePct,-0.3);
 assert.ok(x.symbols.ETH.LONG.bestAlternatives.length>0);
 assert.ok(x.symbols.ETH.SHORT.distantContingencies.every(a=>!('winRate' in a)));
});
