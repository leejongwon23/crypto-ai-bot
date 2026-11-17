import test from 'node:test';
import assert from 'node:assert/strict';
import {canonical,hash,normalizeKlines,visibleAt,appendEvidence,verifyEvidence,takerFlow,bestQuoteOFI,absorptionHypotheses,ksStatistic,purgedWalkForward,holm,movingBlockMeanCI,netTrade,promotionGate} from './lib.mjs';
const market={exchange:'Bybit',category:'linear',symbol:'ETHUSDT'};
const t=Date.parse('2026-11-16T21:00:00Z');
const opts={symbol:'ETHUSDT',interval:'1',collectedAt:'2026-11-16T21:03:00Z',from:t,to:t+120000};
const candle=(start)=>[String(start),'100','102','99','101','10','1005'];
const body=rows=>({retCode:0,result:{category:'linear',symbol:'ETHUSDT',list:rows}});
test('canonical digest rejects undefined and nonfinite instead of ambiguous encoding',()=>{
 assert.equal(canonical({b:2,a:1}),'{"a":1,"b":2}');
 assert.equal(hash({a:1,b:2}),hash({b:2,a:1}));
 for(const x of [undefined,NaN,Infinity,{a:undefined}])assert.throws(()=>hash(x));
});
test('full OHLCV sorts and reports no fabricated missing bars',()=>{
 const r=normalizeKlines(body([candle(t+60000),candle(t)]),opts);
 assert.equal(r.bars[0].start,t);assert.equal(r.missingRatio,0);assert.equal(r.liveVintageVerified,false);
 const missing=normalizeKlines(body([candle(t)]),opts);assert.deepEqual(missing.missing,[t+60000]);assert.equal(missing.missingRatio,.5);
});
test('wrong venue product, duplicates, malformed, negative volume, unclosed candle reject',()=>{
 assert.throws(()=>normalizeKlines({...body([]),result:{category:'spot',symbol:'ETHUSDT',list:[]}},opts));
 assert.throws(()=>normalizeKlines(body([candle(t),candle(t)]),opts),/DUPLICATE/);
 const bad=candle(t);bad[5]='-1';assert.throws(()=>normalizeKlines(body([bad]),opts));
 assert.throws(()=>normalizeKlines(body([candle(t).slice(0,5)]),opts),/OHLCV/);
 assert.throws(()=>normalizeKlines(body([candle(t)]),{...opts,collectedAt:'2026-11-16T21:00:30Z'}),/OPEN/);
});
test('late data cannot become vintage features; replay is explicitly separate',()=>{
 const data=[{eventTime:t,knownAt:t+60000,liveVintageVerified:true},{eventTime:t,knownAt:t,liveVintageVerified:false}];
 assert.deepEqual(visibleAt(data,'2026-11-16T21:00:30Z'),[]);
 assert.equal(visibleAt(data,'2026-11-16T21:00:30Z',{mode:'REPLAY'}).length,1);
 assert.throws(()=>visibleAt([{eventTime:t}],'2026-11-16T21:00:30Z'));
});
test('append-only function does not mutate frozen originals; pinned hash detects tampering',()=>{
 const original={id:'OLD',stop:95};Object.freeze(original);
 const a=appendEvidence([],original),b=appendEvidence(a,{outcome:'STOPPED'});
 assert.equal(a.length,1);assert.equal(b.length,2);assert.ok(verifyEvidence(b));
 b[0].payload.stop=90;assert.equal(verifyEvidence(b),false);assert.equal(original.stop,95);
});
test('taker negative flow and positive response is a hypothesis, never SHORT approval',()=>{
 const trades=[{...market,id:'1',eventTime:t,price:100,size:2,side:'Sell'}];
 const flow=takerFlow(trades,{market,from:t,to:t+1,complete:true});
 assert.equal(flow.delta,-2);assert.equal(absorptionHypotheses({flow,startPrice:100,endPrice:101}).tradeApproval,false);
 assert.equal(takerFlow(trades,{market,from:t,to:t+1,complete:false}).status,'UNVERIFIED');
 assert.throws(()=>takerFlow([...trades,...trades],{market,from:t,to:t+1,complete:true}),/DUPLICATE/);
});
test('OFI respects quote state, continuity, and cannot identify actual order cancellation',()=>{
 const a={...market,bid:100,ask:101,bidSize:5,askSize:8,sequence:1,eventTime:t,epoch:'A',continuityVerified:true};
 const b={...a,bidSize:7,askSize:6,sequence:2,eventTime:t+1};
 assert.equal(bestQuoteOFI(a,b).ofi,4);
 assert.equal(bestQuoteOFI(a,{...b,epoch:'B'}).status,'UNVERIFIED');
 assert.throws(()=>bestQuoteOFI(a,{...b,sequence:1}));
});
test('KS handles ties as empirical CDF distance; no false IID p-value',()=>{
 assert.equal(ksStatistic([1,1],[1,1]).statistic,0);
 assert.equal(ksStatistic([0,0],[1,1]).statistic,1);
 assert.equal(ksStatistic([1],[2]).pValue,null);
});
test('walk-forward purges overlapping labels and feature windows, embargo, untouched holdout',()=>{
 const events=[{featureStart:0,decisionAt:1,labelEnd:4},{featureStart:3,decisionAt:4,labelEnd:10},
 {featureStart:8,decisionAt:10,labelEnd:12},{featureStart:10,decisionAt:11,labelEnd:13}];
 const [f]=purgedWalkForward(events,[{trainStart:0,testStart:10,testEnd:20}],{embargoMs:2,holdoutStart:30});
 assert.equal(f.train.length,1);assert.equal(f.test.length,1);
 assert.throws(()=>purgedWalkForward(events,[{trainStart:0,testStart:10,testEnd:31}],{embargoMs:2,holdoutStart:30}));
});
test('Holm rejects no more than ordered corrected tests',()=>{
 assert.deepEqual(holm([.001,.03,.04],.05),[true,false,false]);
 assert.deepEqual(holm([.001,.01,.04],.05),[true,true,true]);
});
test('block bootstrap is deterministic given locked RNG and uses blocks',()=>{
 const r=movingBlockMeanCI([1,1,1,1],{blockLength:2,reps:100,alpha:.05,random:()=>.5});
 assert.equal(r.lower,1);assert.equal(r.upper,1);assert.equal(r.method,'moving non-circular block bootstrap');
});
test('long and short include explicit adverse execution, fees and signed funding',()=>{
 const costs={entryFeeBps:5,exitFeeBps:5,entrySlippageBps:10,exitSlippageBps:10,fundingCash:-.1};
 for(const [direction,exit] of [['LONG',110],['SHORT',90]]){
 const r=netTrade({direction,entry:100,exit,quantity:1,outcomeVerified:true},costs);
 assert.ok(r.net<10);assert.ok(r.fees>0);
 }
 assert.throws(()=>netTrade({direction:'LONG',entry:100,exit:110,quantity:1,outcomeVerified:true},{}),/EXPLICIT_COST/);
 assert.throws(()=>netTrade({direction:'LONG',entry:100,exit:110,quantity:1,outcomeVerified:false},costs));
});
test('draft protocol and no evidence cannot promote or deploy',()=>{
 const r=promotionGate({status:'DRAFT',spec:{}},{});
 assert.equal(r.eligibleForHumanReview,false);assert.equal(r.deployAuthorized,false);assert.ok(r.reasons.includes('PROTOCOL_NOT_LOCKED'));
});
test('good win-rate or green unit tests alone never authorize deployment',()=>{
 const spec={userApproved:true,minTradesPerRegime:10,maxDrawdown:.1,maxMissingRatio:0,alpha:.05,embargoMs:60000,blockLength:5};
 const protocol={status:'LOCKED',spec,protocolHash:hash(spec)};
 const r=promotionGate(protocol,{unit:true,winRate:.9});
 assert.equal(r.eligibleForHumanReview,false);assert.equal(r.deployAuthorized,false);
});
