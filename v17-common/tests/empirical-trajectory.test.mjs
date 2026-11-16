import {test} from 'node:test';
import assert from 'node:assert/strict';
import {extractObservation,extractClosedMinuteCandles,buildEmpiricalForecast,publicForecast} from '../empirical-trajectory.mjs';
const fmt=x=>new Date(x+9*3600000).toISOString().slice(0,19).replace('T',' ');
const base=Date.parse('2026-11-01T00:00:00+09:00');
const one=(sym,at,partialRows=true)=>{const t=fmt(at),from=fmt(at-2*60000),mid=fmt(at-60000),now=fmt(at);
 return '========== '+sym+' ('+sym+'USDT) ==========\n'+
  'timestamp_kst: '+t+'\nlastPrice: 100 | mark: 100\n'+
  '5m: n=200 close=100 ema12=100.1 ema26=99.5 align=BULL\n'+
  '15m: n=200 close=100 ema12=99.7 ema26=99.4 align=BULL\n'+
  '15m: atr14=2 | swingHigh=103 | swingLow=97\n'+
  'vwap_24h: 99 | vwap_session_6h: 100.2\n'+
  'oi_change_5m/15m/30m/1h/4h: -0.01% / -0.04% / -0.06% / -0.20% / 0.04%\n'+
  'price_change_5m/15m/30m/1h/4h: -0.11% / -0.16% / -0.18% / -0.42% / -0.7%\n'+
  'cvd_net_snapshot(5m): -2 | buy: 40 | sell: 42\n'+
  'cvd_net_snapshot(15m): -12 | buy: 50 | sell: 62\n'+
  'cvd_net_snapshot(30m): -23 | buy: 55 | sell: 78\n'+
  (partialRows?'[10B] BYBIT 1M GAP EVIDENCE (AUTO)\n'+
  'source: Bybit linear '+sym+'USDT | coverage: COMPLETE | expected: 3 | received: 3\n'+
  'minute_range: '+from+' ~ '+now+' KST\n'+
  'first_minute_partial: true | last_minute_partial: true\n'+
  'minute_OHLC_KST: timestamp,open,high,low,close\n'+
  from+',100,101,99,100\n'+mid+',100,100.5,99.5,100\n'+now+',100,105,98,103\n':'');
};
const raw=one('BTC',base+100*60000)+one('ETH',base+100*60000);
test('source features are ATR-normalized and preserve observed negative historical returns',()=>{
 const b=extractObservation(raw,'BTC');
 assert.equal(b.price,100);assert.equal(b.atr,2);assert.equal(b.features.length,12);
 assert.equal(b.features[0]<0,true);assert.equal(b.features[8]<0,true);
 assert.equal(b.atMs,base+100*60000);
});
test('only closed non-boundary minute is retained, never a current partial candle or incomplete first minute',()=>{
 const b=extractClosedMinuteCandles(raw,'BTC');
 assert.equal(b.length,1);
 assert.equal(b[0].t,base+99*60000);
 assert.equal(b[0].high,100.5);
});
test('incomplete labels, wrong source and missing features are never silently fabricated',()=>{
 assert.deepEqual(extractClosedMinuteCandles(raw.replace('expected: 3 | received: 3','expected: 4 | received: 3'),'BTC'),[]);
 assert.deepEqual(extractClosedMinuteCandles(raw.replace('Bybit linear BTCUSDT','Exchange UNKNOWN'),'BTC'),[]);
 assert.equal(extractObservation(raw.replace('15m: atr14=2','15m: atr14=N/A'),'BTC'),null);
});
test('fresh unlabelled snapshot returns explicit insufficient empirical validation',()=>{
 const result=buildEmpiricalForecast({raw,symbol:'ETH'});
 assert.equal(result.horizons[60].status,'INSUFFICIENT_INDEPENDENT_SAMPLES');
 assert.equal(result.horizons[60].measuredDirection,null);
 const v=publicForecast(result,[{side:'SHORT',entry:103,stop:104,tp1:102,tp2:101,priceDistanceAtr:1.5,entryAnchor:'PATH_HIGH'}]);
 assert.equal(v.candidates[0].evidence.status,'INSUFFICIENT_VALIDATED_HISTORY');
 assert.ok(!('winRate' in v.horizons[60]));
});
test('chronological 1m outcomes can train only on genuinely prior non-overlapping observations',()=>{
 const h=[],bars=[];
 for(let i=0;i<51;i++){
  const t=base+i*3600000;
  const trend=i%4===0?1:-1;
  h.push({symbol:'BTC',atMs:t,price:100,atr:2,features:[trend,0,trend,0,0,0,0,0,0,0,0,0]});
  for(let j=1;j<=60;j++){
   const p=100+trend*2*j/60;
   bars.push({t:t+j*60000,open:p,high:p+.02,low:p-.02,close:p});
  }
 }
 const query=base+52*3600000;
 const res=buildEmpiricalForecast({raw:one('BTC',query,false)+one('ETH',query,false),symbol:'BTC',history:h,candles:bars});
 assert.equal(res.horizons[60].samples,51);
 assert.ok(res.horizons[60].tested>=10);
 assert.equal(['OUT_OF_SAMPLE_EDGE_OBSERVED','NO_OUT_OF_SAMPLE_EDGE'].includes(res.horizons[60].status),true);
 assert.equal(res.horizons[60].neighborCount,20);
 const published=publicForecast(res,[{side:'LONG',entry:100,stop:97,tp1:101,tp2:102,priceDistanceAtr:0,entryAnchor:'SPOT'}]);
 assert.equal(published.candidates[0].evidence.status,'HISTORICAL_PATH_FREQUENCIES');
 assert.equal(published.candidates[0].evidence.n,20);
 assert.ok(!('probabilityOfProfit' in published.candidates[0].evidence));
});
test('future-dated labels are excluded even if the database wrongly contains them',()=>{
 const future={symbol:'BTC',atMs:base+999*3600000,price:100,atr:2,features:Array(12).fill(0)};
 const x=buildEmpiricalForecast({raw,symbol:'BTC',history:[future],candles:[]});
 assert.equal(x.horizons[15].samples,0);
 assert.equal(x.horizons[60].status,'INSUFFICIENT_INDEPENDENT_SAMPLES');
});
