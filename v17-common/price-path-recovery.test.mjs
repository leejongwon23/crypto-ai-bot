import test from 'node:test';
import assert from 'node:assert/strict';
import {kstMs,kstTime,floorMinute,rawMarketWindows,parseBybitCandles,verifyCandleCoverage,auditMinuteCandles,recoverMarketGaps,fetchBybitCandles,recoverActiveSignalWindows,earliestUnknownGapOrigins} from './price-path-recovery.mjs';
const when=(s)=>kstMs('2026-11-13 '+s);
const candle=(time,lo=1940,hi=1950)=>({start:when(time),open:(lo+hi)/2,high:hi,low:lo,close:(lo+hi)/2});
const eth={symbol:'ETHUSDT',status:'ENTERED',original:{direction:'SHORT',issuedAt:'2026-11-13 19:01:00',entryLow:1944.8,entryHigh:1948.2,stop:1969,tp1:1933,tp2:1894.1,tp3:1882.5}};
const pending={...eth,status:'PENDING'};
function raw(btcStart='19:12:49',ethStart='19:12:59'){
 return ['BTC','ETH'].map((s,i)=>{
  const p=i?1951.1:69288.4,ts=i?'19:16:29':'19:16:19',start=i?ethStart:btcStart;
  return '========== '+s+' ('+s+'USDT) ==========\n'
   +'timestamp_kst: 2026-11-13 '+ts+'\n[1] REALTIME PRICE (AUTO)\nlastPrice: '+p
   +'\n[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: '+p
   +' @ 2026-11-13 '+start+' | end_price: '+p+' @ 2026-11-13 '+ts
   +'\npath_high/low: '+(i?1956.08:69473.2)+' @ 2026-11-13 19:13:00 / '
   +(i?1950.39:69271.5)+' @ 2026-11-13 19:15:00\n';
 }).join('\n');
}
test('KST times map to UTC',()=>{
 assert.equal(new Date(when('19:04:20')).toISOString(),'2026-11-13T10:04:20.000Z');
 assert.equal(kstTime(when('19:12:00')),'2026-11-13 19:12:00');
 assert.equal(floorMinute(when('19:04:20')),when('19:04:00'));
});
test('window parser extracts BTC and ETH independently',()=>{
 const x=rawMarketWindows(raw());
 assert.equal(x.ETH.pathStart,'2026-11-13 19:12:59');
 assert.equal(x.BTC.timestamp,'2026-11-13 19:16:19');
});
test('Bybit response validation rejects broken and incomplete candles',()=>{
 const stamp=when('19:05:00'),list=[[String(stamp),'1944','1949','1940','1945'],[String(stamp+60000),'1945','1948','1941','1946']];
 const parsed=parseBybitCandles({retCode:0,result:{list}},'ETH');
 assert.equal(verifyCandleCoverage(parsed,stamp,stamp+60000),true);
 assert.equal(verifyCandleCoverage(parsed,stamp,stamp+120000),false);
 assert.throws(()=>parseBybitCandles({retCode:1,result:{list}},'ETH'),/INVALID/);
 assert.throws(()=>parseBybitCandles({retCode:0,result:{list:[[String(stamp),'1944','1930','1940','1945']]}},'ETH'),/VALUES/);
});
test('no target crossed preserves ENTERED with full path',()=>{
 const a=auditMinuteCandles(eth,[candle('19:05:00',1942,1952),candle('19:06:00',1947,1959)],{afterMs:when('19:04:20')});
 assert.equal(a.status,'ENTERED');assert.deepEqual(a.events,[]);assert.equal(a.ambiguous,null);
});
test('PENDING first touch enters without additional signal gates',()=>{
 const a=auditMinuteCandles(pending,[candle('19:00:00',1938,1942),candle('19:01:00',1942,1946.04),candle('19:02:00',1942.12,1945)],{afterMs:when('19:00:00')});
 assert.equal(a.status,'ENTERED');assert.deepEqual(a.events.map(e=>e.kind),['ENTERED']);
});
test('pending entry plus exit in one minute is ambiguous',()=>{
 const a=auditMinuteCandles(pending,[candle('19:01:00',1940,1971)],{afterMs:when('19:00:00')});
 assert.equal(a.events.length,0);assert.equal(a.ambiguous.reason,'ENTRY_AND_EXIT_SAME_1M');
});
test('SHORT traverses TP1, TP2 and TP3 in chronological minutes',()=>{
 const a=auditMinuteCandles(eth,[candle('19:05:00',1931,1943),candle('19:06:00',1893,1932),candle('19:07:00',1870,1900)],{afterMs:when('19:04:20')});
 assert.equal(a.status,'TP3_DONE');assert.deepEqual(a.events.map(e=>e.kind),['TP1','TP2','TP3_DONE']);
});
test('stop after TP1 wins later correctly',()=>{
 const a=auditMinuteCandles(eth,[candle('19:05:00',1931,1943),candle('19:06:00',1940,1971)],{afterMs:when('19:04:20')});
 assert.deepEqual(a.events.map(e=>e.kind),['TP1','STOPPED']);
});
test('same-minute stop and TP cannot be falsely labeled as win',()=>{
 const a=auditMinuteCandles(eth,[candle('19:05:00',1930,1971)],{afterMs:when('19:04:20')});
 assert.equal(a.events.length,0);assert.equal(a.ambiguous.reason,'STOP_AND_TARGET_SAME_1M');
});
test('full missing gap is recovered without any imagined trades',async()=>{
 const mock=async(url)=>{
  const u=new URL(url),start=Number(u.searchParams.get('start')),end=Number(u.searchParams.get('end')),list=[];
  for(let n=start;n<=end;n+=60000)list.push([String(n),'1945','1950','1940','1946']);
  return {ok:true,json:async()=>({retCode:0,result:{list:list.reverse()}})};
 };
 const out=await recoverMarketGaps(raw(),{btcTime:'2026-11-13 19:04:11',ethTime:'2026-11-13 19:04:20'},{fetcher:mock});
 assert.equal(out.ETH.recovered,true);assert.equal(out.ETH.gapSeconds,519);
 assert.equal(out.BTC.recovered,true);
 assert.equal(auditMinuteCandles(eth,out.ETH.bars,{afterMs:when('19:04:20')}).status,'ENTERED');
});
test('incomplete API range never treated as covered',async()=>{
 const mock=async(url)=>{
  const u=new URL(url);
  return {ok:true,json:async()=>({retCode:0,result:{list:[[u.searchParams.get('start'),'1945','1950','1940','1946']]}})};
 };
 const out=await recoverMarketGaps(raw(),{btcTime:'2026-11-13 19:04:11',ethTime:'2026-11-13 19:04:20'},{fetcher:mock});
 assert.equal(out.ETH.recovered,false);assert.match(out.ETH.reason,/INCOMPLETE/);
});
test('HTTP failure cannot fabricate recovered price path',async()=>{
 const out=await recoverMarketGaps(raw(),{btcTime:'2026-11-13 19:04:11',ethTime:'2026-11-13 19:04:20'},{fetcher:async()=>({ok:false,status:403})});
 assert.equal(out.ETH.recovered,false);assert.match(out.ETH.reason,/BYBIT_HTTP_403/);
});
test('LONG stop and TP in same minute stays ambiguous',()=>{
 const long={status:'ENTERED',original:{direction:'LONG',entryLow:100,entryHigh:101,stop:95,tp1:110,tp2:120,tp3:130}};
 const a=auditMinuteCandles(long,[{start:when('19:05:00'),open:100,high:111,low:94,close:105}],{afterMs:when('19:04:20')});
 assert.equal(a.ambiguous.reason,'STOP_AND_TARGET_SAME_1M');
});
test('terminal stop precludes later target milestone',()=>{
 const a=auditMinuteCandles(eth,[candle('19:05:00',1940,1971),candle('19:06:00',1880,1900)],{afterMs:when('19:04:20')});
 assert.deepEqual(a.events.map(e=>e.kind),['STOPPED']);
});

test('fallback uses official Bybit backup only after primary HTTP 403 and verifies every 1m candle',async()=>{
 const start=when('19:04:00'),list=[0,1].map(i=>[String(start+i*60000),'1945','1950','1940','1944','1','100']);
 const seen=[];
 const result=await fetchBybitCandles({symbol:'ETH',fromMs:start,toMs:start+60000,
  fetcher:async url=>{seen.push(url);return seen.length===1?{ok:false,status:403}:{ok:true,json:async()=>({retCode:0,result:{list}})};}});
 assert.equal(result.length,2);
 assert.equal(new URL(seen[0]).hostname,'api.bybit.com');
 assert.equal(new URL(seen[1]).hostname,'api.bytick.com');
});
test('backup rejects incomplete 1m coverage and never invents a winning trade',async()=>{
 const start=when('19:04:00'),seen=[];
 await assert.rejects(fetchBybitCandles({symbol:'ETH',fromMs:start,toMs:start+120000,
  fetcher:async url=>{seen.push(url);return {ok:true,json:async()=>({retCode:0,result:{list:[
   [String(start),'1945','1950','1940','1944'],[String(start+120000),'1944','1950','1940','1945']
  ]}})};}}),/BYBIT_INCOMPLETE_1M_COVERAGE/);
 assert.equal(seen.length,2);
});
test('an ENTERED original has its earlier failed gap retained across later snapshots',()=>{
 const original={id:'ETH-LIVE',symbol:'ETHUSDT',status:'ENTERED'};
 const row=[{evidence:{perSymbol:{ETH:{priorOfficialAt:'2026-11-13 19:04:20',gapSeconds:519,
  couldNotVerifySignals:['ETH-LIVE']}}}}];
 const found=earliestUnknownGapOrigins([original],row);
 assert.equal(found['ETH-LIVE'],'2026-11-13 19:04:20');
 const terminal=earliestUnknownGapOrigins([{...original,status:'TP3_DONE'}],row);
 assert.deepEqual(terminal,{});
});
test('later 1m recovery includes old missing minute instead of querying only recent period',async()=>{
 const old=when('19:04:00'),list=Array.from({length:12},(_,i)=>[
  String(old+i*60000),'1945','1950','1940','1944','1','100']);
 const starts=[];
 const result=await recoverActiveSignalWindows(raw(),{ethTime:'2026-11-13 19:12:20',
  btcTime:'2026-11-13 19:12:20'},[{id:'ETH-LIVE',symbol:'ETHUSDT',
  status:'ENTERED',verifiedUntil:'2026-11-13 19:12:20'}],{
  unverifiedOrigins:{'ETH-LIVE':'2026-11-13 19:04:20'},
  fetcher:async url=>{starts.push(new URL(url).searchParams.get('start'));
   return {ok:true,json:async()=>({retCode:0,result:{list}})};}
 });
 assert.equal(result.ETH.recovered,true);
 assert.equal(result.ETH.coveredFrom,'2026-11-13 19:04:00');
 assert.equal(starts[0],String(old));
});
