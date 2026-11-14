import test from 'node:test';
import assert from 'node:assert/strict';
import {parseSubmittedMinutePaths} from './submitted-minute-path.mjs';
import {planVerifiedMechanicalEvents} from './official-price-tracker.mjs';
import {checkEvent} from './v17-core.mjs';
import {kstMs,kstTime} from './price-path-recovery.mjs';

const t='2026-11-14 ';
const prev={btcTime:t+'18:59:18',ethTime:t+'18:59:28'};
const current={BTC:'2026-11-15 01:21:56',ETH:'2026-11-15 01:22:13'};
const original={id:'V17-ETH-LONG-20261114-185928-RETEST19095',
 symbol:'ETHUSDT',direction:'LONG',issuedAt:prev.ethTime,entryLow:1909,
 entryHigh:1910.5,stop:1897.5,tp1:1919.14,tp2:1936.86,tp3:1952.21};
const pending={id:original.id,symbol:original.symbol,original,status:'PENDING',verifiedUntil:prev.ethTime};

function source({missingMinute=null,entryAndStopSameMinute=false,corruptRange=false}={}){
 const start=kstMs(t+'18:59:00'),end=kstMs(current.ETH),lines=[];
 let last=1912.1;
 for(let at=start;at<=end;at+=60000){
  const time=kstTime(at),key=time.slice(-5);
  const open=last;
  let high=open,low=open,close=open;
  if(key==='19:02'){high=1914.02;close=1911.84;low=1911.71;}
  if(key==='19:04'){low=1910.7;close=1910.71;}
  if(key==='19:05'){high=Math.max(open,1911.38);low=entryAndStopSameMinute?1896.12:1909.7;close=1909.9;}
  if(key==='21:57'){low=1899.97;close=1900.34;}
  if(key==='21:58'){low=1896.12;close=1896.89;}
  if(key==='01:12'){low=1813.76;close=1815.78;}
  if(key==='01:22')close=1827.85;
  high=Math.max(high,open,close);
  low=Math.min(low,open,close);
  if(time!==missingMinute)lines.push([time,open,high,low,close].join(','));
  last=close;
 }
 return '========== BTC (BTCUSDT) ==========\ntimestamp_kst: '+current.BTC+'\n'
  +'[1] REALTIME PRICE (AUTO)\nlastPrice: 65476.10\n'
  +'[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: 67368.30 @ '+prev.btcTime+
   ' | end_price: 65476.10 @ '+current.BTC+'\n'
  +'========== ETH (ETHUSDT) ==========\ntimestamp_kst: '+current.ETH+'\n'
  +'[1] REALTIME PRICE (AUTO)\nlastPrice: 1827.85\n'
  +'[10] PRICE PATH SINCE LAST RUN (AUTO)\nstart_price: 1912.43 @ '+prev.ethTime+
   ' | end_price: 1827.85 @ '+current.ETH+'\n'
  +'[10B] BYBIT 1M GAP EVIDENCE (AUTO)\n'
  +'source: Bybit linear ETHUSDT | coverage: COMPLETE | expected: 384 | received: 384\n'
  +'minute_range: 2026-11-14 18:59:00 ~ '+(corruptRange?'2026-11-15 01:21:00':'2026-11-15 01:22:00')+' KST\n'
  +'first_minute_partial: true | last_minute_partial: true\n'
  +'minute_OHLC_KST: timestamp,open,high,low,close\n'+lines.join('\n')+'\n'
  +'[7] HISTORICAL FLOW SNAPSHOT (AUTO)\n';
}

function audit(src=source()){
 const history=parseSubmittedMinutePaths(src,prev,[pending]);
 const result=planVerifiedMechanicalEvents({
  ledger:[pending],history,currentAtBySymbol:current,
  auditStarts:{[pending.id]:prev.ethTime}
 });
 return {history,result};
}

test('real 2026-11-14 ETH regression: enter 19:05, stop 21:58, never TP1',()=>{
 const {history,result}=audit();
 assert.equal(history.ETH.recovered,true);
 assert.equal(history.ETH.source,'USER_SUBMITTED_BYBIT_1M');
 assert.equal(history.ETH.barCount,383); // 01:22 last candle is still open
 assert.deepEqual(result.events.map(e=>[e.kind,e.observedMinute,e.at]),[
  ['ENTERED',t+'19:05:00',t+'19:06:00'],
  ['STOPPED',t+'21:58:00',t+'21:59:00']
 ]);
 const copy=[structuredClone(pending)];
 for(const evt of result.events){
  assert.doesNotThrow(()=>checkEvent(evt,copy));
  copy[0].status=evt.kind;copy[0].verifiedUntil=evt.at;
 }
 assert.equal(copy[0].status,'STOPPED');
});

test('missing one of the 384 rows is rejected, never turned into confirmed entry',()=>{
 const {history}=audit(source({missingMinute:t+'20:30:00'}));
 assert.equal(history.ETH.recovered,false);
 assert.match(history.ETH.reason,/CONTIGUOUS|INCOMPLETE/);
});

test('incorrect claimed minute range is rejected',()=>{
 const {history}=audit(source({corruptRange:true}));
 assert.equal(history.ETH.recovered,false);
 assert.equal(history.ETH.reason,'SUBMITTED_1M_RANGE_MISMATCH');
});

test('entry and stop inside same 1m cannot fabricate their order',()=>{
 const {history,result}=audit(source({entryAndStopSameMinute:true}));
 assert.equal(history.ETH.recovered,true);
 assert.equal(result.events.some(x=>x.kind==='STOPPED'),false);
 assert.equal(result.events.some(x=>x.kind==='UNVERIFIED'),true);
});

test('without user 1m evidence fallback remains unverified, no fake events',()=>{
 const src=source().replace(/\[10B\][\s\S]*?\[7\] HISTORICAL FLOW SNAPSHOT/,'[7] HISTORICAL FLOW SNAPSHOT');
 const {history,result}=audit(src);
 assert.equal(history.ETH.recovered,false);
 assert.deepEqual(result.events,[]);
});

test('after losing a live signal, an original terminal is never reopened',()=>{
 const closed={...pending,status:'STOPPED',verifiedUntil:t+'21:59:00'};
 const history=parseSubmittedMinutePaths(source(),prev,[closed]);
 assert.equal(history.ETH.needed,false);
 assert.equal(history.ETH.recovered,false);
});
