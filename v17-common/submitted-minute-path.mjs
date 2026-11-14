// Strict chronological input from the user's labelled Bybit 1m [10B] evidence.
// Only closed, contiguous candles may drive V1.7 mechanical state transitions.
// This is user-submitted data validation, NOT an independent Bybit API fetch.
import {kstMs,kstTime,floorMinute,verifyCandleCoverage,rawMarketWindows} from './price-path-recovery.mjs';

const MARKET=['BTC','ETH'];
const live=new Set(['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED']);
const minute=/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),([\d.]+),([\d.]+),([\d.]+),([\d.]+)$/;

function section(raw,symbol){
 const start=raw.indexOf('========== '+symbol+' ('+symbol+'USDT) ==========');
 if(start<0)return null;
 const end=symbol==='BTC'?raw.indexOf('========== ETH (ETHUSDT) ==========',start+1):raw.length;
 return raw.slice(start,end<0?undefined:end);
}

function one(raw,symbol,windows,previousSnapshot,ledger){
 const active=ledger.filter(s=>s.symbol===symbol+'USDT'&&live.has(s.status));
 if(!active.length)return {recovered:false,needed:false,reason:'NO_ACTIVE_SIGNAL'};
 const p=section(raw,symbol);
 if(!p)return {recovered:false,needed:true,reason:'MISSING_SYMBOL'};
 const start=p.indexOf('[10B] BYBIT 1M GAP EVIDENCE (AUTO)');
 if(start<0)return {recovered:false,needed:true,reason:'MISSING_SUBMITTED_1M'};
 const proof=p.slice(start);
 const head=new RegExp('^source:\\s*Bybit linear '+symbol+'USDT\\s*\\|\\s*coverage:\\s*COMPLETE\\s*\\|\\s*expected:\\s*(\\d+)\\s*\\|\\s*received:\\s*(\\d+)','m').exec(proof);
 const range=/^minute_range:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*~\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*KST/m.exec(proof);
 const header=/^minute_OHLC_KST:\s*timestamp,open,high,low,close\s*$/m.exec(proof);
 if(!head||!range||!header)return {recovered:false,needed:true,reason:'INCOMPLETE_SUBMITTED_1M_HEADER'};
 const expected=Number(head[1]),received=Number(head[2]);
 if(!(expected>0&&expected<=1500&&received===expected))return {recovered:false,needed:true,reason:'SUBMITTED_1M_COUNT_MISMATCH'};
 let tail=proof.slice(header.index+header[0].length).replace(/^\r?\n/,'');
 const rows=[];
 for(const line of tail.split(/\r?\n/)){
  const m=minute.exec(line.trim());
  if(!m)break;
  const t=kstMs(m[1]),[open,high,low,close]=m.slice(2).map(Number);
  if(m[1]!==kstTime(t)||t%60000!==0||
     ![open,high,low,close].every(v=>Number.isFinite(v)&&v>0)||
     low>Math.min(open,close)||high<Math.max(open,close)||low>high)
   return {recovered:false,needed:true,reason:'INVALID_SUBMITTED_1M_OHLC'};
  if(rows.length&&t!==rows[rows.length-1].start+60000)
   return {recovered:false,needed:true,reason:'SUBMITTED_1M_NOT_CONTIGUOUS'};
  rows.push({start:t,open,high,low,close});
 }
 if(rows.length!==expected)return {recovered:false,needed:true,reason:'SUBMITTED_1M_ROWS_INCOMPLETE'};
 if(kstTime(rows[0].start)!==range[1]||kstTime(rows.at(-1).start)!==range[2]||
    rows.at(-1).start-rows[0].start!==(rows.length-1)*60000)
  return {recovered:false,needed:true,reason:'SUBMITTED_1M_RANGE_MISMATCH'};
 const now=kstMs(windows[symbol].timestamp),lastClosed=floorMinute(now)-60000;
 const former=previousSnapshot?.[symbol==='BTC'?'btcTime':'ethTime'];
 const starts=active.map(x=>kstMs(x.verifiedUntil||x.original?.issuedAt||x.issuedAt)).filter(Number.isFinite);
 const from=former?kstMs(former):Math.min(...starts);
 if(!Number.isFinite(from)||rows[0].start>floorMinute(from)||rows.at(-1).start<lastClosed||
    !verifyCandleCoverage(rows,from,lastClosed))
  return {recovered:false,needed:true,reason:'SUBMITTED_1M_DOES_NOT_COVER_AUDIT_WINDOW'};
 const closed=rows.filter(x=>x.start<=lastClosed);
 if(!closed.length)return {recovered:false,needed:true,reason:'NO_CLOSED_MINUTE'};
 return {recovered:true,needed:true,source:'USER_SUBMITTED_BYBIT_1M',
  // A path from the previous official minute through the current last closed
  // minute bridges the current interval but is not third-party authenticated.
  bridgeCovered:true,barCount:closed.length,bars:closed,
  coveredFromMs:closed[0].start,coveredThroughMs:closed.at(-1).start,
  coveredFrom:kstTime(closed[0].start),coveredThrough:kstTime(closed.at(-1).start),
  attribution:'USER_SUBMITTED_NOT_INDEPENDENTLY_FETCHED'};
}

export function parseSubmittedMinutePaths(raw,previousSnapshot,ledger){
 const output={};
 const windows=rawMarketWindows(raw);
 for(const symbol of MARKET){
  try{output[symbol]=one(raw,symbol,windows,previousSnapshot,ledger);}
  catch(e){output[symbol]={recovered:false,needed:true,reason:'SUBMITTED_1M_PARSE_ERROR:'+String(e.message||e).slice(0,90)};}
 }
 return output;
}
