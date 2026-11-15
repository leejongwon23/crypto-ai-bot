// Rebuild independently verifiable closed-1m coverage from persisted, user-submitted
// Bybit-labelled snapshots. Never use aggregate path extrema to invent event order.
import {parseSubmittedMinutePaths} from './submitted-minute-path.mjs';
import {floorMinute,kstMs,kstTime,rawMarketWindows} from './price-path-recovery.mjs';

const sameCandle=(a,b)=>['start','open','high','low','close'].every(k=>a[k]===b[k]);

export function assembleSubmittedMinuteChains(savedRaws,previousSnapshot,latestRaw,ledger){
 const result={BTC:{recovered:false},ETH:{recovered:false}};
 if(!previousSnapshot)return result;
 const current=rawMarketWindows(latestRaw);
 const segments=[];
 for(const raw of [...savedRaws,latestRaw]){
  try{
   const windows=rawMarketWindows(raw);
   // Validate each saved snippet against its OWN declared market path start.
   // The previous official checkpoint may be hours earlier; requiring each
   // snippet to cover that entire interval would discard sound newer evidence.
   const ownStart={btcTime:windows.BTC.pathStart,ethTime:windows.ETH.pathStart};
   const parsed=parseSubmittedMinutePaths(raw,ownStart,ledger);
   segments.push(parsed);
  }catch{
   // A malformed saved snippet cannot contribute evidence.
  }
 }
 for(const sym of ['BTC','ETH']){
  const previous=previousSnapshot[sym==='BTC'?'btcTime':'ethTime'];
  if(!previous)continue;
  const from=floorMinute(kstMs(previous));
  const through=floorMinute(kstMs(current[sym].timestamp))-60000;
  if(through<from)continue;
  const merged=new Map();
  let conflicts=false,accepted=0;
  for(const section of segments){
   const source=section[sym];
   if(!source?.recovered)continue;
   accepted++;
   for(const candle of source.bars){
    if(candle.start<from||candle.start>through)continue;
    const previousCandle=merged.get(candle.start);
    if(previousCandle&&!sameCandle(previousCandle,candle)){
     conflicts=true;break;
    }
    merged.set(candle.start,candle);
   }
   if(conflicts)break;
  }
  if(conflicts){result[sym]={recovered:false,reason:'CONFLICTING_SAVED_1M_OHLC'};continue;}
  const bars=[];
  for(let t=from;t<=through;t+=60000){
   const candle=merged.get(t);
   if(!candle)break;
   bars.push(candle);
  }
  const needed=Math.floor((through-from)/60000)+1;
  if(bars.length!==needed){result[sym]={recovered:false,reason:'SAVED_1M_CHAIN_INCOMPLETE',coveredMinutes:bars.length,requiredMinutes:needed};continue;}
  result[sym]={recovered:true,source:'USER_SUBMITTED_BYBIT_1M',
   attribution:'USER_SUBMITTED_NOT_INDEPENDENTLY_FETCHED',
   bridgeCovered:true,barCount:bars.length,bars,
   coveredFromMs:from,coveredThroughMs:through,
   coveredFrom:kstTime(from),coveredThrough:kstTime(through),segmentsAccepted:accepted};
 }
 return result;
}
