// Descriptive evidence only: no decisions, fitting, optimization, strategy PnL or official writes.
import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {hash} from './lib.mjs';
export function aggregateMinutes(bars,minutes){
 if(!Number.isInteger(minutes)||minutes<=0)throw Error('INVALID_INTERVAL');
 const grouped=new Map();
 for(const b of bars){const start=Math.floor(b.start/(minutes*60000))*minutes*60000;
 if(!grouped.has(start))grouped.set(start,[]);grouped.get(start).push(b);}
 const complete=[],missing=[];
 for(const [start,rows]of grouped){
 rows.sort((a,b)=>a.start-b.start);
 if(rows.length!==minutes||rows.some((r,i)=>r.start!==start+i*60000)){missing.push(start);continue;}
 complete.push({start,end:start+minutes*60000,open:rows[0].open,close:rows.at(-1).close,
 high:Math.max(...rows.map(x=>x.high)),low:Math.min(...rows.map(x=>x.low)),
 volume:rows.reduce((a,x)=>a+x.volume,0),turnover:rows.reduce((a,x)=>a+x.turnover,0),
 delta:rows.reduce((a,x)=>a+x.cvdDelta,0)});
 }
 return {minutes,bars:complete,omittedIncompleteBuckets:missing};
}
export async function describe(dir){
 const output={purpose:'DESCRIPTIVE_RESEARCH_ONLY',dataYearsActuallyExamined:[2026],strategyPerformance:'UNVERIFIED',symbols:[]};
 for(const symbol of ['BTCUSDT','ETHUSDT']){
 try{
 const input=JSON.parse(await readFile(join(dir,symbol+'2026-11-15-1m.json'),'utf8'));
 const intervals=[1,5,15,60,240,1440].map(n=>aggregateMinutes(input.bars,n));
 await writeFile(join(dir,symbol+'-derived-intervals.json'),JSON.stringify({sourceDate:'2026-11-15',source:'Bybit daily public raw trades',provenance:'DERIVED_FROM_ARCHIVED_TRADES_NOT_REST_OHLCV',liveVintageVerified:false,intervals},null,2)+'\n');
 const windows=intervals.find(x=>x.minutes===15).bars;
 const sellWithRise=windows.filter(x=>x.delta<0&&x.close>=x.open).length;
 const buyWithFall=windows.filter(x=>x.delta>0&&x.close<=x.open).length;
 const volume=input.bars.reduce((a,x)=>a+x.volume,0),turnover=input.bars.reduce((a,x)=>a+x.turnover,0);
 output.symbols.push({symbol,date:'2026-11-15',rawTrades:input.tradeCount,observedMinutes:input.bars.length,
 fullIntervals:intervals.map(x=>({minutes:x.minutes,count:x.bars.length,omitted:x.omittedIncompleteBuckets.length})),
 sessionVwap:volume?turnover/volume:null,windows15m:windows.length,sellAggressionWithNonnegativeResponse:sellWithRise,buyAggressionWithNonpositiveResponse:buyWithFall,
 inference:'These contemporaneous counts challenge a CVD-sign-only interpretation; they do not establish absorption, intent, future direction or a profitable strategy.',dataHash:hash(input)});
 }catch(e){output.symbols.push({symbol,status:'UNAVAILABLE',reason:String(e.message)});}
 }
 await writeFile(join(dir,'descriptive-study.json'),JSON.stringify(output,null,2)+'\n');console.log(JSON.stringify(output,null,2));
}
if(process.argv[1]?.endsWith('/describe.mjs'))await describe(process.argv[2]??'research-output/archive');
