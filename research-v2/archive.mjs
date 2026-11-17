// Public downloadable research archives, separately documented by Bybit.
// Does not route around REST permissions, switch venue or claim live availability.
import {mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {resolve,join} from 'node:path';
export function aggregateCSV(text,symbol){
 const rows=text.trim().split(/\r?\n/),header=rows.shift().split(','),index=Object.fromEntries(header.map((x,i)=>[x,i]));
 for(const k of ['timestamp','symbol','side','size','price','trdMatchID'])if(!(k in index))throw Error('ARCHIVE_FIELD_MISSING_'+k);
 const bars=new Map();let previous=-Infinity,count=0,first=null,last=null;
 const seen=new Set();
 for(const line of rows){
 const x=line.split(','),timestamp=Number(x[index.timestamp])*1000,price=Number(x[index.price]),size=Number(x[index.size]),side=x[index.side],id=x[index.trdMatchID];
 if(x[index.symbol]!==symbol||!Number.isFinite(timestamp)||timestamp<previous||![price,size].every(v=>Number.isFinite(v)&&v>0)||!['Buy','Sell'].includes(side)||!id||seen.has(id))throw Error('ARCHIVE_TRADE_INVALID_OR_UNORDERED');
 seen.add(id);previous=timestamp;count++;if(first===null)first=timestamp;last=timestamp;
 const start=Math.floor(timestamp/60000)*60000;
 let b=bars.get(start);if(!b){b={start,open:price,high:price,low:price,close:price,volume:0,buy:0,sell:0,count:0};bars.set(start,b);}
 b.high=Math.max(b.high,price);b.low=Math.min(b.low,price);b.close=price;b.volume+=size;b.count++;b[side==='Buy'?'buy':'sell']+=size;
 }
 return {symbol,tradeCount:count,firstTrade:first,lastTrade:last,bars:[...bars.values()].map(x=>({...x,cvdDelta:x.buy-x.sell})),
 provenance:'PUBLIC_BYBIT_DAILY_TRADE_ARCHIVE_RETROSPECTIVE',sameTimestampOrder:'FILE_ORDER_ONLY_NOT_VERIFIED_MATCHING_SEQUENCE',
 coverage:'Archive completeness not independently attested; absent trade minutes are not fabricated candles.'};
}
async function download(url,maxBytes){
 const r=await fetch(url,{signal:AbortSignal.timeout(20000)});
 if(!r.ok)throw Error('ARCHIVE_HTTP_'+r.status);
 if(Number(r.headers.get('content-length'))>maxBytes){await r.body.cancel();throw Error('ARCHIVE_SIZE_LIMIT');}
 const chunks=[];let length=0;for await(const chunk of r.body){length+=chunk.length;if(length>maxBytes)throw Error('ARCHIVE_SIZE_LIMIT');chunks.push(chunk);}
 return Buffer.concat(chunks);
}
export async function collectArchive(output){
 const dir=resolve(output);await mkdir(dir,{recursive:true});const results=[];
 for(const symbol of ['BTCUSDT','ETHUSDT']){
 const date='2026-11-15',url='https://public.bybit.com/trading/'+symbol+'/'+symbol+date+'.csv.gz';
 const row={symbol,date,url,collectedAt:new Date().toISOString(),incidentDay:false,liveVintageVerified:false};
 try{
 const bytes=await download(url,32*1024*1024);row.bytes=bytes.length;row.rawSha256=createHash('sha256').update(bytes).digest('hex');
 await writeFile(join(dir,symbol+date+'.csv.gz'),bytes);
 const data=aggregateCSV(gunzipSync(bytes,{maxOutputLength:256*1024*1024}).toString('utf8'),symbol);
 await writeFile(join(dir,symbol+date+'-1m.json'),JSON.stringify(data,null,2)+'\n');
 Object.assign(row,{status:'CAPTURED_RETROSPECTIVE',tradeCount:data.tradeCount,barCount:data.bars.length,firstTrade:data.firstTrade,lastTrade:data.lastTrade});
 }catch(e){row.status='UNAVAILABLE_OR_INVALID';row.reason=String(e.message);}
 results.push(row);
 }
 const result={results,incidentArchiveStatus:'2026-11-16 not present in directory at audit time; never substitute 2026-11-15 as the incident day.',backtestStatus:'NO_NEW_STRATEGY_PERFORMANCE_TEST'};
 await writeFile(join(dir,'archive-manifest.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));return result;
}
if(process.argv[1]&&resolve(process.argv[1])===resolve(new URL(import.meta.url).pathname))await collectArchive(process.argv[2]??'research-output/archive');
