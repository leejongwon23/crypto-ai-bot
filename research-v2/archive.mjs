// Separate official public archives; no REST permission bypass or live-vintage claim.
import {mkdir,writeFile} from 'node:fs/promises';
import {createReadStream,createWriteStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {createGunzip} from 'node:zlib';
import {Readable,Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {createInterface} from 'node:readline';
import {resolve,join} from 'node:path';
function aggregator(symbol){
 let index=null,previous=-Infinity,count=0,first=null,last=null;const bars=new Map(),seen=new Set();
 return {
 add(line){
 if(!line.trim())return;
 if(index===null){
 index=Object.fromEntries(line.split(',').map((x,i)=>[x,i]));
 for(const k of ['timestamp','symbol','side','size','price','trdMatchID'])if(!(k in index))throw Error('ARCHIVE_FIELD_MISSING_'+k);
 return;
 }
 const x=line.split(','),timestamp=Number(x[index.timestamp])*1000,price=Number(x[index.price]),size=Number(x[index.size]),side=x[index.side],id=x[index.trdMatchID];
 if(x[index.symbol]!==symbol||!Number.isFinite(timestamp)||timestamp<previous||![price,size].every(v=>Number.isFinite(v)&&v>0)||!['Buy','Sell'].includes(side)||!id||seen.has(id))throw Error('ARCHIVE_TRADE_INVALID_OR_UNORDERED');
 seen.add(id);previous=timestamp;count++;if(first===null)first=timestamp;last=timestamp;
 const start=Math.floor(timestamp/60000)*60000;
 let b=bars.get(start);if(!b){b={start,open:price,high:price,low:price,close:price,volume:0,buy:0,sell:0,count:0};bars.set(start,b);}
 b.high=Math.max(b.high,price);b.low=Math.min(b.low,price);b.close=price;b.volume+=size;b.count++;b[side==='Buy'?'buy':'sell']+=size;
 },
 result(){
 if(index===null||count===0)throw Error('EMPTY_ARCHIVE');
 return {symbol,tradeCount:count,firstTrade:first,lastTrade:last,bars:[...bars.values()].map(x=>({...x,cvdDelta:x.buy-x.sell})),
 provenance:'PUBLIC_BYBIT_DAILY_TRADE_ARCHIVE_RETROSPECTIVE',sameTimestampOrder:'FILE_ORDER_ONLY_NOT_VERIFIED_MATCHING_SEQUENCE',
 coverage:'Archive completeness not independently attested; absent trade minutes are not fabricated candles.'};
 }};
}
export function aggregateCSV(text,symbol){const a=aggregator(symbol);for(const line of text.trim().split(/\r?\n/))a.add(line);return a.result();}
async function download(url,path,maxBytes){
 const r=await fetch(url,{signal:AbortSignal.timeout(60000)});
 if(!r.ok)throw Error('ARCHIVE_HTTP_'+r.status);
 const announced=Number(r.headers.get('content-length'));
 if(announced>maxBytes){await r.body.cancel();throw Error('ARCHIVE_SIZE_LIMIT_'+announced);}
 let bytes=0;const digest=createHash('sha256');
 const verify=new Transform({transform(chunk,enc,cb){bytes+=chunk.length;if(bytes>maxBytes)return cb(Error('ARCHIVE_SIZE_LIMIT'));digest.update(chunk);cb(null,chunk);}});
 await pipeline(Readable.fromWeb(r.body),verify,createWriteStream(path,{flags:'wx'}));
 return {bytes,rawSha256:digest.digest('hex')};
}
async function aggregateFile(path,symbol){
 const a=aggregator(symbol),source=createReadStream(path),unzip=createGunzip();let expanded=0;
 const input=source.pipe(unzip);
 const lines=createInterface({input,crlfDelay:Infinity});
 const timeout=setTimeout(()=>input.destroy(Error('ARCHIVE_PARSE_TIMEOUT')),60000);
 source.on('error',e=>input.destroy(e));
 input.on('data',chunk=>{expanded+=chunk.length;if(expanded>2*1024*1024*1024)input.destroy(Error('ARCHIVE_EXPANDED_SIZE_LIMIT'));});
 try{for await(const line of lines)a.add(line);return a.result();}
 finally{clearTimeout(timeout);lines.close();input.destroy();source.destroy();}
}
export async function collectArchive(output){
 const dir=resolve(output);await mkdir(dir,{recursive:true});const results=[];
 for(const symbol of ['BTCUSDT','ETHUSDT']){
 const date='2026-11-15',url='https://public.bybit.com/trading/'+symbol+'/'+symbol+date+'.csv.gz';
 const row={symbol,date,url,collectedAt:new Date().toISOString(),incidentDay:false,liveVintageVerified:false};
 try{
 const path=join(dir,symbol+date+'.csv.gz');Object.assign(row,await download(url,path,128*1024*1024));
 const data=await aggregateFile(path,symbol);
 await writeFile(join(dir,symbol+date+'-1m.json'),JSON.stringify(data,null,2)+'\n');
 Object.assign(row,{status:'CAPTURED_RETROSPECTIVE',tradeCount:data.tradeCount,barCount:data.bars.length,firstTrade:data.firstTrade,lastTrade:data.lastTrade});
 }catch(e){row.status='UNAVAILABLE_OR_INVALID';row.reason=String(e.message);}
 results.push(row);
 }
 const result={results,incidentArchiveStatus:'2026-11-16 not present in directory at audit time; never substitute 2026-11-15 as the incident day.',backtestStatus:'NO_NEW_STRATEGY_PERFORMANCE_TEST'};
 await writeFile(join(dir,'archive-manifest.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));return result;
}
if(process.argv[1]&&resolve(process.argv[1])===resolve(new URL(import.meta.url).pathname))await collectArchive(process.argv[2]??'research-output/archive');
