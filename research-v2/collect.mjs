// Bounded one-shot public data collection. No account endpoint, key, trading or official DB.
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {normalizeKlines,intervalMs,hash} from './lib.mjs';
export async function request(path,params){
 const u=new URL(path,'https://api.bybit.com');for(const [k,v]of Object.entries(params))u.searchParams.set(k,String(v));
 const capturedAt=new Date().toISOString();
 try{
 const r=await fetch(u,{signal:AbortSignal.timeout(5000),headers:{accept:'application/json'}});
 const raw=await r.text();const collectedAt=new Date().toISOString();let data=null;try{data=JSON.parse(raw);}catch{}
 return {url:u.toString(),requestedAt:capturedAt,collectedAt,httpStatus:r.status,rateLimit:{remaining:r.headers.get('x-bapi-limit-status'),reset:r.headers.get('x-bapi-limit-reset-timestamp')},
 raw,rawSha256:hash(raw),data,error:r.ok&&data?.retCode===0?null:'HTTP_OR_PROVIDER_ERROR'};
 }catch(e){return {url:u.toString(),requestedAt:capturedAt,collectedAt:new Date().toISOString(),error:String(e.message),data:null};}
}
export async function collect(output){
 const dir=resolve(output);await mkdir(dir,{recursive:true});const summary={mode:'RETROSPECTIVE_RESEARCH_ONLY',exchange:'Bybit',category:'linear',startedAt:new Date().toISOString(),datasets:[],unavailableModels:['historical OFI without archived sequenced books','historical CVD without complete trades','order cancellation identity','whale/account identity','all-market regime validation']};
 const from=Date.parse('2026-11-16T15:35:00Z'),to=Date.parse('2026-11-16T22:08:00Z');
 for(const symbol of ['BTCUSDT','ETHUSDT']){
 for(const interval of ['1','5','15','60','240','D']){
 const dt=intervalMs[interval],start=Math.floor(from/dt)*dt,end=Math.floor(to/dt)*dt;
 if(end<=start){summary.datasets.push({symbol,interval,status:'NO_FULL_INTERVAL_IN_WINDOW'});continue;}
 const r=await request('/v5/market/kline',{category:'linear',symbol,interval,start,end:end-1,limit:1000});
 const name=symbol+'-'+interval;await writeFile(join(dir,name+'-raw.json'),JSON.stringify(r,null,2)+'\n');
 if(r.error){summary.datasets.push({symbol,interval,status:'UNAVAILABLE',reason:r.error,httpStatus:r.httpStatus??null});continue;}
 try{
 const dataset=normalizeKlines(r.data,{symbol,interval,collectedAt:r.collectedAt,from:start,to:end});
 await writeFile(join(dir,name+'-normalized.json'),JSON.stringify(dataset,null,2)+'\n');
 summary.datasets.push({symbol,interval,status:dataset.missing.length?'GAPPED':'CAPTURED',bars:dataset.bars.length,missing:dataset.missing.length,from:start,to:end,liveVintageVerified:false,rawSha256:r.rawSha256});
 }catch(e){summary.datasets.push({symbol,interval,status:'INVALID',reason:String(e.message)});}
 }
 // These are present-time observations, explicitly never backfilled as incident-time data.
 for(const [kind,path,params]of [
 ['trades','/v5/market/recent-trade',{limit:1000}],
 ['book','/v5/market/orderbook',{limit:50}],
 ['oi','/v5/market/open-interest',{intervalTime:'5min',startTime:from,endTime:to,limit:200}],
 ['funding','/v5/market/funding/history',{startTime:from,endTime:to,limit:200}] ]){
 const r=await request(path,{category:'linear',symbol,...params});
 await writeFile(join(dir,symbol+'-'+kind+'-raw.json'),JSON.stringify({...r,scope:kind==='book'||kind==='trades'?'CURRENT_SAMPLE_NOT_INCIDENT_HISTORY':'RETROSPECTIVE_PARTIAL_WINDOW'},null,2)+'\n');
 summary.datasets.push({symbol,kind,status:r.error?'UNAVAILABLE':'RAW_CAPTURED_NOT_VALIDATED',reason:r.error,complete:false});
 }
 }
 summary.completedAt=new Date().toISOString();summary.performance='UNVERIFIED_NO_STRATEGY_BACKTEST';
 await writeFile(join(dir,'manifest.json'),JSON.stringify(summary,null,2)+'\n');return summary;
}
if(process.argv[1]&&resolve(process.argv[1])===resolve(new URL(import.meta.url).pathname)){
 const summary=await collect(process.argv[2]??'research-output');
 console.log(JSON.stringify({datasets:summary.datasets,performance:summary.performance},null,2));
}
