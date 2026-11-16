// Narrow prospective gate for range-boundary entries fighting an established
// execution-timeframe trend. Evidence only: no prediction or retrospective mutation.
const readNumber=(str,regex)=>{const m=str.match(regex);return m?Number(m[1]):null;};
const okNumber=n=>typeof n==='number'&&Number.isFinite(n)&&n>0;
export function evaluateRangeStopContext({raw,signal}={}){
 if(signal?.entryTactic!=='RANGE_BOUNDARY')return {allowed:true,reason:'NOT_RANGE_ENTRY'};
 if(!['BTCUSDT','ETHUSDT'].includes(signal.symbol))return {allowed:false,reason:'INVALID_SOURCE_SYMBOL'};
 const start=raw.indexOf('========== '+(signal.symbol==='BTCUSDT'?'BTC':'ETH')+' ('+signal.symbol+') ==========');
 if(start<0)return {allowed:false,reason:'RANGE_SOURCE_SYMBOL_MISSING'};
 const next=raw.indexOf('========== ',start+12),s=raw.slice(start,next<0?undefined:next);
 const price=readNumber(s,/^lastPrice:\s*([\d.]+)/m);
 const v24=readNumber(s,/^vwap_24h:\s*([\d.]+)/m);
 const v6=readNumber(s,/\bvwap_session_6h:\s*([\d.]+)/m);
 const high=readNumber(s,/^session_6h_high\/low:\s*([\d.]+)/m);
 const low=readNumber(s,/^session_6h_high\/low:\s*[\d.]+\s*\/\s*([\d.]+)/m);
 const align=['5m','15m','30m','1h'].map(tf=>s.match(new RegExp('^'+tf+':\\s*n=\\d+[^\\n]*?\\balign=(BULL|BEAR)','m'))?.[1]||null);
 if(![price,v24,v6,high,low,signal.stop].every(okNumber)||align.some(x=>!x))
  return {allowed:false,reason:'RANGE_COUNTERTREND_CONTEXT_MISSING',observed:{price,v24,v6,high,low,align}};
 const bullishAgainstShort=signal.direction==='SHORT'&&align.every(x=>x==='BULL')&&price>v24&&price>v6;
 const bearishAgainstLong=signal.direction==='LONG'&&align.every(x=>x==='BEAR')&&price<v24&&price<v6;
 if(bullishAgainstShort&&signal.stop<=high)
  return {allowed:false,reason:'RANGE_SHORT_STOP_INSIDE_UNCLEARED_6H_HIGH',observed:{price,v24,v6,high,stop:signal.stop,align}};
 if(bearishAgainstLong&&signal.stop>=low)
  return {allowed:false,reason:'RANGE_LONG_STOP_INSIDE_UNCLEARED_6H_LOW',observed:{price,v24,v6,low,stop:signal.stop,align}};
 return {allowed:true,reason:'NO_PROVEN_CONTEXT_CONFLICT',observed:{price,v24,v6,high,low,align}};
}
