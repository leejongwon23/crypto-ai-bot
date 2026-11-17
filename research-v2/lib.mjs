// Research infrastructure only. No exchange orders, official ledger writes or trade approvals.
import {createHash} from 'node:crypto';
export function canonical(x){
 if(x===null||typeof x==='boolean'||typeof x==='string')return JSON.stringify(x);
 if(typeof x==='number'&&Number.isFinite(x))return JSON.stringify(x);
 if(Array.isArray(x))return '['+x.map(canonical).join(',')+']';
 if(x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype)
  return '{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}';
 throw Error('NON_CANONICAL_VALUE');
}
export const hash=x=>createHash('sha256').update(typeof x==='string'?x:canonical(x)).digest('hex');
const check=(p,m)=>{if(!p)throw Error(m);};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const nonnegative=x=>finite(x)&&x>=0;
const positive=x=>finite(x)&&x>0;
export function time(x){check(typeof x==='string'&&/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{3})?Z$/.test(x),'UTC_TIMESTAMP_REQUIRED');const t=Date.parse(x);check(Number.isFinite(t)&&new Date(t).toISOString().replace('.000Z','Z')===x.replace('.000Z','Z'),'INVALID_UTC_TIME');return t;}
export const intervalMs=Object.freeze({'1':60000,'5':300000,'15':900000,'60':3600000,'240':14400000,D:86400000});
export function identity(m){
 check(m?.exchange==='Bybit'&&m.category==='linear'&&['BTCUSDT','ETHUSDT'].includes(m.symbol),'PRODUCT_IDENTITY_MISMATCH');
 return m.exchange+':'+m.category+':'+m.symbol;
}
export function normalizeKlines(body,{symbol,interval,collectedAt,from,to}){
 identity({exchange:'Bybit',category:'linear',symbol});const captured=time(collectedAt),dt=intervalMs[interval];
 check(dt&&Number.isSafeInteger(from)&&Number.isSafeInteger(to)&&to>from,'INVALID_CANDLE_WINDOW');
 check(body?.retCode===0&&body.result?.category==='linear'&&body.result?.symbol===symbol&&Array.isArray(body.result.list),'BYBIT_RESPONSE_IDENTITY_INVALID');
 const seen=new Set(),bars=[];
 for(const row of body.result.list){
  check(Array.isArray(row)&&row.length>=7,'FULL_OHLCV_REQUIRED');
  const [start,open,high,low,close,volume,turnover]=row.slice(0,7).map(Number);
  check([start,open,high,low,close,volume,turnover].every(finite)&&Number.isSafeInteger(start)&&start%dt===0,'CANDLE_NUMBER_OR_ALIGNMENT_INVALID');
  check([open,high,low,close].every(positive)&&low<=Math.min(open,close)&&high>=Math.max(open,close)&&low<=high&&nonnegative(volume)&&nonnegative(turnover),'CANDLE_VALUES_INVALID');
  check(!seen.has(start),'DUPLICATE_CANDLE');seen.add(start);
  if(start<from||start>=to)continue;
  check(start+dt<=captured,'OPEN_OR_FUTURE_CANDLE');
  bars.push({start,end:start+dt,open,high,low,close,volume,turnover,availableAt:collectedAt});
 }
 bars.sort((a,b)=>a.start-b.start);
 const missing=[];for(let t=Math.ceil(from/dt)*dt;t<to;t+=dt)if(!seen.has(t))missing.push(t);
 return {identity:'Bybit:linear:'+symbol,interval,provenance:'PUBLIC_REST_RETROSPECTIVE',
 collectedAt,from,to,bars,missing,missingRatio:missing.length/(bars.length+missing.length||1),
 // A retrospective download cannot prove that the live system had these data at issuance.
 liveVintageVerified:false,rawSha256:hash(body)};
}
export function visibleAt(records,asOf,{mode='LIVE'}={}){
 const t=time(asOf);check(['LIVE','REPLAY'].includes(mode),'INVALID_VINTAGE_MODE');
 for(const r of records){check(finite(r.eventTime)&&finite(r.knownAt),'BITEMPORAL_TIME_REQUIRED');}
 return records.filter(r=>r.eventTime<=t&&r.knownAt<=t&&
  (mode==='REPLAY'||r.liveVintageVerified===true));
}
export function appendEvidence(log,payload){
 const previous=log.at(-1)?.digest??null;
 const event={sequence:log.length+1,previous,payload:structuredClone(payload)};
 event.digest=hash(event);return [...log,event];
}
export function verifyEvidence(log){
 let previous=null;for(let i=0;i<log.length;i++){
 const {digest,...event}=log[i];if(event.sequence!==i+1||event.previous!==previous||digest!==hash(event))return false;previous=digest;
 }return true;
}
// Local hash chains detect modification relative to a pinned root, not a privileged full-chain rewrite.
export function takerFlow(trades,{market,from,to,complete}){
 const id=identity(market);check(finite(from)&&finite(to)&&to>from,'INVALID_FLOW_WINDOW');
 if(complete!==true)return {status:'UNVERIFIED',reason:'COMPLETE_TRADE_COVERAGE_REQUIRED'};
 const seen=new Set();let buy=0,sell=0,last=-Infinity;
 for(const x of trades){
  check(identity(x)===id,'MIXED_TRADE_PRODUCTS');check(typeof x.id==='string'&&x.id.length>0&&!seen.has(x.id),'DUPLICATE_OR_MISSING_TRADE_ID');seen.add(x.id);
  check(finite(x.eventTime)&&x.eventTime>=last,'TRADE_ORDER_INVALID');last=x.eventTime;
  check(positive(x.price)&&positive(x.size)&&['Buy','Sell'].includes(x.side),'INVALID_TRADE');
  if(x.eventTime>=from&&x.eventTime<to){if(x.side==='Buy')buy+=x.size;else sell+=x.size;}
 }
 return {status:'OBSERVED_FLOW',buy,sell,delta:buy-sell,
 interpretation:'Aggressor-side volume only; sign is not a directional approval or participant intent.'};
}
export function bestQuoteOFI(a,b){
 check(identity(a)===identity(b),'MIXED_BOOK_PRODUCTS');
 if(a.continuityVerified!==true||b.continuityVerified!==true||a.epoch!==b.epoch)
  return {status:'UNVERIFIED',reason:'BOOK_CONTINUITY_REQUIRED'};
 for(const x of [a,b])check([x.bid,x.ask,x.bidSize,x.askSize].every(positive)&&x.bid<x.ask&&finite(x.eventTime)&&Number.isSafeInteger(x.sequence),'INVALID_BOOK');
 check(b.eventTime>=a.eventTime&&b.sequence>a.sequence,'BOOK_ORDER_INVALID');
 const ofi=(b.bid>=a.bid?b.bidSize:0)-(b.bid<=a.bid?a.bidSize:0)
 -(b.ask<=a.ask?b.askSize:0)+(b.ask>=a.ask?a.askSize:0);
 return {status:'OBSERVED_OFI',ofi,interpretation:'Quote-size changes do not identify cancellations versus executions or whale identity.'};
}
export function absorptionHypotheses({flow,startPrice,endPrice}){
 if(flow.status!=='OBSERVED_FLOW'||![startPrice,endPrice].every(positive))return {status:'UNVERIFIED',hypotheses:[]};
 const change=endPrice-startPrice,hypotheses=[];
 if(flow.delta<0&&change>=0)hypotheses.push('SELL_AGGRESSION_WITH_NONNEGATIVE_PRICE_RESPONSE');
 if(flow.delta>0&&change<=0)hypotheses.push('BUY_AGGRESSION_WITH_NONPOSITIVE_PRICE_RESPONSE');
 return {status:'HYPOTHESIS_ONLY',hypotheses,alternatives:['window mismatch','passive liquidity replenishment','unobserved venue flows'],tradeApproval:false};
}
export function ksStatistic(a,b){
 check(a.length>0&&b.length>0&&a.every(finite)&&b.every(finite),'KS_FINITE_SAMPLES_REQUIRED');
 a=[...a].sort((x,y)=>x-y);b=[...b].sort((x,y)=>x-y);
 let i=0,j=0,d=0;for(const x of [...new Set([...a,...b])].sort((x,y)=>x-y)){
 while(i<a.length&&a[i]<=x)i++;while(j<b.length&&b[j]<=x)j++;
 d=Math.max(d,Math.abs(i/a.length-j/b.length));}
 return {statistic:d,n1:a.length,n2:b.length,pValue:null,status:'DESCRIPTIVE_ONLY',reason:'No IID p-value assumed for dependent market data.'};
}
export function purgedWalkForward(events,folds,{embargoMs,holdoutStart}){
 check(finite(embargoMs)&&embargoMs>=0&&finite(holdoutStart),'FROZEN_SPLIT_BOUNDARIES_REQUIRED');
 for(const e of events)check(finite(e.featureStart)&&finite(e.decisionAt)&&finite(e.labelEnd)&&e.featureStart<=e.decisionAt&&e.decisionAt<=e.labelEnd,'INVALID_EVENT_INFORMATION_SPAN');
 const sorted=[...folds].sort((a,b)=>a.testStart-b.testStart);
 for(let i=0;i<sorted.length;i++){
 const f=sorted[i];check(finite(f.trainStart)&&finite(f.testStart)&&finite(f.testEnd)&&f.trainStart<f.testStart&&f.testStart<f.testEnd&&f.testEnd<=holdoutStart,'FOLD_TOUCHES_HOLDOUT_OR_INVALID');
 if(i)check(sorted[i-1].testEnd<=f.testStart,'OVERLAPPING_EVALUATION_FOLDS');
 }
 return sorted.map(f=>({
 train:events.filter(e=>e.decisionAt>=f.trainStart&&e.labelEnd<f.testStart-embargoMs),
 test:events.filter(e=>e.featureStart>=f.testStart&&e.decisionAt<f.testEnd&&e.labelEnd<f.testEnd),
 bounds:f
 }));
}
export function holm(pValues,alpha){
 check(positive(alpha)&&alpha<1&&pValues.length>0&&pValues.every(p=>nonnegative(p)&&p<=1),'INVALID_HOLM_INPUT');
 const ordered=pValues.map((p,index)=>({p,index})).sort((a,b)=>a.p-b.p),out=Array(pValues.length).fill(false);
 for(let i=0;i<ordered.length;i++){if(ordered[i].p>alpha/(ordered.length-i))break;out[ordered[i].index]=true;}
 return out;
}
export function movingBlockMeanCI(values,{blockLength,reps,alpha,random}){
 check(values.length>=2&&values.every(finite),'BOOTSTRAP_SAMPLE_REQUIRED');
 check(Number.isInteger(blockLength)&&blockLength>0&&blockLength<=values.length&&Number.isInteger(reps)&&reps>=100&&positive(alpha)&&alpha<1&&typeof random==='function','BOOTSTRAP_PROTOCOL_REQUIRED');
 const means=[];for(let r=0;r<reps;r++){const sample=[];while(sample.length<values.length){
 const u=random();check(nonnegative(u)&&u<1,'INVALID_RANDOM');const start=Math.floor(u*(values.length-blockLength+1));
 for(let j=0;j<blockLength&&sample.length<values.length;j++)sample.push(values[start+j]);
 }means.push(sample.reduce((a,b)=>a+b,0)/sample.length);}
 means.sort((a,b)=>a-b);
 return {n:values.length,mean:values.reduce((a,b)=>a+b,0)/values.length,
 lower:means[Math.floor(alpha/2*(reps-1))],upper:means[Math.ceil((1-alpha/2)*(reps-1))],
 method:'moving non-circular block bootstrap',blockLength,reps,alpha};
}
export function netTrade(t,cost){
 check(['LONG','SHORT'].includes(t.direction)&&[t.entry,t.exit,t.quantity].every(positive),'VERIFIED_PRICES_AND_SIZE_REQUIRED');
 check(t.outcomeVerified===true,'UNVERIFIED_OUTCOME_NOT_PNL');
 for(const k of ['entryFeeBps','exitFeeBps','entrySlippageBps','exitSlippageBps'])check(nonnegative(cost?.[k]),'EXPLICIT_COST_REQUIRED_'+k);
 check(finite(cost.fundingCash),'SIGNED_FUNDING_CASH_REQUIRED');
 const sign=t.direction==='LONG'?1:-1;
 const entry=t.entry*(1+sign*cost.entrySlippageBps/10000);
 const exit=t.exit*(1-sign*cost.exitSlippageBps/10000);
 const gross=sign*(exit-entry)*t.quantity;
 const fees=(entry*cost.entryFeeBps+exit*cost.exitFeeBps)/10000*t.quantity;
 return {gross,fees,fundingCash:cost.fundingCash,net:gross-fees+cost.fundingCash,
 assumption:'Linear USDT perpetual, externally specified filled quantity; no account fill assertion.'};
}
export function promotionGate(protocol,evidence){
 const reasons=[];
 const require=(ok,reason)=>{if(!ok)reasons.push(reason);};
 require(protocol?.status==='LOCKED','PROTOCOL_NOT_LOCKED');
 require(typeof protocol?.protocolHash==='string'&&protocol.protocolHash===hash(protocol.spec),'PROTOCOL_HASH_MISMATCH');
 const s=protocol?.spec??{};
 require(s.userApproved===true,'USER_CRITERIA_NOT_APPROVED');
 for(const k of ['minTradesPerRegime','maxDrawdown','maxMissingRatio','alpha','embargoMs','blockLength'])
  require(finite(s[k])&&s[k]>=0,'MISSING_PREREGISTERED_'+k);
 require(positive(s.minTradesPerRegime)&&positive(s.alpha)&&s.alpha<1&&Number.isInteger(s.blockLength)&&s.blockLength>0,'INVALID_STATISTICAL_PROTOCOL');
 for(const k of ['unit','integration','regression','leakage','ledgerIntegrity','costs','holdoutUnspent','versionsMatch','shadow','canaryRollbackRehearsal'])
  require(evidence?.[k]===true,'UNVERIFIED_'+k);
 require(evidence?.protocolHash===protocol?.protocolHash,'RESULT_PROTOCOL_MISMATCH');
 require(positive(evidence?.improvementCiLower),'NO_POSITIVE_NET_EXPECTANCY_IMPROVEMENT_BOUND');
 require(finite(evidence?.drawdown)&&finite(s.maxDrawdown)&&evidence.drawdown<=s.maxDrawdown,'DRAWDOWN_LIMIT');
 require(finite(evidence?.baselineDrawdown)&&finite(evidence?.drawdown)&&evidence.drawdown<=evidence.baselineDrawdown,'DRAWDOWN_WORSE_THAN_BASELINE');
 require(finite(evidence?.missingRatio)&&finite(s.maxMissingRatio)&&evidence.missingRatio<=s.maxMissingRatio,'MISSING_DATA_LIMIT');
 const regimes=evidence?.regimes??[];
 require(regimes.length>0&&regimes.every(r=>Number.isInteger(r.n)&&r.n>=s.minTradesPerRegime&&r.independent===true),'INSUFFICIENT_REGIME_SAMPLES');
 require(evidence?.automaticOrdersEnabled===false,'AUTOMATIC_ORDERS_FORBIDDEN');
 return {eligibleForHumanReview:reasons.length===0,reasons,
 deployAuthorized:false,warning:'Evidence contract only; not an attestation verifier or implemented deployment controller.'};
}
