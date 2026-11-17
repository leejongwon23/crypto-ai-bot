import crypto from 'node:crypto';
export const canonical=x=>Array.isArray(x)?'['+x.map(canonical).join(',')+']':x&&typeof x==='object'?'{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}':JSON.stringify(x);
export const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:canonical(x)).digest('hex');
const finite=Number.isFinite;
const ms=s=>typeof s==='number'?s:Date.parse(String(s).includes('T')?s:String(s).replace(' ','T')+'+09:00');
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
export function empiricalQuantile(a,q){if(!a.length||!a.every(finite)||!(q>=0&&q<=1))return null;const b=[...a].sort((x,y)=>x-y);return b[Math.min(b.length-1,Math.ceil(q*b.length)-1<0?0:Math.ceil(q*b.length)-1)];}
export function empiricalKS(a,b){if(!a.length||!b.length)return null;const x=[...a].sort((x,y)=>x-y),y=[...b].sort((x,y)=>x-y);let i=0,j=0,d=0;for(const v of [...new Set([...x,...y])].sort((x,y)=>x-y)){while(i<x.length&&x[i]<=v)i++;while(j<y.length&&y[j]<=v)j++;d=Math.max(d,Math.abs(i/x.length-j/y.length));}return d;}
export function validateBars(rows,asOf){
 if(!Array.isArray(rows)||rows.length>2000)throw Error('QUANT_BAR_ARRAY_INVALID');
 const out=[];let previous=-Infinity;
 for(const r of rows){const t=ms(r.t??r.start),dt=r.intervalMs??60000;const o=Number(r.open),h=Number(r.high),l=Number(r.low),c=Number(r.close);
  if(![t,dt,o,h,l,c].every(finite)||!(dt>0)||t<=previous||t%dt!==0||[o,h,l,c].some(x=>x<=0)||l>Math.min(o,c)||h<Math.max(o,c)||l>h)throw Error('QUANT_OHLC_ORDER_OR_VALUES_INVALID');
  previous=t;if(t+dt>asOf)continue;
  if(r.knownAt!==undefined&&(!finite(ms(r.knownAt))||ms(r.knownAt)>asOf))throw Error('QUANT_FUTURE_KNOWN_AT');
  if(r.volume!==undefined&&(!finite(Number(r.volume))||Number(r.volume)<0))throw Error('QUANT_VOLUME_INVALID');
  out.push({t,end:t+dt,intervalMs:dt,open:o,high:h,low:l,close:c,...(r.volume!==undefined?{volume:Number(r.volume)}:{})});
 }
 const gaps=out.slice(1).flatMap((r,i)=>r.t!==out[i].end?[{from:out[i].end,to:r.t}]:[]);
 return {bars:out,gaps};
}
export function confirmedPivots(bars){
 const out=[];for(let i=1;i<bars.length-1;i++){const p=bars[i-1],b=bars[i],n=bars[i+1];
  if(b.high>p.high&&b.high>n.high)out.push({kind:'HIGH',price:b.high,at:b.t,knownAt:n.end});
  if(b.low<p.low&&b.low<n.low)out.push({kind:'LOW',price:b.low,at:b.t,knownAt:n.end});
 }return out;
}
export function measureStructure(bars){
 if(bars.length<3)return {status:'UNVERIFIED',reason:'LESS_THAN_THREE_CLOSED_BARS',regime:'UNVERIFIED',pivots:[]};
 const pivots=confirmedPivots(bars),hi=pivots.filter(x=>x.kind==='HIGH').slice(-2),lo=pivots.filter(x=>x.kind==='LOW').slice(-2);
 const returns=bars.slice(1).map((b,i)=>Math.log(b.close/bars[i].close));
 const deltas=bars.slice(1).map((b,i)=>b.close-bars[i].close),tr=bars.map((b,i)=>Math.max(b.high-b.low,i?Math.abs(b.high-bars[i-1].close):0,i?Math.abs(b.low-bars[i-1].close):0));
 const m=mean(returns),v=mean(returns.map(x=>(x-m)**2));
 const up=hi.length===2&&lo.length===2&&hi[1].price>hi[0].price&&lo[1].price>lo[0].price;
 const down=hi.length===2&&lo.length===2&&hi[1].price<hi[0].price&&lo[1].price<lo[0].price;
 const close=bars.at(-1).close,lastHigh=hi.at(-1),lastLow=lo.at(-1);
 const breakingUp=lastHigh&&close>lastHigh.price,breakingDown=lastLow&&close<lastLow.price;
 const invalidatedUp=up&&lastLow&&close<lastLow.price,invalidatedDown=down&&lastHigh&&close>lastHigh.price;
 const regime=invalidatedUp||invalidatedDown?'TRANSITION':breakingUp||breakingDown?'BREAKOUT':up?'TREND_UP':down?'TREND_DOWN':'MIXED';
 const mid=Math.floor(returns.length/2),absolute=returns.map(Math.abs),path=bars.at(-1).close-bars[0].close;
 return {status:'OBSERVED_STRUCTURE_NOT_VALIDATED_FORECAST',regime,pivots,lastHigh:lastHigh??null,lastLow:lastLow??null,
  structuralEvents:{higherHighs:hi.length===2&&hi[1].price>hi[0].price,higherLows:lo.length===2&&lo[1].price>lo[0].price,lowerHighs:hi.length===2&&hi[1].price<hi[0].price,lowerLows:lo.length===2&&lo[1].price<lo[0].price,breakingUp:!!breakingUp,breakingDown:!!breakingDown,invalidatedUp:!!invalidatedUp,invalidatedDown:!!invalidatedDown},
  measurements:{closedBars:bars.length,atr:mean(tr),returnMean:m,returnVariance:v,realizedVariation:returns.reduce((a,x)=>a+x*x,0),
   lagOneAutocorrelation:v>0?mean(returns.slice(1).map((r,i)=>(r-m)*(returns[i]-m)))/v:null,
   pathEfficiency:deltas.reduce((a,x)=>a+Math.abs(x),0)>0?Math.abs(path)/deltas.reduce((a,x)=>a+Math.abs(x),0):null,
   absoluteReturnQuantiles:{median:empiricalQuantile(absolute,.5),q90:empiricalQuantile(absolute,.9),q95:empiricalQuantile(absolute,.95)},
   chronologicalHalfKS:empiricalKS(returns.slice(0,mid),returns.slice(mid)),ksInference:'DESCRIPTIVE_ONLY_NO_IID_P_VALUE'},
  regimeHypotheses:[
   {kind:'DIRECTIONAL_TREND',status:up||down?'STRUCTURALLY_SUPPORTED':'UNIDENTIFIED',counterEvidence:invalidatedUp||invalidatedDown?'LATEST_CLOSE_BROKE_CONFIRMED_PIVOT':'HIGHER_TIMEFRAME_AND_SAMPLE_VALIDATION_REQUIRED'},
   {kind:'RANGE_OR_COMPRESSION',status:'UNIDENTIFIED',counterEvidence:'Boundary duration, repeated reactions, variance comparison and higher-timeframe data must be independently examined.'},
   {kind:'STRONG_EXPANSION_OR_STAIRCASE',status:'UNIDENTIFIED',counterEvidence:'A trend label alone cannot establish strength or staircase persistence.'},
   {kind:'MULTIPLE_FALSE_BREAKOUTS',status:'UNIDENTIFIED',counterEvidence:'Requires frozen prior boundaries and observed excursions/returns; never inferred from MACD alone.'},
   {kind:'ACCUMULATION_DISTRIBUTION',status:'HYPOTHESIS_ONLY',counterEvidence:'Participant identities and intentions are unobserved.'}
  ]};
}
export function analyzeTrades(rows,asOf,symbol){
 if(!Array.isArray(rows)||!rows.length)return {status:'UNAVAILABLE',reason:'RAW_TRADES_MISSING',hypotheses:[]};
 let buy=0,sell=0,last=-Infinity;const seen=new Set();const trades=[];
 for(const r of rows){const t=ms(r.t),price=Number(r.price),size=Number(r.size);
  if(!finite(t)||t<last||t>asOf||!([price,size].every(finite)&&price>0&&size>0)||!['Buy','Sell'].includes(r.side)||r.symbol!==symbol+'USDT'||r.exchange!=='Bybit'||r.category!=='linear'||typeof r.id!=='string'||!r.id.trim()||seen.has(r.id))throw Error('QUANT_TRADE_IDENTITY_TIME_OR_DUPLICATE');
  if(r.knownAt!==undefined&&(!finite(ms(r.knownAt))||ms(r.knownAt)>asOf))throw Error('QUANT_FUTURE_TRADE');
  seen.add(r.id);last=t;if(r.side==='Buy')buy+=size;else sell+=size;trades.push({t,price,size,side:r.side});
 }
 const delta=buy-sell,priceChange=trades.at(-1).price-trades[0].price;
 const conflict=delta<0&&priceChange>=0?'SELL_AGGRESSION_WITH_NONNEGATIVE_PRICE_RESPONSE':delta>0&&priceChange<=0?'BUY_AGGRESSION_WITH_NONPOSITIVE_PRICE_RESPONSE':null;
 return {status:'OBSERVED_SAMPLE_NOT_COMPLETE_TRADE_COVERAGE',trades:trades.length,firstAt:trades[0].t,lastAt:last,buy,sell,delta,priceChange,
  signedImpact:delta!==0?priceChange/delta:null,participantIntent:'UNIDENTIFIABLE',hypotheses:conflict?[{kind:conflict,status:'HYPOTHESIS_ONLY',alternatives:['passive replenishment','window mismatch','other-venue flow'],directionApproved:false}]:[],
  warning:'A raw sample is not complete CVD or proof of order cancellations; no CVD sign-only entry.'};
}
function plan(dir,entry,stops,targets,label){
 const side=dir==='LONG'?1:-1,s=stops.filter(x=>side*(entry-x)>0).sort((a,b)=>Math.abs(a-entry)-Math.abs(b-entry))[0];
 const t=[...new Set(targets.filter(x=>side*(x-entry)>0))].sort((a,b)=>side*(a-b)).slice(0,3);
 if(!finite(s)||t.length<3)return {kind:'NO_PLAN',label,missing:!finite(s)?'STRUCTURAL_STOP': 'THREE_OBSERVED_ORDERED_TARGETS',entry};
 const risk=Math.abs(entry-s);return {kind:'ARITHMETIC_CANDIDATE_NOT_FORMAL',label,direction:dir,entryLow:entry,entryHigh:entry,stop:s,tp1:t[0],tp2:t[1],tp3:t[2],rr2:Math.abs(t[1]-entry)/risk,stopPct:-risk/entry*100,
  noiseRisk:'Empirical noise and structural invalidation must be compared; pivot price is an anchor, not a validated stop buffer.',executionRisk:'Spread, adverse fill, funding and fees unverified unless source supplies them.',approved:false};
}
export function directionCandidates({bars,price,structure,ledger=[],symbol}){
 const highs=structure.pivots?.filter(x=>x.kind==='HIGH').map(x=>x.price)??[],lows=structure.pivots?.filter(x=>x.kind==='LOW').map(x=>x.price)??[];
 return Object.fromEntries(['LONG','SHORT'].map(dir=>{const stops=dir==='LONG'?lows:highs,targets=dir==='LONG'?highs:lows;
  const near=stops.filter(x=>dir==='LONG'?x<price:x>price).sort((a,b)=>Math.abs(a-price)-Math.abs(b-price))[0];
  const far=stops.filter(x=>dir==='LONG'?x<near:x>near).sort((a,b)=>Math.abs(a-price)-Math.abs(b-price))[0];
  return [dir,{A:plan(dir,price,stops,targets,'CURRENT_PRICE'),B:finite(near)?plan(dir,near,stops,targets,'NEAR_OBSERVED_PIVOT'):{kind:'NO_PLAN',missing:'NEAR_ANCHOR'},C:finite(far)?plan(dir,far,stops,targets,'FAR_OBSERVED_PIVOT'):{kind:'NO_PLAN',missing:'FAR_ANCHOR'},D:ledger.filter(s=>s.symbol===symbol+'USDT'&&s.original?.direction===dir&&s.status==='PENDING').map(s=>({id:s.id,original:s.original})),E:{kind:'NO_NEW_SIGNAL',reason:'COMPARE_A_TO_D_WITH_COUNTEREVIDENCE_BEFORE_DECISION'}}];
 }));
}
const FRAMES={'1m':60000,'5m':300000,'15m':900000,'1h':3600000,'4h':14400000,'1D':86400000};
export function aggregateObservedMinutes(bars,intervalMs){
 const groups=new Map();for(const b of bars){if(b.intervalMs!==60000)continue;const t=Math.floor(b.t/intervalMs)*intervalMs;if(!groups.has(t))groups.set(t,[]);groups.get(t).push(b);}
 const result=[];for(const [t,rows]of groups){if(rows.length!==intervalMs/60000||rows.some((b,i)=>b.t!==t+i*60000))continue;const volume=rows.every(b=>'volume' in b)?{volume:rows.reduce((n,b)=>n+b.volume,0)}:{};
 result.push({t,intervalMs,open:rows[0].open,close:rows.at(-1).close,high:Math.max(...rows.map(b=>b.high)),low:Math.min(...rows.map(b=>b.low)),...volume});}return result;
}
function timeframeContexts(json,primary,asOf){
 return Object.fromEntries(Object.entries(FRAMES).map(([tf,dt])=>{const supplied=json.timeframes?.[tf];const rows=supplied?supplied.map(b=>({...b,intervalMs:dt})):primary.every(b=>b.intervalMs===60000)?aggregateObservedMinutes(primary,dt):[];const checked=validateBars(rows,asOf);return [tf,{provenance:supplied?'USER_SUPPLIED_RAW':'DERIVED_ONLY_FROM_COMPLETE_OBSERVED_MINUTES',closedBars:checked.bars.length,gaps:checked.gaps,structure:checked.gaps.length?{status:'UNVERIFIED',reason:'TIMEFRAME_GAPS'}:measureStructure(checked.bars)}];}));
}
function rawSections(raw){const match=raw.match(/\[QUANT_RAW_JSON\]\s*([\s\S]*?)\s*\[\/QUANT_RAW_JSON\]/);if(!match)return {};try{return JSON.parse(match[1]);}catch{throw Error('QUANT_RAW_JSON_INVALID');}}
export function buildQuantContext(raw,ledger=[]){
 const json=rawSections(raw),symbols={};
 for(const sym of ['BTC','ETH']){
  const start=raw.indexOf('========== '+sym+' ('+sym+'USDT) =========='),next=raw.indexOf('========== ',start+12),part=raw.slice(start,next<0?undefined:next);
  const at=part.match(/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m)?.[1],asOf=ms(at),price=Number(part.match(/^lastPrice:\s*([\d.]+)/m)?.[1]);
  const j=json[sym]??{};if(j.exchange!==undefined&&(j.exchange!=='Bybit'||j.category!=='linear'||j.symbol!==sym+'USDT'))throw Error('QUANT_MARKET_IDENTITY_MISMATCH');
  let rows=j.bars??j.timeframes?.['1m']??[];if(!rows.length&&part.includes('[10B]')){const text=part.slice(part.indexOf('minute_OHLC_KST:'));rows=text.split(/\r?\n/).flatMap(l=>{const m=l.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),([\d.]+),([\d.]+),([\d.]+),([\d.]+)$/);return m?[{t:m[1],open:+m[2],high:+m[3],low:+m[4],close:+m[5]}]:[];});}
  const checked=validateBars(rows,asOf),structure=checked.gaps.length?{status:'UNVERIFIED',reason:'NONCONTIGUOUS_BARS',regime:'UNVERIFIED',pivots:[]}:measureStructure(checked.bars);
  const flow=analyzeTrades(j.trades,asOf,sym);
  symbols[sym]={asOf:at,identity:'Bybit linear '+sym+'USDT',provenance:'USER_SUPPLIED_NOT_INDEPENDENTLY_AUTHENTICATED',dataStatus:{closedBars:checked.bars.length,gaps:checked.gaps,fullOHLCV:checked.bars.length>0&&checked.bars.every(b=>'volume' in b),rawTrades:flow.trades??0,bookUpdates:'UNAVAILABLE',orderCancellationIdentity:'UNIDENTIFIABLE',liquidations:'UNAVAILABLE',knownAtVintage:j.liveVintageVerified===true?'USER_ASSERTED_NOT_ATTESTED':'UNVERIFIED'},structure,flow,timeframes:timeframeContexts(j,checked.bars,asOf),
   directions:directionCandidates({bars:checked.bars,price,structure,ledger,symbol:sym}),
   modelErrorReview:ledger.filter(x=>x.symbol===sym+'USDT'&&['NEW_ENTRY','ENTERED','TP1','TP2','PENDING'].includes(x.status)).map(x=>({id:x.id,modelId:x.original.modelId??null,forecastAt:x.original.issuedAt,frozenSourceSha256:x.original.frozenSourceSha256??null,reviewRequiredBeforeStop:true,observedStructuralEvents:structure.structuralEvents??{},predictionInterval:x.original.predictionInterval??null,intervalBreached:x.original.predictionInterval?price<x.original.predictionInterval.low||price>x.original.predictionInterval.high:null,action:'INDEPENDENT_MATHEMATICAL_COUNTEREVIDENCE_REVIEW_REQUIRED'}))};
 }
 const core={version:'V2.0',sourceSha256:sha(raw),symbols,mechanisms:['ORDERED_PRICE_STRUCTURE','CONDITIONAL_FLOW_PRICE_RESPONSE','EMPIRICAL_VARIATION','DISTRIBUTION_CHANGE','PRE_STOP_MODEL_FALSIFICATION'],combination:'NO_WEIGHTS_NO_SCORES',autonomy:{proposalGeneration:true,independentPromotionRequired:true,ordersEnabled:false}};
 return {...core,sha256:sha(core)};
}
export function validateQuantAudit(analysis,context){
 const q=analysis?.quantAudit;if(!q||q.contextSha256!==context.sha256)throw Error('V20_QUANT_CONTEXT_SHA_REQUIRED');
 for(const sym of ['BTC','ETH']){const x=q[sym];if(x?.asOf!==context.symbols[sym].asOf)throw Error('V20_QUANT_ASOF_'+sym);
  for(const k of ['regimeMechanism','flowImpactMechanism','noiseRisk','invalidationRisk','executionCostRisk','preStopErrorReview','competingHypotheses','modelRevisionDecision','sourceLimitations'])if(typeof x[k]!=='string'||!x[k].trim())throw Error('V20_QUANT_EVIDENCE_'+sym+'_'+k);
  if(!['OBSERVED','HYPOTHESIS','UNVERIFIED'].includes(x.intentEvidenceLevel))throw Error('V20_INTENT_EVIDENCE_LEVEL_'+sym);
  if(x.participantIdentityKnown!==false)throw Error('V20_UNOBSERVED_PARTICIPANT_IDENTITY');
  for(const dir of ['LONG','SHORT']){const d=x.directions?.[dir];if(!d)throw Error('V20_QUANT_BOTH_DIRECTIONS');for(const k of ['mathematicalBasis','counterexample','entryStopTargetBasis','costAndCapacityLimitations'])if(typeof d[k]!=='string'||!d[k].trim())throw Error('V20_DIRECTION_EVIDENCE_'+sym+'_'+dir+'_'+k);}
 }
 if(!Array.isArray(q.researchProposals))throw Error('V20_RESEARCH_PROPOSALS_ARRAY');
 for(const p of q.researchProposals)for(const k of ['mechanismChange','support','refutationTest','alternativeExplanation','requiredData','status'])if(typeof p[k]!=='string'||!p[k].trim())throw Error('V20_RESEARCH_PROPOSAL_'+k);
 return true;
}
