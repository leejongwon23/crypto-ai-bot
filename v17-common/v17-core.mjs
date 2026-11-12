import crypto from 'node:crypto';
import {POLICY_TEXT,POLICY_SHA256,POLICY_BYTES,POLICY_LINES} from './policy-v17.mjs';

export const OFFICIAL_POLICY={name:'MASTER PROMPT V1.7 FINAL',sha256:POLICY_SHA256,bytes:POLICY_BYTES,lines:POLICY_LINES};
export const VERSION='V17_VERBATIM_AI_OFFICIAL_REGISTRY_1.0.0';
const STATES=new Set(['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','TP3_DONE','STOPPED','CANCELLED','REDESIGNED_CANCELLED','EARLY_EXIT','REVERSE_EXIT','UNVERIFIED']);
const TERMINAL=new Set(['TP3_DONE','STOPPED','CANCELLED','REDESIGNED_CANCELLED','EARLY_EXIT','REVERSE_EXIT']);
const DIRECTIONS=['BTC_LONG','BTC_SHORT','ETH_LONG','ETH_SHORT'];
const ALTERNATIVES=['A','B','C','D','E'];
export const STEPS=Array.from({length:16},(_,i)=>i+1);
export const CLAUSES=[...Array.from({length:24},(_,i)=>String(i)),'APPENDIX_A','APPENDIX_B','FINAL_EXECUTION'];
const has=(x)=>typeof x==='string'&&x.trim().length>0;
const validTime=x=>typeof x==='string'&&/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(x)&&Number.isFinite(Date.parse(x.replace(' ','T')+'+09:00'));
const ms=x=>Date.parse(x.replace(' ','T')+'+09:00');
const finite=x=>typeof x==='number'&&Number.isFinite(x);
export const digest=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)?x:typeof x==='string'?x:canon(x)).digest('hex');
export function canon(x){if(Array.isArray(x))return '['+x.map(canon).join(',')+']';if(x!==null&&typeof x==='object')return '{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canon(x[k])).join(',')+'}';return JSON.stringify(x);}
export function sessionId(s='OFFICIAL_V17'){if(!/^[a-zA-Z0-9_-]{1,80}$/.test(s))throw Error('INVALID_OFFICIAL_SESSION');return s;}
export function validateSnapshot(raw){
 if(typeof raw!=='string'||raw.length<80||Buffer.byteLength(raw,'utf8')>480000)throw Error('SNAPSHOT_TEXT_REQUIRED');
 const btc=[...raw.matchAll(/^==========\s*BTC\s*\(BTCUSDT\)\s*==========\s*$/gm)];
 const eth=[...raw.matchAll(/^==========\s*ETH\s*\(ETHUSDT\)\s*==========\s*$/gm)];
 if(btc.length!==1||eth.length!==1)throw Error('REQUIRES_EXACTLY_ONE_BTC_AND_ETH_SNAPSHOT');
 const time=[...raw.matchAll(/^timestamp_kst:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/gm)].map(m=>m[1]);
 if(time.length!==2||!time.every(validTime))throw Error('MISSING_OR_INVALID_TIMESTAMP_KST');
 const getPrice=(idx,end)=>{const frag=raw.slice(idx,end);const m=frag.match(/^lastPrice:\s*([\d.]+)/m);return m?Number(m[1]):null;};
 const btcPrice=getPrice(btc[0].index,eth[0].index>btc[0].index?eth[0].index:undefined);
 const ethPrice=getPrice(eth[0].index,btc[0].index>eth[0].index?btc[0].index:undefined);
 if(!(Number.isFinite(btcPrice)&&btcPrice>0&&Number.isFinite(ethPrice)&&ethPrice>0))throw Error('BTC_ETH_LAST_PRICE_REQUIRED');
 return {sha256:digest(raw),btcTime:time[btc[0].index<eth[0].index?0:1],ethTime:time[btc[0].index<eth[0].index?1:0],btcPrice,ethPrice,bytes:Buffer.byteLength(raw,'utf8')};
}
/**
 * Deterministic evidence-health audit: no trade recommendations, no new indicators.
 * The previous official snapshot timestamp is essential: original signal verifiedUntil
 * does not advance when an analysis has no state-change event.
 */
export function assessSnapshotEvidence(raw,{previousSnapshot=null,ledger=[]}={}){
 const snapshot=validateSnapshot(raw);
 const entryKeys=['[1]','[2]','[2B]','[3]','[5]','[8]','[CVD]','[10]','[7]'];
 const btcI=raw.indexOf('========== BTC (BTCUSDT) ==========');
 const ethI=raw.indexOf('========== ETH (ETHUSDT) ==========');
 const parts={BTC:raw.slice(btcI,ethI),ETH:raw.slice(ethI)};
 const perSymbol={},warnings=[];
 for(const symbol of ['BTC','ETH']){
  const text=parts[symbol],snapshotAt=symbol==='BTC'?snapshot.btcTime:snapshot.ethTime;
  const missing=entryKeys.filter(x=>!text.includes(x));
  const hasVWAP=/\[5B\][^\n]*VWAP[^\n]*\bFAIL\b/i.test(text);
  const hasCvdReliability=/cvd_reliability:\s*(HIGH|MEDIUM|LOW)/i.test(text);
  const path=/^start_price:\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*\|\s*end_price:\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(text);
  const prev=previousSnapshot?.[symbol==='BTC'?'btcTime':'ethTime']??null;
  const pathStart=path?.[2]??null,pathEnd=path?.[4]??null;
  const gapSeconds=prev&&pathStart?Math.max(0,Math.round((ms(pathStart)-ms(prev))/1000)):null;
  const reversed=Boolean(prev&&ms(snapshotAt)<ms(prev));
  const gap=Boolean(gapSeconds!==null&&gapSeconds>60);
  // The interval extremes are not a tick-by-tick trace. They can prove
  // a post-issuance zone crossing in a continuous, nonconflicting path,
  // but cannot establish the chronology of simultaneous entry and stop/TP.
  const pathExtremes=/^path_high\/low:\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*\/\s*([\d.]+)\s*@\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)/m.exec(text);
  const high=pathExtremes?Number(pathExtremes[1]):null;
  const low=pathExtremes?Number(pathExtremes[3]):null;
  const knownContinuous=Boolean(pathStart&&pathEnd&&pathExtremes&&prev&&!reversed&&!gap&&ms(pathStart)<=ms(prev)+60000&&ms(pathEnd)>=ms(prev));
  const last24=/^24h high\/low:\s*([\d.]+)\s*\/\s*([\d.]+)/m.exec(text);
  const priorActive=ledger.filter(s=>s.symbol===symbol+'USDT'&&['PENDING','NEW_ENTRY','ENTERED','TP1','TP2','UNVERIFIED'].includes(s.status));
  const uncertainSignals=[],requiredEntrySignals=[],requiredExitSignals=[],ambiguousSignals=[];
  for(const sig of priorActive){
   const orig=sig.original||sig;
   // A complete 24-hour high/low can disprove a touch in a shorter gap;
   // it cannot prove that a touch occurred or establish SL/TP event order.
   const disprovePendingTouch=Boolean(gap&&gapSeconds<=24*3600&&last24&&sig.status==='PENDING'&&(
     orig.direction==='SHORT'?Number(last24[1])<orig.entryLow:Number(last24[2])>orig.entryHigh
   ));
   if(gap&&!disprovePendingTouch)uncertainSignals.push(sig.id);
   if(knownContinuous&&sig.status==='PENDING'&&orig.issuedAt&&ms(pathStart)>=ms(orig.issuedAt)&&
      Number.isFinite(high)&&Number.isFinite(low)&&high>=orig.entryLow&&low<=orig.entryHigh){
     const exited=orig.direction==='SHORT'?(high>=orig.stop||low<=orig.tp1):(low<=orig.stop||high>=orig.tp1);
     if(exited)ambiguousSignals.push(sig.id);
     else requiredEntrySignals.push(sig.id);
   }
   if(knownContinuous&&['NEW_ENTRY','ENTERED','TP1','TP2'].includes(sig.status)&&
      ms(pathStart)>=ms(sig.verifiedUntil||orig.issuedAt)&&Number.isFinite(high)&&Number.isFinite(low)){
     const hitStop=orig.direction==='SHORT'?high>=orig.stop:low<=orig.stop;
     const hitTP1=orig.direction==='SHORT'?low<=orig.tp1:high>=orig.tp1;
     const hitTP2=orig.direction==='SHORT'?low<=orig.tp2:high>=orig.tp2;
     const hitTP3=orig.direction==='SHORT'?low<=orig.tp3:high>=orig.tp3;
     if(hitStop&&hitTP1)ambiguousSignals.push(sig.id);
     else if(hitStop)requiredExitSignals.push({id:sig.id,kind:'STOPPED'});
     else if(hitTP3)requiredExitSignals.push({id:sig.id,kind:'TP3_DONE'});
     else if(hitTP2&&sig.status!=='TP2')requiredExitSignals.push({id:sig.id,kind:'TP2'});
     else if(hitTP1&&!['TP1','TP2'].includes(sig.status))requiredExitSignals.push({id:sig.id,kind:'TP1'});
   }
  }
  const flags=[];
  if(missing.length)flags.push('MISSING_MARKET_SECTIONS');
  if(hasVWAP)flags.push('VWAP_TAKER_RATE_LIMIT');
  if(!hasCvdReliability)flags.push('CVD_RELIABILITY_NOT_PROVIDED');
  if(!path)flags.push('PRICE_PATH_MISSING');
  if(gap)flags.push('PRICE_PATH_GAP');
  if(reversed)flags.push('OUT_OF_ORDER_SNAPSHOT');
  if(uncertainSignals.length)flags.push('EXISTING_SIGNAL_STATE_UNVERIFIED');
  if(requiredEntrySignals.length)flags.push('PENDING_ENTRY_TOUCH_MUST_BE_REGISTERED');
  if(requiredExitSignals.length)flags.push('EXISTING_TP_OR_STOP_MUST_BE_REGISTERED');
  if(ambiguousSignals.length)flags.push('ENTRY_AND_EXIT_SEQUENCE_AMBIGUOUS');
  perSymbol[symbol]={snapshotAt,priorOfficialAt:prev,pathStart,pathEnd,gapSeconds,
   pathHigh:high,pathLow:low,knownContinuous,missingSections:missing,flags,
   rateLimitVWAP:hasVWAP,couldNotVerifySignals:uncertainSignals,
   requiredEntrySignals,requiredExitSignals,ambiguousSignals};
  for(const flag of flags)warnings.push(symbol+':'+flag);
 }
 return {snapshotSha256:snapshot.sha256,perSymbol,warnings,hasMaterialWarnings:warnings.length>0};
}
export function ledgerHash(rows){return digest(rows.map(r=>({id:r.id,symbol:r.symbol,original:r.original,status:r.status,verifiedUntil:r.verifiedUntil||null,version:r.version??0})).sort((a,b)=>a.id.localeCompare(b.id)));}
export const sourcePolicy=()=>({ ...OFFICIAL_POLICY,text:POLICY_TEXT });
export function prepare({raw,session='OFFICIAL_V17',ledger=[],saved=null,previousSnapshot=null}){
 const snapshot=validateSnapshot(raw),id=sessionId(session),beforeLedgerSha256=ledgerHash(ledger);
 const evidenceQuality=assessSnapshotEvidence(raw,{previousSnapshot,ledger});
 const key=digest({session:id,snapshotSha256:snapshot.sha256,ledgerSha256:beforeLedgerSha256,policySha256:POLICY_SHA256});
 if(saved)return {status:'OFFICIAL_REPLAY',session:id,snapshot,policy:OFFICIAL_POLICY,officialDecision:saved,evidenceQuality,source:'PERSISTED_FIRST_OFFICIAL_RESULT',sameState:true};
 return {status:'NEEDS_AI_ANALYSIS',session:id,snapshot,policy:sourcePolicy(),ledger:ledger.map(r=>({...r})),evidenceQuality,beforeLedgerSha256,decisionKey:key,requirements:{
  role:'AI must read and apply THE ENTIRE original Korean V1.7 policy below, not approximate with fixed indicator gates.',
  noExternalScriptedSignals:true,noNewIndicatorThresholds:true,noHistoricalSignalImport:true,
  interpretationSafeguards:{pendingSignalMayBeIssuedBeforeFutureTouch:true,futureRetestIsNotRequiredBeforeFormalPendingIssue:true,requiresPresentStructuralReasonForPlannedRetest:true,holdMustCompareExecutableOptionsNotDefaultToWaiting:true,neverBackfillHypotheticalMissedProfit:true},
  requiredDirections:DIRECTIONS,requiredAlternatives:ALTERNATIVES,requiredSteps:STEPS,requiredClauses:CLAUSES,
  next:'After complete V1.7 analysis call publish_official_v17_analysis with exact prompt sha, snapshot and ledger hashes, decisionKey and structured decision. If the installed plugin has not refreshed its tools list, use existing register_formal_btc_eth_signal with signal={kind:V17_OFFICIAL_SUBMISSION,raw,policySha256,snapshotSha256,beforeLedgerSha256,decisionKey,analysis}. Never present an unpublished preliminary candidate as official.',
  noOutcomeIfDataMissing:'If evidence is insufficient, use HOLD/NONE and report what is unverified; never fabricate prices.'}};
}
function err(s){throw Error(s);}
function assert(c,s){if(!c)err(s);}
export function signalMetrics(s){
 const e=s.direction==='LONG'?s.entryHigh:s.entryLow,risk=Math.abs(e-s.stop),tp=[s.tp1,s.tp2,s.tp3];
 const rr2=risk>0?Math.abs(tp[1]-e)/risk:NaN;
 return {referenceEntry:e,stopPct:-(risk/e*100),tp1Pct:Math.abs(tp[0]-e)/e*100,tp2Pct:Math.abs(tp[1]-e)/e*100,tp3Pct:Math.abs(tp[2]-e)/e*100,rr2};
}
export function validateSignal(s,snapshot){
 assert(s&&typeof s==='object','FORMAL_SIGNAL_OBJECT_REQUIRED');
 for(const k of ['id','issuedAt','dataAt','symbol','direction','tradeType','exchange','product','invalidation','supportingEvidence','opposingEvidence','issuedVersion'])assert(has(s[k]),'MISSING_SIGNAL_'+k);
 assert(['BTCUSDT','ETHUSDT'].includes(s.symbol),'INVALID_SYMBOL');
 assert(['LONG','SHORT'].includes(s.direction),'INVALID_DIRECTION');
 assert(['추세 지속형','지지·저항 반응형','추세 반전형'].includes(s.tradeType),'INVALID_TRADE_TYPE');
 assert(validTime(s.issuedAt)&&validTime(s.dataAt),'INVALID_ISSUE_TIME');
 assert(ms(s.issuedAt)>=ms(s.dataAt),'ISSUE_MUST_NOT_PRECEDE_DATA');
 for(const k of ['entryLow','entryHigh','stop','tp1','tp2','tp3'])assert(finite(s[k])&&s[k]>0,'INVALID_PRICE_'+k);
 assert(s.entryLow<=s.entryHigh,'ENTRY_RANGE_INVALID');
 const p=[s.tp1,s.tp2,s.tp3];
 if(s.direction==='LONG')assert(s.stop<s.entryLow&&s.entryHigh<p[0]&&p[0]<p[1]&&p[1]<p[2],'LONG_PRICE_ORDER_VIOLATION');
 else assert(s.stop>s.entryHigh&&s.entryLow>p[0]&&p[0]>p[1]&&p[1]>p[2],'SHORT_PRICE_ORDER_VIOLATION');
 assert(STATES.has(s.status)&&['PENDING','NEW_ENTRY'].includes(s.status),'NEW_SIGNAL_STATUS_INVALID');
 assert(s.issuedVersion==='V1.7','SOURCE_POLICY_VERSION_MISMATCH');
 const observed=s.symbol==='BTCUSDT'?snapshot.btcPrice:snapshot.ethPrice;
 if(s.status==='NEW_ENTRY')assert(s.entryLow<=observed&&observed<=s.entryHigh,'NEW_ENTRY_PRICE_OUTSIDE_ACTUAL_SNAPSHOT');
 if(s.status==='PENDING')assert(observed<s.entryLow||observed>s.entryHigh,'PENDING_PRICE_ALREADY_IN_RANGE');
 const snapshotAt=s.symbol==='BTCUSDT'?snapshot.btcTime:snapshot.ethTime;
 assert(s.dataAt===snapshotAt,'SIGNAL_DATA_TIME_MUST_EQUAL_SYMBOL_SNAPSHOT_TIME');
 assert(has(s.comparisonEvidence),'V17_COMPARE_A_TO_E_MISSING');
 assert(has(s.couplingEvidence),'V17_COUPLING_JUSTIFICATION_MISSING');
 const m=signalMetrics(s);assert(Number.isFinite(m.rr2),'RISK_METRIC_INVALID');
 if(m.rr2<2)assert(has(s.rr2ExceptionEvidence),'RR2_UNDER_2_NEEDS_EXPLICIT_ORIGINAL_POLICY_JUSTIFICATION');
 assert(snapshot&&snapshot.sha256,'SNAPSHOT_UNVERIFIED');
 return m;
}
export function checkEvent(evt,ledger){
 assert(evt&&has(evt.id)&&has(evt.at)&&has(evt.reason)&&has(evt.evidence),'EVENT_MISSING_REQUIRED_FIELDS');
 assert(validTime(evt.at),'EVENT_INVALID_TIME');
 const signal=ledger.find(s=>s.id===evt.id);
 assert(signal,'EVENT_UNKNOWN_SIGNAL');
 assert(!TERMINAL.has(signal.status),'TERMINAL_SIGNAL_IMMUTABLE');
 assert(ms(evt.at)>=ms(signal.original?.issuedAt||signal.issuedAt),'RETROACTIVE_EVENT_FORBIDDEN');
 if(signal.verifiedUntil&&ms(evt.at)<ms(signal.verifiedUntil))err('EVENT_TIME_BEFORE_PREVIOUS_EVENT');
 if(evt.kind==='REVISION'){
  assert(['NEW_ENTRY','ENTERED','TP1','TP2'].includes(signal.status),'REVISION_REQUIRES_ENTERED_SIGNAL');
  assert(evt.plan&&typeof evt.plan==='object','REVISION_PLAN_REQUIRED');
  for(const k of ['stop','tp1','tp2','tp3']) assert(finite(evt.plan[k]),'REVISION_PRICE_MISSING_'+k);
  if(signal.original.direction==='LONG')assert(evt.plan.stop>=signal.original.stop,'LONG_PROTECTIVE_STOP_MAY_NOT_WIDEN');
  else assert(evt.plan.stop<=signal.original.stop,'SHORT_PROTECTIVE_STOP_MAY_NOT_WIDEN');
  assert(has(evt.structureEvidence)&&has(evt.flowEvidence)&&has(evt.priorVersionNote),'REVISION_REQUIRES_REASON_AND_VERSION_EVIDENCE');
  return true;
 }
 assert(STATES.has(evt.kind)&&evt.kind!=='NEW_ENTRY','EVENT_KIND_INVALID');
 const from=signal.status;
 const transitions={PENDING:['ENTERED','CANCELLED','REDESIGNED_CANCELLED','UNVERIFIED'],NEW_ENTRY:['ENTERED','TP1','TP2','TP3_DONE','STOPPED','EARLY_EXIT','REVERSE_EXIT','UNVERIFIED'],ENTERED:['TP1','TP2','TP3_DONE','STOPPED','EARLY_EXIT','REVERSE_EXIT','UNVERIFIED'],TP1:['TP2','TP3_DONE','STOPPED','EARLY_EXIT','REVERSE_EXIT','UNVERIFIED'],TP2:['TP3_DONE','STOPPED','EARLY_EXIT','REVERSE_EXIT','UNVERIFIED'],UNVERIFIED:['PENDING','ENTERED','TP1','TP2','TP3_DONE','STOPPED','CANCELLED','REDESIGNED_CANCELLED','EARLY_EXIT','REVERSE_EXIT']};
 assert((transitions[from]||[]).includes(evt.kind),'INVALID_STATUS_TRANSITION_'+from+'_TO_'+evt.kind);
 if(['ENTERED','TP1','TP2','TP3_DONE','STOPPED'].includes(evt.kind))assert(has(evt.pricePathEvidence),'PRICE_PATH_EVIDENCE_REQUIRED');
 if(['CANCELLED','REDESIGNED_CANCELLED','EARLY_EXIT','REVERSE_EXIT'].includes(evt.kind))assert(has(evt.strongCounterEvidence),'CANCELLATION_OR_EXIT_REQUIRES_POLICY_EVIDENCE');
 return true;
}
export function validateOfficialSubmission(args,ledger,previousSnapshot=null){
 assert(args&&typeof args==='object','SUBMISSION_MISSING');
 const {raw,session='OFFICIAL_V17',policySha256,snapshotSha256,beforeLedgerSha256,decisionKey,analysis}=args;
 const shot=validateSnapshot(raw),sess=sessionId(session),lhs=ledgerHash(ledger);
 const evidenceQuality=assessSnapshotEvidence(raw,{previousSnapshot,ledger});
 for(const sym of ['BTC','ETH']){
  assert(!evidenceQuality.perSymbol[sym].flags.includes('OUT_OF_ORDER_SNAPSHOT'),
   'OUT_OF_ORDER_SNAPSHOT_CANNOT_CHANGE_OFFICIAL_LEDGER_'+sym);
 }
 assert(policySha256===POLICY_SHA256,'SOURCE_PROMPT_SHA_MISMATCH');
 assert(snapshotSha256===shot.sha256,'SNAPSHOT_SHA_MISMATCH');
 assert(beforeLedgerSha256===lhs,'LEDGER_HAS_CHANGED_FETCH_AGAIN');
 assert(decisionKey===digest({session:sess,snapshotSha256:shot.sha256,ledgerSha256:lhs,policySha256:POLICY_SHA256}),'DECISION_KEY_INVALID');
 assert(analysis&&typeof analysis==='object','ANALYSIS_REQUIRED');
 assert(has(analysis.reportMarkdown),'COMPLETE_USER_REPORT_REQUIRED');
 assert(analysis.directions&&typeof analysis.directions==='object','FOUR_DIRECTIONS_NOT_ANALYZED');
 for(const side of DIRECTIONS){
  const d=analysis.directions[side];
  assert(d&&has(d.marketEvidence)&&has(d.counterEvidence)&&d.alternatives&&typeof d.alternatives==='object','DIRECTION_EVIDENCE_REQUIRED_'+side);
  for(const a of ALTERNATIVES)assert(d.alternatives[a]&&has(d.alternatives[a].judgement)&&has(d.alternatives[a].evidence),'MISSING_A_TO_E_'+side+'_'+a);
 }
 assert(analysis.coupling&&has(analysis.coupling.status)&&has(analysis.coupling.evidence),'COUPLING_CROSSCHECK_REQUIRED');
 assert(analysis.checks&&typeof analysis.checks==='object','V17_FULL_16_STEP_CHECK_REQUIRED');
 for(const n of STEPS){const c=analysis.checks[String(n)];assert(c&&['CONFIRMED','UNVERIFIED','VIOLATION'].includes(c.status)&&has(c.evidence),'MISSING_V17_EXECUTION_STEP_'+n);}
 assert(analysis.clauses&&typeof analysis.clauses==='object','V17_ALL_0_TO_23_CLAUSES_REQUIRED');
 for(const id of CLAUSES){const c=analysis.clauses[id];assert(c&&['CONFIRMED','UNVERIFIED','VIOLATION'].includes(c.status)&&has(c.evidence),'V17_CLAUSE_MISSING_'+id);}
 assert(analysis.audit20&&has(analysis.audit20.rules)&&has(analysis.audit20.bias)&&has(analysis.audit20.evidence),'FINAL_20_AUDIT_REQUIRED');
 assert(analysis.conclusions&&analysis.conclusions.BTC&&analysis.conclusions.ETH,'BTC_ETH_CONCLUSION_REQUIRED');
 for(const sym of ['BTC','ETH'])assert(['LONG','SHORT','HOLD','MANAGE','UNVERIFIED'].includes(analysis.conclusions[sym].decision)&&has(analysis.conclusions[sym].evidence),'INVALID_CONCLUSION_'+sym);
 // Do not allow a confidently asserted status when the previous official snapshot
 // and incoming price path leave an active signal's intervening events unknowable.
 for(const sym of ['BTC','ETH']){
  if(evidenceQuality.perSymbol[sym].couldNotVerifySignals.length){
   assert(analysis.conclusions[sym].decision==='UNVERIFIED','GAP_REQUIRES_UNVERIFIED_CONCLUSION_'+sym);
  }
 }
 const news=analysis.newSignals??[],events=analysis.events??[];
 assert(Array.isArray(news)&&Array.isArray(events),'SIGNAL_EVENTS_ARRAY_REQUIRED');
 const ids=new Set(ledger.map(s=>s.id));
 for(const s of news){
  validateSignal(s,shot);
  const sym=s.symbol==='BTCUSDT'?'BTC':'ETH';
  assert(!evidenceQuality.perSymbol[sym].couldNotVerifySignals.length,
   'NEW_SIGNAL_BLOCKED_BY_UNVERIFIED_EXISTING_STATE_'+sym);
  assert(!evidenceQuality.perSymbol[sym].ambiguousSignals.length,
   'NEW_SIGNAL_BLOCKED_BY_AMBIGUOUS_OLD_ENTRY_'+sym);
  assert(!ids.has(s.id),'FORMAL_SIGNAL_ID_COLLISION');ids.add(s.id);
 }
 // A nonconflicting, continuous post-issuance price-zone crossing must not be
 // silently discarded as HOLD. It requires an original-ID ENTERED event.
 for(const sym of ['BTC','ETH']){
  const q=evidenceQuality.perSymbol[sym];
  for(const id of q.requiredEntrySignals){
   assert(events.some(e=>e.id===id&&e.kind==='ENTERED'),
    'CONFIRMED_PENDING_TOUCH_EVENT_MISSING_'+id);
  }
  for(const target of q.requiredExitSignals){
   const acceptable=target.kind==='TP1'?['TP1','TP2','TP3_DONE']:target.kind==='TP2'?['TP2','TP3_DONE']:[target.kind];
   assert(events.some(e=>e.id===target.id&&acceptable.includes(e.kind)),
    'CONFIRMED_EXIT_OR_TP_EVENT_MISSING_'+target.id+'_'+target.kind);
  }
  if(q.ambiguousSignals.length){
   assert(analysis.conclusions[sym].decision==='UNVERIFIED',
    'ENTRY_EXIT_ORDER_REQUIRES_UNVERIFIED_'+sym);
   assert(!events.some(e=>q.ambiguousSignals.includes(e.id)&&e.kind!=='UNVERIFIED'),
    'AMBIGUOUS_EVENT_REQUIRES_ORDERED_TRADE_EVIDENCE_'+sym);
  }
  if(q.couldNotVerifySignals.length){
   assert(!events.some(e=>q.couldNotVerifySignals.includes(e.id)&&e.kind!=='UNVERIFIED'),
    'GAP_EVENT_REQUIRES_MISSING_PRICE_PATH_'+sym);
  }
 }
 // Events are checked sequentially against the preceding committed or in-flight state.
 const staged=ledger.map(r=>({...r}));
 for(const newSignal of news)staged.push({id:newSignal.id,symbol:newSignal.symbol,original:newSignal,status:newSignal.status,verifiedUntil:newSignal.issuedAt,version:0});
 let lastEventAt=null;
 for(const event of events){
  if(lastEventAt)assert(ms(event.at)>=ms(lastEventAt),'EVENTS_NOT_IN_TIME_ORDER');
  checkEvent(event,staged);
  const row=staged.find(r=>r.id===event.id);
  if(event.kind==='REVISION')row.version=(row.version??0)+1;
  else row.status=event.kind;
  row.verifiedUntil=event.at;
  lastEventAt=event.at;
 }
 if(news.length){
  const syms=new Set(news.map(x=>x.symbol));
  for(const sym of syms){const existing=ledger.filter(x=>x.symbol===sym&&!TERMINAL.has(x.status)&&x.status!=='UNVERIFIED');const inMarket=existing.some(x=>['NEW_ENTRY','ENTERED','TP1','TP2'].includes(x.status));const next=news.filter(x=>x.symbol===sym);
   if(inMarket){
    const was=existing.find(x=>['NEW_ENTRY','ENTERED','TP1','TP2'].includes(x.status));
    for(const s of next.filter(x=>x.status==='NEW_ENTRY'&&x.direction!==was.original.direction)) {
     const exited=events.some(e=>e.id===was.id&&e.kind==='REVERSE_EXIT'&&ms(e.at)<=ms(s.issuedAt));
     assert(exited,'OPPOSITE_SIMULTANEOUS_ENTRY_FORBIDDEN');
    }
   }
   assert(next.filter(x=>x.status==='NEW_ENTRY').length<=1,'MULTIPLE_SIMULTANEOUS_ENTRY_FORBIDDEN');
  }
 }
 const alert=STEPS.some(n=>analysis.checks[String(n)].status==='VIOLATION')||CLAUSES.some(n=>analysis.clauses[n].status==='VIOLATION');
 assert(!(alert&&(news.length||events.length)),'RULE_VIOLATION_BLOCKS_SIGNAL_MUTATION');
 return {snapshot:shot,session:sess,newSignals:news,events,analysis,evidenceQuality,policySha256:POLICY_SHA256,decisionKey};
}
