import {buildQuantContext,sha} from './quant-structure.mjs';
import {generateResearchProposals,synthesizeMechanismRevisions} from './quant-evolution.mjs';
import {freezeResearchForecasts,reviewResearchPath,observedCounterexamples} from './quant-lifecycle.mjs';

export const RESEARCH_DDL=`
CREATE TABLE IF NOT EXISTS v20_research_observations(source_sha TEXT PRIMARY KEY,observed JSONB NOT NULL,received_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v20_research_forecasts(id TEXT PRIMARY KEY,source_sha TEXT NOT NULL,forecast JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v20_research_reviews(forecast_id TEXT NOT NULL,observation_sha TEXT NOT NULL,review JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),PRIMARY KEY(forecast_id,observation_sha));
CREATE TABLE IF NOT EXISTS v20_research_candidates(id TEXT PRIMARY KEY,candidate JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v20_research_rejections(source_sha TEXT PRIMARY KEY,reason TEXT NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v20_research_counterexamples(id TEXT PRIMARY KEY,counterexample JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS v20_integrity_checkpoints(id TEXT PRIMARY KEY,checkpoint JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE OR REPLACE FUNCTION v20_research_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'V20_RESEARCH_APPEND_ONLY'; END $$;
DO $$ DECLARE t TEXT; BEGIN FOREACH t IN ARRAY ARRAY['v20_research_observations','v20_research_forecasts','v20_research_reviews','v20_research_candidates','v20_research_counterexamples','v20_research_rejections','v20_integrity_checkpoints'] LOOP
 EXECUTE format('DROP TRIGGER IF EXISTS v20_research_immutable ON %I',t);
 EXECUTE format('CREATE TRIGGER v20_research_immutable BEFORE UPDATE OR DELETE OR TRUNCATE ON %I FOR EACH STATEMENT EXECUTE FUNCTION v20_research_reject_mutation()',t);
END LOOP; END $$;
CREATE OR REPLACE FUNCTION v20_preserve_official_original() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
 IF TG_OP='DELETE' THEN RAISE EXCEPTION 'V20_OFFICIAL_ORIGINAL_IMMUTABLE'; END IF;
 IF OLD.id IS DISTINCT FROM NEW.id OR OLD.session IS DISTINCT FROM NEW.session OR OLD.symbol IS DISTINCT FROM NEW.symbol OR OLD.original IS DISTINCT FROM NEW.original THEN RAISE EXCEPTION 'V20_OFFICIAL_ORIGINAL_IMMUTABLE'; END IF;
 RETURN NEW; END $$;
DROP TRIGGER IF EXISTS v20_official_original_guard ON v17_ai_signals;
CREATE TRIGGER v20_official_original_guard BEFORE UPDATE OR DELETE ON v17_ai_signals FOR EACH ROW EXECUTE FUNCTION v20_preserve_official_original();
DROP TRIGGER IF EXISTS v20_official_signals_truncate_guard ON v17_ai_signals;
CREATE TRIGGER v20_official_signals_truncate_guard BEFORE TRUNCATE ON v17_ai_signals FOR EACH STATEMENT EXECUTE FUNCTION v20_research_reject_mutation();
DROP TRIGGER IF EXISTS v20_official_events_guard ON v17_ai_events;
CREATE TRIGGER v20_official_events_guard BEFORE UPDATE OR DELETE OR TRUNCATE ON v17_ai_events FOR EACH STATEMENT EXECUTE FUNCTION v20_research_reject_mutation();
`;
function minuteBars(raw,symbol,asOf){
 const start=raw.indexOf('========== '+symbol+' ('+symbol+'USDT) =========='),next=raw.indexOf('========== ',start+12),part=raw.slice(start,next<0?undefined:next);
 const block=raw.match(/\[QUANT_RAW_JSON\]\s*([\s\S]*?)\s*\[\/QUANT_RAW_JSON\]/);const j=block?JSON.parse(block[1])[symbol]??{}:{};
 let rows=j.bars??j.timeframes?.['1m']??[];
 if(!rows.length&&part.includes('[10B]'))rows=part.slice(part.indexOf('minute_OHLC_KST:')).split(/\r?\n/).flatMap(l=>{const m=l.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),([\d.]+),([\d.]+),([\d.]+),([\d.]+)$/);return m?[{t:m[1],open:+m[2],high:+m[3],low:+m[4],close:+m[5]}]:[];});
 return rows.map(r=>({t:typeof r.t==='number'?r.t:Date.parse(String(r.t??r.start).replace(' ','T')+(String(r.t??r.start).includes('T')?'':'+09:00')),intervalMs:r.intervalMs??60000,open:Number(r.open),high:Number(r.high),low:Number(r.low),close:Number(r.close)})).filter(b=>b.intervalMs===60000&&b.t+60000<=asOf).map(b=>({...b,end:b.t+60000}));
}
export async function captureResearchObservation(db,raw,{now=Date.now()}={}){
 const context=buildQuantContext(raw,[]),source=context.sourceSha256;
 const exists=await db.query('SELECT source_sha FROM v20_research_observations WHERE source_sha=$1',[source]);if(exists.rows.length)return {replay:true,sourceSha256:source};
 const symbols={};for(const symbol of ['BTC','ETH']){const s=context.symbols[symbol],at=Date.parse(s.asOf.replace(' ','T')+'+09:00'),start=raw.indexOf('========== '+symbol+' ('+symbol+'USDT) =========='),next=raw.indexOf('========== ',start+12),part=raw.slice(start,next<0?undefined:next);if(!Number.isFinite(at)||at>now)throw Error('RESEARCH_FUTURE_SOURCE_NOT_AVAILABLE_AT_RECEIVE');symbols[symbol]={asOf:s.asOf,lastPrice:Number(part.match(/^lastPrice:\s*([\d.]+)/m)?.[1]),structuralEvents:s.structure.structuralEvents??{},bars:minuteBars(raw,symbol,at)};}
 // Only the DB retains source observations and prospective research plans.
 // The official ledger is never written by these functions.
 const forecasts=freezeResearchForecasts(context,{frozenAt:now});
 await db.query('INSERT INTO v20_research_observations(source_sha,observed,received_at) VALUES($1,$2,$3) ON CONFLICT DO NOTHING',[source,JSON.stringify({sourceSha256:source,symbols,receivedAt:now}),new Date(now)]);
 for(const forecast of forecasts)await db.query('INSERT INTO v20_research_forecasts(id,source_sha,forecast) VALUES($1,$2,$3) ON CONFLICT DO NOTHING',[forecast.id,source,JSON.stringify(forecast)]);
 for(const candidate of generateResearchProposals(context))await db.query('INSERT INTO v20_research_candidates(id,candidate) VALUES($1,$2) ON CONFLICT DO NOTHING',[candidate.id,JSON.stringify(candidate)]);
 return {replay:false,sourceSha256:source,frozen:forecasts.filter(x=>x.status==='FROZEN_PENDING').length,abstained:forecasts.filter(x=>x.status==='ABSTAIN').length};
}
export async function runShadowCycle(db){
 const backlog=await db.query(`SELECT s.snapshot_sha,s.raw FROM v17_ai_prepared_snapshots s WHERE s.session='OFFICIAL_V17'
 AND NOT EXISTS(SELECT 1 FROM v20_research_observations o WHERE o.source_sha=s.snapshot_sha)
 AND NOT EXISTS(SELECT 1 FROM v20_research_rejections r WHERE r.source_sha=s.snapshot_sha)
 ORDER BY s.created_at ASC LIMIT 25`);
 for(const row of backlog.rows){try{if(sha(row.raw)!==row.snapshot_sha)throw Error('PRESERVED_SOURCE_HASH_MISMATCH');await captureResearchObservation(db,row.raw);}catch{await db.query('INSERT INTO v20_research_rejections(source_sha,reason) VALUES($1,$2) ON CONFLICT DO NOTHING',[row.snapshot_sha,'INVALID_OR_INCOMPLETE_MARKET_SOURCE']);}}
 const obs=await db.query('SELECT source_sha,observed FROM v20_research_observations ORDER BY received_at ASC LIMIT 1000');
 const forecasts=await db.query(`SELECT f.id,f.forecast FROM v20_research_forecasts f WHERE forecast->>'status'='FROZEN_PENDING'
 AND COALESCE((SELECT review->>'status' FROM v20_research_reviews r WHERE r.forecast_id=f.id ORDER BY created_at DESC LIMIT 1),'') NOT IN ('STOPPED','TP3_COMPLETE','UNVERIFIED','NOT_ENTERED_INVALIDATED')
 ORDER BY (SELECT MAX(created_at) FROM v20_research_reviews r WHERE r.forecast_id=f.id) ASC NULLS FIRST,f.created_at ASC LIMIT 64`);
 let reviews=0;
 for(const f of forecasts.rows){
  const received=obs.rows.filter(o=>o.observed.receivedAt>f.forecast.frozenAt),bars=new Map(),conflicts=new Set();
  for(const row of received){for(const b of row.observed.symbols[f.forecast.symbol]?.bars??[]){const old=bars.get(b.t);if(old&&sha(old)!==sha(b))conflicts.add(b.t);else bars.set(b.t,b);}}
  const sourceSetSha=sha(received.map(o=>o.source_sha));
  const counterexamples=observedCounterexamples(f.forecast,received.map(x=>x.observed));
  for(const example of counterexamples)await db.query('INSERT INTO v20_research_counterexamples(id,counterexample) VALUES($1,$2) ON CONFLICT DO NOTHING',[sha({forecast:f.id,...example}),JSON.stringify({forecastId:f.id,...example})]);
  for(const revision of synthesizeMechanismRevisions({...f.forecast.mechanism,id:f.forecast.modelSha256},counterexamples)){
   const candidate={...revision,symbol:f.forecast.symbol,sourceForecastId:f.id};await db.query('INSERT INTO v20_research_candidates(id,candidate) VALUES($1,$2) ON CONFLICT DO NOTHING',[sha(candidate),JSON.stringify(candidate)]);
  }
  const review=conflicts.size?{status:'UNVERIFIED',reason:'CONFLICTING_OBSERVED_CANDLES'}:reviewResearchPath(f.forecast,[...bars.values()]);
  const r=await db.query('INSERT INTO v20_research_reviews(forecast_id,observation_sha,review) VALUES($1,$2,$3) ON CONFLICT DO NOTHING',[f.id,sourceSetSha,JSON.stringify({...review,modelSha256:f.forecast.modelSha256,symbol:f.forecast.symbol,regime:f.forecast.regime,sourceSetSha})]);reviews+=r.rowCount;
 }
 return {...await researchStatus(db),newReviews:reviews,backfilled:backlog.rows.length,backfillMayRemain:backlog.rows.length===25,truncated:obs.rows.length===1000||forecasts.rows.length===64};
}
export async function researchStatus(db){
 const count=await db.query(`SELECT (SELECT COUNT(*)::int FROM v20_research_observations) AS observations,(SELECT COUNT(*)::int FROM v20_research_forecasts) AS forecasts,(SELECT COUNT(*)::int FROM v20_research_candidates) AS candidates,(SELECT COUNT(*)::int FROM v20_research_reviews) AS reviews,(SELECT COUNT(*)::int FROM v20_research_counterexamples) AS counterexamples`);
 const statuses=await db.query("SELECT review->>'status' AS status,COUNT(*)::int AS n FROM (SELECT DISTINCT ON(forecast_id) forecast_id,review FROM v20_research_reviews ORDER BY forecast_id,created_at DESC,observation_sha) r GROUP BY review->>'status'");
 return {schema:'V20_RESEARCH_CONTROL_1',...count.rows[0],statuses:statuses.rows,netPerformanceVerified:false,promotionBlocked:true,reasons:['INDEPENDENT_BASELINE_COST_AND_LOCKED_PROTOCOL_REQUIRED'],ordersEnabled:false};
}
export async function takeIntegritySnapshot(db){
 const signals=await db.query("SELECT id,original FROM v17_ai_signals WHERE session='OFFICIAL_V17' ORDER BY id");
 const events=await db.query("SELECT seq::text,id,event,at_kst FROM v17_ai_events WHERE session='OFFICIAL_V17' ORDER BY seq");
 return {originals:signals.rows.map(r=>({idSha256:sha(r.id),sha256:sha(r.original)})),events:events.rows.map(r=>sha(r))};
}
export function compareIntegrity(before,after){
 const current=new Map(after.originals.map(r=>[r.idSha256,r.sha256]));
 const originalsIntact=before.originals.every(r=>current.get(r.idSha256)===r.sha256),eventsIntact=before.events.every((x,i)=>after.events[i]===x);
 return {intact:originalsIntact&&eventsIntact,originalsIntact,eventsIntact,beforeSignals:before.originals.length,currentSignals:after.originals.length,beforeEvents:before.events.length,currentEvents:after.events.length};
}
export async function researchControl(db,body){
 if(body.action==='STATUS')return researchStatus(db);
 if(body.action==='CYCLE')return runShadowCycle(db);
 if(body.action==='CHECKPOINT'){
  const checkpoint=await takeIntegritySnapshot(db),id=sha(checkpoint);
  await db.query('INSERT INTO v20_integrity_checkpoints(id,checkpoint) VALUES($1,$2) ON CONFLICT DO NOTHING',[id,JSON.stringify(checkpoint)]);
  return {checkpoint:id,signals:checkpoint.originals.length,events:checkpoint.events.length};
 }
 if(body.action==='BASELINE'){
  const prior=await db.query('SELECT id,checkpoint FROM v20_integrity_checkpoints ORDER BY created_at DESC,id DESC LIMIT 1');
  if(!prior.rows.length)throw Error('PREVIOUS_INTEGRITY_BASELINE_REQUIRED');
  const p=prior.rows[0];return {checkpoint:p.id,signals:p.checkpoint.originals.length,events:p.checkpoint.events.length,previouslyPersisted:true};
 }
 if(body.action==='VERIFY'){
  if(!/^[a-f0-9]{64}$/.test(body.checkpoint??''))throw Error('CHECKPOINT_HASH_REQUIRED');
  const r=await db.query('SELECT checkpoint FROM v20_integrity_checkpoints WHERE id=$1',[body.checkpoint]);if(!r.rows.length)throw Error('CHECKPOINT_NOT_FOUND');
  return compareIntegrity(r.rows[0].checkpoint,await takeIntegritySnapshot(db));
 }
 throw Error('RESEARCH_CONTROL_ACTION_FORBIDDEN');
}
