/**
 * Opt-in additive, read-only forecast query over append-only market evidence.
 * No formal signal/ledger writes. Observations and source-labeled 1m bars are
 * immutable to protect forward evaluation from silent hindsight revision.
 */
import {extractObservation,extractClosedMinuteCandles,buildEmpiricalForecast,publicForecast} from './empirical-trajectory.mjs';
export const FORECAST_TABLE_SQL=String.raw`
CREATE TABLE IF NOT EXISTS v17_ai_forecast_observations (
 session TEXT NOT NULL, symbol TEXT NOT NULL, snapshot_sha TEXT NOT NULL,
 at_ms BIGINT NOT NULL, price DOUBLE PRECISION NOT NULL, atr DOUBLE PRECISION NOT NULL,
 features JSONB NOT NULL, source TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 PRIMARY KEY(session,symbol,snapshot_sha)
);
CREATE INDEX IF NOT EXISTS v17_ai_forecast_obs_time ON v17_ai_forecast_observations(session,symbol,at_ms DESC);
CREATE TABLE IF NOT EXISTS v17_ai_forecast_minutes (
 session TEXT NOT NULL, symbol TEXT NOT NULL, at_ms BIGINT NOT NULL,
 o DOUBLE PRECISION NOT NULL, h DOUBLE PRECISION NOT NULL,
 l DOUBLE PRECISION NOT NULL, c DOUBLE PRECISION NOT NULL,
 source TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 PRIMARY KEY(session,symbol,at_ms)
);
`;
export async function persistForecastEvidence(db,session,sha,raw){
 const results={};
 for(const sym of ['BTC','ETH']){
  const o=extractObservation(raw,sym);
  if(!o){results[sym]={status:'FEATURES_INCOMPLETE'};continue;}
  await db.query('INSERT INTO v17_ai_forecast_observations(session,symbol,snapshot_sha,at_ms,price,atr,features,source) VALUES($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT DO NOTHING',
   [session,sym,sha,o.atMs,o.price,o.atr,JSON.stringify(o.features),o.source]);
  const bars=extractClosedMinuteCandles(raw,sym);
  // Batch the complete, closed source-candle records (not the live minute).
  // Existing observations are NEVER overwritten by later conflicting reports.
  for(let offset=0;offset<bars.length;offset+=200){
   const batch=bars.slice(offset,offset+200),parameters=[];
   const values=batch.map((b,i)=>{
    const j=i*8;parameters.push(session,sym,b.t,b.open,b.high,b.low,b.close,'USER_SUBMITTED_BYBIT_1M_UNVERIFIED');
    return '('+Array.from({length:8},(_,k)=>'$'+(j+k+1)).join(',')+')';
   });
   await db.query('INSERT INTO v17_ai_forecast_minutes(session,symbol,at_ms,o,h,l,c,source) VALUES '+values.join(',')+' ON CONFLICT DO NOTHING',parameters);
  }
  results[sym]={status:'STORED',atMs:o.atMs,closedMinutesReceived:bars.length,source:'USER_SUBMITTED_NOT_AUTHENTICATED'};
 }
 return results;
}
function normalizeRows(rows,sym){
 return rows.map(x=>({symbol:sym,atMs:Number(x.at_ms),price:Number(x.price),atr:Number(x.atr),features:x.features})).filter(x=>Number.isFinite(x.atMs)&&Number.isFinite(x.price)&&x.price>0);
}
function normalizeCandles(rows){
 return rows.map(x=>({t:Number(x.at_ms),open:Number(x.o),high:Number(x.h),low:Number(x.l),close:Number(x.c)}));
}
export async function researchForecast(db,session,raw,opportunityContext){
 const out={model:'V17_EMPIRICAL_ANALOGUE_RESEARCH_V1',
  source:'USER_SUBMITTED_SNAPSHOTS_AND_BYBIT_LABELLED_MINUTES_NOT_INDEPENDENTLY_AUTHENTICATED',
  doesNotTrade:true,estimatedProfitability:false,status:'RESEARCH_ONLY',symbols:{}};
 for(const sym of ['BTC','ETH']){
  const q=extractObservation(raw,sym);
  if(!q){out.symbols[sym]={status:'SOURCE_FEATURES_INCOMPLETE'};continue;}
  const earliest=q.atMs-14*86400000;
  const past=await db.query('SELECT at_ms,price,atr,features FROM v17_ai_forecast_observations WHERE session=$1 AND symbol=$2 AND at_ms>=$3 AND at_ms<$4 ORDER BY at_ms DESC LIMIT 600',
    [session,sym,earliest,q.atMs]);
  const last=Number(past.rows.at(-1)?.at_ms);
  const from=Number.isFinite(last)?Math.max(last,q.atMs-14*86400000):q.atMs-86400000;
  const minutes=await db.query('SELECT at_ms,o,h,l,c FROM v17_ai_forecast_minutes WHERE session=$1 AND symbol=$2 AND at_ms>=$3 AND at_ms<$4 ORDER BY at_ms DESC LIMIT 22000',
    [session,sym,from,q.atMs]);
  const model=buildEmpiricalForecast({raw,symbol:sym,history:normalizeRows(past.rows,sym),candles:normalizeCandles(minutes.rows)});
  const plans=opportunityContext?.symbols?.[sym];
  // A distant RR opportunity is listed for comparison but never promoted to a
  // historical forecast unless its entry path has been observed in analogues.
  const examples=[];
  for(const side of ['LONG','SHORT']){
   for(const x of (plans?.[side]?.nearbyAlternatives||[]).slice(0,2))examples.push(x);
   const remote=plans?.[side]?.distantContingencies?.[0];if(remote)examples.push(remote);
  }
  out.symbols[sym]=publicForecast(model,examples);
 }
 return out;
}
