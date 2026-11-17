// Effective-at plans are observations of management events, never rewritten originals.
const keys=['stop','tp1','tp2','tp3'];
const stamp=s=>{if(typeof s!=='string'||!/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(s))throw Error('REVISION_TIME_INVALID');const n=Date.parse(s.replace(' ','T')+'+09:00');if(!Number.isFinite(n))throw Error('REVISION_TIME_INVALID');return n;};
const plan=p=>{if(!p||!keys.every(k=>typeof p[k]==='number'&&Number.isFinite(p[k])&&p[k]>0))throw Error('REVISION_PLAN_INVALID');return Object.fromEntries(keys.map(k=>[k,p[k]]));};
export function planTimeline(signal){
 const original=signal.original||signal,base=plan(original);
 let revisions;
 if(Array.isArray(signal.managementRevisions)){
  revisions=signal.managementRevisions;
  if(signal.version!==undefined&&revisions.length!==signal.version)throw Error('REVISION_HISTORY_INCOMPLETE');
 }else if(signal.effectivePlan){
  if((signal.version??1)>1)throw Error('REVISION_HISTORY_INCOMPLETE');
  revisions=[{at:signal.planEffectiveAt,plan:signal.effectivePlan}];
 }else{if((signal.version??0)>0)throw Error('REVISION_HISTORY_INCOMPLETE');revisions=[];}
 let last=-Infinity;const out=[{at:-Infinity,plan:base}];
 for(const r of revisions){
  const at=stamp(r.at),p=plan(r.plan);
  if(at<last||original.issuedAt&&at<stamp(original.issuedAt))throw Error('REVISION_HISTORY_ORDER_INVALID');
  last=at;out.push({at,plan:p});
 }
 if(signal.effectivePlan&&keys.some(k=>out.at(-1).plan[k]!==signal.effectivePlan[k]))throw Error('REVISION_LATEST_PLAN_MISMATCH');
 return out;
}
export function planForSpan(signal,start,end){
 if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)throw Error('PLAN_SPAN_INVALID');
 const timeline=planTimeline(signal);let selected=timeline[0].plan;
 for(const x of timeline)if(x.at<=start)selected=x.plan;
 const interior=timeline.filter(x=>x.at>start&&x.at<end).map(x=>x.plan);
 return {plan:selected,plans:[selected,...interior],hasInteriorRevision:interior.length>0};
}
