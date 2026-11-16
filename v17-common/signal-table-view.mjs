// Read-only presentation of the official V1.7 ledger. Never creates or changes
// a signal, a price touch, a position, a target, or a formal outcome.
const TERMINAL = new Set(['TP3_DONE','STOPPED','CANCELLED','REDESIGNED_CANCELLED','EARLY_EXIT','REVERSE_EXIT','USER_CLOSED']);
const ENTERED = new Set(['NEW_ENTRY','ENTERED','TP1','TP2']);
// UNVERIFIED may mean an earlier entry was PROVEN, while a later exit is not.
// Never erase the observed entry or declare the still-open position as fact.
const enteredEarlier = row=>row.status==='UNVERIFIED'&&ENTERED.has(row.unverifiedFrom);
const hasKnownEntry = row=>ENTERED.has(row.status)||enteredEarlier(row);
// This is a separate user account claim; it never resolves the exchange price-path gap.
const userConfirmedNoAccountEntry = row => row.status==='UNVERIFIED' && row.unverifiedFrom==='PENDING' && row.userAccountEntry==='NOT_ENTERED';
const LABEL = {
 PENDING:'대기',NEW_ENTRY:'신규 진입',ENTERED:'진입',
 TP1:'TP1 도달',TP2:'TP2 도달',UNVERIFIED:'미확인',USER_CLOSED:'사용자 요청 추적 종료'
};
const CHANGE_LABEL = {
 ENTERED:'대기 → 진입 자동전환',NEW_ENTRY:'즉시 진입',
 CANCELLED:'신호 취소',REDESIGNED_CANCELLED:'기존 신호 재설계 취소',
 EARLY_EXIT:'조기 종료',REVERSE_EXIT:'반대 방향 전환 종료',
 TP1:'TP1 도달',TP2:'TP2 도달',TP3_DONE:'TP3 완료',
 STOPPED:'손절 종료',UNVERIFIED:'과거 상태 미확인 전환',
 REVISION:'손절·목표 관리변경',USER_CLOSED:'사용자 요청 추적 종료'
};
export const SIGNAL_TABLE_OUTPUT_RULE='최종 응답은 signalView.markdown 전체만 출력한다. BTC/ETH 고정 2열 표를 나란히 보여주고 신호가 없는 종목에는 진입 시그널 없음이라고 표시한다. 표 밑에는 BTC·ETH 각각 실제 관측 시황과 조건부 방향 예측 및 반대 시나리오를 짧게 출력한다. marketOutlook 미저장 구형 분석은 전망 미기록으로 표기하고 소급 예측하지 않는다. 검증된 신호 변경만 changeNotices로 추가한다. V1.9 분석·검수·원장 규칙은 유지하며 새 대화창도 동일하다.';
const num=x=>typeof x==='number'&&Number.isFinite(x);
const money=x=>num(x)?'$'+x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';
function signedPct(v){const n=Math.abs(v)<0.0000001?0:v;return (n>0?'+':'')+n.toFixed(2)+'%';}
function pctInterval(row,price){
 const o=row.original||row;
 if(!num(price)||!num(o.entryLow)||!num(o.entryHigh)||!['LONG','SHORT'].includes(o.direction))return '—';
 const side=o.direction==='LONG'?1:-1;
 const values=[o.entryLow,o.entryHigh].map(e=>side*(price-e)/e*100).sort((a,b)=>a-b);
 if(Math.abs(values[0]-values[1])<0.005)return signedPct((values[0]+values[1])/2);
 return signedPct(values[0])+'~'+signedPct(values[1]);
}
const cell=(row,value)=>money(value)+' ('+pctInterval(row,value)+')';
function currentPctDisplay(row,price){
 if(userConfirmedNoAccountEntry(row))return '— (미진입·사용자 확인)';
 if(row.status==='PENDING')return '— (미진입)';
 if(!num(price)||!hasKnownEntry(row))return '미확인';
 const pct=pctInterval(row,price);
 if(enteredEarlier(row))return pct+' (보유 가정·청산 미확인)';
 return pct+' (진입범위·현재가 기준)';
}
const status=row=>{
 if(row.status==='UNVERIFIED')return userConfirmedNoAccountEntry(row)?'미진입 (사용자 확인·시장경로 미확인)':enteredEarlier(row)?'진입 확인·이후 청산 미확인':'진입 여부 미확인';
 return LABEL[row.status]||row.status||'—';
};
function notes(changes=[],ledger=[]){
 const byId=new Map(ledger.map(x=>[x.id,x]));
 const result=[];
 for(const x of changes){
  if(!x||typeof x!=='object')continue;
  const name=CHANGE_LABEL[x.kind];
  if(name){const row=byId.get(x.id);const s=row?.symbol?.replace('USDT','')||x.symbol?.replace('USDT','')||'신호';result.push(s+' '+name);}
  else if(x.relationToExisting==='SAME_DIRECTION_ADD_ON')result.push((x.symbol||'').replace('USDT','')+' 추가 진입 신호');
  else if(x.relationToExisting==='OPPOSITE_PENDING')result.push((x.symbol||'').replace('USDT','')+' 반대 방향 대기 스위칭 신호');
 }
 return [...new Set(result)];
}
const byCoin=(all,sym)=>all.filter(r=>r&&r.original&&!TERMINAL.has(r.status)&&r.symbol===sym+'USDT').sort((a,b)=>String(a.original.issuedAt||'').localeCompare(String(b.original.issuedAt||''))||String(a.id).localeCompare(String(b.id)));
function rr(row,p){const o=row.original,e=o.direction==='LONG'?o.entryHigh:o.entryLow;if(!num(e)||!num(p.stop)||!num(p.tp2)||e===p.stop)return '—';if(o.direction==='LONG'&&!(p.stop<e&&p.tp2>e)||o.direction==='SHORT'&&!(p.stop>e&&p.tp2<e))return '—';return '1:'+Math.abs((p.tp2-e)/(e-p.stop)).toFixed(2);}
function marketComment(sym,outlook){const a=outlook?.[sym];if(!a||!['currentSituation','expectedPath','alternative'].every(k=>typeof a[k]==='string'&&a[k].trim()))return '**'+sym+' 시황·전망:** 해당 분석의 시황·전망 코멘트 미기록 (미확인).';return '**'+sym+' 현재 시황:** '+a.currentSituation+'\n\n**'+sym+' 예상 흐름(조건부 전망):** '+a.expectedPath+' (반대 시나리오: '+a.alternative+')';}
export function buildSignalView({ledger=[],snapshot=null,changes=[],outlook=null,management=null}={}){
 const rows={BTC:byCoin(ledger,'BTC'),ETH:byCoin(ledger,'ETH')};
 const price=s=>s==='BTC'?snapshot?.btcPrice:snapshot?.ethPrice;
 const time=s=>s==='BTC'?snapshot?.btcTime:snapshot?.ethTime;
 const plan=r=>({stop:r.original.stop,tp1:r.original.tp1,tp2:r.original.tp2,tp3:r.original.tp3,...(r.effectivePlan||{})});
 const show=(s,f,empty='—')=>rows[s].length?rows[s].map(f).join(' / '):empty;
 const cellRow=(name,fn)=>'| '+[name,fn('BTC'),fn('ETH')].join(' | ')+' |';
 const lines=['| 항목 | BTC | ETH |','| --- | --- | --- |',
  cellRow('기준시각(KST)',s=>time(s)||'미확인'),
  cellRow('현재가',s=>money(price(s))),
  cellRow('진입 시그널',s=>show(s,r=>r.original.direction+' · '+status(r),'진입 시그널 없음')),
  cellRow('신호 ID',s=>show(s,r=>r.id)),
  cellRow('진입가',s=>show(s,r=>r.original.entryLow===r.original.entryHigh?money(r.original.entryLow):money(r.original.entryLow)+'~'+money(r.original.entryHigh))),
  cellRow('현재가 대비 손익률',s=>show(s,r=>currentPctDisplay(r,price(s)))),
  cellRow('손절',s=>show(s,r=>cell(r,plan(r).stop))),
  ...['tp1','tp2','tp3'].map(k=>cellRow(k.toUpperCase(),s=>show(s,r=>cell(r,plan(r)[k])))),
  cellRow('TP2 손익비',s=>show(s,r=>rr(r,plan(r)))),
  cellRow('상태',s=>show(s,status,'진입 시그널 없음')),
  cellRow('현재 판단',s=>management?.[s]?.action||'미확인'),
  cellRow('추가 진입',s=>show(s,r=>r.original?.relationToExisting==='SAME_DIRECTION_ADD_ON'?'추가 진입 '+r.id:'없음','없음'))];
 const entries=[...rows.BTC,...rows.ETH].map(row=>({id:row.id,symbol:row.symbol,direction:row.original.direction,status:row.status,unverifiedFrom:row.unverifiedFrom||null,userAccountEntry:row.userAccountEntry||null,userAccountEntryAt:row.userAccountEntryAt||null,currentPrice:num(price(row.symbol.replace('USDT','')))?price(row.symbol.replace('USDT','')):null,entryLow:row.original.entryLow,entryHigh:row.original.entryHigh,stop:plan(row).stop,tp1:plan(row).tp1,tp2:plan(row).tp2,tp3:plan(row).tp3,stopPct:pctInterval(row,plan(row).stop),tp1Pct:pctInterval(row,plan(row).tp1),tp2Pct:pctInterval(row,plan(row).tp2),tp3Pct:pctInterval(row,plan(row).tp3),currentIndicativePct:hasKnownEntry(row)&&num(price(row.symbol.replace('USDT','')))?pctInterval(row,price(row.symbol.replace('USDT',''))):null,currentIndicativeConditional:enteredEarlier(row),historicalExitUnknown:row.status==='UNVERIFIED'}));
 return {mode:'BTC_ETH_HORIZONTAL_WITH_OUTLOOK',markdown:lines.join('\n')+'\n\n'+marketComment('BTC',outlook)+'\n\n'+marketComment('ETH',outlook),changeNotices:notes(changes,ledger),entries,marketOutlook:outlook||null};
}
