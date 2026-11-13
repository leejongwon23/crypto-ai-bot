// Read-only presentation of the official V1.7 ledger. Never creates or changes
// a signal, a price touch, a position, a target, or a formal outcome.
const TERMINAL = new Set(['TP3_DONE','STOPPED','CANCELLED','REDESIGNED_CANCELLED','EARLY_EXIT','REVERSE_EXIT']);
const ENTERED = new Set(['NEW_ENTRY','ENTERED','TP1','TP2']);
// UNVERIFIED may mean an earlier entry was PROVEN, while a later exit is not.
// Never erase the observed entry or declare the still-open position as fact.
const enteredEarlier = row=>row.status==='UNVERIFIED'&&ENTERED.has(row.unverifiedFrom);
const hasKnownEntry = row=>ENTERED.has(row.status)||enteredEarlier(row);
const LABEL = {
 PENDING:'대기',NEW_ENTRY:'신규 진입',ENTERED:'진입',
 TP1:'TP1 도달',TP2:'TP2 도달',UNVERIFIED:'미확인'
};
const CHANGE_LABEL = {
 ENTERED:'대기 → 진입 자동전환',NEW_ENTRY:'즉시 진입',
 CANCELLED:'신호 취소',REDESIGNED_CANCELLED:'기존 신호 재설계 취소',
 EARLY_EXIT:'조기 종료',REVERSE_EXIT:'반대 방향 전환 종료',
 TP1:'TP1 도달',TP2:'TP2 도달',TP3_DONE:'TP3 완료',
 STOPPED:'손절 종료',UNVERIFIED:'과거 상태 미확인 전환',
 REVISION:'손절·목표 관리변경'
};
export const SIGNAL_TABLE_OUTPUT_RULE =
 '최종 사용자 응답은 signalView.markdown 표만 출력한다. 시장설명·검수·미확인 사유·분석과정의 별도 문단을 쓰지 않는다. '+
 '새 진입/대기→진입 자동전환, 취소, 재설계, 추가진입, 방향 스위칭, 손절/익절/관리변경이 실제 공식원장에서 일어난 경우에만 signalView.changeNotices를 표 밑에 짧게 추가한다. '+
 '종료되지 않은 기존 BTC/ETH 정식 신호는 UNVERIFIED도 포함하여 모두 표시한다. 손절과 TP1/2/3에 최초 진입범위 대비 예상 손익률(%)을 병기한다. '+
 '대기 신호의 미체결 수익은 계산하지 않는다. 과거 진입이 확인된 UNVERIFIED 신호는 현재가 대비 손익률을 반드시 계산하되 (보유 가정·청산 미확인)으로 표시하고 실제 보유·실현수익으로 단정하지 않는다. '+
 '실제 분석·4방향 비교·V1.7 검수·원장 등록 단계는 축약하거나 생략하지 않는다. 새 대화창에서도 동일 규칙이다.';
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
 if(row.status==='PENDING')return '— (미진입)';
 if(!num(price)||!hasKnownEntry(row))return '미확인';
 const pct=pctInterval(row,price);
 if(enteredEarlier(row))return pct+' (보유 가정·청산 미확인)';
 return pct+' (진입범위·현재가 기준)';
}
const status=row=>{
 if(row.status==='UNVERIFIED')return enteredEarlier(row)?'진입 확인·이후 청산 미확인':'진입 여부 미확인';
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
export function buildSignalView({ledger=[],snapshot=null,changes=[]}={}){
 const active=ledger.filter(row=>row&&row.original&&!TERMINAL.has(row.status)&&['BTCUSDT','ETHUSDT'].includes(row.symbol))
  .sort((a,b)=>a.symbol.localeCompare(b.symbol)||String(a.original.issuedAt||'').localeCompare(String(b.original.issuedAt||''))||a.id.localeCompare(b.id));
 const current=(row)=>row.symbol==='BTCUSDT'?snapshot?.btcPrice:snapshot?.ethPrice;
 const plan=row=>({...row.original?.stop!==undefined?{stop:row.original.stop,tp1:row.original.tp1,tp2:row.original.tp2,tp3:row.original.tp3}:{},...(row.effectivePlan||{})});
 const headings=active.map(row=>row.symbol.replace('USDT','')+' '+row.original.direction);
 const mk=(label,fn)=>'| '+[label,...active.map(fn)].join(' | ')+' |';
 const lines=['| '+['항목',...headings].join(' | ')+' |','| '+['---',...active.map(()=>'---')].join(' | ')+' |'];
 if(active.length){
  lines.push(mk('현재가',row=>money(current(row))));
  lines.push(mk('진입가',row=>row.original.entryLow===row.original.entryHigh?money(row.original.entryLow):money(row.original.entryLow)+'~'+money(row.original.entryHigh)));
  lines.push(mk('현재가 대비 손익률',row=>currentPctDisplay(row,current(row))));
  lines.push(mk('손절',row=>cell(row,plan(row).stop)));
  for(const k of ['tp1','tp2','tp3'])lines.push(mk(k.toUpperCase(),row=>cell(row,plan(row)[k])));
  lines.push(mk('상태',status));
 }else lines.push('| 활성 정식 신호 없음 |');
 const changeNotices=notes(changes,ledger);
 return {mode:'SIGNAL_TABLE_ONLY',markdown:lines.join('\n'),changeNotices,
  entries:active.map(row=>({id:row.id,symbol:row.symbol,direction:row.original.direction,status:row.status,unverifiedFrom:row.unverifiedFrom||null,
   currentPrice:num(current(row))?current(row):null,
   entryLow:row.original.entryLow,entryHigh:row.original.entryHigh,
   stop:plan(row).stop,tp1:plan(row).tp1,tp2:plan(row).tp2,tp3:plan(row).tp3,
   stopPct:pctInterval(row,plan(row).stop),tp1Pct:pctInterval(row,plan(row).tp1),tp2Pct:pctInterval(row,plan(row).tp2),tp3Pct:pctInterval(row,plan(row).tp3),
   currentIndicativePct:hasKnownEntry(row)&&num(current(row))?pctInterval(row,current(row)):null,
   currentIndicativeConditional:enteredEarlier(row),
   historicalExitUnknown:row.status==='UNVERIFIED'
  }))};
}
