import test from 'node:test';
import assert from 'node:assert/strict';
import {buildSignalView,SIGNAL_TABLE_OUTPUT_RULE} from './signal-table-view.mjs';

const BTC={id:'btc-short',symbol:'BTCUSDT',status:'PENDING',original:{
 id:'btc-short',symbol:'BTCUSDT',direction:'SHORT',entryLow:69500,entryHigh:69540,
 stop:69820,tp1:69254.3,tp2:68820,tp3:68274,issuedAt:'2026-11-13 21:11:13'}};
const ETH={id:'eth-short',symbol:'ETHUSDT',status:'UNVERIFIED',unverifiedFrom:'ENTERED',original:{
 id:'eth-short',symbol:'ETHUSDT',direction:'SHORT',entryLow:1944.8,entryHigh:1948.2,
 stop:1969,tp1:1933,tp2:1894.1,tp3:1882.5,issuedAt:'2026-11-13 17:00:23'}};
const shot={btcPrice:69234.20,ethPrice:1958.37};
test('BTC and still-open UNVERIFIED ETH are both visible with entry-relative stop/TP percentages',()=>{
 const x=buildSignalView({ledger:[ETH,BTC],snapshot:shot});
 assert.equal(x.mode,'SIGNAL_TABLE_ONLY');
 assert.match(x.markdown,/BTC SHORT \| ETH SHORT/);
 assert.match(x.markdown,/\$69,500\.00~\$69,540\.00/);
 assert.match(x.markdown,/\$69,820\.00 \(-0\.46%~-0\.40%\)/);
 assert.match(x.markdown,/\$68,820\.00 \(\+0\.98%~\+1\.04%\)/);
 assert.match(x.markdown,/\$1,969\.00 \(-1\.24%~-1\.07%\)/);
 assert.match(x.markdown,/\$1,894\.10 \(\+2\.61%~\+2\.78%\)/);
 assert.match(x.markdown,/진입 이력·미확인/);
 assert.match(x.markdown,/— \(미진입\)/);
 assert.match(x.markdown,/\| 미확인 \|/);
 assert.deepEqual(x.changeNotices,[]);
 assert.equal(x.entries.length,2);
 assert.equal(x.entries[0].currentIndicativePct,null);
 assert.equal(x.entries[1].currentIndicativePct,null);
});
test('verified entered position displays indicative current price percentage, never a fill claim',()=>{
 const entered={...BTC,status:'ENTERED'};
 const result=buildSignalView({ledger:[entered],snapshot:{btcPrice:69234.2}});
 assert.match(result.markdown,/\+0\.38%~\+0\.44% \(진입범위 기준\)/);
 assert.equal(result.entries[0].currentIndicativePct,'+0.38%~+0.44%');
});
test('terminal signals omitted from default table, but changes carry concise notifications',()=>{
 const rows=[BTC,{...ETH,status:'CANCELLED'}];
 const result=buildSignalView({ledger:rows,snapshot:shot,changes:[
  {id:'eth-short',kind:'CANCELLED',at:'2026-11-13 21:30:00'},
  {id:'btc-short',kind:'ENTERED',at:'2026-11-13 21:31:00'}]});
 assert.equal(result.entries.length,1);
 assert.doesNotMatch(result.markdown,/ETH SHORT/);
 assert.ok(result.changeNotices.includes('ETH 신호 취소'));
 assert.ok(result.changeNotices.includes('BTC 대기 → 진입 자동전환'));
});
test('add-on and opposite-side switching alerts are exceptional notes',()=>{
 const added={id:'new',symbol:'BTCUSDT',direction:'SHORT',relationToExisting:'SAME_DIRECTION_ADD_ON'};
 const switchPlan={id:'new2',symbol:'BTCUSDT',direction:'LONG',relationToExisting:'OPPOSITE_PENDING'};
 const x=buildSignalView({ledger:[BTC],snapshot:shot,changes:[added,switchPlan]});
 assert.deepEqual(x.changeNotices,['BTC 추가 진입 신호','BTC 반대 방향 대기 스위칭 신호']);
});
test('LONG and revised effective stop/targets use correct percentage sign and original entry range',()=>{
 const long={id:'L',symbol:'BTCUSDT',status:'TP1',original:{
  id:'L',symbol:'BTCUSDT',direction:'LONG',entryLow:100,entryHigh:102,
  stop:95,tp1:105,tp2:110,tp3:115},
  effectivePlan:{stop:99,tp1:105,tp2:112,tp3:118}};
 const r=buildSignalView({ledger:[long],snapshot:{btcPrice:108}});
 assert.match(r.markdown,/\$99\.00 \(-2\.94%~-1\.00%\)/);
 assert.match(r.markdown,/\$112\.00 \(\+9\.80%~\+12\.00%\)/);
 assert.match(r.markdown,/\+5\.88%~\+8\.00% \(진입범위 기준\)/);
});
test('view generation does not alter the official ledger or invent a record',()=>{
 const input=[structuredClone(BTC),structuredClone(ETH)];
 const orig=JSON.stringify(input);
 const x=buildSignalView({ledger:input,snapshot:shot});
 assert.equal(JSON.stringify(input),orig);
 assert.equal(x.entries.length,2);
 const empty=buildSignalView({ledger:[],snapshot:shot});
 assert.match(empty.markdown,/활성 정식 신호 없음/);
 assert.match(SIGNAL_TABLE_OUTPUT_RULE,/새 대화창/);
});
