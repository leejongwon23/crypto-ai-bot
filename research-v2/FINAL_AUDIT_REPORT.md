# BTC·ETH 재설계 실행·검증 보고서
2026-11-17 KST / 연구 후보 / 운영 미승격

## 결론 및 현재 범위

**두 소프트웨어 결함을 재현하고 후보 수정·회귀·DB 시험까지 수행했다.
원시 시장자료를 실제 확보하여 재현 가능한 관측 연구를 시작했으나,
요청된 수학적 자율진화 매매 시스템 전체와 통계적 성과 개선은 아직 완성·검증되지 않았다.**

사용 가능한 터미널이 없어 로컬 첨부파일 작업은 수행할 수 없었다.
연결된 GitHub·플러그인·공식 API를 직접 조사하고 GitHub Actions에서 실제 Node/PostgreSQL 시험을 실행했다.
새 매매모형을 검증한 것처럼 정책을 V2.0 FINAL로 올리거나 운영에 배포하지 않았다.

## A. 기존 원본·운영 상태 감사와 재현

|항목|직접 확인|한계|
|---|---|---|
|canonical 저장소|leejongwon23/crypto-ai-bot|public repository|
|운영 branch head|4cb79078ebef4dc5f113e20dd44e25068a9e34ec|Render 실행 커밋과 동일하다는 증거는 없음|
|V1.9 원문|96602 bytes,1263 lines,SHA03910e1c418de6a83ed8191ec59801481c933fd1358b33b6b3071a8d81d8505a|첨부 txt는 미독|
|플러그인|1.4.2, pluginrel_6afb81b633f88191ae2a3a315c3e89ed|이번 작업에서 릴리스하지 않음|
|원문 대조|플러그인 원문=policy-v19.mjs embedded UTF-8 전문, 완전 문자열·SHA 일치|기억/파일명 비교가 아님|
|공식 원장|OFFICIAL_V17,27신호·58사건|실 계좌 포지션/체결 미연동|
|공개 health|HTTP200,ready,V1.9 SHA 일치|서비스 ID·deploy ID·실제 실행커밋은 미확인|
|Render 연결|My Workspace / tea-d5nj03cmrvns73fp6520만 조회됨|선택되지 않아 서비스/DB 조회가 거부됨|

### 실제 호출 및 필드 경로

server.beginAnalysis -> validateSnapshot -> v17_ai_prepared_snapshots에 원문 저장 ->
reconcileVerifiedPricePath(제출 1m/연결경로/공개복구) -> prepare ->
ChatGPT가 4방향·A–E·국면·동적관리 판단 ->
publishAnalysis -> 세션 FOR UPDATE 잠금 -> 원문/정책/원장 SHA 및 decisionKey 확인 ->
validateOfficialSubmission -> validateSignal -> evaluateRangeStopContext ->
v17_ai_signals.original 신규 INSERT / v17_ai_events INSERT / 현재 status·verified_until 갱신 ->
공식 분석 JSON 저장 -> buildSignalView.

따라서 현재 방향·진입을 자체 학습하는 수학 서버가 있는 것은 아니다.
서버는 AI 분석을 검증·저장·재생하며 시간순 가격교차를 추적한다.
설명 문자열과 완료된 검증의 의미를 구분했다.
정상 쓰기 경로는 원본을 보존하나 별도 관리자 reset 코드도 존재한다.
이번 감사에서는 prepare/publication/reset/admin-close/order를 호출하지 않았다.

### ETH 사고 및 반증

공식 ID V19-ETH-SHORT-20261117-063519-RANGE1826:
SHORT1823.63,STOP1828.91,TP1=1819.37,TP2=1811.19,TP3=1809.22.
원장에는 6h high1836.88,5m/15m/30m/1h BULL,두 VWAP 위와 음수CVD가 함께 기록된다.

후속 STOP 이벤트는 USER_SUBMITTED_BYBIT_1M 자료의
06:43 high1829.60/low1827.44를 근거로 06:44에 기록되었다.
관측분과 분봉 종료 사건시각의 차이는 실제 코드로 설명된다.
정확한 틱 체결·계좌 손절과 동일시하지 않는다.
전체 원시 입력을 독립 거래소에서 다시 수집한 사고 재현은 아직 아니다.

fetched gate를 직접 실행한 결과:
- 원형 후보와 STOP=6h high 경계는 차단.
- STOP1839.50은 이 관문만 통과하며 동일 TP2 RR 약0.784.
- 반대EMA,비RANGE 전술은 이 특정 gate에서 전면 거부되지 않음.
- 필요한 VWAP 누락은 승인되지 않음.
- 거울 LONG 반례도 차단.

반추세·잘못된 박스·매도흡수·너무 짧은 STOP·국면 오인·시각오차 가설을 분리했다.
정확한 실패 원인을 이 중 하나로 확정하지 않았다.
음수CVD가 매도흡수를 증명하거나 0.707ATR STOP이 통계적으로 잘못됐다고 단정하지 않았다.
support/refutation/대안/필요자료는 hypotheses.json에 있다.

추가 재현:
1. 공개 MCP 도구 publish_official_v19_analysis는 수정 전 dispatcher에서 Unknown tool.
2. 원본 LONG entry100/STOP90,06:36부터 보호STOP95,06:37 low94의 합성 반례에서
   기존 minute 재생기는 ENTERED 유지. prepared aggregate 재생기도 original만 읽음.
이는 실제 과거 거래의 손실/누락을 새로 인정한 것이 아니라 코드 반례이다.

## B. 원인별 최소 후보 수정

|확정 원인|후보 수정|보존|
|---|---|---|
|공개 v19 별칭이 dispatcher 분기에 없음|기존 publication 분기에 v19 이름 추가|기존 검증/권한/저장 handler|
|두 재생기가 original만 참조|revision event 전체를 seq순 읽고 effective-at 계획 선택|original·발행시각·과거 사건|
|기존 표 시험이 옛 제목 기대|현행 가로표 header와 방향/상태 행으로 기대값 교정|출력 함수 자체|

관리계획 수정시각을 가로지르는 분봉/집계경로에서 STOP/TP 교차 순서가 불명확하면
유리한 순서를 고르지 않고 미확인으로 남긴다.
다중 이력 누락·순서오류·최종계획 불일치도 정상으로 간주하지 않는다.
현재 최종 STOP으로 과거 전체 경로를 덮어쓰지 않는다.

이 후보 변경은 운영에 배포하지 않았다. 과거 원장을 다시 판정하지 않았다.

## C. 데이터 상태·수학적 모형과 식별 가능성

Bybit REST를 실제 요청했으나 미국 CI에서 HTTP403으로 실패했다.
공식 integration guide의 미국IP 제한과 CI의 미국 실행지역은 이 현상과 부합한다.
토큰·권한 설정을 임의로 바꾸거나 다른 거래소 가격으로 대체하지 않았다.

공식 공개 아카이브에서 **2026-11-15 UTC 하루**를 실제 확보:

|관측|BTC|ETH|
|---|---:|---:|
|원시 체결|3,580,461|2,203,441|
|gzip bytes|126,793,002|79,941,375|
|1m/5m/15m|1440/288/96|1440/288/96|
|1h/4h/1D|24/6/1|24/6/1|
|파생시 누락된 버킷|0|0|
|음수흐름·비하락 15m|13|10|
|양수흐름·비상승 15m|8|18|

봉은 실제 체결에서 파생했다. REST 원시 OHLCV를 받았다고 표현하지 않는다.
이 수치는 미래 예측·승률·흡수 주체·고래 의도를 보여주지 않는다.
일별 자료의 파일 순서는 관측되지만 같은 timestamp의 매칭엔진 순서까지 증명하지 못한다.
11-16 UTC 사고일 아카이브는 조회 당시 목록에 없었다.
11-15 자료를 11-17 KST 사고 자료로 사용하지 않았다.

BTC archive SHA800505d9eaa5f2c7faff5f0faf59fce00c1a3dab6fd4768520df9262b7a3ea7b.
ETH archive SHA39af30c976ea01b6921fd6127d3521a2493ab00c0172d57aafa88910d3e3a6f8.

구현된 연구 기반:
- raw hash·수집시각·상품·시각·완결봉·중복·누락 계약.
- eventTime/knownAt와 retrospective/live vintage 분리.
- raw archive 저장·stream parsing·거래량/체결흐름·VWAP·다중 봉 파생.
- 가설용 OFI/가격반응 함수. 완전한 호가 연속성이 없으면 비활성.
- 표본 의존성을 가정하지 않는 KS 기술통계, Holm 검정, moving-block bootstrap.
- 정보구간 purging/embargo 및 홀드아웃 침범 방지 유틸리티.
- 명시적인 선물 수수료·불리한 슬리피지·signed funding 계산.
- 별도 append-only 연구 스키마와 digest chain.
  특권 DDL/전체 chain 재작성에 대한 외부 불변 보관까지 구현된 것은 아님.

미구현/비활성:
연속 WebSocket 호가·취소/체결 식별·청산 수집,장기 캐시/백필 스케줄러,
국면별 수학 모형 자체의 독립 채택,4방향 신규 수학모형의 공식 엔진 통합,
자동 경쟁모형 탐색·원인별 patch 생성·연속 shadow/canary/rollback controller.
모형 계열은 사건순서/분포변화/조건부 충격/지속시간/비모수 구간 등의 미선정 연구가설이다.
이를 가중 합산 거래점수로 바꾸지 않았다.

## D. 수정 전후 파일·해시·diff

runtime-change-manifest.json에 기준 commit/Git blob/SHA256/대상 함수를 기록했다.
기존 runtime 수정파일:
server.mjs,price-path-recovery.mjs,official-price-tracker.mjs,prepared-path-chain.mjs.
신규 pure helper: effective-plan-timeline.mjs.
기존 v19-core.mjs,regime-entry-risk.mjs,signal-table-view.mjs,원문은 변경하지 않았다.
DB 테이블/저장키 변경 없이 SELECT의 revision history 조회만 추가했다.
기존 테스트 하나의 오래된 출력 기대값을 수정하고 독립 반례시험을 추가했다.

Diff: https://github.com/leejongwon23/crypto-ai-bot/pull/23/files
검증한 runtime 변경 commit:522263abae57a1d924c0ea729fd6efaaab2d4528.
연구 수집·기술통계 최종 검증 commit:5d84bf2ddf7b4e4b924bc68e3058fb2b0687e2aa.
DB 실험 추가 commit:7463df3ca607f9a9d71c9b99a2f50dee9ce08372.
이후 보고서 파일만 추가해도 이 runtime hash들과 대조할 수 있다.

## E. 실제 시험과 미실행

- 문법: 후보 모든 mjs node --check 통과.
- 기준본:234개 중233통과/1실패. 현행 표와 맞지 않는 regex 실패를 확인.
- 후보 서비스:252/252 통과,skip0.
- 연구:17/17 통과,skip0.
- 별도 PostgreSQL16:2/2 통과. 실제 source DDL,revision SELECT,append-only UPDATE/DELETE/TRUNCATE 거부.
- 기존 gate·반대방향·경계·누락·동일입력·첫미완결봉·동일봉순서 불명·시점별계획·다중수정·원본보존 시험 수행.
- 운영 health200 및 정책SHA 일치 확인.
- 공식 원장·신호 원본·이벤트를 읽기 전후 각각 hash하여 일치 확인.

원장SHA:8bbcfe954e7f04326d30114afd4692c691cf5addb4e7c22f14062405a92d67cf.
원본SHA:e4a402a0dc2c008e5378bcefe7328440446abe1800a4ab89d2063d4576ca2e21.
이벤트SHA:c5377047c43083f0343c95d9c3922bee315330ce615075a915e68a3fcf34747e.
각 before=after. ledgerHash가 이벤트 전체를 포함하지 않으므로 이벤트 digest를 별도 비교했다.

미실행:
실제 운영 DB 카탈로그/마이그레이션 대조,Render deploy commit 확인,
사고 전체 독립 틱 경로,장기 전략 재현,국면별 walk-forward,잠긴 미사용 holdout,
실비용 후 순기대손익·낙폭·기회상실·잘못된 switching 측정,실시간 shadow/canary/rollback.
단위 함수 시험은 실제 데이터 누출 부재나 시장 성능의 증거가 아니다.

## F. BEFORE/AFTER 표본

|종류|BEFORE|AFTER|의미|
|---|---|---|---|
|실제 ETH 원본1건|당시 SHORT 발행 후 원장 STOP|현재 기존gate는 원형 후보 거부; 역사 기록 유지|새 정책의 개선 표본이 아님|
|실제 dispatcher code|v19 별칭 Unknown tool|기존 handler 도달|호출 수정|
|합성 보호STOP|95 유효 뒤 low94인데 ENTERED|STOP95 교차|시간순 추적 수정|
|합성 수정 전 low94|최종계획을 덮어씌우면 과거 STOP 오판 위험|원본90 적용,STOP 아님|미래정보 차단|
|합성 분봉내수정|유리한 순서 선택 가능성|미확인|체결순서 조작 방지|

실제 신규 거래 표본3~5개와 전후 실현손익은 확보하지 못했다.
합성 반례나 gate 재생을 실제 거래 성과로 세지 않았다.

## G. 승격·배포·미완료 이유

통계적으로 확인된 매매 성능 개선은 없다.
V2.0 운영 정책,승격된 새 국면 모형,자동배포가 준비됐다고 주장하지 않는다.
원문 전문+연구계약 초안 MASTER_PROMPT_V2_RESEARCH_NOT_ACTIVE.txt만 별도로 저장했다.

현재 운영 branch,플러그인1.4.2,공식 원장은 유지한다.
프로토콜 기간/표본/alpha/비용/허용낙폭 값은 사용자 합의 전 null이다.
promotionGate는 증거 입력 계약이며 서명된 시험증명이나 배포 controller가 아니다.
신규 자동주문/권한 확대는 구현·활성화하지 않았다.

Render 조회 실패는 workspace 미선택이다.
연결은 My Workspace 사용에 대한 사용자 확인 후 workspaceId를 넘기도록 요구한다.
그 확인 전 serviceId/deployId/live commit을 추측하지 않았다.
따라서 운영 배포/스테이징/카나리를 실행하지 않았고 되돌릴 배포도 없다.
향후 코드만 rollback하고 원장 데이터는 덮어쓰지 않는 절차는 DEPLOYMENT.md에 있다.

## H. 파일 및 증거 링크

- 검토용 PR: https://github.com/leejongwon23/crypto-ai-bot/pull/23
- 소스/문서: https://github.com/leejongwon23/crypto-ai-bot/tree/v20-research-audit-20261117
- 연구 CI: https://github.com/leejongwon23/crypto-ai-bot/actions/runs/43992446138
- 실제 DB 시험: https://github.com/leejongwon23/crypto-ai-bot/actions/runs/43992705215
- 원시자료·파생봉·TAP·runtime diff·소스 묶음 ZIP:
  https://github.com/leejongwon23/crypto-ai-bot/actions/runs/43992446138/artifacts/14287083748
- Artifact SHA256:314663e1341aea426aaca3a37a2db3cf8ed821cd3fbb54e664a158feea3153fa.
- 데이터 artifact는30일 보존. 저장소 코드/문서는 commit으로 보존한다.
- 성과표 regime-performance.csv는 미측정 항목을 NA로 명시했다. 60%/65%를 실측 성과로 쓰지 않았다.

참조한 1차 문서:
https://bybit-exchange.github.io/docs/v5/guide
https://bybit-exchange.github.io/docs/v5/market/kline
https://bybit-exchange.github.io/docs/v5/market/recent-trade
https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
https://public.bybit.com/trading/BTCUSDT/
https://public.bybit.com/trading/ETHUSDT/
