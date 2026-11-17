# BTC·ETH 연구·감사 후보 — 운영 미승격

현재 결과는 **연구 기반과 재현 시험**이다. 검증된 자율진화 매매 시스템의 완성이 아니다.
운영 정책은 MASTER PROMPT V1.9 FINAL, 플러그인은 1.4.2로 유지한다.
운영 API·공식 원장·자동 주문을 이 코드에서 호출하거나 변경하지 않는다.

## 실제 기준본

- 저장소: leejongwon23/crypto-ai-bot.
- 기준 브랜치: v17-canonical-engine.
- 기준 커밋: 4cb79078ebef4dc5f113e20dd44e25068a9e34ec.
- V1.9 원문: 96,602 bytes / 1,263 lines.
- SHA256: 03910e1c418de6a83ed8191ec59801481c933fd1358b33b6b3071a8d81d8505a.
- 플러그인 1.4.2 원문과 저장소 policy-v19.mjs의 UTF-8 문자열이 정확히 일치함을 직접 비교.
- 공식 API 조회: OFFICIAL_V17 27 signals / 58 events.
- 조회 당시 ledgerSha256: 8bbcfe954e7f04326d30114afd4692c691cf5addb4e7c22f14062405a92d67cf.
- Render 실제 서비스 ID·배포 커밋은 워크스페이스 선택이 없어 미확인.
- 첨부 txt는 실행기 부재로 읽지 못했음. 이름만 보고 내용/해시를 추정하지 않음.

## 수정 전 감사 결과와 호출경로

1. snapshot -> beginAnalysis -> validateSnapshot -> prepared raw 영속 저장.
2. reconcileVerifiedPricePath -> 사용자 1m/연결 경로/공개 Bybit 경로 복구 -> 시간순 기계적 사건.
3. prepare -> 정책·원장·4방향/A–E·국면·동적관리 요구를 AI에게 제공.
4. ChatGPT가 방향과 신규 신호를 판단. 서버는 자체 학습/예측 모형을 돌리지 않는다.
5. publishAnalysis -> 세션 행 잠금 -> 원문/정책/원장 hash 검사 -> validateOfficialSubmission.
6. validateSignal -> RANGE_BOUNDARY일 때 evaluateRangeStopContext -> 사건순서와 기존 ID 관리 검사.
7. new signal original JSONB 삽입 + event 추가 + 현재 상태 갱신 -> official 분석을 한 트랜잭션으로 저장.
8. buildSignalView -> BTC/ETH 가로표 + 저장된 종목별 시황/조건부 전망. 표시 함수는 쓰기를 하지 않는다.

DB 스키마는 server.mjs에서 직접 읽었으나 실제 DB의 카탈로그·마이그레이션 상태와 동일하다는 증거는 없다.
signals의 original은 INSERT로 보존되고 상태/verifiedUntil/version만 UPDATE된다.
events는 정상 발행 경로에서 INSERT만 하지만 별도 관리자 reset 경로도 존재한다.
이 감사는 reset/관리자 종료/실거래를 호출하지 않는다.

## 확정된 소프트웨어 결함과 최소 패치

MCP 목록에는 publish_official_v19_analysis가 있지만 dispatcher는 v17/v18 이름만 처리한다.
실제 dispatcher를 의존성 stub과 함께 실행하면 v19만 -32601 Unknown tool을 반환했다.
패치는 mcp의 기존 분기에 v19 이름 하나를 추가하고 테스트 접근용 mcp export만 추가한다.
원장 저장 SQL·신호 판단·손절·타겟·파서·기존 방어관문은 수정하지 않는다.
새 테스트는 실 DB 없이 세 별칭이 동일 handler의 PERSISTENT_DATABASE_REQUIRED에 도달하는지,
write scope가 모두 필요한지, 미등록 이름은 여전히 거부되는지 검사한다.
이 수정은 **도구 호출 오류 수정**이며 시장 수익 개선이 아니다.

## ETH 사고: 사실과 가설

공식 원본 ID: V19-ETH-SHORT-20261117-063519-RANGE1826.
원본 SHORT 1823.63 / STOP 1828.91 / TP 1819.37, 1811.19, 1809.22.
6h high 1836.88, 5m/15m/30m/1h BULL, 가격은 두 VWAP 위.
음수 CVD는 원본 설명에 기록되어 있으나 그 부호가 향후 하락을 증명하지 않는다.
후속 공식 이벤트는 사용자 제출 Bybit 표기 1m 06:43 high1829.60/low1827.44를 근거로 STOP을 기록.
관측 분봉은 06:43이고 사건 저장시각은 분봉 종료 06:44이다. 정확한 tick 체결시각이 아니다.
전체 원시 경로를 독립 거래소에서 다시 받은 것이 아니므로 실제 계좌 손절 체결과 동일시하지 않는다.

현재 gate를 그대로 실행하면 원형 후보를 차단한다. STOP=6h high 경계도 차단한다.
넓은 STOP1839.50은 이 gate만 통과하며 같은 TP2의 RR은 약0.784이므로 승인 근거가 되지 않는다.
반대 EMA 또는 비RANGE 전술은 이 특정 gate에서 차단되지 않는다.
이는 한 형태의 승인 누락 방어가 작동한다는 뜻이며 일반적 성과시험이 아니다.
경쟁가설과 반증·필요 자료는 hypotheses.json에 별도로 기록했다.

추가로 auditMinuteCandles가 effectivePlan을 참조하지 않는 합성 반례를 재현했다.
진입100/원본STOP90/미래 유효 보호STOP95에서 이후 low94를 넣으면 ENTERED가 유지된다.
실제 수정 이력 전체를 시간구간별로 재생해야 하므로 단순히 현재 STOP으로 과거 봉을 덮어씌우지 않았다.
해당 문제는 **미수정/운영 승격 차단 사유**이다. 과거 공식 결과는 변경하지 않는다.

## 데이터와 수학적 연구 구조

lib.mjs는 제품 정합성·완결 OHLCV·누락·중복·knownAt 시점·소급자료 구분을 검사한다.
collect.mjs는 bounded 공개 REST 수집기이며 현재 호가/최근 체결을 과거 사고자료로 바꾸지 않는다.
원문응답·수집시각·서버응답·HTTP오류·rate-limit header·raw hash를 보존한다.
OHLCV 한 페이지 범위를 넘는 장기 수집, 웹소켓 지속 수집, 재접속 snapshot/delta 재동기화,
호가 취소/체결 구분, 완전 CVD, 청산 이벤트는 아직 미구현 또는 확보되지 않았다.
그 자료를 요구하는 모형은 비활성이다. 실제 확보 범위는 CI artifact의 data/manifest.json만이 근거이다.

OFI는 연속성이 확인된 같은 상품의 best bid/ask 상태 변화에서만 산출한다.
완전 체결창이 없으면 taker/CVD 추론을 거부한다. 음수 흐름+상승은 흡수 **가설**이다.
KS 통계는 경험분포 차이만 보고하고 독립표본 p값을 붙이지 않는다.
purgedWalkForward는 featureStart~labelEnd 범위와 embargo를 분리하고 홀드아웃 진입을 거부한다.
Holm 검정과 moving-block bootstrap은 의존성과 반복탐색 검증을 위한 도구이다.
어떤 통계량도 가중 합산 거래점수 또는 거래승인 커트라인으로 결합하지 않는다.
모형 계열은 가설 등록 상태이며, 특정 국면 모형·예측 우열·실전 성능을 확정하지 않는다.

## 시험·배포 상태

단계별 실제 실행 상태는 TEST_MATRIX.md 및 CI 로그/아티팩트에 기록한다.
연구 스키마는 별도 DB용 초안이며 운영 DB에 적용하지 않았다.
프로토콜의 기간/표본수/허용낙폭/비용/검정수준은 null이다. 사용자가 승인한 값인 척하지 않는다.
promotionGate는 누락값과 미검증을 거부하는 **증거 계약 검사기**이고 배포 controller가 아니다.
서명된 CI 증명, 자동 patch 생성, 정책 승인, shadow/paper 연속운영, canary 배포,
실패시 자동 rollback은 아직 연결되어 있지 않다.
실제 새로운 매매 후보 3~5개 비교와 국면별 장기 성과표는 자료 부재로 미제공이다.
1개 실제 기록 사례와 합성 경계/반대사례를 혼동하지 않는다.

연구 artifact 생성: node research-v2/audit.mjs research-output
연구시험: node --test research-v2/*.test.mjs
기존 회귀: V17_NO_LISTEN=1 node --test v17-common/*.test.mjs v17-common/tests/*.test.mjs
수집: node research-v2/collect.mjs research-output/data

운영 승격 조건 및 rollback 절차는 DEPLOYMENT.md에 있다.
