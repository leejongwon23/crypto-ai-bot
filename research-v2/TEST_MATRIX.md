# 실제 시험 결과

|검증|수행 결과|증거 범위|
|---|---|---|
|문법|후보의 mjs 전체 node --check 통과|GitHub Node22|
|기준본 회귀|234중233통과/1실패|예전 표 regex 기대값 실패|
|후보 서비스 회귀|252/252 통과|기존기능+발행별칭+시점별관리계획|
|연구 단위|17/17 통과|knownAt·누락·OFI·비용·분리·bootstrap·파생봉|
|동일입력 반복|실제 fetched gate V8 동일 결과|실 DB 부하시험 아님|
|사고 후보 gate|원형 차단; 6h 경계 차단; 넓은STOP 이 gate만 통과|축약한 당시 수치; 전체raw 재현 아님|
|반대방향/경계|거울 LONG·반대EMA·비RANGE·누락 입력 검사|합성 반례; 시장성과 아님|
|관리계획 유효시각|원본보존·사전/사후·다중수정·양방향·분봉내불명·집계경로 통과|새로운 과거 거래 성과를 생성하지 않음|
|DB 통합|2/2 통과|일회성 PostgreSQL16; 실제 source DDL과 readLedger SELECT|
|append-only SQL|UPDATE/DELETE/TRUNCATE 모두 거부 확인|별도 연구 schema; 특권 DDL 재작성 방어 아님|
|공식 원장|전후 원장/원본/이벤트 digest 일치|27신호/58사건 read-only API|
|live health|HTTP200 ready, V1.9 정책SHA 일치|DB 설정상태만; 배포커밋 증명 아님|
|독립 과거 데이터|11-15 UTC BTC3580461/ETH2203441체결 확보|일별 공식 archive; live-vintage 아님|
|사고 전체 역사경로|미검증|11-16 UTC archive 미게시, REST403|
|실제 바뀐 신규신호3~5개|미확보|1실제 기록과 합성검증 사례를 분리|
|국면별 전략 walk-forward|미실행|기간·표본·비용·검정 protocol 미확정|
|실비용 후 전략성과|미실행|비용 함수 시험 통과와 다름|
|운영 DB 실물/Render deploy|미조회|workspace 선택이 필요|
|shadow/canary/자동rollback|미구현/미실행|절차 초안만|
|첨부txt 비교|미수행|현재 로컬 파일 실행기 없음|

완전 회귀 성공과 연구 단위시험 성공은 시장 정확도/승률/수익성의 증거가 아니다.
