# 실행 범위 및 검증 상태

|항목|현재 증거|제한|
|---|---|---|
|원문 대조|플러그인/저장소 본문 완전 문자열 일치, 순수 JS SHA256 계산|첨부 파일 미독|
|사고 원본/결과|공식 원장 조회 27신호/58사건|개인 계좌 체결 미확인|
|ETH gate/경계/반대방향|동일 fetched 함수를 isolated V8에서 실제 실행|전체 raw 및 Node suite와 다름|
|동일입력 반복|V8 동일 결과 확인|실 DB 동시성 시험 아님|
|사고 STOP 교차|원장에 인용된 06:43 봉으로 STOP 교차 재생|전체 선행경로 독립 검증 미완료|
|발행 별칭 버그|실 dispatcher+stub: v19 Unknown tool, v17/v18 handler 진입|원격 서버 재현/운영 배포 아님|
|별칭 후보 패치|같은 stub 시험에서 세 별칭 handler 진입|Node CI 별도 확인 필요|
|관리버전 tracker|합성 effectivePlan95 vs original90/low94에서 누락 재현|미수정; 실거래 손실 추정 금지|
|문법/단위/통합/전체회귀|GitHub Actions 결과 참조|로그 없이 통과 주장 금지|
|역사적 재현/다년 수집|미실행|공개 수집 결과는 data/manifest만 확인|
|국면별 walk-forward 성과|미실행|승률·순기대손익·DD 미확인|
|실비용 계산|새 lib에 명시 비용 시험 준비|실제 수수료·슬리피지 추정 미확인|
|데이터 공백/제품/knownAt|연구 단위시험 준비|운영 수집기 통합 미완료|
|원장 불변|전후 read-only hash 비교 필요|이벤트 hash는 ledgerSha에 포함되지 않음|
|DB 스키마 및 append-only|운영 source schema 읽음; 별도 연구 schema 초안|실 DB 카탈로그/SQL 실행 미검증|
|live 응답|공식 ledger API 응답 성공; 공개 health는 CI 진단|Render commit/deploy ID 미확인|
|shadow/canary/rollback|절차 초안만|구현·운영 미완료|

데이터 누출 방지용 합성 시험이 통과해도 실제 데이터의 누출 없음이 증명되는 것은 아니다.
추가 실제표본 3~5개는 동결된 raw와 전후 결정이 확보된 후에만 작성한다.
