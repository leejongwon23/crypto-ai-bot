# BTC·ETH 研究·검증 후보

**최종 상태: 코드·자료 수집·기술 검증 완료 범위가 있으며, 자율진화 매매 시스템 전체와 시장 성능은 미검증. 운영 미배포.**

[최종 감사 보고서](FINAL_AUDIT_REPORT.md)에서 실제 수행/미수행과 남은 제약을 먼저 확인한다.
[검증 결과](verification-result.json), [원장 불변 증명](ledger-integrity-result.json),
[변경 함수·해시](runtime-change-manifest.json), [실제 관측 연구](descriptive-study-result.json)를 제공한다.

## 보존 및 수정

- 기존 V1.9 원문, 1.4.2 SKILL·manifest를 baseline/에 보존했다.
- 운영 기준 커밋 4cb79078ebef4dc5f113e20dd44e25068a9e34ec는 유지한다.
- 공식 OFFICIAL_V17의 27신호·58사건과 원본/이벤트 digest를 전후 대조해 동일함을 확인했다.
- 발행 v19 별칭 누락과 관리계획 유효시각 추적 누락만 후보 코드에서 수정했다.
- 가로표 출력 함수는 변경하지 않고, 오래된 테스트 기대값만 현행 계약으로 고쳤다.
- 신호 생성 모형·공식 정책·기존 방어관문·DB 저장키를 임의 교체하지 않았다.

## 연구 도구와 실제 자료

lib.mjs: OHLCV 정합성, 누락/중복, known-at 조회, hash-chain, 관측 흐름/OFI,
purged walk-forward 분리, Holm 보정, moving-block bootstrap, 명시적 비용계산, 승격 증거계약.
가중 합산 점수·임의 신뢰도·보상/벌점·가격예측 자동 승인은 없다.

REST는 미국 CI에서 HTTP403으로 실패했다.
공식 공개 일별 아카이브 2026-11-15 UTC는 BTC3,580,461체결/ETH2,203,441체결을 실제 확보했다.
각 1,440개 1분 봉에서 5m/15m/1h/4h/1D를 파생하고 거래량 기반 일간 VWAP와 signed flow를 계산했다.
과거 다운로드 자료이므로 live-vintage가 아니며 사고 당일 자료로 대체하지 않는다.

## 미완료

장기 데이터셋·완전 실시간 호가/체결/청산 수집, 독립 국면 모형·4방향 수학 후보의 운영 통합,
경쟁 전략 역사적 재현과 walk-forward/미사용 holdout 성과, 자동 원인별 patch 생성,
shadow/paper 연속운영·canary 승격·자동 rollback controller는 완성되지 않았다.
promotionGate는 입력 증거 계약 검사이며 실제 배포 권한을 부여하지 않는다.
protocol.draft.json은 기간/기준/한도를 승인 전 null로 보존한다.

## 파일 안내

- FINAL_AUDIT_REPORT.md: 요구 A~H 순서의 결과와 한계.
- MASTER_PROMPT_V2_RESEARCH_NOT_ACTIVE.txt: 기존 V1.9 전문을 포함한 연구 계약 초안. 운영 정책 아님.
- hypotheses.json: 사고 경쟁가설과 반증 자료.
- TEST_MATRIX.md, verification-result.json: 실제 실행 수준.
- schema.sql: 별도 연구 DB용 스키마. PostgreSQL16에서 DDL/UPDATE·DELETE·TRUNCATE 거부 시험.
- DEPLOYMENT.md: 승격/롤백 절차 초안.
- regime-performance.csv: 실제 전략 시험 미수행을 NA로 명시한 표.
- GitHub CI artifact: 원시 gzip·파생자료·시험 TAP·runtime diff·소스 묶음. 30일 보존.

전체 서비스: V17_NO_LISTEN=1 node --test v17-common/*.test.mjs v17-common/tests/*.test.mjs
연구시험: node --test research-v2/*.test.mjs
DB시험: 별도 workflow의 disposable localhost PostgreSQL만 허용한다.
