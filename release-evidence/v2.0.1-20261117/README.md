# BTC·ETH 2.0.1 완료·미완료·검증 구분

2026-11-17 KST. 최초 원문을 바꾸어 만든 V2.0 분석 규칙은 유지하고, 빠져 있던 연구·안전 배포 연결을 보완했다. 이 보고서는 이전 보고서의 미완료 상태 중 이번에 실제로 바뀐 부분을 갱신한다.

## 아주 쉽게

새 판단을 먼저 저장한다 → 이후 실제 가격과 비교한다 → 반대 증거가 있으면 판단 조건을 바꾼 연구 후보를 만든다 → 별도 자료에서도 확인된 후보만 단계적으로 적용한다 → 배포 검사가 실패하면 이전 검증 코드로 돌아간다.

컴퓨터가 이런 순서로 일할 장치를 추가했다. 이 장치가 생겼다고 시장 예측이 더 정확해진 것으로 보지는 않는다.

## 완성·교체·배포한 것

|항목|실제 완료|
|---|---|
|판단과 원자료 보존|별도 연구 DB에 실제 수신·동결 시각과 원본 SHA를 보존. UPDATE/DELETE/TRUNCATE 방어|
|후행 가격 시험|동결 이후 완결 1분 경로만 사용. 소급 진입·같은 봉의 유리한 순서 선택 금지|
|반례와 수정 후보|손절 전 반대 구조와 동결 구조 무효화를 관측해 저장. 조건 그래프 수정 후보 생성. 단일 손실을 원인 확정이나 승격 근거로 쓰지 않음|
|단계 적용|SHADOW→BTC 한정 CANARY→LIVE. 각 단계는 새 독립 자료와 별도 증거 필요. 미승격 모형의 modelId로 정식 신호 발행 거부|
|실제 성과 자료 검사|원시 경로·고정 비용을 재생하여 제출 순손익과 대조. 실제 TAP, 별도로 커밋한 검정계획, 표본 적정성, 손실 한도, 자료 중첩·중복 검사|
|검정자료 재사용 방지|자동 시험에서는 holdout 소모를 평가 전에 저장. 실패나 코드 복귀 후에도 소모 이력 보존|
|배포 감시·자동 복귀|실행 커밋·정책 SHA·모형 SHA·원본·이벤트 prefix 대조. 직전 검증 소스를 새 커밋으로 복귀하고 live 응답 확인|
|기존 원장 보호|DB에서 원본 신호 변경, 이벤트 변경·삭제 및 원장 TRUNCATE 거부. 현재 상태 갱신과 정상적인 사건 추가는 허용|
|기존 UX|BTC/ETH 2열 표, 네 방향 검토, 최초 터치 추적, TP1~3·STOP, 기존 원본·사건 유지|

## 아직 완성됐다고 말할 수 없는 것

실제 독립 성과시험 자료를 자동으로 완성하고 새 모형을 정식 승격한 전체 무인 연구 사이클은 아직 관측되지 않았다. 현재 자동 작업은 보존 관측을 수집하고 연구 판단·반례·후행 경로를 기록한다. 성과 runner는 여전히 실제 동결 연구 파일, V1.9 기준선, 비용·순서 자료, 사전 측정 기간·표본 기준·손실 한도와 증거 파일을 요구한다. 현재 이 독립 연구 파일은 없다. 자동으로 모두 조립·검증·승격까지 끝났다고 보고하지 않는다.

수정 언어는 지원되는 수학적 조건 그래프다. 모든 종류의 새 통계 알고리즘 소스를 스스로 만들고 검증하는 범용 연구자는 아니다. 장기 원시 호가·청산 자료가 새로 확보됐다고 주장하지 않는다.

## 실제 시장 검증이 필요한 것

- 방향·진입·손절·목표의 정확도가 실제로 개선됐는지.
- 국면별 순기대손익·낙폭·오진입·기회 상실·스위칭 오류가 개선됐는지.
- 신규 모형의 충분한 미래 shadow/canary 표본과 실비용 후 성과.

현재 승격 연구 모형은 0개다. 새 정식 신호를 시험 목적으로 발행하지 않았다. 실제 V2 신규 신호 3~5건의 전후 성과 비교는 제공할 표본이 없으므로 만들지 않았다. 승률 60/65%는 실측 결과가 아니다.

## 이번에 실제 실행한 기술 검증

- Node 단위·회귀·통합 성격의 코드 시험 289/289, 실패0·skip0.
- 실제 PostgreSQL16 별도 시험 1/1. DDL, 원래 행 보존, 정책별 저장, 마이크로초 페이지 순서, 연구 재생 방지, 원본 변경·이벤트 삭제·원장 초기화 거부와 정상 상태 갱신 검사.
- 실제 연구 실행에서 시장 특징 123개와 후보6개를 조회. 해당 실행의 연구 DB에는 관측122개·4방향 연구 검토488개를 저장했다. 이것은 독립 거래 수가 아니다. 순성과 검증 false, 승격 차단 true다.
- 실제 서버와 플러그인의 전체 V2.0 전문을 다시 읽어 바이트 일치를 확인했다.
- 최초 사용자 V1.9 파일을 실제로 내려받아 감사 원문과 바이트를 비교했다. 96,602바이트·1,263줄 및 지정 SHA 일치.
- 배포 후 오류 로그 조회 구간에서 error 로그가 없었다. 브라우저 화면과 실계좌 체결 시험은 수행하지 않았다.

## 실제 자동 복귀 시험

분석 코드와 신호를 바꾸지 않는 표식 파일만 추가했다. 감시 장치가 `CONTROLLED_ROLLBACK_REHEARSAL`로 이를 거부하고, GitHub에서 자동 복귀 커밋을 작성했다. Render가 실제 배포한 뒤 정책·모형·원장 보존까지 다시 확인했다.

|증거|실측|
|---|---|
|시험용 커밋|a99eceefe21aee238f92fc6c8608b43235449acf|
|복귀한 검증 소스|2dd178dfa41ca80963b1e19d83faa22ada1cc6bc|
|자동 작성된 복귀 커밋·현재 운영 커밋|c4bc05936767b6c5333187d2df4e534d6c4fd96b|
|현재 live deploy|dep-dbu39jlbedkc73evq500|
|자동 복귀 결과|ROLLED_BACK, verified=true|
|실제 원본·이벤트 보존|27신호·58이벤트, 각각 동일|

시험 표식이 제거된 뒤에도 새 기능이 포함된 검증 코드가 운영 중이다. 한 번의 통제된 복귀 시험이 모든 장애 상황의 복구를 증명하는 것은 아니다. DB나 원장을 과거 시점으로 되돌리지 않았다.

## 버전·해시

- 플러그인: 2.0.1, 릴리스 pluginrel_6afc333bfc508191b0d69f57436a4bb6.
- 엔진: V20_MATHEMATICAL_HYPOTHESIS_ENGINE_1.1.0.
- 전체 V2.0 정책은 이번 보완에서 그대로: 118,535바이트·1,492줄, SHA256 a6b389650a36e5140bab2929824d13b2c60fe0ec74648a1989e94fe095943265.
- 최초 V1.9 SHA256: 03910e1c418de6a83ed8191ec59801481c933fd1358b33b6b3071a8d81d8505a.
- 공식 원장 전후 SHA256: 8bbcfe954e7f04326d30114afd4692c691cf5addb4e7c22f14062405a92d67cf.
- 원본 신호 배열과 이벤트 배열도 각각 깊은 비교로 동일하다. 원장 해시만으로 이벤트 보존을 추정하지 않았다.
- 자동 거래 주문은 false. 새 수익성 모형 승격은 0개.

## 파일과 실제 실행 링크

- [전체 코드](https://github.com/leejongwon23/crypto-ai-bot/tree/c4bc05936767b6c5333187d2df4e534d6c4fd96b)
- [최초 V1.9 감사 전문](https://github.com/leejongwon23/crypto-ai-bot/blob/c4bc05936767b6c5333187d2df4e534d6c4fd96b/v17-common/MASTER_PROMPT_V1.9_ORIGINAL.txt)
- [새 V2.0 전체 전문](https://github.com/leejongwon23/crypto-ai-bot/blob/c4bc05936767b6c5333187d2df4e534d6c4fd96b/v17-common/MASTER_PROMPT_V2.0_FINAL.txt)
- [이번 연구·배포 연결 diff](https://github.com/leejongwon23/crypto-ai-bot/pull/26/files)
- [원장·배포 기준 보강 diff](https://github.com/leejongwon23/crypto-ai-bot/pull/27/files)
- [실제 자동 연구 실행](https://github.com/leejongwon23/crypto-ai-bot/actions/runs/44069697561)
- [실제 배포 검증과 자동 복귀 실행](https://github.com/leejongwon23/crypto-ai-bot/actions/runs/44070595118)
- [운영 health](https://btc-eth-v17-canonical-api.onrender.com/health)
- [플러그인](https://chatgpt.com/plugins/plugin_asdk_app_6af55fe9f77c8191b938f4418ebc3913)

이 폴더의 release-proof.json, deployment-guard.json, live-health.json, ledger-integrity.json은 실행 증거다. unit.tap/integration.tap은 실제 CI 결과이며, file-hashes.json은 수정 전후 파일 해시다. regime-performance.csv의 성과 빈칸은 미측정이며 0%가 아니다.

- [릴리스 증거](release-proof.json)
- [실제 자동 복귀 기록](deployment-guard.json)
- [운영 서버 응답](live-health.json)
- [원장 불변 대조](ledger-integrity.json)
- [실제 단위·회귀 시험](unit.tap) / [실제 DB 시험](integration.tap)
- [수정 전후 파일 해시](file-hashes.json)
- [국면별 성과 미검증 표](regime-performance.csv)
- [시장 특징 수집 결과](collection.json) / [연구 운영 기록](shadow-cycle.json) / [승격 차단 결과](promotion.json)

## 장애 복귀 절차

감시 장치가 마지막 검증 참조와 실행 소스를 대조한다. 실패하면 동시 변경 여부를 검사한 뒤 마지막 검증 소스로 새 커밋을 만든다. 소모 holdout 이력은 유지하고 DB는 변경하지 않는다. 복귀 서버의 실제 커밋·정책·모형·원본·사건 보존 확인에 실패하면 ROLLBACK_UNVERIFIED로 남긴다. 검증되지 않은 복귀를 성공이라고 보고하지 않는다.
