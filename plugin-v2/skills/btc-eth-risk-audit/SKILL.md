---
name: btc-eth-risk-audit
description: "Analyze BTC/ETH snapshots with the full V2.0 mathematical market-structure, flow-impact hypothesis, pre-stop error review and evidence-gated self-revision policy; preserve the official ledger."
---

# BTC·ETH mathematical hypothesis engine — V2.0

## Mandatory complete policy

Read the ENTIRE `references/MASTER_PROMPT_V2.0_FINAL.txt` before market analysis. This is the active amended full policy, including the preserved original operational safeguards, the section-specific mathematical amendments in 0–31, and new sections 32–43. Do not substitute the old V1.9 prompt or an operational bug-fix addendum. The original V1.9 source remains `references/MASTER_PROMPT_V1.9_FINAL.txt` for audit only.

Current policy name must be `MASTER PROMPT V2.0 FINAL`. Verify the exact policy SHA in `references/ENGINE_V2_MANIFEST.json` against the live server before publishing a new V2.0 result. Never reset `OFFICIAL_V17` because of an upgrade. Existing originals, prices, versions, issuance times, events and completed outcomes remain unchanged.

## What changed in the analysis

1. Use the server's actual `quantContext` computed from the complete raw text. It contains source/closed-bar/volume/trade/gap status; confirmed pivots with knowledge timestamps; structural transitions; empirical returns/variation/lag correlation/quantiles/distribution differences; same-window aggressor-flow/price-response hypotheses; independent LONG/SHORT A–E arithmetic candidates; and per-ID pre-stop error reviews. Values not calculable from provided raw data remain explicitly unavailable.
2. Analyze market mechanisms, not indicator votes. A negative CVD while price rises is a competing seller-absorption hypothesis, with window mismatch, replenishment and other-venue flows as alternatives. It is not a SHORT instruction or proof of a whale's intention. Never identify an unseen participant or claim hidden orders. OFI/order cancellation/liquidation models require the corresponding real data.
3. Independently evaluate BTC LONG, BTC SHORT, ETH LONG, ETH SHORT. In every direction compare A immediate current price, B nearby structure, C distant retest/breakout, D original existing pending, E no new issue. Explain structural entry, invalidation stop, actual TP1–3, worst-entry percentages/RR, noise, failure, execution costs and contrary evidence. The arithmetic explorer does not itself qualify a formal signal or prove future retest.
4. Analyze higher and execution regimes separately: trend up/down, strong expansion, staircase, range, compression, repeated false breaks, suspected accumulation/distribution, transition, mixed/unidentified. Do not pretend that all are mathematically identified from a few EMA summaries. Measure what is identifiable and distinguish alternatives.
5. Evaluate model errors before a STOP: structural invalidation, conditional price-impact reversal, forecast interval breach when a frozen interval exists, distribution/regime change and coupling change. A losing mark alone is not a model falsification. Compare maintain/protect/early-exit/switch/redesign/add independently for each original active ID.
6. Generate competing mechanism revisions in `researchProposals`; modify causal/structural assumptions and testable mathematical relationships, never weights, weighted scores, confidence points, reward/punishment or cosmetic parameter fitting. A single STOP or lucky win never promotes a model.

## Immutable official trading semantics

- Existing pending first actual postissue touch becomes analytical ENTERED with no added flow/candle condition. It is not an account fill claim.
- Before touch cancellation/redesign needs actual pre-entry invalidation; a new entry plan gets a new ID. Entered trades are not retroactively cancelled.
- STOP/TP1/TP2/TP3 use the actual chronological source path. Ambiguous intraminute ordering and missing intervals remain unverified. Future revised stops/targets never rewrite earlier observations.
- Opposite entry requires every relevant existing opposing entered tranche to have a separately verified exit first. Same-direction add requires a new independently valid ID and exposure disclosure.
- Quantities/margin/leverage are required for combined account/liquidation risk; otherwise mark unverified. No exchange orders or account state is invented.
- Do not move targets outward or shrink STOP merely to satisfy TP2 RR>=2. Respect the full original risk/exception policy and explain any policy-permitted exception explicitly; no invented anchor.
- Exclude HEATMAP PROXY/its percentages and the user's market view as independent evidence. CVD and taker can overlap, so do not count them twice.

## Official tool workflow and contract

1. For each new complete BTC+ETH raw snapshot call `run_official_btc_eth_snapshot` with EXACT identical raw and session `OFFICIAL_V17`. Use each source KST timestamp, not current chat time. The response supplies full active policy, official original ledger, evidence quality, numerical alternatives, quant context and mechanism proposals.
2. If `OFFICIAL_REPLAY`, display the stored official view, never issue again. A saved old-version analysis is historical evidence, not proof of a new V2.0 decision.
3. Actually perform all 16 checks and all 47 clauses: 0–43, APPENDIX_A, APPENDIX_B, FINAL_EXECUTION, plus the four independent directions/A–E and all current IDs. Mark missing evidence UNVERIFIED; do not copy generic evidence as if verified.
4. Submit `analysis.quantAudit` with exact `contextSha256`. BTC/ETH each require exact `asOf`, `regimeMechanism`, `flowImpactMechanism`, `noiseRisk`, `invalidationRisk`, `executionCostRisk`, `preStopErrorReview`, `competingHypotheses`, `modelRevisionDecision`, `sourceLimitations`, `intentEvidenceLevel` (OBSERVED/HYPOTHESIS/UNVERIFIED), `participantIdentityKnown:false`.
5. Each symbol's quantAudit `directions.LONG` and `.SHORT` require `mathematicalBasis`, `counterexample`, `entryStopTargetBasis`, `costAndCapacityLimitations`. `researchProposals` is an array; each item requires `mechanismChange`, `support`, `refutationTest`, `alternativeExplanation`, `requiredData`, `status`. Use [] when no grounded proposal exists.
6. Preserve all existing structured fields: reportMarkdown; directions BTC_LONG/BTC_SHORT/ETH_LONG/ETH_SHORT with A–E; coupling; checks; clauses; audit20; conclusions; regimeAudit; dynamicManagement; marketOutlook; newSignals; events; and actual required historicalUncertainty/autoSwitchAudit/reversalReview. Regime and management asOf must exactly match each symbol's raw timestamp.
7. A NEW V2.0 formal signal uses `issuedVersion:'V2.0'`, `frozenSourceSha256` equal to the exact source snapshot SHA, optional `modelId` only when that registered model was actually used, existing complete identity/entry/STOP/TP/evidence fields, and `modelHypothesis`, `modelVersion`, `structuralInvalidation`, `predictionIntervalEvidence` (state precise insufficiency if no defensible interval). Preserve old V1.7/V1.8/V1.9 issuedVersion originals.
8. Publish through available `publish_official_v20_analysis`, or the existing cached adapter `register_formal_btc_eth_signal` with `signal.kind:'V17_OFFICIAL_SUBMISSION'` and the exact policy/snapshot/ledger hashes, decisionKey and full analysis. The legacy spelling is an adapter, not the governing policy version. Source raw may be recovered from the exact prepared SHA under the existing contract.
9. Publication is complete only after OFFICIAL_STORED/OFFICIAL_REPLAY, followed by official read-back. Never narrate an unpublished plan as formal or validation failure as successful HOLD/switch.

## Autonomous research and deployment truthfulness

The runtime generates explicit supported-family mechanism proposals from observed new data. The repository contains an evidence-gated study/promotion runner and a workflow that can commit validated future model-registry changes, with Render deploying the linked production branch.

Distinguish these capability states: proposal generated; frozen real study available; independent validation run; shadow/rollback proof verified; promotion eligible; registry changed; actual deploy live. Never infer the later states from the earlier ones. If the real frozen study/data feed/protocol/proofs are absent, promotion remains blocked. The initial registry has no promoted research model; do not pretend it has learned a profitable model already.

Lock datasets and test families before evaluation; purge overlapping feature/label spans, embargo, use dependence-aware intervals and multiple-test correction, compare same-cost baseline/opposite/abstain and regime/asset strata. Consume a holdout after opening even when the candidate fails. Reject worse drawdown, missing costs/source/path/order, skipped regression or unverified shadow/rollback. A stable full prompt need not be reissued for every future supported symbolic-model registry version; keep code/model/policy hashes auditable.

No observed number in this system is a weighted trade score. No automatic exchange orders are enabled. Do not promise all-market success, exact whale intention, high win rate, full autonomous research connectivity, or market-performance improvement without actual evidence.

## Compact output preserved

Display the entire official `signalView.markdown`: fixed `항목 | BTC | ETH`, source time/price, formal direction/ID, entry, STOP and percentage, TP1–3 and percentages, TP2 RR, state, management and add decision. A coin without a formal signal says `진입 시그널 없음`.

Below it keep separate BTC then ETH observed situation and conditional expected path plus opposite/invalidation scenario. Do not add long internal audit dumps unless asked. Do not reconstruct an unstored historical forecast. Append genuine state changes and only essential uncertainty. A star, if used, identifies at most one currently superior qualifying entry and is not a win probability.
