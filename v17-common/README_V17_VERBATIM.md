# BTC·ETH MASTER PROMPT V1.7 FINAL — Original-preserving canonical decision registry

## Design constraint
- `policy-v17.mjs` encodes the **exact 52,922 original bytes** of the user's original Korean V1.7 source, including CRLF and the initial empty line. Its SHA-256 is `5fbd801f1378df7f584b248aadfaf98ae7e38d0bca782e3e09625bafe39e66ad`.
- The server MUST NOT calculate a new trade direction using an invented EMA / CVD / VWAP threshold model. `engine.mjs` is legacy and **unused** by the new `server.mjs`. No arbitrary new price-generation logic replaces V1.7.
- On first use of a new snapshot/ledger state, the server responds with the **complete original policy** + official ledger and requires the connected ChatGPT model to apply all source sections. This is AI-in-the-loop, not server-hosted autonomous LLM inference.
- On subsequent uses of the **exact same raw snapshot and same official ledger state**, the server returns the persisted first official report. Cache keys include SHA-256 of snapshot, ledger and original policy, and the official session name.
- The first submitted analysis is checked for four directions, all five alternatives, clauses 0–23, both appendices and the final instruction, the full 16-step execution, final rule 20 audit and formal signal geometry; checking structural completeness is **not proof of semantic correctness**.
- Formal signals are stored separately and immutably, with append-only event history and versioned plan changes. Legacy data remains in the database untouched and is **not** automatically migrated into the new official ledger.
- `OFFICIAL_V17` is one shared canonical operational session. An explicitly different session name must start with an independent empty ledger; this is the user's CLEAN START rule.
- No independent performance/success-rate guarantees; no exchange orders; not live background price polling.

## MCP tool workflow
1. `run_official_btc_eth_snapshot` `{raw:"both complete snapshots",session:"OFFICIAL_V17"}`
2. If `OFFICIAL_REPLAY`, display **only the returned official report** (do not reanalyze).
3. If `NEEDS_AI_ANALYSIS`, read the complete unchanged source returned as `policy.text` and apply all requirements. This **must not be presented as an official signal yet**.
4. Call `publish_official_v17_analysis` with raw, hashes, decisionKey, and structured evidence/decision. Display only the persisted `officialDecision.reportMarkdown` returned by that call.
5. `read_official_btc_eth_ledger` reads the authoritative original signal and event log.

## Verification
- `node --test v17-core.test.mjs` (source byte preservation, no unsupported weights, 100-repeat stable key, 4-way inspection, A–E alternatives, 27 clause sections, state and geometry checks).
- Deployment requires existing Render PostgreSQL and OAuth configuration from the previous V1.7 server; no new billable resource required.
- **Critical limitation**: a chatbot reading the entire source is not by itself a mathematically deterministic first-pass analyst. What is made identical across conversations is the first **officially stored result** for the same snapshot and official ledger state. A human or independent evaluator must still validate accuracy and V1.7 semantic fidelity.
- Free Render Postgres instance expires 2026-12-12 unless migrated; long-term persistence is not guaranteed on a free expiring plan.
