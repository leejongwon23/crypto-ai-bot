# BTC·ETH shared canonical engine — V1.7 companion

This is **not** a claim of literal equivalence to all 901 lines of V1.7. Human-language discretion has no machine-unique new entry decision. This deterministic, audited companion is a reproducible alternative. No backtested profitability or exchange orders are implied.

## Deployment

- `npm install --prefix v17-common`
- `node v17-common/server.mjs`
- Required cloud env vars: `DEPLOYMENT_MODE=cloud`, `SIGNAL_API_KEY` (high entropy), `DATABASE_URL` (durable Postgres), `PORT` (set by Render).
- Exposes `/health`, `/` paste-form, `/api/run` and `/api/signal` protected by bearer token; `/mcp` tool-discovery publicly accessible, tool calls auth-protected.
- Never upload proprietary user V1.7 original, snapshots, account identifiers, or server credentials to this public repository.

## ChatGPT availability

Plan-specific OpenAI permissions may prevent installing arbitrary remote MCP servers. Backend deployment alone does not connect the service to ordinary chats. MCP tool selection is per-message when supported. No trading or exchange account is connected.

## Source provenance

Canonical unmodified original prompt SHA256: `dd53a208898124cb815433caca4348b95347d2bd06a49f7cda4b6cc57bcf8e52`.

## Consistency boundary

Exact snapshot + same state + same engine version -> deterministic decision. Across-chat consistency does not prove profit, nor does it establish absolute semantic equivalence to discretionary V1.7. CLEAN START by default `OFFICIAL_V17`; distinct sessions do not merge.