-- DRAFT: only for a separate disposable research database. NOT applied to production.
BEGIN;
CREATE SCHEMA IF NOT EXISTS btc_eth_research_v2;
REVOKE ALL ON SCHEMA btc_eth_research_v2 FROM PUBLIC;
CREATE TABLE btc_eth_research_v2.evidence (
 sequence BIGSERIAL PRIMARY KEY,
 digest TEXT NOT NULL UNIQUE CHECK (digest ~ '^[a-f0-9]{64}$'),
 previous_digest TEXT,
 event_time TIMESTAMPTZ NOT NULL,
 known_at TIMESTAMPTZ NOT NULL,
 captured_at TIMESTAMPTZ NOT NULL,
 exchange TEXT NOT NULL CHECK (exchange='Bybit'),
 category TEXT NOT NULL CHECK (category='linear'),
 symbol TEXT NOT NULL CHECK (symbol IN ('BTCUSDT','ETHUSDT')),
 source_kind TEXT NOT NULL,
 source_uri TEXT NOT NULL,
 vintage_status TEXT NOT NULL CHECK (vintage_status IN ('LIVE_CAPTURE','RETROSPECTIVE','USER_SUBMITTED','UNVERIFIED')),
 payload JSONB NOT NULL
);
CREATE INDEX evidence_asof ON btc_eth_research_v2.evidence(symbol,event_time,known_at);
CREATE FUNCTION btc_eth_research_v2.deny_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'APPEND_ONLY_RESEARCH_EVIDENCE'; END $$;
CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON btc_eth_research_v2.evidence
 FOR EACH ROW EXECUTE FUNCTION btc_eth_research_v2.deny_mutation();
CREATE TRIGGER immutable_truncate BEFORE TRUNCATE ON btc_eth_research_v2.evidence
 FOR EACH STATEMENT EXECUTE FUNCTION btc_eth_research_v2.deny_mutation();
COMMIT;
-- Not a defense against privileged DDL/full-chain rewrite. Pin roots externally.
-- Official v17_ai_* tables, reset archive and original signals are deliberately absent.
