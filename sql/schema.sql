-- ============================================================
-- Health Insurance AI Platform – PostgreSQL Schema
-- Lumen Technologies
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ─── Members ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS members (
    member_id               VARCHAR(20)    PRIMARY KEY,
    age                     SMALLINT       NOT NULL CHECK (age BETWEEN 18 AND 100),
    gender                  VARCHAR(10),
    state                   VARCHAR(5),
    bmi                     DECIMAL(5,2)   CHECK (bmi > 0),
    smoker                  BOOLEAN        DEFAULT FALSE,
    plan_type               VARCHAR(20)    CHECK (plan_type IN ('Bronze','Silver','Gold','Platinum')),
    tenure_months           SMALLINT       DEFAULT 0,
    num_chronic_conditions  SMALLINT       DEFAULT 0,
    primary_diagnosis       VARCHAR(10),
    annual_premium          DECIMAL(10,2),
    deductible              DECIMAL(10,2),
    member_since            DATE,
    created_at              TIMESTAMPTZ    DEFAULT NOW(),
    updated_at              TIMESTAMPTZ    DEFAULT NOW()
);

-- ─── Providers ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS providers (
    provider_id             VARCHAR(20)    PRIMARY KEY,
    provider_name           VARCHAR(200),
    provider_type           VARCHAR(50),
    state                   VARCHAR(5),
    avg_claim_amount        DECIMAL(12,2),
    total_claims            INTEGER        DEFAULT 0,
    fraud_rate              DECIMAL(6,4)   DEFAULT 0,
    accreditation           VARCHAR(50),
    years_active            SMALLINT,
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

-- ─── Claims ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims (
    claim_id                VARCHAR(20)    PRIMARY KEY,
    member_id               VARCHAR(20)    REFERENCES members(member_id),
    provider_id             VARCHAR(20)    REFERENCES providers(provider_id),
    claim_date              DATE           NOT NULL,
    claim_type              VARCHAR(30)    CHECK (claim_type IN
                                ('Inpatient','Outpatient','Emergency','Pharmacy','Lab')),
    diagnosis_code          VARCHAR(10),
    procedure_code          VARCHAR(10),
    claim_amount            DECIMAL(12,2)  CHECK (claim_amount > 0),
    approved_amount         DECIMAL(12,2)  DEFAULT 0,
    claim_status            VARCHAR(20)    CHECK (claim_status IN ('Approved','Denied','Pending')),
    days_in_hospital        SMALLINT       DEFAULT 0,
    num_procedures          SMALLINT       DEFAULT 1,
    prior_auth              BOOLEAN        DEFAULT FALSE,
    is_fraud                BOOLEAN        DEFAULT FALSE,
    processed_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

-- ─── ML Predictions ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS claim_predictions (
    id                      SERIAL         PRIMARY KEY,
    claim_id                VARCHAR(20),
    member_id               VARCHAR(20),
    model_version           VARCHAR(50),
    approval_probability    DECIMAL(6,4),
    risk_score              DECIMAL(6,4),
    risk_band               VARCHAR(20),
    decision                VARCHAR(20),
    processing_time_ms      DECIMAL(8,2),
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fraud_predictions (
    id                      SERIAL         PRIMARY KEY,
    claim_id                VARCHAR(20),
    model_version           VARCHAR(50),
    anomaly_score           DECIMAL(6,4),
    fraud_flag              BOOLEAN,
    combined_fraud_score    DECIMAL(6,4),
    alert_level             VARCHAR(10),
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS plan_recommendations (
    id                      SERIAL         PRIMARY KEY,
    member_id               VARCHAR(20),
    recommended_plan        VARCHAR(20),
    recommendation_score    DECIMAL(6,4),
    model_version           VARCHAR(50),
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

-- ─── Clinical Notes ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id                 VARCHAR(20)    PRIMARY KEY,
    member_id               VARCHAR(20)    REFERENCES members(member_id),
    note_date               DATE,
    clinical_note           TEXT,
    diagnosis_code          VARCHAR(10),
    diagnosis_label         VARCHAR(100),
    predicted_diagnosis     VARCHAR(20),
    nlp_confidence          DECIMAL(5,4),
    sentiment_label         VARCHAR(20),
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

-- ─── Model Registry ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id                      SERIAL         PRIMARY KEY,
    model_name              VARCHAR(100)   NOT NULL,
    model_type              VARCHAR(50),
    version                 VARCHAR(20),
    auc_roc                 DECIMAL(6,4),
    precision_score         DECIMAL(6,4),
    recall_score            DECIMAL(6,4),
    f1_score                DECIMAL(6,4),
    training_rows           INTEGER,
    artifact_path           TEXT,
    is_active               BOOLEAN        DEFAULT FALSE,
    deployed_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ    DEFAULT NOW()
);

-- ─── Indexes ──────────────────────────────────────────────
CREATE INDEX idx_claims_member      ON claims(member_id);
CREATE INDEX idx_claims_provider    ON claims(provider_id);
CREATE INDEX idx_claims_date        ON claims(claim_date);
CREATE INDEX idx_claims_status      ON claims(claim_status);
CREATE INDEX idx_claims_fraud       ON claims(is_fraud);
CREATE INDEX idx_preds_claim        ON claim_predictions(claim_id);
CREATE INDEX idx_fraud_pred_claim   ON fraud_predictions(claim_id);
CREATE INDEX idx_notes_member       ON clinical_notes(member_id);

-- ─── Useful Analytics Views ───────────────────────────────
CREATE OR REPLACE VIEW v_claim_summary AS
SELECT
    DATE_TRUNC('month', claim_date)    AS month,
    claim_type,
    claim_status,
    COUNT(*)                           AS num_claims,
    SUM(claim_amount)                  AS total_claimed,
    SUM(approved_amount)               AS total_approved,
    AVG(claim_amount)                  AS avg_claim,
    SUM(is_fraud::INT)                 AS fraud_count
FROM claims
GROUP BY 1, 2, 3;

CREATE OR REPLACE VIEW v_provider_fraud_risk AS
SELECT
    p.provider_id,
    p.provider_name,
    p.provider_type,
    p.state,
    COUNT(c.claim_id)                   AS total_claims,
    AVG(c.claim_amount)                 AS avg_claim_amount,
    SUM(c.is_fraud::INT)                AS fraud_claims,
    ROUND(SUM(c.is_fraud::INT) * 100.0 / NULLIF(COUNT(*), 0), 2) AS fraud_rate_pct
FROM providers p
LEFT JOIN claims c USING (provider_id)
GROUP BY 1, 2, 3, 4;

CREATE OR REPLACE VIEW v_member_risk_profile AS
SELECT
    m.member_id,
    m.age,
    m.plan_type,
    m.num_chronic_conditions,
    COUNT(c.claim_id)                   AS total_claims,
    SUM(c.claim_amount)                 AS lifetime_spend,
    AVG(cp.risk_score)                  AS avg_risk_score,
    AVG(fp.anomaly_score)               AS avg_anomaly_score
FROM members m
LEFT JOIN claims c                  USING (member_id)
LEFT JOIN claim_predictions cp      ON c.claim_id = cp.claim_id
LEFT JOIN fraud_predictions fp      ON c.claim_id = fp.claim_id
GROUP BY 1, 2, 3, 4;
