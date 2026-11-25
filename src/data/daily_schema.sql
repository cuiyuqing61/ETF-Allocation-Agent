CREATE TABLE IF NOT EXISTS daily_sessions (
 run_id TEXT PRIMARY KEY REFERENCES experiment_runs(run_id), config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_state (
 run_id TEXT PRIMARY KEY REFERENCES daily_sessions(run_id), version INTEGER NOT NULL,
 state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_snapshots (
 run_id TEXT NOT NULL REFERENCES daily_sessions(run_id), version INTEGER NOT NULL,
 state_json TEXT NOT NULL, processed_at TEXT NOT NULL,
 PRIMARY KEY(run_id,version)
);
CREATE TABLE IF NOT EXISTS daily_inputs (
 decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id), account_version INTEGER NOT NULL,
 snapshot_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_attempts (
 attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES daily_sessions(run_id),
 operation TEXT NOT NULL, logical_date TEXT NOT NULL, started_at TEXT NOT NULL,
 finished_at TEXT, status TEXT NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS daily_attempt_lookup ON daily_attempts(run_id,operation,logical_date);

-- A crashed worker cannot mark itself failed. Infer recovery from committed facts,
-- while retaining the original started/failed attempt for diagnosis.
CREATE VIEW IF NOT EXISTS daily_attempt_diagnostics AS
SELECT a.*, CASE
 WHEN a.status IN ('started','failed') AND (
  (a.operation='decide' AND EXISTS(SELECT 1 FROM decisions d WHERE d.run_id=a.run_id AND d.decision_date=a.logical_date)) OR
  (a.operation='settle' AND EXISTS(SELECT 1 FROM executions e WHERE e.run_id=a.run_id AND e.execution_date=a.logical_date))
 ) THEN a.status || '_result_available'
 WHEN a.status='started' THEN 'unfinished_retryable'
 ELSE a.status END AS recovery_status
FROM daily_attempts a;

CREATE VIEW IF NOT EXISTS daily_decision_audit AS
SELECT d.*, r.model_id, r.dataset_id, i.account_version, i.snapshot_json,
 e.execution_id,e.execution_date,e.status AS execution_status,e.reason AS execution_reason,
 e.signed_quantity,e.fill_price,e.fee_cash,e.slippage_cash,
 a.cash,a.quantity,a.dividend_receivable,a.dividend_cash,a.equity,a.net_return,
 s.processed_at AS settled_at
FROM decisions d
JOIN daily_inputs i USING(decision_id)
JOIN experiment_runs r ON r.run_id=d.run_id
LEFT JOIN executions e ON e.decision_id=d.decision_id
LEFT JOIN daily_account a ON a.run_id=d.run_id AND a.trade_date=e.execution_date
LEFT JOIN daily_snapshots s ON s.run_id=d.run_id AND s.version=i.account_version+1;
