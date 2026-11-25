CREATE TABLE IF NOT EXISTS position_schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS datasets (
 dataset_id TEXT PRIMARY KEY, source TEXT NOT NULL, fetched_at TEXT NOT NULL,
 raw_hash TEXT NOT NULL, normalized_hash TEXT NOT NULL, raw_path TEXT NOT NULL,
 instrument TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
 adjustment_policy TEXT NOT NULL, corporate_actions_version TEXT NOT NULL,
 metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS market_bars (
 dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id), instrument TEXT NOT NULL,
 trade_date TEXT NOT NULL, raw_open REAL NOT NULL CHECK(raw_open>0),
 raw_high REAL NOT NULL CHECK(raw_high>0), raw_low REAL NOT NULL CHECK(raw_low>0),
 raw_close REAL NOT NULL CHECK(raw_close>0), volume REAL NOT NULL CHECK(volume>=0),
 tradable INTEGER NOT NULL CHECK(tradable IN (0,1)), availability_at TEXT NOT NULL,
 PRIMARY KEY(dataset_id,instrument,trade_date)
);
CREATE TABLE IF NOT EXISTS corporate_actions (
 dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id), action_id TEXT NOT NULL,
 instrument TEXT NOT NULL, ex_date TEXT NOT NULL, record_date TEXT NOT NULL,
 pay_date TEXT NOT NULL, cash_per_unit REAL NOT NULL CHECK(cash_per_unit>=0),
 split_ratio REAL NOT NULL CHECK(split_ratio>0), source TEXT NOT NULL,
 PRIMARY KEY(dataset_id,action_id)
);
CREATE TABLE IF NOT EXISTS models (
 model_id TEXT PRIMARY KEY, algorithm TEXT NOT NULL, train_start TEXT NOT NULL,
 train_end TEXT NOT NULL, feature_version TEXT NOT NULL, scaler_path TEXT,
 model_path TEXT, artifact_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_runs (
 run_id TEXT PRIMARY KEY, strategy_version TEXT NOT NULL, model_id TEXT REFERENCES models(model_id),
 dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id), seed INTEGER,
 split TEXT NOT NULL CHECK(split IN ('train','validation','test','forward')),
 evaluation_mode TEXT NOT NULL CHECK(evaluation_mode IN ('historical','forward_paper')),
 period_start TEXT NOT NULL, period_end TEXT NOT NULL,
 initial_capital REAL NOT NULL CHECK(initial_capital>0), config_json TEXT NOT NULL,
 code_commit TEXT NOT NULL, status TEXT NOT NULL, payload_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
 account_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE REFERENCES experiment_runs(run_id),
 instrument TEXT NOT NULL, cash REAL NOT NULL CHECK(cash>=-0.000001),
 quantity INTEGER NOT NULL CHECK(quantity>=0), last_valuation_date TEXT,
 dividend_receivable REAL NOT NULL DEFAULT 0 CHECK(dividend_receivable>=0),
 account_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS decisions (
 decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
 decision_date TEXT NOT NULL, created_at TEXT NOT NULL, data_as_of TEXT NOT NULL,
 snapshot_hash TEXT NOT NULL, current_weight REAL NOT NULL,
 raw_target_weight REAL CHECK(raw_target_weight BETWEEN 0 AND 1),
 target_weight REAL CHECK(target_weight BETWEEN 0 AND 1),
 planned_execution_date TEXT NOT NULL, guard_reason TEXT NOT NULL,
 UNIQUE(run_id,decision_date), UNIQUE(run_id,decision_id),
 CHECK(planned_execution_date>decision_date)
);
CREATE TABLE IF NOT EXISTS executions (
 execution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
 decision_id TEXT NOT NULL, execution_date TEXT NOT NULL, status TEXT NOT NULL,
 reference_open REAL NOT NULL, fill_price REAL, signed_quantity INTEGER NOT NULL,
 notional REAL NOT NULL, fee_cash REAL NOT NULL CHECK(fee_cash>=0),
 slippage_cash REAL NOT NULL CHECK(slippage_cash>=0), reason TEXT NOT NULL,
 UNIQUE(run_id,decision_id),
 FOREIGN KEY(run_id,decision_id) REFERENCES decisions(run_id,decision_id)
);
CREATE TABLE IF NOT EXISTS daily_account (
 run_id TEXT NOT NULL REFERENCES experiment_runs(run_id), trade_date TEXT NOT NULL,
 prev_equity REAL NOT NULL CHECK(prev_equity>0), equity REAL NOT NULL CHECK(equity>0),
 cash REAL NOT NULL CHECK(cash>=-0.000001), quantity INTEGER NOT NULL CHECK(quantity>=0),
 position_value REAL NOT NULL, net_return REAL NOT NULL, traded_notional REAL NOT NULL,
 fee_cash REAL NOT NULL, slippage_cash REAL NOT NULL, dividend_cash REAL NOT NULL DEFAULT 0,
 dividend_receivable REAL NOT NULL DEFAULT 0 CHECK(dividend_receivable>=0),
 dividend_accrued REAL NOT NULL DEFAULT 0, prev_cash REAL, prev_quantity INTEGER,
 prev_receivable REAL, trade_cash REAL, signed_quantity INTEGER, quantity_change INTEGER,
 close_price REAL, reference_price REAL, is_session INTEGER NOT NULL DEFAULT 1,
 exposure REAL NOT NULL, market_regime TEXT NOT NULL DEFAULT 'unclassified',
 PRIMARY KEY(run_id,trade_date),
 CHECK(abs(equity-cash-position_value-dividend_receivable)<0.000001)
);
CREATE TABLE IF NOT EXISTS corporate_action_sets (
 version TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
 manifest_json TEXT NOT NULL, manifest_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corporate_action_events (
 run_id TEXT NOT NULL REFERENCES experiment_runs(run_id), action_id TEXT NOT NULL,
 event_date TEXT NOT NULL, phase TEXT NOT NULL CHECK(phase IN ('record','ex','pay')),
 entitled_quantity INTEGER NOT NULL CHECK(entitled_quantity>=0), amount REAL NOT NULL CHECK(amount>=0),
 PRIMARY KEY(run_id,action_id,phase)
);
