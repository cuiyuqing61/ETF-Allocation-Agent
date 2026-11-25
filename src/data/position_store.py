"""Versioned business records inside the existing FinRL-X SQLite service."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd


def migrate(store):
    with store.connect() as conn:
        schema=Path(__file__).with_name('position_schema.sql').read_text()
        columns=[r[1] for r in conn.execute('PRAGMA table_info(daily_account)')]
        if columns and 'dividend_receivable' not in columns:
            # Rebuild only the CHECK-constrained table, retaining every old value.
            # SQLite DDL/data copy are in one transaction; no writable_schema hacks.
            conn.execute('BEGIN IMMEDIATE')
            for operation in ('UPDATE','DELETE'):
                conn.execute(f'DROP TRIGGER IF EXISTS immutable_daily_account_{operation}')
            conn.execute('ALTER TABLE daily_account RENAME TO daily_account_v1')
            start=schema.index('CREATE TABLE IF NOT EXISTS daily_account (')
            end=schema.index(';',start)+1
            conn.execute(schema[start:end])
            names=','.join(columns)
            conn.execute(f'INSERT INTO daily_account ({names}) SELECT {names} FROM daily_account_v1')
            conn.execute('DROP TABLE daily_account_v1')
            conn.commit()
        conn.executescript(schema)
        conn.executescript(Path(__file__).with_name('daily_schema.sql').read_text())
        conn.execute('INSERT OR IGNORE INTO position_schema_migrations VALUES(3,?)', (datetime.now(timezone.utc).isoformat(),))
        if 'dividend_receivable' not in [r[1] for r in conn.execute('PRAGMA table_info(accounts)')]:
            conn.execute('ALTER TABLE accounts ADD COLUMN dividend_receivable REAL NOT NULL DEFAULT 0 CHECK(dividend_receivable>=0)')
        conn.execute('INSERT OR IGNORE INTO position_schema_migrations VALUES(2,?)',
                     (datetime.now(timezone.utc).isoformat(),))
        conn.execute('INSERT OR IGNORE INTO position_schema_migrations VALUES(1,?)',
                     (datetime.now(timezone.utc).isoformat(),))
        for table in ('datasets','market_bars','corporate_actions','models','experiment_runs','decisions','executions','daily_account','corporate_action_sets','corporate_action_events','daily_sessions','daily_snapshots','daily_inputs'):
            for operation in ('UPDATE','DELETE'):
                conn.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable record'); END")


def save_dataset(store, metadata, bars):
    from src.data.akshare_provider import validate_bars
    b, _ = validate_bars(bars, metadata['start_date'], metadata['end_date'])
    b = b[['trade_date', 'open', 'high', 'low', 'close', 'volume', 'tradable', 'availability_at']]
    raw_path = Path(metadata['raw_path'])
    if sha256(raw_path.read_bytes()).hexdigest() != metadata['raw_hash']:
        raise ValueError('Raw snapshot hash mismatch')
    normalized_hash = sha256(b.to_csv(index=False).encode()).hexdigest()
    m = {**metadata, 'normalized_hash': normalized_hash}
    frozen_csv=raw_path.parent/'bars.csv'
    payload=b.to_csv(index=False).encode()
    if frozen_csv.exists() and frozen_csv.read_bytes()!=payload:
        raise ValueError('Immutable normalized CSV conflict')
    if not frozen_csv.exists():
        with frozen_csv.open('xb') as stream: stream.write(payload)
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        old = conn.execute('SELECT normalized_hash,raw_hash FROM datasets WHERE dataset_id=?', (m['dataset_id'],)).fetchone()
        if old:
            if tuple(old) != (normalized_hash, m['raw_hash']):
                raise ValueError('Dataset id conflicts with immutable content')
            return m['dataset_id']
        conn.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                     tuple(m[k] for k in ('dataset_id','source','fetched_at','raw_hash','normalized_hash','raw_path','instrument','start_date','end_date','adjustment_policy','corporate_actions_version')) + (json.dumps(m, sort_keys=True),))
        conn.executemany('INSERT INTO market_bars VALUES(?,?,?,?,?,?,?,?,?,?)',
                         [(m['dataset_id'],m['instrument'],r.trade_date,r.open,r.high,r.low,r.close,r.volume,int(r.tradable),r.availability_at) for r in b.itertuples()])
    return m['dataset_id']


def load_dataset(store, dataset_id):
    with store.connect() as conn:
        row = conn.execute('SELECT metadata_json FROM datasets WHERE dataset_id=?', (dataset_id,)).fetchone()
        if not row:
            raise KeyError(dataset_id)
        m = json.loads(row[0])
        b = pd.read_sql_query('SELECT trade_date,raw_open AS open,raw_high AS high,raw_low AS low,raw_close AS close,volume,tradable,availability_at FROM market_bars WHERE dataset_id=? ORDER BY trade_date', conn, params=(dataset_id,))
    if sha256(Path(m['raw_path']).read_bytes()).hexdigest() != m['raw_hash']:
        raise ValueError('Raw snapshot corrupted')
    # SQLite serializes numeric values as REAL; canonicalize before checking.
    from src.data.akshare_provider import validate_bars
    b, _ = validate_bars(b, m['start_date'], m['end_date'])
    if sha256(b.to_csv(index=False).encode()).hexdigest() != m['normalized_hash']:
        raise ValueError('Normalized dataset corrupted')
    return m, b


def save_run(store, run, decisions, executions, daily, events=None):
    """One transaction: a retry returns the existing identical run, never another fill."""
    events=events or []
    run = dict(run)
    if daily and daily[0].get('prev_cash') is not None:
        from src.backtest.position_replay import reconcile
        reconcile(daily,run['initial_capital'])
    canonical_decisions = [{k: v for k, v in d.items() if k != 'created_at'} for d in decisions]
    payload_hash = sha256(json.dumps([run, canonical_decisions, executions, daily, events], sort_keys=True, allow_nan=False).encode()).hexdigest()
    run['payload_hash'] = payload_hash
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        old = conn.execute('SELECT config_json,payload_hash FROM experiment_runs WHERE run_id=?', (run['run_id'],)).fetchone()
        if old:
            if old[0] != run['config_json'] or old[1] != payload_hash:
                raise ValueError('Run id conflict')
            return False
        def insert(table, records):
            for record in records:
                cols = list(record)
                conn.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", tuple(record[c] for c in cols))
        insert('experiment_runs', [run])
        insert('decisions', decisions)
        insert('executions', executions)
        insert('daily_account', daily)
        insert('corporate_action_events',[dict(run_id=run['run_id'],**e) for e in events])
        last = daily[-1]
        insert('accounts', [dict(account_id=run['run_id'],run_id=run['run_id'],instrument='510300',cash=last['cash'],quantity=last['quantity'],dividend_receivable=last.get('dividend_receivable',0),last_valuation_date=last['trade_date'],account_version=len(daily))])
    return True


METRICS_SQL = """
WITH points AS (
 SELECT d.*, r.initial_capital,
 MAX(d.equity) OVER(PARTITION BY d.run_id ORDER BY d.trade_date ROWS UNBOUNDED PRECEDING) AS peak,
 ROW_NUMBER() OVER(PARTITION BY d.run_id ORDER BY d.trade_date DESC) AS last_row
 FROM daily_account d JOIN experiment_runs r USING(run_id)
)
SELECT run_id, MAX(CASE WHEN last_row=1 THEN equity/initial_capital-1 END) AS cumulative_return,
 MAX(1-equity/MAX(initial_capital,peak)) AS max_drawdown, SUM(fee_cash) AS fees,
 SUM(slippage_cash) AS diagnostic_slippage, AVG(exposure) AS average_exposure
FROM points GROUP BY run_id
"""
