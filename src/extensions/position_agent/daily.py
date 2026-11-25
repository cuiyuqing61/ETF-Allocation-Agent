"""Offline, restartable daily inference and settlement on the existing SQLite ledger.

No broker/network/training entry point. Historical as-of time is distinct from
actual processing time. BEGIN IMMEDIATE serializes writers; all financial writes
for a settlement commit together. An unfinished attempt can always be retried.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.backtest.position_replay import reconcile
from src.data.akshare_provider import validate_bars
from src.extensions.position_agent.model_versions import revision_matches
from src.strategies.etf_ppo import FrozenScaler, OBSERVATIONS, features
from src.trading.corporate_actions import CorporateActionLedger, reference_price
from src.trading.position_execution import Account, execute_target, mark_to_market, D

ROOT = Path(__file__).resolve().parents[3]
MAX_DATE = '2024-12-31'
CODE_FILES = ['src/extensions/position_agent/daily.py', 'src/trading/position_execution.py',
              'src/trading/corporate_actions.py', 'src/strategies/etf_ppo.py',
              'src/backtest/position_replay.py', 'src/data/daily_schema.sql',
              'src/data/akshare_provider.py', 'src/data/position_store.py',
              'src/data/position_schema.sql', 'src/data/data_store.py']


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def insert(conn, table, row):
    conn.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))


def sessions(start, end):
    return mcal.get_calendar('SSE').valid_days(start, end).strftime('%Y-%m-%d').tolist()


def bounded_date(date):
    if pd.Timestamp(date).strftime('%Y-%m-%d') != date or date > MAX_DATE:
        raise ValueError('Offline service accepts ISO dates through 2024 only; test set blocked')


class DailyService:
    MINIMUM_SEQUENCE_HISTORY = 0
    POLICY_SOURCE_FILES = ()

    def __init__(self, store, run_id, policy=None):
        self.store, self.run_id = store, run_id
        with store.connect() as conn:
            row = conn.execute('SELECT config_json FROM daily_sessions WHERE run_id=?', (run_id,)).fetchone()
        if not row:
            raise ValueError('Unknown daily account; initialize explicitly')
        self.cfg = json.loads(row[0])
        for path, expected in self.cfg['source_hashes'].items():
            actual = sha256((ROOT/path).read_bytes()).hexdigest()
            if not revision_matches(Path(path).stem, expected, actual):
                raise ValueError('Service code changed; review/migrate account explicitly')
        for path, expected in self.cfg['dependency_locks'].items():
            if sha256((ROOT/path).read_bytes()).hexdigest() != expected:
                raise ValueError('Dependency lock changed; review account environment')
        self.model, self.scaler, _, manifest = self.load_checkpoint(self.cfg['checkpoint'])
        self.sequence_policy = manifest['variant'] == 'transformer20-v1'
        if self.artifact_identity(self.cfg['checkpoint']) != self.cfg['artifact_identity']:
            raise ValueError('Frozen model/config/scaler changed')
        if policy is not None:  # explicit dependency injection for synthetic offline contracts
            self.model = policy

    @staticmethod
    def artifact_identity(directory):
        p = Path(directory)
        return {n: sha256((p/n).read_bytes()).hexdigest()
                for n in ('manifest.json', 'config.json', 'model.zip', 'scaler.json')}

    @staticmethod
    def load_checkpoint(directory):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        p = Path(directory)
        manifest = json.loads((p/'manifest.json').read_text())
        if manifest['mapping'] != 'clip(a,0,1)' or manifest['variant'] not in ('original', 'transformer20-v1'):
            raise ValueError('Only verified original action mapping is supported')
        if set(manifest['hashes']) != {'model.zip', 'scaler.json', 'config.json'}:
            raise ValueError('Incomplete checkpoint hashes')
        for name, expected in manifest['hashes'].items():
            if sha256((p/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Checkpoint hash mismatch')
        config = json.loads((p/'config.json').read_text())
        base = config['frozen_base_plan']
        scaler = FrozenScaler(**json.loads((p/'scaler.json').read_text()))
        if base['observations'] != OBSERVATIONS or scaler.train_end != '2021-12-31':
            raise ValueError('Incompatible observation order or scaler interval')
        model = PPO.load(p/'model.zip', device='cpu')
        expected_shape = (10,)
        if manifest['variant'] == 'transformer20-v1':
            from src.strategies.etf_transformer import ETFTransformerExtractor, OBS_SIZE, OBS_SPEC
            if manifest.get('observations') != OBS_SPEC or config.get('observation') != OBS_SPEC:
                raise ValueError('Incompatible Transformer observation specification')
            if not isinstance(model.policy.features_extractor, ETFTransformerExtractor):
                raise ValueError('Unexpected feature extractor')
            expected_shape = (OBS_SIZE,)
        if model.observation_space.shape != expected_shape or model.action_space.shape != (1,):
            raise ValueError('Incompatible policy shape')
        return model, scaler, base, manifest

    @classmethod
    def create(cls, store, run_id, checkpoint, start, end):
        bounded_date(start); bounded_date(end)
        if start > end or start < '2022-01-01':
            raise ValueError('Offline acceptance uses validation dates only')
        model, _, base, checkpoint_manifest = cls.load_checkpoint(checkpoint)
        sequence_policy = checkpoint_manifest['variant'] == 'transformer20-v1'
        source_files = CODE_FILES + (['src/strategies/etf_transformer.py'] if sequence_policy else [])
        source_files += cls.POLICY_SOURCE_FILES
        days = sessions(start, end)
        if not days:
            raise ValueError('No execution sessions')
        if end != days[-1]:
            raise ValueError('Freeze the offline end on a trading session, including intervening payment days')
        prior = sessions((pd.Timestamp(days[0])-pd.Timedelta(days=30)).strftime('%Y-%m-%d'), days[0])[-2]
        with store.connect() as conn:
            conn.row_factory = __import__('sqlite3').Row
            data = conn.execute('SELECT normalized_hash FROM datasets WHERE dataset_id=?', (base['dataset_id'],)).fetchone()
            if not data or data[0] != base['normalized_hash']:
                raise ValueError('Dataset differs from model provenance')
            actions = [dict(r) for r in conn.execute(
                'SELECT action_id,instrument,record_date,ex_date,pay_date,cash_per_unit,split_ratio,source FROM corporate_actions WHERE dataset_id=? AND record_date<=? ORDER BY ex_date',
                (base['dataset_id'], end))]
            version = base['project']['corporate_actions_version']
            evidence = conn.execute('SELECT manifest_json FROM corporate_action_sets WHERE dataset_id=? AND version=?', (base['dataset_id'], version)).fetchone()
            if not evidence:
                raise ValueError('Reviewed corporate actions version missing')
            manifest = json.loads(evidence[0])
            if manifest['coverage_start'] > prior or manifest['coverage_end'] < end or manifest['share_changes_in_coverage']:
                raise ValueError('Unverified corporate action coverage/share conversion')
            keys = ('action_id','record_date','ex_date','pay_date','cash_per_unit','split_ratio','source')
            expected_actions = [{**{k:a[k] for k in keys},'instrument':'510300'}
                                for a in manifest['actions'] if a['record_date'] <= end]
            if sorted(actions,key=lambda a:a['action_id']) != sorted(expected_actions,key=lambda a:a['action_id']):
                raise ValueError('Corporate action database differs from reviewed manifest')
        for path, expected in base['dependency_locks'].items():
            if sha256((ROOT/path).read_bytes()).hexdigest() != expected:
                raise ValueError('Dependency lock differs from frozen model')
        from src.extensions.position_agent.cli import code_fingerprint
        cfg = dict(source_code_hash=code_fingerprint(), checkpoint=str(Path(checkpoint).resolve()), artifact_identity=cls.artifact_identity(checkpoint),
                   dataset_id=base['dataset_id'], actions=actions, actions_version=version,
                   rules=base['market_rules'], costs=base['costs'], capital=base['project']['initial_capital'],
                   start=days[0], end=end, prior=prior, sessions=days, mode='offline_historical',
                   source_hashes={p: sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
                   upstream_commit=base['upstream_commit'], dependency_locks=base['dependency_locks'])
        ledger = CorporateActionLedger(actions, days[0])
        state = dict(account=asdict(Account(cfg['capital'])), date=prior, equity=cfg['capital'], peak=cfg['capital'],
                     entitlements=ledger.entitlements, applied=[], version=0)
        with store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            old = conn.execute('SELECT config_json FROM daily_sessions WHERE run_id=?', (run_id,)).fetchone()
            if old:
                if old[0] != canonical(cfg):
                    raise ValueError('Account id conflicts with frozen configuration')
                return run_id
            model_id = digest(cfg['artifact_identity'])
            if not conn.execute('SELECT 1 FROM models WHERE model_id=?', (model_id,)).fetchone():
                insert(conn, 'models', dict(model_id=model_id, algorithm='PPO-frozen-daily', train_start='2015-01-01',
                       train_end='2021-12-31', feature_version='transformer20-v1' if sequence_policy else 'causal-relative-v1', scaler_path=str(Path(checkpoint)/'scaler.json'),
                       model_path=str(Path(checkpoint)/'model.zip'), artifact_hash=model_id))
            insert(conn, 'experiment_runs', dict(run_id=run_id, strategy_version='daily-offline-v1', model_id=model_id,
                   dataset_id=cfg['dataset_id'], seed=int(model.seed), split='validation', evaluation_mode='historical',
                   period_start=cfg['start'], period_end=end, initial_capital=cfg['capital'], config_json=canonical(cfg),
                   code_commit=cfg['upstream_commit'], status='incremental_offline', payload_hash=digest(cfg)))
            insert(conn, 'daily_sessions', dict(run_id=run_id, config_json=canonical(cfg)))
            insert(conn, 'daily_state', dict(run_id=run_id, version=0, state_json=canonical(state)))
            insert(conn, 'daily_snapshots', dict(run_id=run_id, version=0, state_json=canonical(state), processed_at=now()))
            insert(conn, 'accounts', dict(account_id=run_id, run_id=run_id, instrument='510300', cash=cfg['capital'],
                   quantity=0, dividend_receivable=0, last_valuation_date=prior, account_version=0))
        return run_id

    def _bars(self, conn, date, as_of):
        bounded_date(date)
        from src.strategies.etf_transformer import HISTORY
        history = max(HISTORY, self.MINIMUM_SEQUENCE_HISTORY) if self.sequence_policy else 21
        b = pd.read_sql_query('''SELECT trade_date,raw_open AS open,raw_high AS high,raw_low AS low,
            raw_close AS close,volume,tradable,availability_at FROM market_bars
            WHERE dataset_id=? AND instrument='510300' AND trade_date<=? ORDER BY trade_date DESC LIMIT ?''',
            conn, params=(self.cfg['dataset_id'], date, history)).iloc[::-1].reset_index(drop=True)
        b, _ = validate_bars(b)
        if len(b) != history or b.iloc[-1].trade_date != date or sessions(b.iloc[0].trade_date, date) != b.trade_date.tolist():
            raise ValueError('Missing/stale session or feature warmup; no forward fill')
        if pd.Timestamp(as_of).tzinfo is None or pd.Timestamp(as_of) < pd.Timestamp(date+'T15:30:00+08:00'):
            raise ValueError('Daily bar unavailable before close')
        if any(pd.Timestamp(x) > pd.Timestamp(as_of) for x in b.availability_at):
            raise ValueError('Input unavailable at requested logical as-of time')
        return b

    def _job(self, operation, date, work, failpoint=None):
        bounded_date(date)
        attempt = uuid.uuid4().hex
        with self.store.connect() as conn:
            insert(conn, 'daily_attempts', dict(attempt_id=attempt, run_id=self.run_id, operation=operation,
                   logical_date=date, started_at=now(), status='started'))
        try:
            with self.store.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                state = json.loads(conn.execute('SELECT state_json FROM daily_state WHERE run_id=?', (self.run_id,)).fetchone()[0])
                result, reused = work(conn, state)
                if failpoint == 'before_commit':
                    raise RuntimeError('Injected interruption before commit')
                conn.execute('UPDATE daily_attempts SET status=?,finished_at=? WHERE attempt_id=?',
                             ('reused' if reused else 'committed', now(), attempt))
            if failpoint == 'after_commit':
                raise RuntimeError('Injected response loss after commit')
            return dict(result=result, reused=reused)
        except Exception as exc:
            with self.store.connect() as conn:
                conn.execute("UPDATE daily_attempts SET status='failed',finished_at=?,error=? WHERE attempt_id=? AND status='started'", (now(), str(exc), attempt))
            raise

    def policy_observation(self, bars, account, equity, peak):
        close = float(bars.close.iloc[-1])
        indicators = features(bars)
        if self.sequence_policy:
            from src.strategies.etf_transformer import WINDOW, observation
            window = self.scaler.transform(indicators.iloc[-WINDOW:])
            return observation(window, account, close, equity, peak)
        return np.asarray([*self.scaler.transform(indicators.iloc[-1]),
                           account.quantity * close / equity, account.cash / equity,
                           1 - equity / peak], dtype=np.float32)

    def decide(self, date, as_of=None, failpoint=None):
        as_of = as_of or date+'T15:30:00+08:00'
        def work(conn, state):
            old = conn.execute('SELECT decision_id FROM decisions WHERE run_id=? AND decision_date=?', (self.run_id, date)).fetchone()
            if old:
                return old[0], True
            if state['date'] != date:
                raise ValueError('Settle previous sessions first; account date differs from decision date')
            next_days = [d for d in self.cfg['sessions'] if d > date]
            if not next_days:
                raise ValueError('No next session inside frozen offline period')
            b = self._bars(conn, date, as_of)
            a = Account(**state['account']); close = float(b.iloc[-1].close)
            equity = a.cash+a.quantity*close+a.receivable
            if abs(equity-state['equity']) > 1e-7:
                raise ValueError('Stored account valuation mismatch')
            obs = self.policy_observation(b, a, equity, state['peak'])
            action, _ = self.model.predict(obs, deterministic=True)
            target = float(action[0])
            if not np.isfinite(target) or not 0 <= target <= 1:
                raise ValueError('Invalid PPO target')
            snapshot = dict(bars=b.to_dict('records'), observation=obs.tolist(), account_state=state,
                            artifact_identity=self.cfg['artifact_identity'], logical_as_of=as_of)
            decision_id = digest([self.run_id, date])
            insert(conn, 'decisions', dict(decision_id=decision_id, run_id=self.run_id, decision_date=date,
                   created_at=now(), data_as_of=as_of, snapshot_hash=digest(snapshot), current_weight=a.quantity*close/equity,
                   raw_target_weight=target, target_weight=target, planned_execution_date=next_days[0], guard_reason='validated_offline'))
            insert(conn, 'daily_inputs', dict(decision_id=decision_id, account_version=state['version'], snapshot_json=canonical(snapshot)))
            return decision_id, False
        return self._job('decide', date, work, failpoint)

    def settle(self, date, as_of=None, failpoint=None):
        as_of = as_of or date+'T15:30:00+08:00'
        def work(conn, state):
            old = conn.execute('SELECT execution_id FROM executions WHERE run_id=? AND execution_date=?', (self.run_id, date)).fetchone()
            if old:
                return old[0], True
            expected = next((d for d in self.cfg['sessions'] if d > state['date']), None)
            if date != expected:
                raise ValueError('Out-of-order settlement; resume the next unprocessed session')
            row = conn.execute('''SELECT d.decision_id,d.target_weight,i.account_version,i.snapshot_json
                FROM decisions d JOIN daily_inputs i USING(decision_id)
                WHERE d.run_id=? AND d.decision_date=? AND d.planned_execution_date=?''', (self.run_id, state['date'], date)).fetchone()
            if not row or row[2] != state['version']:
                raise ValueError('Missing decision or stale account version')
            b = self._bars(conn, date, as_of); bar = b.iloc[-1].to_dict()
            snapshot = json.loads(row[3]); previous = snapshot['bars'][-1]
            if b.iloc[-2].trade_date != previous['trade_date'] or float(b.iloc[-2].close) != previous['close']:
                raise ValueError('Decision/settlement source conflict')
            account = Account(**state['account'])
            ledger = CorporateActionLedger(self.cfg['actions'], self.cfg['start'])
            ledger.entitlements = state['entitlements']; ledger.applied = set(map(tuple, state['applied']))
            equity = state['equity']; peak = state['peak']; daily = []
            days = sorted({a['pay_date'] for a in self.cfg['actions'] if state['date'] < a['pay_date'] < date})+[date]
            for day in days:
                opening = account
                if day == date:
                    account = replace(account, sellable=account.quantity)
                    account, accrued = ledger.open(account, day)
                    reference = reference_price(previous['close'], day, self.cfg['actions'], self.cfg['rules']['tick_size'])
                    account, execution = execute_target(account, row[1], bar, reference, self.cfg['rules'], self.cfg['costs'])
                    ex = execution; close = float(bar['close'])
                else:
                    accrued = 0.; reference = None; close = previous['close']
                    ex = dict(signed_quantity=0, notional=0., fee_cash=0., slippage_cash=0., fill_price=None)
                account, paid = ledger.close(account, day)
                valuation = mark_to_market(account, close, equity)
                daily.append(dict(run_id=self.run_id, trade_date=day, **valuation, traded_notional=ex['notional'],
                    fee_cash=ex['fee_cash'], slippage_cash=ex['slippage_cash'], dividend_cash=paid, dividend_accrued=accrued,
                    prev_cash=opening.cash, prev_quantity=opening.quantity, prev_receivable=opening.receivable,
                    trade_cash=float(-D(ex['signed_quantity'])*D(ex['fill_price'] or 0)), signed_quantity=ex['signed_quantity'],
                    quantity_change=0, close_price=close, reference_price=reference, is_session=int(day == date), market_regime='unclassified'))
                equity = valuation['equity']; peak = max(peak, equity)
            execution_id = digest([self.run_id, row[0], 'execution'])
            insert(conn, 'executions', dict(execution_id=execution_id, run_id=self.run_id, decision_id=row[0], execution_date=date, **execution))
            for d in daily: insert(conn, 'daily_account', d)
            for event in ledger.events: insert(conn, 'corporate_action_events', dict(run_id=self.run_id, **event))
            # Independent full-chain identities before the transaction is committed.
            all_daily = pd.read_sql_query('SELECT * FROM daily_account WHERE run_id=? ORDER BY trade_date', conn, params=(self.run_id,))
            reconcile(all_daily.to_dict('records'), self.cfg['capital'])
            new_state = dict(account=asdict(account), date=date, equity=equity, peak=peak,
                             entitlements=ledger.entitlements, applied=sorted(ledger.applied), version=state['version']+1,
                             execution_bar=bar, execution_as_of=as_of, execution_input_hash=digest(bar))
            conn.execute('UPDATE daily_state SET version=?,state_json=? WHERE run_id=? AND version=?',
                         (new_state['version'], canonical(new_state), self.run_id, state['version']))
            insert(conn, 'daily_snapshots', dict(run_id=self.run_id, version=new_state['version'], state_json=canonical(new_state), processed_at=now()))
            conn.execute('UPDATE accounts SET cash=?,quantity=?,dividend_receivable=?,last_valuation_date=?,account_version=? WHERE run_id=?',
                         (account.cash, account.quantity, account.receivable, date, new_state['version'], self.run_id))
            return execution_id, False
        return self._job('settle', date, work, failpoint)

    def status(self):
        with self.store.connect() as conn:
            state = json.loads(conn.execute('SELECT state_json FROM daily_state WHERE run_id=?', (self.run_id,)).fetchone()[0])
            attempts = pd.read_sql_query('SELECT * FROM daily_attempt_diagnostics WHERE run_id=? ORDER BY started_at DESC LIMIT 10', conn, params=(self.run_id,)).to_dict('records')
            pending = conn.execute('SELECT planned_execution_date FROM decisions d WHERE run_id=? AND NOT EXISTS (SELECT 1 FROM executions e WHERE e.decision_id=d.decision_id)', (self.run_id,)).fetchone()
        next_day = next((d for d in self.cfg['sessions'] if d > state['date']), None)
        return dict(mode='offline_historical', state=state, next_operation='complete' if next_day is None else ('settle' if pending else 'decide'),
                    next_date=next_day if pending else state['date'], recent_attempts=attempts)
