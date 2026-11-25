"""Start a local historical replay from reproducible synthetic prices."""

from hashlib import sha256
from math import sin
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import pandas as pd

from src.data.akshare_provider import validate_bars
from src.data.corporate_actions import import_actions
from src.data.data_store import DataStore
from src.extensions.position_agent.agent_historical import AgentAccount, AgentTrial
from src.extensions.position_agent.daily import canonical, now, sessions
from src.extensions.position_agent.simulation import DB


DEMO_START = '2024-10-09'
DEMO_END = '2024-10-31'
DATA_START = '2024-05-06'
DATASET_ID = 'etf-allocation-synthetic-ohlcv-v1'
ACTIONS_MANIFEST = {
    'dataset_id': DATASET_ID,
    'instrument': '510300',
    'coverage_start': DATA_START,
    'coverage_end': DEMO_END,
    'share_changes_in_coverage': [],
    'actions': [],
    'sources': {},
    'verification': 'Synthetic fixture. No corporate actions are simulated in this replay.',
}
ACTIONS_PAYLOAD = canonical(ACTIONS_MANIFEST).encode()
ACTIONS_VERSION = sha256(ACTIONS_PAYLOAD).hexdigest()


def default_db():
    directory = Path(os.environ.get('POSITION_AGENT_DEMO_DIR', str(DB.parent)))
    return directory.expanduser().resolve() / DB.name


def _snapshot():
    dates = sessions(DATA_START, DEMO_END)
    decline = dates.index('2024-10-22')
    rows = []
    previous_close = 3560
    for index, date in enumerate(dates):
        close = 3560 + 4 * index + round(24 * sin(index / 5))
        close -= 24 * max(0, index - decline + 1)
        opening = previous_close + (index % 7) - 3
        rows.append((date, opening / 1000, (max(opening, close) + 7) / 1000,
                     (min(opening, close) - 7) / 1000, close / 1000,
                     3_000_000 + (index % 13) * 37_000))
        previous_close = close
    bars = pd.DataFrame(rows, columns=['trade_date', 'open', 'high', 'low', 'close', 'volume'])
    return validate_bars(bars, DATA_START, DEMO_END)


def _write_fixture(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('演示行情文件已变更，请使用新的演示数据目录。')
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as pending:
        temporary = Path(pending.name)
        try:
            pending.write(payload)
            pending.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _prepare_database(path):
    store = DataStore(str(path.parent))
    if store.db_path != path:
        raise ValueError(f'演示数据库文件名必须为 {store.db_path.name}。')
    fixtures = path.parent / DATASET_ID
    fixtures.mkdir(exist_ok=True)
    bars, quality = _snapshot()
    payload = bars.to_csv(index=False).encode()
    raw_path = fixtures / 'bars.csv'
    actions_path = fixtures / 'corporate-actions.json'
    _write_fixture(raw_path, payload)
    _write_fixture(actions_path, ACTIONS_PAYLOAD)
    metadata = dict(dataset_id=DATASET_ID, source='synthetic_demo', fetched_at=now(),
                    raw_hash=sha256(payload).hexdigest(), raw_path=str(raw_path),
                    instrument='510300', start_date=DATA_START, end_date=DEMO_END,
                    adjustment_policy='none', corporate_actions_version=ACTIONS_VERSION,
                    quality=quality)
    store.save_position_dataset(metadata, bars)
    import_actions(store, actions_path)


class DemoAccount(AgentAccount):
    POLICY_SOURCE_FILES = (*AgentAccount.POLICY_SOURCE_FILES, 'src/extensions/position_agent/demo.py')

    @staticmethod
    def load_checkpoint(directory):
        model, scaler, base, manifest = AgentAccount.load_checkpoint(directory)
        bars, _ = _snapshot()
        return model, scaler, {
            **base,
            'dataset_id': DATASET_ID,
            'normalized_hash': sha256(bars.to_csv(index=False).encode()).hexdigest(),
            'project': {**base['project'], 'corporate_actions_version': ACTIONS_VERSION},
        }, manifest

    def suggest(self, date):
        suggestion = super().suggest(date)
        suggestion['facts']['synthetic'] = True
        return suggestion


class DemoTrial(AgentTrial):
    NAMESPACE = 'etf-allocation-demo-v1'
    PROTOCOL = 'etf-allocation-synthetic-replay-v1'
    ACCOUNT_SERVICE = DemoAccount

    @classmethod
    def start(cls, path=None):
        path = Path(path).expanduser().resolve() if path is not None else default_db()
        _prepare_database(path)
        return cls.create(uuid4().hex, 'demo', DEMO_START, DEMO_END, 'daily', path=path)
