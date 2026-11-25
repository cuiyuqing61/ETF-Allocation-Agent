# Modified from FinRL-X for this 510300 ETF Agent release (Apache-2.0).
"""Local SQLite store for frozen ETF bars, accounts and backtests."""

from pathlib import Path
import sqlite3

from src.data.position_store import load_dataset, migrate, save_dataset, save_run


class DataStore:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / 'finrl_trading.db'
        migrate(self)

    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    def save_position_dataset(self, metadata, bars):
        return save_dataset(self, metadata, bars)

    def load_position_dataset(self, dataset_id):
        return load_dataset(self, dataset_id)

    def save_position_run(self, run, decisions, executions, daily, events=None):
        return save_run(self, run, decisions, executions, daily, events)
