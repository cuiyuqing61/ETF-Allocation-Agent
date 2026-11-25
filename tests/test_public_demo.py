from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.extensions.position_agent.agent_paper import (
    CAPITAL, ZONE, OpeningQuote, PaperStore, PaperWorker, ReviewedActions,
)
from src.extensions.position_agent.daily import sessions


APP = Path(__file__).resolve().parents[1] / 'src/web/app.py'


def test_public_historical_demo_starts_in_an_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('POSITION_AGENT_DEMO_DIR', str(tmp_path))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=60)

    def click(label):
        next(button for button in app.button if button.label == label).click().run(timeout=60)
        assert not app.exception

    click('开始历史模拟')
    assert any('演示数据' in note.value for note in app.info)
    assert all('参与码' not in field.label for field in app.text_input)
    click('采纳并进入下一交易日')
    assert any('最近成交回执' in note.value for note in app.markdown)

    click('不采纳，和 Agent 聊聊')
    app.chat_input[0].set_value('把仓位调到 35%').run(timeout=60)
    click('确认自选并进入下一交易日')
    assert any('35.00%' in note.value for note in app.caption)

    with sqlite3.connect(tmp_path / 'finrl_trading.db') as conn:
        tables = {name for name, in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'user_events', 'study_feedback', 'study_sessions', 'feedback'}.isdisjoint(tables)
        assert conn.execute('SELECT source FROM datasets').fetchall() == [('synthetic_demo',)]
        assert conn.execute("SELECT COUNT(*) FROM executions WHERE run_id LIKE '%-user'").fetchone() == (2,)


def test_realtime_paper_requires_confirmation_and_matches_ledger(tmp_path):
    clock = datetime(2026, 11, 20, 15, 31, tzinfo=ZONE)

    class Feed:
        def closes(self, start, end):
            days = sessions(start, end)
            prices = [3 + .015 * index for index in range(len(days))]
            return pd.DataFrame(dict(trade_date=days, open=prices, high=prices,
                                     low=prices, close=prices, volume=1_000_000.))

        def opening(self, day):
            return OpeningQuote(day, 3.92, 500_000., clock)

    store = PaperStore(tmp_path / 'paper.sqlite3')
    token = store.create(clock)
    idle_token = store.create(clock)
    actions = ReviewedActions('2026-01-01', '2026-12-31', [], 'test-review')
    worker = PaperWorker(store, Feed(), actions, clock=lambda: clock)
    worker.tick()
    _, card, _ = store.view(token, clock)
    assert card['facts']['execution_date'] == '2026-11-23'
    store.choose(token, card, adopt=True, clock=clock)
    store.choose(token, card, adopt=True, clock=clock)

    clock = datetime(2026, 11, 23, 9, 31, tzinfo=ZONE)
    worker.tick()
    worker.tick()
    state, _, receipt = store.view(token, clock)
    assert receipt['signed_quantity'] > 0
    assert state['account']['cash'] == pytest.approx(CAPITAL - receipt['notional'] - receipt['fee_cash'])
    assert store.view(idle_token, clock)[2]['reason'] == 'expired'
    with store.connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM paper_executions').fetchone()[0] == 2
