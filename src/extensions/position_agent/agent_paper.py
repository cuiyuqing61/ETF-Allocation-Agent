"""Time-driven 510300 paper accounts using the frozen Agent policy and spot prices."""
import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, time, timedelta
import json
import logging
import math
import os
from pathlib import Path
import secrets
import sqlite3
from time import sleep
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.akshare_provider import AkshareDataProvider, validate_bars
from src.data.corporate_actions import read_manifest
from src.extensions.position_agent.agent_policy import ROOT, SETTINGS, VERSION, load_release, make_suggestion
from src.extensions.position_agent.daily import canonical, digest, sessions
from src.trading.corporate_actions import CorporateActionLedger, reference_price
from src.trading.position_execution import Account, execute_target, mark_to_market

ZONE = ZoneInfo('Asia/Shanghai')
OPEN = time(9, 30)
OPEN_END = time(9, 40)
CLOSE = time(15, 30)
CAPITAL = 100000.
PAPER_DB = ROOT / 'data/position_agent/agent-paper.sqlite3'
SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_accounts(
    id TEXT PRIMARY KEY, created_on TEXT NOT NULL, state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_cards(
    id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES paper_accounts(id),
    decision_date TEXT NOT NULL, execution_date TEXT NOT NULL, snapshot TEXT NOT NULL,
    declined INTEGER NOT NULL DEFAULT 0, request TEXT, confirmed_at TEXT,
    UNIQUE(account_id, execution_date)
);
CREATE TABLE IF NOT EXISTS paper_executions(
    account_id TEXT NOT NULL REFERENCES paper_accounts(id), day TEXT NOT NULL, receipt TEXT NOT NULL,
    PRIMARY KEY(account_id, day)
);
CREATE TABLE IF NOT EXISTS paper_closes(
    account_id TEXT NOT NULL REFERENCES paper_accounts(id), day TEXT NOT NULL, valuation TEXT NOT NULL,
    PRIMARY KEY(account_id, day)
);
"""
log = logging.getLogger(__name__)


def default_db():
    return Path(os.environ.get('POSITION_AGENT_PAPER_DB', PAPER_DB))


def market_now():
    return datetime.now(ZONE)


def local(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('行情和执行时间需要包含时区')
    return stamp.to_pydatetime().astimezone(ZONE)


def at(day, wall_time):
    return datetime.combine(datetime.fromisoformat(day).date(), wall_time, ZONE)


def next_session(day):
    end = at(day, OPEN) + timedelta(days=31)
    return next(d for d in sessions(day, end.date().isoformat()) if d > day)


def no_fill(reason):
    return dict(signed_quantity=0, notional=0., fee_cash=0., slippage_cash=0.,
                fill_price=None, reference_open=None, status='no_fill', reason=reason)


@dataclass(frozen=True)
class ReviewedActions:
    start: str
    end: str
    entries: list
    version: str

    @classmethod
    def load(cls, path):
        version, evidence = read_manifest(path)
        return cls(evidence['coverage_start'], evidence['coverage_end'], evidence['actions'], version)

    def through(self, previous, day):
        if not self.start <= previous <= day <= self.end:
            raise ValueError('缺少该区间已核对的分红和份额变动资料')
        return [action for action in self.entries if action['record_date'] <= day]


@dataclass(frozen=True)
class OpeningQuote:
    day: str
    open: float
    volume: float
    as_of: datetime

    def execution_bar(self, day, clock):
        if (self.day != day or not at(day, OPEN) <= local(self.as_of) <= clock < at(day, OPEN_END)
                or not all(math.isfinite(v) and v > 0 for v in (self.open, self.volume))):
            raise ValueError('缺少当日开盘时段的有效行情')
        return dict(open=self.open, close=self.open, volume=self.volume, tradable=True)


class EastMoneyFeed:
    def __init__(self, archive_dir):
        self.daily = AkshareDataProvider(str(archive_dir))

    def closes(self, start, end):
        metadata, frame = self.daily.fetch_snapshot('510300', start, end)
        if local(metadata['fetched_at']) < at(end, CLOSE):
            raise ValueError('收盘行情尚未到达可用时间')
        return frame

    def opening(self, day):
        import akshare as ak
        row = ak.fund_etf_spot_em().loc[lambda f: f['代码'].astype(str) == '510300']
        if len(row) != 1:
            raise ValueError('缺少 510300 的开盘报价')
        value = row.iloc[0]
        return OpeningQuote(pd.Timestamp(value['数据日期']).date().isoformat(), float(value['开盘价']),
                            float(value['成交量']), local(value['更新时间']))


class SinaFeed:
    """Timestamped single-ETF quotes, with AKShare's Sina history decoder."""

    @staticmethod
    def _get(url):
        import requests
        response = requests.get(url, headers={'Referer': 'https://finance.sina.com.cn/'}, timeout=12)
        response.raise_for_status()
        return response.text

    def _quote(self):
        raw = self._get('https://hq.sinajs.cn/list=sh510300')
        name, separator, message = raw.partition('=')
        if not separator or name.strip() != 'var hq_str_sh510300':
            raise ValueError('新浪返回了意外的证券代码')
        value = json.loads(message.strip().removesuffix(';')).split(',')
        if len(value) < 32:
            raise ValueError('新浪行情字段不足')
        stamp = local(value[30]+'T'+value[31]+'+08:00')
        quote = dict(trade_date=stamp.date().isoformat(), open=float(value[1]), close=float(value[3]),
                     high=float(value[4]), low=float(value[5]), volume=float(value[8]))
        amount = float(value[9])
        if quote['volume'] > 0 and not quote['low']-.002 <= amount/quote['volume'] <= quote['high']+.002:
            raise ValueError('成交金额与新浪份额单位无法核对')
        return quote, stamp

    def opening(self, day):
        quote, stamp = self._quote()
        return OpeningQuote(quote['trade_date'], quote['open'], quote['volume'], stamp)

    def closes(self, start, end):
        from akshare.fund import fund_etf_sina
        raw = self._get('https://finance.sina.com.cn/realstock/company/sh510300/hisdata_klc2/klc_kl.js')
        # Treat the remote payload as encoded data, not executable JavaScript.
        encoded = json.loads(raw.split('=', 1)[1].split(';', 1)[0].strip())
        decoder = fund_etf_sina.py_mini_racer.MiniRacer()
        decoder.eval(fund_etf_sina.hk_js_decode)
        frame = pd.DataFrame(decoder.call('d', encoded)).rename(columns={'date': 'trade_date'})
        columns = ['trade_date','open','high','low','close','volume']
        frame['trade_date'] = pd.to_datetime(frame['trade_date'], errors='raise').dt.strftime('%Y-%m-%d')
        frame = frame.loc[frame.trade_date.ge(start) & frame.trade_date.lt(end), columns]
        # The historical endpoint may publish today's close on the following day.
        quote, stamp = self._quote()
        if quote['trade_date'] != end or not at(end, CLOSE) <= stamp <= market_now():
            raise ValueError('尚未收到当日收盘后的 510300 行情')
        bars, _ = validate_bars(pd.concat([frame, pd.DataFrame([quote])], ignore_index=True), start, end)
        bars.loc[bars.trade_date.eq(end), 'availability_at'] = stamp.isoformat()
        return bars


class PaperStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self, write=False):
        with closing(sqlite3.connect(self.path, timeout=15)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA foreign_keys=ON')
            if write:
                conn.execute('BEGIN IMMEDIATE')
            yield conn

    def create(self, clock=None):
        clock = local(clock or market_now())
        token = secrets.token_urlsafe(32)
        account_id = self.account_id(token)
        state = dict(date=None, first_day=None, close=None, open_day=None, equity=CAPITAL, peak=CAPITAL, version=0,
                     account=asdict(Account(CAPITAL)), entitlements={}, applied=[], reviewed_actions={},
                     policy_version=VERSION)
        with self.connection(write=True) as conn:
            conn.execute('INSERT INTO paper_accounts VALUES(?,?,?)',
                         (account_id, clock.date().isoformat(), canonical(state)))
        return token

    @staticmethod
    def account_id(token):
        return digest(['paper-v1', token])

    def _state(self, conn, account_id):
        row = conn.execute('SELECT state FROM paper_accounts WHERE id=?', (account_id,)).fetchone()
        if row is None:
            raise ValueError('找不到实时模拟账户，请核对恢复码')
        return json.loads(row['state'])

    def _save(self, conn, account_id, state):
        conn.execute('UPDATE paper_accounts SET state=? WHERE id=?', (canonical(state), account_id))

    @staticmethod
    def _card(row, clock, receipt):
        snapshot = json.loads(row['snapshot'])
        completed = receipt is not None and receipt['execution_date'] == row['execution_date']
        return dict(id=row['id'], day=row['decision_date'], facts=snapshot['facts'],
                    declined=bool(row['declined']), request=row['request'],
                    status='complete' if completed else 'ready',
                    actionable=not completed and clock < at(row['execution_date'], OPEN))

    def view(self, token, clock=None):
        clock = local(clock or market_now())
        account_id = self.account_id(token)
        with self.connection() as conn:
            state = self._state(conn, account_id)
            card = conn.execute('SELECT * FROM paper_cards WHERE account_id=? ORDER BY decision_date DESC LIMIT 1',
                                (account_id,)).fetchone()
            trade = conn.execute('''SELECT e.receipt,c.valuation FROM paper_executions e
                LEFT JOIN paper_closes c ON c.account_id=e.account_id AND c.day=e.day
                WHERE e.account_id=? ORDER BY e.day DESC LIMIT 1''', (account_id,)).fetchone()
        receipt = json.loads(trade['receipt']) if trade else None
        if receipt is not None and trade['valuation']:
            valuation = json.loads(trade['valuation'])
            receipt.update({k: valuation[k] for k in ('exposure', 'equity', 'quantity')})
        return state, self._card(card, clock, receipt) if card else None, receipt

    def inbox(self, token, clock=None):
        clock = local(clock or market_now())
        account_id = self.account_id(token)
        with self.connection() as conn:
            rows = conn.execute('''SELECT c.decision_date,c.execution_date,c.snapshot,c.declined,c.request,
                e.receipt FROM paper_cards c LEFT JOIN paper_executions e
                ON e.account_id=c.account_id AND e.day=c.execution_date
                WHERE c.account_id=? ORDER BY c.decision_date DESC LIMIT 20''', (account_id,)).fetchall()
        notices = []
        for row in rows:
            if row['receipt']:
                receipt = json.loads(row['receipt'])
                status = '已成交' if receipt['signed_quantity'] else '已处理 · 未成交'
            elif clock >= at(row['execution_date'], OPEN):
                status = '等待开盘处理' if row['request'] else '已到期'
            else:
                status = '已确认' if row['request'] else '已拒绝' if row['declined'] else '待确认'
            target = json.loads(row['snapshot'])['target']
            notices.append({'建议日': row['decision_date'], '计划执行日': row['execution_date'],
                            '模型建议': '保持份额' if target is None else f'目标 {target:.2%}', '状态': status})
        return notices

    def choose(self, token, card, target=None, *, adopt=False, decline=False, clock=None):
        clock = local(clock or market_now())
        account_id = self.account_id(token)
        with self.connection(write=True) as conn:
            state = self._state(conn, account_id)
            saved = conn.execute('SELECT * FROM paper_cards WHERE account_id=? AND id=?',
                                 (account_id, card['id'])).fetchone()
            if not saved:
                raise ValueError('找不到这条建议')
            snapshot = json.loads(saved['snapshot'])
            if adopt:
                target = snapshot['target']
            if target is not None and (not math.isfinite(target) or not 0 <= target <= 1):
                raise ValueError('仓位请输入 0% 至 100%')
            request = canonical({'target': target, 'adopt': adopt})
            if saved['request']:
                if not decline and saved['request'] == request:
                    return
                raise ValueError('这个执行日已有已确认操作')
            if (clock >= at(saved['execution_date'], OPEN) or state['version'] != snapshot['state']['version']
                    or state['open_day'] is not None or state['date'] != saved['decision_date']):
                raise ValueError('卡片已过期，请查看新的建议')
            if decline:
                conn.execute('UPDATE paper_cards SET declined=1 WHERE id=?', (card['id'],))
            else:
                if adopt and saved['declined']:
                    raise ValueError('原建议已拒绝，请确认自选仓位')
                conn.execute('UPDATE paper_cards SET request=?,confirmed_at=? WHERE id=?',
                             (request, clock.isoformat(), card['id']))


class PaperWorker:
    def __init__(self, store, feed, review, release=None, clock=market_now):
        self.store, self.feed, self.review, self.clock = store, feed, review, clock
        self.model, self.scaler, base, _ = release or load_release()
        self.rules, self.costs = base['market_rules'], base['costs']

    def _ledger(self, state, day):
        actions = self.review.through(state['date'], day)
        earlier = {a['action_id']: digest(a) for a in actions if a['record_date'] <= state['date']}
        if earlier != state['reviewed_actions']:
            raise ValueError('分红资料与上次收盘已记账的事件不符')
        ledger = CorporateActionLedger(actions, state['first_day'])
        ledger.entitlements.update(state['entitlements'])
        ledger.applied.update(map(tuple, state['applied']))
        return actions, ledger

    @staticmethod
    def _entitlements(state, ledger):
        state.update(entitlements=ledger.entitlements, applied=sorted(ledger.applied))

    def _opening(self, conn, account_id, state, day, clock, bar, quote):
        if state['open_day'] == day:
            return
        if state['open_day'] is not None or next_session(state['date']) != day:
            raise ValueError('请先完成前一交易日结算')
        actions, ledger = self._ledger(state, day)
        account = Account(**state['account'])
        prev = account
        paid_between = 0.
        for pay_day in sorted({a['pay_date'] for a in actions if state['date'] < a['pay_date'] < day}):
            account, paid = ledger.close(account, pay_day)
            paid_between += paid
        account, accrued = ledger.open(replace(account, sellable=account.quantity), day)
        saved = conn.execute('SELECT request,confirmed_at,snapshot FROM paper_cards WHERE account_id=? AND execution_date=?',
                             (account_id, day)).fetchone()
        request = json.loads(saved['request']) if saved and saved['request'] else None
        if request:
            snapshot = json.loads(saved['snapshot'])
            if local(saved['confirmed_at']) >= at(day, OPEN) or snapshot['state']['version'] != state['version']:
                raise ValueError('确认已过期或账户版本不符')
            if request['adopt'] and request['target'] != snapshot['target']:
                raise ValueError('已确认的目标与建议快照不符')
        target = request['target'] if request else None
        if target is None:
            result = no_fill('HOLD' if request else 'expired')
        elif bar is None:
            result = no_fill('opening_quote_unavailable')
        else:
            reference = reference_price(state['close'], day, actions, self.rules['tick_size'])
            account, result = execute_target(account, target, bar, reference, self.rules, self.costs)
        signed = result['signed_quantity']
        if (account.quantity != prev.quantity + signed
                or abs(account.cash - (prev.cash + paid_between - signed*(result['fill_price'] or 0.)
                                       - result['fee_cash'])) > 1e-7
                or abs(account.receivable - (prev.receivable - paid_between + accrued)) > 1e-7):
            raise ValueError('开盘份额、现金与回执无法核对')
        self._entitlements(state, ledger)
        receipt = dict(execution_date=day, recorded_at=clock.isoformat(), target=target, **result,
                       quote_as_of=quote.as_of.isoformat() if quote and bar is not None else None,
                       account_before=asdict(prev), account_after=asdict(account),
                       dividends_accrued=accrued, dividends_paid_before_open=paid_between,
                       corporate_actions_version=self.review.version)
        conn.execute('INSERT INTO paper_executions VALUES(?,?,?)', (account_id, day, canonical(receipt)))
        state.update(account=asdict(account), open_day=day)
        self.store._save(conn, account_id, state)

    def _closing(self, conn, account_id, state, bars):
        last = bars.iloc[-1]
        day = str(last.trade_date)
        if state['date'] == day:
            return
        if state.get('policy_version', VERSION) != VERSION:
            raise ValueError('实时模拟账户的策略版本已冻结')
        if state['date'] is None:
            state['first_day'] = day
            actions = self.review.through(day, day)
            ledger = CorporateActionLedger(actions, day)
            account = Account(**state['account'])
        else:
            if state['open_day'] != day:
                raise ValueError('收盘估值缺少开盘记录')
            actions, ledger = self._ledger(state, day)
            account = Account(**state['account'])
        account, paid = ledger.close(account, day)
        self._entitlements(state, ledger)
        valuation = mark_to_market(account, float(last.close), state['equity'])
        if abs(valuation['equity']-account.cash-account.quantity*float(last.close)-account.receivable) > 1e-7:
            raise ValueError('实时模拟账本核对失败')
        state.update(date=day, open_day=None, account=asdict(account), close=float(last.close),
                     equity=valuation['equity'], peak=max(state['peak'], valuation['equity']),
                     version=state['version']+1,
                     reviewed_actions={a['action_id']: digest(a) for a in actions})
        conn.execute('INSERT INTO paper_closes VALUES(?,?,?)',
                     (account_id, day, canonical(dict(valuation, dividend_cash=paid))))
        if len(bars) >= SETTINGS.trend_sessions:
            suggestion = make_suggestion(bars.tail(SETTINGS.trend_sessions), state, next_session(day),
                                         self.model, self.scaler)
            card_id = digest([account_id, suggestion])
            conn.execute('INSERT INTO paper_cards(id,account_id,decision_date,execution_date,snapshot) VALUES(?,?,?,?,?)',
                         (card_id, account_id, day, suggestion['facts']['execution_date'], canonical(suggestion)))
        self.store._save(conn, account_id, state)

    def tick(self):
        clock = local(self.clock())
        today = clock.date().isoformat()
        if today not in sessions(today, today) or clock < at(today, OPEN):
            return
        with self.store.connection() as conn:
            accounts = [(r['id'], r['created_on'], json.loads(r['state']))
                        for r in conn.execute('SELECT * FROM paper_accounts')]
        if clock >= at(today, CLOSE):
            self._after_close(today, clock, accounts)
            return
        due = [aid for aid, _, state in accounts
               if state['date'] is not None and state['open_day'] is None and next_session(state['date']) == today]
        if not due:
            return
        bar = quote = None
        if clock < at(today, OPEN_END):
            quote = self.feed.opening(today)
            clock = local(self.clock())
            if clock < at(today, OPEN_END):
                bar = quote.execution_bar(today, clock)
        for aid in due:
            with self.store.connection(write=True) as conn:
                self._opening(conn, aid, self.store._state(conn, aid), today, clock, bar, quote)

    def _after_close(self, today, clock, accounts):
        pending = [(aid, created, state) for aid, created, state in accounts if state['date'] != today]
        if not pending:
            return
        oldest = min(state['date'] or created for _, created, state in pending)
        start = (at(oldest, OPEN)-timedelta(days=110)).date().isoformat()
        fetched = self.feed.closes(start, today)
        data, _ = validate_bars(fetched, start, today)
        if 'availability_at' in fetched:
            data['availability_at'] = data.trade_date.map(fetched.set_index('trade_date').availability_at)
        for aid, created, _ in pending:
            with self.store.connection(write=True) as conn:
                state = self.store._state(conn, aid)
                if state['date'] is None:
                    # A new account begins with current cash after its first available close.
                    if created <= today:
                        self._closing(conn, aid, state, data)
                    continue
                closes = data.index[data.trade_date > state['date']]
                for i in closes:
                    day = str(data.at[i, 'trade_date'])
                    if state['open_day'] is None:
                        self._opening(conn, aid, state, day, clock, None, None)
                    self._closing(conn, aid, state, data.iloc[:i+1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=default_db())
    parser.add_argument('--actions', type=Path, required=True, help='Reviewed 510300 corporate action manifest')
    parser.add_argument('--feed', choices=('sina', 'eastmoney'), default='sina')
    parser.add_argument('--once', action='store_true', help='Process the current Shanghai market time once')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    store = PaperStore(args.db)
    feed = SinaFeed() if args.feed == 'sina' else EastMoneyFeed(args.db.parent / 'paper-market-snapshots')
    worker = PaperWorker(store, feed, ReviewedActions.load(args.actions))
    if args.once:
        worker.tick()
        return
    while True:
        try:
            # Refresh reviewed evidence so a running worker can pick up newly published events.
            worker.review = ReviewedActions.load(args.actions)
            worker.tick()
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning('实时模拟等待下次处理：%s', exc)
        sleep(30)


if __name__ == '__main__':
    main()
