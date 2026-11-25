"""Approval and receipt surface for the versioned historical Agent trial."""
import json
import sqlite3

from src.extensions.position_agent.agent_policy import CHECKPOINT, SETTINGS, VERSION, load_release, make_suggestion
from src.extensions.position_agent.daily import canonical, digest, insert, now
from src.extensions.position_agent.simulation import AccountService, Trial


class AgentAccount(AccountService):
    MINIMUM_SEQUENCE_HISTORY = SETTINGS.trend_sessions
    POLICY_SOURCE_FILES = (
        'src/extensions/position_agent/agent_policy.py',
        'src/extensions/position_agent/agent_historical.py',
        'src/extensions/position_agent/policy_rules.py',
    )

    @staticmethod
    def load_checkpoint(directory):
        model, scaler, base, hashes = load_release(directory)
        return model, scaler, base, {'variant': 'transformer20-v1', 'hashes': hashes}

    def suggest(self, date):
        with self.store.connect() as conn:
            state = json.loads(conn.execute('SELECT state_json FROM daily_state WHERE run_id=?',
                                            (self.run_id,)).fetchone()[0])
            if state['date'] != date:
                raise ValueError('账户尚未到达建议时点')
            bars = self._bars(conn, date, date+'T15:30:00+08:00')
        execution = next(d for d in self.cfg['sessions'] if d > date)
        return make_suggestion(bars, state, execution, self.model, self.scaler)

    def decide(self, date, as_of=None, failpoint=None):
        if as_of is not None and as_of != date+'T15:30:00+08:00':
            raise ValueError('历史 Agent 使用已冻结的 15:30 收盘决策时点')
        suggestion = self.suggest(date)
        return self.choose(date, suggestion['target'])


class AgentTrial(Trial):
    NAMESPACE = 'simulation-agent-v1'
    PROTOCOL = VERSION
    CHECKPOINT = CHECKPOINT
    ACCOUNT_SERVICE = AgentAccount

    def _declined(self, conn, day):
        return conn.execute('SELECT 1 FROM trial_events WHERE id=?',
                            (digest([self.id, day, 'agent-decline']),)).fetchone() is not None

    def validate_choice(self, conn, day, adopt):
        complete = {row[0] for row in conn.execute(
            "SELECT day FROM trial_nodes WHERE trial=? AND status='complete'", (self.id,))}
        next_day = next((d for d in self.nodes[:-1] if d not in complete), None)
        if day != next_day:
            raise ValueError('这条建议已过期，请查看当前卡片')
        if adopt and self._declined(conn, day):
            raise ValueError('这条建议已拒绝，请确认一个自选操作')

    def _card(self, node):
        suggestion = json.loads(node['suggestion'])['user']
        day = node['day']
        with self.store.connect() as conn:
            declined = self._declined(conn, day)
        return dict(id=digest([self.id, day, suggestion]), day=day, facts=suggestion['facts'],
                    status=node['status'], request=node['request'], declined=declined)

    def card(self):
        node = self.prepare()
        return None if node is None else self._card(node)

    def _matching_card(self, day, card_id):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            node = conn.execute('SELECT * FROM trial_nodes WHERE trial=? AND day=?',
                                (self.id, day)).fetchone()
        if not node:
            raise ValueError('这条建议已过期，请查看当前卡片')
        saved = self._card(node)
        if saved['id'] != card_id:
            raise ValueError('建议快照已改变，请重新确认')
        return saved

    def decline(self, card):
        current = self._matching_card(card['day'], card['id'])
        day = current['day']
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            request, status = conn.execute('SELECT request,status FROM trial_nodes WHERE trial=? AND day=?',
                                           (self.id, day)).fetchone()
            if request or status != 'ready':
                raise ValueError('这条建议已经确认或过期')
            if not self._declined(conn, day):
                insert(conn, 'trial_events', dict(id=digest([self.id, day, 'agent-decline']),
                       trial=self.id, day=day, kind='agent_declined',
                       payload=canonical({'card': current['id']}), created=now()))

    def confirm(self, card, target=None, *, adopt=False, failpoint=None):
        current = self._matching_card(card['day'], card['id'])
        if adopt:
            target = current['facts']['target']
        return self.submit(current['day'], target, reason='Agent 采纳' if adopt else 'Agent 自选',
                           adopt=adopt, failpoint=failpoint)

    def receipt(self):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('''
                SELECT e.*, d.target_weight, a.exposure, a.quantity, a.equity
                FROM trial_nodes n
                JOIN decisions d ON d.decision_date=n.day AND d.run_id=?
                JOIN executions e ON e.decision_id=d.decision_id
                JOIN daily_account a ON a.run_id=e.run_id AND a.trade_date=e.execution_date
                WHERE n.trial=? AND n.status='complete'
                ORDER BY n.day DESC LIMIT 1
                ''', (self.id+'-user', self.id)).fetchone()
        return dict(row) if row else None
