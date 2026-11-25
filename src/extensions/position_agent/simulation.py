"""Durable two-account simulation, using DailyService settlement unchanged."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from src.extensions.position_agent.daily import DailyService, ROOT, canonical, digest, now, insert, sessions
from src.extensions.position_agent.model_versions import revision_matches
from src.trading.position_execution import Account

DB=ROOT/'data/demo/finrl_trading.db'
SCHEMA='''
CREATE TABLE IF NOT EXISTS trial_sessions(id TEXT PRIMARY KEY, config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trial_nodes(trial TEXT NOT NULL REFERENCES trial_sessions(id), day TEXT NOT NULL,
 next_day TEXT NOT NULL, suggestion TEXT NOT NULL, request TEXT, status TEXT NOT NULL DEFAULT 'ready',
 created TEXT NOT NULL, finished TEXT, PRIMARY KEY(trial,day));
CREATE TABLE IF NOT EXISTS trial_events(id TEXT PRIMARY KEY, trial TEXT NOT NULL, day TEXT, kind TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
CREATE VIEW IF NOT EXISTS trial_diagnostics AS SELECT t.id,t.config,n.day,n.next_day,n.status,n.request,n.suggestion,n.created,n.finished FROM trial_sessions t JOIN trial_nodes n ON n.trial=t.id;
'''
class Store:
    def __init__(self,path): self.db_path=Path(path)
    def connect(self):
        c=sqlite3.connect(self.db_path.as_uri()+'?mode=rw',uri=True,timeout=30)
        c.execute('PRAGMA foreign_keys=ON');return c

def initialize(path=DB):
    with Store(path).connect() as c:c.executescript(SCHEMA)

class AccountService(DailyService):
    def suggest(self,date):
        with self.store.connect() as c:
            state=json.loads(c.execute('SELECT state_json FROM daily_state WHERE run_id=?',(self.run_id,)).fetchone()[0])
            if state['date']!=date:raise ValueError('账户尚未到达操作日')
            b=self._bars(c,date,date+'T15:30:00+08:00')
        a=Account(**state['account']);eq=state['equity']
        obs = self.policy_observation(b, a, eq, state['peak'])
        value=float(self.model.predict(obs,deterministic=True)[0][0])
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('模型输出不合法')
        return dict(target=value,observation=obs.tolist(),state=state,bars=b.to_dict('records'))
    def choose(self,date,target):
        def work(c,state):
            old=c.execute('SELECT decision_id,target_weight FROM decisions WHERE run_id=? AND decision_date=?',(self.run_id,date)).fetchone()
            if old:
                if old[1]!=target:raise ValueError('既有决策冲突')
                return old[0],True
            if state['date']!=date:raise ValueError('决策顺序错误')
            b=self._bars(c,date,date+'T15:30:00+08:00');a=Account(**state['account'])
            snapshot=dict(bars=b.to_dict('records'),account_state=state,artifact_identity=self.cfg['artifact_identity'])
            did=digest([self.run_id,date]);following=next(d for d in self.cfg['sessions'] if d>date)
            insert(c,'decisions',dict(decision_id=did,run_id=self.run_id,decision_date=date,created_at=now(),data_as_of=date+'T15:30:00+08:00',snapshot_hash=digest(snapshot),current_weight=a.quantity*float(b.iloc[-1].close)/state['equity'],raw_target_weight=target,target_weight=target,planned_execution_date=following,guard_reason='trial_HOLD' if target is None else 'trial_target'))
            insert(c,'daily_inputs',dict(decision_id=did,account_version=state['version'],snapshot_json=canonical(snapshot)))
            return did,False
        return self._job('decide',date,work)

class Trial:
    NAMESPACE = 'simulation-v1'
    PROTOCOL = 'simulation-transformer20-v1'
    CHECKPOINT = None
    ACCOUNT_SERVICE = AccountService

    @classmethod
    def create(cls,participant,round_id,start,end,frequency,path=DB,origin='participant'):
        import re
        if any(not re.fullmatch('[A-Za-z0-9_-]{1,32}',x) for x in (participant,round_id)):raise ValueError('参与码和轮次限字母、数字、下划线和短横线')
        if frequency not in ('daily','weekly') or origin not in ('participant','automated_test'):raise ValueError('配置不合法')
        if not '2022-01-01'<=start<=end<='2024-12-31':raise ValueError('只支持 2022—2024 区间')
        days=sessions(start,end)
        if not days:raise ValueError('区间内没有交易日')
        initialize(path);store=Store(path)
        tid=digest([participant,round_id,cls.NAMESPACE])
        with store.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            old=c.execute('SELECT config FROM trial_sessions WHERE id=?',(tid,)).fetchone()
            saved = json.loads(old[0]) if old else {}
            cfg = dict(participant=participant, round=round_id, origin=origin, start=days[0],
                       end=days[-1], frequency=frequency,
                       checkpoint=saved.get('checkpoint', str(cls.CHECKPOINT)),
                       protocol=saved.get('protocol', cls.PROTOCOL),
                       service_hash=saved.get('service_hash', digest(Path(__file__).read_text())))
            if not Path(cfg['checkpoint'], 'manifest.json').is_file():
                raise ValueError('找不到本轮模型，请先完成 20 日 Transformer 训练或恢复原模型文件')
            if old and old[0]!=canonical(cfg):raise ValueError('本轮配置已经冻结，请恢复原试用或使用新轮次')
            c.execute('INSERT OR IGNORE INTO trial_sessions VALUES(?,?)',(tid,canonical(cfg)))
        for role in ('user','model'):
            with store.connect() as c: exists=c.execute('SELECT 1 FROM daily_sessions WHERE run_id=?',(tid+'-'+role,)).fetchone()
            if not exists:cls.ACCOUNT_SERVICE.create(store,tid+'-'+role,cfg['checkpoint'],cfg['start'],cfg['end'])
        return cls(tid,path)
    def __init__(self,tid,path=DB):
        self.id=tid;self.store=Store(path)
        with self.store.connect() as c:
            self.cfg=json.loads(c.execute('SELECT config FROM trial_sessions WHERE id=?',(tid,)).fetchone()[0])
        if not revision_matches('simulation', self.cfg['service_hash'], digest(Path(__file__).read_text())):raise ValueError('服务版本已改变，需显式迁移或新开轮次')
        self.accounts={r:self.ACCOUNT_SERVICE(self.store,tid+'-'+r) for r in ('user','model')}
        cfg=self.accounts['user'].cfg;days=cfg['sessions'];self.prior=cfg['prior']
        self.nodes=[self.prior]+[d for i,d in enumerate(days[:-1]) if self.cfg['frequency']=='daily' or pd.Timestamp(d).isocalendar()[:2]!=pd.Timestamp(days[i+1]).isocalendar()[:2]]+[days[-1]]
    def current(self):
        with self.store.connect() as c:
            done={r[0] for r in c.execute("SELECT day FROM trial_nodes WHERE trial=? AND status='complete'",(self.id,))}
        return next((d for d in self.nodes[:-1] if d not in done),self.nodes[-1])
    def prepare(self):
        day=self.current()
        if day==self.nodes[-1]:return None
        with self.store.connect() as c:
            c.row_factory=sqlite3.Row;row=c.execute('SELECT * FROM trial_nodes WHERE trial=? AND day=?',(self.id,day)).fetchone()
        if row:return dict(row)
        suggestion={r:a.suggest(day) for r,a in self.accounts.items()}
        with self.store.connect() as c:
            c.execute('INSERT OR IGNORE INTO trial_nodes(trial,day,next_day,suggestion,created) VALUES(?,?,?,?,?)',(self.id,day,self.nodes[self.nodes.index(day)+1],canonical(suggestion),now()))
        return self.prepare()
    def validate_choice(self, conn, day, adopt):
        """Optional policy-specific check inside the same write transaction."""

    def submit(self,day,target,reason='',adopt=False,failpoint=None):
        if target is not None and (not np.isfinite(target) or not 0<=target<=1):raise ValueError('仓位必须在 0%—100%')
        if len(reason)>2000:raise ValueError('备注过长')
        request=canonical(dict(target=target,reason=reason,adopt=bool(adopt)))
        with self.store.connect() as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT suggestion,request,status,next_day FROM trial_nodes WHERE trial=? AND day=?',(self.id,day)).fetchone()
            if not row:raise ValueError('请先查看当前节点建议')
            if row[1] and row[1]!=request:raise ValueError('本节点已提交不同选择，不能覆盖')
            if row[2]=='complete':return True
            if adopt and target!=json.loads(row[0])['user']['target']:raise ValueError('采纳标记与建议不一致')
            self.validate_choice(c, day, adopt)
            c.execute("UPDATE trial_nodes SET request=?,status='processing' WHERE trial=? AND day=?",(request,self.id,day))
            c.execute('INSERT OR IGNORE INTO trial_events VALUES(?,?,?,?,?,?)',(digest([self.id,day,'submit']),self.id,day,'submitted',request,now()))
        if failpoint=='after_request':raise RuntimeError('Injected after request')
        for role,service in self.accounts.items():
            for date in [d for d in service.cfg['sessions'] if day<d<=row[3]]:
                state=service.status()['state']
                if state['date']>=date:continue
                weight=(target if role=='user' else json.loads(row[0])['model']['target']) if state['date']==day else None
                service.choose(state['date'],weight);service.settle(date)
            if failpoint=='after_user' and role=='user':raise RuntimeError('Injected after user settlement')
        with self.store.connect() as c:c.execute("UPDATE trial_nodes SET status='complete',finished=COALESCE(finished,?) WHERE trial=? AND day=?",(now(),self.id,day))
        if failpoint=='after_commit':raise RuntimeError('Injected response loss')
        return False
    def view(self):
        day=self.current()
        with self.store.connect() as c:
            bars=pd.read_sql_query("SELECT trade_date,raw_open AS open,raw_high AS high,raw_low AS low,raw_close AS close,volume FROM market_bars WHERE dataset_id=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 250",c,params=(self.accounts['user'].cfg['dataset_id'],day)).iloc[::-1]
            daily=pd.read_sql_query('SELECT * FROM daily_account WHERE run_id IN (?,?) AND trade_date<=? ORDER BY trade_date',c,params=(self.id+'-user',self.id+'-model',day))
            trades=pd.read_sql_query('SELECT * FROM executions WHERE run_id IN (?,?) AND execution_date<=? ORDER BY execution_date',c,params=(self.id+'-user',self.id+'-model',day))
        return day,bars,daily,trades
