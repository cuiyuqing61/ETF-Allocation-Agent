"""ETF target-weight environment for PPO."""
from dataclasses import dataclass, replace
import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.data.akshare_provider import validate_bars
from src.trading.position_execution import Account, execute_target, mark_to_market
from src.trading.corporate_actions import CorporateActionLedger, reference_price

FEATURES=['return_1','return_5','return_20','relative_ma10','relative_ma20','volatility20','relative_volume20']
OBSERVATIONS=FEATURES+['exposure','cash_fraction','drawdown']


def features(bars):
    c=bars.close.astype(float);v=bars.volume.astype(float)
    returns=c.pct_change()
    return pd.DataFrame({'return_1':returns,'return_5':c.pct_change(5),'return_20':c.pct_change(20),
                         'relative_ma10':c/c.rolling(10).mean()-1,'relative_ma20':c/c.rolling(20).mean()-1,
                         'volatility20':returns.rolling(20).std(),'relative_volume20':v/v.rolling(20).mean()-1})[FEATURES]


@dataclass
class FrozenScaler:
    mean: list
    scale: list
    feature_order: list
    train_start: str
    train_end: str

    @classmethod
    def fit(cls,bars,start,end):
        x=features(bars)
        selected=x[(bars.trade_date>=start)&(bars.trade_date<=end)].dropna()
        if len(selected)<20 or bars.trade_date.max()>end:
            raise ValueError('Fit accepts only train-bounded data with sufficient warmup')
        scale=selected.std(ddof=0).replace(0,1)
        return cls(selected.mean().tolist(),scale.tolist(),FEATURES.copy(),start,end)

    def transform(self,row):
        if self.feature_order!=FEATURES or len(self.mean)!=7 or len(self.scale)!=7:
            raise ValueError('Scaler feature order mismatch')
        scale=np.asarray(self.scale)
        if not np.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Invalid scaler')
        out=(np.asarray(row,dtype=float)-np.asarray(self.mean))/scale
        if not np.isfinite(out).all(): raise ValueError('Nonfinite feature/scaler')
        return out


class TargetWeightEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,bars,scaler,rules,costs,initial_capital=100000,start=None,end=None,corporate_actions=None):
        self.actions=corporate_actions or []
        bounded=bars[pd.to_datetime(bars.trade_date)<=pd.Timestamp(end)] if end else bars
        self.bars,_=validate_bars(bounded)
        self.scaler=scaler;self.rules=dict(rules);self.costs=dict(costs)
        self.initial_capital=float(initial_capital)
        self.raw_features=features(self.bars)
        valid=self.raw_features.notna().all(axis=1)
        if start:
            # start denotes the FIRST EXECUTION session, just as in BacktestEngine.
            executions=np.flatnonzero((self.bars.trade_date>=start)&(self.bars.trade_date<=(end or self.bars.trade_date.iloc[-1])))
            if not len(executions) or executions[0]==0: raise ValueError('Need a prior decision session')
            self.first=int(executions[0])-1;self.last=int(executions[-1])
            if not valid.iloc[self.first:self.last+1].all(): raise ValueError('Insufficient feature warmup')
        else:
            if end: valid &= self.bars.trade_date<=end
            ids=np.flatnonzero(valid)
            if len(ids)<2 or not np.all(np.diff(ids)==1): raise ValueError('Insufficient/noncontiguous feature history')
            self.first=int(ids[0]);self.last=int(ids[-1])
        if self.first>=self.last: raise ValueError('No executable sessions')
        self.bar_records=self.bars.to_dict('records')
        self.closes=self.bars.close.to_numpy(dtype=float)
        self.scaled_features=np.array([self.scaler.transform(self.raw_features.iloc[i]) for i in range(self.first,self.last+1)])
        self.action_space=spaces.Box(0.,1.,shape=(1,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(10,),dtype=np.float32)
        self.reset()

    def observation(self):
        if self.scaler.feature_order!=FEATURES: raise ValueError('Scaler feature order mismatch')
        close=float(self.closes[self.i])
        position=self.account.quantity*close
        extra=[position/self.equity,self.account.cash/self.equity,1-self.equity/self.peak]
        return np.asarray([*self.scaled_features[self.i-self.first],*extra],dtype=np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.i=self.first;self.account=Account(self.initial_capital)
        self.ledger=CorporateActionLedger(self.actions,self.bars.iloc[self.first+1].trade_date)
        self.equity=self.peak=self.initial_capital
        self.terminated=False
        return self.observation(),{}

    def step(self,action):
        if self.terminated: raise RuntimeError('Episode finished; call reset')
        value=np.asarray(action)
        if value.shape!=(1,) or not self.action_space.contains(value.astype(np.float32)):
            raise ValueError('Action must be one finite weight in [0,1]')
        return self._advance(float(value[0]))

    def hold_shares(self):
        if self.terminated: raise RuntimeError('Episode finished; call reset')
        return self._advance(None)

    def _advance(self, target):
        previous=self.bar_records[self.i]
        next_bar=self.bar_records[self.i+1]
        account=replace(self.account,sellable=self.account.quantity)
        # Non-session payment dates retain the prior close mark; entitlement
        # operations use the very same ledger as historical replay.
        for day in sorted({a['pay_date'] for a in self.actions if previous['trade_date']<a['pay_date']<next_bar['trade_date']}):
            account,_=self.ledger.close(account,day)
        account,accrued=self.ledger.open(account,next_bar['trade_date'])
        reference=reference_price(previous['close'],next_bar['trade_date'],self.actions,self.rules['tick_size'])
        self.account,execution=execute_target(account,target,next_bar,reference,self.rules,self.costs)
        self.account,paid=self.ledger.close(self.account,next_bar['trade_date'])
        daily=mark_to_market(self.account,next_bar['close'],self.equity)
        daily.update(dividend_cash=paid,dividend_accrued=accrued)
        reward=math.log(daily['equity']/self.equity)
        self.equity=daily['equity'];self.peak=max(self.peak,self.equity)
        self.i+=1;self.terminated=self.i>=self.last
        return self.observation(),reward,self.terminated,False,{'execution':execution,'daily':daily}
