"""Next-open ETF replay, adapted from the FinRL-X backtest interface."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from src.trading.position_execution import Account, execute_target, mark_to_market, D
from src.trading.corporate_actions import CorporateActionLedger, reference_price
from src.data.akshare_provider import validate_bars


@dataclass(frozen=True)
class ReplayConfig:
    start_date: str
    end_date: str
    initial_capital: float
    market_rules: dict
    costs: dict
    corporate_actions: list


@dataclass
class ReplayResult:
    strategy_name: str
    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    daily_account: list
    decisions: list
    corporate_action_events: list


def replay(config, strategy_name, bars, weights):
    b, _ = validate_bars(bars)
    if len(b)<2 or list(weights.columns)!=['510300']:
        raise ValueError('Need a prior decision bar, execution bars and one ETF target column')
    w=weights.copy()
    w.index=pd.to_datetime(w.index).strftime('%Y-%m-%d')
    if w.index.duplicated().any() or not w.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered decisions')
    if not set(w.index).issubset(set(b.trade_date)):
        raise ValueError('Decision date has no bar')
    finite=w['510300'].dropna()
    if not np.isfinite(finite).all() or ((finite<0)|(finite>1)).any():
        raise ValueError('Targets outside [0,1]')
    actions=config.corporate_actions or []
    ledger=CorporateActionLedger(actions,config.start_date)
    for a in actions:
        for field in ('record_date','ex_date'):
            if config.start_date<=a[field]<=config.end_date and a[field] not in set(b.trade_date):
                raise ValueError('Missing corporate action session')
    account=Account(config.initial_capital)
    decision_account=account
    prev_equity=config.initial_capital
    decisions, executions, daily=[] ,[],[]
    trading={b.iloc[i].trade_date:i for i in range(1,len(b)) if config.start_date<=b.iloc[i].trade_date<=config.end_date}
    days=sorted(set(trading)|{a['pay_date'] for a in actions if config.start_date<=a['pay_date']<=config.end_date})
    last_close=float(b.iloc[0].close)
    for date in days:
        opening=account
        decision=None
        if date in trading:
            i=trading[date];previous,bar=b.iloc[i-1],b.iloc[i]
            account=replace(account,sellable=account.quantity)
            value=w.loc[previous.trade_date,'510300'] if previous.trade_date in w.index else np.nan
            target=None if pd.isna(value) else float(value)
            snapshot={'bar':previous.to_dict(),'cash':decision_account.cash,'quantity':decision_account.quantity,'receivable':decision_account.receivable}
            decision_equity=decision_account.cash+decision_account.quantity*previous.close+decision_account.receivable
            decision=dict(decision_date=previous.trade_date,created_at=datetime.now(timezone.utc).isoformat(),
                          data_as_of=previous.availability_at,snapshot_hash=sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest(),
                          current_weight=decision_account.quantity*previous.close/decision_equity,raw_target_weight=target,target_weight=target,
                          planned_execution_date=date,guard_reason='HOLD' if target is None else 'validated')
            account,accrued=ledger.open(account,date)
            reference=reference_price(previous.close,date,actions,config.market_rules['tick_size'])
            account,execution=execute_target(account,target,bar.to_dict(),reference,config.market_rules,config.costs)
            last_close=float(bar.close)
            decisions.append(decision)
            executions.append(dict(execution_date=date,**execution))
        else:
            accrued=0.0;reference=None
            execution=dict(signed_quantity=0,notional=0.0,fee_cash=0.0,slippage_cash=0.0,fill_price=None)
        account,paid=ledger.close(account,date)
        if date in trading: decision_account=account
        valuation=mark_to_market(account,last_close,prev_equity)
        trade_cash=-D(execution['signed_quantity'])*D(execution['fill_price'] or 0)
        daily.append(dict(trade_date=date,**valuation,traded_notional=execution['notional'],
                          fee_cash=execution['fee_cash'],slippage_cash=execution['slippage_cash'],dividend_cash=paid,
                          dividend_accrued=accrued,prev_cash=opening.cash,prev_quantity=opening.quantity,
                          prev_receivable=opening.receivable,trade_cash=float(trade_cash),
                          signed_quantity=execution['signed_quantity'],quantity_change=0,
                          close_price=last_close,reference_price=reference,is_session=int(date in trading),
                          market_regime='unclassified'))
        prev_equity=valuation['equity']
    if not daily: raise ValueError('No executable sessions')
    reconcile(daily,config.initial_capital)
    frame=pd.DataFrame(daily).set_index('trade_date');frame.index=pd.to_datetime(frame.index)
    equity=frame.equity;peak=equity.cummax().clip(lower=config.initial_capital)
    metrics={'total_return':float(equity.iloc[-1]/config.initial_capital-1),
             'max_drawdown':float((1-equity/peak).max()),'fees':float(frame.fee_cash.sum()),
             'dividend_cash':float(frame.dividend_cash.sum()),'dividend_receivable':float(frame.dividend_receivable.iloc[-1])}
    return ReplayResult(strategy_name, frame.net_return, equity, w, pd.DataFrame(executions),
                        metrics, daily, decisions, ledger.events)


def reconcile(daily,initial_capital):
    """Independent accounting identities; raise before persisting inconsistent runs."""
    errors=[];prev=dict(cash=initial_capital,quantity=0,dividend_receivable=0,equity=initial_capital)
    for d in daily:
        expectations={
            'prev_cash':prev['cash'],'prev_quantity':prev['quantity'],'prev_receivable':prev['dividend_receivable'],
            'prev_equity':prev['equity'],
            'cash':d['prev_cash']+d['trade_cash']-d['fee_cash']+d['dividend_cash'],
            'quantity':d['prev_quantity']+d['signed_quantity']+d['quantity_change'],
            'dividend_receivable':d['prev_receivable']+d['dividend_accrued']-d['dividend_cash'],
            'position_value':d['quantity']*d['close_price'],
            'equity':d['cash']+d['position_value']+d['dividend_receivable'],
            'net_return':d['equity']/d['prev_equity']-1}
        for key,value in expectations.items():
            if not np.isfinite(d[key]) or abs(d[key]-value)>1e-7: errors.append((d['trade_date'],key,d[key],value))
        prev=d
    if errors: raise ValueError(f'Ledger reconciliation failed: {errors[:5]}')
    return {'rows':len(daily),'failed_identities':0,'tolerance':1e-7}
