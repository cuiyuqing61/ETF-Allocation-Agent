"""FinRL-X StrategyResult adapters; targets always refer to total account equity."""
import pandas as pd
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult


class ETFBaselineStrategy(BaseStrategy):
    def __init__(self, name='cash'):
        if name not in ('cash','buy_hold','ma10_20'):
            raise ValueError(name)
        super().__init__(StrategyConfig(name=name))

    def generate_weights(self, data, target_date=None):
        bars = data['bars'].copy().sort_values('trade_date')
        if target_date is not None:
            bars = bars[bars.trade_date <= str(target_date)]
        close = bars.set_index(pd.to_datetime(bars.trade_date))['close']
        if self.config.name == 'cash':
            signal = pd.Series(0.0,index=close.index)
        elif self.config.name == 'buy_hold':
            # The evaluation slice determines the first buy; later rows are HOLD.
            signal = pd.Series(float('nan'),index=close.index)
            from src.trading.position_execution import Account, execute_target
            from src.trading.corporate_actions import reference_price
            execution=data['execution']
            account=Account(execution['initial_capital'])
            for i in range(len(bars)):
                if account.quantity > 0: break
                signal.iloc[i]=1.0
                if i+1 < len(bars):
                    account,_=execute_target(account,1.0,bars.iloc[i+1].to_dict(),reference_price(bars.iloc[i].close,bars.iloc[i+1].trade_date,data.get('corporate_actions',[])),
                                             execution['market_rules'],execution['costs'])
        else:
            fast, slow = close.rolling(10).mean(),close.rolling(20).mean()
            signal = (fast>slow).astype(float).where(slow.notna())
        return StrategyResult(self.config.name,signal.to_frame('510300'),
                              {'weight_semantics':'total_equity','nan_semantics':'HOLD',
                               'feature_version':'ma10_20_v1','decision_timing':'close'})
