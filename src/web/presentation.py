"""Display-only translations and calculations over committed account records."""
import pandas as pd

REASONS={
 'target':'按模型保存的目标仓位调整。',
 'HOLD':'建议保持原有份额，因此没有主动买卖。',
 'target_within_lot':'仓位差异不足一手（100 份），按交易单位取整后无需买卖。',
 'untradable':'当日记录显示不可交易或成交量为零。',
 'limit_open':'开盘价格触及本模拟规则的买卖限制。',
 'slippage_outside_limit':'计入滑点后的成交价超出价格限制。',
 'cash_limited':'可用现金不足以达到完整目标，按实际可买数量处理。',
 'sellable_limited':'可卖份额受到限制，按实际可卖数量处理。',
 'fee_exceeds_cash':'可用现金不足以支付这笔交易的费用。',
}


def recommendation_text(d):
    current=float(d['current_weight']);target=d['target_weight']
    if pd.isna(target):
        return f"模型建议保持原有 ETF 份额，不主动买卖。当时 ETF 占账户总资产的 {current:.1%}。"
    target=float(target);delta=100*(target-current)
    direction='提高' if delta>0 else '降低' if delta<0 else '维持'
    change=f"，{'增加' if delta>0 else '减少'} {abs(delta):.1f} 个百分点" if delta else ''
    return f"模型计划将 ETF 占总资产的比例从 {current:.1%} {direction}到 {target:.1%}{change}。"


def execution_text(e):
    if e is None:return '还没有这条建议的结算结果。是否成交、成交价格和费用均尚未确定。'
    qty=int(e['signed_quantity']);reason=REASONS.get(e['reason'],f"记录原因：{e['reason']}")
    if qty:
        return f"{e['execution_date']} 模拟{'买入' if qty>0 else '卖出'} {abs(qty):,} 份 ETF，每份成交价 {e['fill_price']:.3f} 元，交易手续费 {e['fee_cash']:.2f} 元。{reason}"
    return f"{e['execution_date']} 没有成交。{reason}"


def latest_account(records,service):
    if service:
        s=service['state'];a=s['account']
        return dict(date=s['date'],equity=s['equity'],cash=a['cash'],quantity=a['quantity'],
                    receivable=a['receivable'],position_value=s['equity']-a['cash']-a['receivable'])
    if records.empty:return None
    r=records.sort_values('trade_date').iloc[-1]
    return dict(date=r.trade_date,equity=r.equity,cash=r.cash,quantity=int(r.quantity),
                receivable=r.dividend_receivable,position_value=r.position_value)


def comparison_contract(config):
    # Missing costs/rules are not evidence of matching assumptions.
    rules=config.get('market_rules',config.get('rules'))
    costs=config.get('costs')
    if rules is None or costs is None:return None
    return dict(rules=rules,costs=costs,
                return_basis=config.get('return_basis','cash_dividends_accrual' if config.get('actions_version') else None),
                actions=config.get('corporate_actions_version',config.get('actions_version')))
