"""Single-ETF simulation shared by backtest and the RL adapter. No broker calls."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math


def D(value):
    return Decimal(str(value))


@dataclass(frozen=True)
class Account:
    cash: float
    quantity: int = 0
    sellable: int = 0
    receivable: float = 0.0


def execute_target(account, target, bar, previous_close, rules, costs):
    """At next open, size from pretrade open equity; None means HOLD shares.

    Daily volume is used only for retrospective no-fill eligibility; it is never
    observed by a strategy. A daily bar cannot prove intraday queue priority.
    """
    if not all(math.isfinite(x) for x in (account.cash,account.receivable)) or account.receivable < 0 or account.cash < 0 or account.quantity < 0 or not 0 <= account.sellable <= account.quantity:
        raise ValueError('Invalid account')
    if target is not None and (not math.isfinite(float(target)) or not 0 <= target <= 1):
        raise ValueError('Target must be finite and in [0,1], or None for HOLD')
    for key in ('open','close','volume'):
        if not math.isfinite(float(bar[key])):
            raise ValueError('Missing/nonfinite execution bar')
    if bar['open'] <= 0 or bar['close'] <= 0 or previous_close <= 0 or bar['volume'] < 0:
        raise ValueError('Invalid bar')
    if any(not math.isfinite(float(costs[k])) or costs[k] < 0 for k in ('fee_rate','minimum_fee','slippage_rate')):
        raise ValueError('Invalid costs')
    if costs.get('stamp_tax_rate',0) != 0:
        raise ValueError('This ETF adapter requires the frozen zero stamp-tax assumption')
    op, cash = D(bar['open']), D(account.cash)
    result = dict(reference_open=float(op), fill_price=None, signed_quantity=0,
                  notional=0.0, fee_cash=0.0, slippage_cash=0.0, status='no_fill', reason='HOLD')
    if target is None:
        return account, result
    lot = int(rules['buy_lot'])
    if lot <= 0 or int(rules['sell_lot']) != lot or D(rules['tick_size']) <= 0:
        raise ValueError('Unsupported market units')
    desired = int((D(target)*(cash+D(account.quantity)*op+D(account.receivable))/op/D(lot)).to_integral_value(rounding=ROUND_FLOOR))*lot
    delta = desired-account.quantity
    if not delta:
        return account, {**result,'reason':'target_within_lot'}
    if not bar.get('tradable',True) or bar['volume'] == 0:
        return account, {**result,'reason':'untradable'}
    tick, limit = D(rules['tick_size']), D(rules['price_limit_fraction'])
    upper = (D(previous_close)*(1+limit)/tick).to_integral_value(rounding=ROUND_HALF_UP)*tick
    lower = (D(previous_close)*(1-limit)/tick).to_integral_value(rounding=ROUND_HALF_UP)*tick
    if (delta > 0 and op >= upper) or (delta < 0 and op <= lower):
        return account, {**result,'reason':'limit_open'}
    side = 1 if delta > 0 else -1
    raw_fill = op*(1+D(side)*D(costs['slippage_rate']))
    fill = (raw_fill/tick).to_integral_value(rounding=ROUND_CEILING if side>0 else ROUND_FLOOR)*tick
    if not lower <= fill <= upper or fill <= 0:
        return account, {**result,'reason':'slippage_outside_limit'}
    def fee(q):
        return max(D(costs['minimum_fee']),abs(D(q))*fill*D(costs['fee_rate'])).quantize(D('0.01'),rounding=ROUND_HALF_UP) if q else D(0)
    reason = 'target'
    if side > 0:
        affordable = max(0,int((cash/(fill*(1+D(costs['fee_rate'])))/D(lot)).to_integral_value(rounding=ROUND_FLOOR))*lot)
        delta = min(delta,affordable)
        while delta > 0 and D(delta)*fill+fee(delta)>cash:
            delta -= lot
        if delta < desired-account.quantity:
            reason = 'cash_limited'
    else:
        delta = -min(-delta,account.sellable)
        # Odd-lot liquidation is supported only when selling the complete remainder.
        if -delta != account.quantity:
            delta = -(abs(delta)//lot)*lot
        if -delta < account.quantity-desired:
            reason = 'sellable_limited'
    if not delta:
        return account,{**result,'reason':reason}
    charged = fee(delta)
    new_cash = cash-D(delta)*fill-charged
    if new_cash < 0:  # Tiny sale whose minimum fee exceeds available cash.
        return account,{**result,'reason':'fee_exceeds_cash'}
    new_quantity = account.quantity+delta
    updated = Account(float(new_cash),new_quantity,account.sellable+min(delta,0),account.receivable)
    return updated, dict(reference_open=float(op),fill_price=float(fill),signed_quantity=delta,
                         notional=float(abs(D(delta))*fill),fee_cash=float(charged),
                         slippage_cash=float(abs(D(delta))*abs(fill-op)),status='filled',reason=reason)


def mark_to_market(account, close, previous_equity):
    position = D(account.quantity)*D(close)
    equity = D(account.cash)+position+D(account.receivable)
    if equity <= 0 or previous_equity <= 0:
        raise ValueError('Nonpositive equity')
    return dict(prev_equity=previous_equity,equity=float(equity),cash=account.cash,
                quantity=account.quantity,position_value=float(position),dividend_receivable=account.receivable,
                net_return=float(equity/D(previous_equity)-1),exposure=float(position/equity))
