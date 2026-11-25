"""Record-close entitlement, ex-open receivable, pay-close cash ledger.

Cash availability within pay day is not known from announcements. Conservatively
credit after trading, available to orders from the following session. Entitlements
survive sale. No reinvestment is forced. All prices remain unadjusted.
"""
from dataclasses import replace
from decimal import ROUND_HALF_UP
import math
from src.trading.position_execution import D


def validate_actions(actions):
    ids=set()
    for a in actions:
        if a['action_id'] in ids: raise ValueError('Duplicate corporate action')
        ids.add(a['action_id'])
        if a.get('instrument','510300') != '510300': raise ValueError('Wrong action instrument')
        if not a['record_date'] < a['ex_date'] <= a['pay_date']:
            raise ValueError('Invalid corporate action chronology')
        if a.get('announcement_date',a['record_date']) > a['record_date']:
            raise ValueError('Late announcement requires manual review')
        if not math.isfinite(float(a['cash_per_unit'])) or a['cash_per_unit'] < 0:
            raise ValueError('Invalid dividend')
        if a.get('split_ratio',1) != 1:
            raise ValueError('Share conversion requires verified allocation/rounding rules; blocked')
    return actions


def reference_price(previous_close,date,actions,tick=.001):
    cash=sum((D(a['cash_per_unit']) for a in actions if a['ex_date']==date),D(0))
    value=((D(previous_close)-cash)/D(tick)).to_integral_value(rounding=ROUND_HALF_UP)*D(tick)
    if value<=0: raise ValueError('Nonpositive ex-dividend reference')
    return float(value)


class CorporateActionLedger:
    def __init__(self,actions,start):
        self.actions=validate_actions(actions)
        # Every new historical account starts in cash with no carried-in rights.
        self.entitlements={a['action_id']:0 for a in actions if a['record_date']<start}
        self.applied=set()
        self.events=[]

    def _event(self,a,date,phase,quantity,amount):
        key=(a['action_id'],phase)
        if key in self.applied: raise ValueError('Corporate action phase applied twice')
        self.applied.add(key)
        self.events.append(dict(action_id=a['action_id'],event_date=date,phase=phase,
                                entitled_quantity=quantity,amount=float(amount)))

    def open(self,account,date):
        accrued=D(0)
        for a in self.actions:
            if a['ex_date']==date:
                if a['action_id'] not in self.entitlements: raise ValueError('Missing record-date holdings')
                qty=self.entitlements[a['action_id']]
                amount=(D(qty)*D(a['cash_per_unit'])).quantize(D('.01'),rounding=ROUND_HALF_UP)
                accrued+=amount
                self._event(a,date,'ex',qty,amount)
        return replace(account,receivable=float(D(account.receivable)+accrued)),float(accrued)

    def close(self,account,date):
        paid=D(0)
        for a in self.actions:
            if a['record_date']==date:
                qty=account.quantity
                self.entitlements[a['action_id']]=qty
                amount=(D(qty)*D(a['cash_per_unit'])).quantize(D('.01'),rounding=ROUND_HALF_UP)
                self._event(a,date,'record',qty,amount)
            if a['pay_date']==date:
                qty=self.entitlements.get(a['action_id'])
                if qty is None: raise ValueError('Missing entitlement for payment')
                amount=(D(qty)*D(a['cash_per_unit'])).quantize(D('.01'),rounding=ROUND_HALF_UP)
                if amount and (a['action_id'],'ex') not in self.applied:
                    raise ValueError('Payment without accrued receivable')
                paid+=amount
                self._event(a,date,'pay',qty,amount)
        remaining=D(account.receivable)-paid
        if remaining<0: raise ValueError('Negative receivable')
        return replace(account,cash=float(D(account.cash)+paid),receivable=float(remaining)),float(paid)
