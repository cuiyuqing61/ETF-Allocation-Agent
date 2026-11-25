"""Explicit domestic ETF provider; never fabricates prices or falls back to US data."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.trading_calendar import get_trading_days_set


def validate_bars(bars, start=None, end=None):
    required = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
    if bars.empty or not set(required).issubset(bars.columns):
        raise ValueError('Empty data or missing required OHLCV columns')
    b = bars.copy()
    b['trade_date'] = pd.to_datetime(b.trade_date, errors='raise').dt.strftime('%Y-%m-%d')
    if b.trade_date.duplicated().any():
        raise ValueError('Duplicate trading dates')
    for c in required[1:]:
        b[c] = pd.to_numeric(b[c], errors='raise').astype(float)
    if not np.isfinite(b[required[1:]].to_numpy()).all():
        raise ValueError('Nonfinite OHLCV')
    if (b[['open', 'high', 'low', 'close']] <= 0).any().any() or (b.volume < 0).any():
        raise ValueError('Invalid price or volume')
    if ((b.high < b[['open','close','low']].max(axis=1)) | (b.low > b[['open','close','high']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC')
    b = b.sort_values('trade_date').reset_index(drop=True)
    first, last = start or b.trade_date.iloc[0], end or b.trade_date.iloc[-1]
    expected = get_trading_days_set(first, last, 'SSE')
    actual = set(b.trade_date)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(f'SSE calendar mismatch: missing={missing}, extra={extra}')
    if 'tradable' in b and not b.tradable.isin([0,1,False,True]).all():
        raise ValueError('Invalid tradability flag')
    b['tradable'] = ((b.volume > 0) & b.get('tradable',True).astype(bool)).astype(int) if 'tradable' in b else (b.volume > 0).astype(int)
    b['availability_at'] = b.trade_date + 'T15:30:00+08:00'
    return b, {'rows': len(b), 'duplicates': 0, 'missing_sessions': missing,
               'non_sessions': extra, 'zero_volume': int((b.volume == 0).sum()),
               'exchange': 'SSE', 'volume_unit': 'shares', 'adjustment': 'none',
               'return_basis': 'price_only_excluding_dividends',
               'availability_semantics': 'historical_assumption_not_point_in_time_capture'}


class AkshareDataProvider:
    def __init__(self, cache_dir='data/position_agent'):
        self.base_dir = Path(cache_dir)

    def is_available(self):
        return importlib.util.find_spec('akshare') is not None

    def fetch_snapshot(self, instrument, start, end):
        if instrument != '510300':
            raise ValueError('Frozen v1 universe contains only 510300')
        import akshare as ak
        request = {'symbol': instrument, 'period': 'daily', 'start_date': start.replace('-', ''),
                   'end_date': end.replace('-', ''), 'adjust': ''}
        raw = ak.fund_etf_hist_em(**request)
        fetched_at = datetime.now(timezone.utc).isoformat()
        # Preserve the unmodified provider DataFrame, not a claim of HTTP wire bytes.
        payload = raw.to_json(orient='table', date_format='iso', force_ascii=False).encode()
        raw_hash = hashlib.sha256(payload).hexdigest()
        provenance = {'source': 'akshare.fund_etf_hist_em', 'provider_version': ak.__version__,
                      'request': request, 'raw_hash': raw_hash}
        dataset_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        raw_dir = self.base_dir / 'raw' / dataset_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / 'provider_frame.json'
        if raw_file.exists() and raw_file.read_bytes() != payload:
            raise ValueError('Immutable raw snapshot conflict')
        if not raw_file.exists():
            raw_file.write_bytes(payload)
            (raw_dir / 'provenance.json').write_text(json.dumps({**provenance, 'fetched_at': fetched_at}, indent=2))
        columns = {'日期':'trade_date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'}
        if not set(columns).issubset(raw.columns):
            raise ValueError(f'Missing AKShare columns; raw preserved at {raw_file}')
        bars = raw[list(columns)].rename(columns=columns)
        bars['volume'] = pd.to_numeric(bars.volume, errors='raise') * 100  # EM lots -> shares
        # Check unit conversion against raw traded amount, not a guessed constant.
        if '成交额' not in raw:
            raise ValueError('Cannot verify volume units without raw traded amount')
        amount = pd.to_numeric(raw['成交额'], errors='raise')
        positive = bars.volume > 0
        vwap = amount[positive] / bars.loc[positive, 'volume']
        if not np.isfinite(vwap).all() or ((vwap < pd.to_numeric(bars.loc[positive,'low']) - .002) | (vwap > pd.to_numeric(bars.loc[positive,'high']) + .002)).any():
            raise ValueError('Volume unit / amount check failed; raw snapshot preserved')
        try:
            bars, quality = validate_bars(bars, start, end)
        except Exception as exc:
            (raw_dir / 'quality_failure.json').write_text(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
            raise
        metadata = {**provenance, 'dataset_id': dataset_id, 'fetched_at': fetched_at,
                    'instrument': instrument, 'start_date': start, 'end_date': end,
                    'raw_path': str(raw_file.resolve()), 'adjustment_policy': 'none',
                    'corporate_actions_version': 'unavailable', 'quality': quality}
        return metadata, bars

    def get_price_data(self, tickers, start_date, end_date):
        if list(tickers) != ['510300']:
            raise ValueError('Pass the frozen ticker list ["510300"]')
        from src.data.data_store import DataStore
        metadata, bars = self.fetch_snapshot('510300', start_date, end_date)
        DataStore(str(self.base_dir)).save_position_dataset(metadata, bars)
        result = bars.rename(columns={'trade_date':'datadate','open':'prcod','high':'prchd',
                                      'low':'prcld','close':'prccd','volume':'cshtrd'}).copy()
        result['tic'] = result['gvkey'] = '510300'
        result['adj_close'] = result.prccd  # explicitly UNADJUSTED in metadata
        result.attrs.update(metadata)
        return result

    def get_sp500_components(self, *args, **kwargs):
        raise NotImplementedError('Domestic ETF provider has no S&P 500 universe')

    def get_fundamental_data(self, *args, **kwargs):
        raise NotImplementedError('ETF provider does not supply stock fundamentals')
