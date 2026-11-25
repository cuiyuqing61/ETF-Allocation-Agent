"""Serving adapter for the checkpoint frozen before the 2025 holdout."""
from functools import lru_cache
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from src.extensions.position_agent.model_versions import ROOT
from src.extensions.position_agent.policy_rules import SETTINGS, VARIANT, planned_action
from src.strategies.etf_ppo import FrozenScaler, features
from src.strategies.etf_transformer import ETFTransformerExtractor, OBS_SIZE, OBS_SPEC, observation
from src.trading.position_execution import Account

SEED = 11  # Highest development-period return, before the holdout was opened.
DIRECTORY = ROOT / 'models' / VARIANT
CHECKPOINT = DIRECTORY / f'seed{SEED}'
PROTOCOL = DIRECTORY / 'independent-2025/protocol.json'
VERSION = 'agent-trend-sized-10bp-seed11-v1'


@lru_cache(maxsize=1)
def _load(weights_hash):
    torch.set_num_threads(1)
    model = PPO.load(CHECKPOINT / 'model.zip', device='cpu')
    if (model.seed != SEED or model.num_timesteps != SETTINGS.steps
            or model.observation_space.shape != (OBS_SIZE,) or model.action_space.shape != (1,)
            or not isinstance(model.policy.features_extractor, ETFTransformerExtractor)):
        raise ValueError('检查点模型身份与冻结方案不符')
    return model


def load_release(directory=CHECKPOINT):
    if Path(directory).resolve() != CHECKPOINT.resolve():
        raise ValueError('找不到本轮对应的已冻结版本')
    protocol = json.loads(PROTOCOL.read_text())
    hashes = protocol['checkpoints'][str(SEED)]
    manifest = json.loads((CHECKPOINT / 'manifest.json').read_text())
    if hashes != manifest or set(hashes) != {'config.json', 'scaler.json', 'model.zip'}:
        raise ValueError('冻结清单与检查点不符')
    for filename, expected in hashes.items():
        if sha256((CHECKPOINT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'冻结文件校验失败：{filename}')
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    base = config['frozen_base_plan']
    if (config['variant'] != VARIANT or config['observation'] != OBS_SPEC
            or config['settings'] != protocol['settings'] or base['market_rules'] != protocol['market_rules']
            or base['costs'] != protocol['costs'] or base['dataset_id'] != protocol['dataset']['dataset_id']):
        raise ValueError('策略或成交口径与冻结测试不符')
    if asdict(SETTINGS) != config['settings']:
        raise ValueError('运行策略参数与冻结测试不符')
    scaler = FrozenScaler(**json.loads((CHECKPOINT / 'scaler.json').read_text()))
    return _load(hashes['model.zip']), scaler, base, hashes


def make_suggestion(bars, state, execution_date, model, scaler):
    history = SETTINGS.trend_sessions
    if len(bars) < history or bars.trade_date.iloc[-1] != state['date']:
        raise ValueError('缺少决策当日或最近 50 个交易日的行情')
    last = bars.iloc[-1]
    close = float(last.close)
    average = float(bars.close.rolling(history).mean().iloc[-1])
    prices = features(bars)
    account = Account(**state['account'])
    obs = observation(scaler.transform(prices.iloc[-OBS_SPEC['window']:]),
                      account, close, state['equity'], state['peak'])
    action, _ = model.predict(obs, deterministic=True)
    raw = float(action[0])
    if not np.isfinite(raw) or not 0 <= raw <= 1:
        raise ValueError('模型输出不合法')
    exposure = float(obs[-3])
    risk_on = close > average
    target = planned_action(action, exposure, SETTINGS, risk_on)[0]
    target = None if np.isnan(target) else float(target)
    mapped = float(np.float32(SETTINGS.trend_floor + SETTINGS.trend_span * raw)) if risk_on else 0.0
    if not risk_on:
        reason = 'exit' if account.quantity else 'cash'
    elif target is None:
        reason = 'hold'
    elif target > exposure:
        reason = 'buy'
    else:
        reason = 'sell'
    return dict(
        target=target, observation=obs.tolist(), state=state,
        facts=dict(version=VERSION, instrument='510300', seed=SEED, date=state['date'],
                   data_as_of=str(last.availability_at), execution_date=execution_date,
                   cash=account.cash, equity=state['equity'], quantity=account.quantity,
                   close=close, ma50=average, exposure=exposure, raw=raw, mapped=mapped, target=target,
                   difference=mapped-exposure, band=SETTINGS.hold_band, reason=reason,
                   return_1=float(prices.return_1.iloc[-1]), return_5=float(prices.return_5.iloc[-1]),
                   return_20=float(prices.return_20.iloc[-1])),
    )
