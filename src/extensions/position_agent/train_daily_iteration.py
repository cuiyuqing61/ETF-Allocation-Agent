"""Train and validate the released daily Transformer on your local 510300 data."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from src.backtest.position_replay import ReplayConfig, replay
from src.data.corporate_actions import load_actions
from src.data.data_store import DataStore
from src.extensions.position_agent.agent_policy import CHECKPOINT, ROOT
from src.extensions.position_agent.policy_rules import SETTINGS, VARIANT, planned_action
from src.strategies.etf_ppo import FrozenScaler
from src.strategies.etf_transformer import ETFTransformerExtractor, OBS_SPEC, TransformerEnv
from src.trading.corporate_actions import CorporateActionLedger


class ActionEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        above_mean = env.bars.close.gt(env.bars.close.rolling(SETTINGS.trend_sessions).mean())
        self.risk_on = dict(zip(env.bars.trade_date, above_mean, strict=True))

    def target(self, action, exposure):
        day = self.env.bar_records[self.env.i]['trade_date']
        return planned_action(action, exposure, SETTINGS, self.risk_on[day])

    def step(self, target):
        return self.env.hold_shares() if np.isnan(target[0]) else self.env.step(target)


class TrainingEnv(ActionEnv):
    def step(self, action):
        target = self.target(action, float(self.env.observation()[-3]))
        obs, reward, done, truncated, info = super().step(target)
        traded = info['execution']['notional'] / info['daily']['prev_equity']
        return obs, reward - SETTINGS.turnover_bps * traded / 10000, done, truncated, info


class RandomWindowEnv(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        base = self.unwrapped
        first = int(base.np_random.integers(base.first + 1, base.last - SETTINGS.episode_sessions + 2))
        base.i = first - 1
        base.ledger = CorporateActionLedger(base.actions, base.bar_records[first]['trade_date'])
        return base.observation(), info


def validate(model, env, bars, first, last, base):
    obs, _ = env.reset()
    decisions, daily = [], []
    while env.env.i < env.env.last:
        action, _ = model.predict(obs, deterministic=True)
        target = env.target(action, float(obs[-3]))
        decisions.append((env.env.bar_records[env.env.i]['trade_date'], float(target[0])))
        obs, _, _, _, info = env.step(target)
        daily.append(info['daily'])
    weights = pd.DataFrame(decisions, columns=['date', '510300']).set_index('date')
    config = ReplayConfig(first, last, base['project']['initial_capital'],
                          base['market_rules'], base['costs'], env.env.actions)
    selected = bars.loc[bars.trade_date.between(decisions[0][0], last)]
    result = replay(config, 'transformer20-validation', selected, weights)
    settled = [row for row in result.daily_account if row['is_session']]
    for env_row, replay_row in zip(daily, settled, strict=True):
        columns = ('cash', 'quantity', 'dividend_receivable', 'dividend_cash', 'equity')
        if any(abs(env_row[key] - replay_row[key]) > 1e-7 for key in columns):
            raise ValueError('Training environment and validation replay disagree')
    return result.metrics


def train(store, dataset_id, actions_version, seed, steps, output):
    frozen = json.loads((CHECKPOINT / 'config.json').read_text())
    base, ppo = frozen['frozen_base_plan'], frozen['ppo']
    if steps <= 0 or steps % ppo['n_steps']:
        raise ValueError(f"Choose a positive multiple of {ppo['n_steps']} training steps")
    if output.exists():
        raise FileExistsError(output)
    metadata, bars = store.load_position_dataset(dataset_id)
    if metadata['instrument'] != '510300' or metadata['adjustment_policy'] != 'none':
        raise ValueError('Training requires unadjusted 510300 prices')
    train_start, train_end = base['project']['splits']['train']
    validation_start, validation_end = base['project']['splits']['validation']
    actions = load_actions(store, actions_version, dataset_id, train_start, validation_end)
    train_bars = bars.loc[bars.trade_date.le(train_end)].reset_index(drop=True)
    validation_bars = bars.loc[bars.trade_date.le(validation_end)].reset_index(drop=True)
    scaler = FrozenScaler.fit(train_bars, train_start, train_end)
    capital = base['project']['initial_capital']
    train_tape = TransformerEnv(train_bars, scaler, base['market_rules'], base['costs'],
                                capital, train_start, train_end,
                                [action for action in actions if action['record_date'] <= train_end])
    env = gym.wrappers.TimeLimit(RandomWindowEnv(TrainingEnv(train_tape)),
                                 max_episode_steps=SETTINGS.episode_sessions)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = PPO(ActorCriticPolicy, env, seed=seed, **ppo, policy_kwargs={
        'features_extractor_class': ETFTransformerExtractor, 'net_arch': [],
        'log_std_init': math.log(frozen['initial_std']),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    model.learn(steps)
    validation_tape = TransformerEnv(validation_bars, scaler, base['market_rules'], base['costs'],
                                     capital, validation_start, validation_end, actions)
    config = dict(variant=VARIANT, seed=seed, steps=steps, observation=OBS_SPEC,
                  settings={**asdict(SETTINGS), 'steps': steps}, ppo=ppo, dataset_id=dataset_id,
                  training=[train_start, train_end], validation=[validation_start, validation_end],
                  market_rules=base['market_rules'], costs=base['costs'],
                  corporate_actions_version=actions_version)
    with TemporaryDirectory(prefix='.training-', dir=output.parent) as temp:
        pending = Path(temp)
        model_path = pending / 'model.zip'
        model.save(model_path)
        loaded = PPO.load(model_path, device='cpu')
        if loaded.num_timesteps != steps or loaded.seed != seed:
            raise ValueError('Reloaded checkpoint identity differs from training configuration')
        metrics = validate(loaded, ActionEnv(validation_tape), validation_bars,
                           validation_start, validation_end, base)
        (pending / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (pending / 'scaler.json').write_text(json.dumps(asdict(scaler), indent=2) + '\n')
        manifest = {name: sha256((pending / name).read_bytes()).hexdigest()
                    for name in ('config.json', 'scaler.json', 'model.zip')}
        (pending / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (pending / 'validation.json').write_text(json.dumps(metrics, indent=2) + '\n')
        pending.rename(output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=ROOT / 'data/position_agent')
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--actions-version', required=True)
    parser.add_argument('--seed', type=int, choices=(11, 23, 47), default=11)
    parser.add_argument('--steps', type=int, default=SETTINGS.steps)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metrics = train(DataStore(str(args.base_dir)), args.dataset_id, args.actions_version,
                    args.seed, args.steps, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
