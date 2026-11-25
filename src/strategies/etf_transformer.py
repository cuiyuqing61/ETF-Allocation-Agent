"""Twenty-session market sequence and current account state for ETF PPO."""
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.strategies.etf_ppo import FEATURES, TargetWeightEnv

WINDOW = 20
HISTORY = 40
VERSION = 'transformer20-v1'
ACCOUNT_FIELDS = ['exposure', 'cash_fraction', 'drawdown']
OBS_SIZE = WINDOW * len(FEATURES) + len(ACCOUNT_FIELDS)
OBS_SPEC = dict(version=VERSION, window=WINDOW, market=FEATURES, account=ACCOUNT_FIELDS,
                order='oldest_first', history=HISTORY)


def observation(window, account, close, equity, peak):
    if window.shape != (WINDOW, len(FEATURES)):
        raise ValueError('Expected 20 complete market observations')
    state = [account.quantity * close / equity, account.cash / equity, 1 - equity / peak]
    result = np.concatenate((window.ravel(), state)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Invalid market or account observation')
    return result


class TransformerEnv(TargetWeightEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_SIZE,), dtype=np.float32)

    def observation(self):
        # TargetWeightEnv resets once during initialization.
        if not hasattr(self, 'sequence_features'):
            first = self.first - WINDOW + 1
            if first < HISTORY - WINDOW:
                raise ValueError('The first decision needs 40 consecutive trading sessions')
            self.sequence_features = self.scaler.transform(self.raw_features.iloc[first:self.last + 1])
        offset = self.i - self.first
        window = self.sequence_features[offset:offset + WINDOW]
        return observation(window, self.account, self.closes[self.i], self.equity, self.peak)


class ETFTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=35)
        if observation_space.shape != (OBS_SIZE,):
            raise ValueError('Incompatible Transformer observation shape')
        self.project = nn.Linear(len(FEATURES), 32)
        self.position = nn.Parameter(torch.empty(1, WINDOW, 32))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0,
                                       batch_first=True, activation='gelu')
            for _ in range(2)
        ])

    def forward(self, inputs):
        market = inputs[:, :-3].reshape(-1, WINDOW, len(FEATURES))
        encoded = self.project(market) + self.position
        for layer in self.layers:
            encoded = layer(encoded)
        return torch.cat((encoded[:, -1], inputs[:, -3:]), dim=1)
