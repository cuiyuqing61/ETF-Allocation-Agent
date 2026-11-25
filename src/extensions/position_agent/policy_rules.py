"""Position rules shared by the frozen policy and the training environment."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Settings:
    steps: int = 100000
    turnover_bps: int = 10
    exposure_penalty_bps: int = 0
    hold_band: float = 0.
    learning_rate: float = .0003
    episode_sessions: int = 0
    trend_sessions: int = 0
    trend_floor: float = 0.
    trend_span: float = 1.


VARIANT = 'trend-sized-10bp'
SETTINGS = Settings(episode_sessions=252, hold_band=.075, trend_sessions=50,
                    trend_floor=.40, trend_span=.20)


def planned_action(action, exposure, settings, risk_on=True):
    if settings.trend_sessions:
        if not risk_on:
            return np.array([0.], dtype=np.float32)
        action = np.array([settings.trend_floor + settings.trend_span * action[0]], dtype=np.float32)
    if abs(float(action[0]) - exposure) < settings.hold_band:
        return np.array([np.nan], dtype=np.float32)
    return action
