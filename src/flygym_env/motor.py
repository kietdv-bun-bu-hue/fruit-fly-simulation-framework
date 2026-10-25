"""Map motor-neuron spike rates → flee locomotion commands."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FleeMotorMapper:
    """Convert MN spikes into HybridTurningController descending signals.

    FlyGym v2 position-controls joints (target angles), not raw torques. We map
    motor-neuron activity onto the demo turning controller's bilateral descending
    amplitudes ``[left, right]``.

    In FlyGym's turning CPG, **reducing** the left descending amplitude relative
    to the right produces a leftward turn (and vice versa). So:

    - ``mn_turn_l`` (escape-from-right) → suppress left drive / boost right
    - ``mn_turn_r`` → suppress right drive / boost left
    - ``mn_speed`` raises overall step vigor on both sides

    Rates are EMA-smoothed so binary LIF spikes still steer smoothly.
    """

    base_drive: float = 0.90
    turn_gain: float = 0.85
    speed_gain: float = 0.45
    min_drive: float = 0.15
    max_drive: float = 1.45
    smooth: float = 0.08
    # Integrate recent MN spikes so single-tick binary events still steer.
    rate_tau_steps: int = 40
    _ema: np.ndarray = field(
        default_factory=lambda: np.array([0.90, 0.90], dtype=np.float64)
    )
    _rate: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def reset(self) -> None:
        self._ema[:] = self.base_drive
        self._rate.fill(0.0)

    def map_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Return descending_signal shape ``(2,)`` for left / right CPG drive."""
        alpha = 1.0 / max(self.rate_tau_steps, 1)
        instantaneous = np.array(
            [float(spikes[8]), float(spikes[9]), float(spikes[10])],
            dtype=np.float64,
        )
        self._rate = (1.0 - alpha) * self._rate + alpha * instantaneous
        turn_l, turn_r, speed = self._rate

        # mn_turn_l means "command a left turn" → lower left descending amp.
        left = (
            self.base_drive
            - self.turn_gain * turn_l
            + 0.35 * self.turn_gain * turn_r
            + self.speed_gain * speed
        )
        right = (
            self.base_drive
            - self.turn_gain * turn_r
            + 0.35 * self.turn_gain * turn_l
            + self.speed_gain * speed
        )

        target = np.clip(
            np.array([left, right], dtype=np.float64),
            self.min_drive,
            self.max_drive,
        )
        self._ema = (1.0 - self.smooth) * self._ema + self.smooth * target
        return self._ema.copy()
