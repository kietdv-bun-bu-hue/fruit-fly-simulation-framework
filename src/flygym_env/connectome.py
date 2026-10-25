"""Hand-built LIF network with CSV adjacency (FlyWire placeholder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Canonical neuron roles for the demo connectome.
NEURON_NAMES = (
    "jo_l_low",  # 0 — left Johnston's organ, low band
    "jo_l_high",  # 1 — left JO, high/screech band
    "jo_r_low",  # 2 — right JO, low band
    "jo_r_high",  # 3 — right JO, high/screech band
    "in_l",  # 4 — left interneuron
    "in_r",  # 5 — right interneuron
    "in_aversive",  # 6 — aversive hub
    "in_escape",  # 7 — escape hub
    "mn_turn_l",  # 8 — left descending / turn MN
    "mn_turn_r",  # 9 — right descending / turn MN
    "mn_speed",  # 10 — locomotion-speed MN
)

SENSORY_INDICES = np.array([0, 1, 2, 3], dtype=int)
MOTOR_INDICES = np.array([8, 9, 10], dtype=int)


def load_adjacency_csv(path: Path | str, n_neurons: int) -> np.ndarray:
    """Load a sparse (pre, post, weight) CSV into a dense adjacency matrix.

    Matrix convention: ``(Adjacency @ Spikes)[post]`` sums weighted presynaptic spikes.
    """
    path = Path(path)
    adjacency = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("pre_idx"):
                continue
            pre_s, post_s, w_s = line.split(",")[:3]
            pre, post, weight = int(pre_s), int(post_s), float(w_s)
            if not (0 <= pre < n_neurons and 0 <= post < n_neurons):
                raise ValueError(f"Index out of range in {path}: {pre}->{post}")
            adjacency[post, pre] += weight
    return adjacency


class ConnectomeLIFNetwork:
    """Minimal leaky-integrate-and-fire network over a fixed adjacency matrix.

    Voltage update (per step)::

        V_next = V_curr + (I_sensory + (Adjacency_Matrix @ Spikes)) * decay_factor

    Spikes are binary events emitted when ``V >= threshold``; those neurons then
    reset to ``v_reset``.
    """

    def __init__(
        self,
        adjacency: np.ndarray | None = None,
        *,
        connectivity_csv: Path | str | None = None,
        n_neurons: int | None = None,
        decay_factor: float = 0.28,
        threshold: float = 0.75,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        names: tuple[str, ...] = NEURON_NAMES,
    ) -> None:
        self.names = names
        n = n_neurons if n_neurons is not None else len(names)
        if adjacency is None:
            if connectivity_csv is None:
                raise ValueError("Provide adjacency or connectivity_csv")
            adjacency = load_adjacency_csv(connectivity_csv, n)
        adjacency = np.asarray(adjacency, dtype=np.float64)
        if adjacency.shape != (n, n):
            raise ValueError(f"adjacency shape {adjacency.shape} != ({n}, {n})")

        self.n_neurons = n
        self.adjacency = adjacency
        self.decay_factor = float(decay_factor)
        self.threshold = float(threshold)
        self.v_reset = float(v_reset)
        self.v_rest = float(v_rest)

        self.v = np.full(n, self.v_rest, dtype=np.float64)
        self.spikes = np.zeros(n, dtype=np.float64)
        self.spike_counts = np.zeros(n, dtype=np.int64)

    @classmethod
    def from_default_csv(cls, csv_path: Path | str, **kwargs) -> "ConnectomeLIFNetwork":
        return cls(connectivity_csv=csv_path, **kwargs)

    def reset(self) -> None:
        self.v.fill(self.v_rest)
        self.spikes.fill(0.0)
        self.spike_counts.fill(0)

    def step(self, i_sensory: np.ndarray) -> np.ndarray:
        """Advance one discrete network tick.

        Args:
            i_sensory: Length-``n_neurons`` current injection (A.U.). Non-sensory
                entries may be zero.

        Returns:
            Binary spike vector for this tick (same length as ``n_neurons``).
        """
        i_sensory = np.asarray(i_sensory, dtype=np.float64)
        if i_sensory.shape != (self.n_neurons,):
            raise ValueError(
                f"i_sensory shape {i_sensory.shape} != ({self.n_neurons},)"
            )

        synaptic = self.adjacency @ self.spikes
        self.v = self.v + (i_sensory + synaptic) * self.decay_factor

        spiked = self.v >= self.threshold
        self.spikes = spiked.astype(np.float64)
        self.spike_counts += spiked.astype(np.int64)
        self.v[spiked] = self.v_reset
        return self.spikes.copy()

    def motor_rates(self, window_spikes: np.ndarray | None = None) -> dict[str, float]:
        """Map recent motor spikes to turn-left / turn-right / speed rates in [0, 1]."""
        if window_spikes is None:
            window_spikes = self.spikes
        rates = {
            "turn_l": float(window_spikes[8]),
            "turn_r": float(window_spikes[9]),
            "speed": float(window_spikes[10]),
        }
        return rates
