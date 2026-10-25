"""Map screech audio → Johnston's-organ / chordotonal-like sensory currents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flygym_env.connectome import NEURON_NAMES, SENSORY_INDICES
from flygym_env.stimulus import ScreechStimulus, band_energies


@dataclass
class AntennaeSensoryState:
    """Demo sensory vector approximating bilateral JO / chordotonal drive.

    FlyGym 2.x does **not** expose ``obs["antennae"]`` (that key belonged to the
    sketch / older Gymnasium-style API). This project therefore:

    1. Synthesizes the fork-ceramic screech offline.
    2. Extracts band energies each physics step.
    3. Applies a fixed world-frame laterality (stimulus louder on the fly's
       right) so escape turns leftward / away.
    4. Optionally blends weak proprioceptive modulation from thorax heading
       when a live ``Simulation`` is available (see ``heading_bias``).
    """

    vector: np.ndarray  # shape (4,) — jo_l_low, jo_l_high, jo_r_low, jo_r_high
    band_energies: np.ndarray
    laterality: float  # >0 means stronger on right antenna


class JohnstonOrganEncoder:
    """Encode screech time series into bilateral JO-like currents."""

    def __init__(
        self,
        stimulus: ScreechStimulus,
        *,
        laterality: float = 0.55,
        gain: float = 3.4,
        n_neurons: int = len(NEURON_NAMES),
    ) -> None:
        self.stimulus = stimulus
        self.laterality = float(np.clip(laterality, -1.0, 1.0))
        self.gain = float(gain)
        self.n_neurons = n_neurons

    def encode(
        self,
        sim_time_s: float,
        *,
        heading_bias: float = 0.0,
    ) -> tuple[np.ndarray, AntennaeSensoryState]:
        """Return full ``I_sensory`` (len n_neurons) and the antennae state."""
        # Loop the stimulus waveform for longer runs.
        t_loop = sim_time_s % self.stimulus.duration_s
        bands = band_energies(
            self.stimulus.waveform,
            self.stimulus.sample_rate,
            center_time_s=t_loop,
        )
        # Collapse 3 analysis bands → low / high JO channels.
        low = 0.65 * bands[0] + 0.35 * bands[1]
        high = 0.40 * bands[1] + 0.60 * bands[2]

        # Bilateral split: laterality > 0 → right louder (stimulus from right).
        lat = np.clip(self.laterality + 0.25 * heading_bias, -1.0, 1.0)
        left_gain = self.gain * (1.0 - 0.5 * lat)
        right_gain = self.gain * (1.0 + 0.5 * lat)

        antennae = np.array(
            [
                left_gain * low,
                left_gain * high,
                right_gain * low,
                right_gain * high,
            ],
            dtype=np.float64,
        )
        i_sensory = np.zeros(self.n_neurons, dtype=np.float64)
        i_sensory[SENSORY_INDICES] = antennae
        state = AntennaeSensoryState(
            vector=antennae, band_energies=bands, laterality=float(lat)
        )
        return i_sensory, state


def document_antennae_fallback() -> str:
    """Human-readable note for README / CLI about obs key absence."""
    return (
        "FlyGym 2.1 has no Gymnasium observation dict and no obs['antennae'] key. "
        "Sensory drive is synthesized from the fork-ceramic screech and injected "
        "as I_sensory into ConnectomeLIFNetwork (JO L/R low+high channels). "
        "Optional heading bias uses thorax orientation from Simulation.get_body_rotations."
    )
