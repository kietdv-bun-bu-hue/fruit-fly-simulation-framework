"""Synthesize a metallic fork-on-ceramic-plate screech."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScreechStimulus:
    """Time-domain waveform plus analysis sample rate."""

    waveform: np.ndarray
    sample_rate: int
    duration_s: float

    def save_wav(self, path: Path | str) -> Path:
        """Write a mono 16-bit WAV (stdlib only)."""
        import wave

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(self.waveform, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return path


def synthesize_fork_ceramic_screech(
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    *,
    seed: int = 7,
    base_hz: float = 2800.0,
    rasp_hz: float = 18.0,
) -> ScreechStimulus:
    """Synthesize a harsh metallic scrape against ceramic.

    Construction (perceptual, not a physical contact model):
    - inharmonic partials around a high carrier (fork ringing)
    - amplitude-modulated broadband noise (ceramic grit / stick-slip)
    - slow rasp envelope (scraping strokes)
    - rising attack then irregular sustain
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Metallic inharmonic stack (slightly stretched partials).
    partials = np.zeros(n, dtype=np.float64)
    for k, (ratio, amp) in enumerate(
        [(1.0, 0.55), (1.47, 0.35), (2.11, 0.28), (2.83, 0.18), (3.71, 0.12)]
    ):
        freq = base_hz * ratio * (1.0 + 0.002 * k)
        phase = rng.uniform(0, 2 * np.pi)
        vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * (6.0 + 0.7 * k) * t)
        partials += amp * np.sin(2 * np.pi * freq * vibrato * t + phase)

    # Gritty noise shaped toward high frequencies (ceramic scrape).
    white = rng.standard_normal(n)
    # Simple one-pole high-emphasis filter.
    grit = np.zeros(n, dtype=np.float64)
    prev = 0.0
    for i, x in enumerate(white):
        prev = 0.15 * prev + 0.85 * x
        grit[i] = x - 0.65 * prev

    # Stick-slip rasp (periodic scrape strokes with irregular duty).
    rasp = 0.55 + 0.45 * np.sin(2 * np.pi * rasp_hz * t)
    rasp = np.clip(rasp, 0.0, 1.0) ** 1.6
    stroke_jitter = 0.85 + 0.15 * np.sin(2 * np.pi * 3.3 * t + 0.4)
    rasp *= stroke_jitter

    # Overall attack / sustain envelope.
    attack = np.clip(t / 0.04, 0.0, 1.0)
    sustain = 0.65 + 0.35 * np.sin(2 * np.pi * 1.7 * t)
    env = attack * sustain

    mix = env * rasp * (0.55 * partials + 0.9 * grit)
    peak = np.max(np.abs(mix)) + 1e-12
    mix = 0.95 * mix / peak
    return ScreechStimulus(waveform=mix.astype(np.float64), sample_rate=sample_rate, duration_s=duration_s)


def band_energies(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    center_time_s: float,
    window_s: float = 0.02,
    bands_hz: tuple[tuple[float, float], ...] = (
        (400.0, 1600.0),
        (1600.0, 5000.0),
        (5000.0, 12000.0),
    ),
) -> np.ndarray:
    """Return RMS energies in frequency bands around ``center_time_s``."""
    n = waveform.shape[0]
    half = max(1, int(0.5 * window_s * sample_rate))
    center = int(center_time_s * sample_rate)
    start = max(0, center - half)
    end = min(n, center + half)
    chunk = waveform[start:end]
    if chunk.size < 8:
        return np.zeros(len(bands_hz), dtype=np.float64)

    # rFFT magnitude spectrum.
    spec = np.fft.rfft(chunk * np.hanning(chunk.size))
    freqs = np.fft.rfftfreq(chunk.size, d=1.0 / sample_rate)
    mag2 = (spec.real**2 + spec.imag**2).astype(np.float64)

    energies = np.zeros(len(bands_hz), dtype=np.float64)
    for i, (lo, hi) in enumerate(bands_hz):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        energies[i] = float(np.sqrt(np.mean(mag2[mask])))
    peak = np.max(energies) + 1e-12
    return energies / peak
