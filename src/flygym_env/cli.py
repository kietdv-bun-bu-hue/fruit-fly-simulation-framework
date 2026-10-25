"""CLI entrypoint for the fork-screech flee demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from flygym_env.simulation import FleeSimConfig, run_flee_simulation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fly-flee-sim",
        description=(
            "Closed-loop FlyGym demo: metallic fork-on-ceramic screech drives a "
            "placeholder LIF connectome; motor spikes turn the fly away."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Physics steps (default: 2000 ≈ 0.2 s at dt=1e-4).",
    )
    p.add_argument(
        "--timestep",
        type=float,
        default=1e-4,
        help="MuJoCo timestep in seconds (default: 1e-4).",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=Path("fly_simulation.mp4"),
        help="Output MP4 path (default: ./fly_simulation.mp4).",
    )
    p.add_argument(
        "--connectivity",
        type=Path,
        default=Path("data/synaptic_connectivity.csv"),
        help="Placeholder adjacency CSV (pre_idx,post_idx,weight).",
    )
    p.add_argument(
        "--laterality",
        type=float,
        default=0.55,
        help=">0 means screech louder on the right antenna (flee left).",
    )
    p.add_argument(
        "--no-wav",
        action="store_true",
        help="Skip writing the synthesized screech WAV under outputs/.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = FleeSimConfig(
        n_steps=args.steps,
        timestep=args.timestep,
        video_path=args.video,
        connectivity_csv=args.connectivity,
        laterality=args.laterality,
        save_stimulus_wav=None if args.no_wav else Path("outputs/fork_ceramic_screech.wav"),
    )
    result = run_flee_simulation(cfg)
    displacement = result.final_xy - result.start_xy
    print("Flee simulation complete.")
    print(f"  steps:          {result.n_steps}")
    print(f"  start xy (mm):  {result.start_xy}")
    print(f"  final xy (mm):  {result.final_xy}")
    print(f"  Δxy (mm):       {displacement}")
    print(f"  mean descend L/R: {result.mean_descending}")
    print(f"  total spikes:   {result.total_spikes.tolist()}")
    print(f"  video:          {result.video_path.resolve()}")
    print(f"  sensory note:   {result.antennae_fallback_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
