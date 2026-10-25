"""Closed-loop FlyGym flee simulation driven by screech → LIF → turning CPG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Headless / CI rendering (EGL). Safe to set before mujoco imports by callers.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from flygym import Simulation
from flygym.compose import ActuatorType, FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import (
    apply_locomotion_action,
    get_default_locomotion_dof_order,
    make_locomotion_fly,
)
from flygym_demo.complex_terrain.cpg_controller import make_tripod_cpg_network
from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
from flygym_demo.complex_terrain.turning_controller import HybridTurningController

from flygym_env.connectome import ConnectomeLIFNetwork
from flygym_env.motor import FleeMotorMapper
from flygym_env.sensory import JohnstonOrganEncoder
from flygym_env.stimulus import synthesize_fork_ceramic_screech


@dataclass
class FleeSimConfig:
    n_steps: int = 2000
    timestep: float = 1e-4
    spawn_xyz: tuple[float, float, float] = (0.0, 0.0, 0.7)
    video_path: Path = Path("fly_simulation.mp4")
    connectivity_csv: Path = Path("data/synaptic_connectivity.csv")
    camera_res: tuple[int, int] = (240, 320)
    playback_speed: float = 0.25
    output_fps: int = 25
    stimulus_seed: int = 7
    laterality: float = 0.55  # >0 → stimulus louder on right → flee left
    save_stimulus_wav: Path | None = Path("outputs/fork_ceramic_screech.wav")
    seed: int = 0


@dataclass
class FleeSimResult:
    video_path: Path
    n_steps: int
    final_xy: np.ndarray
    start_xy: np.ndarray
    total_spikes: np.ndarray
    mean_descending: np.ndarray
    antennae_fallback_note: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _thorax_heading_bias(sim: Simulation, fly_name: str) -> float:
    """Project thorax forward (x) onto world +y as a weak laterality bias."""
    fly = sim.world.fly_lookup[fly_name]
    body_order = fly.get_bodysegs_order()
    bodyseg_cls = type(fly).BODY_SEGMENT_CLASS
    thorax_idx = body_order.index(bodyseg_cls("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly_name][thorax_idx]
    heading = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)[:, 0]
    return float(np.clip(heading[1], -1.0, 1.0))


def run_flee_simulation(config: FleeSimConfig | None = None) -> FleeSimResult:
    """Build world, run closed loop, save ``fly_simulation.mp4``."""
    from flygym_env.sensory import document_antennae_fallback

    cfg = config or FleeSimConfig()
    root = _repo_root()
    csv_path = cfg.connectivity_csv
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    video_path = cfg.video_path
    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Sensory stimulus ---
    stimulus = synthesize_fork_ceramic_screech(
        duration_s=max(1.0, cfg.n_steps * cfg.timestep * 1.5),
        seed=cfg.stimulus_seed,
    )
    if cfg.save_stimulus_wav is not None:
        wav_path = cfg.save_stimulus_wav
        if not wav_path.is_absolute():
            wav_path = root / wav_path
        stimulus.save_wav(wav_path)

    encoder = JohnstonOrganEncoder(stimulus, laterality=cfg.laterality)
    network = ConnectomeLIFNetwork.from_default_csv(csv_path)
    motor = FleeMotorMapper()

    # --- FlyGym v2 scene (FlyMechEnv sketch → Simulation + FlatGroundWorld) ---
    fly = make_locomotion_fly(name="nmf", colorize=True)
    tracking_cam = fly.add_tracking_camera()

    world = FlatGroundWorld(name="flee_arena")
    world.add_fly(
        fly,
        cfg.spawn_xyz,
        Rotation3D(format="quat", values=[1.0, 0.0, 0.0, 0.0]),
    )

    sim = Simulation(world, timestep=cfg.timestep)
    sim.set_renderer(
        tracking_cam,
        camera_res=cfg.camera_res,
        playback_speed=cfg.playback_speed,
        output_fps=cfg.output_fps,
        buffer_frames=True,
    )

    cpg = make_tripod_cpg_network(cfg.timestep, seed=cfg.seed)
    controller = HybridTurningController(
        timestep=cfg.timestep,
        cpg_network=cpg,
        preprogrammed_steps=PreprogrammedSteps(),
        output_dof_order=get_default_locomotion_dof_order(),
    )

    sim.reset()
    sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=bool))
    sim.warmup()

    fly_name = fly.name
    start_pos = sim.get_body_positions(fly_name)[
        fly.get_bodysegs_order().index(type(fly).BODY_SEGMENT_CLASS("c_thorax"))
    ][:2].copy()

    descending_hist = []
    for step_i in range(cfg.n_steps):
        t = step_i * cfg.timestep
        heading_bias = _thorax_heading_bias(sim, fly_name)
        i_sensory, _ant = encoder.encode(t, heading_bias=heading_bias)
        spikes = network.step(i_sensory)
        descending = motor.map_spikes(spikes)
        descending_hist.append(descending)

        obs = HybridControllerObservation.from_sim(sim, fly_name)
        action = controller.step(descending, obs)
        apply_locomotion_action(sim, fly_name, action, actuator_type=ActuatorType.POSITION)

        sim.step()
        # API note: sketch used env.render(); FlyGym 2.x uses render_as_needed().
        sim.render_as_needed()

    end_pos = sim.get_body_positions(fly_name)[
        fly.get_bodysegs_order().index(type(fly).BODY_SEGMENT_CLASS("c_thorax"))
    ][:2].copy()

    # API note: sketch used env.save_video(...); FlyGym 2.x saves via renderer.
    sim.renderer.save_video(video_path)
    sim.close()

    return FleeSimResult(
        video_path=video_path,
        n_steps=cfg.n_steps,
        final_xy=end_pos,
        start_xy=start_pos,
        total_spikes=network.spike_counts.copy(),
        mean_descending=np.mean(np.stack(descending_hist), axis=0),
        antennae_fallback_note=document_antennae_fallback(),
    )
