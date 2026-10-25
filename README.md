# Fruit Fly Brain Simulation — Flee Demo (first slice)

Closed-loop **FlyGym** neuromechanical demo: a synthesized metallic *fork scratching a ceramic plate* screech drives a small placeholder LIF “connectome,” which modulates a turning walking controller so the digital fruit fly attempts to **flee** (turn away / locomote away from the louder side).

This is a **demo-quality first slice** — runnable loop + video — not the full FlyWire connectome.

## Requirements

- Python 3.10–3.12
- Linux recommended (headless EGL rendering on servers)
- MuJoCo (pulled in by `flygym`)

## Create the `flygym_env` venv and install

From the repo root:

```bash
python3 -m venv flygym_env
source flygym_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Headless / CI machines (no display):
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Optional: add those two `export` lines to your shell profile so every session is ready.

## Run the flee demo

Any of:

```bash
python -m flygym_env --steps 2000 --video fly_simulation.mp4
# or
fly-flee-sim --steps 2000 --video fly_simulation.mp4
# or
python scripts/run_flee_sim.py --steps 2000 --video fly_simulation.mp4
```

Useful flags: `--laterality 0.55` (positive = louder on the **right** antenna → flee left), `--connectivity data/synaptic_connectivity.csv`, `--no-wav`.

Outputs:

- `fly_simulation.mp4` — tracking-camera recording (default CWD)
- `outputs/fork_ceramic_screech.wav` — synthesized stimulus (unless `--no-wav`)

## Project layout

```
src/flygym_env/
  connectome.py   # ConnectomeLIFNetwork + CSV adjacency loader
  stimulus.py     # fork-on-ceramic screech synthesizer
  sensory.py      # screech → Johnston’s-organ / chordotonal-like I_sensory
  motor.py        # MN spikes → bilateral descending / flee drive
  simulation.py   # FlyGym world, closed loop, video save
  cli.py          # CLI
data/synaptic_connectivity.csv   # placeholder adjacency (FlyWire later)
scripts/run_flee_sim.py
```

### Network layout (placeholder)

| idx | name | role |
|----:|------|------|
| 0–1 | `jo_l_low/high` | left JO sensory |
| 2–3 | `jo_r_low/high` | right JO sensory |
| 4–7 | `in_*` | interneurons / aversive / escape hubs |
| 8–9 | `mn_turn_l/r` | turn motor neurons |
| 10 | `mn_speed` | locomotion vigor |

Voltage update each tick:

```text
V_next = V_curr + (I_sensory + (Adjacency_Matrix @ Spikes)) * decay_factor
```

## FlyGym API mapping (sketch → flygym 2.1)

The product sketch mentioned `FlyMechEnv(render_mode="saved", …)`, `obs["antennae"]`, `env.render()`, and `env.save_video(...)`. **Installed FlyGym 2.1.1 does not provide those symbols.** This repo adapts as follows:

| Sketch | FlyGym 2.1.1 used here |
|--------|-------------------------|
| `FlyMechEnv(...)` | `FlatGroundWorld` + `NeuroMechFly` via `make_locomotion_fly`, then `Simulation(world, timestep=1e-4)` |
| Gymnasium `reset` / `step(action)` returning `obs` | Imperative API: `sim.reset()`, `sim.set_actuator_inputs(...)`, `sim.step()` (no obs dict) |
| `obs["antennae"]` | **Not available.** Screech → band energies → bilateral JO current vector injected as `I_sensory`. Optional thorax heading bias from `mj_data.xmat`. |
| Joint torques / target angles | Position actuators (`ActuatorType.POSITION`); MN rates → `HybridTurningController` descending signal `[left, right]` → preprogrammed step angles (+ adhesion) |
| `env.render()` | `sim.render_as_needed()` |
| `env.save_video("fly_simulation.mp4")` | `sim.set_renderer(tracking_cam, buffer_frames=True)` then `sim.renderer.save_video(path)` |

Locomotion helpers come from the `flygym_demo.complex_terrain` package shipped with FlyGym 2.x (CPG + turning controller + preprogrammed steps).

`gymnasium` remains a declared dependency for ecosystem compatibility / future wrappers; the closed-loop demo itself uses FlyGym’s native Simulation API.

## Design notes

- **Stimulus:** inharmonic metallic partials + high-emphasized grit noise + stick-slip rasp envelope (`stimulus.py`).
- **Flee:** laterality makes the right JO louder by default; the LIF graph biases `mn_turn_l`. In FlyGym’s turning CPG, a **lower** left descending amplitude (vs right) yaws the fly left — away from the screech — while `mn_speed` keeps locomotion going.
- **Connectome:** hand-built CSV only. Swap `data/synaptic_connectivity.csv` (or load a denser matrix) when wiring FlyWire Codex.

## License

Demo code: Apache-2.0. FlyGym / NeuroMechFly retain their upstream licenses.
