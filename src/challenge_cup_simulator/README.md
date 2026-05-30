# Challenge Cup Simulator

This package contains a standalone Challenge Cup MuJoCo scene for the Kuavo
`biped_s52` robot.

## Start

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch
```

`with_estimation` defaults to `false` so the scene can start in workspaces that
do not include `humanoid_estimation`. Enable it explicitly if that package is
available:

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch with_estimation:=true
```

The CRAIC rosbag nodelet include is also optional and defaults to off. Enable it
with `use_rosbag_nodelet:=true` only when the `humanoid_interface` package is
available.

The scene loaded by default is:

```text
models/biped_s52/xml/scene1.xml
```

The XML is generated from a YAML scene description, matching the CRAIC
simulator workflow:

```bash
rosrun challenge_cup_simulator scene_builder.py
```

Generate all configured competition scenes:

```bash
rosrun challenge_cup_simulator scene_builder.py --all
```

or directly from the source tree:

```bash
python3 src/challenge_cup_simulator/utils/scene_builder.py
```

The YAML source is:

```text
config/scenes/scene1.yaml
config/scenes/scene2.yaml
config/scenes/scene3.yaml
```

Switch scenes at launch time:

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene2
```

## LiDAR

The simulated mid360 LiDAR uses the PyPI `mujoco-lidar` package for scan
patterns. Install the dependency in the Docker workspace before launching:

```bash
pip3 install --no-deps -r src/challenge_cup_simulator/requirements.txt
```

`--no-deps` avoids optional visualization dependencies that are not needed by
the Challenge Cup LiDAR node. The default launch starts the CPU backend and
publishes `/lidar/points`, `/lidar_imu`, and `/mujoco/qpos`:

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch enable_lidar:=true lidar_backend:=cpu
```

## Scene Layout

- One CRAIC-style table, built from the same simple tabletop and leg geometry.
- Four movable parcels on the table.
- One `0.2 m x 0.2 m` weighing area on the tabletop.
- One `0.4 m x 0.3 m x 0.3 m` open sorting box on the tabletop.
- The robot starts in front of the table, facing the parcels.
