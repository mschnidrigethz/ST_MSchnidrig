# Cube-Lift Trajectory Recorder

This repository contains a script that can be used for lift-cube trajectory recording as for the thesis "Adversarial Inverse Reinforcement Learning for Small Batch Production Automation with Cobot".

## Repository layout

- `README.md` - this file.
- `trajectory_recorder.py` - the adapted file that changes the original recording of the cube-stack task to the cube-lift task.

## Where to integrate in franka_ros2_ws

- `trajectory_recorder.py`
	- replace: `/src/franka_trajectory_recorder/franka_trajectory_recorder/trajectory_recorder.py`