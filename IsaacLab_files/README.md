# IsaacLab - AIRL-enabled Lift Cube Environment

This repository contains an Isaac-Lab based environment and training scripts used for the thesis "Adversarial Inverse Reinforcement Learning for Small Batch Production Automation with Cobot".

## Repository layout

- `README.md` - this file.
- `requirements.yml` - requirements file.
- `franka.py` - a Franka robot variant more closely matching the lab robot. (Originally created by Jan Frischknecht; integrated here.)
- `observations.py` - observation processing adapted to provide inputs expected by the reward networks.
- `franka/` - contains environment configuration files and registration for the modified Franka environments:
	- `__init__.py` - registers `Isaac-Lift-Cube-IK-Abs-AIRL-v0`.
	- `ik_abs_airl_env_cfg.py` - a new environment configuration modified to support a reward network (AIRL).
	- `ik_abs_env_cfg.py` - baseline IK absolute environment config (adjusted custom robot).
	- `joint_pos_env_cfg.py` - uses the cube object geometry actually used in experiments (a slightly larger cube than the standard one).

- `sb3_AIRL/` - contains the training script that runs RL training using a learned reward network (from AIRL):
	- `train.py` - entry point for RL training using a reward network (AIRL).

## Where to integrate in IsaacLab

- `franka.py`
	- replace: `source/isaaclab_assets/isaaclab_assets/robots/franka.py`

- `franka/`
	- target folder: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/`
		- add: `ik_abs_airl_env_cfg.py`
		- replace: `ik_abs_env_cfg.py`
		- replace: `joint_pos_env_cfg.py`
        - replace: `__init__.py`

- `observations.py`
	- replace: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/observations.py`

- `sb3_AIRL/`
	- add: `scripts/reinforcement_learning/sb3_AIRL`

## Usage

If you want to use the custom environment, make sure to use `Isaac-Lift-Cube-IK-Abs-AIRL-v0`when specifying the environment in the commands.


Below is an example bash command you can copy/paste.
```bash

# Run sb3 PPO training with reward net
python /IsaacLab/scripts/reinforcement_learning/sb3_AIRL/train.py

```
