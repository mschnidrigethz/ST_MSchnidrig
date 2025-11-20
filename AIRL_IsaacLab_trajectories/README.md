# AIRL with Trjaectories, Recorded or Generated in Isaac Lab

This repository contains scripts that can be used to train AIRL with Isaac Lab data in a Robosuite environment, as in the thesis "Adversarial Inverse Reinforcement Learning for Small Batch Production Automation with Cobot".

## Repository layout

- `README.md` - this file.
- `requirements.yml` - requirements file.
- `main.py` - PPO-AIRL training.
- `play_demos_RoboSuite.py` - simulate trajectories in the Robosuite environment.
- `export_reward_net.py` - traces the reward net to create a .pt file, which is compatible with IsaacLab.
- `Recorded_Data/` - these files are tailored for the use of data, collected with the trajectory recorder:
	- `hdf5TOnpz_converter.py` - converts the .hdf5 dataset from Isaac Lab in a .npz dataset (compatible with Robosuite).
	- `prepare_training_npz.py` - translates the .npz dataset in Robosuite dimensions.
- `Synthetic_Data/` - these files are tailored for the use of data, collected with the trajectory cutter:
    - `hdf5TOnpz_converter.py` - converts the .hdf5 dataset from Isaac Lab in a .npz dtatset (compatible with Robosuite).
	- `prepare_training_npz.py` - translates the .npz dataset in Robosuite dimensions.

## Usage

Below are example bash commands you can copy/paste. Replace placeholder paths and names (e.g. /path/to/..., ENV_NAME) as needed.

```bash

# Run the main training script (PPO + AIRL)
python main.py

# Play or simulate recorded demos in Robosuite
python play_demos_RoboSuite.py /path/to/demo_file.npz

# Export / trace the reward network for IsaacLab (produces a .pt file)
python export_reward_net.py /path/to/checkpoint.pth /path/to/output_reward_net.pt

# Recorded data utilities (from Recorded_Data/)
python Recorded_Data/hdf5TOnpz_converter.py /path/to/input_dataset.hdf5 /path/to/output_dataset.npz
python Recorded_Data/prepare_training_npz.py /path/to/input_dataset.npz /path/to/output_prepared.npz

# Synthetic data utilities (from Synthetic_Data/)
python Synthetic_Data/hdf5TOnpz_converter.py /path/to/input_dataset.hdf5 /path/to/output_dataset.npz
python Synthetic_Data/prepare_for_robosuite.py /path/to/input_dataset.npz /path/to/output_prepared.npz

```
