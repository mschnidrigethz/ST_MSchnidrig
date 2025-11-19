# Trajectory Cutter - Stack-Cube to Lift-Cube

This repository contains scripts that can be used for translating stack-cube hdf5-files in IsaacLab format to lift-cube hdf5-files in IsaacLab format, as used in the thesis "Adversarial Inverse Reinforcement Learning for Small Batch Production Automation with Cobot".

## Repository layout

- `README.md` - this file.
- `requirements.yml` - requirements file.
- `process_cube_stack_to_lift.py` - the main file, that cuts the data.
- `fix_hdf5_metadata.py` - a helper file, that is called in `process_cube_stack_to_lift.py`, that fixes missing metadata.

## Usage

Below is an example bash command you can copy/paste. Replace placeholder paths and names (e.g. /path/to/...) as needed.

```bash

# Convert stack-cube trajectoriies to lift-cube trajectories
python process_cube_stack_to_lift.py --input /full/path/to/generated_dataset_real_dynamics.hdf5 --output /full/path/to/franka_lift_cube_trajectories.hdf5

```