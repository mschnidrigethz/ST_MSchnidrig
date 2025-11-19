# AIRL with Imitation, Robosuite, Robomimic

This repository contains scripts that can be used to enhance the original Imitation repository for more AIRL training possibilities, as in the thesis "Adversarial Inverse Reinforcement Learning for Small Batch Production Automation with Cobot".

## Repository layout

- `README.md` - this file.
- `requirements.yml` - requirements file.
- `4a_airl_franka_lift.ipynb` - AIRL, applied on the cube-lift task.
- `4b_airl_cartpole.ipynb` - Preexisting cartpole-AIRL demonstration (`4_train_airl.ipynb`), enhanced with visulizations of policies in various stages and additional RL training afterwards, using the reward net.
## Where to integrate in Imitation

- `4a_airl_franka_lift.ipynb` and `4b_airl_cartpole.ipynb`
	- add: `/imitation/docs/tutorials`
