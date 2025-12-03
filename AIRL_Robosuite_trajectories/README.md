# AIRL with Imitation, Robosuite, Robomimic

This repository contains scripts that can be used to enhance the original Imitation repository for more AIRL training possibilities, as in the thesis "Adversarial Inverse Reinforcement Learning for Small Batch Production Automation with Cobot".

## Repository layout

- `README.md` - this file.
- `requirements.yml` - requirements file.
- `4a_airl_franka_lift.ipynb` - AIRL, applied on the cube-lift task.
- `4b_airl_cartpole.ipynb` - preexisting cartpole-AIRL demonstration (`4_train_airl.ipynb`), enhanced with visulizations of policies in various stages and additional RL training afterwards, using the reward net.
- `4c_airl_franka_lift_newDataset.ipynb` - AIRL, applied on the cube-lift task with larger dataset and short BC in the beginning. BC is needed because the grasp action gets lost in AIRL.
-  `convert_reward_net.py` - convert reward net in IsaacLab compatible format.
## Where to integrate in Imitation

- `4a_airl_franka_lift.ipynb`, `4b_airl_cartpole.ipynb` and `convert_reward_net.py`
	- add: `/imitation/docs/tutorials`
