"""
Use reexport_reward_net to transform the resulting reward_net into a IsaacLab-compatible TorchScript model.
"""


import numpy as np
import os
import time
import torch
import random
import argparse

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

from imitation.algorithms.adversarial import airl
from imitation.algorithms import bc
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.data.types import Trajectory
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


# -----------------------------------------------------
# Custom VecEnv Wrapper for Rendering
# -----------------------------------------------------

class RenderWrapper(VecEnvWrapper):
    """Wrapper to enable rendering in vectorized environments."""
    def __init__(self, venv):
        super().__init__(venv)
    
    def reset(self):
        obs = self.venv.reset()
        return obs
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Render first environment only (to avoid multiple windows)
        if hasattr(self.venv.envs[0].unwrapped, 'render'):
            self.venv.envs[0].unwrapped.render()
        return obs, rewards, dones, infos

#from scripts.converter import convert_h5_to_npz

# -----------------------------------------------------
# 1. Load Expert Data (HDF5 format)
# -----------------------------------------------------

def load_npz_trajectories(path):
    """Load trajectories stored in NPZ format."""
    trajectories = []
    data = np.load(path, allow_pickle=True)

    obs = data["observations"]
    acts = data["actions"]
    dones = data["dones"]

    start_idx = 0
    for i, done in enumerate(dones):
        if done:
            trajectories.append(
                Trajectory(
                    obs=obs[start_idx:i+1],
                    acts=acts[start_idx:i],
                    infos=None,
                    terminal=True,
                )
            )
            start_idx = i+1
    return trajectories


# -----------------------------------------------------
# 1. Parse Arguments First
# -----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--expert-npz', default="/home/chris/Imitation/trajectories/test2.1_delta_robosuite.npz")
parser.add_argument('--n-steps', type=int, default=None, help='RL training steps (override FAST preset)')
parser.add_argument('--show', action='store_true', help='Show Robosuite GUI during training (slows down training)')
args = parser.parse_args()

if args.show:
    print("\n⚠️  GUI rendering enabled! Training will be MUCH slower.")
    print("   Press Ctrl+C to stop training early.\n")


# -----------------------------------------------------
# 2. Create Robosuite Environment Wrapper
# -----------------------------------------------------

def make_franka_env(rank=0):
    """
    Create Robosuite environment.
    Only rank 0 (first environment) gets a renderer to avoid segfaults.
    """
    env = suite.make(
        env_name="Lift",  # Franka cube lifting task
        robots="Panda",   # Franka Panda robot
        has_renderer=(args.show and rank == 0),  # Only first env renders
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
    )
    wrapped = GymWrapper(env)
    return wrapped


# Vectorized environment using DummyVecEnv
# Reproducibility / seeding
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Vectorized environment using DummyVecEnv (pass rank to each env)
venv = DummyVecEnv([lambda i=i: make_franka_env(rank=i) for i in range(4)])
venv.seed(SEED)

# Wrap with rendering if --show flag is set
if args.show:
    venv = RenderWrapper(venv)
    print("✓ Rendering enabled for first environment")


# -----------------------------------------------------
# 3. Expert Trajectories
# -----------------------------------------------------

expert_trajectories = load_npz_trajectories(args.expert_npz)


# -----------------------------------------------------
# 4. Define RL Policy & AIRL
# -----------------------------------------------------

# Use explicit PPO hyperparameters (more stable and reproducible)
policy = PPO(
    env=venv,
    policy=MlpPolicy,
    verbose=1,
    batch_size=128,
    ent_coef=0.01,  # Add entropy bonus to keep policy stochastic after BC pretraining
    learning_rate=0.001,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
)

#reward_net = BasicRewardNet(
#    observation_space=venv.observation_space,
#    action_space=venv.action_space,
#    hidden_sizes=(64, 64),
#)

env = make_franka_env()
print("Generated observation shape:", env.observation_space.shape)
# Quick validation: ensure expert obs dim matches environment obs dim
if len(expert_trajectories) == 0:
    raise RuntimeError(f'No expert trajectories loaded; check NPZ path {args.expert_npz}')

# Flatten env observation size
env_obs_dim = int(np.prod(env.observation_space.shape))
expert_obs_dim = int(np.ravel(expert_trajectories[0].obs[0]).shape[0])
print(f"Expert obs dim: {expert_obs_dim}; Env obs dim: {env_obs_dim}")
if expert_obs_dim != env_obs_dim:
    raise ValueError(
        f"Expert observation dimension ({expert_obs_dim}) does not match environment observation dimension ({env_obs_dim}).\n"
        "If you intended to use the robosuite-53 representation, create an aligned dataset first using the helper script:\n"
        "  conda run -n env_imitation python3 scripts/align_expert_obs.py --to-robosuite --npz-in trajectories/merged_real_dataset_1.1to1.6.npz --hdf5 trajectories/merged_real_dataset_1.1to1.6.hdf5 --npz-out trajectories/merged_real_dataset_1.1to1.6_robosuite.npz\n"
    )


reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    reward_hid_sizes=(32,),  # MUCH smaller: single layer with 32 units
    potential_hid_sizes=(32,),  # MUCH smaller: single layer with 32 units
    use_state=True,
    use_action=True,
    use_next_state=False,  # AIRL: reward should only depend on (s,a), not next state
    use_done=False,
    discount_factor=0.95,  # Match PPO's gamma
    normalize_input_layer=RunningNorm,
)

# -----------------------------------------------------
# AIRL Trainer Setup (NO BC Pretraining!)
# -----------------------------------------------------
# Note: BC pretraining was removed because it makes the policy too deterministic,
# which prevents the discriminator from learning properly.

airl_trainer = airl.AIRL(
    demonstrations=expert_trajectories,
    venv=venv,
    gen_algo=policy,
    # MUCH smaller batches to slow down discriminator learning
    demo_batch_size=256,  # Reduced from 1024 to 256
    demo_minibatch_size=64,  # Process in even smaller chunks
    gen_replay_buffer_capacity=4096,
    # VERY REDUCED discriminator updates to prevent discriminator collapse
    n_disc_updates_per_round=1,  # Reduced from 4 to 1
    reward_net=reward_net,
    # Higher discriminator learning rate but with strong regularization
    disc_opt_kwargs=dict(lr=1e-4, weight_decay=1e-3),  # Increased weight_decay for stronger regularization
)

# -----------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------
#print("Expert observation shape:", expert_trajectories[0].obs.shape)
#print("Generated observation shape:", env.observation_space.shape)
#print("Starting AIRL training...")


#airl_trainer.train(n_epochs=50)
# Use a named constant for RL training budget
FAST = True  # False
if args.n_steps is not None:
    N_RL_TRAIN_STEPS = args.n_steps
else:
    N_RL_TRAIN_STEPS = 100_000 if FAST else 2_000_000

# Evaluate learner before training
venv.seed(SEED)
try:
    learner_rewards_before_training, _ = evaluate_policy(policy, venv, 10, return_episode_rewards=True)
    print(f"Learner mean reward before training: {np.mean(learner_rewards_before_training):.2f}")
except Exception as e:
    print("Warning: evaluate_policy before training failed:", e)

airl_trainer.train(N_RL_TRAIN_STEPS)

# Evaluate learner after training (on AIRL-wrapped env)
venv.seed(SEED)
try:
    learner_rewards_after_training, _ = evaluate_policy(policy, venv, 10, return_episode_rewards=True)
    print(f"Learner mean reward after training (AIRL-wrapped): {np.mean(learner_rewards_after_training):.2f}")
except Exception as e:
    print("Warning: evaluate_policy after training failed:", e)

# Additional post-training evaluations: raw env reward vs AIRL reward on fresh eval envs
print("\n=== Post-Training Evaluation (10 episodes each) ===")
try:
    # 1. Raw environment reward (no AIRL wrapper)
    eval_venv_raw = DummyVecEnv([make_franka_env for _ in range(4)])
    eval_venv_raw.seed(SEED)
    raw_rewards, _ = evaluate_policy(policy, eval_venv_raw, 10, return_episode_rewards=True)
    print(f"Raw env reward: {np.mean(raw_rewards):.2f} +/- {np.std(raw_rewards):.2f}")
    
    # 2. AIRL learned reward (wrapped with trained reward_net)
    eval_venv_airl = DummyVecEnv([make_franka_env for _ in range(4)])
    eval_venv_airl = RewardVecEnvWrapper(eval_venv_airl, reward_net.predict_processed)
    eval_venv_airl.seed(SEED)
    airl_rewards, _ = evaluate_policy(policy, eval_venv_airl, 10, return_episode_rewards=True)
    print(f"AIRL learned reward: {np.mean(airl_rewards):.2f} +/- {np.std(airl_rewards):.2f}")
    print("===================================================\n")
except Exception as e:
    print(f"Warning: post-training dual evaluation failed: {e}\n")

# Print mean +/- std in requested format if evaluations succeeded
if 'learner_rewards_before_training' in globals():
    print(
        "Rewards before training:",
        np.mean(learner_rewards_before_training),
        "+/-",
        np.std(learner_rewards_before_training),
    )
else:
    print("Rewards before training: (evaluation failed or not run)")

if 'learner_rewards_after_training' in globals():
    print(
        "Rewards after training:",
        np.mean(learner_rewards_after_training),
        "+/-",
        np.std(learner_rewards_after_training),
    )
else:
    print("Rewards after training: (evaluation failed or not run)")

# Save trained policy and reward net to a timestamped folder so we don't lose artifacts
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("output", "manual_runs", f"franka_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

# policy
policy_path = os.path.join(out_dir, "gen_policy")
policy.save(policy_path)

# reward net: save state_dict and full object where possible
try:
    reward_state_path = os.path.join(out_dir, "reward_net_state.pth")
    torch.save(reward_net.state_dict(), reward_state_path)
    # also try to save the whole object (may fail if there are lambdas or non-picklable members)
    reward_full_path = os.path.join(out_dir, "reward_net_full.pth")
    torch.save(reward_net, reward_full_path)
    print(f"Saved reward net state to {reward_state_path} and full object to {reward_full_path}")
except Exception as e:
    print("Warning: saving full reward_net object failed:", e)
    print(f"Reward state was saved to {reward_state_path} if available.")

print(f"Training complete. Artifacts saved under {out_dir}")

