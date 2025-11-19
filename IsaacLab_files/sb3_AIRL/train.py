# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
python scripts/reinforcement_learning/sb3_AIRL/train.py   --task=Isaac-Lift-Cube-Franka-v0   --agent=sb3_cfg_entry_point   --num_envs=1   --max_iterations=200   --seed=42
"""

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

# (AIRL discriminator removed from this file; trainGEN.py contains the AIRL training flow)

# (expert-data loader removed; AIRL-style training is handled in trainGEN.py)


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# parser.add_argument("--expert_data", type=str, default=None, help="Path to expert HDF5 data file (observations/actions).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

import omni
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize
#mps Hier Imitation AIRL importieren

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import traceback


def try_load_reward_net_from_dir(rn_dir: str, device: str | torch.device = "cpu"):
    """Search for a TorchScript (.pt) model in rn_dir and load it. If none found, try .pth via torch.load.
    Returns a callable module.
    """
    device = torch.device(device)
    # prefer *.pt
    if not os.path.exists(rn_dir):
        raise FileNotFoundError(rn_dir)
    # find any .pt
    for fname in os.listdir(rn_dir):
        if fname.endswith('.pt'):
            p = os.path.join(rn_dir, fname)
            try:
                mod = torch.jit.load(p, map_location=device)
                print(f"[INFO] Loaded TorchScript reward net from {p}")
                return mod
            except Exception as e:
                print(f"[WARN] torch.jit.load failed for {p}: {e}")
    # fallback: look for full pth pickles
    for fname in os.listdir(rn_dir):
        if fname.endswith('.pth') or fname.endswith('.pt.tar'):
            p = os.path.join(rn_dir, fname)
            try:
                mod = torch.load(p, map_location=device)
                # if this is a dict-like state dict, we cannot instantiate automatically
                if isinstance(mod, dict) and any(k in mod for k in ("state_dict", "model_state_dict", "params")):
                    raise RuntimeError("Found state-dict like archive; please provide a TorchScript (.pt) or full module.")
                try:
                    mod.eval()
                except Exception:
                    pass
                print(f"[INFO] Loaded reward net module from {p}")
                return mod
            except Exception as e:
                print(f"[WARN] torch.load failed for {p}: {e}")
    raise FileNotFoundError(f"No usable reward net found in {rn_dir}")

# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Safety clamp for number of environments to avoid huge memory allocations on a single machine.
    # Very large values (e.g. thousands) can cause malloc failures in native libraries or exhaust GPU/CPU memory.
    MAX_ENVS = 128
    try:
        if int(env_cfg.scene.num_envs) > MAX_ENVS:
            print(f"[WARN] Requested num_envs={env_cfg.scene.num_envs} is large — clamping to {MAX_ENVS} to avoid OOM or malloc errors."
                  " If you really want more, increase this limit in the script.")
            env_cfg.scene.num_envs = MAX_ENVS
    except Exception:
        # If env_cfg.scene.num_envs is not an int or missing, ignore
        pass

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")


    #mps Dieser Teil existiert im Vorschlag nicht. START
    # set the IO descriptors export flag if requested
    #if isinstance(env_cfg, ManagerBasedRLEnvCfg):
    #    env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    #else:
    #    omni.log.warn(
    #        "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
    #    )
    #mps Dieser Teil existiert im Vorschlag nicht. ENDE
    
    
    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ---------- Attempt to load TorchScript reward net and wrap env ----------
    def _flatten_obs(obs):
        if isinstance(obs, dict):
            parts = []
            # First pass: infer batch size from any 2D tensor in the dict
            batch_size = None
            for k in obs.keys():
                v = obs[k]
                if torch.is_tensor(v):
                    arr = v.detach().cpu().numpy()
                else:
                    arr = np.array(v)
                if arr.ndim == 2:
                    batch_size = arr.shape[0]
                    break
            
            # Second pass: flatten all observations
            for k in sorted(obs.keys()):
                v = obs[k]
                if torch.is_tensor(v):
                    arr = v.detach().cpu().numpy()
                else:
                    arr = np.array(v)
                # ensure 2D - preserve batch dimension
                if arr.ndim == 0:
                    # Scalar
                    arr = arr.reshape(1, 1) if batch_size is None else np.full((batch_size, 1), arr)
                elif arr.ndim == 1:
                    # Ambiguous: could be (batch,) or (features,)
                    # If we detected batch_size from other obs, use it
                    if batch_size is not None and arr.shape[0] == batch_size:
                        # This is (batch,) with 1 feature each
                        arr = arr.reshape(-1, 1)
                    else:
                        # This is (features,) for single env
                        arr = arr.reshape(1, -1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                parts.append(arr)
            return np.concatenate(parts, axis=-1)
        else:
            if torch.is_tensor(obs):
                arr = obs.detach().cpu().numpy()
            else:
                arr = np.array(obs)
            # ensure 2D
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                # Assume (features,) for single env
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            return arr

    class RewardNetWrapper(gym.Wrapper):
        def __init__(self, env, reward_net, device="cpu", obs_adapter=None, gamma=0.99, log_dir: str = None):
            super().__init__(env)
            self.reward_net = reward_net
            self.device = torch.device(device)
            self.obs_adapter = obs_adapter
            self.gamma = float(gamma)
            self._last_obs = None
            # debug counters
            self._debug_steps = 0
            self._debug_max = 50
            # flag to print an informational message once when the reward net is first used
            self._reward_net_reported = False
            # store the run-specific log directory (if provided) so we can write marker files there
            self.log_dir = log_dir

        
        # default adapter builder: creates a function that maps the raw env observation
        # (dict or tensor) into the 53-d state vector expected by the reward net.
        @staticmethod
        def _build_default_adapter(target_state_dim: int = 53):
            import math

            def _get_array(x):
                # accept torch tensor or numpy or python scalar
                if torch.is_tensor(x):
                    a = x.detach().cpu().numpy()
                else:
                    a = np.array(x)
                # ensure 2D (batch, features)
                # Isaac Lab returns (num_envs, features) already, so typically a.ndim == 2
                # If ndim == 1, it's ambiguous but we assume it's features for a single env
                if a.ndim == 0:
                    # scalar
                    a = a.reshape(1, 1)
                elif a.ndim == 1:
                    # Assume (features,) for 1 env → (1, features)
                    a = a.reshape(1, -1)
                elif a.ndim > 2:
                    # Flatten extra dims
                    a = a.reshape(a.shape[0], -1)
                # a is now always 2D: (batch, features)
                return a

            def adapter(raw_obs):
                """Return torch.Tensor of shape (batch, target_state_dim).
                raw_obs: either a dict of arrays/tensors or a single tensor/ndarray.
                This implements the mapping you supplied (robot joints, cos/sin, vel, eef, cube, etc.).
                """
                # If the env returns (obs, info) tuple, allow passing the tuple
                if isinstance(raw_obs, tuple) and len(raw_obs) >= 1:
                    raw_obs = raw_obs[0]

                # Prefer dict-style observations
                if isinstance(raw_obs, dict):
                    o = raw_obs
                else:
                    # fallback: assume already flattened tensor/ndarray
                    arr = _get_array(raw_obs)
                    # if arr is (batch, N) and matches target dim, return tensor
                    if arr.shape[1] == target_state_dim:
                        return torch.tensor(arr, dtype=torch.float32)
                    # else we cannot auto-derive fields from flat array; return zeros padded
                    pad = np.zeros((arr.shape[0], target_state_dim - arr.shape[1])) if arr.shape[1] < target_state_dim else arr[:, :target_state_dim]
                    res = arr if arr.shape[1] >= target_state_dim else np.concatenate([arr, pad], axis=1)
                    return torch.tensor(res, dtype=torch.float32)

                # Infer batch size from any available observation
                batch_size = 1
                for k, v in o.items():
                    try:
                        if torch.is_tensor(v):
                            arr = v.detach().cpu().numpy()
                        else:
                            arr = np.array(v)
                        if arr.ndim >= 1:
                            batch_size = arr.shape[0]
                            break
                    except Exception:
                        continue

                # helper to read possible keys
                def read_key(preferred, fallback_shape=None):
                    for k in preferred:
                        if k in o:
                            try:
                                a = _get_array(o[k])
                                # sanitize NaNs/Infs
                                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                                return a
                            except Exception:
                                continue
                    if fallback_shape is not None:
                        return np.zeros((batch_size, fallback_shape))
                    return None

                # 1) robot joint pos (7)
                joint_pos = read_key(["joint_pos", "robot0_joint_pos", "robot_joint_pos", "robot0_joint_position"], 7)
                if joint_pos is None:
                    joint_pos = np.zeros((batch_size, 7))

                # 2) joint vel (7)
                joint_vel = read_key(["joint_vel", "robot0_joint_vel", "robot_joint_vel"], 7)
                if joint_vel is None:
                    joint_vel = np.zeros((joint_pos.shape[0], 7))

                # 3) eef pos (3)
                eef_pos = read_key(["eef_pos", "ee_frame_pos", "end_effector_pos", "robot0_eef_pos"], 3)
                if eef_pos is None:
                    eef_pos = np.zeros((joint_pos.shape[0], 3))

                # 4) eef quat (4)
                eef_quat = read_key(["eef_quat", "ee_frame_quat", "end_effector_quat", "robot0_eef_quat"], 4)
                if eef_quat is None:
                    eef_quat = np.zeros((joint_pos.shape[0], 4))

                # 5) gripper qpos approx from gripper_pos
                gripper_pos = read_key(["gripper_pos", "robot0_gripper_qpos"], 2)
                if gripper_pos is None:
                    gripper_pos = np.zeros((joint_pos.shape[0], 2))

                # 6) cube position/quaternion: try a few possible keys and be robust to multiple objects
                cube_pos = None
                cube_quat = None

                # PRIORITY: Check for IsaacLab's observation keys first
                cube_pos = read_key(["object_position"], 3)
                cube_quat = read_key(["object_orientation"], 4)

                def _choose_closest(arr, per_elem):
                    """Given arr shaped (B, K), try to select a single element of width per_elem per batch.
                    If K == per_elem -> return arr
                    If K % per_elem == 0 -> reshape to (B, N, per_elem) and pick item closest to eef_pos (if available)
                    Else if K > per_elem -> take first per_elem
                    Else pad with zeros to per_elem
                    """
                    if arr is None:
                        return None
                    B = arr.shape[0]
                    K = arr.shape[1]
                    if K == per_elem:
                        return arr
                    if K % per_elem == 0:
                        N = K // per_elem
                        try:
                            arr3 = arr.reshape(B, N, per_elem)
                        except Exception:
                            # fallback to taking first per_elem
                            if K >= per_elem:
                                return arr[:, :per_elem]
                            pad = np.zeros((B, per_elem - K))
                            return np.concatenate([arr, pad], axis=1)
                        # if we have an eef position, pick the closest object per batch
                        try:
                            eef = eef_pos.reshape(B, 1, 3)
                            dists = np.linalg.norm(arr3 - eef, axis=2)
                            idx = np.argmin(dists, axis=1)
                            chosen = arr3[np.arange(B), idx, :]
                            return chosen
                        except Exception:
                            return arr3[:, 0, :]
                    if K > per_elem:
                        return arr[:, :per_elem]
                    # K < per_elem
                    pad = np.zeros((B, per_elem - K))
                    return np.concatenate([arr, pad], axis=1)

                if "cube_positions" in o:
                    cp = read_key(["cube_positions"])  # could be (B,3) or (B,3*M) etc
                    cube_pos = _choose_closest(cp, 3)
                elif "cube_positions_in_world_frame" in o:
                    cp = read_key(["cube_positions_in_world_frame"])
                    cube_pos = _choose_closest(cp, 3)
                elif "object" in o:
                    ob = read_key(["object"])  # object_obs: maybe multiple objects flattened
                    if ob is not None and ob.shape[1] >= 7:
                        # try detect multiple objects: if width%7==0, reshape
                        try:
                            if ob.shape[1] % 7 == 0:
                                N = ob.shape[1] // 7
                                ob3 = ob.reshape(ob.shape[0], N, 7)
                                # pick closest to eef if possible
                                try:
                                    eef = eef_pos.reshape(ob.shape[0], 1, 3)
                                    dists = np.linalg.norm(ob3[:, :, :3] - eef, axis=2)
                                    idx = np.argmin(dists, axis=1)
                                    cube_pos = ob3[np.arange(ob.shape[0]), idx, :3]
                                    cube_quat = ob3[np.arange(ob.shape[0]), idx, 3:7]
                                except Exception:
                                    cube_pos = ob[:, :3]
                                    cube_quat = ob[:, 3:7]
                            else:
                                cube_pos = ob[:, :3]
                                cube_quat = ob[:, 3:7]
                        except Exception:
                            cube_pos = ob[:, :3]
                            cube_quat = ob[:, 3:7]
                # also try cube_orientations
                if cube_quat is None:
                    if "cube_orientations" in o:
                        cq = read_key(["cube_orientations"])
                        cube_quat = _choose_closest(cq, 4)

                # If we still don't have cube_pos/quats, try 'cube_single' or single 'object' keys
                if cube_pos is None:
                    # try keys with 'cube_single'
                    for k in ["cube_single_pos", "cube_single_position", "cube_single"]:
                        if k in o:
                            val = read_key([k])
                            cube_pos = _choose_closest(val, 3)
                            break
                if cube_quat is None:
                    for k in ["cube_single_quat", "cube_single_orientation"]:
                        if k in o:
                            val = read_key([k])
                            cube_quat = _choose_closest(val, 4)

                if cube_pos is None:
                    cube_pos = np.zeros((joint_pos.shape[0], 3))
                if cube_quat is None:
                    cube_quat = np.zeros((joint_pos.shape[0], 4))

                # compute gripper_to_cube = eef_pos - cube_pos  
                gripper_to_cube = eef_pos - cube_pos

                # Now build robot block according to the user's spec:
                # robot0_joint_pos (7)
                # robot0_joint_pos_cos (7)
                # robot0_joint_pos_sin (7)
                # robot0_joint_vel (7)
                # robot0_eef_pos (3)
                # robot0_eef_quat (4)
                # robot0_eef_quat_site (4)  (duplicate of eef_quat)
                # robot0_gripper_qpos (2)
                # robot_gripper_qvel (2)

                jp = joint_pos
                jp_cos = np.cos(jp)
                jp_sin = np.sin(jp)
                jvel = joint_vel
                epos = eef_pos
                equat = eef_quat
                equat_site = equat
                gqpos = gripper_pos[:, :2]
                gqvel = np.zeros((joint_pos.shape[0], 2))

                robot_block = np.concatenate([jp, jp_cos, jp_sin, jvel, epos, equat, equat_site, gqpos, gqvel], axis=1)

                # cube block: cube_pos (3), cube_quat (4), gripper_to_cube (3)
                cube_block = np.concatenate([cube_pos, cube_quat, gripper_to_cube], axis=1)

                state = np.concatenate([robot_block, cube_block], axis=1)
                # ensure correct width
                if state.shape[1] > target_state_dim:
                    state = state[:, :target_state_dim]
                elif state.shape[1] < target_state_dim:
                    pad = np.zeros((state.shape[0], target_state_dim - state.shape[1]))
                    state = np.concatenate([state, pad], axis=1)

                # final sanitization: replace NaN/Inf and ensure shape
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                if state.shape[1] > target_state_dim:
                    state = state[:, :target_state_dim]
                elif state.shape[1] < target_state_dim:
                    pad = np.zeros((state.shape[0], target_state_dim - state.shape[1]))
                    state = np.concatenate([state, pad], axis=1)

                t = torch.tensor(state, dtype=torch.float32, device=torch.device('cpu'))
                return t
            # expose target dim for callers
            adapter._target_state_dim = target_state_dim
            return adapter

        def reset(self, **kwargs):
            out = self.env.reset(**kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                obs, info = out
            else:
                obs = out
                info = {}
            self._last_obs = obs
            return (obs, info) if info is not None else obs

        def step(self, action):
            step_out = self.env.step(action)
            if len(step_out) == 5:
                next_obs, env_reward, terminated, truncated, info = step_out
            else:
                next_obs, env_reward, done, info = step_out
                terminated = done
                truncated = False

            try:
                # If obs_adapter is provided, use it on the raw obs dicts (preferred), otherwise
                # fall back to the generic flattening logic.
                if self.obs_adapter is not None:
                    # obs_adapter should accept raw obs (dict or tuple) and return torch.Tensor(batch, state_dim)
                    try:
                        obs_t = self.obs_adapter(self._last_obs)
                        next_obs_t = self.obs_adapter(next_obs)
                        # adapter MAY return numpy arrays or torch tensors on CPU — normalize to torch tensor on wrapper device
                        target_dim = getattr(self.obs_adapter, '_target_state_dim', None)
                        def _ensure_tensor(x):
                            if torch.is_tensor(x):
                                t = x.to(self.device).to(torch.float32)
                            else:
                                arr = np.array(x)
                                if arr.ndim == 1:
                                    arr = arr[None, ...]
                                t = torch.tensor(arr, dtype=torch.float32, device=self.device)
                            return t

                        obs_t = _ensure_tensor(obs_t)
                        next_obs_t = _ensure_tensor(next_obs_t)

                        # validate width if adapter provided a target
                        if target_dim is not None:
                            if obs_t.ndim == 1:
                                obs_t = obs_t.unsqueeze(0)
                            if next_obs_t.ndim == 1:
                                next_obs_t = next_obs_t.unsqueeze(0)
                            if obs_t.shape[1] != target_dim or next_obs_t.shape[1] != target_dim:
                                # Adapter produced an unexpected width (can happen for unusual spawns).
                                # Instead of aborting immediately, fallback to generic flattening to try to recover.
                                try:
                                    print(f"[WARN] Adapter returned width obs={obs_t.shape[1]} next_obs={next_obs_t.shape[1]} expected={target_dim}. Falling back to generic flattening for this step.")
                                except Exception:
                                    pass
                                # fallback to flattening
                                obs_np = _flatten_obs(self._last_obs)
                                next_obs_np = _flatten_obs(next_obs)
                                if obs_np.ndim == 1:
                                    obs_np = obs_np[None, ...]
                                    next_obs_np = next_obs_np[None, ...]
                                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                                next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
                    except Exception:
                        # fallback to flattening if adapter fails
                        obs_np = _flatten_obs(self._last_obs)
                        next_obs_np = _flatten_obs(next_obs)
                        if obs_np.ndim == 1:
                            obs_np = obs_np[None, ...]
                            next_obs_np = next_obs_np[None, ...]
                        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                        next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
                else:
                    obs_np = _flatten_obs(self._last_obs)
                    next_obs_np = _flatten_obs(next_obs)
                    acts_np = action if not torch.is_tensor(action) else action.detach().cpu().numpy()

                    if obs_np.ndim == 1:
                        obs_np = obs_np[None, ...]
                        next_obs_np = next_obs_np[None, ...]
                        acts_np = np.array(acts_np)[None, ...]

                    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                    acts_t = torch.tensor(acts_np, dtype=torch.float32, device=self.device)
                    next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)

                # If obs_adapter is provided and we haven't created acts_t yet, create acts_t from action
                if 'acts_t' not in locals():
                    acts_np = action if not torch.is_tensor(action) else action.detach().cpu().numpy()
                    if isinstance(acts_np, np.ndarray) and acts_np.ndim == 1:
                        acts_np = acts_np[None, ...]
                    acts_t = torch.tensor(acts_np, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    out = None
                    model_reward_val = None
                    used_model = False
                    # prepare a 'done' tensor matching the batch size — some ScriptModules expect (state, action, next_state, done)
                    try:
                        batch = int(obs_t.shape[0])
                    except Exception:
                        batch = 1
                    # create done tensor (float32) on the same device
                    done_t = torch.zeros(batch, dtype=torch.float32, device=self.device)

                    # normalize and prepare action tensor to expected width
                    try:
                        # ensure acts_t exists, is float tensor on correct device and has batch dim
                        if not torch.is_tensor(acts_t):
                            acts_t = torch.tensor(np.array(acts_t), dtype=torch.float32, device=self.device)
                        else:
                            acts_t = acts_t.to(self.device).to(torch.float32)
                        if acts_t.ndim == 1:
                            acts_t = acts_t.unsqueeze(0)
                    except Exception:
                        # best-effort conversion
                        acts_np = np.array(acts_t)
                        if acts_np.ndim == 1:
                            acts_np = acts_np[None, ...]
                        acts_t = torch.tensor(acts_np, dtype=torch.float32, device=self.device)

                    # The trained reward-net expects action dim == 7 (arm only). Slice if extra dims (e.g. gripper at idx -1) exist.
                    acts_for_model = acts_t
                    try:
                        if acts_t.ndim == 2 and acts_t.shape[1] > 7:
                            acts_for_model = acts_t[:, :7].contiguous()
                    except Exception:
                        acts_for_model = acts_t

                    # ensure obs_t/next_obs_t are float tensors on correct device with batch dim
                    if not torch.is_tensor(obs_t):
                        obs_t = torch.tensor(np.array(obs_t), dtype=torch.float32, device=self.device)
                    else:
                        obs_t = obs_t.to(self.device).to(torch.float32)
                    if obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)
                    if not torch.is_tensor(next_obs_t):
                        next_obs_t = torch.tensor(np.array(next_obs_t), dtype=torch.float32, device=self.device)
                    else:
                        next_obs_t = next_obs_t.to(self.device).to(torch.float32)
                    if next_obs_t.ndim == 1:
                        next_obs_t = next_obs_t.unsqueeze(0)

                    # Debug: print shapes and dtypes before calling model
                    if self._debug_steps < self._debug_max:
                        try:
                            print(f"[REWARD_DEBUG][SHAPES] obs_t={tuple(obs_t.shape)} acts_t={tuple(acts_t.shape)} acts_for_model={tuple(acts_for_model.shape)} next_obs_t={tuple(next_obs_t.shape)} done_t={tuple(done_t.shape)} device={self.device} dtype={obs_t.dtype}")
                        except Exception:
                            pass

                    # 1) preferred: get_reward with done (if available)
                    if hasattr(self.reward_net, 'get_reward'):
                        try:
                            out = self.reward_net.get_reward(obs_t, acts_for_model, next_obs_t, done_t)
                            used_model = out is not None
                        except Exception as e:
                            raise RuntimeError(f"Reward net get_reward raised an exception: {e}\n{traceback.format_exc()}")

                    # 2) try forward(state, action, next_state, done)
                    if out is None:
                        try:
                            out = self.reward_net(obs_t, acts_for_model, next_obs_t, done_t)
                            used_model = out is not None
                        except Exception as e:
                            raise RuntimeError(f"Reward net forward(state,action,next_state,done) raised an exception: {e}\n{traceback.format_exc()}")

                    # 3) legacy AIRL signature: (obs, acts, next_obs, gamma) - try with gamma float as fallback
                    if out is None:
                        try:
                            out = self.reward_net(obs_t, acts_for_model, next_obs_t, float(self.gamma))
                            used_model = out is not None
                        except Exception as e:
                            raise RuntimeError(f"Reward net forward(obs,acts,next_obs,gamma) raised an exception: {e}\n{traceback.format_exc()}")

                    # 4) other fallbacks (less likely to work for this model)
                    if out is None:
                        try:
                            out = self.reward_net(obs_t, acts_for_model)
                            used_model = out is not None
                        except Exception as e:
                            raise RuntimeError(f"Reward net forward(obs,acts) raised an exception: {e}\n{traceback.format_exc()}")
                    if out is None:
                        try:
                            out = self.reward_net(obs_t)
                            used_model = out is not None
                        except Exception as e:
                            raise RuntimeError(f"Reward net forward(obs) raised an exception: {e}\n{traceback.format_exc()}")

                # If model failed to produce an output, raise an error and stop — no fallback to environment reward
                if out is None:
                    raise RuntimeError(f"Reward net returned no output (None) at step {self._debug_steps}; aborting training.")
                else:
                    # Convert model output to a 1-D torch tensor of shape (batch,)
                    try:
                        if torch.is_tensor(out):
                            out_t = out.to(self.device).to(torch.float32)
                        else:
                            out_t = torch.tensor(np.array(out), dtype=torch.float32, device=self.device)
                        # reshape to (batch,) if possible
                        if out_t.ndim == 0:
                            out_t = out_t.unsqueeze(0)
                        elif out_t.ndim > 1 and out_t.shape[1] == 1:
                            out_t = out_t.view(out_t.shape[0])
                        elif out_t.ndim > 1 and out_t.shape[0] == 1:
                            out_t = out_t.reshape(-1)
                    except Exception:
                        raise RuntimeError(f"Failed to convert reward net output to tensor: {out}\n{traceback.format_exc()}")

                    # For debug printing, capture scalar/array view
                    try:
                        model_reward_val = float(out_t[0].detach().cpu().numpy()) if out_t.numel() == 1 else out_t.detach().cpu().numpy()
                    except Exception:
                        model_reward_val = None

                    # final reward is the tensor out_t (shape (batch,))
                    reward = out_t

                    # If the model returned NaN or Inf, treat as fatal and dump the observation for debugging
                    try:
                        if not torch.isfinite(reward).all():
                            # dump and raise
                            try:
                                import time, pickle
                                logs_root = os.path.join('logs', 'sb3', getattr(self.env, 'spec', {}).get('id', 'unknown_task'))
                                if not os.path.exists(logs_root):
                                    try:
                                        os.makedirs(logs_root, exist_ok=True)
                                    except Exception:
                                        logs_root = '.'
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                dump_path = os.path.join(logs_root, f'nonfinite_reward_{ts}.pkl')
                                with open(dump_path, 'wb') as f:
                                    pickle.dump({'last_obs': self._last_obs, 'next_obs': next_obs, 'action': action, 'reward': reward.detach().cpu().numpy().tolist()}, f)
                                print(f"[ERROR] RewardNetWrapper detected non-finite reward and dumped obs to: {dump_path}")
                            except Exception:
                                pass
                            raise RuntimeError('Reward net returned non-finite values; aborting training.')
                    except Exception:
                        # if torch functions fail here, propagate the original out->reward conversion error below
                        pass

                # debug print for first N steps: show env reward and model reward
                try:
                    if self._debug_steps < self._debug_max:
                        print(f"[REWARD_DEBUG] step={self._debug_steps} env_reward={env_reward} model_reward={model_reward_val} used_model={used_model}")
                        self._debug_steps += 1
                except Exception:
                    pass
                # Print a one-time info line when we actually use the learned reward net
                try:
                    if used_model and not getattr(self, '_reward_net_reported', False):
                        print("INFO: reward_net is being used")
                        # persist a small marker file into the logs directory so the usage
                        # can be detected after the run even if stdout didn't show the print.
                        try:
                            # Prefer to write into the run-specific log_dir if available
                            logs_root = None
                            try:
                                if getattr(self, 'log_dir', None):
                                    logs_root = self.log_dir
                                else:
                                    logs_root = os.path.join('logs', 'sb3', getattr(self.env, 'spec', {}).get('id', 'unknown_task'))
                                os.makedirs(logs_root, exist_ok=True)
                                ts = datetime.now().isoformat()
                                with open(os.path.join(logs_root, 'reward_net_used.txt'), 'a') as _f:
                                    _f.write(f"{ts}\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        self._reward_net_reported = True
                except Exception:
                    pass
            except Exception as e:
                # Any exception during reward computation is fatal when using a learned reward net
                # Dump the failing observations and action to disk to help debugging variable spawn/obs layouts
                try:
                    import time, pickle
                    logs_root = os.path.join('logs', 'sb3', getattr(self.env, 'spec', {}).get('id', 'unknown_task'))
                    # Fallback if env.spec is not available
                    if not os.path.exists(logs_root):
                        try:
                            os.makedirs(logs_root, exist_ok=True)
                        except Exception:
                            logs_root = '.'
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dump_path = os.path.join(logs_root, f'failing_obs_{ts}.pkl')
                    with open(dump_path, 'wb') as f:
                        pickle.dump({'last_obs': self._last_obs, 'next_obs': next_obs, 'action': action, 'error': str(e), 'trace': traceback.format_exc()}, f)
                    print(f"[ERROR] RewardNetWrapper dumped failing obs to: {dump_path}")
                except Exception:
                    try:
                        print('[ERROR] Failed to dump failing obs for RewardNetWrapper')
                    except Exception:
                        pass
                raise RuntimeError(f"RewardNetWrapper failed to compute reward and fallback is disabled. Error: {e}\n{traceback.format_exc()}")

            # Ensure we return a torch.Tensor reward (expected by downstream Vec wrappers)
            try:
                if not torch.is_tensor(reward):
                    reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device)
                else:
                    # move to wrapper device and ensure dtype
                    reward = reward.to(self.device).to(torch.float32)
            except Exception:
                # last resort: wrap scalar
                reward = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)

            # debug reward type/shape for first steps
            #if self._debug_steps <= self._debug_max:
            #    try:
            #        print(f"[REWARD_DEBUG][RET] reward_type={type(reward)} shape={tuple(reward.shape) if torch.is_tensor(reward) else None} device={getattr(reward, 'device', None)}")
            #    except Exception:
            #        pass

            self._last_obs = next_obs
            return next_obs, reward, terminated, truncated, info

    # Enforce using a learned reward net only. Any failure to load or wrap should
    # abort training immediately (no fallback to the environment's built-in reward).
    rn_dir = os.path.join("reward_nets")
    try:
        device = getattr(env.unwrapped, "device", "cpu") if hasattr(env, 'unwrapped') else 'cpu'
        reward_net = try_load_reward_net_from_dir(rn_dir, device=device)
        # build default adapter that maps raw env obs -> 53-d state expected by reward net
        try:
            default_adapter = RewardNetWrapper._build_default_adapter(53)
        except Exception:
            default_adapter = None
        env = RewardNetWrapper(env, reward_net, device=device, obs_adapter=default_adapter, gamma=agent_cfg.get("gamma", 0.99), log_dir=log_dir)
        print(f"[INFO] Wrapped environment with learned reward net from {rn_dir}")
        if default_adapter is not None:
            print("[INFO] Using default 53-d observation adapter for reward net")
    except Exception as e:
        # Fatal error: do not continue training with environment reward. Provide a clear message.
        try:
            sys.stderr.write(f"[ERROR] Failed to load or wrap learned reward net from '{rn_dir}': {e}\n")
        except Exception:
            pass
        # print traceback to stderr for debugging
        try:
            traceback.print_exc()
        except Exception:
            pass
        # ensure simulator is closed and exit with non-zero code
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.exit(3)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    #mps Hier wird PPO gestartet, dies durch AIRL Start ersetzen.
    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    #mps Expertendaten laden START (optional)
    # expert_data / AIRL discriminator training is not used in this script.
    expert_data = None
    #mps Expertendaten laden ENDE
    
    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]
#mps Worked til here--------------------------------------------------------------------------------------
    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # Validate required CLI args early to give a clear error message instead of a cryptic AttributeError
    if not getattr(args_cli, "task", None):
        print("[ERROR] Missing required argument: --task. Please call the script with e.g. --task=franka_lift_cube --agent=sb3_cfg_entry_point")
        simulation_app.close()
        sys.exit(2)

    # run the main function
    main()
    # close sim app
    simulation_app.close()



#Rest von Vorschlag
#obs = env.reset()
#    episode_obs, episode_acts, episode_next_obs = [], [], []
#
#    try:
#        for step in range(n_timesteps):
#            actions, _ = agent.predict(obs)
#            next_obs, _, done, infos = env.step(actions)
#            episode_obs.append(obs)
#            episode_acts.append(actions)
#            episode_next_obs.append(next_obs)
#            obs = next_obs
#
#            if done:
#                obs = env.reset()
#
#            if len(episode_obs) >= 256:
#                obs_batch = torch.tensor(np.array(episode_obs), dtype=torch.float32)
#                acts_batch = torch.tensor(np.array(episode_acts), dtype=torch.float32)
#                next_obs_batch = torch.tensor(np.array(episode_next_obs), dtype=torch.float32)
#
#                expert_obs, expert_acts, expert_next_obs = expert_data
#                f_exp = discriminator(expert_obs, expert_acts, expert_next_obs, agent_cfg["gamma"])
#                f_gen = discriminator(obs_batch, acts_batch, next_obs_batch, agent_cfg["gamma"])
#                loss = -(torch.log(torch.sigmoid(f_exp)).mean() + torch.log(1 - torch.sigmoid(f_gen)).mean())
#
#                disc_optimizer.zero_grad()
#                loss.backward()
#                disc_optimizer.step()
#                episode_obs, episode_acts, episode_next_obs = [], [], []
#
#            rewards = discriminator.get_reward(
#                torch.tensor(obs, dtype=torch.float32),
#                torch.tensor(actions, dtype=torch.float32),
#                torch.tensor(next_obs, dtype=torch.float32),
#                agent_cfg["gamma"]
#            ).cpu().numpy()
#            infos[0]["reward"] = rewards
#            agent.learn(total_timesteps=step + 1, callback=callbacks, progress_bar=True)
#
#    except KeyboardInterrupt:
#        print("Training interrupted.")
#    agent.save(os.path.join(log_dir,


