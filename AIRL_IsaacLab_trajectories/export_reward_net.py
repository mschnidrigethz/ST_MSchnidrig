#!/usr/bin/env python3
"""

Verwende dieses Script!!!

Create a traced TorchScript from a reward checkpoint and compare it to the original Python net.

Usage (example):
  PYTHONPATH=. python scripts/reexport_reward_net.py \
    --checkpoint output/manual_runs/franka_20251008_123837/reward_net_state.pth \
    --expert-npz trajectories/merged_real_dataset_1.1to1.6_for_training.npz \
    --n-samples 200
    
The script will:
- load the checkpoint (state_dict or full module) using the helper in scripts/check_reward_net.py
- reconstruct a Python RewardNet (or load the full model) and evaluate it on expert samples
- create a traced TorchScript using a representative input and save it as
  reward_net_ts_<timestamp>.pt inside the same folder as the checkpoint. The <timestamp>
  is extracted from the parent folder name (e.g. franka_20251008_123837 -> 20251008_123837).
- compare Python net outputs vs TorchScript outputs on N samples and print max/mean absolute diffs.

Notes:
- This script uses tracing (torch.jit.trace) by default as requested. A scripting fallback is not used.
- If the checkpoint is a pickled full module and loading fails, pass --allow-unsafe to permit unsafe unpickling.
"""

import argparse
import os
import re
import sys
from datetime import datetime
import numpy as np
import torch

# Import helpers from repo
from scripts.check_reward_net import load_reward_model, make_franka_env


def find_timestamp_from_path(path):
    # look for pattern YYYYMMDD_hhmmss in path
    m = re.search(r"(\d{8}_\d{6})", path)
    if m:
        return m.group(1)
    # fallback: try folder name like franka_YYYYMMDD_HHMMSS
    base = os.path.basename(os.path.abspath(path))
    m2 = re.search(r"(\d{8}_\d{6})", base)
    if m2:
        return m2.group(1)
    # final fallback: current timestamp
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def choose_trace_inputs(net, obs, acts):
    """Return a tuple of example inputs to trace the net with.
    Prefer (o,a,next_o,done) if supported; otherwise fall back to (o,a,next_o) or (o,a).
    obs: (T+1, obs_dim) or (T, obs_dim)
    acts: (T, act_dim)
    """
    o = torch.tensor(obs[0].astype(np.float32).reshape(1, -1))
    a = torch.tensor(acts[0].astype(np.float32).reshape(1, -1))
    next_o = None
    if obs.shape[0] >= 2:
        next_o = torch.tensor(obs[1].astype(np.float32).reshape(1, -1))
    done = torch.tensor([0.0], dtype=torch.float32)

    # Try calling net to see what it accepts
    try:
        # try the most complete signature
        net(o, a, next_o, done)
        return (o, a, next_o, done)
    except TypeError:
        pass
    try:
        net(o, a, next_o)
        return (o, a, next_o)
    except TypeError:
        pass
    # final fallback
    net(o, a)
    return (o, a)


def compare_models(py_net, ts_net, obs, acts, nsamples=200, device='cpu'):
    L = acts.shape[0]
    ns = min(nsamples, max(1, L - 1))
    inds = np.linspace(0, L - 2, ns, dtype=int)
    max_abs = 0.0
    sum_abs = 0.0
    cnt = 0
    for i in inds:
        o = torch.tensor(obs[i].astype(np.float32).reshape(1, -1), device=device)
        a = torch.tensor(acts[i].astype(np.float32).reshape(1, -1), device=device)
        next_o = None
        if obs.shape[0] >= i + 2:
            next_o = torch.tensor(obs[i + 1].astype(np.float32).reshape(1, -1), device=device)
        done = torch.tensor([0.0], dtype=torch.float32, device=device)

        with torch.no_grad():
            # python net
            try:
                out_py = py_net(o, a, next_o, done)
            except TypeError:
                try:
                    out_py = py_net(o, a, next_o)
                except Exception:
                    out_py = py_net(o, a)

            # torchscript net
            try:
                out_ts = ts_net(o, a, next_o, done)
            except TypeError:
                try:
                    out_ts = ts_net(o, a, next_o)
                except Exception:
                    out_ts = ts_net(o, a)

        out_py = torch.as_tensor(out_py).cpu()
        out_ts = torch.as_tensor(out_ts).cpu()
        err = float(torch.abs(out_py - out_ts).max().item())
        max_abs = max(max_abs, err)
        sum_abs += err
        cnt += 1

    return {'compared': cnt, 'max_abs': max_abs, 'mean_abs': (sum_abs / cnt) if cnt > 0 else float('nan')}


def main():
    p = argparse.ArgumentParser(description='Trace reward net checkpoint into TorchScript and compare')
    p.add_argument('--checkpoint', required=True, help='Path to reward checkpoint file or run dir containing reward_net_state.pth or reward_net_full.pth')
    p.add_argument('--expert-npz', default='trajectories/merged_real_dataset_1.1to1.6_for_training.npz')
    p.add_argument('--n-samples', type=int, default=200, help='Number of samples to use for the numeric comparison')
    p.add_argument('--allow-unsafe', action='store_true', help='Allow unsafe unpickling for full-model checkpoints')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    ckpt = args.checkpoint

    # If user passed a directory, try to detect checkpoint file inside
    if os.path.isdir(ckpt):
        # look for reward_net_state.pth or reward_net_full.pth
        cand1 = os.path.join(ckpt, 'reward_net_state.pth')
        cand2 = os.path.join(ckpt, 'reward_net_full.pth')
        if os.path.exists(cand1):
            ckpt = cand1
        elif os.path.exists(cand2):
            ckpt = cand2
        else:
            print('No reward checkpoint found in directory', ckpt)
            sys.exit(2)

    if not os.path.exists(ckpt):
        print('Checkpoint not found:', ckpt)
        sys.exit(2)

    run_dir = os.path.dirname(os.path.abspath(ckpt))
    timestamp = find_timestamp_from_path(ckpt)
    out_name = f'reward_net_ts_{timestamp}.pt'
    out_path = os.path.join(run_dir, out_name)

    print('Checkpoint:', ckpt)
    print('Run dir:', run_dir)
    print('Output TorchScript path:', out_path)

    # load expert npz
    if not os.path.exists(args.expert_npz):
        print('Expert npz not found:', args.expert_npz)
        sys.exit(2)
    data = np.load(args.expert_npz, allow_pickle=True)
    obs = data['observations']
    acts = data['actions']

    # create env and reconstruct python net
    print('Reconstructing Python reward net from checkpoint...')
    try:
        env = make_franka_env()
    except Exception as e:
        print('Failed to create robosuite env (needed for spaces):', e)
        env = None

    if env is None:
        # try to create dummy spaces by inferring shapes from npz
        class DummySpace:
            def __init__(self, low, high, shape):
                self.low = low
                self.high = high
                self.shape = shape

        obs_dim = obs.shape[1] if obs.ndim == 2 else obs.shape[-1]
        act_dim = acts.shape[1]
        from gym import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32)
    else:
        obs_space = env.observation_space
        act_space = env.action_space

    py_net = load_reward_model(ckpt, obs_space, act_space, device=args.device, allow_unsafe=args.allow_unsafe)
    py_net.eval()

    # prepare inputs for trace
    print('Preparing representative input for tracing...')
    example_inputs = choose_trace_inputs(py_net, obs, acts)

    # do the trace
    print('Tracing the model (torch.jit.trace) ...')
    try:
        traced = torch.jit.trace(py_net, example_inputs)
        torch.jit.save(traced, out_path)
        print('Saved traced TorchScript to', out_path)
    except Exception as e:
        print('Tracing failed:', e)
        sys.exit(3)

    # load saved TS and compare
    print('Loading traced TorchScript and comparing outputs...')
    ts_net = torch.jit.load(out_path, map_location=args.device).eval()

    cmp_res = compare_models(py_net, ts_net, obs, acts, nsamples=args.n_samples, device=args.device)
    print('Comparison results:', cmp_res)

    if cmp_res['max_abs'] > 1e-3:
        print('WARNING: Max abs diff > 1e-3 — reexport may not be numerically identical. Check RunningNorm or use a larger test set.')
    else:
        print('OK: Traced TorchScript matches Python net within tolerance on tested samples.')


if __name__ == '__main__':
    main()
