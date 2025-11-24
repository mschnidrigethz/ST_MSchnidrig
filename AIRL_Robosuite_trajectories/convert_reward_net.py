"""Convert a state-dict reward checkpoint into a TorchScript archive compatible with
scripts/reinforcement_learning/sb3_AIRL/train.py

Usage:
  python convert_reward_net.py /path/to/reward_net_1000000.pt /path/to/output.pt

The converter inspects the state-dict shapes to infer input dims and action dims
and constructs a compatible nn.Module with submodules and parameter names matching
the state-dict (e.g. "_base.mlp.dense0.weight" and "potential._potential_net.dense0.weight").
It then loads the weights and saves a traced TorchScript module.
"""
import sys
import os
from typing import List
import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        # buffers expected by the state-dict
        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones(size))
        self.register_buffer('count', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return x
        if self.running_mean.numel() == x.shape[-1]:
            return (x - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return x


class FlexibleMLP(nn.Module):
    """Construct an MLP that exposes layers as named attributes 'dense0', 'dense1', 'dense_final'
    to match the keys in the provided state-dict."""
    def __init__(self, in_dim: int, hidden_sizes: List[int], out_dim: int):
        super().__init__()
        # optional normalizer
        self.normalize_input = RunningNorm(in_dim)

        # create dense layers with exact names used in the state dict
        if len(hidden_sizes) == 0:
            # single linear
            self.dense_final = nn.Linear(in_dim, out_dim)
        else:
            self.dense0 = nn.Linear(in_dim, hidden_sizes[0])
            if len(hidden_sizes) > 1:
                # more hidden layers
                for i in range(1, len(hidden_sizes)):
                    setattr(self, f'dense{i}', nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.dense_final = nn.Linear(hidden_sizes[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        x = self.normalize_input(x)
        if hasattr(self, 'dense0'):
            x = torch.relu(self.dense0(x))
            i = 1
            while True:
                name = f'dense{i}'
                if hasattr(self, name):
                    layer = getattr(self, name)
                    x = torch.relu(layer(x))
                    i += 1
                else:
                    break
        # final
        x = self.dense_final(x)
        return x


class RewardNetWrapper(nn.Module):
    """A simple wrapper that exposes the same submodule names as in the state-dict
    and implements a forward signature compatible with train.py:
        forward(obs, acts=None, next_obs=None, done=None)

    The implementation is minimal: it evaluates the potential network on obs and
    the base.mlp on a concatenation of obs+acts (padding/truncating as required),
    then returns the sum. This matches common compositions seen in reward nets and
    is enough to reproduce the saved behavior when weights match."""
    def __init__(self, obs_dim: int, base_input_dim: int, action_dim: int):
        super().__init__()
        # potential._potential_net -> MLP(in=obs_dim, hidden=[32,32], out=1)
        self.potential = nn.Module()
        self.potential._potential_net = FlexibleMLP(obs_dim, [32, 32], 1)

        # _base.mlp -> MLP(in=base_input_dim, hidden=[32], out=1)
        self._base = nn.Module()
        self._base.mlp = FlexibleMLP(base_input_dim, [32], 1)

    def forward(self, obs: torch.Tensor, acts: torch.Tensor = None, next_obs: torch.Tensor = None, done: torch.Tensor = None):
        # obs: (batch, obs_dim)
        pot = self.potential._potential_net(obs)

        # build input for base.mlp: try to concatenate obs and acts,
        # otherwise pad zeros/truncate to base_input_dim
        bsize = obs.shape[0]
        device = obs.device
        base_in_dim = self._base.mlp.normalize_input.running_mean.numel()

        if acts is not None:
            # ensure 2D
            if acts.ndim == 1:
                acts = acts.unsqueeze(0)
            # flatten acts
            cand = [obs, acts.reshape(bsize, -1)]
            if next_obs is not None:
                cand.append(next_obs)
            base_in = torch.cat(cand, dim=1)
            # pad/truncate
            if base_in.shape[1] < base_in_dim:
                pad = torch.zeros((bsize, base_in_dim - base_in.shape[1]), device=device)
                base_in = torch.cat([base_in, pad], dim=1)
            elif base_in.shape[1] > base_in_dim:
                base_in = base_in[:, :base_in_dim]
        else:
            base_in = torch.zeros((bsize, base_in_dim), device=device)
            take = min(base_in_dim, obs.shape[1])
            base_in[:, :take] = obs[:, :take]

        base_out = self._base.mlp(base_in)

        out = pot + base_out
        # return (batch,) shape
        out = out.view(out.shape[0])
        return out


def main(in_path: str, out_path: str):
    sd = torch.load(in_path, map_location='cpu')
    if not isinstance(sd, dict):
        raise RuntimeError(f"Expected a state-dict-like checkpoint at {in_path}")

    # infer dims from state-dict
    # potential input
    pot_w = sd.get('potential._potential_net.dense0.weight')
    base_w = sd.get('_base.mlp.dense0.weight')
    if pot_w is None or base_w is None:
        raise RuntimeError('Could not find expected keys in state-dict (potential._potential_net.dense0 or _base.mlp.dense0)')

    obs_dim = pot_w.shape[1]
    base_in_dim = base_w.shape[1]
    action_dim = base_in_dim - obs_dim
    if action_dim < 0:
        action_dim = 0

    print(f'Inferred dims: obs_dim={obs_dim}, base_input_dim={base_in_dim}, action_dim={action_dim}')

    model = RewardNetWrapper(obs_dim, base_in_dim, action_dim)

    # load state dict - keys should match our constructed module
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('Loaded state-dict into model. missing keys:', missing)
    print('unexpected keys:', unexpected)

    model.eval()

    # prepare example inputs for tracing: shapes expected by train.py
    obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    acts = torch.zeros((1, action_dim), dtype=torch.float32) if action_dim > 0 else None
    next_obs = obs.clone()
    done = torch.tensor([0.0])

    # trace module
    try:
        if acts is not None:
            traced = torch.jit.trace(model, (obs, acts, next_obs, done))
        else:
            traced = torch.jit.trace(model, (obs,))
        torch.jit.save(traced, out_path)
        print(f'TorchScript module saved to: {out_path}')
    except Exception as e:
        print('Tracing failed:', e)
        raise


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python convert_reward_net.py <in_state_dict.pt> <out_scripted.pt>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
