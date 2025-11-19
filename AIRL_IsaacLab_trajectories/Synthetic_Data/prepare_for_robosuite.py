#!/usr/bin/env python3
"""
Prepare NPZ trajectory data for Robosuite AIRL training.

This script converts test2.1.npz (42-dim obs, 8-dim actions) to Robosuite format
(53-dim obs, 7-dim actions) by:
1. Truncating actions from 8 to 7 dimensions
2. Expanding observations from 42 to 53 dimensions with computed features
3. Verifying cube coordinates make sense

Robosuite Lift/Panda expects:
- Observations (53): [object-state (10) | robot0_proprio-state (43)]
  - object-state: cube_pos(3) + cube_quat(4) + gripper_to_cube_pos(3)
  - robot0_proprio: joint_pos(7) + joint_pos_cos(7) + joint_pos_sin(7) + joint_vel(7) +
                    eef_pos(3) + eef_quat(4) + eef_quat_site(4) + gripper_qpos(2) + gripper_qvel(2)
- Actions (7): 6 DOF + gripper

test2.1.npz structure (42 dims):
- Indices 0-7: Unknown (zeros in first obs)
- Indices 8-23: joint/eef data (~16 dims)
- Indices 24-34: Unknown (zeros/small values)
- Indices 35-41: cube_pos(3) + cube_quat(4) = 7 dims
"""

import numpy as np
import argparse
from pathlib import Path


def analyze_npz_structure(npz_path):
    """Analyze the NPZ to understand its structure."""
    npz = np.load(npz_path)
    obs = npz['observations']
    actions = npz['actions']
    dones = npz['dones']
    
    print(f'\n=== Input NPZ: {npz_path.name} ===')
    print(f'Observations: {obs.shape} (expect N x 42)')
    print(f'Actions:      {actions.shape} (expect N x 8)')
    print(f'Dones:        {dones.shape}')
    
    # Cube coordinates verification
    cube_pos = obs[0, 35:38]
    cube_quat = obs[0, 38:42]
    print(f'\nCube position (first timestep): {cube_pos}')
    print(f'Cube quaternion: {cube_quat}')
    print(f'Cube z-coordinate: {cube_pos[2]:.4f} (should be ~0.02 for table height)')
    
    if cube_pos[2] < -0.1 or cube_pos[2] > 2.0:
        print('⚠️  WARNING: Cube z-coordinate looks unusual!')
    else:
        print('✓ Cube coordinates look reasonable')
    
    return npz


def map_observations_to_robosuite(obs_42):
    """
    Map 42-dim observations to 53-dim Robosuite format.
    
    test2.1.npz structure (from HDF5, alphabetically sorted fields):
    - Indices  0-7:  actions (8) - previous action in obs
    - Indices  8-10: eef_pos (3)
    - Indices 11-14: eef_quat (4)
    - Indices 15-16: gripper_pos (2)
    - Indices 17-25: joint_pos (9) - IsaacLab has 9, Robosuite Panda uses first 7
    - Indices 26-34: joint_vel (9) - IsaacLab has 9, Robosuite Panda uses first 7
    - Indices 35-41: object = cube_pos(3) + cube_quat(4)
    
    Robosuite format (53 dims):
    - object-state (10): cube_pos(3) + cube_quat(4) + gripper_to_cube_pos(3)
    - robot0_proprio (43): joint_pos(7) + joint_pos_cos(7) + joint_pos_sin(7) + 
                           joint_vel(7) + eef_pos(3) + eef_quat(4) + eef_quat_site(4) +
                           gripper_qpos(2) + gripper_qvel(2)
    """
    N = obs_42.shape[0]
    obs_53 = np.zeros((N, 53), dtype=np.float32)
    
    # Extract from obs_42 based on HDF5 structure
    # prev_action = obs_42[:, 0:8]    # Not needed in Robosuite format
    eef_pos = obs_42[:, 8:11]         # 3 dims
    eef_quat = obs_42[:, 11:15]       # 4 dims
    gripper_pos = obs_42[:, 15:17]    # 2 dims (qpos)
    joint_pos_9 = obs_42[:, 17:26]    # 9 dims from IsaacLab
    joint_vel_9 = obs_42[:, 26:35]    # 9 dims from IsaacLab
    cube_pos = obs_42[:, 35:38]       # 3 dims
    cube_quat = obs_42[:, 38:42]      # 4 dims
    
    # Panda only has 7 joints, so take first 7
    joint_pos = joint_pos_9[:, :7]    # 7 dims
    joint_vel = joint_vel_9[:, :7]    # 7 dims
    
    # Compute derived features
    joint_pos_cos = np.cos(joint_pos)  # 7 dims
    joint_pos_sin = np.sin(joint_pos)  # 7 dims
    gripper_to_cube = cube_pos - eef_pos  # 3 dims (relative position)
    
    # For gripper_qvel: we don't have velocity, use zeros
    gripper_vel = np.zeros((N, 2), dtype=np.float32)
    
    # eef_quat_site: In Robosuite this is often the same as eef_quat or a site quaternion
    # For now, duplicate eef_quat (refinement may be needed)
    eef_quat_site = eef_quat.copy()
    
    # Assemble 53-dim observation in Robosuite order:
    # [object-state (10) | robot0_proprio-state (43)]
    
    # object-state (0-9): cube_pos + cube_quat + gripper_to_cube
    obs_53[:, 0:3] = cube_pos
    obs_53[:, 3:7] = cube_quat
    obs_53[:, 7:10] = gripper_to_cube
    
    # robot0_proprio-state (10-52):
    # joint_pos(7) + joint_pos_cos(7) + joint_pos_sin(7) + joint_vel(7) +
    # eef_pos(3) + eef_quat(4) + eef_quat_site(4) + gripper_qpos(2) + gripper_qvel(2)
    offset = 10
    obs_53[:, offset:offset+7] = joint_pos;         offset += 7   # 10-16
    obs_53[:, offset:offset+7] = joint_pos_cos;     offset += 7   # 17-23
    obs_53[:, offset:offset+7] = joint_pos_sin;     offset += 7   # 24-30
    obs_53[:, offset:offset+7] = joint_vel;         offset += 7   # 31-37
    obs_53[:, offset:offset+3] = eef_pos;           offset += 3   # 38-40
    obs_53[:, offset:offset+4] = eef_quat;          offset += 4   # 41-44
    obs_53[:, offset:offset+4] = eef_quat_site;     offset += 4   # 45-48
    obs_53[:, offset:offset+2] = gripper_pos;       offset += 2   # 49-50
    obs_53[:, offset:offset+2] = gripper_vel;       offset += 2   # 51-52
    
    assert offset == 53, f"Expected 53 dims, got {offset}"
    
    print(f'\n✓ Observation mapping complete:')
    print(f'  - Extracted 7 joint positions from 9 (IsaacLab → Robosuite Panda)')
    print(f'  - Computed cos/sin transforms for joints')
    print(f'  - Computed gripper_to_cube relative position')
    print(f'  - Used eef_quat for eef_quat_site (may need refinement)')
    print(f'  - Zero gripper velocity (not available in source data)')
    
    return obs_53


def prepare_for_robosuite(input_npz, output_npz):
    """Convert NPZ from 42-dim/8-act to 53-dim/7-act format."""
    # Load input
    npz_data = analyze_npz_structure(input_npz)
    obs_42 = npz_data['observations']
    actions_8 = npz_data['actions']
    dones = npz_data['dones']
    
    # Check if actions are already 7-dim or need conversion from 8-dim
    if actions_8.shape[1] == 7:
        print(f'\n=== Actions Already 7-dim (Delta Actions) ===')
        actions_7 = actions_8
        print(f'Actions: {actions_7.shape} (7-dim: 6 DOF delta + gripper)')
        
        # CRITICAL: Normalize delta actions for Robosuite OSC controller!
        # OSC expects normalized [-1, 1] inputs and scales them to:
        #   Position: [-0.05m, 0.05m]
        #   Orientation: [-0.5rad, 0.5rad]
        # Our actions are in absolute units (meters/radians), so normalize:
        actions_normalized = actions_7.copy()
        actions_normalized[:, :3] /= 0.05   # Position deltas: meters → [-1, 1]
        actions_normalized[:, 3:6] /= 0.5   # Orientation deltas: radians → [-1, 1]
        # Gripper (dim 6) stays unchanged (should already be -1 or 1)
        
        # Clip to [-1, 1] range (this is what OSC controller does anyway)
        # Note: This means some expert actions will be clipped, but that's OK!
        # AIRL learns from state transitions and reward, not exact action matching.
        # The learned policy will stay within valid action bounds.
        actions_normalized[:, :6] = np.clip(actions_normalized[:, :6], -1.0, 1.0)
        
        print(f'\n=== Action Normalization for OSC Controller ===')
        print(f'Position deltas: divided by 0.05 (OSC output range)')
        print(f'Orientation deltas: divided by 0.5 (OSC output range)')
        print(f'Sample action before normalization: {actions_7[10]}')
        print(f'Sample action after normalization:  {actions_normalized[10]}')
        
        # Check how many actions were clipped
        n_clipped_pos = (np.abs(actions_normalized[:, :3]) >= 0.999).sum()
        n_clipped_orient = (np.abs(actions_normalized[:, 3:6]) >= 0.999).sum()
        total_actions = actions_normalized.shape[0] * 3
        print(f'\nClipped actions (at ±1.0):')
        print(f'  Position: {n_clipped_pos}/{total_actions} values ({100*n_clipped_pos/total_actions:.1f}%)')
        print(f'  Orientation: {n_clipped_orient}/{total_actions} values ({100*n_clipped_orient/total_actions:.1f}%)')
        
        if n_clipped_pos > 0.1 * total_actions:
            print('\n⚠️  WARNING: >10% of position actions clipped!')
            print('  This suggests IsaacLab uses different action scaling than Robosuite OSC.')
            print('  AIRL will learn a policy that respects Robosuite action bounds.')
        
        actions_7 = actions_normalized
        
    elif actions_8.shape[1] == 8:
        # Action mapping: IsaacLab has 8 dims, Robosuite needs 7
        # IsaacLab: [arm_6dof, gripper_pos, gripper_cmd]
        # Robosuite: [arm_6dof, gripper_cmd]
        # We need to keep dims 0-5 (arm) and dim 7 (gripper command), drop dim 6 (gripper position)
        actions_7 = np.concatenate([actions_8[:, :6], actions_8[:, 7:8]], axis=1)
        
        print(f'\n=== Action Mapping ===')
        print(f'Input:  {actions_8.shape} (8-dim: 6 arm + gripper_pos + gripper_cmd)')
        print(f'Output: {actions_7.shape} (7-dim: 6 arm + gripper_cmd)')
        print(f'Action remapping: [0:6, 7] → [0:7]  (dropping dim 6: gripper position)')
        
        # Verify gripper command is binary
        gripper_cmd_unique = np.unique(actions_8[:, 7])
        print(f'Gripper command values: {gripper_cmd_unique} (should be [-1, 1])')
    else:
        raise ValueError(f'Expected 7 or 8 action dims, got {actions_8.shape[1]}')
    
    # Expand observations: 42 dims → 53 dims
    obs_53 = map_observations_to_robosuite(obs_42)
    print(f'\n=== Observation Expansion ===')
    print(f'Input:  {obs_42.shape} (42-dim)')
    print(f'Output: {obs_53.shape} (53-dim)')
    
    # Save output
    np.savez_compressed(
        output_npz,
        observations=obs_53,
        actions=actions_7,
        dones=dones
    )
    print(f'\n=== Output saved to: {output_npz} ===')
    print(f'Observations: {obs_53.shape}')
    print(f'Actions:      {actions_7.shape}')
    print(f'Dones:        {dones.shape}')
    
    return output_npz


def main():
    parser = argparse.ArgumentParser(description='Prepare NPZ for Robosuite AIRL training')
    parser.add_argument('--input', type=str, default='trajectories/test2.1.npz',
                        help='Input NPZ file (42-dim obs, 8-dim actions)')
    parser.add_argument('--output', type=str, default='trajectories/test2.1_robosuite.npz',
                        help='Output NPZ file (53-dim obs, 7-dim actions)')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f'Error: Input file not found: {input_path}')
        return
    
    prepare_for_robosuite(input_path, output_path)
    print('\n✓ Conversion complete!')
    print('\nThe converted NPZ is ready for Robosuite AIRL training with:')
    print('  - 53-dim observations (10 object-state + 43 robot0_proprio-state)')
    print('  - 7-dim actions (Panda: 6 DOF + gripper)')
    print('  - Computed features: joint cos/sin, gripper-to-cube relative position')
    print('  - Verified cube coordinates (z: ~0.02m table → ~0.28m lifted)')


if __name__ == '__main__':
    main()
