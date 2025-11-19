#!/usr/bin/env python3
"""
Script to process Isaac Lab cube stack trajectories for Franka lift cube task.
- Identifies the first grasped cube
- Renames it to "object"
- Removes the other two cubes
- Truncates trajectory at the maximum height of the first cube
"""

import h5py
import numpy as np
import argparse
import subprocess
import sys
from pathlib import Path


def find_first_grasped_cube(cube_positions):
    """
    Identifies which cube is grasped first based on when each cube reaches max height.
    
    Args:
        cube_positions: Array of shape (T, 9) with [x,y,z] for 3 cubes
        
    Returns:
        tuple: (cube_index, max_height_timestep)
    """
    z_changes = []
    for i in range(3):
        z_pos = cube_positions[:, i*3 + 2]
        max_z = np.max(z_pos)
        max_timestep = np.argmax(z_pos)
        z_changes.append((i, max_z, max_timestep))
    
    # Sort by timestep when maximum is reached
    z_changes.sort(key=lambda x: x[2])
    
    first_cube_idx = z_changes[0][0]
    max_timestep = z_changes[0][2]
    
    return first_cube_idx, max_timestep


def extract_single_cube_data(cube_idx, cube_positions, cube_orientations):
    """
    Extract position and orientation data for a single cube.
    
    Args:
        cube_idx: Index of the cube (0, 1, or 2)
        cube_positions: Array of shape (T, 9)
        cube_orientations: Array of shape (T, 12) - quaternions as 4D vectors
        
    Returns:
        tuple: (positions, orientations) for single cube
    """
    # Extract position (3D)
    pos = cube_positions[:, cube_idx*3:(cube_idx+1)*3]
    
    # Extract orientation (4D quaternion)
    ori = cube_orientations[:, cube_idx*4:(cube_idx+1)*4]
    
    return pos, ori


def process_demo(demo_group, output_group):
    """
    Process a single demonstration.
    
    Args:
        demo_group: HDF5 group containing the input demo
        output_group: HDF5 group where processed demo will be written
    """
    # Read cube positions
    cube_positions = demo_group['obs']['cube_positions'][:]
    cube_orientations = demo_group['obs']['cube_orientations'][:]
    
    # Find first grasped cube and truncation point
    first_cube_idx, max_timestep = find_first_grasped_cube(cube_positions)
    
    print(f"  First grasped cube: {first_cube_idx}")
    print(f"  Maximum height at timestep: {max_timestep}")
    print(f"  Truncating from {len(cube_positions)} to {max_timestep + 1} timesteps")
    
    # Truncate all data at max height timestep
    truncate_idx = max_timestep + 1
    
    # Extract single cube data
    single_cube_pos, single_cube_ori = extract_single_cube_data(
        first_cube_idx, cube_positions[:truncate_idx], cube_orientations[:truncate_idx]
    )
    
    # Create output structure
    # Copy actions
    output_group.create_dataset('actions', data=demo_group['actions'][:truncate_idx])
    
    # Process initial_state if it exists - only keep first grasped cube
    if 'initial_state' in demo_group:
        init_state_group = output_group.create_group('initial_state')
        
        # Copy articulation (robot) initial state
        if 'articulation' in demo_group['initial_state']:
            art_group = init_state_group.create_group('articulation')
            if 'robot' in demo_group['initial_state']['articulation']:
                robot_group = art_group.create_group('robot')
                for key in demo_group['initial_state']['articulation']['robot'].keys():
                    robot_group.create_dataset(
                        key, 
                        data=demo_group['initial_state']['articulation']['robot'][key][:]
                    )
        
        # Process rigid_object initial state - only keep the first grasped cube
        if 'rigid_object' in demo_group['initial_state']:
            rigid_group = init_state_group.create_group('rigid_object')
            
            # Map cube index to cube name (cube_1, cube_2, cube_3)
            cube_name = f'cube_{first_cube_idx + 1}'
            
            if cube_name in demo_group['initial_state']['rigid_object']:
                # Rename to 'object' for lift task
                object_group = rigid_group.create_group('object')
                for key in demo_group['initial_state']['rigid_object'][cube_name].keys():
                    object_group.create_dataset(
                        key,
                        data=demo_group['initial_state']['rigid_object'][cube_name][key][:]
                    )
    
    # Create obs group
    obs_group = output_group.create_group('obs')
    
    # Copy robot state observations
    obs_group.create_dataset('actions', data=demo_group['obs']['actions'][:truncate_idx])
    obs_group.create_dataset('eef_pos', data=demo_group['obs']['eef_pos'][:truncate_idx])
    obs_group.create_dataset('eef_quat', data=demo_group['obs']['eef_quat'][:truncate_idx])
    obs_group.create_dataset('gripper_pos', data=demo_group['obs']['gripper_pos'][:truncate_idx])
    obs_group.create_dataset('joint_pos', data=demo_group['obs']['joint_pos'][:truncate_idx])
    obs_group.create_dataset('joint_vel', data=demo_group['obs']['joint_vel'][:truncate_idx])
    
    # Create single object data - rename to "object"
    # Combine position and orientation into single "object" observation
    # Format: [x, y, z, qx, qy, qz, qw]
    object_data = np.concatenate([single_cube_pos, single_cube_ori], axis=1)
    obs_group.create_dataset('object', data=object_data)
    
    # Process states if they exist
    if 'states' in demo_group:
        states_group = output_group.create_group('states')
        
        # Copy articulation (robot) states - truncated
        if 'articulation' in demo_group['states']:
            art_group = states_group.create_group('articulation')
            if 'robot' in demo_group['states']['articulation']:
                robot_group = art_group.create_group('robot')
                for key in demo_group['states']['articulation']['robot'].keys():
                    robot_group.create_dataset(
                        key, 
                        data=demo_group['states']['articulation']['robot'][key][:truncate_idx]
                    )
        
        # Process rigid_object states - only keep the first grasped cube and truncate
        if 'rigid_object' in demo_group['states']:
            rigid_group = states_group.create_group('rigid_object')
            
            # Map cube index to cube name (cube_1, cube_2, cube_3)
            cube_name = f'cube_{first_cube_idx + 1}'
            
            if cube_name in demo_group['states']['rigid_object']:
                # Rename to 'object' for lift task
                object_group = rigid_group.create_group('object')
                for key in demo_group['states']['rigid_object'][cube_name].keys():
                    object_group.create_dataset(
                        key,
                        data=demo_group['states']['rigid_object'][cube_name][key][:truncate_idx]
                    )


def process_hdf5_file(input_path, output_path):
    """
    Process the entire HDF5 file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
    """
    print(f"Processing: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            # Create data group
            data_group = f_out.create_group('data')
            
            # Process each demo
            demo_keys = sorted([k for k in f_in['data'].keys() if k.startswith('demo_')],
                             key=lambda x: int(x.split('_')[1]))
            
            total_demos = len(demo_keys)
            print(f"\nProcessing {total_demos} demonstrations...")
            
            for i, demo_key in enumerate(demo_keys):
                if i % 100 == 0:
                    print(f"Processing demo {i}/{total_demos}...")
                
                demo_in = f_in['data'][demo_key]
                demo_out = data_group.create_group(demo_key)
                
                process_demo(demo_in, demo_out)
    
    print(f"\nProcessing complete! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process cube stack trajectories for Franka lift cube task'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='generated_dataset_real_dynamics_16_07_final_5000.hdf5',
        help='Input HDF5 file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='franka_lift_cube_trajectories.hdf5',
        help='Output HDF5 file path'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    if output_path.exists():
        response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    process_hdf5_file(input_path, output_path)
    
    # Automatisch fix_hdf5_metadata.py ausführen
    print("\n" + "="*60)
    print("Running metadata fixer...")
    print("="*60)
    
    fix_script = Path(__file__).parent / "fix_hdf5_metadata.py"
    
    if not fix_script.exists():
        print(f"Warning: fix_hdf5_metadata.py not found at {fix_script}")
        print("Skipping metadata fix. Run it manually if needed.")
        return 0
    
    try:
        # Führe fix_hdf5_metadata.py aus
        result = subprocess.run(
            [sys.executable, str(fix_script), 
             "--input", str(output_path),
             "--env_name", "Isaac-Lift-Cube-Franka-IK-Abs-v0",
             "--no-backup"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        print("\n✅ Dataset processing and metadata fix complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running fix_hdf5_metadata.py: {e}")
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
