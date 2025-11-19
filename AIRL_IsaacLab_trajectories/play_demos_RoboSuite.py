import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from imitation.data.types import Trajectory
import time  # For controlling playback speed

# === Parameter ===
NPZ_PATH = "/home/chris/Imitation/trajectories/test2.1_delta_NO_CLIP_robosuite.npz"   # DELTA ACTIONS (NO CLIPPING!)
TASK = "Lift"
ROBOT = "Panda"
VERIFY_OBS = True  # Verify that observations match between NPZ and environment

# === Load data like main.py does ===
def load_npz_trajectories(path):
    """Load trajectories stored in NPZ format (same as main.py)."""
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

trajectories = load_npz_trajectories(NPZ_PATH)
print(f"Loaded {len(trajectories)} trajectories from {NPZ_PATH}")
print(f"First trajectory: {trajectories[0].obs.shape[0]} obs, {trajectories[0].acts.shape[0]} acts")
print(f"Last trajectory: {trajectories[-1].obs.shape[0]} obs, {trajectories[-1].acts.shape[0]} acts")

# === Initialisiere Robosuite Umgebung ===
env = suite.make(
    TASK,
    robots=ROBOT,
    has_renderer=True,           # GUI aktivieren
    has_offscreen_renderer=False,
    use_camera_obs=False,        # nur state obs
    use_object_obs=True,
    control_freq=20,             # muss mit Daten übereinstimmen
)

# Wrap with GymWrapper to get consistent action/obs space
wrapped_env = GymWrapper(env)

# === Playback trajectories (same structure as main.py uses) ===
for traj_idx, traj in enumerate(trajectories[:10]):  # Play first 10 trajectories
    print(f"\n=== Playing trajectory {traj_idx + 1}/{len(trajectories)} ===")
    print(f"  Observations: {traj.obs.shape}")
    print(f"  Actions: {traj.acts.shape}")
    
    reset_result = wrapped_env.reset()
    # Handle both old gym (returns obs) and new gym (returns obs, info)
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    
    # Verify first observation matches
    if VERIFY_OBS:
        obs_diff = np.abs(obs - traj.obs[0]).max()
        if obs_diff > 0.01:
            print(f"  ⚠️  WARNING: Initial obs differs by {obs_diff:.6f}")
            print(f"    NPZ obs[0][:5]: {traj.obs[0][:5]}")
            print(f"    Env obs[:5]:    {obs[:5]}")
        else:
            print(f"  ✓ Initial observation matches (diff: {obs_diff:.6f})")
    
    # Play actions from this trajectory
    # CRITICAL INSIGHT: Frequency doesn't matter for training!
    # What matters is that state→action→next_state relationships are preserved.
    # 
    # OSC controller expects normalized [-1, 1] inputs:
    # Controller maps [-1,1] → [-0.05m, 0.05m] for position, [-0.5rad, 0.5rad] for orientation
    # Our delta actions are in meters/radians, so we normalize by dividing by the max output.
    
    USE_REALTIME_PLAYBACK = True  # Set False for faster playback during debugging
    CONTROL_FREQ = 20  # Hz - for visualization timing only
    
    for step_idx, act in enumerate(traj.acts):
        if USE_REALTIME_PLAYBACK:
            step_start_time = time.time()
        
        # Normalize actions to [-1, 1] range (NO frequency scaling needed!)
        act_normalized = act.copy()
        act_normalized[:3] /= 0.05   # Position: meters → normalized to [-1, 1]
        act_normalized[3:6] /= 0.5   # Orientation: radians → normalized to [-1, 1]
        # Gripper: INVERT because IsaacLab and Robosuite have opposite conventions
        act_normalized[6] *= -1.0
        
        result = wrapped_env.step(act_normalized)
        if len(result) == 5:
            obs, rew, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, rew, done, info = result
        
        env.render()
        
        # Control the playback speed to match real-time (optional, for visualization only)
        if USE_REALTIME_PLAYBACK:
            step_duration = time.time() - step_start_time
            target_step_duration = 1.0 / CONTROL_FREQ
            if step_duration < target_step_duration:
                time.sleep(target_step_duration - step_duration)
        
        # Verify observation matches stored data
        if VERIFY_OBS and step_idx < len(traj.obs) - 1:
            expected_obs = traj.obs[step_idx + 1]
            obs_diff = np.abs(obs - expected_obs).max()
            if obs_diff > 0.01 and step_idx % 20 == 0:
                print(f"    Step {step_idx}: obs diff = {obs_diff:.6f}")
    
    print(f"  Trajectory {traj_idx + 1} completed ({len(traj.acts)} steps)")

wrapped_env.close()
print("\n✓ Playback complete")
