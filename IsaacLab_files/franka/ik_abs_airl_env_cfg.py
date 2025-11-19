# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Franka Lift environment with AIRL-compatible observations.

This config adds end-effector pose and gripper position observations that are expected
by reward networks trained on Robosuite's lift task.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import ik_abs_env_cfg


@configclass
class FrankaCubeLiftEnvCfg_AIRL(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
    """Configuration for Franka lift environment with AIRL-compatible observations."""
    
    @configclass
    class ObservationsCfg:
        """Observation specifications for the MDP with AIRL compatibility."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""

            # Core observations (matching IsaacLab's lift task)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
            object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
            target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
            actions = ObsTerm(func=mdp.last_action)
            
            # Additional observations for AIRL reward network (matching Robosuite)
            eef_pos = ObsTerm(func=mdp.ee_frame_pose_in_robot_root_frame, params={"return_key": "pos"})
            eef_quat = ObsTerm(func=mdp.ee_frame_pose_in_robot_root_frame, params={"return_key": "quat"})
            gripper_pos = ObsTerm(func=mdp.gripper_pos)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        # observation groups
        policy: PolicyCfg = PolicyCfg()
    
    observations: ObservationsCfg = ObservationsCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Define gripper joint names for observation function
        self.gripper_joint_names = ["panda_finger_.*"]


@configclass
class FrankaCubeLiftEnvCfg_AIRL_PLAY(FrankaCubeLiftEnvCfg_AIRL):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
