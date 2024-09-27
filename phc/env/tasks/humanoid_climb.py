import torch
import time
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags
from enum import Enum
from poselib.poselib.core.rotation3d import quat_inverse, quat_mul

TAR_ACTOR_ID = 1

class HumanoidClimb(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        
        self._build_climbing_wall()
        # self.set_initial_root_state()
        self.goal = torch.tensor([0, 0, 20]).to(self.device)  # Goal is 20 meters high
        
        self.statistics = False
        if flags.test:
            self._enable_early_termination = False
            self.statistics = True
            
        if self.statistics:
            self.build_statistics_tensor()
        
        return

    def build_statistics_tensor(self):
        self.height_buffer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.time_buffer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reach_goal_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.height_result = []
        self.time_result = []

    def set_initial_root_state(self):
        initial_root_states = self._initial_humanoid_root_states.clone()
        initial_root_states[:, 7:13] = 0
        initial_root_states[..., 0] = 0  # x position
        initial_root_states[..., 1] = 0  # y position
        initial_root_states[..., 2] = 1  # z position (slightly above ground)
        initial_root_states[..., 3:7] = torch.tensor([0, 0, 0, 1])  # rotation to face the wall
        self._initial_humanoid_root_states = initial_root_states

    def _build_climbing_wall(self):
        # Load the climbing wall asset
        asset_root = "phc/data/assets/urdf/"
        wall_asset_file = "climbing_wall.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True

        wall_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_file, asset_options)

        # Set the wall pose
        wall_pose = gymapi.Transform()
        wall_pose.p = gymapi.Vec3(0.5, 0, 10)  # position the wall
        wall_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi)  # rotate to face the humanoid

        # Create the wall actor in each environment
        for i in range(self.num_envs):
            wall_handle = self.gym.create_actor(self.envs[i], wall_asset, wall_pose, "wall", i, 2)
            self.gym.set_rigid_body_color(self.envs[i], wall_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.8))

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 6  # distance to goal (3) + current height (1) + wall normal (2)
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
        else:
            root_states = self._humanoid_root_states[env_ids]
        
        obs = compute_climb_observations(root_states, self.goal)
        return obs

    def _compute_reward(self, actions):
        root_states = self._humanoid_root_states
        self.rew_buf[:] = compute_climb_reward(root_states, self._prev_root_pos, self.goal, self.dt)
        self._prev_root_pos[:] = root_states[:, 0:3]
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights)

    def post_physics_step(self):
        super().post_physics_step()
        
        if self.statistics:
            self.get_statistics()
        
        return

    def get_statistics(self):
        root_pos = self._humanoid_root_states[:, 0:3]
        
        self.height_buffer = torch.max(self.height_buffer, root_pos[:, 2])
        self.reach_goal_flag += (root_pos[:, 2] > self.goal[2]).float()
        
        self.time_result.append(self.progress_buf[self.reach_goal_flag == 1].clone())
        
class HumanoidClimbZ(HumanoidClimb):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        super().step_z(actions)
        return
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        return


def compute_climb_observations(root_states, goal):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    dist_to_goal = goal - root_pos
    dist_to_goal_local = quat_rotate(heading_rot, dist_to_goal)
    
    obs = torch.cat([
        dist_to_goal_local,
        root_pos[:, 2:3],  # current height
        torch.tensor([1.0, 0.0]).repeat(root_states.shape[0], 1)  # wall normal (assuming it's always [1, 0])
    ], dim=-1)
    
    return obs

def compute_climb_reward(root_states, prev_root_pos, goal, dt):
    root_pos = root_states[:, 0:3]
    
    # Height reward
    height_diff = root_pos[:, 2] - prev_root_pos[:, 2]
    height_reward = torch.clamp(height_diff / dt, min=0) * 2.0
    
    # Goal distance reward
    dist_to_goal = torch.norm(goal - root_pos, dim=-1)
    goal_reward = 1.0 / (1.0 + dist_to_goal)
    
    # Climbing direction reward
    direction_reward = torch.clamp(root_pos[:, 0], max=0) * 0.1  # penalize moving away from the wall
    
    total_reward = height_reward + goal_reward + direction_reward
    
    return total_reward
