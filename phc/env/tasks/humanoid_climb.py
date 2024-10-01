import torch
import time
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags
from enum import Enum
from smpl_sim.poselib.core.rotation3d import quat_inverse, quat_mul

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
        
        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))
        
        # self.set_initial_root_state()
        self.goal = torch.tensor([0, -0.5, 4]).to(self.device)  # Goal is 20 meters high
        
        self.statistics = False
        # if flags.test:
        #     self._enable_early_termination = False
        #     self.statistics = True
            
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

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        super()._build_env(env_id, env_ptr, humanoid_asset_list)
        self._build_target(env_id, env_ptr)
        return

    
    def _load_target_asset(self): 
        

        asset_root = "phc/data/assets/urdf/"
        asset_file = "climbing_wall.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        self._wall_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return

    def _build_target(self, env_id, env_ptr):
        
        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(0, -0.5, 0.0)  
        default_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0)  # Rotate if needed
        target_handle = self.gym.create_actor(env_ptr, self._wall_asset, default_pose, "target", env_id, 0)
        
        return 
        
    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self.num_agents, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents
        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 6  # distance to goal (3) + current height (1) + wall normal (2)
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        obs_list = []

        for i in range(self.num_agents):
            if env_ids is None:
                root_states = self._humanoid_root_states_list[i]
            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
            
            obs = compute_climb_observations(root_states, self.goal)
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            root_states = self._humanoid_root_states_list[i]
            key_bodies_pos = self._rigid_body_pos_list[i][:, self._key_body_ids[:2], :]
            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_climb_reward(
                root_states, 
                self._prev_root_pos_list[i], 
                self.goal, 
                self.dt
            )

        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        
        for i in range(self.num_agents):
            self._prev_root_pos_list[i] = self._humanoid_root_states_list[i][..., 0:3].clone()

        return

    def _compute_reset(self):
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        
        return

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
    dist_to_goal_local = torch_utils.my_quat_rotate(heading_rot, dist_to_goal)
    
    obs = torch.cat([
        dist_to_goal_local,
        root_pos[:, 2:3],  # current height
        torch.tensor([1.0, 0.0]).repeat(root_states.shape[0], 1).to(root_states.device)  # wall normal (assuming it's always [1, 0])
    ], dim=-1)
    
    return obs

def compute_climb_reward(root_states, prev_root_pos, goal, dt):
    root_pos = root_states[:, 0:3]
    root_vel = root_states[:, 7:10]
    
    
    # Goal distance reward (30% of total reward)
    dist_to_goal = torch.norm(goal - root_pos, dim=-1)
    goal_reward =  torch.exp(-10 * dist_to_goal)
    
    # Climbing direction reward (10% of total reward)
    direction_reward = torch.clamp(root_pos[:, 0], max=0) 
    
    
    # Weighting
    w_goal = 0.8
    w_direction = 0.2
    
    reward = goal_reward * w_goal + direction_reward * w_direction
    
    return reward

def is_touching_wall(contact_forces, contact_body_ids):
    wall_contact_threshold = 1.0  # Adjust this value as needed
    wall_contact_forces = contact_forces[:, contact_body_ids, 0]  # Assuming the wall is in the x-direction
    return torch.any(torch.abs(wall_contact_forces) > wall_contact_threshold, dim=-1)


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           max_episode_length,
                           enable_early_termination, termination_heights, num_agents):
    # type: (Tensor, Tensor, list, Tensor, list, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf_list[0].clone()
        
        masked_contact_buf[:, contact_body_ids, :] = 0
        force_threshold = 50
        fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        body_height = rigid_body_pos_list[0][..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        body_y = rigid_body_pos_list[0][..., 0, 1]
        body_out = torch.abs(body_y)>0.8
        
        has_fallen = torch.logical_or(fall_contact, fall_height) # don't touch the hurdle. 
        has_fallen = torch.logical_or(has_fallen, body_out)
        

        if num_agents>1:
            for i in range(1, num_agents):
                masked_contact_buf = contact_buf_list[i].clone()
                masked_contact_buf[:, contact_body_ids, :] = 0
                force_threshold = 50
                fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

                body_height = rigid_body_pos_list[i][..., 2]
                fall_height = body_height < termination_heights
                fall_height[:, contact_body_ids] = False
                fall_height = torch.any(fall_height, dim=-1)

                has_fallen_temp = torch.logical_or(fall_contact, fall_height)

                has_fallen = torch.logical_or(has_fallen, has_fallen_temp)

                body_y = rigid_body_pos_list[i][..., 0, 1]
                body_out = torch.abs(body_y)>0.8
            
                has_fallen = torch.logical_or(has_fallen, body_out)
        
        has_failed = has_fallen
        
        touching_wall = is_touching_wall(contact_buf_list[0].clone(), contact_body_ids)
        # Terminate if not touching the wall after a certain height
        min_wall_contact_height = 1.2  # Adjust this value as needed
        should_touch_wall = rigid_body_pos_list[0][..., 0, 2] > min_wall_contact_height
        not_touching_wall_when_should = should_touch_wall & ~touching_wall
        
        has_failed = torch.logical_or(has_failed, not_touching_wall_when_should)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated