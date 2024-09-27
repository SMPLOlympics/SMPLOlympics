

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

TAR_ACTOR_ID = 1



class HumanoidHurdle(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._tar_dist_min = 0.5
        self._tar_dist_max = 1.0
        self._near_dist = 1.5
        self._near_prob = 0.5
        
        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))
        
        # strike_body_names = cfg["env"]["strikeBodyNames"]
        
        # self._build_target_tensors()
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        self._build_target_tensors()
        self.set_initial_root_state()
        self.goal = torch.tensor([110,0,1]).to(self.device)
        self.hurdle_location = torch.tensor([[13.72,0,1.067],[22.86,0,1.067],[32.0,0,1.067],[41.14,0,1.067],
                                             [50.28,0,1.067],[59.42,0,1.067],[68.56,0,1.067],[77.7,0,1.067],
                                             [86.84,0,1.067],[95.98,0,1.067]])[:,None,:].repeat(1,self.num_envs,1).to(self.device)
        self.tar_speed = 4
        self.statistics = False
        if flags.test:
            self._enable_early_termination = False
            self.statistics = True
            
        if self.statistics:
            self.build_statistics_tensor()
            
        
        return
    
    def build_statistics_tensor(self):
        self.distance_buffer = torch.zeros(self.num_envs,
                                    device=self.device,
                                    dtype=torch.float)
        # self.time_finish_buffer = torch.zeros(self.num_envs,
        #                             device=self.device,
        #                             dtype=torch.float)
        self.pass_line_flag = torch.zeros(self.num_envs,
                                    device=self.device,
                                    dtype=torch.float)
        
        self.distance_result = []
        self.time_result = []
        
    def set_initial_root_state(self):
        for i in range(self.num_agents):
            initial_humanoid_root_states = self._initial_humanoid_root_states_list[i].clone()
            initial_humanoid_root_states[:, 7:13] = 0

            initial_humanoid_root_states[..., 0] = 0
            initial_humanoid_root_states[..., 1] = i*2
            
            initial_humanoid_root_states[..., 3] = 0
            initial_humanoid_root_states[..., 4] = 0
            initial_humanoid_root_states[..., 5] = 0
            initial_humanoid_root_states[..., 6] = 1

            self._initial_humanoid_root_states_list[i] = initial_humanoid_root_states
    
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3 + 10 * 3
        return obs_size
    
    def post_physics_step(self):
        # self.out_bound, self.red_win, self.green_win = self.check_game_state()
        if self.statistics:
            self.get_statistics()
            
        super().post_physics_step()

        return
    
    def get_statistics(self):
        root_x = self._humanoid_root_states_list[0][:,0].clone()
        
        self.distance_buffer = torch.max(self.distance_buffer, root_x)
        
        self.pass_line_flag += 1 * (root_x>110)
        
        self.time_result.append(self.progress_buf[self.pass_line_flag==1].clone())

        # if root_x.max()>110:
        #     import pdb;pdb.set_trace()
    
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
        asset_file = "hurdle.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        self._hurdle_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return

    def _build_target(self, env_id, env_ptr):
        
        default_pose = gymapi.Transform()
        target_handle = self.gym.create_actor(env_ptr, self._hurdle_asset, default_pose, "target", env_id, 0)
        
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self.num_agents, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents
        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self.reset_target(env_ids)
        return

    def reset_target(self, env_ids):
        n = len(env_ids)
        if n>0:
            if flags.test:
                self._target_states[env_ids,2] = torch.Tensor(n).uniform_(-0., -0.).to(self.device, torch.float) 
            else:
                self._target_states[env_ids,2] = torch.Tensor(n).uniform_(-1.067, 0.1).to(self.device, torch.float) 
            self.hurdle_location[:, env_ids, 2] = 1.067 + self._target_states[env_ids, 2].clone()
    
    
    def _reset_env_tensors(self, env_ids):
        
        if self.statistics:
            self.distance_result.append(self.distance_buffer[env_ids].clone())
        
        self.distance_buffer[env_ids] = 0
        self.pass_line_flag[env_ids] = 0
        
        if self.statistics and len(self.distance_result)>2:
            distance = torch.cat(self.distance_result[2:])
            distance = distance[distance>0]
            print("len", len(distance))
            print("success_rate:",((distance>110)*1.).mean())
            print("average_distance:", distance.mean())
            print("average_time:", (torch.cat(self.time_result)/30).mean())
                
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        
        for i in range(self.num_agents):
            self._prev_root_pos_list[i] = self._humanoid_root_states_list[i][..., 0:3].clone()

        return
    
      
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []

        if (env_ids is None):
            hurdle_location = self.hurdle_location
        else:
            hurdle_location = self.hurdle_location[:,env_ids,:]

        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                
            
            obs = compute_hurdle_observations(root_states, self.goal, hurdle_location)
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            # reward = 1
            root_states = self._humanoid_root_states_list[i]
            key_bodies_pos = self._rigid_body_pos_list[i][:, self._key_body_ids[:2], :]
            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_hurdle_reward(root_states, self._prev_root_pos_list[i], self.goal, key_bodies_pos, self.tar_speed, self.dt)

        return

    def _compute_reset(self):
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        
        return

    def _draw_task(self):

        
        return


class HumanoidHurdleZ(HumanoidHurdle):
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
    



#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_hurdle_observations(root_states, goal, hurdle_location):
    
    
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    # print(hurdle_location.shape)

    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = goal - root_pos
    # local_tar_pos[..., -1] = goal[..., -1]
    local_tar_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_tar_pos)
    obs = local_tar_pos
    # import pdb;pdb.set_trace()

    for i in range(len(hurdle_location)):
        local_hurdle_pos = hurdle_location[i] - root_pos
        # local_hurdle_pos[..., -1] = goal[..., -1]
        local_hurdle_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_hurdle_pos)
        obs = torch.cat([obs, local_hurdle_pos],dim=-1)
    return obs



# @torch.jit.script
# def compute_hurdle_reward(root_states, prev_root_pos, tar_pos,key_body_pos, tar_speed, dt):
#     # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
#     root_pos = root_states[:, 0:3]
#     root_rot = root_states[:, 3:7]
#     feet_z = key_body_pos[...,-1]
#     feet_x = key_body_pos[...,0]

#     dist_threshold = 0.5

#     pos_err_scale = 0.05
#     vel_err_scale = 0.3

#     pos_reward_w = 0.45
#     vel_reward_w = 0.45
#     face_reward_w = 0.1
#     feet_reward_w = 0.
    
#     pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
#     pos_err = torch.norm(pos_diff, dim=-1)
#     pos_reward = torch.exp(-pos_err_scale * pos_err)

#     tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
#     tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
#     delta_root_pos = root_pos - prev_root_pos
#     root_vel = delta_root_pos / dt

    
#     tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
#     tar_vel_err = tar_speed - tar_dir_speed
#     tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
#     vel_reward = torch.exp(-vel_err_scale * tar_vel_err)
#     speed_mask = tar_dir_speed <= 0
#     vel_reward[speed_mask] = 0


#     heading_rot = torch_utils.calc_heading_quat(root_rot)
#     facing_dir = torch.zeros_like(root_pos)
#     facing_dir[..., 0] = 1.0
#     facing_dir = quat_rotate(heading_rot, facing_dir)
#     facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
#     facing_reward = torch.clamp_min(facing_err, 0.0)

#     feet_reward = torch.max(feet_z,dim=-1)[0]
#     feet_x = torch.min(feet_x,dim=-1)[0]

#     dist_mask = pos_err < dist_threshold
#     facing_reward[dist_mask] = 1.0
#     vel_reward[dist_mask] = 1.0
#     # feet_reward[dist_mask] = 0.5
#     # print(pos_reward[0], vel_reward[0],root_vel[0])
#     reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward + feet_reward_w * feet_reward
#     # reward = (1 + ((feet_x[...,0])//5)) * reward
#     reward += torch.clamp_min(feet_x[...,0] * 0.1, 0)
    
#     return reward


# @torch.jit.script
def compute_hurdle_reward(root_states, prev_root_pos, tar_pos, key_body_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    feet_z = key_body_pos[...,-1]
    
    prev_dist = torch.norm(prev_root_pos - tar_pos, dim=-1)
    curr_dist = torch.norm(root_pos - tar_pos, dim=-1)
    closer_target_r = torch.clamp(prev_dist - curr_dist, min=0, max=1)  # player getting closer to the ball
    # print(prev_dist[0], curr_dist[0])
    dist_threshold = 0.5


     
    reward =  closer_target_r 
    
    return reward

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
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
