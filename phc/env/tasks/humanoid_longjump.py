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



class HumanoidLongjump(humanoid_amp_task.HumanoidAMPTask):
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
        self.first_in = True

        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))
 
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
 
        self.set_initial_root_state()
        self.goal = torch.tensor([30,0,1]).to(self.device)
        self.jump_start = 20
 
        self.tar_speed = 4
        return
    
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
            obs_size = 4
        return obs_size
    
  
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        
        for i in range(self.num_agents):
            self._prev_root_pos_list[i] = self._humanoid_root_states_list[i][..., 0:3].clone()

        return
    
    def _draw_task(self):
        if self.first_in:
            self.first_in=False
            self.gym.clear_lines(self.viewer)
            cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            for i, env_ptr in enumerate(self.envs):
                vertices = np.array([
                    [0, -0.5, 0],
                    [0, 0.5, 0],
                    [self.jump_start, 0.5, 0],
                    [self.jump_start, -0.5, 0]
                ], dtype=np.float32)

                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]]
                ])
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)
                
                vertices = np.array([
                    [self.jump_start, -1.5, 0],
                    [self.jump_start, 1.5, 0],
                    [self.goal[0], 1.5, 0],
                    [self.goal[0], -1.5, 0]
                ], dtype=np.float32)

                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]]
                ])
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)

      
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []
 
        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                
            
            obs = compute_longjump_observations(root_states, self.goal, self.jump_start)
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            # reward = 1
            root_states = self._humanoid_root_states_list[i]
            key_bodies_pos = self._rigid_body_pos_list[i][:, self._key_body_ids[:2], :]
            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_longjump_reward(root_states, self._prev_root_pos_list[i], 
                                                                                        self.goal, self.jump_start, self._rigid_body_pos_list, 
                                                                                        self._contact_forces_list, self._contact_body_ids)

        return

    def _compute_reset(self):
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, 
                                                           self.num_agents, self.jump_start)
        
        return

 
class HumanoidLongjumpZ(HumanoidLongjump):
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
def compute_longjump_observations(root_states, goal, jump_start):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
    local_tar_pos = goal - root_pos
    local_tar_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_tar_pos)

    humanoid_jumpstart_diff = jump_start - root_pos[:, 0: 1]
    obs = torch.concatenate((local_tar_pos, humanoid_jumpstart_diff ), dim=-1)
    return obs


 

# @torch.jit.script
def compute_longjump_reward(root_states, prev_root_pos, goal, jump_start,rigid_body_pos_list, contact_buf_list, contact_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    root_pos = root_states[:, 0:3]
    prev_dist = torch.norm(prev_root_pos - goal, dim=-1)
    curr_dist = torch.norm(root_pos - goal, dim=-1)
    closer_target_r = torch.clamp(prev_dist - curr_dist, min=0, max=1) 
 
    #root velocity in x reward
    vel_reward = root_states[:, 7]
 
    #jump height reward
    x_over_40 = torch.any(rigid_body_pos_list[0][:, contact_body_ids, 0] > jump_start, dim=-1) #shape 1024
    jump_height_reward= torch.zeros(root_states.shape[0]).to(root_states.device)
    jump_height_reward[x_over_40] = root_states[x_over_40, 2] 
    
    #end reward for jump length
    jump_length_reward = torch.zeros(root_states.shape[0]).to(root_states.device)
    force_threshold = 50
    contact_force_not_zero =  torch.sqrt(torch.sum(torch.sum(torch.square(contact_buf_list[0] ), dim=-1) , dim=-1) )>force_threshold 
    reset_x_over_40_and_contact_force_not_zero = torch.logical_and(x_over_40, contact_force_not_zero)
    
    jump_length_reward[reset_x_over_40_and_contact_force_not_zero] = torch.mean(rigid_body_pos_list[0][reset_x_over_40_and_contact_force_not_zero][:, :, 0], dim=-1) - jump_start
    
    #parameters
    closer_target_r *= 1
    vel_reward *= 0.01
    jump_height_reward *= 0.1
    jump_length_reward *= 30
    reward = closer_target_r + vel_reward + jump_height_reward   + jump_length_reward
    #print("closer_target_r", closer_target_r[0], "vel", vel_reward[0], "height", jump_height_reward[0], "length", jump_length_reward[0], "total", reward[0])
    return reward

# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           max_episode_length, 
                           enable_early_termination, termination_heights, num_agents, jump_start):
    # type: (Tensor, Tensor, list, Tensor, list, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    force_threshold = 50
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
 
        # ------------contact_buf_list[0].shape
        # torch.Size([1024, 24, 3])
        # ----------(Pdb) rigid_body_pos_list[0].shape
        # torch.Size([1024, 24, 3])
        # ----------(Pdb) contact_body_ids
        # tensor([7, 3, 8, 4], device='cuda:0')
        # -----------reset_buf.shape
        # torch.Size([1024])
        # -----------progress_buf.shape
        # torch.Size([1024])
        # ------------termination_heights
        # tensor([0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500,
        # 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500,
        # 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.3000], device='cuda:0')

        
        x_over_40 = torch.any(rigid_body_pos_list[0][:, contact_body_ids, 0] > jump_start, dim=-1) #shape 1024
 
        contact_force_not_zero =  torch.sqrt(torch.sum(torch.sum(torch.square(contact_buf_list[0] ), dim=-1) , dim=-1) )>force_threshold 
    
        reset_x_over_40_and_contact_force_not_zero = torch.logical_and(x_over_40, contact_force_not_zero)
        
        jump_length = torch.mean(rigid_body_pos_list[0][reset_x_over_40_and_contact_force_not_zero][:, :, 0], dim=-1) - jump_start
        if not torch.equal(jump_length, torch.tensor([]).to(jump_length.device)):
            print("jump length is ", jump_length)




 
        masked_contact_buf = contact_buf_list[0].clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        body_height = rigid_body_pos_list[0][..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        body_y = rigid_body_pos_list[0][..., 0, 1]
        body_out = torch.abs(body_y)>0.5
        
        has_fallen = torch.logical_or(fall_contact, fall_height) # don't touch the hurdle. 
        has_fallen = torch.logical_or(has_fallen, body_out)
        has_fallen = torch.logical_or(has_fallen, reset_x_over_40_and_contact_force_not_zero)


        if num_agents>1:
            for i in range(1, num_agents):
      
                masked_contact_buf = contact_buf_list[i].clone()
                masked_contact_buf[:, contact_body_ids, :] = 0
                fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

                body_height = rigid_body_pos_list[i][..., 2]
                fall_height = body_height < termination_heights
                fall_height[:, contact_body_ids] = False
                fall_height = torch.any(fall_height, dim=-1)

                has_fallen_temp = torch.logical_or(fall_contact, fall_height)

                has_fallen = torch.logical_or(has_fallen, has_fallen_temp)

                body_y = rigid_body_pos_list[i][..., 0, 1]
                body_out = torch.abs(body_y)>0.5
            
                has_fallen = torch.logical_or(has_fallen, body_out)
                has_fallen = torch.logical_or(has_fallen, reset_x_over_40_and_contact_force_not_zero)

        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

