

import torch
import warnings
warnings.filterwarnings("ignore")
from isaacgym import gymtorch, gymapi, gymutil

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
from env.tasks.humanoid import dof_to_obs_smpl

TAR_ACTOR_ID = 1

class GameState(Enum):
    out_bound = 0
    green_win = 1
    red_win = 2
    idle = 3

class HumanoidFencing(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                            sim_params=sim_params,
                            physics_engine=physics_engine,
                            device_type=device_type,
                            device_id=device_id,
                            headless=headless)

        self.first_in = True
        
        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))
        
        # strike_body_names = cfg["env"]["strikeBodyNames"]
        strikeBodyNames=["L_Knee","L_Ankle","L_Toe", "R_Knee","R_Ankle","R_Toe",]
        self._strike_body_ids = self._build_key_body_ids_tensor(strikeBodyNames)
        footNames=["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
        self._foot_ids = self._build_key_body_ids_tensor(footNames)
        handNames = ["R_Hand"] # FEncing only needs one hand 
        self._hand_ids = self._build_key_body_ids_tensor(handNames)
        targetNames = ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"]
        self._target_ids = self._build_key_body_ids_tensor(targetNames)
        self.extend_body_pos = torch.tensor([[0.0, -0.98, 0.0]]).repeat(self.num_envs, 1, 1).to(self.device)

        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        self.sword_hit_history = torch.zeros((self.num_envs, 2)).to(self.device)
        self.win_frame_hit = 45 # hit more than this frames, you win. 
        
        ########## building for boxing area ##########
        self.bounding_box = torch.tensor([-1, 1, -7, 7,]).to(self.device) # x_min, x_max, y_min, y_max
        self.bounding_box_points = torch.tensor([[[-1, -7, 0], [1, 7, 0]]]).repeat(self.num_envs, 1, 1).to(self.device)
        self.reward_weights = {
            'reward_f': 0.1,
            'reward_v': 0.1,
            'reward_s': 0.2,
            'reward_t': 1,
            'reward_h': 0.6
        }

        

        self.env_ids_all = torch.arange(self.num_envs).to(self.device)
        self.warmup_time = int(250/self.dt) # 10 minutes wall time
        self.step_counter = 0

    def sample_position_on_field(self, n):
        inside_x, inside_y = 1, 1 # 1m inside the bounding box
        x = torch.FloatTensor(n).uniform_(self.bounding_box[0] + inside_x, self.bounding_box[1] - inside_x).to(self.device)
        y = torch.FloatTensor(n).uniform_(self.bounding_box[2] + inside_y, self.bounding_box[3] - inside_y).to(self.device)
        return torch.stack([x, y], dim=-1)
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 154
        return obs_size

    def post_physics_step(self):
        self.out_bound, self.red_win, self.green_win = self.check_game_state()
        self.step_counter += 1
        super().post_physics_step()
        
        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return
  
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(env_ids)
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel
    
    def _reset_ref_state_init(self, env_ids):
        self.sword_hit_history[:] = 0
        
        num_envs = env_ids.shape[0]
        
        root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list, rb_pos_list, rb_rot_list, body_vel_list, body_ang_vel_list = [], [], [], [], [], [], [], [], [], []
        for i in range(self.num_agents):
            motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._sample_ref_state(env_ids)
            
            if i%2 == 0:
                root_pos[:, :3] = torch.tensor([0, -1.5, 0.9]).to(self.device)
                root_rot[:] = torch.tensor([[0.    , 0.    , 0.7071, 0.7071]]).to(self.device)
            else:
                root_pos[:, :3] = torch.tensor([0, 1.5, 0.9]).to(self.device)
                root_rot[:] = torch.tensor([[0.    ,  0.    , -0.7071,  0.7071]]).to(self.device)

            root_vel[:] = 0
            dof_vel[:] = 0
            dof_pos[:] = 0 # zero pose for start. 

            root_pos_list.append(root_pos); root_rot_list.append(root_rot); dof_pos_list.append(dof_pos); root_vel_list.append(root_vel); root_ang_vel_list.append(root_ang_vel); dof_vel_list.append(dof_vel)
            rb_pos_list.append(rb_pos); rb_rot_list.append(rb_rot); body_vel_list.append(body_vel); body_ang_vel_list.append(body_ang_vel)
        
        self._set_env_state(env_ids=env_ids, root_pos_list=root_pos_list, root_rot_list=root_rot_list, dof_pos_list=dof_pos_list, \
            root_vel_list=root_vel_list, root_ang_vel_list=root_ang_vel_list, dof_vel_list=dof_vel_list, rigid_body_pos_list=rb_pos_list, rigid_body_rot_list=rb_rot_list,\
                rigid_body_vel_list=body_vel_list, rigid_body_ang_vel_list=body_ang_vel_list)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids] = motion_times
        self._sampled_motion_ids[env_ids] = motion_ids
        if flags.follow:
            self.start = True  ## Updating camera when reset
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        return
    
    def _push_robots(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = 5
        random_agent = np.random.randint(0, self.num_agents)
        
        self._humanoid_root_states_list[random_agent][env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        
        for i in range(self.num_agents):
            self._prev_root_pos_list[i] = self._humanoid_root_states_list[i][..., 0:3].clone()

        return
    
    def get_sword_tip_pos(self):
        extend_curr_pos_list = []
        for i in range(self.num_agents):
            body_pos = self._rigid_body_pos_list[i]
            body_rot = self._rigid_body_rot_list[i]
            extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self._hand_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self._hand_ids]
            extend_curr_pos_list.append(extend_curr_pos)
        return extend_curr_pos_list
    
    def check_sword_hit(self):
        sword_tip_pos_list = self.get_sword_tip_pos()
        sword_hit_list = []
        for i in range(self.num_agents):
            sword_tip_pos = sword_tip_pos_list[i] # my sword tip
            target_pos = self._rigid_body_pos_list[1 - i][:, self._target_ids] # hit target pos
            dist = torch.linalg.norm(sword_tip_pos - target_pos, dim=-1)
            distance_check =  dist < 0.1
            contact_check = torch.norm(self._contact_forces_list[1 - i][:, self._target_ids], dim = -1) > 50
            
            sword_hit = torch.any(torch.logical_and(distance_check, contact_check), dim = -1)[:, None] # my sowrd hit. 
            if flags.test and sword_hit.sum() > 0:
                if i == 0: print("Green Hit")
                if i == 1: print("Red Hit")
            
            sword_hit_list.append(sword_hit)
                
        return sword_hit_list
    
    def check_game_state(self):
        fuzzy = 0.1
        out_bound = torch.zeros(self.num_envs).to(self.device).bool()
        for i in range(self.num_agents):
            root_pos = self._humanoid_root_states_list[i][..., 0:3]
            out_bound += torch.logical_or(torch.logical_or(root_pos[..., 0] < self.bounding_box[0] - fuzzy,   root_pos[..., 0] > self.bounding_box[1] + fuzzy),   torch.logical_or(root_pos[..., 1] < self.bounding_box[2] - fuzzy,  root_pos[..., 1] > self.bounding_box[3] + fuzzy))

        #### TODO check hitting
        self.sword_hit_list = self.check_sword_hit()
        self.sword_hit_history[:, 0:1] += self.sword_hit_list[0].float()
        self.sword_hit_history[:, 1:2] += self.sword_hit_list[1].float()
        
        
        green_win = self.sword_hit_history[:, 0] > self.win_frame_hit
        red_win = self.sword_hit_history[:, 1] > self.win_frame_hit
        return out_bound, red_win, green_win
    
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        num_envs = env_ids.shape[0]
        sword_tip_pos_list = self.get_sword_tip_pos()
        for i in range(self.num_agents):
            root_states = self._humanoid_root_states_list[i][env_ids]
            body_pos = self._rigid_body_pos_list[i][env_ids]
            contact_force = self._contact_forces_list[i][env_ids]
            contact_force_norm = torch.linalg.norm(contact_force, dim=-1)
            self_sword_pos = sword_tip_pos_list[i][env_ids]
            
            if i%2 == 0:
                opponent_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 13)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_body_rot = torch.stack([self._rigid_body_rot_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 4)).to(self.device)
                opponent_dof_pos = torch.stack([self._dof_pos_list[j][env_ids]for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_dof_vel = torch.stack([self._dof_vel_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_contact_force_norm = torch.linalg.norm(opponent_contact_force, dim=-1)

            else:
                opponent_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 13)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_body_rot = torch.stack([self._rigid_body_rot_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 4)).to(self.device)
                opponent_dof_pos = torch.stack([self._dof_pos_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_dof_vel = torch.stack([self._dof_vel_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_contact_force_norm = torch.linalg.norm(opponent_contact_force, dim=-1)

        
            obs = compute_fencing_observation(root_states, body_pos, opponent_root_states[:, 0], opponent_body_pos[:, 0], opponent_body_rot[:, 0], 
                                             opponent_dof_pos[:, 0], opponent_dof_vel[:, 0], contact_force_norm, opponent_contact_force_norm[:, 0], self.bounding_box_points[env_ids], 
                                             self_sword_pos, self._target_ids)                        
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):
        sword_tip_pos_list = self.get_sword_tip_pos()
        for i in range(self.num_agents):
            root_state = self._humanoid_root_states_list[i]
            prev_root_state = self._prev_root_pos_list[i]
            contact_force = self._contact_forces_list[i]
            body_pos = self._rigid_body_pos_list[i]


            if i % 2 == 0:
                opponent_root_state = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)
            else:
                opponent_root_state = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)

            contact_force_norm = torch.linalg.norm(contact_force, dim=-1)
            opponent_contact_force_norm = torch.linalg.norm(opponent_contact_force, dim=-1)
            lower_clipped_force = 100.0
            upper_clipped_force = 1200
            
            contact_force_norm[contact_force_norm<lower_clipped_force] = 0
            contact_force_norm[contact_force_norm>upper_clipped_force] = upper_clipped_force
            opponent_contact_force_norm[:, 0][opponent_contact_force_norm[:, 0]<lower_clipped_force] = 0
            opponent_contact_force_norm[:, 0][opponent_contact_force_norm[:, 0]>upper_clipped_force] = upper_clipped_force

            self_fallen = compute_humanoid_reset_in_reward(self.reset_buf, self.progress_buf, self._foot_ids, body_pos, self._enable_early_termination, self._termination_heights)
            opponent_fallen = compute_humanoid_reset_in_reward(self.reset_buf, self.progress_buf, self._foot_ids, opponent_body_pos[:, 0], self._enable_early_termination, self._termination_heights)
            sword_tip_pos = sword_tip_pos_list[i]
            
            sword_hit_state = torch.cat([self.sword_hit_list[i], self.sword_hit_list[1-i]], dim = -1)
            
            reward, reward_raw = compute_fencing_reward(root_state, prev_root_state, body_pos, self_fallen, opponent_fallen,
                                                        contact_force_norm, opponent_contact_force_norm[:,0], 
                                                        opponent_root_state[:, 0],  opponent_body_pos[:, 0], sword_tip_pos, sword_hit_state, self._target_ids, self.dt, self.reward_weights)
            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = reward
            # if flags.test:
                # print(i, reward, reward_raw)
            
        if flags.test:
            np.set_printoptions(precision=4, suppress=1)
            if self.green_win.sum() > 0:
                print("Green Win!!")
                # import ipdb; ipdb.set_trace()
            elif self.red_win.sum() > 0:
                print("Red Win!!")
                
        return
    
    

    def _compute_reset(self):
        game_done = torch.logical_or(self.out_bound, torch.logical_or(self.red_win, self.green_win))
            
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return
    
    def _draw_task(self):
        # build retangle for boxing area
        if self.first_in:
            self.gym.clear_lines(self.viewer)
            fuzzy = 0.01
            x_min, x_max, y_min, y_max = self.bounding_box[0], self.bounding_box[1], self.bounding_box[2], self.bounding_box[3]
            for i, env_ptr in enumerate(self.envs):
                bbox = np.array([x_min + fuzzy , x_max - fuzzy, y_min + fuzzy, y_max - fuzzy , 0.05])
                vertices = np.array([
                        [bbox[0], bbox[2], bbox[4]],  # x_min, y_min, z_i
                        [bbox[1], bbox[2], bbox[4]],  # x_max, y_min, z_i
                        [bbox[1], bbox[3], bbox[4]],  # x_max, y_max, z_i
                        [bbox[0], bbox[3], bbox[4]],  # x_min, y_max, z_i
                    ])
                
                lines = np.array([
                        [vertices[0], vertices[1]],
                        [vertices[1], vertices[2]],
                        [vertices[2], vertices[3]],
                        [vertices[3], vertices[0]],], dtype=np.float32)
                
                cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)
        
        sword_tip_pos_list = self.get_sword_tip_pos()
        for i in range(self.num_agents):
            for env_id in range(self.num_envs):
                sword_tip_pos = sword_tip_pos_list[i]
                sphere_geom_marker = gymutil.WireframeSphereGeometry(0.05, 10, 10, None, color=(0.0, 1, 0.0) )
                sphere_pose = gymapi.Transform(gymapi.Vec3(sword_tip_pos[env_id,  0, 0], sword_tip_pos[env_id,  0, 1], sword_tip_pos[env_id, 0, 2]), r=None)
                gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 


    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_pos')):
            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        tar_pos = self._target_states[0, 0:3].cpu().numpy()
        self._output_motion_target_pos.append(tar_pos)

        tar_rot = self._target_states[0, 3:7].cpu().numpy()
        self._output_motion_target_rot.append(tar_rot)
        
        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_pos) > 1):
            output_tar_pos = np.array(self._output_motion_target_pos)
            output_tar_rot = np.array(self._output_motion_target_rot)
            output_data = np.concatenate([output_tar_pos, output_tar_rot], axis=-1)
            np.save('output/record_tar_motion.npy', output_data)

            self._output_motion_target_pos = []
            self._output_motion_target_rot = []
        return


class HumanoidFencingZ(HumanoidFencing):
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
    
    def _compute_reset(self):
        
        game_done = torch.logical_or(self.out_bound, torch.logical_or(self.red_win, self.green_win))
        if self.step_counter > self.warmup_time or flags.test:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset_z(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, 
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        else:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

# borrow from NCP https://github.com/Tencent-RoboticsX/NCP/blob/master/ncp/env/tasks/humanoid_boxing.py
@torch.jit.script
def compute_fencing_observation(self_root_state, self_body_pos, oppo_root_state, oppo_body_pos, oppo_body_rot,
                                  oppo_dof_pos, oppo_dof_vel, self_contact_norm, oppo_contact_norm, bounding_box_points,
                                  self_hand_pos, target_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    # root info
    self_root_pos = self_root_state[:, 0:3]
    self_root_rot = self_root_state[:, 3:7]
    oppo_root_pos = oppo_root_state[:, 0:3]
    oppo_root_rot = oppo_root_state[:, 3:7]
    oppo_root_vel = oppo_root_state[:, 7:10]

    heading_rot_inv = torch_utils.calc_heading_quat_inv(self_root_rot)
    root_pos_diff = oppo_root_pos - self_root_pos
    root_pos_diff[..., -1] = oppo_root_pos[..., -1]
    local_root_pos_diff = torch_utils.quat_rotate(heading_rot_inv, root_pos_diff)

    local_tar_rot = torch_utils.quat_mul(heading_rot_inv, oppo_root_rot)
    local_root_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    local_oppo_vel = torch_utils.quat_rotate(heading_rot_inv, oppo_root_vel)

    oppo_body_pos_diff = oppo_body_pos - self_root_pos[:, None]
    flat_oppo_body_pos_diff = oppo_body_pos_diff.view(oppo_body_pos_diff.shape[0] * oppo_body_pos_diff.shape[1],
                                              oppo_body_pos_diff.shape[2])
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, oppo_body_pos_diff.shape[1], 1))

    flat_heading_rot = heading_rot_inv_expand.view(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
                                               heading_rot_inv_expand.shape[2])
    local_flat_oppo_body_pos_diff =  torch_utils.quat_rotate(flat_heading_rot, flat_oppo_body_pos_diff)
    local_flat_oppo_body_pos_diff = local_flat_oppo_body_pos_diff.view(oppo_body_pos_diff.shape[0], oppo_body_pos_diff.shape[1] * oppo_body_pos_diff.shape[2])

    
    self_hand_pos = self_hand_pos.unsqueeze(-2)
    oppo_target_pos = oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot = heading_rot_inv.unsqueeze(-2).repeat((1, global_target_hand_pos_diff.shape[0] // heading_rot_inv.shape[0], 1))
    
    local_target_hand_pos_diff = torch_utils.quat_rotate(flat_heading_rot.view(-1, 4), global_target_hand_pos_diff)
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot.shape[0], -1)

    local_bounding_pos = bounding_box_points - self_root_pos[:, None]
    local_bounding_pos = torch_utils.my_quat_rotate(heading_rot_inv[:, None].repeat(1, 2, 1).view(-1, 4), local_bounding_pos.view(-1, 3))[..., :2].reshape(self_root_pos.shape[0], -1)

    obs = torch.cat([local_root_pos_diff, local_root_rot_obs, local_oppo_vel, local_flat_oppo_body_pos_diff,
                     local_target_hand_pos_diff, self_contact_norm, oppo_contact_norm, local_bounding_pos], dim=-1)
    return obs


@torch.jit.script
def compute_fencing_reward(root_state, prev_root_pos, body_pos, self_terminated, oppo_terminated, self_force_norm, oppo_force_norm, tar_pos, tar_body_pos, sword_tip_pos, hit_state, target_ids, dt, reward_weights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, float]) -> Tuple[Tensor, Tensor]
    reward_f, reward_v, reward_s, reward_t, reward_h = reward_weights['reward_f'], reward_weights['reward_v'], reward_weights['reward_s'], reward_weights['reward_t'], reward_weights['reward_h']

    tar_speed = 1.0
    vel_err_scale = 4.0
    facing_err_scale = 2.0

    root_pos = root_state[..., 0:3]
    root_rot = root_state[..., 3:7]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    tar_dir_3d = tar_pos[..., 0:3] - root_pos[..., 0:3]
    tar_dir_3d[..., -1] = 0.0
    tar_dir_3d = torch.nn.functional.normalize(tar_dir_3d, dim=-1)
    tar_dir_local = torch_utils.quat_rotate(heading_rot_inv, tar_dir_3d)
    facing_reward = torch.exp(-facing_err_scale * (1 - tar_dir_local[..., 0]))
    strike_reward = torch.mean((oppo_force_norm - self_force_norm)[:, target_ids], dim=-1) / 10

    terminate_reward = hit_state[:, 0].float() - hit_state[:, 1].float()
    
    target_pos = tar_body_pos[:, target_ids, :]
    hit_dist = torch.linalg.norm(sword_tip_pos - target_pos, dim=-1).min(dim=-1).values
    hit_reward = torch.exp(10 * -hit_dist)

    reward = vel_reward * reward_v + facing_reward * reward_f + strike_reward * reward_s + terminate_reward * reward_t + hit_reward * reward_h
    reward_raw = torch.stack([vel_reward, facing_reward, terminate_reward, hit_reward], dim=-1)
    
    return reward, reward_raw


#@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                        strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, num_agents):
    # type: (Tensor, Tensor, list, list, Tensor, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos_list[0][..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = fall_height
        for i in range(1, num_agents):
            body_height = rigid_body_pos_list[i][..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)
            has_fallen_temp = fall_height

            has_fallen = torch.logical_or(has_fallen, has_fallen_temp)

        
        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

#@torch.jit.script
def compute_humanoid_reset_z(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, num_agents):
    # type: (Tensor, Tensor, list, Tensor, list, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    
    terminated = torch.zeros_like(reset_buf)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

# borrow from NCP https://github.com/Tencent-RoboticsX/NCP/blob/master/ncp/env/tasks/humanoid_boxing.py
@torch.jit.script
def compute_humanoid_reset_in_reward(reset_buf, progress_buf, foot_ids, rigid_body_pos,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor) -> Tensor    
    

    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, foot_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = fall_height
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    return terminated