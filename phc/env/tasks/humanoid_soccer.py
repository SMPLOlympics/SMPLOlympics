import torch

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

class GameState(Enum):
    out_bound = 0
    green_win = 1
    red_win = 2
    idle = 3

##### Positive y Green, green goal post (red score for 1),  Negative y red, red goal post (Green Score for 0)
class HumanoidSoccer(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.first_in = True
        
        
        self._prev_ball_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))
        
        # strike_body_names = cfg["env"]["strikeBodyNames"]
        strikeBodyNames=["L_Knee","L_Ankle","L_Toe", "R_Knee","R_Ankle","R_Toe",]
        self._strike_body_ids = self._build_key_body_ids_tensor(strikeBodyNames)
        self._build_target_tensors()
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        self.bounding_box = torch.tensor([-10, 10, -16, 16,]).to(self.device) # x_min, x_max, y_min, y_max
        self.bounding_box_points = torch.tensor([[[-10, -16, 0], [10, 16, 0]]]).repeat(self.num_envs, 1, 1).to(self.device)
        
        self.goal_bound_green = torch.tensor([-1.9, 1.9, 16, 18,  0, 2]).to(self.device) # x_min, x_max, y_min, y_max, z_min, z_max
        self.goal_bound_red = torch.tensor([-1.9, 1.9, -18, -16, 0, 2]).to(self.device)
        
        self.goal_pos_green = torch.tensor([[0, 17, 1]]).to(self.device).repeat(self.num_envs, 1)
        self.goal_pos_red = torch.tensor([[0, -17, 1]]).to(self.device).repeat(self.num_envs, 1)
        self.env_ids_all = torch.arange(self.num_envs).to(self.device)
        
        self.warmup_time = int(250/self.dt) # 10 minutes wall time
        self.push_interval = self.cfg.env.get("push_interval", 300) # Push one of the humanoid every 60 steps. 
        self.step_counter = 0
        
        return
    
    def sample_position_on_field(self, n):
        inside_x, inside_y = 1, 3
        x = torch.FloatTensor(n).uniform_(self.bounding_box[0] + inside_x, self.bounding_box[1] - inside_x).to(self.device)
        y = torch.FloatTensor(n).uniform_(self.bounding_box[2] + inside_y, self.bounding_box[3] - inside_y).to(self.device)
        return torch.stack([x, y], dim=-1)
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 18 + np.ceil(self.num_agents / 2).astype(int) * (3 + 3 + 6 + 3) + 4
            
            if self.num_agents > 2:
                obs_size += (np.ceil(self.num_agents / 2).astype(int) - 1) * (3 + 3 + 6 + 3)
            # obs_size = 18 
        return obs_size
    
    def post_physics_step(self):
        self.out_bound, self.red_win, self.green_win = self.check_game_state()
        self.step_counter += 1
        super().post_physics_step()
        
        if self.cfg.env.get("push_robot", False):
            env_ids = (self.progress_buf % self.push_interval == 0).nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                self._push_robots(env_ids)

        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return
    
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
        asset_file = "soccer.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 60
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        prp = self.gym.get_asset_rigid_shape_properties(self._target_asset)
        for each in prp:
            each.friction = 1
            each.restitution = 0.8
            each.contact_offset = 0.02
            each.rolling_friction = 0.1

        self.gym.set_asset_rigid_shape_properties(self._target_asset,prp)


        asset_root = "phc/data/assets/urdf/"
        asset_file = "soccer_bound.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 25
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        # asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.fix_base_link = True
        self._soccer_bound_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        asset_root = "phc/data/assets/urdf/"
        asset_file = "goalpost.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 25
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        # asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.fix_base_link = True
        self._goalpost_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return

    def _build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", env_id, 0)
        self._target_handles.append(target_handle)
        default_pose = gymapi.Transform()
        if self.cfg.env.get("has_soccer_bound", True):
            _ = self.gym.create_actor(env_ptr, self._soccer_bound_asset, default_pose, "soccer_bound", env_id, 0)
        _ = self.gym.create_actor(env_ptr, self._goalpost_asset, default_pose, "goalpost", env_id, 0)
        
        
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self.num_agents, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_target(self, env_ids):
        n = len(env_ids)
        if flags.test:
            self._target_states[env_ids, 0] = 0
            self._target_states[env_ids, 1] = 0
            random_pos = self.sample_position_on_field(n)
            self._target_states[env_ids, 0] = random_pos[..., 0]
            self._target_states[env_ids, 1] = random_pos[..., 1]
        else:
            random_pos = self.sample_position_on_field(n)
            self._target_states[env_ids, 0] = random_pos[..., 0]
            self._target_states[env_ids, 1] = random_pos[..., 1]
        self._target_states[env_ids, 2] = 0.5
        
        
        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        return
    
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(env_ids)
        random_pos = self.sample_position_on_field(len(env_ids))
        root_pos[..., :2] = random_pos
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
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

        self._prev_ball_pos = self._target_states[..., 0:3].clone()
        return
    
    def check_game_state(self):
        tar_pos = self._target_states[..., 0:3]
        if flags.test:
            fuzzy = 30
        else:
            fuzzy = 0.1
        out_bound = torch.logical_or(torch.logical_or(tar_pos[..., 0] < self.bounding_box[0] - fuzzy,   tar_pos[..., 0] > self.bounding_box[1] + fuzzy),   torch.logical_or(tar_pos[..., 1] < self.bounding_box[2] - fuzzy,  tar_pos[..., 1] > self.bounding_box[3] + fuzzy))
        
        
        # red_win = torch.logical_and(torch.logical_and(torch.logical_and(tar_pos[..., 0] >= self.goal_bound_green[0], tar_pos[..., 0] <= self.goal_bound_green[1]), torch.logical_and(tar_pos[..., 1] >= self.goal_bound_green[2], tar_pos[..., 1] <= self.goal_bound_green[3])),  torch.logical_and(tar_pos[..., 2] >= self.goal_bound_green[4], tar_pos[..., 2] <= self.goal_bound_green[5]))
        # green_win = torch.logical_and(torch.logical_and(torch.logical_and(tar_pos[..., 0] >= self.goal_bound_red[0], tar_pos[..., 0] <= self.goal_bound_red[1]), torch.logical_and(tar_pos[..., 1] >= self.goal_bound_red[2], tar_pos[..., 1] <= self.goal_bound_red[3])),  torch.logical_and(tar_pos[..., 2] >= self.goal_bound_red[4], tar_pos[..., 2] <= self.goal_bound_red[5]))
        red_win = torch.zeros_like(out_bound)
        green_win = red_win
        
        return out_bound, red_win, green_win
    
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []
        for i in range(self.num_agents):
            goal_pos = self.goal_pos_red if i % 2 == 0 else self.goal_pos_green
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
                tar_states = self._target_states
                bounding_box_points = self.bounding_box_points
            else:
                goal_pos = goal_pos[env_ids]
                root_states = self._humanoid_root_states_list[i][env_ids]
                tar_states = self._target_states[env_ids]
                bounding_box_points = self.bounding_box_points[env_ids]
            

            if i % 2 == 0:
                if (env_ids is None):
                    opponent_root_states = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                else:
                    opponent_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((len(env_ids), 1, 13)).to(self.device)
            else:
                if (env_ids is None):
                    opponent_root_states = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                else:
                    opponent_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((len(env_ids), 1, 13)).to(self.device)
            
            if self.num_agents > 2 :
                if i % 2 == 0:
                    if (env_ids is None):
                        teammate_root_states = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j != i and j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                    else:
                        teammate_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j != i and j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((len(env_ids), 1, 13)).to(self.device)
                else:
                    if (env_ids is None):
                        teammate_root_states = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j != i and j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                    else:
                        teammate_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j != i and j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((len(env_ids), 1, 13)).to(self.device)
            else:
                teammate_root_states = torch.zeros((0, 1, 13)).to(self.device)            
                
            obs = compute_soccer_observations(root_states, tar_states, goal_pos, bounding_box_points,  opponent_root_states, teammate_root_states)
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            char_root_state = self._humanoid_root_states_list[i]
            goal_pos = self.goal_pos_red if i % 2 == 0 else self.goal_pos_green
            
            reward, reward_raw = compute_soccer_reward(self._target_states, char_root_state, goal_pos, self._prev_ball_pos, self._prev_root_pos_list[i], self.dt)
            
            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = reward
            ending_reward = 10
            if i % 2 == 0: # Green 
                self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] += self.green_win.float() * ending_reward
                self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] -= self.red_win.float() * ending_reward
            else: # Red
                self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] += self.red_win.float() * ending_reward
                self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] -= self.green_win.float() * ending_reward
                
            # if flags.test:
            #     np.set_printoptions(precision=4, suppress=1)
            #     print(i, ":", reward_raw.numpy())
            #     if self.green_win.sum() > 0:
            #         print("Green Win!!")
            #         # import ipdb; ipdb.set_trace()
            #     elif self.red_win.sum() > 0:
            #         print("Red Win!!")
            #         # import ipdb; ipdb.set_trace()
                    
        return

    def _compute_reset(self):
        
        game_done = torch.logical_or(self.out_bound, torch.logical_or(self.red_win, self.green_win))
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, self._tar_contact_forces,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        self.reset_buf[:] = torch.logical_or(self.reset_buf, game_done)
        return

    def _draw_task(self):
        if self.first_in:
            self.first_in = False
            self.gym.clear_lines(self.viewer)
            fuzzy = 0.05
            for i, env_ptr in enumerate(self.envs):
                x_min, x_max, y_min, y_max, z_min, z_max = self.goal_bound_green[0], self.goal_bound_green[1], self.goal_bound_green[2], self.goal_bound_green[3], self.goal_bound_green[4], self.goal_bound_green[5]
                bbox = np.array([x_min - fuzzy , x_max + fuzzy, y_min - fuzzy, y_max + fuzzy , z_min - fuzzy, z_max + fuzzy ])
                vertices = np.array([
                    [bbox[0], bbox[2], bbox[4]],  # x_min, y_min, z_min
                    [bbox[1], bbox[2], bbox[4]],  # x_max, y_min, z_min
                    [bbox[1], bbox[3], bbox[4]],  # x_max, y_max, z_min
                    [bbox[0], bbox[3], bbox[4]],  # x_min, y_max, z_min
                    [bbox[0], bbox[2], bbox[5]],  # x_min, y_min, z_max
                    [bbox[1], bbox[2], bbox[5]],  # x_max, y_min, z_max
                    [bbox[1], bbox[3], bbox[5]],  # x_max, y_max, z_max
                    [bbox[0], bbox[3], bbox[5]]   # x_min, y_max, z_max
                ])
                # Define the lines for the bounding box (each line is a pair of vertices)
                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]],
                    [vertices[4], vertices[5]],
                    [vertices[5], vertices[6]],
                    [vertices[6], vertices[7]],
                    [vertices[7], vertices[4]],
                    [vertices[0], vertices[4]],
                    [vertices[1], vertices[5]],
                    [vertices[2], vertices[6]],
                    [vertices[3], vertices[7]]
                ])
                cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)

                x_min, x_max, y_min, y_max, z_min, z_max = self.goal_bound_red[0], self.goal_bound_red[1], self.goal_bound_red[2], self.goal_bound_red[3], self.goal_bound_red[4], self.goal_bound_red[5]
                bbox = np.array([x_min - fuzzy , x_max + fuzzy, y_min - fuzzy, y_max + fuzzy , z_min - fuzzy, z_max + fuzzy ])
                vertices = np.array([
                    [bbox[0], bbox[2], bbox[4]],  # x_min, y_min, z_min
                    [bbox[1], bbox[2], bbox[4]],  # x_max, y_min, z_min
                    [bbox[1], bbox[3], bbox[4]],  # x_max, y_max, z_min
                    [bbox[0], bbox[3], bbox[4]],  # x_min, y_max, z_min
                    [bbox[0], bbox[2], bbox[5]],  # x_min, y_min, z_max
                    [bbox[1], bbox[2], bbox[5]],  # x_max, y_min, z_max
                    [bbox[1], bbox[3], bbox[5]],  # x_max, y_max, z_max
                    [bbox[0], bbox[3], bbox[5]]   # x_min, y_max, z_max
                ])
                # Define the lines for the bounding box (each line is a pair of vertices)
                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]],
                    [vertices[4], vertices[5]],
                    [vertices[5], vertices[6]],
                    [vertices[6], vertices[7]],
                    [vertices[7], vertices[4]],
                    [vertices[0], vertices[4]],
                    [vertices[1], vertices[5]],
                    [vertices[2], vertices[6]],
                    [vertices[3], vertices[7]]
                ])
                cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)
                    
                fuzzy = 0.01
                x_min, x_max, y_min, y_max = self.bounding_box[0], self.bounding_box[1], self.bounding_box[2], self.bounding_box[3]
                bbox = np.array([x_min + fuzzy , x_max - fuzzy, y_min + fuzzy, y_max - fuzzy , z_min + fuzzy, z_max - fuzzy ])
                vertices = np.array([
                    [bbox[0], bbox[2], bbox[4]],  # x_min, y_min, z_min
                    [bbox[1], bbox[2], bbox[4]],  # x_max, y_min, z_min
                    [bbox[1], bbox[3], bbox[4]],  # x_max, y_max, z_min
                    [bbox[0], bbox[3], bbox[4]],  # x_min, y_max, z_min
                    [bbox[0], bbox[2], bbox[5]],  # x_min, y_min, z_max
                    [bbox[1], bbox[2], bbox[5]],  # x_max, y_min, z_max
                    [bbox[1], bbox[3], bbox[5]],  # x_max, y_max, z_max
                    [bbox[0], bbox[3], bbox[5]]   # x_min, y_max, z_max
                ])
                # Define the lines for the bounding box (each line is a pair of vertices)
                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]],
                    [vertices[4], vertices[5]],
                    [vertices[5], vertices[6]],
                    [vertices[6], vertices[7]],
                    [vertices[7], vertices[4]],
                    [vertices[0], vertices[4]],
                    [vertices[1], vertices[5]],
                    [vertices[2], vertices[6]],
                    [vertices[3], vertices[7]]
                ])
                cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)
                
        return

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
    
class HumanoidSoccerZ(HumanoidSoccer):
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
                                                           self._rigid_body_pos_list, self._tar_contact_forces,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        else:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, self._tar_contact_forces,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return

    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_soccer_observations(root_states, tar_states, goal_location, bounding_box_points, opponent_root_states, teammate_root_states):
    # type: (Tensor, Tensor , Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_tar_pos)
    local_tar_vel = torch_utils.my_quat_rotate(heading_rot_inv, tar_vel)
    local_tar_ang_vel = torch_utils.my_quat_rotate(heading_rot_inv, tar_ang_vel)
    
    local_goal_pos = goal_location - root_pos
    local_goal_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_goal_pos)

    local_tar_rot = quat_mul(heading_rot_inv, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)
    
    
    B, A, _ = opponent_root_states.shape
    opponent_root_pos = opponent_root_states[..., 0:3]
    opponent_root_rot = opponent_root_states[..., 3:7]
    opponent_root_vel = opponent_root_states[..., 7:10]
    opponent_root_angvel = opponent_root_states[..., 10:13]
    
    local_op_pos = opponent_root_pos - root_pos.unsqueeze(1)
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).repeat(1, A, 1)
    
    local_op_pos = torch_utils.my_quat_rotate(heading_rot_inv_expand.view(-1, 4), local_op_pos.view(-1, 3)).view(B, -1)
    local_op_rot = quat_mul(heading_rot_inv_expand.view(-1, 4), opponent_root_rot.view(-1, 4))
    local_op_rot = torch_utils.quat_to_tan_norm(local_op_rot).view(B, -1)
    local_op_vel = torch_utils.my_quat_rotate(heading_rot_inv_expand.view(-1, 4), opponent_root_vel.view(-1, 3)).view(B, -1)
    local_op_angvel = torch_utils.my_quat_rotate(heading_rot_inv_expand.view(-1, 4), opponent_root_angvel.view(-1, 3)).view(B, -1)
    # obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel, local_goal_pos, local_op_pos, local_op_rot, local_op_vel, local_op_angvel], dim=-1)
    
 
    
    local_bounding_pos = bounding_box_points - root_pos[:, None]
    local_bounding_pos = torch_utils.my_quat_rotate(heading_rot_inv[:, None].repeat(1, 2, 1).view(-1, 4), local_bounding_pos.view(-1, 3))[..., :2].reshape(B, -1)
    
    
    if len(teammate_root_states) > 0:
        B, A, _ = teammate_root_states.shape
        teammate_root_pos = teammate_root_states[..., 0:3]
        teammate_root_rot = teammate_root_states[..., 3:7]
        teammate_root_vel = teammate_root_states[..., 7:10]
        teammate_root_angvel = teammate_root_states[..., 10:13]
        
        local_teammate_pos = teammate_root_pos - root_pos.unsqueeze(1)
        heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).repeat(1, A, 1)
        
        local_tm_pos = torch_utils.my_quat_rotate(heading_rot_inv_expand.view(-1, 4), local_teammate_pos.view(-1, 3)).view(B, -1)
        local_tm_rot = quat_mul(heading_rot_inv_expand.view(-1, 4), teammate_root_rot.view(-1, 4))
        local_tm_rot = torch_utils.quat_to_tan_norm(local_tm_rot).view(B, -1)
        local_tm_vel = torch_utils.my_quat_rotate(heading_rot_inv_expand.view(-1, 4), teammate_root_vel.view(-1, 3)).view(B, -1)
        local_tm_angvel = torch_utils.my_quat_rotate(heading_rot_inv_expand.view(-1, 4), teammate_root_angvel.view(-1, 3)).view(B, -1)
        obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel, local_goal_pos, local_op_pos, local_op_rot, local_op_vel, local_op_angvel, local_tm_pos, local_tm_rot, local_tm_vel, local_tm_angvel, local_bounding_pos], dim=-1)
    else:
        obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel, local_goal_pos, local_op_pos, local_op_rot, local_op_vel, local_op_angvel, local_bounding_pos], dim=-1)
    
    return obs

@torch.jit.script
def ball_velocity_towards_goal(ball_position, ball_velocity, goal_center):

    # Calculate the direction vector from the ball to the goal center.
    direction_to_goal = goal_center - ball_position
    direction_to_goal_normalized = direction_to_goal / torch.norm(direction_to_goal)

    # Project the ball's velocity onto the direction to goal vector.
    velocity_towards_goal = (ball_velocity[..., None, :] @ direction_to_goal_normalized[..., None]).squeeze(-1).squeeze(-1)

    # The reward is the magnitude of the velocity towards the goal. 
    # If the velocity is negative (i.e., moving away from the goal), the reward is zero.
    reward = torch.clamp(velocity_towards_goal, min=0.0)

    return reward

@torch.jit.script
def player_velocity_towards_ball(player_position, player_velocity, ball_position):
    # Calculate the direction vector from the player to the ball.
    direction_to_ball = ball_position - player_position
    direction_to_ball_normalized = direction_to_ball / torch.norm(direction_to_ball)

    # Project the player's velocity onto the direction to ball vector.
    velocity_towards_ball = (player_velocity[..., None, :] @ direction_to_ball_normalized[..., None]).squeeze(-1).squeeze(-1)

    # Since we're interested in the magnitude, we take the absolute value.
    # This accounts for cases where the player might be moving away from the ball.
    magnitude_velocity_towards_ball = torch.clamp(velocity_towards_ball, min=0.0)

    return magnitude_velocity_towards_ball

@torch.jit.script
def distance_ball_to_target(ball_position, target_position):
    """
    Calculates the Euclidean distance between the ball and a target.
    
    Parameters:
    - ball_position: Tensor representing the ball's current position (x, y, z).
    - target_position: Tensor representing the target's position (x, y, z).
    
    Returns:
    - A scalar tensor representing the distance from the ball to the target.
    """

    # Calculate and return the Euclidean distance between the ball and the target.
    distance = torch.norm(ball_position - target_position, dim = -1)

    return distance

@torch.jit.script
def predict_entry_point(ball_pos, ball_vel, goal_pos):
    # TODO: have to include air drag and magnus force
    v_y = ball_vel[..., 1]
    ball_y = ball_pos[..., 1]
    goal_y = goal_pos[..., 1]

    h = ball_pos[..., 2]
    v_z = ball_vel[..., 2]

    v_x = ball_vel[..., 0]
    ball_x = ball_pos[..., 0]

    # if go to red goal -- vy needs to be positive. green goal negative
    will_reach = ((v_y > 0) & (ball_y < goal_y)) | ((v_y < 0) & (ball_y > goal_y))

    t = torch.zeros_like(v_y)
    t[will_reach] = (goal_y[will_reach] - ball_y[will_reach]) / v_y[will_reach]
    will_reach = torch.logical_and(will_reach, t > 0)
    will_reach = torch.logical_and(will_reach, t < 10)

    ball_entry_height = torch.clamp(h + v_z * t - 0.5 * 9.81 * t * t, min=0)
    ball_entry_x = ball_x + v_x * t

    # ball_entry_height = torch.clamp(ball_entry_height, min=-30, max=30)
    # ball_entry_x = torch.clamp(ball_entry_x, min=-30, max=30)
    
    return ball_entry_x, ball_entry_height

# @torch.jit.script
def compute_soccer_reward(ball_states,  root_state, goal_pos, prev_ball_pos,  prev_root_pos, dt):
    # type: (Tensor, Tensor, Tensor, Tensor,Tensor, float) -> Tuple[Tensor, Tensor]
    pos_err_scale = 5.0
    
    ball_pos = ball_states[..., 0:3]
    ball_vel = ball_states[..., 7:10]
    root_pos = root_state[..., 0:3]
    root_vel = root_state[..., 7:10]
    
    max_vel_player = 5
    max_vel_goal = 30
    
    ball_goal_velocity = ball_velocity_towards_goal(ball_pos, ball_vel, goal_pos)
    ball_goal_velocity_r = torch.clamp( ball_goal_velocity/max_vel_goal, min = 0, max=1.0)
    ball_entry_x, ball_entry_z = predict_entry_point(ball_pos, ball_vel, goal_pos)
    target_pos_x, target_pos_y, target_pos_z = goal_pos[..., 0], goal_pos[..., 1], goal_pos[..., 2]
    target_pos2d = torch.stack([target_pos_x, target_pos_z], dim=-1)
    ball_entry_pos2d = torch.stack([ball_entry_x, ball_entry_z], dim=-1)
    target_pos_diff = ball_entry_pos2d - target_pos2d
    target_pos_err = torch.sum(target_pos_diff * target_pos_diff, dim=-1)
    target_pos_reward = torch.exp(-pos_err_scale * target_pos_err)

    ball_to_goal_curr = distance_ball_to_target(ball_pos, goal_pos)
    ball_to_goal_prev = distance_ball_to_target(prev_ball_pos, goal_pos)
    ball_to_goal_closer_r = torch.clamp(ball_to_goal_prev - ball_to_goal_curr, min=-1,  max=1)  # ball getting closer to the goal

    ############### Testing
    prev_dist = torch.norm(prev_root_pos - ball_pos, dim=-1)
    curr_root_dist = torch.norm(root_pos - ball_pos, dim=-1)
    ball_player_closer_r = torch.clamp(prev_dist - curr_root_dist, min=0, max=1)   # cap max at 1/3
    
    closer_inxy = (torch.norm((ball_pos -  goal_pos)[:, :2], dim = -1) - torch.norm((prev_ball_pos - goal_pos)[:, :2], dim = -1)) < 0
    
    # is_too_far = curr_root_dist > 1
    # is_reaching = curr_root_dist <= 1
    
    # ball_to_goal_closer_r[~closer_inxy] = 0 
    # ball_goal_velocity_r[~closer_inxy] = 0 
    # target_pos_reward[~closer_inxy] = 0.0

    reward  = 0.05 * ball_player_closer_r + 0.2 * ball_to_goal_closer_r + 0.3 * ball_goal_velocity_r  + 0.5 * target_pos_reward

    reward_raw = torch.stack([ball_player_closer_r, ball_to_goal_closer_r, ball_goal_velocity_r, target_pos_reward], dim=-1)
    # reward = ball_to_goal_closer_r * 0.7 + ball_goal_velocity_r * 0.3
    # reward_raw = torch.stack([ball_to_goal_closer_r,  ball_goal_velocity_r], dim=-1)
    
    # reward = ball_to_goal_closer * 0.3  + ball_player_velocity_r * 0.2 + ball_goal_velocity_r * 0.5
    # reward_raw = torch.stack([ball_to_goal_closer, ball_player_velocity_r, ball_goal_velocity_r], dim=-1)
    
    # np.set_printoptions(precision=4, suppress=1)
    # print(reward.detach().numpy())
    # reward =  ball_player_velocity_r * 0.5 + closer_reward * 0.5
    # reward_raw = torch.stack([ball_player_velocity_r, closer_reward], dim=-1)

    return reward, reward_raw


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, num_agents):
    # type: (Tensor, Tensor, list, Tensor, list, Tensor, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf_list[0].clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos_list[0][..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        for i in range(1, num_agents):
            masked_contact_buf = contact_buf_list[i].clone()
            masked_contact_buf[:, contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
            fall_contact = torch.any(fall_contact, dim=-1)

            body_height = rigid_body_pos_list[i][..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)

            has_fallen_temp = torch.logical_and(fall_contact, fall_height)

            has_fallen = torch.logical_or(has_fallen, has_fallen_temp)



        # tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        # #strike_body_force = contact_buf[:, strike_body_id, :]
        # #strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        # nonstrike_body_force = masked_contact_buf
        # nonstrike_body_force[:, strike_body_ids, :] = 0
        
        # nonstrike_body_has_contact = torch.any(torch.sqrt(torch.square(torch.abs(nonstrike_body_force.sum(dim=-2))).sum(dim=-1)) > contact_force_threshold, dim=-1)
        # nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)
        
        # nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        # nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        # tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)
        
        # has_failed = torch.logical_or(has_fallen, tar_fail)
        
        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

# @torch.jit.script
def compute_humanoid_reset_z(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, num_agents):
    # type: (Tensor, Tensor, list, Tensor, list, Tensor, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    
    terminated = torch.zeros_like(reset_buf)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated