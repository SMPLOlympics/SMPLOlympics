import torch
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import torch.nn.functional as F
import math
import pdb
import time
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *
from tqdm import tqdm
from scipy import ndimage
from phc.env.tasks.humanoid_golf_terrain import *
import random
from phc.utils.flags import flags
class HumanoidGolf(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.ball_no_contact_record = 0
        self.ball_pos_at_stop = torch.tensor([])
        self.ball_target_at_stop = torch.tensor([])
        self.total_test = 0

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.sensor_extent = cfg["env"].get("sensor_extent", 0.1)
        self.sensor_res = cfg["env"].get("sensor_res", 32)
        self.square_height_points = self.init_square_height_points()
        self.terrain_obs_type = self.cfg['env'].get("terrain_obs_type",
                                                    "square")
        self.terrain_obs = self.cfg['env'].get("terrain_obs", False)
        self.terrain_obs_root = self.cfg['env'].get("terrain_obs_root",
                                                    "pelvis")
        if self.terrain_obs_type == "fov":
            self.height_points = self.init_fov_height_points()
        elif self.terrain_obs_type == "square_fov":
            self.height_points = self.init_square_fov_height_points()
        elif self.terrain_obs_type == "square":
            self.height_points = self.square_height_points
        self.root_points = self.init_root_points()

        self.center_height_points = self.init_center_height_points()
        self.height_meas_scale = 5
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        self.first_in = True
        self.set_initial_root_state()
        self.set_racket_restitution()
        self._build_predefined_tensors()
        self._build_ball_state_tensors()
        # if not self.headless:
        #     self._build_target_state_tensors()
        return

    def _build_predefined_tensors(self):
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._forces = torch.zeros((self.num_envs, bodies_per_env, 3), device=self.device, dtype=torch.float)
        self._torques = torch.zeros((self.num_envs, bodies_per_env, 3), device=self.device, dtype=torch.float)

        self._ball_actor_id = self.num_agents
        self._tee_actor_id = self.num_agents + 1
        self._target_actor_id = self.num_agents + 2
        self._righthand_body_id = 23
        self._ball_body_id = 24
    
        self._paddle_to_hand = torch.tensor([0, -1.14, 0.], device=self.device, dtype=torch.float).repeat(self.num_envs,1)
        self._ball_paddle_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_ground_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_targets = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self._ball_land_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self._ball_prev_pos_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.int)
        self._ball_prev_vel_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.int)
        self.z_dir = torch.FloatTensor([0, 0, -1]).to(self.device)
    
    def set_initial_root_state(self):

        initial_humanoid_root_states = self._initial_humanoid_root_states_list[0].clone()
        initial_humanoid_root_states[..., 0] = 1
        initial_humanoid_root_states[..., 1] = 0
        initial_humanoid_root_states[..., 2] = 1 
        initial_humanoid_root_states[..., 3] = 0
        initial_humanoid_root_states[..., 4] = 0
        initial_humanoid_root_states[..., 5] = 1
        initial_humanoid_root_states[..., 6] = 0
        initial_humanoid_root_states[:, 7:13] = 0

        center_height = torch.mean( self.get_heights(initial_humanoid_root_states ) , dim=1)
        initial_humanoid_root_states[:, 2] += center_height
        self._initial_humanoid_root_states_list[0] = initial_humanoid_root_states
        
        self._initial_ball_root_states = initial_humanoid_root_states.clone()
        self._initial_ball_root_states[..., 0] = 0
        self._initial_ball_root_states[...,1] = 0
        self._initial_ball_root_states[...,2] = 0.021 + 0.05
        self._initial_ball_root_states[...,3:6] = 0
        self._initial_ball_root_states[...,6] = 1
        self._initial_ball_root_states[...,7] = 1e-9 # Zero speed will crash
        self._initial_ball_root_states[...,8:13] = 0

        self._initial_tee_root_states = initial_humanoid_root_states.clone()
        self._initial_tee_root_states[...,0] = 0
        self._initial_tee_root_states[...,1] = 0
        self._initial_tee_root_states[...,2] = 0.025
        self._initial_tee_root_states[...,3:6] = 0
        self._initial_tee_root_states[...,6] = 1
        self._initial_tee_root_states[...,7] = 1e-9  # Zero speed will crash
        self._initial_tee_root_states[...,8:13] = 0
        center_height =torch.mean( self.get_heights(self._initial_tee_root_states) , dim=-1)
        self._initial_tee_root_states[...,2] += center_height
        self._initial_ball_root_states[..., 2] += center_height
    
    def set_racket_restitution(self):
        for env_id in range(self.num_envs): 
            env_ptr = self.envs[env_id]
            
            rsp = self.gym.get_actor_rigid_shape_properties(env_ptr, self.humanoid_handles_list[env_id][0])
            assert len(rsp) == 25
            rsp[-1].restitution = 0.8
            rsp[-1].friction = 0.8
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.humanoid_handles_list[env_id][0],rsp)
    
    def _build_ball_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._ball_root_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self._ball_actor_id, :]
        self._tee_root_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self._tee_actor_id, :]
        self._ball_pos = self._ball_root_states[..., :3]
        self._ball_vel = self._ball_root_states[..., 7:10]
        self._ball_ang_vel = self._ball_root_states[..., 10:13]
        
        self._ball_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self._ball_actor_id
        self._tee_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self._tee_actor_id
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._ball_handles = []
        self._tee_handles = []
        
        self._load_ball_asset()
        self._load_tee_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_ball_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "golf_ball.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 100
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100
        # asset_options.density = 2000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        prp = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
        for each in prp:
            each.friction = 1
            each.restitution = 0.1
            # each.contact_offset = 0.02
            each.rolling_friction = 1

        self.gym.set_asset_rigid_shape_properties(self._ball_asset,prp)

        return 

    def _load_tee_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "golf_tee.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._tee_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        super()._build_env(env_id, env_ptr, humanoid_asset_list)
        
        self._build_ball(env_id, env_ptr) # id: num_agents
        self._build_tee(env_id, env_ptr) # id: num_agents+1

        return
    
    def _build_ball(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 3
        segmentation_id = 1
        default_pose = gymapi.Transform()
        
        ball_handle = self.gym.create_actor(env_ptr, self._ball_asset, default_pose, "ball", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
        self._ball_handles.append(ball_handle)

        return

    def _build_tee(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 1
        default_pose = gymapi.Transform()
        
        tee_handle = self.gym.create_actor(env_ptr, self._tee_asset, default_pose, "tee", col_group, col_filter, segmentation_id)
        rsp = gymapi.RigidShapeProperties()
        rsp.restitution = 0
        rsp.thickness = 0.2
        rsp.contact_offset = 0.02
        rsp.friction = 0.8
        rsp.rolling_friction = 0.1

        self.gym.set_actor_rigid_shape_properties(env_ptr, tee_handle, [rsp,rsp])

        self.gym.set_rigid_body_color(env_ptr, tee_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.2, 0.2))
        self._tee_handles.append(tee_handle)

        return
 
    def _reset_envs(self, env_ids):
        self._reset_env_ball(env_ids)
        super()._reset_envs(env_ids)

        return

    def _reset_env_ball(self, env_ids):
        n = len(env_ids)
        self.total_test += n
        if n>0:
            self._ball_prev_pos_buf[env_ids] = 0
            self._ball_prev_vel_buf[env_ids] = 0
            self._ball_paddle_contact_buf[env_ids] = 0
            self._ball_ground_contact_buf[env_ids] = 0
            self._ball_root_states[env_ids, :] = self._initial_ball_root_states[env_ids, :]
            self._tee_root_states[env_ids, :] = self._initial_tee_root_states[env_ids, :]

            # Set ball target
            self.max_distance = self.terrain.offset
            self._ball_targets[env_ids, 0] = 0
            self._ball_targets[env_ids, 1] =  (-1) * self.max_distance* torch.rand(len(env_ids)).to(self.device)
            env_ids_int32 = torch.cat([self._ball_actor_ids[env_ids], self._tee_actor_ids[env_ids]])
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 9 + 1024
        return obs_size
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._ball_prev_pos_buf = self._ball_pos.clone()
        self._ball_prev_vel_buf = self._ball_vel.clone()

        return

    def apply_external_force_to_ball(self, ball_root_states):
        R       = 0.021 # m, radius of ball
        rho     = 1.21  # kg/m^3, density of air
        kf      = (rho * math.pi * R * R) / 2
        BASE_CD = 0.55 # physics.usyd.edu.au/~cross/TRAJECTORIES/42.%20Ball%20Trajectories.pdf

        def get_cl(v, vs):
            """ Coefficient of lift, magnus effect """
            return 1 / (2 + abs(v/(vs + 1e-6)))

        def get_cd(v, vs):
            """ Coefficient of drag, affected by magnus. """
            return BASE_CD
            # return BASE_CD + 1 / pow(22.5 + 4.2 * pow(v/vs, 2.5), 0.4)

        vel_scalar = self._ball_vel.norm(dim=1).view(-1, 1)
        vel_scalar[vel_scalar == 0] += 1 # Avoid divide by 0
        vel_norm = self._ball_vel / vel_scalar
        g_tensor = torch.FloatTensor([[0, 0, -1]]).repeat(self.num_envs, 1).to(self.device)
        vel_tan = torch.cross(vel_norm, g_tensor)
        vspin = ball_root_states[:, 10:13].norm(dim=1).view(-1, 1) / (math.pi * 2)

        cd = get_cd(vel_scalar, vspin)
        cl = get_cl(vel_scalar, vspin)
        cl = cl * torch.where(vspin > 0, 
            -1 * torch.ones_like(cl),
            1 * torch.ones_like(cl)
        )
        force_drag = - kf * cd * vel_scalar * self._ball_vel
        force_lift = - kf * cl * vel_scalar ** 2 * torch.cross(vel_tan, vel_norm) 
        
        ball_forces = force_drag + force_lift
        ball_forces[self._ball_pos[:, 2] < R + 0.01, :] = 0 # No external forces when on the ground
        self._forces[:, self._ball_body_id, :] = ball_forces 

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.apply_external_force_to_ball(self._ball_root_states)

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self._forces), 
                gymtorch.unwrap_tensor(self._torques), gymapi.ENV_SPACE)

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            self._forces[:] = 0
            self._torques[:] = 0

            self.render()
    
    def post_physics_step(self):
        super().post_physics_step()
        return

    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []

        if (env_ids is None):
            ball_pos =  self._ball_pos.clone()
            ball_vel = self._ball_vel.clone()
            target_ball_pos = self._ball_targets.clone()
        else:
            ball_pos = self._ball_pos[env_ids].clone()
            ball_vel = self._ball_vel[env_ids].clone()
            target_ball_pos = self._ball_targets[env_ids].clone()

        if (env_ids is None):
            root_states = self._humanoid_root_states_list[0]
            righthand_pos = self._rigid_body_pos_list[0][:, self._righthand_body_id, :].clone()
            righthand_rot = self._rigid_body_rot_list[0][:, self._righthand_body_id, :].clone()
            paddle_pos = righthand_pos + quat_rotate(righthand_rot, self._paddle_to_hand.clone())
        else:
            root_states = self._humanoid_root_states_list[0][env_ids]
            righthand_pos = self._rigid_body_pos_list[0][env_ids, self._righthand_body_id, :].clone()
            righthand_rot = self._rigid_body_rot_list[0][env_ids, self._righthand_body_id, :].clone()
            paddle_pos = righthand_pos + quat_rotate(righthand_rot, self._paddle_to_hand[env_ids].clone())

        obs = compute_golf_ball_observations(root_states, ball_pos, paddle_pos, target_ball_pos)

        if env_ids is None:
            if self.terrain_obs:
                self.measured_heights = self.get_heights(root_states=root_states )
                if self.cfg['env'].get("use_center_height", False):
                    center_heights = self.get_heights(root_states=root_states )
                    center_heights = center_heights.mean(dim=-1, keepdim=True)
                    heights = center_heights - self.measured_heights
                    heights = torch.clip(heights, -3, 3.) * self.height_meas_scale  #
                else:
                    heights = torch.clip(root_states[:, 2:3] - self.measured_heights, -3, 3.) * self.height_meas_scale  #

                obs = torch.cat([obs, heights], dim=1)
        else:
            if self.terrain_obs:
                self.measured_heights = self.get_heights(root_states=root_states, env_ids=env_ids)
                if self.cfg['env'].get("use_center_height", False):
                    center_heights = self.get_heights(root_states=root_states, env_ids=env_ids)
                    center_heights = center_heights.mean(dim=-1, keepdim=True)
                    heights = center_heights - self.measured_heights
                    heights = torch.clip(heights, -3, 3.) * self.height_meas_scale  #
                else:
                    heights = torch.clip(root_states[:, 2:3] - self.measured_heights, -3,
                                         3.) * self.height_meas_scale  #

                obs = torch.cat([obs, heights], dim=1)

        obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):

        ball_pos = self._ball_pos[..., :3].clone()
        ball_prev_pos = self._ball_prev_pos_buf[..., :3].clone()
        ball_vel = self._ball_vel[..., :3].clone()

        height_threshold = (ball_pos[..., 2] < 0.4)
        ball_vel_z_change = torch.logical_and(self._ball_prev_vel_buf[..., 2]<0,  ball_vel[..., 2]>=0)

        predict_ball_land_pos = predict_land_point(ball_pos, ball_vel[..., :3])

        root_states = self._humanoid_root_states_list[0].clone()
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        righthand_pos = self._rigid_body_pos_list[0][:, self._righthand_body_id, :].clone()
        righthand_rot = self._rigid_body_rot_list[0][:, self._righthand_body_id, :].clone()
        paddle_pos = righthand_pos + quat_rotate(righthand_rot, self._paddle_to_hand.clone())

        ball_ground_contact_check = torch.logical_and(height_threshold, ball_vel_z_change)
        self._ball_ground_contact_buf += ball_ground_contact_check

        ball_paddle_contact_check = torch.logical_and(torch.norm(ball_pos - paddle_pos, dim=-1) < 0.2, torch.abs(ball_pos[:, 0]) > 0.01)
        self._ball_paddle_contact_buf += ball_paddle_contact_check

        self.rew_buf[:] = compute_ball_reward(root_pos, root_rot,
                                            ball_pos, ball_prev_pos, ball_vel, paddle_pos, self._ball_paddle_contact_buf, self._ball_ground_contact_buf, 
                                            self._ball_targets, predict_ball_land_pos, self._initial_humanoid_root_states_list[0], self._initial_ball_root_states)
        
        return

    def _compute_reset(self):
        termination_heights = self._termination_heights.unsqueeze(0).repeat(self.num_envs, 1) - torch.mean(self.get_heights(self._humanoid_root_states_list[0]), dim=-1).unsqueeze(1).repeat(1, self._termination_heights.shape[0])
        self.reset_buf[:], self._terminate_buf[:], ball_no_contact, ball_pos_at_stop, ball_target_at_stop = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces_list, self._contact_body_ids,
                                                   self._rigid_body_pos_list, self.max_episode_length,
                                                   self._enable_early_termination, termination_heights ,
                                                   self._ball_paddle_contact_buf, self._ball_pos, self._ball_prev_pos_buf, self._ball_vel, self._ball_targets)
        if flags.test:
            self.ball_pos_at_stop = torch.cat((self.ball_pos_at_stop, ball_pos_at_stop.to(self.ball_pos_at_stop.device)), dim=0)
            self.ball_target_at_stop = torch.cat((self.ball_target_at_stop, ball_target_at_stop.to(self.ball_target_at_stop.device)), dim=0)
            self.ball_no_contact_record += torch.sum(ball_no_contact)
            if self.total_test>5000:
                print("------------------------------------------------------------------------------")
                hit_rate = 1 - self.ball_no_contact_record / self.total_test
                ball_stop_diff = torch.norm(self.ball_target_at_stop - self.ball_pos_at_stop, dim=-1)
                success_rate = torch.sum(ball_stop_diff<1)/ball_stop_diff.shape[0]
                print("success", hit_rate)
                print("ball stop diff percentage is  ", torch.mean(ball_stop_diff))
                print("success rate, diff within 1m is ", success_rate)
                import pdb; pdb.set_trace()

        return

    def _draw_task(self):
        self.gym.clear_lines(self.viewer)

        sphere_geom_marker = gymutil.WireframeSphereGeometry(0.5, 10, 10, None, color=(0.0, 1, 0.0) )
        for env_id in range(1):
            sphere_pose = gymapi.Transform(gymapi.Vec3(self._ball_targets[env_id,  0], self._ball_targets[env_id,  1], self._ball_targets[env_id, 2]), r=None)
            gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        return

    def init_center_height_points(self):
        # center_height_points
        y =  torch.tensor(np.linspace(-0.2, 0.2, 3),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.1, 0.1, 3),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_center_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_center_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_height_points(self):
        # 4mx4m square
        y =  torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent, self.sensor_res),device=self.device,requires_grad=False)
        x = torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent,
                                     self.sensor_res),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_fov_height_points(self):
        y = torch.tensor(np.linspace(-1, 1, 20),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.02, 1.98, 20),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                            device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_root_points(self):
        y = torch.tensor(np.linspace(-0.5, 0.5, 20),
                         device=self.device,
                         requires_grad=False)
        x = torch.tensor(np.linspace(-0.25, 0.25, 10),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_root_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_fov_height_points(self):
        # 3m x 3m fan shaped area
        rs =  np.exp(np.arange(0.2, 2, 0.1)) - 0.9
        rs = rs/rs.max() * 2

        max_angle = 110
        phi = np.exp(np.linspace(0.1, 1.5, 12)) - 1
        phi = phi/phi.max() * max_angle
        phi = np.concatenate([-phi[::-1],[0], phi]) * np.pi/180
        xs, ys = [], []
        for r in rs:
            xs.append(r * np.cos(phi)); ys.append(r * np.sin(phi))

        xs, ys = np.concatenate(xs), np.concatenate(ys)

        xs, ys = torch.from_numpy(xs).to(self.device), torch.from_numpy(ys).to(
            self.device)

        self.num_height_points = xs.shape[0]
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = xs
        points[:, :, 1] = ys
        return points


    def get_heights(self, root_states, env_ids=None):

        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        heading_rot = torch_utils.calc_heading_quat(base_quat)

        if env_ids is None:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        num_envs = self.num_envs if env_ids is None else len(env_ids)
        res =  heights.view(num_envs, -1) * self.terrain.vertical_scale

        return res

    def _create_ground_plane(self):
        self.create_training_ground()

    def create_training_ground(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"],
                               self.cfg["env"]['env_spacing'],
                               num_envs=self.num_envs,
                               device=self.device)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.offset
        tm_params.transform.p.y = -self.terrain.offset
        tm_params.transform.p.z = 0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


class HumanoidGolfZ(HumanoidGolf):

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
@torch.jit.script
def predict_land_point(ball_pos, ball_vel):
    v_z = ball_vel[..., 2]
    h = ball_pos[..., 2]
    discriminant = 4 * v_z**2 + 8 * 9.8 * h

    # Check if the discriminant is non-negative to avoid NaN
    mask = discriminant >= 0
    discriminant = torch.where(mask, discriminant, torch.zeros_like(discriminant))

    t = (2 * v_z + torch.sqrt(discriminant)) / (9.8 * 2)

    ball_land_pos = ball_pos[..., :2] + ball_vel[..., :2] * t.unsqueeze(-1)
    return ball_land_pos

# @torch.jit.script
def compute_golf_ball_observations(root_states, ball_pos, paddle_pos, target_ball_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_ball_pos = quat_rotate(heading_rot, ball_pos - root_pos)

    local_paddle_pos = quat_rotate(heading_rot, paddle_pos - root_pos)

    local_target_ball_pos = quat_rotate(heading_rot, target_ball_pos - root_pos)

    obs = torch.cat([local_ball_pos, local_paddle_pos, local_target_ball_pos],dim=-1)

    return obs


# @torch.jit.script
def compute_ball_reward(root_pos, root_rot, ball_pos, prev_ball_pos, ball_vel, paddle_pos, ball_paddle_contact_buf, 
                        ball_ground_contact_buf, ball_targets, predict_ball_land_pos, _initial_humanoid_root_states, _initial_ball_root_states):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # root_diff = torch.norm(root_pos - _initial_humanoid_root_states[:, 0:3], dim=-1)
    # root_diff_penalty = - torch.clamp(root_diff, min=0, max=0.1)

    # test for root position
    pos_target_diff_normal = F.normalize(ball_targets[..., :2] - _initial_ball_root_states[..., :2], dim=-1)
    pos_root_diff_normal = F.normalize(root_pos[..., :2] - _initial_ball_root_states[..., :2], dim=-1)
    cross_product_z = pos_target_diff_normal[..., 0] * pos_root_diff_normal[..., 1] - pos_target_diff_normal[..., 1] * \
                      pos_root_diff_normal[..., 0]  # from -1 to 1, +1 means robot can hit ball with right hand, which is good
    root_pos_target_reward = torch.clamp(-torch.log(1.0 - cross_product_z), min=0, max=1) * (ball_paddle_contact_buf < 1).float()

    # test for root orientation
    root_rot_project_x = 1 - 2 * root_rot[..., 1].pow(2) - 2 * root_rot[..., 2].pow(2)
    root_rot_project_y = 2 * root_rot[..., 0] * root_rot[..., 1] + 2 * root_rot[..., 3] * root_rot[..., 2]
    root_rot_project_normal = F.normalize(torch.stack([root_rot_project_x, root_rot_project_y], dim=-1), dim=-1)
    dot_product = torch.sum(root_rot_project_normal * pos_root_diff_normal, dim=-1)  # from -1 to 1, -1 means robot facing the ball (good), +1 means the ball is behind robot
    root_rot_reward = torch.clamp(-torch.log(cross_product_z + 1.0), min=0, max=1) * (ball_paddle_contact_buf < 1).float()

    #ready_to_hit = torch.logical_and(cross_product_z>0.9 , dot_product<-0.9).float() * (ball_paddle_contact_buf < 1).float()

    pos_err_scale = 0.1
    ball_paddle_diff = ball_pos - paddle_pos
    ball_paddle_err = torch.sum(ball_paddle_diff * ball_paddle_diff, dim=-1)
    ball_paddle_reward = torch.exp(-100 * ball_paddle_err)
    ball_paddle_reward = (ball_paddle_contact_buf < 1).float() * ball_paddle_reward + (ball_paddle_contact_buf >= 1).float()

    pred_target_pos_diff = predict_ball_land_pos - ball_targets[...,:2]
    pred_target_pos_err = torch.norm(pred_target_pos_diff , dim=-1)
    pred_target_pos_reward = torch.exp(-pos_err_scale * pred_target_pos_err)
    pred_reward = (ball_paddle_contact_buf >= 1).float() * pred_target_pos_reward

    prev_dist = torch.norm(prev_ball_pos - ball_targets, dim=-1)
    curr_dist = torch.norm(ball_pos - ball_targets, dim=-1)
    closer_target_r = torch.clamp(prev_dist - curr_dist, min=0, max=1)
    closer_target_r = (ball_paddle_contact_buf >= 1).float() * closer_target_r
    
    target_pos_diff = ball_pos[..., :2] - ball_targets[...,:2]
    target_pos_err = torch.norm(target_pos_diff, dim=-1)
    target_pos_reward = (ball_paddle_contact_buf >= 1).float() * torch.exp(-pos_err_scale * target_pos_err)

    reward = ball_paddle_reward  + target_pos_reward + closer_target_r + pred_reward # + root_pos_target_reward + root_rot_reward
    #print( "ball_paddle", ball_paddle_reward[0], "predict", pred_reward[0], "target_pos", target_pos_reward[0], "closer_target", closer_target_r[0], "total reward is ", reward[0])
    # if torch.any(torch.isnan(reward)):
    #     print("")
    return reward


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           max_episode_length, enable_early_termination, termination_heights, 
                           ball_paddle_contact_buf, ball_pos, ball_pos_prev, ball_vel, ball_target):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
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

        has_fallen = torch.logical_or(fall_contact, fall_height)
        body_ball_dis = torch.norm(rigid_body_pos_list[0]-ball_pos[:,None,:], dim=-1)
        body_ball_dis[:,23] = 1
        ball_contact_body = torch.any(body_ball_dis<0.15, dim=-1)
        ball_not_closer = torch.logical_and(ball_paddle_contact_buf > 0, torch.norm(ball_target - ball_pos_prev, dim=-1) < torch.norm(ball_target - ball_pos, dim=-1)) 
        ball_stop = torch.logical_and(ball_paddle_contact_buf > 0, torch.norm(ball_vel, dim=-1) < 0.1)

        ball_stop_diff = torch.norm(ball_pos[ball_stop] - ball_target[ball_stop], dim=-1)
        if not torch.equal(ball_stop_diff, torch.tensor([]).to(ball_stop_diff.device)):
            print("ball stop diff length ", torch.mean(ball_stop_diff), "diff/target percentage is ", torch.mean(ball_stop_diff/torch.norm(ball_target[ball_stop], dim=-1)))

        ball_no_contact1 = torch.logical_and(ball_paddle_contact_buf == 0, progress_buf > 60)
        ball_no_contact2 = torch.logical_and(ball_paddle_contact_buf==0, ball_pos[:, 0] < -0.05)
        ball_no_contact = ball_no_contact1 | ball_no_contact2
        ball_fail = torch.logical_or(ball_not_closer, ball_stop)
        ball_fail = torch.logical_or(ball_no_contact, ball_fail)
        ball_fail = torch.logical_or(ball_contact_body, ball_fail)
            
        has_failed = torch.logical_or(has_fallen, ball_fail)
        if has_failed.shape[0]==1 and has_failed[0]:
            print("reset because ", "fall_contact", fall_contact, "fall_height", fall_height, "ball_not_closer", ball_not_closer, "ball_no_contact", ball_no_contact, "ball_contact_body", ball_contact_body, "ball stop", ball_stop)
        has_failed *= (progress_buf > 1)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated,ball_no_contact, ball_pos[ball_stop], ball_target[ball_stop]
