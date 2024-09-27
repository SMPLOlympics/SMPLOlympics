import torch
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import torch.nn.functional as F
import math
import pdb
import time
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags

class HumanoidTennis(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        # Tennis court dimension
        self._doubleline = 10.97 / 2
        self._singleline = 8.23 / 2
        self._baseline = 23.77 / 2
        self._serviceline = 6.4

        # Ball launch parameters
        self._origin_min = torch.FloatTensor(cfg['env'].get('origin_min', [-12.5, -3, 1.2]))
        self._origin_max = torch.FloatTensor(cfg['env'].get('origin_max', [-12.5, 3, 1.2]))
        self._bounce_min = torch.FloatTensor(cfg['env'].get('bounce_min', [12, -3, 0]))
        self._bounce_max = torch.FloatTensor(cfg['env'].get('bounce_max', [12, 3, 0]))
        self._target_min = torch.FloatTensor(cfg['env'].get('target_min', [-11, -3, 0]))
        self._target_max = torch.FloatTensor(cfg['env'].get('target_max', [-8, 3, 0]))
        self._vel_range = torch.FloatTensor(cfg['env'].get('vel_range', [18, 22]))
        self._vspin_range = torch.FloatTensor(cfg['env'].get('vspin_range', [20, 35]))
        self._theta_range = torch.FloatTensor(cfg['env'].get('theta_range', [9, 10]))

        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        self.set_initial_root_state()
        self.set_racket_restitution()
        self._build_predefined_tensors()
        self._build_ball_state_tensors()
        if not self.headless:
            self._build_target_state_tensors()
            
        self.statistics = False
        if flags.test:
            self.statistics = False
            
        if self.statistics:
            self.build_statistics_tensor()
        
        return
        
    def build_statistics_tensor(self):
        self.avg_hits = []
        self.avg_ball_vel = []
        self.land_dis = []

        self.body_ang_vel = []
        self.body_vel = []

        self.hits_buf = torch.zeros(self.num_envs,
                                    device=self.device,
                                    dtype=torch.float)
        self.prev_predict_ball_land_pos = None
        
    def _build_predefined_tensors(self):
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._forces = torch.zeros((self.num_envs, bodies_per_env, 3), device=self.device, dtype=torch.float)
        self._torques = torch.zeros((self.num_envs, bodies_per_env, 3), device=self.device, dtype=torch.float)

        self._ball_actor_id = self.num_agents
        self._net_actor_id = self.num_agents + 1
        self._target_actor_id = self.num_agents + 2
        self._righthand_body_id = 23
        self._ball_body_id = 24
    
        self._racket_to_hand = torch.tensor([0, -0.35, 0.], device=self.device, dtype=torch.float).repeat(self.num_envs,1)
        self._ball_mycourt_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_othercourt_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_racket_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_land_in_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._ball_targets = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self._tar_time_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._tar_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_land_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self._ball_prev_vel_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.int)
        self._target_root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        self.z_dir = torch.FloatTensor([0, 0, -1]).to(self.device)
        # self.ball_mycourt_contact_time_buf = torch.zeros(
        #     self.num_envs,2, device=self.device, dtype=torch.long)
        # self.ball_othercourt_contact_time_buf = torch.zeros(
        #     self.num_envs,2, device=self.device, dtype=torch.long)
    
    def set_initial_root_state(self):
        for i in range(self.num_agents):
            initial_humanoid_root_states = self._initial_humanoid_root_states_list[i].clone()
            initial_humanoid_root_states[:, 7:13] = 0

            initial_humanoid_root_states[..., 0] = 13
            initial_humanoid_root_states[..., 1] = 0
            initial_humanoid_root_states[..., 2] = 1 
            
            initial_humanoid_root_states[..., 3] = 0
            initial_humanoid_root_states[..., 4] = 0
            initial_humanoid_root_states[..., 5] = 1
            initial_humanoid_root_states[..., 6] = 0

            self._initial_humanoid_root_states_list[i] = initial_humanoid_root_states
    
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
        self._ball_pos = self._ball_root_states[..., :3]
        self._ball_vel = self._ball_root_states[..., 7:10]
        self._ball_ang_vel = self._ball_root_states[..., 10:13]
        
        self._ball_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self._ball_actor_id

        return

    def _build_target_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self._target_actor_id, :]
        self._target_pos = self._target_states[..., :3]
        self._target_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self._target_actor_id

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._ball_handles = []
        self._net_handles = []
        
        self._load_ball_asset()
        self._load_net_asset()

        if (not self.headless):
            self._target_handles = []
            self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_net_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "tennis.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._net_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return

    def _load_ball_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "tennis_ball.urdf"

        asset_options = gymapi.AssetOptions()
        # asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.01
        # asset_options.max_angular_velocity = 500
        # asset_options.density = 2000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        prp = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
        for each in prp:
            each.friction = 0.8
            each.restitution = 0.8
            # each.contact_offset = 0.02
            # each.rolling_friction = 0.1

        self.gym.set_asset_rigid_shape_properties(self._ball_asset,prp)

        return 
            
    def _load_target_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        super()._build_env(env_id, env_ptr, humanoid_asset_list)
        
        self._build_ball(env_id, env_ptr) # id: num_agents
        self._build_net(env_id, env_ptr) # id: num_agents+1

        if (not self.headless):
            self._build_target(env_id, env_ptr) # id: num_agents+2
            
        return
    
    def _build_net(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 1
        default_pose = gymapi.Transform()
        
        net_handle = self.gym.create_actor(env_ptr, self._net_asset, default_pose, "net", col_group, col_filter, segmentation_id)
        rsp = gymapi.RigidShapeProperties()
        rsp.restitution = 0.7
        rsp.thickness = 0.2
        rsp.contact_offset = 0.02
        rsp.friction = 0.8
        rsp.rolling_friction = 0.1

        self.gym.set_actor_rigid_shape_properties(env_ptr, net_handle, [rsp,rsp])

        self.gym.set_rigid_body_color(env_ptr, net_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.25, 0.41, 0.88))
        self._net_handles.append(net_handle)

        return

    def _build_ball(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 3
        segmentation_id = 1
        default_pose = gymapi.Transform()
        
        ball_handle = self.gym.create_actor(env_ptr, self._ball_asset, default_pose, "tennis_ball", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 1.0, 0.0))
        self._ball_handles.append(ball_handle)

        return
    
    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 4
        segmentation_id = 2
        default_pose = gymapi.Transform()
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._target_handles.append(target_handle)

        return

    def _reset_envs(self, env_ids):
        self._reset_env_ball(env_ids)
        super()._reset_envs(env_ids)

        if self.statistics:
            if len(env_ids) > 0:
                self.avg_hits.append(self.hits_buf[env_ids])
                self.hits_buf[env_ids] = 0
                
        return

    def get_ball_initial_speed(self, ball_pos, n, y_target_left, y_target_right, vx_min=10, vx_max=15, x_min=2):

        x0 = ball_pos[:,0]
        y0 = ball_pos[:,1]
        z0 = ball_pos[:,2]

        vx = torch.Tensor(n).uniform_(vx_min, vx_max).to(self.device, torch.float) 

        t_net = abs((x0) / vx) # abs((0 - x0) / vx)  # Time to reach the net

        vz_min = (1.1 - z0 + 0.5 * 9.81 * t_net**2) / t_net

        a = -0.5 * 9.81
        b = vz_min
        c = z0 
        t2 = (-b - torch.sqrt(b**2 - 4*a*c)) / (2*a)

        x_position_max = x0 + vx * t2 
        
        x_position_max = torch.clamp(x_position_max, min=x_min)
        
        x_target = torch.rand(n,device=self.device) * ((x_position_max) - (self._baseline)) + (self._baseline)
        y_target = torch.Tensor(n).uniform_(y_target_left, y_target_right).to(self.device, torch.float) 
        
        t_land = (x_target - x0) / vx
        vz = (0  - z0 + 0.5 * 9.81 * t_land**2) / t_land
        vy = (y_target - y0) / t_land

        return vx, vy, vz
    
    def _reset_env_ball(self, env_ids):
        
        def torch_sample_range(size, min, max):
            return torch.rand(size) * (max - min) + min
        
        n = len(env_ids)
        if n>0:
            self._ball_prev_vel_buf[env_ids] = 0
            self._ball_mycourt_contact_buf[env_ids] = 0
            self._ball_othercourt_contact_buf[env_ids] = 0
            self._ball_racket_contact_buf[env_ids] = 0
            self._ball_land_in_buf[env_ids] = 0
            
            # Launch new ball
            origin = torch_sample_range((len(env_ids), 3), self._origin_min, self._origin_max).to(self.device)
            # bounce = torch_sample_range((len(env_ids), 3), self._bounce_min, self._bounce_max)
            # dir = F.normalize(bounce[:, :2] - origin[:, :2], dim=1)
            # import pdb;pdb.set_trace()
            # launch_vel_scalar = torch_sample_range((len(env_ids)), self._vel_range[0], self._vel_range[1])
            # launch_theta = torch_sample_range((len(env_ids)), self._theta_range[0], self._theta_range[1])
            launch_vspin = torch_sample_range((len(env_ids)), self._vspin_range[0], self._vspin_range[1]).to(self.device)

            # launch_vel = torch.stack([
            #     launch_vel_scalar * torch.cos(launch_theta / 180 * np.pi) * dir[:, 0],
            #     launch_vel_scalar * torch.cos(launch_theta / 180 * np.pi) * dir[:, 1],
            #     launch_vel_scalar * torch.sin(launch_theta / 180 * np.pi),
            # ]).T

            launch_vel = torch.stack(self.get_ball_initial_speed(origin, n, -3, 3, 12, 18, 5),dim=-1)
            
            launch_ang_vel = launch_vspin.view(-1, 1) * math.pi * 2 * F.normalize(
                torch.cross(launch_vel, self.z_dir.repeat(len(env_ids), 1)), dim=1)
            
            self._ball_root_states[env_ids, 0:3] = origin
            self._ball_root_states[env_ids, 7:10] = launch_vel
            self._ball_root_states[env_ids, 10:13] = 0#launch_ang_vel

            # Set ball target
            self._ball_targets[env_ids] = torch_sample_range((len(env_ids), 3), self._target_min, self._target_max).to(self.device)
            self._tar_time_total[env_ids] = 70
            self._tar_time[env_ids] = 0

            # Set player target
            ball_side_pos_y = ((torch.abs(self._ball_pos[env_ids, 0]) + self._baseline) / torch.abs(self._ball_vel[env_ids, 0])) * self._ball_vel[env_ids, 1] + self._ball_pos[env_ids, 1]
            # Predict forehand/backhand based on left/right side of the court
            self._target_root_pos[env_ids, 1] =  torch.where(
                self._humanoid_root_states_list[0][env_ids, 1] < ball_side_pos_y,
                ball_side_pos_y.clone() - 1, # forehand
                ball_side_pos_y.clone() + 1, # backhand
            )
            
            # print(ball_side_pos_y, self._target_root_pos[0,1])
            self._target_root_pos[env_ids, 0] = self._baseline
            self._target_root_pos[env_ids, 2] = 0.75

            # Update gym tensors
            if not self.headless:
                self._target_pos[env_ids,:3] = self._ball_targets[env_ids, :3].clone()
                env_ids_int32 = torch.cat([self._ball_actor_ids[env_ids], self._target_actor_ids[env_ids]])
            else:
                env_ids_int32 = torch.cat([self._ball_actor_ids[env_ids]])

            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15
        return obs_size
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        return

    def apply_external_force_to_ball(self, ball_root_states):
        R       = 0.032 # m, radius of ball
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
        self._forces[:, self._ball_body_id, :] = ball_forces 

    def physics_step(self):
        for i in range(self.control_freq_inv):
            self.apply_external_force_to_ball(self._ball_root_states)

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self._forces), 
                gymtorch.unwrap_tensor(self._torques), gymapi.ENV_SPACE)

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            if not self._is_train:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            self._forces[:] = 0
            self._torques[:] = 0

            self.render()
    
    def post_physics_step(self):
        
        super().post_physics_step()
        self._update_ball()

        return

    def _update_ball(self):
        # reset_task_mask = torch.logical_and(torch.logical_or(self._ball_pos[..., 2] < 0.7,self._ball_pos[..., 0] > 1.4), self._ball_othercourt_contact_buf==1) + (self._ball_othercourt_contact_buf>=2) 

        self._tar_time += 1
        # For single player, we will launch a new ball after certain time
        # reset_task_mask = self._tar_time == self._tar_time_total 
        
        reset_task_mask = torch.logical_and(self._ball_pos[..., 0] < (-self._baseline-1), self._ball_othercourt_contact_buf==1) + (self._ball_othercourt_contact_buf>=2) 
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
            
        if len(rest_env_ids) > 0:
            self._reset_env_ball(rest_env_ids)
            self._refresh_sim_tensors()
        return
      
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []

        if (env_ids is None):
            ball_pos =  self._ball_pos.clone()
            ball_vel = self._ball_vel.clone()
            target_root_pos = self._target_root_pos.clone()
            target_ball_pos = self._ball_targets.clone()

        else:
            ball_pos = self._ball_pos[env_ids].clone()
            ball_vel = self._ball_vel[env_ids].clone()
            target_root_pos = self._target_root_pos[env_ids].clone()
            target_ball_pos = self._ball_targets[env_ids].clone()

        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
                righthand_pos = self._rigid_body_pos_list[i][:, self._righthand_body_id, :].clone()
                righthand_rot = self._rigid_body_rot_list[i][:, self._righthand_body_id, :].clone()
                racket_pos = righthand_pos + quat_rotate(righthand_rot, self._racket_to_hand.clone())

            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                righthand_pos = self._rigid_body_pos_list[i][env_ids, self._righthand_body_id, :].clone()
                righthand_rot = self._rigid_body_rot_list[i][env_ids, self._righthand_body_id, :].clone()
                racket_pos = righthand_pos + quat_rotate(righthand_rot, self._racket_to_hand[env_ids].clone())

            obs = compute_tennis_ball_observations(root_states, ball_pos, ball_vel, target_root_pos, racket_pos, target_ball_pos)
            obs_list.append(obs)
        
        return obs_list
    
    
    def _compute_statistics(
        self,
        ball_pos,
        ball_vel,
        ball_targets,
        ball_table_contact,
        predict_ball_land_pos,
    ):
        
        ball_table_id = ball_table_contact == 1
        self.hits_buf[ball_table_id] += 1
        if torch.sum(ball_table_id) > 0:
            
            self.land_dis.append(
                torch.norm(
                    predict_ball_land_pos[ball_table_id, :2] -
                    ball_targets[ball_table_id, :2],
                    dim=-1,
                ))
            
            if len(self.avg_hits) > 2:

                print(
                    "len",
                    len(torch.cat(self.avg_hits[2:])),
                    "hits",
                    torch.mean(torch.cat(
                        self.avg_hits[2:])).item(),
                    "land dis",
                    torch.mean(torch.cat(self.land_dis)).item(),
                    
                    #   'body vel', torch.mean(torch.cat(self.body_vel)).item(),
                    #   'body ang vel', torch.mean(torch.cat(self.body_ang_vel)).item(),
                    #    '0len', torch.sum(torch.cat(self.avg_hits[self.num_envs:])==0).item()
                )
                

    def _compute_reward(self, actions):

        ball_pos = self._ball_pos[..., :3].clone()
        ball_vel = self._ball_vel[..., :3].clone()

        height_threshold = (ball_pos[..., 2] < 0.4)
        # ball_vel_z_change =  (self._ball_prev_vel_buf[...,2] * ball_vel[...,2]) < 0
        # ball_vel_x_change = (self._ball_prev_vel_buf[...,0] * ball_vel[...,0]) < 0
        
        ball_vel_z_change = torch.logical_and(self._ball_prev_vel_buf[..., 2]<0,  ball_vel[..., 2]>0)
        ball_vel_x_change = (self._ball_prev_vel_buf[...,0] * ball_vel[...,0])<0


        self._ball_prev_vel_buf = ball_vel.clone()

        court_x_y_threshold = torch.logical_and(torch.abs(ball_pos[..., 0]) < self._baseline, torch.abs(ball_pos[..., 1]) < self._singleline)
        
        contact_ground = torch.logical_and(height_threshold, ball_vel_z_change)
        contact_court = torch.logical_and(contact_ground, court_x_y_threshold)

        check_mycourt_contact = torch.logical_and(ball_pos[:,0] > 0, contact_ground)
        check_othercourt_contact = torch.logical_and(ball_pos[:,0] < 0, contact_court)
        
        self._ball_mycourt_contact_buf += check_mycourt_contact
        self._ball_othercourt_contact_buf += check_othercourt_contact
        # print(self._ball_othercourt_contact_buf)
        # check_mycourt_contact_id=check_mycourt_contact>0
        # check_othercourt_contact_id = check_othercourt_contact>0

        # self.ball_mycourt_contact_time_buf[check_mycourt_contact_id,0]= (self.ball_mycourt_contact_time_buf[check_mycourt_contact_id,1]).clone()
        # self.ball_othercourt_contact_time_buf[check_othercourt_contact_id,0]=(self.ball_othercourt_contact_time_buf[check_othercourt_contact_id,1]).clone()

        # self.ball_mycourt_contact_time_buf[check_mycourt_contact_id,1]= (self.progress_buf[check_mycourt_contact_id]).clone()
        # self.ball_othercourt_contact_time_buf[check_othercourt_contact_id,1] = (self.progress_buf[check_othercourt_contact_id]).clone()

        predict_ball_land_pos = predict_land_point(ball_pos, ball_vel[..., :3])
        
        # print(predict_ball_land_pos, ball_pos[0,:3])
        # time.sleep(0.5)

        ball_land_in_now = torch.logical_and(~self._ball_land_in_buf, torch.logical_and(check_othercourt_contact, self._ball_mycourt_contact_buf == 1))
        self._ball_land_in_buf += ball_land_in_now
        self._ball_land_pos[ball_land_in_now] = self._ball_pos[ball_land_in_now]
        
        
        if self.statistics:
            # print(check_othertable_contact)
            if self.prev_predict_ball_land_pos is not None:
                self._compute_statistics(
                    self._ball_pos,
                    self._ball_vel,
                    self._ball_targets,
                    torch.logical_and(check_othercourt_contact, self._ball_othercourt_contact_buf==1),
                    self.prev_predict_ball_land_pos,
                )
            self.prev_predict_ball_land_pos = predict_ball_land_pos.clone()
                
                
        

        for i in range(self.num_agents):
            root_states = self._humanoid_root_states_list[i].clone()
            root_pos = root_states[:, 0:3]
            root_rot = root_states[:, 3:7]

            righthand_pos = self._rigid_body_pos_list[i][:, self._righthand_body_id, :].clone()
            righthand_rot = self._rigid_body_rot_list[i][:, self._righthand_body_id, :].clone()
            racket_pos = righthand_pos + quat_rotate(righthand_rot, self._racket_to_hand.clone())

            ball_racket_contact_check = torch.logical_and(torch.norm(ball_pos - racket_pos, dim=-1) < 0.2, ball_vel_x_change)
            self._ball_racket_contact_buf += ball_racket_contact_check

            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_ball_reward(root_pos, root_rot,
                                                ball_pos, ball_vel, racket_pos, self._ball_racket_contact_buf, self._ball_land_in_buf, 
                                                self._ball_targets, self._target_root_pos, predict_ball_land_pos, self._ball_land_pos)

        
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces_list, self._contact_body_ids,
                                                   self._rigid_body_pos_list, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._ball_mycourt_contact_buf, self._ball_othercourt_contact_buf, 
                                                   self._ball_pos, self.num_agents, self._baseline, self._doubleline)

        return

    def _draw_task(self):
        self.gym.clear_lines(self.viewer)
        for i, env_ptr in enumerate(self.envs):
            vertices = np.array([
                [-self._baseline, -self._doubleline, 0],
                [-self._baseline, -self._singleline, 0],
                [-self._baseline, self._singleline, 0],
                [-self._baseline, self._doubleline, 0],
                [-self._serviceline, -self._singleline, 0],
                [-self._serviceline, 0, 0],
                [-self._serviceline, self._singleline, 0],

                [0, -self._doubleline, 0],
                [0, self._doubleline, 0],
                
                [self._baseline, -self._doubleline, 0],
                [self._baseline, -self._singleline, 0],
                [self._baseline, self._singleline, 0],
                [self._baseline, self._doubleline, 0],
                [self._serviceline, -self._singleline, 0],
                [self._serviceline, 0, 0],
                [self._serviceline, self._singleline, 0],
            ])

            lines = np.array([
                # Horizontal lines
                [vertices[0], vertices[3]],
                [vertices[4], vertices[6]],
                [vertices[7], vertices[8]],
                [vertices[9], vertices[12]],
                [vertices[13], vertices[15]],
                # Vertical lines
                [vertices[0], vertices[9]],
                [vertices[1], vertices[10]],
                [vertices[5], vertices[14]],
                [vertices[2], vertices[11]],
                [vertices[3], vertices[12]],
            ]).astype(np.float32)

            cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            for line in lines:
                self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)
            
        return


class HumanoidTennisZ(HumanoidTennis):

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
    # TODO: have to include air drag and magnus force
    v_z = ball_vel[...,2]
    h = ball_pos[..., 2]
    t = (2 * v_z + torch.sqrt(4*v_z**2+8*9.8*h))/(9.8*2)

    ball_land_pos = ball_pos[...,:2] + ball_vel[...,:2] * t[:,None]
    ball_land_pos = torch.clamp(ball_land_pos,min=-15,max=15)
    return ball_land_pos


# @torch.jit.script
def compute_tennis_ball_observations(root_states, ball_pos, ball_vel, target_root_pos, left_racket_pos, target_ball_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_ball_pos = quat_rotate(heading_rot, ball_pos - root_pos)

    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    local_target_root_pos = quat_rotate(heading_rot, target_root_pos - root_pos)

    local_racket_pos = quat_rotate(heading_rot, left_racket_pos - root_pos)

    local_target_ball_pos = quat_rotate(heading_rot, target_ball_pos - root_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_target_root_pos, local_racket_pos, local_target_ball_pos],dim=-1)

    return obs


# @torch.jit.script
def compute_ball_reward(root_pos, root_rot, ball_pos, ball_vel, racket_pos, ball_racket_contact_buf, ball_land_in_buf, ball_targets, root_targets, predict_ball_land_pos, ball_land_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    pos_err_scale = 1.0
    # print(root_pos[0,:3],root_targets[0,:3])
    ball_pos_diff = ball_pos - racket_pos
    ball_pos_err = torch.sum(ball_pos_diff * ball_pos_diff, dim=-1)
    ball_pos_reward = torch.exp(-pos_err_scale * ball_pos_err)
    
    # 
    player_pos_diff = root_targets[:, :3] - root_pos[:, :3]
    player_pos_err = torch.sum(player_pos_diff * player_pos_diff, dim=-1)
    player_pos_reward = torch.exp(-2 * player_pos_err)


    pred_target_pos_diff = predict_ball_land_pos - ball_targets[...,:2]
    pred_target_pos_err = torch.sum(pred_target_pos_diff * pred_target_pos_diff, dim=-1)
    pred_target_pos_reward = torch.exp(-pos_err_scale * pred_target_pos_err)

    target_pos_diff = ball_land_pos[..., :2] - ball_targets[...,:2]
    target_pos_err = torch.sum(target_pos_diff * target_pos_diff, dim=-1)
    target_pos_reward = torch.exp(-pos_err_scale * target_pos_err)

   
    reward = (ball_racket_contact_buf < 1) * ball_pos_reward * player_pos_reward + \
             ((ball_racket_contact_buf >= 1).float() * (ball_vel[:,0]<0) + \
             (ball_racket_contact_buf >= 1) * (ball_land_in_buf == 0) * pred_target_pos_reward + \
             (ball_land_in_buf >= 1) * (1 + target_pos_reward + (target_pos_err < 0.1))) * (ball_racket_contact_buf <= 2) * (player_pos_reward + 0.1) * (ball_pos[:,2]<3)
    return reward


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           max_episode_length, enable_early_termination, termination_heights, ball_mycourt_contact_buf, 
                           ball_othercourt_contact_buf, ball_pos, num_agents, baseline, doubleline):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
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

                body_ball_dis = torch.norm(rigid_body_pos_list[0]-ball_pos[:,None,:], dim=-1)
                body_ball_dis[:,23] = 1
                ball_contact_body = torch.logical_or(ball_contact_body, body_ball_dis<0.15, dim=-1)
        

        ball_fail = torch.logical_and(torch.logical_or(ball_pos[...,0]< (-baseline-1), torch.abs(ball_pos[...,1])>doubleline), ball_othercourt_contact_buf==0)
        
        ball_fail = torch.logical_or(ball_fail, ball_mycourt_contact_buf>=2)
        ball_fail = torch.logical_or(ball_fail, ball_pos[...,0] > (baseline+4))
        ball_fail = torch.logical_or(ball_fail, ball_pos[...,2] > 6)
        ball_fail = torch.logical_or(ball_contact_body, ball_fail)
        
        
            
            
        has_failed = torch.logical_or(has_fallen, ball_fail)


        has_failed *= (progress_buf > 1)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)


    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    # if reset[0]:
    #     import pdb;pdb.set_trace()
    return reset, terminated
