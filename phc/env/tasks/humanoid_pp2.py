
import os
import torch
import time
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import multiprocessing

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags
from enum import Enum

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
import torch.multiprocessing as mp


TAR_ACTOR_ID = 1



class HumanoidPP2(humanoid_amp_task.HumanoidAMPTask):
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

        # strike_body_names = cfg["env"]["strikeBodyNames"]
        
        # self._build_target_tensors()
        self.compete_reward = False
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        
        self.set_initial_root_state()
        self.set_paddle_restitution()
        self._build_predefined_tensors()
        self._build_ball_state_tensors()
        # if not self.headless:
        #     self._build_target_state_tensors()
        
        self.compete_reward = cfg["env"].get("compete_reward", False)
        return
    
    def _build_predefined_tensors(self):

        self._ball_init_pos = torch.tensor([0, 0, 1.4], device=self.device, dtype=torch.float)

        self._paddle_to_hand = torch.tensor([0, -0.12, 0.], device=self.device, dtype=torch.float).repeat(self.num_envs,1)

        self._ball_mytable_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_othertable_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._ball_paddle_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_table_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        #op
        self._ball_paddle_contact_buf_op = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._ball_table_reward_buf_op = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._ball_targets = torch.tensor([-1, 0, 0.8], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self._ball_targets_op = torch.tensor([1, 0, 0.8], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)

        self._ball_targets[:,0] = torch.Tensor(self.num_envs).uniform_(-1.2, -0.1)
        self._ball_targets[:,1] = torch.Tensor(self.num_envs).uniform_(-0.65, 0.65)

        self._ball_targets_op[:,0] = torch.Tensor(self.num_envs).uniform_(0.1, 1.2)
        self._ball_targets_op[:,1] = torch.Tensor(self.num_envs).uniform_(-0.65, 0.65)

        
        self._ball_prev_vel_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.int)

        self._target_root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        self._target_root_pos[..., 0] = 2
        self._target_root_pos[..., 2] = 0.75

        self._target_root_pos_op = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        self._target_root_pos_op[..., 0] = -2
        self._target_root_pos_op[..., 2] = 0.75

        self._target_root_transl_prob = torch.zeros(self.num_envs, device=self.device, dtype=torch.float) + 0.5

        self.target_root_transl = torch.bernoulli(self._target_root_transl_prob) * 0.6

        self.target_root_transl_op = torch.bernoulli(self._target_root_transl_prob) * 0.6
        if self.num_agents==1:
            self._launch_prob = torch.zeros(self.num_envs, device=self.device, dtype=torch.float) + 1
        else:
            self._launch_prob = torch.zeros(self.num_envs, device=self.device, dtype=torch.float) + 0.5
        self.launch_dir = torch.bernoulli(self._launch_prob)

        self.ball_mytable_contact_time_buf = torch.zeros(
            self.num_envs,2, device=self.device, dtype=torch.long)
        
        self.ball_othertable_contact_time_buf = torch.zeros(
            self.num_envs,2, device=self.device, dtype=torch.long)

        self.game_finish_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        
    
    def set_initial_root_state(self):
        for i in range(self.num_agents):
            initial_humanoid_root_states = self._initial_humanoid_root_states_list[i].clone()
            initial_humanoid_root_states[:, 7:13] = 0

            initial_humanoid_root_states[..., 0] = -i*4+2
            initial_humanoid_root_states[..., 1] = 0
            initial_humanoid_root_states[..., 2] = 1 
            
            initial_humanoid_root_states[..., 3] = 0
            initial_humanoid_root_states[..., 4] = 0
            initial_humanoid_root_states[..., 5] = 1
            initial_humanoid_root_states[..., 6] = 0

            if i==1:
                initial_humanoid_root_states[..., 3] = 0
                initial_humanoid_root_states[..., 4] = 0
                initial_humanoid_root_states[..., 5] = 0
                initial_humanoid_root_states[..., 6] = 1


            self._initial_humanoid_root_states_list[i] = initial_humanoid_root_states
    
    def set_paddle_restitution(self):
        # self.humanoid_handles_list
        
        for env_id in range(self.num_envs): 
            env_ptr = self.envs[env_id]
            
            for i in range(self.num_agents):
                
                rsp = self.gym.get_actor_rigid_shape_properties(env_ptr, self.humanoid_handles_list[env_id][i])
                rsp[23].restitution = 0.2 #right hand
                self.gym.set_actor_rigid_shape_properties(env_ptr, self.humanoid_handles_list[env_id][i],rsp)

            

    
    def _load_table_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "table.urdf"
        

        asset_options = gymapi.AssetOptions()
        asset_options.density = 100.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._table_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return

    def _load_ball_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "pingpong.urdf"

        asset_options = gymapi.AssetOptions()

        asset_options.max_angular_velocity = 30.0

        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        prp = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
        for each in prp:
            each.restitution = 0.8
            # each.thickness = 0.001
            each.contact_offset = 0.02
            each.friction = 1.
            each.rest_offset = 0.01

        self.gym.set_asset_rigid_shape_properties(self._ball_asset,prp)

        return
            
    def _load_target_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "target.urdf"

        asset_options = gymapi.AssetOptions()

        # asset_options.max_angular_velocity = 30.0

        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        return
    
    
    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        super()._build_env(env_id, env_ptr, humanoid_asset_list)
        
        
        self._build_table(env_id, env_ptr)
        self._build_ball(env_id, env_ptr)
        # if (not self.headless):
            # self._build_target(env_id, env_ptr)
            
        return
    
    def _build_table(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 1
        default_pose = gymapi.Transform()
        
        table_handle = self.gym.create_actor(env_ptr, self._table_asset, default_pose, "table", col_group, col_filter, segmentation_id)
        
        rsp = gymapi.RigidShapeProperties()
        rsp.restitution = 0.7
        rsp.thickness = 0.2
        rsp.contact_offset = 0.02
        rsp.friction = 0.8
        rsp.rolling_friction = 0.1

        self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, [rsp,rsp])

        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.25, 0.41, 0.88))
        self._table_handles.append(table_handle)

        return

    def _build_ball(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 3
        segmentation_id = 1
        default_pose = gymapi.Transform()

        

        ball_handle = self.gym.create_actor(env_ptr, self._ball_asset, default_pose, "pingpong_ball", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.5, 0.))
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

    def _build_ball_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._ball_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self.num_agents+1, :]
        self._ball_pos = self._ball_states[..., :3]
        self._ball_vel = self._ball_states[..., 7:10]
        self._ball_ang_vel = self._ball_states[..., 10:13]
        
        self._ball_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._ball_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., -2:, :]

        return
    
    

    def _build_target_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., self.num_agents+2, :]
        # self._target_pos = self._target_states[..., :3]
        self._target_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents + 2

        return

    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15
        return obs_size
    
    def post_physics_step(self):        
        super().post_physics_step()
        
        if self.compete_reward:
            terminate_env_ids = (
                self._terminate_buf * (self.game_finish_buf < 1)).nonzero(as_tuple=False).flatten()

            if len(terminate_env_ids > 0):

                me_reward, op_reward = compute_compete_rewards(
                    self.ball_mytable_contact_time_buf[terminate_env_ids, 1], self.ball_othertable_contact_time_buf[terminate_env_ids, 1])
                self.rew_buf[terminate_env_ids] += me_reward
                self.rew_buf[self.num_envs+terminate_env_ids] += op_reward

            self.game_finish_buf[terminate_env_ids] = self.progress_buf[terminate_env_ids]
            
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        
        self._table_handles = []
        self._ball_handles = []
        
        self._load_table_asset()
        self._load_ball_asset()

        if (not self.headless):
            self._target_handles = []
            self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _update_ball(self, ball_paddle_contact_check, ball_paddle_contact_check_op):
        
        ball_vel = self._ball_vel.clone()
        ball_pos = self._ball_pos.clone()
        reset_agent1_ids = ball_paddle_contact_check_op.nonzero(as_tuple=False).flatten()
        reset_agent2_ids = ball_paddle_contact_check.nonzero(as_tuple=False).flatten()

        n1 = len(reset_agent1_ids)
        n2 = len(reset_agent2_ids)
        if n1 > 0:
            self._ball_paddle_contact_buf[reset_agent1_ids] = 0
            self._ball_table_reward_buf[reset_agent1_ids] = 0

            x_distance = 1.37 - ball_pos[reset_agent1_ids, 0 ]
            t = x_distance/ball_vel[reset_agent1_ids, 0]
            ball_side_pos_y = ball_pos[reset_agent1_ids, 1] + t * ball_vel[reset_agent1_ids, 1]
            
            target_root_transl = ((ball_side_pos_y - self._humanoid_root_states_list[0][reset_agent1_ids, 1]) > 0.) * 0.6
            
            self.target_root_transl[reset_agent1_ids] = target_root_transl.clone()
            self._target_root_pos[reset_agent1_ids, 1] =  ball_side_pos_y.clone() - target_root_transl

            
            self._ball_targets[reset_agent1_ids, 0] = torch.Tensor(n1).uniform_(-1.37, -0.1).to(self.device, torch.float) 
            self._ball_targets[reset_agent1_ids, 1] = torch.Tensor(n1).uniform_(-0.75, 0.75).to(self.device, torch.float) 

            
            # if not self.headless:
            #     self._target_pos[reset_agent1_ids,:3] = self._ball_targets[reset_agent1_ids, :3].clone()
            #     env_ids_int32 = env_ids_int32 = self._target_actor_ids[reset_agent1_ids]
            #     self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
            #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        if n2 > 0:
            self._ball_paddle_contact_buf_op[reset_agent2_ids] = 0
            self._ball_table_reward_buf_op[reset_agent2_ids] = 0

            x_distance = 1.37 + ball_pos[reset_agent2_ids, 0 ]
            t = -x_distance/ball_vel[reset_agent2_ids, 0]
            ball_side_pos_y = ball_pos[reset_agent2_ids, 1] + t * ball_vel[reset_agent2_ids, 1]

            target_root_transl = ((ball_side_pos_y - self._humanoid_root_states_list[1][reset_agent2_ids, 1]) < 0.) * 0.6

            self.target_root_transl_op[reset_agent2_ids] = target_root_transl.clone()
            self._target_root_pos_op[reset_agent2_ids, 1] =  ball_side_pos_y.clone() + target_root_transl

            self._ball_targets_op[reset_agent2_ids, 0] = torch.Tensor(n2).uniform_(0.1, 1.37).to(self.device, torch.float) 
            
            self._ball_targets_op[reset_agent2_ids, 1] = torch.Tensor(n2).uniform_(-0.75, 0.75).to(self.device, torch.float)

        return


    def _reset_env_ball(self, env_ids):

        n = len(env_ids)
        if n>0:
            
            self.game_finish_buf[env_ids] = 0
                   
            self._ball_prev_vel_buf[env_ids] = 0
            
            self._ball_mytable_contact_buf[env_ids] = 0

            self._ball_othertable_contact_buf[env_ids] = 0
            
            self.ball_mytable_contact_time_buf[env_ids, :] = 0

            self.ball_othertable_contact_time_buf[env_ids, :] = 0

            self._ball_paddle_contact_buf[env_ids] = 0

            self._ball_table_reward_buf[env_ids] = 0

            self._ball_paddle_contact_buf_op[env_ids] = 0

            self._ball_table_reward_buf_op[env_ids] = 0
            
            self._ball_pos[env_ids, :3] = self._ball_init_pos.clone()

            self._ball_targets[env_ids, 0] = torch.Tensor(n).uniform_(-1.2, -0.3).to(self.device, torch.float) 
            self._ball_targets[env_ids, 1] = torch.Tensor(n).uniform_(-0.6, 0.6).to(self.device, torch.float)

            self._ball_targets_op[env_ids, 0] = torch.Tensor(n).uniform_(0.3, 1.2).to(self.device, torch.float) 
            self._ball_targets_op[env_ids, 1] = torch.Tensor(n).uniform_(-0.6, 0.6).to(self.device, torch.float) 

            self._ball_states[env_ids, 3:6] = 0
            self._ball_states[env_ids, 6] = 1

            self._ball_states[env_ids, 10:13] = 0

            launch_dir = torch.bernoulli(self._launch_prob[env_ids])

            self.launch_dir[env_ids] = launch_dir.clone()
            launch_to_me = env_ids[launch_dir>0]
            launch_to_op = env_ids[launch_dir<1]

            self._target_root_pos[env_ids, 2] = 0.75
            self._target_root_pos_op[env_ids, 2] = 0.75
            
            n1 = len(launch_to_me)
            n2 = len(launch_to_op)
            # print(n1,n2)
            if n1>0:
                self._ball_vel[launch_to_me, 0] = torch.Tensor(n1).uniform_(2.0, 3.0).to(self.device, torch.float) 
                self._ball_vel[launch_to_me, 1] = torch.Tensor(n1).uniform_(-0.4, -0.4).to(self.device, torch.float) 
                self._ball_vel[launch_to_me, 2] = 0
                ball_side_pos_y = (1.37 / self._ball_vel[launch_to_me, 0]) * self._ball_vel[launch_to_me, 1]

                target_root_transl = ((ball_side_pos_y - 0) > 0.) * 0.6

                self.target_root_transl[launch_to_me] = target_root_transl.clone()
                self._target_root_pos[launch_to_me, 1] = ball_side_pos_y.clone() - target_root_transl

                if self.num_agents>1:
                    self.target_root_transl_op[launch_to_me] = 0
                    self._target_root_pos_op[launch_to_me, 1] =  self._humanoid_root_states_list[1][launch_to_me, 1]

            if n2>0: #num_agents>1
                self._ball_vel[launch_to_op, 0] = torch.Tensor(n2).uniform_(-3.0, -2.0).to(self.device, torch.float) 
                self._ball_vel[launch_to_op, 1] = torch.Tensor(n2).uniform_(-0.4, 0.4).to(self.device, torch.float) 
                self._ball_vel[launch_to_op, 2] = 0
                ball_side_pos_y = -(1.37 / self._ball_vel[launch_to_op, 0]) * self._ball_vel[launch_to_op, 1]

                target_root_transl = ((ball_side_pos_y - 0) < 0.) * 0.6
                self.target_root_transl_op[launch_to_op] = target_root_transl.clone()
                self._target_root_pos_op[launch_to_op, 1] = ball_side_pos_y.clone() + target_root_transl
                self.target_root_transl[launch_to_op] = 0
                self._target_root_pos[launch_to_op, 1] =  self._humanoid_root_states_list[0][launch_to_op, 1]

            # if not self.headless:
                # self._target_pos[env_ids,:3] = self._ball_targets[env_ids, :3].clone()
                # env_ids_int32 = torch.cat([self._ball_actor_ids[env_ids],self._target_actor_ids[env_ids]])
            # else:
            env_ids_int32 = self._ball_actor_ids[env_ids]

            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        return
    
    def _reset_envs(self, env_ids):

        self._reset_env_ball(env_ids)
        super()._reset_envs(env_ids)

        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        return
    
      
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []

        if (env_ids is None):
            ball_pos =  self._ball_pos.clone()
            ball_vel = self._ball_vel.clone()
            target_root_pos = self._target_root_pos.clone()
            target_ball_pos = self._ball_targets.clone()
            target_root_pos_op = self._target_root_pos_op.clone()
            target_ball_pos_op = self._ball_targets_op.clone()

        else:
            ball_pos = self._ball_pos[env_ids].clone()
            ball_vel = self._ball_vel[env_ids].clone()
            target_root_pos = self._target_root_pos[env_ids].clone()
            target_ball_pos = self._ball_targets[env_ids].clone()
            target_root_pos_op = self._target_root_pos_op[env_ids].clone()
            target_ball_pos_op = self._ball_targets_op[env_ids].clone()

        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
                righthand_pos = self._rigid_body_pos_list[i][:, self._key_body_ids[-2], :].clone()
                righthand_rot = self._rigid_body_rot_list[i][:, self._key_body_ids[-2], :].clone()
                paddle_pos = righthand_pos + quat_rotate(righthand_rot, self._paddle_to_hand.clone())
                

            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                righthand_pos = self._rigid_body_pos_list[i][env_ids, self._key_body_ids[-2], :].clone()
                righthand_rot = self._rigid_body_rot_list[i][env_ids, self._key_body_ids[-2], :].clone()
                paddle_pos = righthand_pos + quat_rotate(righthand_rot, self._paddle_to_hand[env_ids].clone())
            if i==0:
                obs = compute_pingpong_observations(root_states, ball_pos, ball_vel, target_root_pos, paddle_pos, target_ball_pos)
            else:
                obs = compute_pingpong_observations(root_states, ball_pos, ball_vel, target_root_pos_op, paddle_pos, target_ball_pos_op)
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):

        ball_pos = self._ball_pos[..., :3].clone()
        ball_vel = self._ball_vel[..., :3].clone()

        # height_threshold = (ball_pos[..., 2]<1.0)
        # ball_vel_z_change =  (self._ball_prev_vel_buf[...,2] * ball_vel[...,2])<0
        ball_vel_z_change = torch.logical_and(self._ball_prev_vel_buf[..., 2]<0,  ball_vel[..., 2]>0)
        ball_vel_x_change = (self._ball_prev_vel_buf[...,0] * ball_vel[...,0])<0
        
        self._ball_prev_vel_buf = ball_vel.clone()

        # table_x_y_threshold = torch.logical_and(torch.abs(ball_pos[..., 0])<1.4, torch.abs(ball_pos[..., 1])<0.78)
        
        # contact_table = torch.logical_and(height_threshold, torch.logical_and(ball_vel_z_change, table_x_y_threshold))
        
        contact_table = torch.logical_and(ball_vel_z_change, ~ball_vel_x_change)

        check_mytable_contact = torch.logical_and(ball_pos[:,0]>0, contact_table)
        check_othertable_contact = torch.logical_and(ball_pos[:,0]<0, contact_table)

        self._ball_mytable_contact_buf += check_mytable_contact#*((self._ball_paddle_contact_buf+self._ball_paddle_contact_buf_op)>0)
        self._ball_othertable_contact_buf += check_othertable_contact

        check_mytable_contact_id=check_mytable_contact>0
        check_othertable_contact_id = check_othertable_contact>0

        self.ball_mytable_contact_time_buf[check_mytable_contact_id,0]= (self.ball_mytable_contact_time_buf[check_mytable_contact_id,1]).clone()
        self.ball_othertable_contact_time_buf[check_othertable_contact_id,0]=(self.ball_othertable_contact_time_buf[check_othertable_contact_id,1]).clone()

        self.ball_mytable_contact_time_buf[check_mytable_contact_id,1]= (self.progress_buf[check_mytable_contact_id]).clone()
        self.ball_othertable_contact_time_buf[check_othertable_contact_id,1] = (self.progress_buf[check_othertable_contact_id]).clone()

        predict_ball_land_pos = predict_land_point(ball_pos, ball_vel[..., :3])

        self._ball_table_reward_buf += check_othertable_contact

        self._ball_table_reward_buf_op += check_mytable_contact

        for i in range(self.num_agents):
            # reward = 1
            root_states = self._humanoid_root_states_list[i].clone()
            root_pos = root_states[:, 0:3]
            root_rot = root_states[:, 3:7]

            righthand_pos = self._rigid_body_pos_list[i][:, self._key_body_ids[-2], :].clone()
            righthand_rot = self._rigid_body_rot_list[i][:, self._key_body_ids[-2], :].clone()
            paddle_pos = righthand_pos + quat_rotate(righthand_rot, self._paddle_to_hand.clone())
            ball_paddle_dis = torch.norm(self._ball_pos-paddle_pos,dim=-1)

            if i==0:
                ball_paddle_contact_check = torch.logical_and(ball_paddle_dis < 0.2, ball_vel_x_change)
                ball_paddle_contact_check = torch.logical_or(ball_paddle_contact_check, ball_paddle_dis<0.05)
                self._ball_paddle_contact_buf += ball_paddle_contact_check
                self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_ball_reward(root_pos, root_rot, 
                                                 ball_pos, ball_vel, paddle_pos, self._ball_paddle_contact_buf, self._ball_table_reward_buf, self._ball_targets, self._target_root_pos, predict_ball_land_pos)

            elif i==1:     
                ball_paddle_contact_check_op = torch.logical_and(ball_paddle_dis < 0.2, ball_vel_x_change)
                ball_paddle_contact_check_op = torch.logical_or(ball_paddle_contact_check_op, ball_paddle_dis<0.05)
                self._ball_paddle_contact_buf_op += ball_paddle_contact_check_op
                
                self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_ball_reward(root_pos, root_rot, 
                                                 ball_pos, ball_vel, paddle_pos, self._ball_paddle_contact_buf_op, self._ball_table_reward_buf_op, self._ball_targets_op, self._target_root_pos_op, predict_ball_land_pos)

        self._update_ball(ball_paddle_contact_check, ball_paddle_contact_check_op)
        
        
  
        return

    def _compute_reset(self): 
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces_list, self._contact_body_ids,
                                                   self._rigid_body_pos_list, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self.ball_mytable_contact_time_buf, self.ball_othertable_contact_time_buf, 
                                                   self._ball_pos, self.num_agents)

        return

    def _draw_task(self):

        
        return


class HumanoidPP2Z(HumanoidPP2):
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

def compute_compete_rewards(ball_mytable_contact_time_buf, ball_othertable_contact_time_buf):
    me_win = ball_othertable_contact_time_buf > ball_mytable_contact_time_buf
    op_win = ball_othertable_contact_time_buf <= ball_mytable_contact_time_buf
    me_reward = me_win * 10 - op_win * 10
    op_reward = op_win * 10 - me_win * 10
    return me_reward, op_reward


@torch.jit.script
def predict_land_point(ball_pos,ball_vel):
    v_z = ball_vel[...,2]
    z = ball_pos[..., 2]
    h = torch.abs(z-0.8)
    t = (2 * v_z + torch.sqrt(4*v_z**2+8*9.8*h))/(9.8*2)

    ball_land_pos = ball_pos[...,:2] + ball_vel[...,:2] * t[:,None]
    ball_land_pos = torch.clamp(ball_land_pos,min=-2,max=2)
    return ball_land_pos

# @torch.jit.script
def compute_pingpong_observations(root_states, ball_pos, ball_vel, target_root_pos, left_paddle_pos, target_ball_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # heading_rot_ = heading_rot.clone()

    local_ball_pos = quat_rotate(heading_rot, ball_pos - root_pos)

    local_ball_vel = quat_rotate(heading_rot, ball_vel)
    # if op:
    #     local_ball_vel[..., 0] *= -1
    #     local_ball_vel[..., 1] *= -1

    # print(root_pos[0],target_root_pos[0])
    # print(target_root_pos.shape)
    
    local_target_root_pos = quat_rotate(heading_rot, target_root_pos - root_pos)
    # time.sleep(0.5)
    # print(root_pos[0], target_root_pos[0], local_target_root_pos[0])

    local_paddle_pos = quat_rotate(heading_rot, left_paddle_pos - root_pos)

    local_target_ball_pos = quat_rotate(heading_rot, target_ball_pos - root_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_target_root_pos, local_paddle_pos, local_target_ball_pos],dim=-1)

    return obs



# @torch.jit.script
def compute_ball_reward(root_pos, root_rot, ball_pos, ball_vel, paddle_pos, ball_paddle_contact_buf, ball_table_reward_buf, ball_targets, root_targets, predict_ball_land_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    pos_err_scale = 4.0

    pos_diff = ball_pos - paddle_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    

    pos_diff3 = root_targets[:,:3] - root_pos[:,:3]
    pos_err3 = torch.sum(pos_diff3 * pos_diff3, dim=-1)
    pos_reward3 = torch.exp(-pos_err_scale * pos_err3)

    pos_diff4 = predict_ball_land_pos - ball_targets[...,:2]
    pos_err4 = torch.sum(pos_diff4 * pos_diff4, dim=-1)
    pos_reward4 = torch.exp(-pos_err_scale * pos_err4)

    reward = (ball_paddle_contact_buf<1) *  pos_reward * pos_reward3   \
        +  (ball_paddle_contact_buf>=1) * (torch.abs(ball_vel[:,0])>2.5) * (1 + 1 * pos_reward4 + 1 * (pos_err4 < 0.01)) * (ball_table_reward_buf <= 1) - 1 * (ball_paddle_contact_buf>2)
    return reward   


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           max_episode_length, enable_early_termination, termination_heights, ball_mytable_contact_time_buf, 
                           ball_othertable_contact_time_buf, ball_pos, num_agents):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
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
        ball_contact_body = torch.any(body_ball_dis<0.1, dim=-1)

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
                ball_contact_body = torch.logical_or(ball_contact_body, torch.any(body_ball_dis<0.1, dim=-1))
   
        # ball_fail = torch.logical_and(torch.logical_or(ball_pos[...,0]<0, ball_othertable_contact_buf==0), ball_pos[...,2]<0.5)
        ball_fail = torch.logical_or(ball_pos[..., 2] < 0.6, ball_pos[..., 2] > 1.8)
        ball_fail = torch.logical_or(ball_fail, torch.abs(ball_pos[..., 0]) > 2.5)
        
        ball_fail = torch.logical_or(ball_fail, ball_contact_body)
        
        ball_fail_hit = torch.logical_or(ball_mytable_contact_time_buf[:, 0] > ball_othertable_contact_time_buf[:, 1],
                                         ball_mytable_contact_time_buf[:, 1] < ball_othertable_contact_time_buf[:, 0])
        ball_fail = torch.logical_or(ball_fail, ball_fail_hit)
        
        # has_failed = has_fallen
        has_failed = torch.logical_or(has_fallen, ball_fail)


        has_failed *= (progress_buf > 1)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)


    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    
    return reset, terminated
