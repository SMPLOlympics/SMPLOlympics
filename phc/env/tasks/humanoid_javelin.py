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
import joblib

TAR_ACTOR_ID = 1

class HumanoidJavelin(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if device_type == "cuda" or device_type == "GPU":
            self.device = "cuda" + ":" + str(device_id)
        
        # self.jevalin_default_pose = torch.tensor([[ 0.2433, -0.59,  1.61 , 0, 0, 0, 1]]).float().repeat(cfg.env.num_envs, 1).to(self.device)
        self.jevalin_default_pose = torch.tensor([[ 0.2433, -0.59,  1.61 ,0, -0.1736482, 0, 0.9848078  ]]).float().repeat(cfg.env.num_envs, 1).to(self.device) # 45 degree
        self.javelin_state = joblib.load("sample_data/javelin_state.pkl")
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.first_in = True
        self._build_target_tensors()
        self.bounding_box = torch.tensor([-2, 2, -2, 2,]).to(self.device) # x_min, x_max, y_min, y_max
        self.bounding_box_points = torch.tensor([[[self.bounding_box[0], self.bounding_box[2], 0], [self.bounding_box[1], self.bounding_box[3], 0]]]).repeat(self.num_envs, 1, 1).to(self.device)
        
        self._prev_target_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
 
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
 
        self.goal = torch.tensor([50,0,1]).to(self.device)
        self.step_counter=torch.zeros((self.num_envs)).to(self.device)
        self.warmup_time = torch.tensor([20]).to(self.device)
        self.tar_speed = 4
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
        
    def post_physics_step(self):
        self.step_counter += 1
        super().post_physics_step()
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
        asset_file = "javelin.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 2000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        prp = self.gym.get_asset_rigid_shape_properties(self._target_asset)
        for each in prp:
            each.friction = 1
            each.restitution = 0.8
            each.contact_offset = 0.02
            each.rolling_friction = 0.1

        self.gym.set_asset_rigid_shape_properties(self._target_asset,prp)

        return

    def _build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = self.jevalin_default_pose[0, 0]
        default_pose.p.y = self.jevalin_default_pose[0, 1]
        default_pose.p.z = self.jevalin_default_pose[0, 2]
        default_pose.r = gymapi.Quat(self.jevalin_default_pose[0, 3], self.jevalin_default_pose[0, 4], self.jevalin_default_pose[0, 5], self.jevalin_default_pose[0, 6])
        
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", env_id, 0)
        self._target_handles.append(target_handle)
        self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 0))
        
        return

    def _build_target_tensors(self):
        self._righthand_rigid_body_ids = []
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles_list[0][0]
        self._rigid_body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, actor_handle)
        for name in ["R_Index1","R_Middle1", "R_Pinky1", "R_Ring1", "R_Thumb1"]:
            self._righthand_rigid_body_ids.append( self._rigid_body_dict[name])

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
       
        self._target_states[env_ids, :7] = self.jevalin_default_pose[env_ids].clone()
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        self.step_counter[env_ids] = 0
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 13 + 6 + 1 + 1
        return obs_size
    
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(env_ids)
        
        motion_ids_jv, motion_times_jv, root_pos_jv, root_rot_jv, dof_pos_jv, root_vel_jv, root_ang_vel_jv, dof_vel_jv,  rb_pos_jv, rb_rot_jv, body_vel_jv, body_ang_vel_jv = self.javelin_state
        
        root_pos[:] = root_pos_jv
        root_rot[:] = root_rot_jv
        dof_pos[:] = dof_pos_jv
        root_ang_vel[:] = root_ang_vel_jv
        root_vel[:] = root_vel_jv
        dof_vel[:] = dof_vel_jv
        body_vel[:] = body_vel_jv
        body_ang_vel[:] = body_ang_vel_jv
        rb_pos[:] = rb_pos_jv
        rb_rot[:] = rb_rot_jv
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel
    
  
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        
        self._prev_target_pos= self._target_states[:, 0:3].clone()
        return
    
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []
 
        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
                tar_states = self._target_states
                bounding_box_points = self.bounding_box_points
                step_counter = self.step_counter
                default_pose = self.jevalin_default_pose
            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                tar_states = self._target_states[env_ids]
                bounding_box_points = self.bounding_box_points[env_ids]
                step_counter = self.step_counter[env_ids]
                default_pose = self.jevalin_default_pose[env_ids]
            
            obs = compute_javelin_observations(root_states, tar_states, default_pose, bounding_box_points, step_counter, self.max_episode_length)

            obs_list.append(obs)

        
        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            target_states = self._target_states
            root_states = self._humanoid_root_states_list[i]
            right_hand_states = self._rigid_body_pos_list[i][:, self._righthand_rigid_body_ids, :].clone()
            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = compute_javelin_reward(self.jevalin_default_pose, self.step_counter, self.warmup_time, root_states, self._prev_target_pos, target_states, right_hand_states, self.jevalin_default_pose)

        return

    def _compute_reset(self):
        fuzzy = 0.01
        out_bound = torch.zeros(self.num_envs).to(self.device).bool()
        for i in range(self.num_agents):
            root_pos = self._humanoid_root_states_list[i][..., 0:3]
            out_bound += torch.logical_or(torch.logical_or(root_pos[..., 0] < self.bounding_box[0] - fuzzy,   root_pos[..., 0] > self.bounding_box[1] + fuzzy),   torch.logical_or(root_pos[..., 1] < self.bounding_box[2] - fuzzy,  root_pos[..., 1] > self.bounding_box[3] + fuzzy))

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_javelin_reset(self.step_counter, self.warmup_time, self._righthand_rigid_body_ids, self.jevalin_default_pose,
                                                                                   self.reset_buf, self.progress_buf,
                                                                                   self._contact_forces_list, self._contact_body_ids,
                                                                                   self._rigid_body_pos_list, self._target_states, self.max_episode_length,
                                                                                   self._enable_early_termination, self._termination_heights, self.num_agents)
        
        self._terminate_buf[:] = torch.logical_or(self._terminate_buf, out_bound) # Going out of the bound means termination.
        self.reset_buf[:] = torch.logical_or(self.reset_buf, out_bound)
        return

 
class HumanoidJavelinZ(HumanoidJavelin):
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
def compute_javelin_observations(root_states, javelin_states, javelin_default_pose, bounding_box_points, step_counter, max_episode_length):
    B, J = root_states.shape
    obs = []
    phase = step_counter.view(-1, 1) / max_episode_length
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    
    target_pos = javelin_states[:, 0:3]
    target_rot = javelin_states[:, 3:7]
    
    default_rot = javelin_default_pose[:, 3:7]
    default_pos = javelin_default_pose[:, 0:3]

    
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    
    local_tar_pos = target_pos - root_pos
    local_tar_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_tar_pos)
    
    local_tar_rot = torch_utils.quat_mul(heading_rot_inv, target_rot)
    local_tar_rot = torch_utils.quat_to_tan_norm(local_tar_rot)
    
    tar_rot_diff = torch_utils.quat_mul(default_rot, torch_utils.quat_conjugate(target_rot))
    local_tar_rot_diff = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot_inv, tar_rot_diff.view(-1, 4)), heading_rot.view(-1, 4))  # Need to be change of basis
    local_tar_rot_diff = torch_utils.quat_to_tan_norm(local_tar_rot_diff)
    
    diff_y = default_pos[:, 1:2] - target_pos[:, 1:2]
    
    local_bounding_pos = bounding_box_points - root_pos[:, None]
    local_bounding_pos = torch_utils.my_quat_rotate(heading_rot_inv[:, None].repeat(1, 2, 1).view(-1, 4), local_bounding_pos.view(-1, 3))[..., :2].reshape(B, -1)
    
    obs.append(local_tar_pos)
    obs.append(local_tar_rot)
    obs.append(local_tar_rot_diff)
    obs.append(diff_y)
    obs.append(local_bounding_pos)
    obs.append(phase)
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs



def compute_javelin_reward(jevalin_default_pose, step_counter, warmup_time, root_states, prev_target_pos, object_states, right_hand_states, target_default_pose):
    is_warmup = step_counter<warmup_time
    is_throwing = torch.logical_and(step_counter >= warmup_time , step_counter <= warmup_time + 20)
    is_flying = step_counter > warmup_time + 20
    
    k_pos, k_rot, k_vel, k_ang_vel = 100, 10, 10, 0.1

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    r_humanoid_root_diff = torch.exp(-1  * (root_pos** 2).mean(dim = -1))
    javelin_pos = object_states[:, 0:3]
    javelin_rot = object_states[:, 3:7]
    javelin_lin_vel = object_states[:, 7:10]
    javelin_ang_vel = object_states[:, 10:13]

    right_hand_to_target_diff = torch.norm(javelin_pos - torch.mean(right_hand_states, dim=1), dim=-1 )
    r_righthand_closer_to_target = torch.exp(-1 * right_hand_to_target_diff)
    r_righthand_further_to_target = torch.clamp(right_hand_to_target_diff, min=0, max=1)
    r_pos_diff = torch.exp(-k_pos * ((javelin_pos - jevalin_default_pose[:, 0:3]) ** 2).mean(dim = -1))
    target_rot_diff, axis = torch_utils.quat_to_angle_axis(torch_utils.quat_mul(javelin_rot, torch_utils.quat_conjugate(jevalin_default_pose[:, 3:7])))
    r_target_rot_diff = torch.clamp(target_rot_diff, min=0, max=1)
    r_vel_diff = torch.exp(-k_vel * (javelin_lin_vel ** 2).mean(dim = -1))
    r_ang_diff = torch.exp(-k_vel * (javelin_ang_vel ** 2).mean(dim = -1)) 
    r_state_diff = r_pos_diff * 0.7 + r_target_rot_diff * 0.2 + r_vel_diff * 0.05 + r_ang_diff* 0.05 # warm up stage aim at holding the javelin in hand stabely 
    

    goal_pos = torch.tensor([50, 0, 0]).to(root_states)
    javelin_to_goal_curr = goal_pos - javelin_pos 
    javelin_to_goal_prev = goal_pos - prev_target_pos 
    
    javelin_to_goal_curr = javelin_to_goal_curr[:, 0].abs()
    javelin_to_goal_prev = javelin_to_goal_prev[:, 0].abs()
    
    r_throw_closer = torch.clamp((javelin_to_goal_prev - javelin_to_goal_curr) * 1/5, min=0,  max=1)
    # r_throw_r = torch.exp(-1 * ((javelin_pos - goal_pos) ** 2).mean(dim=-1))
    
    target_default_rot = target_default_pose[:, 3:7] # javalin heading reward
    diff_global_body_rot = torch_utils.quat_mul(torch_utils.calc_heading_quat(target_default_rot), torch_utils.calc_heading_quat_inv(javelin_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    r_heading = torch.exp(-k_rot * (diff_global_body_angle ** 2).mean(dim=-1))
    
    reward = is_warmup.float() * (0.9 * r_righthand_closer_to_target + 0.1 * r_state_diff) * 0.1 + \
             is_throwing.float() * (0.9 * r_throw_closer  + 0.05 * r_humanoid_root_diff + 0.05 * r_righthand_further_to_target ) + \
             is_flying.float() * (0.9 * r_throw_closer  + 0.1 * r_heading)
    # import ipdb; ipdb.set_trace()
    return reward

# @torch.jit.script
def compute_humanoid_javelin_reset(step_counter, warmup_time, right_hand_rigid_body_id, javelin_default_pose, reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list, target_states,
                           max_episode_length,
                           enable_early_termination, termination_heights, num_agents):
    # type: (Tensor, Tensor, list, Tensor, list, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    is_warmup = step_counter < warmup_time
    is_throw = torch.logical_and(step_counter >= warmup_time, step_counter <= warmup_time + 20)
    is_flying = step_counter > warmup_time + 20

    terminated = torch.zeros_like(reset_buf)

    masked_contact_buf = contact_buf_list[0].clone()
    masked_contact_buf[:, contact_body_ids, :] = 0
    ## torch.sum to disable self-collision.
    force_threshold = 50
    
    body_contact_force = torch.sqrt(torch.square(masked_contact_buf.sum(dim=-2)).sum(dim=-1)) > force_threshold

    has_body_contact = body_contact_force
    has_body_contact *= (progress_buf > 1)

    # first timestep can sometimes still have nonzero contact forces
    # so only check after first couple of steps
    terminated = torch.where(has_body_contact, torch.ones_like(reset_buf), terminated)
    
    tar_pos = target_states[:, 0:3]
    tar_rot = target_states[:, 3:7]
    target_default_rot = javelin_default_pose[:, 3:7] # javalin heading reward
    diff_global_body_rot = torch_utils.quat_mul(torch_utils.calc_heading_quat_inv(target_default_rot), torch_utils.calc_heading_quat(tar_rot))
    diff_global_heading_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    target_rot_too_much_when_warmup = torch.logical_and(diff_global_heading_angle>0.25, ~is_flying)

    tar_h = tar_pos[:, 2]
    tar_x = tar_pos[:, 0]
    tar_y = tar_pos[:, 1]
    
    tar_y_out = torch.logical_or(tar_y < javelin_default_pose[:, 1] - 0.1, tar_y > javelin_default_pose[:, 1] + 0.1) # default is 0.59
    tar_h_fallen = tar_h < 0.3
    tar_x_out_bound = tar_x > 1
    
    target_righthand_diff = torch.norm(tar_pos - torch.mean(rigid_body_pos_list[0][:, right_hand_rigid_body_id], dim=1), dim=-1)
    target_not_in_hand_when_warmup = torch.logical_and (target_righthand_diff>0.15, is_warmup)
    target_in_hand_when_time_to_fly = torch.logical_and (target_righthand_diff<0.15, is_flying)
    
    tar_terminate = torch.logical_and(tar_h_fallen, ~tar_x_out_bound) # javelin fallen and still close is a terminate
    tar_terminate = torch.logical_or(tar_terminate, torch.logical_and(tar_y_out, is_warmup)) # javelin should not move more than 20 cm in the y direction and thrown directly
    tar_terminate = torch.logical_or(tar_terminate, target_not_in_hand_when_warmup)
    tar_terminate = torch.logical_or(tar_terminate, target_rot_too_much_when_warmup)
    tar_terminate = torch.logical_or(tar_terminate, target_in_hand_when_time_to_fly)
    terminated = torch.logical_or(terminated, tar_terminate)
    
    tar_reset = torch.logical_and(tar_h_fallen, tar_x_out_bound) # javelin fallen and out of the bound is a reset 
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    reset = torch.logical_or(reset, tar_reset)
    # if reset.sum() > 0:
        # import ipdb; ipdb.set_trace()
        
    throw_length = tar_pos[tar_h_fallen, 0]
    if flags.test and not torch.equal(throw_length, torch.tensor([]).to(throw_length.device)):
        print("reset, javelin length",throw_length)
    # if reset[0]:
        #  print("reset 0, javelin pos x:", tar_pos[0,0],"rotate", target_rot_too_much_when_warmup[0], "warmup not in hand", target_not_in_hand_when_warmup[0], "contact", body_contact_force[0], "throw but in hand", target_in_hand_when_time_to_throw[0])
    # reset[:] = 0
    return reset, terminated

