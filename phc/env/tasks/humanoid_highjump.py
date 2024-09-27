
import numpy as np
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


class HumanoidHighjump(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        if flags.test:
            self.all_leap_over = torch.zeros(self.num_envs).to(self.device)
            self.all_height = torch.zeros(self.num_envs).to(self.device)
            self.all_x_near_pole = torch.zeros(self.num_envs).to(self.device)
            self.all_jumps = torch.zeros(self.num_envs).to(self.device)


        self._tar_dist_min = 0.5
        self._tar_dist_max = 1.0
        self._near_dist = 1.5
        self._near_prob = 0.5
        self.first_in = True
        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))

        # strike_body_names = cfg["env"]["strikeBodyNames"]
        self.goal = torch.tensor([22, 6, 1]).to(self.device)
        self.highjump_pole_pos_init = torch.tensor([20., 6., 0.]).unsqueeze(0).unsqueeze(0).repeat(1, self.num_envs, 1).to(
            self.device)
        env_id = torch.arange(0, self.num_envs, 4).to(dtype=torch.long, device=self.device)

        self.highjump_pole_pos_init[:, env_id, 2] = 2
        self.highjump_pole_pos_init[:, env_id + 1, 2] = 1.5
        self.highjump_pole_pos_init[:, env_id + 2, 2] = 1
        self.highjump_pole_pos_init[:, env_id + 3, 2] = 0.5

        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)

        self.set_initial_root_state()
        self._build_predefined_tensors()
        self._build_pole_state_tensors()
        self.tar_speed = 4
        return
    def _build_predefined_tensors(self):
        self._pole_actor_id = self.num_agents
        return
    def _build_pole_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._pole_root_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[...,
                                 self._pole_actor_id, :]

        self._pole_pos = self._pole_root_states[..., :3]

        self._pole_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device,
                                        dtype=torch.int32) + self._pole_actor_id

        return
    def set_initial_root_state(self):
        initial_humanoid_root_states = self._initial_humanoid_root_states_list[0].clone()

        initial_humanoid_root_states[..., 0] = 0
        initial_humanoid_root_states[..., 1] = 0
        initial_humanoid_root_states[..., 2] = 1
        initial_humanoid_root_states[..., 3] = 0
        initial_humanoid_root_states[..., 4] = 0
        initial_humanoid_root_states[..., 5] = 0
        initial_humanoid_root_states[..., 6] = 1
        initial_humanoid_root_states[:, 7:13] = 0
        self._initial_humanoid_root_states_list[0] = initial_humanoid_root_states

        self._initial_pole_root_states = initial_humanoid_root_states.clone()
        self._initial_pole_root_states[..., 0:3] = self.highjump_pole_pos_init
        self._initial_pole_root_states[...,3:6] = 0
        self._initial_pole_root_states[...,6] = 1
        self._initial_pole_root_states[...,7] = 0 # Zero speed will crash
        self._initial_pole_root_states[...,8:13] = 0

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3 + 3
        return obs_size

    def post_physics_step(self):
        # self.out_bound, self.red_win, self.green_win = self.check_game_state()

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
        asset_file = "highjump_pole.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = False
        self._highjump_pole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_root = "phc/data/assets/urdf/"
        asset_file = "highjump.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        self._highjump_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_root = "phc/data/assets/urdf/"
        asset_file = "highjump_small.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        self._highjump_small_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_root = "phc/data/assets/urdf/"
        asset_file = "highjump_short.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        self._highjump_short_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_root = "phc/data/assets/urdf/"
        asset_file = "highjump_tiny.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        self._highjump_tiny_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        color_vec = gymapi.Vec3(1, 0, 0)

        self._pole_handle = self.gym.create_actor(env_ptr, self._highjump_pole_asset, default_pose, "pole", env_id, 0)
        self.gym.set_rigid_body_color(env_ptr, self._pole_handle, 0, gymapi.MESH_VISUAL, color_vec)

        if env_id % 4 == 0:
            self._target_handle = self.gym.create_actor(env_ptr, self._highjump_asset, default_pose, "target", env_id, 0)
        elif env_id % 4 == 1:
            self._target_handle = self.gym.create_actor(env_ptr, self._highjump_short_asset, default_pose, "target", env_id, 0)
        elif env_id % 4 == 2:
            self._target_handle = self.gym.create_actor(env_ptr, self._highjump_small_asset, default_pose, "target", env_id, 0)
        elif env_id % 4 == 3:
            self._target_handle = self.gym.create_actor(env_ptr, self._highjump_tiny_asset, default_pose, "target", env_id, 0)

        self.gym.set_rigid_body_color(env_ptr, self._target_handle, 0, gymapi.MESH_VISUAL, color_vec)

        return
    def _reset_envs(self, env_ids):
        self._reset_env_pole(env_ids)
        super()._reset_envs(env_ids)

        return

    def _reset_env_pole(self, env_ids):
        n = len(env_ids)
        if n>0:
            self._pole_root_states[env_ids, :] = self._initial_pole_root_states[env_ids, :]
            env_ids_int32 = self._pole_actor_ids[env_ids]
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

        for i in range(self.num_agents):
            if (env_ids is None):
                root_states = self._humanoid_root_states_list[i]
                highjump_location = self.highjump_pole_pos_init[i]
            else:
                root_states = self._humanoid_root_states_list[i][env_ids]
                highjump_location = self.highjump_pole_pos_init[i][env_ids]

            obs = compute_highjump_observations(root_states, self.goal, highjump_location)
            obs_list.append(obs)

        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            # reward = 1
            root_states = self._humanoid_root_states_list[i]
            key_bodies_pos = self._rigid_body_pos_list[i][:, self._key_body_ids[:2], :]
            highjump_location = self.highjump_pole_pos_init[i]
            self.rew_buf[i * self.num_envs:(i + 1) * self.num_envs] = compute_highjump_reward(root_states,
                                                                                              self._prev_root_pos_list[i], self.goal,
                                                                                              highjump_location,
                                                                                              key_bodies_pos,
                                                                                              self.tar_speed, self.dt)

        return

    def _compute_reset(self):
        for i in range(self.num_agents):
            if flags.test:
                self.reset_buf[:], self._terminate_buf[:], has_leaped_over, jump_height, x_near_highjump_location= compute_humanoid_reset_evaluation(self.reset_buf, self.progress_buf,
                                                                                                                                                     self._contact_forces_list[0],
                                                                                                                                                     self._contact_body_ids,
                                                                                                                                                     self._rigid_body_pos_list[0],
                                                                                                                                                     self._humanoid_root_states_list[i],
                                                                                                                                                     self.max_episode_length,
                                                                                                                                                     self._enable_early_termination,
                                                                                                                                                     self._termination_heights,
                                                                                                                                                     self.num_agents,
                                                                                                                                                     self.highjump_pole_pos_init[i])

                self.all_jumps += self.reset_buf
                self.all_height += jump_height
                self.all_x_near_pole += x_near_highjump_location
                self.all_leap_over += has_leaped_over


                env_ids = torch.arange(0, self.num_envs, 4).to(dtype=torch.long, device=self.device)
                def calc_average(height, env_num):
                    sum_height = torch.sum(height)/torch.sum(env_num)
                    return sum_height
                for i in range(4):
                    curr_envid = env_ids + i%4
                    if torch.sum(self.all_jumps[curr_envid])>10000:

                        print("----------------------------------------", i%4, "jumps " , torch.sum(self.all_jumps[curr_envid]), "times")
                        average_height = calc_average(self.all_height[curr_envid], self.all_x_near_pole[curr_envid])
                        print( i%4, 'average_height',average_height)

                        average_leap_over_rate = calc_average(self.all_leap_over[curr_envid], self.all_jumps[curr_envid])
                        print( i%4, 'leap_over_rate',average_leap_over_rate)

                    #import pdb; pdb.set_trace()

            else:
                self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                                   self._contact_forces_list[0],
                                                                                   self._contact_body_ids,
                                                                                   self._rigid_body_pos_list[0],
                                                                                   self._humanoid_root_states_list[i],
                                                                                   self.max_episode_length,
                                                                                   self._enable_early_termination,
                                                                                   self._termination_heights,
                                                                                   self.num_agents,
                                                                                   self.highjump_pole_pos_init[i],
                                                                                   self._pole_pos, self._prev_root_pos_list[0], self.goal)

        return

    def _draw_task(self):
        if self.first_in:
            self.first_in = False
            self.gym.clear_lines(self.viewer)
            cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            for i, env_ptr in enumerate(self.envs):
                vertices = np.array([
                    [0, 0, 0],
                    [self.highjump_pole_pos_init[0, i, 0], 0, 0],
                    [self.highjump_pole_pos_init[0, i, 0], self.highjump_pole_pos_init[0, i, 1] + 2, 0],
                    [0, self.highjump_pole_pos_init[0, i, 1] + 2, 0]
                ], dtype=np.float32)

                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]]
                ])
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 2, line, cols)
                # vertices = np.array([self.goal.numpy(), self.goal.numpy() + 0.1], dtype=np.float32)
                # lines = np.array([vertices[0], vertices[1]])
                # self.gym.add_lines(self.viewer, env_ptr, 5, lines[0], cols)


class HumanoidHighjumpZ(HumanoidHighjump):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type,
                         device_id=device_id, headless=headless)
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
def compute_highjump_observations(root_states, goal, highjump_location):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = goal - root_pos
    # local_tar_pos[..., -1] = goal[..., -1]
    local_tar_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_tar_pos)
    obs = local_tar_pos

    local_highjump_pos = highjump_location - root_pos
    # local_highjump_pos[..., -1] = goal[..., -1]
    local_highjump_pos = torch_utils.my_quat_rotate(heading_rot_inv, local_highjump_pos)
    obs = torch.cat([obs, local_highjump_pos], dim=-1)
    return obs


# @torch.jit.script
def compute_highjump_reward(root_states, prev_root_pos, tar_pos, highjump_location, key_body_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    feet_z = key_body_pos[..., -1]

    prev_dist = torch.norm(prev_root_pos - tar_pos, dim=-1)
    curr_dist = torch.norm(root_pos - tar_pos, dim=-1)
    closer_target_r = torch.clamp(prev_dist - curr_dist, min=0, max=1)  # player getting closer to the ball

    root_pos = root_states[:, 0:3]

    x_near_highjump_location = torch.logical_and(root_pos[:, 0] > highjump_location[:, 0] - 0.5,
                                                 root_pos[:, 0] < highjump_location[:, 0] + 0.5)

    jump_height_reward = torch.zeros_like(root_pos[:, 2])
    jump_height_reward[x_near_highjump_location] = root_pos[x_near_highjump_location, 2]
    reward = closer_target_r + jump_height_reward

    return reward


# @torch.jit.script
def compute_humanoid_reset_evaluation(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, root_states,
                           max_episode_length,
                           enable_early_termination, termination_heights, num_agents, highjump_location):
    # type: (Tensor, Tensor, list, Tensor, list, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    terminated = torch.zeros_like(reset_buf)

    root_pos = root_states[:, 0:3]
    x_near_highjump_location = torch.logical_and(root_pos[:, 0] > highjump_location[:, 0] - 0.1,
                                                 root_pos[:, 0] < highjump_location[:, 0] + 0.1)

    has_leaped_over = torch.min(rigid_body_pos[:, :, 0], dim=1)[0] > highjump_location[:, 0] + 0.1

    contact_force_not_zero = torch.sqrt(
        torch.sum(torch.sum(torch.square(contact_buf), dim=-1), dim=-1)) > contact_force_threshold
    hit_pole = torch.logical_and(x_near_highjump_location, contact_force_not_zero)
    lower_than_pole = torch.logical_and(x_near_highjump_location, root_pos[:, 2] < highjump_location[:, 2])
    jump_height = torch.zeros_like(root_pos[:, 2])
    jump_height[x_near_highjump_location] = root_pos[x_near_highjump_location, 2]

    masked_contact_buf = contact_buf.clone()

    masked_contact_buf[:, contact_body_ids, :] = 0
    force_threshold = 50
    fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

    body_height = rigid_body_pos[..., 2]
    fall_height = body_height < termination_heights
    fall_height[:, contact_body_ids] = False
    fall_height = torch.any(fall_height, dim=-1)


    has_fallen_when_running = torch.logical_or(fall_contact, fall_height)  # don't touch the highjump.
    has_fallen_when_running = torch.logical_and(has_fallen_when_running, root_pos[:,0]<highjump_location[:,0])
    has_failed = torch.logical_or(has_fallen_when_running, hit_pole)
    has_failed = torch.logical_or(has_failed, lower_than_pole)

    # first timestep can sometimes still have nonzero contact forces
    # so only check after first couple of steps
    has_failed *= (progress_buf > 1)
    terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    reset = torch.logical_or(reset, has_leaped_over)
    return reset, terminated, has_leaped_over, jump_height, x_near_highjump_location


def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, root_states,
                           max_episode_length, enable_early_termination, termination_heights, num_agents,
                           highjump_location, pole_pos, prev_root_states, goal):
    # type: (Tensor, Tensor, list, Tensor, list, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0

    terminated = torch.zeros_like(reset_buf)
    root_pos = root_states[:, 0:3]
    root_vel = root_states[:, 7:10]
    prev_root_pos = prev_root_states[:, 0:3]
    if (enable_early_termination):

        x_near_highjump_location = torch.logical_and(root_pos[:, 0] > highjump_location[:, 0] - 0.5,
                                                     root_pos[:, 0] < highjump_location[:, 0] + 0.2)

        root_z_lower_than_highjump_location_z = root_pos[:, 2] <= highjump_location[:, 2]
        root_y_out_of_bound = torch.logical_or(root_pos[:, 1] > highjump_location[:, 1] + 2,
                                               root_pos[:, 1] < highjump_location[:, 1] - 2)
        pole_fall = pole_pos[:, 2] < highjump_location[:, 2]

        failed_at_highjump_location = torch.logical_or(root_z_lower_than_highjump_location_z, root_y_out_of_bound)
        failed_at_highjump_location = torch.logical_and(x_near_highjump_location, failed_at_highjump_location)
        failed_at_highjump_location = torch.logical_or(failed_at_highjump_location, pole_fall)

        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        force_threshold = 50
        fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        body_y = rigid_body_pos[..., 0, 1]
        body_out = torch.logical_or(body_y < -0.5, body_y > highjump_location[:, 1] + 2)
        body_out = torch.logical_or(body_out, rigid_body_pos[..., 0, 0] < -0.5)

        humanoid_not_closer = torch.norm(goal[0] - prev_root_pos[:, 0], dim=-1) < torch.norm(
                                                goal[0] - root_pos[:, 0], dim=-1)
        humanoid_stop = torch.norm(root_vel, dim=-1) < 0.1
        has_fallen = torch.logical_or(fall_contact, fall_height)  # don't touch the highjump.
        has_fallen = torch.logical_or(has_fallen, body_out)
        has_fallen = torch.logical_or(has_fallen, humanoid_not_closer)
        has_fallen = torch.logical_or(has_fallen, humanoid_stop)
        has_fallen = torch.logical_and(has_fallen, root_pos[:, 0] < highjump_location[:, 0] + 0.1)
        has_fallen = torch.logical_or(has_fallen, failed_at_highjump_location)

        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
