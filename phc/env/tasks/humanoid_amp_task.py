import torch
import time
import phc.env.tasks.humanoid_amp as humanoid_amp
import phc.env.tasks.humanoid_amp_z as humanoid_amp_z
from phc.utils.flags import flags
class HumanoidAMPTask(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.has_task = True
        return


    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
       
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer or flags.server_mode:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        humanoid_obs_list = self._compute_humanoid_obs(env_ids)

        if (self._enable_task_obs):
            task_obs_list = self._compute_task_obs(env_ids)
            
            
        for i in range(self.num_agents):
            
            self.obs_buf[env_ids+i*self.num_envs] = torch.cat([humanoid_obs_list[i], task_obs_list[i]],dim=-1)
        # time.sleep(5)
        # else:
        #     obs = humanoid_obs
        #     obs_op = humanoid_obs_op
        
                
        # if self.obs_v == 2:
        #     # Double sub will return a copy.
        #     B, N = obs.shape
        #     sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
        #     zeros = sums == 0
        #     nonzero = ~zeros
        #     obs_slice = self.obs_buf[env_ids]
        #     obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
        #     obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
        #     self.obs_buf[env_ids] = obs_slice
        # else:
        #     self.obs_buf[env_ids] = obs
        #     self.obs_buf[env_ids+self.num_envs] = obs_op

        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return


class HumanoidAMPZTask(humanoid_amp_z.HumanoidAMPZ):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.has_task = True
        return


    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer or flags.server_mode:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        humanoid_obs, humanoid_obs_op = self._compute_humanoid_obs(env_ids)

        if (self._enable_task_obs):
            task_obs, task_obs_op = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
            obs_op = torch.cat([humanoid_obs_op, task_obs_op], dim=-1)


        else:
            obs = humanoid_obs
            obs_op = humanoid_obs_op

        if self.motion_sym_loss:
            flip_obs = self._compute_flip_humanoid_obs(env_ids)

            if (self._enable_task_obs):
                flip_task_obs = self._compute_flip_task_obs(task_obs, env_ids)
                flip_obs = torch.cat([flip_obs, flip_task_obs], dim=-1)

            if (env_ids is None):
                self._flip_obs_buf[:] = flip_obs
            else:
                self._flip_obs_buf[env_ids] = flip_obs
                
        if self.obs_v == 2:
            # Double sub will return a copy.
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return