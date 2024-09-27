import time
import torch
import phc.env.tasks.humanoid as humanoid
from phc.env.tasks.humanoid_amp import remove_base_rot
from phc.utils import torch_utils
from typing import OrderedDict

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.utils.torch_utils import project_to_norm

from phc.learning.network_loader import load_z_encoder, load_z_decoder

from easydict import EasyDict

HACK_MOTION_SYNC = False

class HumanoidZ(humanoid.Humanoid):

    def initialize_z_models(self):
        check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
        ### Loading Distill Model ###
        self.distill_model_config = self.cfg['env']['distill_model_config']
        self.embedding_size_distill = self.distill_model_config['embedding_size']
        self.embedding_norm_distill = self.distill_model_config['embedding_norm']
        self.fut_tracks_distill = self.distill_model_config['fut_tracks']
        self.num_traj_samples_distill = self.distill_model_config['numTrajSamples']
        self.traj_sample_timestep_distill = self.distill_model_config['trajSampleTimestepInv']
        self.fut_tracks_dropout_distill = self.distill_model_config['fut_tracks_dropout']
        self.z_activation = self.distill_model_config['z_activation']
        self.distill_z_type = self.distill_model_config.get("z_type", "sphere")
        
        self.embedding_partition_distill = self.distill_model_config.get("embedding_partion", 1)
        self.dict_size_distill = self.distill_model_config.get("dict_size", 1)
        ### Loading Distill Model ###
        
        self.z_all = self.cfg['env'].get("z_all", False)
        
        self.use_vae_prior_loss = self.cfg['env'].get("use_vae_prior_loss", False)
        self.use_vae_prior = self.cfg['env'].get("use_vae_prior", False)
        self.use_vae_fixed_prior = self.cfg['env'].get("use_vae_fixed_prior", False)
        self.use_vae_sphere_prior = self.cfg['env'].get("use_vae_sphere_prior", False)
        self.use_vae_sphere_posterior = self.cfg['env'].get("use_vae_sphere_posterior", False)
        
        
        self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device) 
        self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.power_usage_coefficient = self.cfg["env"].get("power_usage_coefficient", 0.005)

        self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']

        if self.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
                "gt_z": torch.zeros([self.num_envs,self.cfg['env'].get("embedding_size", 256) ]),
                }) # current root pos + root for future aggergration
    
    def _setup_character_props_z(self):
        self._num_actions = self.cfg['env'].get("embedding_size", 256)
        return

    def get_task_obs_size_detail_z(self):
        task_obs_detail = OrderedDict()

        ### For Z
        task_obs_detail['proj_norm'] = self.cfg['env'].get("proj_norm", True)
        task_obs_detail['embedding_norm'] = self.cfg['env'].get("embedding_norm", 3)
        task_obs_detail['embedding_size'] = self.cfg['env'].get("embedding_size", 256)
        task_obs_detail['z_readout'] = self.cfg['env'].get("z_readout", False)
        task_obs_detail['z_type'] = self.cfg['env'].get("z_type", "sphere")
        task_obs_detail['num_unique_motions'] = self._motion_lib._num_unique_motions
        return task_obs_detail

    def step_z(self, actions_z):

        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
            # t_s = time.time()
        # t_s = time.time()
        with torch.no_grad():
            # Apply trained Model.
            
            ################ GT-Z ################
            self_obs_size = self.get_self_obs_size()
            if self.obs_v == 2:
                self_obs_size = self_obs_size//self.past_track_steps
                obs_buf = self.obs_buf.view(self.num_envs * self.num_agents, self.past_track_steps, -1)
                curr_obs = obs_buf[:, -1]
                self_obs = ((curr_obs[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
            else:
                self_obs = (self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05)
                
            if self.distill_z_type == "hyper":
                actions_z = self.decoder.hyper_layer(actions_z)
            if self.distill_z_type == "vq_vae":
                if self.is_discrete:
                    indexes = actions_z
                else:
                    B, F = actions_z.shape
                    indexes = actions_z.reshape(B, -1, self.embedding_size_distill).argmax(dim = -1)
                task_out_proj = self.decoder.quantizer.embedding.weight[indexes.view(-1)]
                print(f"\r {indexes.numpy()[0]}", end = '')
                actions_z = task_out_proj.view(-1, self.embedding_size_distill)
            elif self.distill_z_type == "vae":
                if self.use_vae_prior:
                    z_prior_out = self.decoder.z_prior(self_obs)
                    prior_mu = self.decoder.z_prior_mu(z_prior_out)
                    
                    actions_z = prior_mu + actions_z
                
                if self.use_vae_sphere_posterior:
                    actions_z = project_to_norm(actions_z, 1, "sphere")
                else:
                    actions_z = project_to_norm(actions_z, self.cfg['env'].get("embedding_norm", 5), "none")
                    
            else:
                actions_z = project_to_norm(actions_z, self.cfg['env'].get("embedding_norm", 5), self.distill_z_type)

           
            if self.z_all:
                x_all = self.decoder.decoder(actions_z)
            else:
                self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
                x_all = self.decoder.decoder(torch.cat([self_obs, actions_z], dim = -1))
                
                # z_prior_out = self.decoder.z_prior(self_obs); prior_mu, prior_log_var = self.decoder.z_prior_mu(z_prior_out), self.decoder.z_prior_logvar(z_prior_out); print(prior_mu.max(), prior_mu.min())
                # print('....')
                
            actions = x_all

        # actions = x_all[:, 3]  # Debugging

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if flags.server_mode:
            dt = time.time() - t_s
            print(f'\r {1/dt:.2f} fps', end='')
            
        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

