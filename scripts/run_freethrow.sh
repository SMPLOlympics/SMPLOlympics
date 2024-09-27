python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=ppo exp_name=freethrow_ppo \
    env=env_x_amp_z env.task=HumanoidBasketballShoot  robot=smplx_humanoid env.enableTaskObs=True  \
    env.motion_file=sample_data/basketball_all_x.pkl env.num_envs=1024  env.shape_resampling_interval=500000  +env.has_basketball_bound=False env.power_reward=True \
    env.episode_length=300 +env.randomrize_heading=False  sim=hand_sim learning.params.config.max_epochs=20000 


python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=amp exp_name=freethrow_amp \
    env=env_x_amp_z env.task=HumanoidBasketballShoot  robot=smplx_humanoid env.enableTaskObs=True  \
    env.motion_file=sample_data/basketball_all_x.pkl env.num_envs=1024  env.shape_resampling_interval=500000  +env.has_basketball_bound=False env.power_reward=True \
    env.episode_length=300 +env.randomrize_heading=False  sim=hand_sim learning.params.config.max_epochs=20000 


python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=pulse exp_name=freethrow_pulse \
    env=env_x_amp_z env.task=HumanoidBasketballShootZ  robot=smplx_humanoid env.enableTaskObs=True  \
    env.motion_file=sample_data/basketball_all_x.pkl env.num_envs=1024  env.shape_resampling_interval=500000 +env.has_basketball_bound=False env.power_reward=True  \
    env.episode_length=300 +env.randomrize_heading=False  sim=hand_sim learning.params.config.max_epochs=20000 

python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=pulse_amp exp_name=freethrow_pulse_amp \
    env=env_x_amp_z env.task=HumanoidBasketballShootZ  robot=smplx_humanoid env.enableTaskObs=True  \
    env.motion_file=sample_data/basketball_all_x.pkl env.num_envs=1024  env.shape_resampling_interval=500000 +env.has_basketball_bound=False env.power_reward=True  \
    env.episode_length=300 +env.randomrize_heading=False  sim=hand_sim learning.params.config.max_epochs=20000 