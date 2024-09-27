export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=ppo exp_name=javelin_ppo \
    env=env_x_amp_z env.task=HumanoidJavelin  robot=smplx_humanoid  env.enableTaskObs=True \
    env.motion_file=sample_data/javelin_x.pkl env.num_envs=1024 \
    env.episode_length=300 +env.randomrize_heading=False sim=hand_sim learning.params.config.max_epochs=20000

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=amp exp_name=javelin_amp \
    env=env_x_amp_z env.task=HumanoidJavelin  robot=smplx_humanoid  env.enableTaskObs=True \
    env.motion_file=sample_data/javelin_x_amp.pkl env.num_envs=1024 \
    env.episode_length=300 +env.randomrize_heading=False sim=hand_sim learning.params.config.max_epochs=20000

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=pulse exp_name=javelin_pulse \
    env=env_x_amp_z env.task=HumanoidJavelinZ  robot=smplx_humanoid  env.enableTaskObs=True \
    env.motion_file=sample_data/javelin_x.pkl env.num_envs=1024 \
    env.episode_length=300 +env.randomrize_heading=False sim=hand_sim learning.params.config.max_epochs=20000

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=4 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=pulse_amp exp_name=javelin_pulse_amp \
    env=env_x_amp_z env.task=HumanoidJavelinZ  robot=smplx_humanoid env.enableTaskObs=True  \
    env.motion_file=sample_data/javelin_x_amp.pkl env.num_envs=1024 \
    env.episode_length=300 +env.randomrize_heading=False  sim=hand_sim learning.params.config.max_epochs=20000



