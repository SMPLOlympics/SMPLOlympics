export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=ppo exp_name=hurdle_ppo  \
    env=env_amp env.num_envs=2048 env.task=HumanoidHurdle env.enableTaskObs=True env.shape_resampling_interval=1000 \
    robot=smpl_humanoid  robot.has_upright_start=True no_log=True learning.params.config.max_epochs=10000 \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl headless=True env.stateInit=Default env.episode_length=600

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=amp exp_name=hurdle_amp  \
    env=env_amp env.num_envs=2048 env.task=HumanoidHurdle env.enableTaskObs=True env.shape_resampling_interval=1000 \
    robot=smpl_humanoid  robot.has_upright_start=True no_log=True learning.params.config.max_epochs=10000 \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl headless=True env.stateInit=Default env.episode_length=600

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse exp_name=hurdle_pulse  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidHurdleZ env.enableTaskObs=True env.shape_resampling_interval=1000 \
    robot=smpl_humanoid  robot.has_upright_start=True no_log=True learning.params.config.max_epochs=10000 \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl headless=True env.stateInit=Default env.episode_length=600



#### H1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=ppo exp_name=hurdle_ppo  \
    env=env_amp_h1 env.num_envs=2048 env.task=HumanoidHurdle env.enableTaskObs=True env.shape_resampling_interval=1000 \
    robot=unitree_h1_nohead   learning.params.config.max_epochs=10000 \
    env.motion_file=./sample_data/h1_simplerun.pkl headless=True env.stateInit=Default env.episode_length=600


python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse exp_name=hurdle_pulse  \
    env=env_amp_z_h1 env.num_envs=2048 env.task=HumanoidHurdleZ env.enableTaskObs=True env.shape_resampling_interval=1000 \
    robot=unitree_h1_nohead   learning.params.config.max_epochs=10000 \
    env.motion_file=./sample_data/h1_simplerun.pkl headless=True env.stateInit=Default env.episode_length=600
