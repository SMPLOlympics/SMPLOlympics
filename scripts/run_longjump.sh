export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=ppo exp_name=longjump_ppo  \
    env=env_amp_z_longjump env.num_envs=2048 env.task=HumanoidLongjump env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default env.episode_length=600 learning.params.config.max_epochs=100000


export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=amp exp_name=longjump_amp  \
    env=env_amp_z_longjump env.num_envs=2048 env.task=HumanoidLongjumpZ env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default env.episode_length=600 learning.params.config.max_epochs=100000

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse exp_name=longjump_pulse  \
    env=env_amp_z_longjump env.num_envs=2048 env.task=HumanoidLongjumpZ env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default env.episode_length=600 learning.params.config.max_epochs=100000

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse_amp exp_name=longjump_pulse_amp  \
    env=env_amp_z_longjump env.num_envs=2048 env.task=HumanoidLongjumpZ env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default env.episode_length=600 learning.params.config.max_epochs=100000