python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=ppo exp_name=highjump_ppo  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidHighjump env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl headless=True env.stateInit=Default \
    env.episode_length=300 learning.params.config.max_epochs=20000  

python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=amp exp_name=highjump_ppo  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidHighjump env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl headless=True env.stateInit=Default \
    env.episode_length=300 learning.params.config.max_epochs=20000  

python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse exp_name=highjump_pulse  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidHighjumpZ env.enableTaskObs=True \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl headless=True env.stateInit=Default \
    env.episode_length=300 learning.params.config.max_epochs=20000  