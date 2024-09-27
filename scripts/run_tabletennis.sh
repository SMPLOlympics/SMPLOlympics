export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=ppo exp_name=tabletennis__ppo  \
    env=env_amp env.num_envs=2048 env.task=HumanoidPP env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_pp  robot.has_upright_start=True env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default no_log=True \
    learning.params.config.max_epochs=100000

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=amp exp_name=tabletennis_amp  \
    env=env_amp env.num_envs=2048 env.task=HumanoidPP env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_pp  robot.has_upright_start=True env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/pingpong1after_upright.pkl \
    headless=True env.stateInit=Default env.numAMPObsSteps=10 no_log=True \
    learning.params.config.max_epochs=100000


export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse exp_name=tabletennis_pulse  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidPPZ env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_pp  robot.has_upright_start=True env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True env.stateInit=Default no_log=True \
    learning.params.config.max_epochs=100000

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=pulse_amp exp_name=tabletennis_pulse_amp  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidPPZ env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_pp  robot.has_upright_start=True env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/pingpong1after_upright.pkl \
    headless=True env.stateInit=Default env.numAMPObsSteps=10 no_log=True \
    learning.params.config.max_epochs=100000


# Finetuning two-person play
export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=2 \
    learning=pulse_amp exp_name=tabletennis_pulse_amp  \
    env=env_amp_z env.num_envs=512 env.task=HumanoidPP2Z env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"] \
    robot=smpl_humanoid_pp  robot.has_upright_start=True env.numAMPObsSteps=10 env.shape_resampling_interval=500000 \
    env.motion_file=./sample_data/pingpong1after_upright.pkl headless=True env.stateInit=Default epoch=70000 no_log=True