export OMP_NUM_THREADS=1
python phc/run_hydra.py project_name=SMPLOlympics num_agents=1 \
      learning=ppo exp_name=kick_ppo \
      env=env_amp_z env.num_envs=2048 env.task=HumanoidSoccerPenaltyKick \
      env.enableTaskObs=True +env.strikeBodyNames=[L_Knee,L_Ankle,L_Toe] env.plane.restitution=0.8 \
      robot=smpl_humanoid robot.has_upright_start=True env.shape_resampling_interval=500000 env.motion_file=sample_data/video_football_afterproc_upright.pkl \
      headless=True env.episode_length=120 

export OMP_NUM_THREADS=1
python phc/run_hydra.py project_name=SMPLOlympics num_agents=1 \
      learning=amp exp_name=kick_amp \
      env=env_amp_z env.num_envs=2048 env.task=HumanoidSoccerPenaltyKick \
      env.enableTaskObs=True +env.strikeBodyNames=[L_Knee,L_Ankle,L_Toe] env.plane.restitution=0.8 \
      robot=smpl_humanoid robot.has_upright_start=True env.shape_resampling_interval=500000 env.motion_file=sample_data/video_football_afterproc_upright.pkl \
      headless=True env.episode_length=120 learning.params.config.task_reward_w=0.8  learning.params.config.disc_reward_w=0.2 no_log=True

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=no_amp_z exp_name=kick_pulse  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidSoccerPenaltyKickZ env.enableTaskObs=True +env.strikeBodyNames=["L_Knee","L_Ankle","L_Toe"] env.plane.restitution=0.8 \
    robot=smpl_humanoid  robot.has_upright_start=True  \
    env.motion_file=./sample_data/video_football_afterproc_upright.pkl headless=True env.episode_length=120

export OMP_NUM_THREADS=1
python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 \
    learning=amp_z exp_name=kick_pulse_amp  \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidSoccerPenaltyKickZ env.enableTaskObs=True +env.strikeBodyNames=["L_Knee","L_Ankle","L_Toe"] env.plane.restitution=0.8 \
    robot=smpl_humanoid robot.has_upright_start=True env.shape_resampling_interval=500000 env.motion_file=sample_data/video_football_afterproc_upright.pkl \
    headless=True env.episode_length=120 learning.params.config.task_reward_w=0.8  learning.params.config.disc_reward_w=0.2 no_log=True


