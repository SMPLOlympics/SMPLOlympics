export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=1 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=ppo exp_name=golf_ppo env=env_amp_golf \
    env.num_envs=256 env.task=HumanoidGolf \
    env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"]  robot=smpl_humanoid_golf \
    robot.has_upright_start=True env.shape_resampling_interval=500000  env.motion_file=./sample_data/golf_after_1_upright.pkl \
    headless=True robot.real_weight_porpotion_boxes=False  env.plane.restitution=0.6  env.stateInit=Default \
    env.numAMPObsSteps=10 env.episode_length=300 learning.params.config.max_epochs=30000
#
##
#
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=amp exp_name=golf_amp env=env_amp_golf \
    env.num_envs=256 env.task=HumanoidGolf \
    env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"]  robot=smpl_humanoid_golf \
    robot.has_upright_start=True env.shape_resampling_interval=500000  env.motion_file=./sample_data/golf_after_1_upright.pkl \
    headless=True robot.real_weight_porpotion_boxes=False  env.plane.restitution=0.6  env.stateInit=Default \
    env.numAMPObsSteps=10 env.episode_length=300 learning.params.config.max_epochs=30000
#
#
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=3 python phc/run_hydra.py \
project_name=SMPLOlympics num_agents=1 learning=pulse exp_name=golf_pulse env=env_amp_z_golf \
env.num_envs=256 env.task=HumanoidGolfZ   env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"]  robot=smpl_humanoid_golf \
  robot.has_upright_start=True env.shape_resampling_interval=500000  env.motion_file=./sample_data/golf_after_1_upright.pkl \
  headless=True robot.real_weight_porpotion_boxes=False  env.plane.restitution=0.6  env.stateInit=Default \
  env.numAMPObsSteps=10 env.episode_length=300 learning.params.config.max_epochs=30000
 

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=1 python phc/run_hydra.py \
    project_name=SMPLOlympics num_agents=1 learning=pulse_amp exp_name=golf_pulse_amp env=env_amp_z_golf \
    env.num_envs=256 env.task=HumanoidGolfZ \
    env.enableTaskObs=True +env.contact_bodies=["R_Ankle","L_Ankle","R_Toe","L_Toe","R_Hand"]  robot=smpl_humanoid_golf \
    robot.has_upright_start=True env.shape_resampling_interval=500000  env.motion_file=./sample_data/golf_after_1_upright.pkl \
    headless=True robot.real_weight_porpotion_boxes=False  env.plane.restitution=0.6  env.stateInit=Default \
    env.numAMPObsSteps=10 env.episode_length=300 learning.params.config.max_epochs=30000