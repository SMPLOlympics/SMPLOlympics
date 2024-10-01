export OMP_NUM_THREADS=1
python phc/run_hydra.py     project_name=SMPLOlympics num_agents=2     \
        learning=amp_z_self_play exp_name=boxing_pulse \
        env=env_amp_z env.num_envs=2048 env.task=HumanoidBoxingZ env.enableTaskObs=True  \
        env.stateInit=Start    robot=smpl_humanoid_boxing +env.models=["output/HumanoidIm/pulse_vae_iclr/Humanoid.pth"]      \
        env.motion_file=./sample_data/video_boxing_afterproc_upright.pkl headless=True env.episode_length=300  learning.params.config.switch_frequency=250


python phc/run_hydra.py     project_name=SMPLOlympics num_agents=2     \
        learning=amp_z_self_play exp_name=h1_boxing_pulse \
        env=env_amp_z_h1 env.num_envs=2048 env.task=HumanoidBoxingZ env.enableTaskObs=True  \
        env.stateInit=Start    robot=unitree_h1_nohead      \
        env.motion_file=./sample_data/h1_simplerun.pkl headless=True \
        +env.strikeBodyNames=['left_knee_link','left_ankle_link','right_knee_link','right_ankle_link'] +env.footNames=['left_ankle_link','right_ankle_link'] \
        +env.handNames=['left_elbow_link','right_elbow_link'] +env.targetNames=['torso_link','pelvis']
