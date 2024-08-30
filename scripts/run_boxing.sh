export OMP_NUM_THREADS=1
python phc/run_hydra.py     project_name=SMPLOlympics num_agents=2     \
        learning=amp_z_self_play exp_name=boxing_pulse \
        env=env_amp_z env.num_envs=2048 env.task=HumanoidBoxingZ env.enableTaskObs=True  \
        env.stateInit=Start    robot=smpl_humanoid_boxing +env.models=["output/HumanoidIm/pulse_vae_iclr/Humanoid.pth"]      \
        env.motion_file=./sample_data/video_boxing_afterproc_upright.pkl headless=True env.episode_length=300  learning.params.config.switch_frequency=250


