export OMP_NUM_THREADS=1
python phc/run_hydra.py     project_name=SMPLOlympics num_agents=2     \
        learning=amp_z_self_play exp_name=fencing_pulse \
        env=env_amp_z env.num_envs=2048 env.task=HumanoidFencingZ env.enableTaskObs=True  \
        env.stateInit=Start    robot=smpl_humanoid_fencing +env.models=["output/HumanoidIm/pulse_vae/Humanoid.pth"]      \
        env.motion_file=./sample_data/fencing_all.pkl headless=True env.episode_length=300  learning.params.config.switch_frequency=250


