import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_games.algos_torch import torch_ext
import cv2
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


upright_start = False
print("upright_start", upright_start)
print("upright_start", upright_start)
print("upright_start", upright_start)

robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, 
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False, 
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
    }

smpl_local_robot = LocalRobot(robot_cfg,)
data_load = joblib.load("sample_data/basketball_all_x.pkl")
# data_load = joblib.load("sample_data/standing_x.pkl")
data_full_dump = {}
pbar = tqdm(data_load.items())

beta = np.zeros((16))
gender_number, beta[:], gender = [0], 0, "neutral"
print("using neutral model")
smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")

for data_key, entry_data in pbar:
    pose_aa = entry_data['pose_aa']
    root_trans = entry_data['trans_orig']
    betas = entry_data['beta']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    
    
    ########################################  Fixed data overwrite
    # pose_aa = pose_aa[:3]
    # root_trans = root_trans[:3]
    
    # pose_aa[:, :3] = (sRot.from_euler("xyz", [0, 0, np.pi/2]) * sRot.from_rotvec(pose_aa[0, :3])).as_rotvec()
    # root_trans[:, :2] = 0
    # pose_aa = pose_aa.reshape(3, -1, 3)
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("R_Elbow")] = sRot.from_euler("xyz", [0, np.pi/2, 0]).as_rotvec() # jv
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("R_Shoulder")] = sRot.from_euler("xyz", [-np.pi/2, np.pi/3, 0]).as_rotvec() # jv
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("R_Thumb1")] = sRot.from_euler("xyz", [0, np.pi/4, 0]).as_rotvec() # jv
    
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("R_Elbow")] = sRot.from_euler("xyz", [0, np.pi/2, 0]).as_rotvec() # bb
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("R_Shoulder")] = sRot.from_euler("xyz", [-np.pi/3, np.pi/3, 0]).as_rotvec() # bb
    
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("L_Elbow")] = sRot.from_euler("xyz", [0, -np.pi/2, 0]).as_rotvec() # bb
    # pose_aa[:, SMPLH_BONE_ORDER_NAMES.index("L_Shoulder")] = sRot.from_euler("xyz", [-np.pi/3, -np.pi/3, 0]).as_rotvec() # bb
    ########################################
    
    root_trans = root_trans
    pose_aa = pose_aa
    N = pose_aa.shape[0]

    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 52, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
    

    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)
    
    if robot_cfg['upright_start']:
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()


    pose_quat_global = new_sk_state.global_rotation.numpy()
    pose_quat = new_sk_state.local_rotation.numpy()
    fps = 30

    new_motion_out = {}
    new_motion_out['pose_quat_global'] = pose_quat_global
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['trans_orig'] = root_trans
    new_motion_out['root_trans_offset'] = root_trans_offset
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = gender
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['fps'] = fps

    data_full_dump[data_key] = new_motion_out
    
print(f"Total number of sequences {len(data_full_dump)}")
import ipdb; ipdb.set_trace()
# joblib.dump(data_full_dump, "sample_data/javelin_x.pkl", compress=True)
joblib.dump(data_full_dump, "sample_data/basketball_all_x.pkl", compress=True)
