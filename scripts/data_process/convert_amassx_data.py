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

smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/smpl/smplx_amass")
all_pkls = glob.glob("/hdd/zen/data/ActBound/AMASS/AMASS_X_G_Complete/**/*.npz", recursive=True)
amass_occlusion = joblib.load("/hdd/zen/data/ActBound/AMASS/amassx_occlusion_v1.pkl")
# raw_data_entry = "/hdd/zen/data/ActBound/AMASS/AMASS_X_G_download/ACCAD/Male2MartialArtsKicks_c3d/G12-__cresent_left_stageii.npz"
amass_full_motion_dict = {}
pbar = tqdm(all_pkls)

beta = np.zeros((16))
gender_number, beta[:], gender = [0], 0, "neutral"
print("using neutral model")
smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")

for data_path in pbar:
    bound = 0
    
    key_name_dump = "_".join(data_path.split("/")[7:]).replace(".npz", "")
    # if not "KIT".lower() in key_name_dump.lower():
        # continue
    
    if key_name_dump in amass_occlusion:
        issue = amass_occlusion[key_name_dump]["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
            bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
            if bound < 10:
                pbar.set_description_str(f"bound too small {key_name_dump}, {bound}")
                continue
        else:
            pbar.set_description_str(f"issue irrecoverable, {key_name_dump}, {issue}")
            continue
    
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if not 'mocap_frame_rate' in  entry_data:
        continue
    framerate = entry_data['mocap_frame_rate']

    if "totalcapture" in key_name_dump.lower() or "ssm" in key_name_dump.lower():
        framerate = 60 # total capture's framerate is wrong. 
    elif "KIT".lower() in key_name_dump.lower():
        framerate = 100 # KIT's framerate
    elif "CMU".lower() in key_name_dump.lower():
        orig_file = data_path
        orig_file = orig_file.replace("AMASS_X_G_Complete", "AMASS_Complete")
        orig_file = orig_file.replace("stageii", "poses")
        if osp.isfile(orig_file):
            entry_data_orig = dict(np.load(open(orig_file, "rb"), allow_pickle=True))
            if (entry_data['mocap_frame_rate'] !=  entry_data_orig['mocap_framerate']):
                framerate = entry_data_orig['mocap_framerate']
        else:
            import ipdb; ipdb.set_trace()
            print('.....')
            
    skip = int(np.floor(framerate/30))
    pose_aa = np.concatenate([entry_data['poses'][::skip, :66], entry_data['poses'][::skip, 75:]], axis = -1)
    root_trans = entry_data['trans'][::skip, :]
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    
    if bound == 0:
        bound = N
            
    root_trans = root_trans[:bound]
    pose_aa = pose_aa[:bound]
    N = pose_aa.shape[0]
    if N < 10:
        continue

    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 52, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
    
    # gender_number, beta[:], gender = [0], 0, "neutral" # For updating bodies
    # print("using neutral model")
    # smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
    # smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    # skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")

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
    amass_full_motion_dict[key_name_dump] = new_motion_out
    
print(f"Total number of sequences {len(amass_full_motion_dict)}")
import ipdb; ipdb.set_trace()
# joblib.dump(amass_full_motion_dict, "data/amass_x/singles/cmu_patch.pkl", compress=True)
joblib.dump(amass_full_motion_dict, "data/amass_x/amass_clean_v1.pkl", compress=True)
# joblib.dump(amass_full_motion_dict, "data/amass_x/upright/amass_clean_upright_v1.pkl", compress=True)