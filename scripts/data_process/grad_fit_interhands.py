import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
import joblib
from smpl_sim.smpllib.torch_smpl_humanoid_batch import Humanoid_Batch
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from collections import defaultdict

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES

from smpl_sim.smpllib.smpl_parser import (
    SMPLX_Parser,
)
from tqdm import tqdm
from smplx import MANO


device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )



hb = Humanoid_Batch(smpl_model='smplx')
dump_data = defaultdict(dict)
data_split = "test"
# pkl_files = sorted(glob.glob("output/dgx/smpl_im_z_fut_kp_sparse_1/adt/*"))
# pkl_files = pkl_files[50:75]
mano_data = joblib.load("data/reinterhand/test.pkl")
standing_pose = joblib.load(f"data/amass_x/singles/standing.pkl")['standing']

smplx_parser_n = SMPLX_Parser(model_path="data/smpl", gender="neutral").to(device)

layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
mano_layer_right = MANO(model_path = "data/smpl", is_rhand=True, use_pca=False, flat_hand_mean=True, **layer_arg)
mano_layer_left = MANO(model_path = "data/smpl", is_rhand=False, use_pca=False, flat_hand_mean=True, **layer_arg)

# right_output = mano_layer_right(betas=torch.zeros((1, 10)), hand_pose=torch.zeros((1, 45)), global_orient=torch.zeros((1, 3)), transl=torch.zeros((1, 3)))
# left_output = mano_layer_left(betas=torch.zeros((1, 10)), hand_pose=torch.zeros((1, 45)), global_orient=torch.zeros((1, 3)), transl=torch.zeros((1, 3)))
# left_hand_offset = left_output.joints[:, 0]
# right_hand_offset = right_output.joints[:, 0]


mujoco_hand_joints = ['L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
smpl_hand_joints = ["L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3","L_Middle1","L_Middle2","L_Middle3","L_Pinky1","L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",]

# smpl_2_mujoco_hands = [smpl_hand_joints.index(q) for q in mujoco_hand_joints]

smpl_joint_pick_idx = [SMPLH_MUJOCO_NAMES.index(q) for q in mujoco_hand_joints]
# smpl_joint_pick_idx = [SMPLH_MUJOCO_NAMES.index(q) for q in ["L_Wrist", "R_Wrist"]]
wrist_joints = [SMPLH_MUJOCO_NAMES.index(q) for q in ["L_Wrist", "R_Wrist"]]
optimizing_joint_pick_idx = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_BONE_ORDER_NAMES if not q in ["L_Elbow", "R_Elbow",  "L_Wrist", "R_Wrist"] ]

eps = 1e-7
for seq_key, data_entry in mano_data.items():

    data_entry = {k: torch.from_numpy(v).float() for k, v in data_entry.items()}
    left_pose = data_entry['left_pose']
    right_pose = data_entry['right_pose']
    left_shape = data_entry['left_shape']
    right_shape = data_entry['right_shape']
    left_trans = data_entry['left_trans']
    right_trans = data_entry['right_trans']
    left_root_pose = data_entry['left_root_pose']
    right_root_pose = data_entry['right_root_pose']
    
    ###### Rotating the Interhand Data ########
    head_up_rot = sRot.from_euler("xyz", [-np.pi/2, 0, 0])
    left_root_pose = torch.from_numpy((head_up_rot * sRot.from_rotvec(left_root_pose)).as_rotvec()).float()
    right_root_pose = torch.from_numpy((head_up_rot * sRot.from_rotvec(right_root_pose)).as_rotvec()).float()
    left_height, right_height = left_trans[0, :], right_trans[:, 2]
    left_trans = (left_trans @ head_up_rot.as_matrix().T).float()
    right_trans = (right_trans @ head_up_rot.as_matrix().T).float()
    diff = (left_trans[0, :] - left_height)
    diff[2] -= 0.15
    left_trans = left_trans - diff
    right_trans = right_trans - diff
    ###### Rotating the Interhand Data ########

    with torch.no_grad():
        # left_output = mano_layer_left(betas=left_shape, hand_pose=left_pose, global_orient=left_root_pose, transl=left_trans)
        # right_output = mano_layer_right(betas=right_shape, hand_pose=right_pose, global_orient=right_root_pose, transl=right_trans)
        left_output = mano_layer_left(betas=torch.zeros_like(left_shape), hand_pose=left_pose, global_orient=left_root_pose, transl=left_trans) # using mean shape
        right_output = mano_layer_right(betas=torch.zeros_like(right_shape), hand_pose=right_pose, global_orient=right_root_pose, transl=right_trans)
    
    
    
    N = left_pose.shape[0]
    pose_aa = torch.from_numpy(standing_pose['pose_aa'][0:1].repeat(N, axis = 0)).float()
    trans = torch.from_numpy(standing_pose['trans_orig'][0:1].repeat(N, axis = 0)).float()
    starting_pos = (left_trans + right_trans)[0]/2
    
    trans[:, :2] = starting_pos[:2]
    trans[:, 1] += 0.6                                ##### Hack 
    trans[:, 0] += 0.2                             ##### Hack 
    
    hb.update_model(torch.zeros([1, 10]).to(device), torch.tensor([0]).to(device))
    
    pose_aa[:, -90:] = torch.cat([left_pose.float(), right_pose.float()], dim = -1)
    
    pose_aa_torch_new = Variable(pose_aa.reshape(N, -1, 3).clone(), requires_grad=True)
    
    # optimizer_pose = torch.optim.Adadelta([pose_aa_torch_new],lr=500)
    optimizer_pose = torch.optim.Adam([pose_aa_torch_new],lr=0.01)
    
    scheduler_pose = StepLR(optimizer_pose, step_size=1, gamma=0.9999)
    num_epochs = 100
    print(f"processing: {seq_key}")
    pbar = tqdm(range(num_epochs))
    
    
    # hands_joints_gt = torch.cat([left_output.joints[:, 0:1], right_output.joints[:, 0:1]], dim = 1).float()
    hands_joints_gt = torch.cat([left_output.joints, right_output.joints], dim = 1).float()
    

    head_rots_gt_mat = torch.from_numpy(np.stack([sRot.from_rotvec(left_root_pose).as_matrix(), sRot.from_rotvec(right_root_pose).as_matrix()], axis = 1)).float()
    
    for i in pbar:
        fk_res = hb.fk_batch(pose_aa_torch_new[:, None], trans[:, None])
        fk_pos = fk_res['wbpos'].squeeze() # Mujoco order
        fk_rot = fk_res['wbmat'].squeeze() # Mujoco order
        
        diff = fk_pos[:, smpl_joint_pick_idx] - hands_joints_gt
        
        loss_g = diff.norm(dim = -1).mean() *1000
        
        R_diffs = fk_res['wbmat'].squeeze()[:, wrist_joints] @ head_rots_gt_mat.permute(0, 1, 3, 2)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
        loss_rot = dists.mean()
        loss = loss_g + loss_rot
        print_str = f"loss_g: {loss_g.item() :.3f} loss_rot: {loss_rot.item() :.3f}  Epoch: {i} LR pose: {scheduler_pose.get_last_lr()[0]:.3f}"

        # loss = loss_g
        # print_str = f"loss_g: {loss_g.item() :.3f}   Epoch: {i} LR pose: {scheduler_pose.get_last_lr()[0]:.3f}"
        
        if i > 200:
            optimizing_joint_pick_idx = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_BONE_ORDER_NAMES if not q in ["L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder"] + mujoco_hand_joints]

        pbar.set_description_str(print_str)
        
        optimizer_pose.zero_grad()
        loss.backward()
        pose_aa_torch_new.grad[:, optimizing_joint_pick_idx] = 0 # zero out these grad, not changing them
        optimizer_pose.step()
        scheduler_pose.step()
        
        
    dump_data[seq_key]["pose_aa"] = pose_aa_torch_new.detach().cpu().numpy()
    dump_data[seq_key]["trans_orig"] = trans.detach().cpu().numpy()
    
    dump_data[seq_key]["beta"] = standing_pose['beta']
    dump_data[seq_key]["fps"] = standing_pose['fps']
    dump_data[seq_key]["gender"] = standing_pose['gender']
    
    
joblib.dump(dump_data, f"data/reinterhand/fitted/test.pkl")
import ipdb; ipdb.set_trace()
print('...')


        