import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from phc.utils.flags import flags
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from phc.learning.world_model import WorldModel
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_pnn, load_mcp_mlp
from phc.env.tasks.humanoid import compute_humanoid_observations_smpl_max
from phc.env.tasks.humanoid_im import compute_imitation_observations_v6
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import rescale_actions

import torch
import joblib
from easydict import EasyDict
import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as sRot

def reset_func(action):
    global rigid_body_pos, rigid_body_rot, rigid_body_vel, rigid_body_ang_vel, dof_pos, dof_vel, motion_res
    ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
    rigid_body_pos[:] = ref_rb_pos
    rigid_body_rot[:] = ref_rb_rot
    rigid_body_vel[:] = ref_body_vel
    rigid_body_ang_vel[:] = ref_body_ang_vel
    dof_pos[:] = ref_dof_pos
    dof_vel[:] = ref_dof_pos

def next_func(action):
    global motion_lib, start_idx, num_motions, skeleton_trees, gender_beta, num_motions
    start_idx += 1
    motion_lib.load_motions(skeleton_trees=skeleton_trees, 
                        gender_betas=[torch.from_numpy(gender_beta)] * num_motions,
                        limb_weights=[np.zeros(10)] * num_motions,
                        random_sample=False,
                        start_idx = start_idx, 
                        max_len=-1)
    
    


motion_file = "/hdd/zen/dev/meta/phc/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl"
device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
dt = 1/30
time_steps = 1
has_upright_start = True
motion_lib_cfg = EasyDict({
                    "has_upright_start": has_upright_start,
                    "motion_file": motion_file,
                    "fix_height": FixHeightMode.full_fix,
                    "min_length": -1,
                    "max_length": -1,
                    "im_eval": flags.im_eval,
                    "multi_thread": False ,
                    "smpl_type": 'smpl',
                    "randomrize_heading": True,
                    "device": device,
                })
robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": has_upright_start,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smpl",
        "big_ankle": True, 
        "box_body": True, 
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }

smpl_robot = SMPL_Robot(
    robot_cfg,
    data_dir="data/smpl",
)
    
gender_beta = np.zeros((17))
smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
test_good = f"/tmp/smpl/test_good.xml"
smpl_robot.write_xml(test_good)
sk_tree = SkeletonTree.from_mjcf(test_good)
num_motions = 1
skeleton_trees = [sk_tree] * num_motions
world_config_dict = EasyDict({
            "num_dofs": 69, 
            "num_rigid_bodies": 24,
            "dim_in": 565, # max coordinates self obs + dof pos and dof_vel and actions
            "upright": has_upright_start, 
            "device": device,
            "skeleton_trees": skeleton_trees,
        })



#################### Load World Model and Motion Lib ####################
start_idx = 0
world_model = WorldModel(world_config_dict)
world_model.load_state_dict(torch.load("output/HumanoidIm/phc_3_world/world_model_0000010500.pth"))
motion_lib = MotionLibSMPL(motion_lib_cfg)
motion_lib.load_motions(skeleton_trees=skeleton_trees, 
                        gender_betas=[torch.from_numpy(gender_beta)] * num_motions,
                        limb_weights=[np.zeros(10)] * num_motions,
                        random_sample=False,
                        start_idx = start_idx, 
                        max_len=-1)
motion_id, time_step = torch.zeros(1).to(device).long(), torch.zeros(1).to(device).float()
motion_len = motion_lib.get_motion_length(motion_id).item()
motion_time = time_step % motion_len


############## Load Policy  ####################
policy_path = "output/HumanoidIm/phc_3_world/Humanoid_00010500.pth"
check_points = [torch_ext.load_checkpoint(policy_path)]
pnn = load_pnn(check_points[0], num_prim = 3, has_lateral = False, activation = "silu", device = device)
running_mean, running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
action_offset = joblib.load("action_offset.pkl")
pd_action_offset = action_offset[0]
pd_action_scale = action_offset[1]


######################  Visualizer  ######################
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
o3d_vis.create_window()

box = o3d.geometry.TriangleMesh()
ground_size, height = 5, 0.01
box = box.create_box(width=ground_size, height=height, depth=ground_size)
box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
box.compute_vertex_normals()
box.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.1, 0.1]]).repeat(8, axis=0))
mujoco_2_smpl = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]
with torch.no_grad():
    verts, joints = motion_lib.mesh_parsers[0].get_joints_verts(pose = torch.zeros(1, len(SMPL_MUJOCO_NAMES) * 3))
np_triangles = motion_lib.mesh_parsers[0].faces
pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
mesh_parser = copy.deepcopy(motion_lib.mesh_parsers[0])
mesh_parser = mesh_parser.cuda()

sim_mesh = o3d.geometry.TriangleMesh()
sim_mesh.vertices = o3d.utility.Vector3dVector(verts.numpy()[0])
sim_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
sim_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0, 0.5, 0.5]]).repeat(verts.shape[1], axis=0))
ref_mesh = o3d.geometry.TriangleMesh()
ref_mesh.vertices = o3d.utility.Vector3dVector(verts.numpy()[0])
ref_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
ref_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.5, 0., 0.]]).repeat(verts.shape[1], axis=0))
o3d_vis.add_geometry(ref_mesh)
o3d_vis.add_geometry(sim_mesh)
o3d_vis.add_geometry(box)
coord_trans = torch.from_numpy(sRot.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()).float().cuda()


# o3d_vis.register_key_callback(32, self.pause_func) # space
o3d_vis.register_key_callback(82, reset_func) # R
# o3d_vis.register_key_callback(76, self.record_func) # L
o3d_vis.register_key_callback(84, next_func) # T
# o3d_vis.register_key_callback(75, self.hide_ref) # K

######################  Simulation Properties  ######################
rigid_body_pos = torch.zeros(1, 24, 3).cuda()
rigid_body_rot = torch.zeros(1, 24, 4).cuda()
rigid_body_vel = torch.zeros(1, 24, 3).cuda()
rigid_body_ang_vel = torch.zeros(1, 24, 3).cuda()
dof_pos = torch.zeros(1, 69).cuda()
dof_vel = torch.zeros(1, 69).cuda()


motion_res = motion_lib.get_motion_state(motion_id, motion_time)
ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
rigid_body_pos[:] = ref_rb_pos
rigid_body_rot[:] = ref_rb_rot
rigid_body_vel[:] = ref_body_vel
rigid_body_ang_vel[:] = ref_body_ang_vel
dof_pos[:] = ref_dof_pos
dof_vel[:] = ref_dof_pos


while True:
    ################ Sampling reference motion ################
    motion_time = time_step % motion_len
    motion_res = motion_lib.get_motion_state(motion_id, motion_time)
    ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

    body_pos = rigid_body_pos
    body_rot = rigid_body_rot
    body_vel = rigid_body_vel
    body_ang_vel = rigid_body_ang_vel
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    body_pos_subset = body_pos
    body_rot_subset = body_rot
    body_vel_subset = body_vel
    body_ang_vel_subset = body_ang_vel
    ref_rb_pos_subset = ref_rb_pos
    ref_rb_rot_subset = ref_rb_rot
    ref_body_vel_subset = ref_body_vel
    ref_body_ang_vel_subset = ref_body_ang_vel
    
    self_obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, ref_smpl_params, ref_limb_weights, True, True, has_upright_start, False, False)
    task_obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, has_upright_start)

    full_obs = torch.cat([self_obs, task_obs], dim = -1)
    full_obs = ((full_obs - running_mean.float()) / torch.sqrt(running_var.float() + 1e-05))
    full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)

    with torch.no_grad():
        actions, _ = pnn(full_obs, idx=0)
        actions = rescale_actions(-1, 1, torch.clamp(actions, -1, 1))
        actions = actions * pd_action_scale + pd_action_offset
    
    world_model_dict = EasyDict({
        "rg_pos": rigid_body_pos[None, ],
        "rg_rot": rigid_body_rot[None, ],
        "rg_vel": rigid_body_vel[None, ],
        "rg_ang_vel": rigid_body_ang_vel[None, ],

        "dof_pos": dof_pos[None, ],
        "dof_vel": dof_vel[None, ],
    })
    world_model_dict['self_obs_and_action'] = torch.cat([self_obs, dof_pos, dof_vel, actions], dim=-1)[None, ]
    with torch.no_grad():
        next_state = world_model.forward(world_model_dict)
    
    rigid_body_pos = next_state.body_pos[0]
    rigid_body_rot = next_state.body_quat[0]
    rigid_body_vel = next_state.lin_vel[0]
    rigid_body_ang_vel = next_state.ang_vel[0]
    dof_pos = next_state.dof_pos[0].reshape(1, -1)
    dof_vel = next_state.dof_vel[0].reshape(1, -1)
    
                    
    ################# Simulation Values #################
    body_quat = rigid_body_rot
    root_trans = rigid_body_pos[:, 0, :]
    offset = skeleton_trees[0].local_translation[0].cuda()
    
    ref_body_quat = motion_res['rb_rot']
    ref_root_trans = motion_res['root_pos']
    body_quat = torch.cat([body_quat, ref_body_quat])
    root_trans = torch.cat([root_trans, ref_root_trans])
    
    
    N = body_quat.shape[0]
    pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * pre_rot).as_quat().reshape(N, -1, 4)
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[0], torch.from_numpy(pose_quat), root_trans.cpu(), is_local=False)
    local_rot = new_sk_state.local_rotation
    ref_pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
    ref_pose_aa = torch.from_numpy(ref_pose_aa[:, mujoco_2_smpl, :].reshape(N, -1)).cuda()
    root_trans_offset = root_trans - offset
    
    with torch.no_grad():
        verts, joints = mesh_parser.get_joints_verts(pose=ref_pose_aa, th_trans=root_trans_offset.cuda())
        sim_verts = verts.numpy()[0]
        ref_verts = verts.numpy()[1]
    ref_mesh.vertices = o3d.utility.Vector3dVector(ref_verts); sim_mesh.vertices = o3d.utility.Vector3dVector(sim_verts)
    sim_mesh.compute_vertex_normals(); ref_mesh.compute_vertex_normals()
    o3d_vis.update_geometry(sim_mesh); o3d_vis.update_geometry(ref_mesh)
    o3d_vis.poll_events()
    o3d_vis.update_renderer()
    
    time_step += dt
    