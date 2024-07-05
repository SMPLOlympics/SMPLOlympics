import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
from datetime import datetime
paused, reset, recording, image_list, writer, control, curr_zoom = False, False, False, [], None, None, 0.01


def pause_func(action):
    global paused
    paused = not paused
    print(f"Paused: {paused}")
    return True


def reset_func(action):
    global reset
    reset = not reset
    print(f"Reset: {reset}")
    return True


def record_func(action):
    global recording, writer
    if not recording:
        fps = 30
        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        curr_video_file_name = f"output/renderings/o3d/{curr_date_time}-test.mp4"
        writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
    elif not writer is None:
        writer.close()
        writer = None

    recording = not recording

    print(f"Recording: {recording}")
    return True


def zoom_func(action):
    global control, curr_zoom
    curr_zoom = curr_zoom * 0.9
    control.set_zoom(curr_zoom)
    print(f"Reset: {reset}")
    return True

colorpicker = mpl.colormaps['Blues']
mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

Name = "getting_started"
Title = "Getting Started"

data_dir = "data/smpl"
smplx_parser_n = SMPLX_Parser(model_path=data_dir, gender="neutral", use_pca=False, create_transl=False, flat_hand_mean = True)
smplx_parser_m = SMPLX_Parser(model_path=data_dir, gender="male", use_pca=False, create_transl=False, flat_hand_mean = True)
smplx_parser_f = SMPLX_Parser(model_path=data_dir, gender="female", use_pca=False, create_transl=False, flat_hand_mean = True)

# motion_file = "data/amass_x/singles/GRAB_s1_banana_eat_1_stageii.pkl"
# motion_file = "data/amass_x/singles/CMU_61_61_04_stageii.pkl"
motion_file = "data/amass_x/singles/standing.pkl"
# motion_file = "data/reinterhand/fitted/test.pkl"
# motion_file = "data/amass_x/test.pkl"

Name = motion_file.split("/")[-1].split(".")[0]
pkl_data = joblib.load(motion_file)
mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]


def main():
    global reset, paused, recording, image_list, control
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    smpl_meshes = dict()
    items = list(pkl_data.items())

    for entry_key, data_seq in tqdm(items):
        
        gender, beta = 0, data_seq['beta'][0]
        if gender == 0:
            smpl_parser = smplx_parser_n
        elif gender == 1:
            smpl_parser = smplx_parser_m
        else:
            smpl_parser = smplx_parser_f

        pose_aa = data_seq['pose_aa']
        root_trans_offset = data_seq['trans_orig']
        with torch.no_grad():
            vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa), th_betas=torch.zeros((1, 20)), th_trans=torch.from_numpy(root_trans_offset), )
        vertices = vertices.numpy()
        faces = smpl_parser.faces
        smpl_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
        smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)

        ######################## Smampling texture ########################
        
        vertex_colors = colorpicker(0.6)[:3]
        smpl_mesh.paint_uniform_color(vertex_colors)
        
        smpl_mesh.compute_vertex_normals()
        ######################## Smampling texture ########################
        vis.add_geometry(smpl_mesh)
        smpl_meshes[entry_key] = {
            'mesh': smpl_mesh,
            "vertices": vertices,
        }
        break

    box = o3d.geometry.TriangleMesh()
    ground_size, height = 15,  0.01
    box = box.create_box(width=ground_size, height=height, depth=ground_size)
    box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
    box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
    box.compute_vertex_normals()
    box.vertex_colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]).repeat(8, axis=0))
    num_jts = 51
    spheres = []
    for _ in range(num_jts):
        sphere = o3d.geometry.TriangleMesh()
        sphere = sphere.create_sphere(radius=0.05)
        sphere.compute_vertex_normals()
        sphere.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.9, 0.1]]).repeat(len(sphere.vertices), axis=0))
        spheres.append(sphere)

    sphere_pos = np.zeros([num_jts, 3])
    [vis.add_geometry(sphere) for sphere in spheres]
    vis.add_geometry(box)

    control = vis.get_view_control()

    control.unset_constant_z_far()
    control.unset_constant_z_near()
    i = 0
    N = vertices.shape[0]

    vis.register_key_callback(32, pause_func)
    vis.register_key_callback(82, reset_func)
    vis.register_key_callback(76, record_func)
    vis.register_key_callback(90, zoom_func)

    control.set_up(np.array([0, 0, 1]))
    control.set_front(np.array([1, 0, 0]))
    control.set_lookat(vertices[0, 0])

    control.set_zoom(0.5)
    dt = 1 / 30

    
    # tracker_pos = data_seq['joints_gt']

    while True:
        vis.poll_events()
        for smpl_mesh_key, smpl_mesh_data in smpl_meshes.items():
            verts = smpl_mesh_data["vertices"]
            smpl_mesh_data["mesh"].vertices = o3d.utility.Vector3dVector(verts[i % verts.shape[0]])
            vis.update_geometry(smpl_mesh_data["mesh"])
            smpl_mesh_data["mesh"].compute_vertex_normals()

            # motion_res = motion_lib.get_motion_state(torch.tensor([0]), torch.tensor([(i % verts.shape[0]) * dt]))
            # curr_pos = motion_res['rg_pos'][0, [13, 18 ,23]].numpy()
            
            # curr_pos = tracker_pos[i % verts.shape[0]] # joints gt

            # for idx, s in enumerate(spheres):
            #     s.translate((curr_pos - sphere_pos)[idx])
            #     vis.update_geometry(s)
            # sphere_pos = curr_pos
            
            # sphere.translate(verts[0, 0])
            # vis.update_geometry(sphere)

        if not paused:
            i = (i + 1)

        if reset:
            i = 0
            reset = False
        if recording:
            rgb = vis.capture_screen_float_buffer()
            rgb = (np.asarray(rgb) * 255).astype(np.uint8)
            writer.append_data(rgb)

        vis.update_renderer()


if __name__ == "__main__":
    main()