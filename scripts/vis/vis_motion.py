"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags
from phc.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict
import argparse

flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:

    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments
        
if __name__ == "__main__":
    


    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": False,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "model": "smplx",
        "big_ankle": True, 
        "box_body": True, 
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl/smplx",
    )

    # gender_beta = np.array([1.0000, -0.2141, -0.1140, 0.3848, 0.9583, 1.7619, 1.5040, 0.5765, 0.9636, 0.2636, -0.4202, 0.5075, -0.7371, -2.6490, 0.0867, 1.4699, -1.1865])
    gender_beta = np.zeros((21))
    smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
    test_good = f"/tmp/smpl/test_good.xml"
    smpl_robot.write_xml(test_good)
    sk_tree = SkeletonTree.from_mjcf(test_good)

    asset_descriptors = [
        AssetDesc(test_good, False),
    ]

    # parse arguments
    args = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges",
                                custom_parameters=[{
                                    "name": "--asset_id",
                                    "type": int,
                                    "default": 0,
                                    "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)
                                }, {
                                    "name": "--speed_scale",
                                    "type": float,
                                    "default": 1.0,
                                    "help": "Animation speed scale"
                                }, {
                                    "name": "--show_axis",
                                    "action": "store_true",
                                    "help": "Visualize DOF axis"
                                }, 
                                {
                                    "name": "--motion_file",
                                    "type": str,
                                    "default": "data/amass_x/pkls/singles/smplx_failed_1.pkl"
                                }
                                ])
    motion_file = args.motion_file
    
    if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
        print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
        quit()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if not args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # load asset
    # asset_root = "amp/data/assets"
    # asset_root = "./"
    asset_root = "/"
    asset_file = asset_descriptors[args.asset_id].file_name

    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.flip_visual_attachments = asset_descriptors[
    #     args.asset_id].flip_visual_attachments
    asset_options.use_mesh_materials = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # set up the env grid
    num_envs = 1
    num_per_row = 5
    spacing = 5
    env_lower = gymapi.Vec3(-spacing, spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(0, -10.0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    num_dofs = gym.get_asset_dof_count(asset)
    print("Creating %d environments" % num_envs)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        # set default DOF positions
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # Setup Motion
    body_ids = []
    key_body_names = ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]
    for body_name in key_body_names:
        body_id = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], body_name)
        assert (body_id != -1)
        body_ids.append(body_id)
    gym.prepare_sim(sim)
    body_ids = np.array(body_ids)


    motion_data = joblib.load(motion_file)
    # print(motion_keys)

    device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))
    motion_lib_cfg = EasyDict({
                    "motion_file": motion_file,
                    "device": torch.device("cpu"),
                    "fix_height": FixHeightMode.full_fix,
                    "min_length": -1,
                    "max_length": -1,
                    "im_eval": flags.im_eval,
                    "multi_thread": False ,
                    "smpl_type": 'smplx',
                    "randomrize_heading": True,
                    "device": device,
                })
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    num_motions = 10
    curr_start = 0
    motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.from_numpy(gender_beta)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
    motion_keys = motion_lib.curr_motion_keys

    current_dof = 0
    speeds = np.zeros(num_dofs)

    time_step = 0
    rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
    rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
    rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    actor_root_state = gymtorch.wrap_tensor(actor_root_state)

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "previous")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "add")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "next_batch")
    motion_id = 0
    motion_acc = set()
    env_ids = torch.arange(num_envs).int().to(args.sim_device)
    while not gym.query_viewer_has_closed(viewer):
        # step the physics

        motion_len = motion_lib.get_motion_length(motion_id).item()
        motion_time = time_step % motion_len
        # motion_time = 0

        motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([motion_time]).to(args.compute_device_id))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        if args.show_axis:
            gym.clear_lines(viewer)

        #################### Heading invarance check: ####################
        # from phc.env.tasks.humanoid_im import compute_imitation_observations
        # from phc.env.tasks.humanoid import compute_humanoid_observations_smpl_mmt_max
        # from phc.env.tasks.humanoid_amp import build_amp_observations_smpl_mmt

        # motion_res_10 = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([0]).to(args.compute_device_id))
        # motion_res_100 = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([3]).to(args.compute_device_id))

        # root_pos_10, root_rot_10, dof_pos_10, root_vel_10, root_ang_vel_10, dof_vel_10, smpl_params_10, limb_weights_10, pose_aa_10, rb_pos_10, rb_rot_10, body_vel_10, body_ang_vel_10 = \
        #             motion_res_10["root_pos"], motion_res_10["root_rot"], motion_res_10["dof_pos"], motion_res_10["root_vel"], motion_res_10["root_ang_vel"], motion_res_10["dof_vel"], \
        #              motion_res_10["motion_bodies"], motion_res_10["motion_limb_weights"], motion_res_10["motion_aa"], motion_res_10["rg_pos"], motion_res_10["rb_rot"], motion_res_10["body_vel"], motion_res_10["body_ang_vel"]

        # root_pos_100, root_rot_100, dof_pos_100, root_vel_100, root_ang_vel_100, dof_vel_100, smpl_params_100, limb_weights_100, pose_aa_100, rb_pos_100, rb_rot_100, body_vel_100, body_ang_vel_100 = \
        #             motion_res_100["root_pos"], motion_res_100["root_rot"], motion_res_100["dof_pos"], motion_res_100["root_vel"], motion_res_100["root_ang_vel"], motion_res_100["dof_vel"], \
        #             motion_res_100["motion_bodies"], motion_res_100["motion_limb_weights"], motion_res_100["motion_aa"], motion_res_100["rg_pos"], motion_res_100["rb_rot"], motion_res_100["body_vel"], motion_res_100["body_ang_vel"]
        # # 10 is the current, 100 is target. we need to change it's heading so load up another scirpt. 
        # obs = compute_imitation_observations(root_pos_100, root_rot_100, rb_pos_100, rb_rot_100, body_vel_100, body_ang_vel_100, rb_pos_10, rb_rot_10, body_vel_10, body_ang_vel_10, 1, True)
        # # obs_im = compute_humanoid_observations_smpl_mmt_max(rb_pos_100, rb_rot_100, body_vel_100, body_ang_vel_100, smpl_params_100, limb_weights_100, True, False, True, True, True)
        # # obs_amp = build_amp_observations_smpl_mmt(
        # #     root_pos_100, root_rot_100, body_vel_100[:, 0, :],
        # #     body_ang_vel_100[:, 0, :], dof_pos_100, dof_vel_100, rb_pos_100,
        # #     smpl_params_100, limb_weights_100, None, True, False, False, True, True, True)

        # # motion_lib.load_motions(skeleton_trees = [sk_tree] * num_motions, gender_betas = [torch.zeros(17)] * num_motions, limb_weights = [np.zeros(10)] * num_motions, random_sample=False)
        # joblib.dump(obs, "a.pkl")
        # import ipdb
        # ipdb.set_trace()

        #################### Heading invarance check: ####################

        ###########################################################################
        # root_pos[:, 1] *= -1
        # key_pos[:, 1] *= -1  # Will need to flip these as well
        # root_rot[:, 0] *= -1
        # root_rot[:, 2] *= -1

        # dof_vel = dof_vel.reshape(len(left_to_right_index), 3)[left_to_right_index]
        # dof_vel[:, 0] = dof_vel[:, 0] * -1
        # dof_vel[:, 2] = dof_vel[:, 2] * -1
        # dof_vel = dof_vel.reshape(1, len(left_to_right_index) * 3)

        # dof_pos = dof_pos.reshape(len(left_to_right_index), 3)[left_to_right_index]
        # dof_pos[:, 0] = dof_pos[:, 0] * -1
        # dof_pos[:, 2] = dof_pos[:, 2] * -1
        # dof_pos = dof_pos.reshape(1, len(left_to_right_index) * 3)
        ###########################################################################
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
        # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.refresh_actor_root_state_tensor(sim)

        # dof_pos = dof_pos.cpu().numpy()
        # dof_states['pos'] = dof_pos
        # speed = speeds[current_dof]

        dof_state = torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1).squeeze().repeat(num_envs, 1)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.fetch_results(sim, True)

        # print((rigidbody_state[None, ] - rigidbody_state[:, None]).sum().abs())
        # print((actor_root_state[None, ] - actor_root_state[:, None]).sum().abs())

        # pose_quat = motion_lib._motion_data['0-ACCAD_Female1Running_c3d_C5 - walk to run_poses']['pose_quat_global']
        # diff = quat_mul(quat_inverse(rb_rot[0, :]), rigidbody_state[0, :, 3:7]); np.set_printoptions(precision=4, suppress=1); print(diff.cpu().numpy()); print(torch_utils.quat_to_angle_axis(diff)[0])

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        # time_step += 1/5
        time_step += dt

        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "previous" and evt.value > 0:
                motion_id = (motion_id - 1) % num_motions
                print(f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}")
            elif evt.action == "next" and evt.value > 0:
                motion_id = (motion_id + 1) % num_motions
                print(f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}")
            elif evt.action == "add" and evt.value > 0:
                motion_acc.add(motion_keys[motion_id])
                print(f"Adding motion {motion_keys[motion_id]}")
            elif evt.action == "print" and evt.value > 0:
                print(motion_acc)
            elif evt.action == "next_batch" and evt.value > 0:
                curr_start += num_motions
                motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
                motion_keys = motion_lib.curr_motion_keys
                print(f"Next batch {curr_start}")

            time_step = 0
    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
