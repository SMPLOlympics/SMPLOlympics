# poselib

`poselib` is a library for loading, manipulating, and retargeting skeleton poses and motions. It is separated into three modules: `smpl_sim.poselib.core` for basic data loading and tensor operations, `smpl_sim.poselib.skeleton` for higher-level skeleton operations, and `smpl_sim.poselib.visualization` for displaying skeleton poses.

## smpl_sim.poselib.core
- `smpl_sim.poselib.core.rotation3d`: A set of Torch JIT functions for dealing with quaternions, transforms, and rotation/transformation matrices.
    - `quat_*` manipulate and create quaternions in [x, y, z, w] format (where w is the real component).
    - `transform_*` handle 7D transforms in [quat, pos] format.
    - `rot_matrix_*` handle 3x3 rotation matrices.
    - `euclidean_*` handle 4x4 Euclidean transformation matrices.
- `smpl_sim.poselib.core.tensor_utils`: Provides loading and saving functions for PyTorch tensors.

## smpl_sim.poselib.skeleton
- `smpl_sim.poselib.skeleton.skeleton3d`: Utilities for loading and manipulating skeleton poses, and retargeting poses to different skeletons.
    - `SkeletonTree` is a class that stores a skeleton as a tree structure.
    - `SkeletonState` describes the static state of a skeleton, and provides both global and local joint angles.
    - `SkeletonMotion` describes a time-series of skeleton states and provides utilities for computing joint velocities.

## smpl_sim.poselib.visualization
- `smpl_sim.poselib.visualization.common`: Functions used for visualizing skeletons interactively in `matplotlib`.
