import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp
from smpl_sim.utils.torch_ext import dict_to_torch

sys.path.append(os.getcwd())

from smpl_sim.poselib.core.rotation3d import transform_mul


class Humanoid_Batch:

    def __init__(self, sk_trees, device="cpu"):
        self.device = device
        self.update_model(sk_trees=sk_trees)

    def update_model(self, sk_trees):
        self.sk_trees = sk_trees
        self.local_translation_batch = torch.stack([sk_tree.local_translation for sk_tree in sk_trees]).to(self.device)
        self.num_joints = len(self.sk_trees[-1])
        self.parent_indices = sk_trees[0].parent_indices

    def batch_fk_quat(self, env_ids, local_rot, root_trans):
        # local_rot in quat.
        local_translation_batch = self.local_translation_batch[env_ids].clone()
        local_translation_batch[:, 0] = root_trans
        local_rotation_batch = local_rot
        local_transformation = torch.cat([local_rotation_batch, local_translation_batch], dim=-1).to(local_rot)

        global_transformation = []

        for node_index in range(self.num_joints):
            parent_index = self.parent_indices[node_index]
            if parent_index == -1:
                global_transformation.append(local_transformation[..., node_index, :])
            else:
                global_transformation.append(transform_mul(
                    global_transformation[parent_index],
                    local_transformation[..., node_index, :],
                ))
        global_transformation = torch.stack(global_transformation, axis=-2)
        return {"global_rotation": global_transformation[..., :4], "global_translation": global_transformation[..., 4:]}

    # def batch_fk_expmap(self, local_rot, root_trans):
    #     # local_rot in quat.
    #     local_translation_batch = self.local_translation_batch.clone()
    #     local_translation_batch[:, 0] = root_trans
    #     local_rotation_batch = local_rot
    #     local_transformation = torch.cat([local_rotation_batch, local_translation_batch], dim = -1)

    #     global_transformation = []

    #     for node_index in range(self.num_joints):
    #         parent_index = self.parent_indices[node_index]
    #         if parent_index == -1:
    #             global_transformation.append(local_transformation[..., node_index, :])
    #         else:
    #             global_transformation.append(
    #                 transform_mul_expmap(
    #                     global_transformation[parent_index],
    #                     local_transformation[..., node_index, :],
    #                 ))
    #     global_transformation = torch.stack(global_transformation, axis=-2)
    #     return {
    #         "global_translation": global_transformation[..., :3],
    #         "global_rotation": global_transformation[..., 3:]
    #     }
