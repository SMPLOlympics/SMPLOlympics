import torch
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import torch.nn.functional as F
import math
import pdb
import time
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *
from tqdm import tqdm
from scipy import ndimage


class Terrain:
    def __init__(self, cfg, env_spacing, num_envs, device) -> None:

        self.type = cfg["terrainType"]
        self.device = device
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1# resolution 0.1
        self.vertical_scale = 0.005
        self.border_size = 0
        self.offset = 20
        self.env_length = 20
        self.env_width = 20
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.env_rows = 15
        self.env_cols = 15
        self.num_maps = self.env_rows * self.env_cols
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length /
                                         self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(
            self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(
            self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.walkable_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.randomized_terrain()

        self.heightsamples = torch.from_numpy(self.height_field_raw).to(self.device) # ZL: raw height field, first dimension is x, second is y
        self.change_height_map()
        self.walkable_field = torch.from_numpy(self.walkable_field_raw).to(self.device)
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale,cfg["slopeTreshold"])
        self.sample_extent_x = int((self.tot_rows - self.border * 2) * self.horizontal_scale)
        self.sample_extent_y = int((self.tot_cols - self.border * 2) * self.horizontal_scale)

        coord_x, coord_y = torch.where(self.walkable_field == 0)
        coord_x_scale = coord_x * self.horizontal_scale
        coord_y_scale = coord_y * self.horizontal_scale
        walkable_subset = torch.logical_and(
                torch.logical_and(coord_y_scale < coord_y_scale.max() - self.border * self.horizontal_scale, coord_x_scale < coord_x_scale.max() - self.border * self.horizontal_scale),
                torch.logical_and(coord_y_scale > coord_y_scale.min() + self.border * self.horizontal_scale, coord_x_scale > coord_x_scale.min() +  self.border * self.horizontal_scale)
            )
        # import ipdb; ipdb.set_trace()
        # joblib.dump(self.walkable_field_raw, "walkable_field.pkl")

        self.coord_x_scale = coord_x_scale[walkable_subset]
        self.coord_y_scale = coord_y_scale[walkable_subset]
        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, max_num_envs, env_ids, group_num_people = 16, sample_groups = False):
        if sample_groups:
            num_groups = max_num_envs// group_num_people
            group_centers = torch.stack([torch_rand_float(0., self.sample_extent_x, (num_groups, 1),device=self.device).squeeze(1), torch_rand_float(0., self.sample_extent_y, (num_groups, 1),device=self.device).squeeze(1)], dim = -1)
            group_diffs = torch.stack([torch_rand_float(-8., 8, (num_groups, group_num_people) ,device=self.device), torch_rand_float(8., -8, (num_groups, group_num_people),device=self.device)], dim = -1)
            valid_locs = (group_centers[:, None, ] + group_diffs).reshape(max_num_envs, -1)

            if not env_ids is None:
                valid_locs = valid_locs[env_ids]
        else:
            num_envs = env_ids.shape[0]
            idxes = np.random.randint(0, self.num_samples, size=num_envs)
            valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)

        return valid_locs

    def world_points_to_map(self, points):
        points = (points / self.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0] - 2)
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py

    def change_height_map(self):
        # center_height_points
        y = torch.tensor(np.linspace(-0.1, 0.1, 3), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(-0.1, 0.1, 3), device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        num_center_height_points = grid_x.numel()

        x_points_tee  = torch.arange(0, np.sqrt(self.num_envs) ) * self.env_spacing * 2 + self.offset
        y_points_tee = torch.arange(0, self.num_envs / np.sqrt(self.num_envs)) * self.env_spacing * 2 + self.offset
        grid_x_tee, grid_y_tee = torch.meshgrid(x_points_tee, y_points_tee)
        tee_total = grid_x_tee.numel()
        points = torch.zeros(tee_total,
                             num_center_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        tee_pos = torch.stack([grid_x_tee.flatten(), grid_y_tee.flatten(), torch.zeros_like(grid_y_tee).flatten()],
                              dim=-1)
        tee_pos=tee_pos.to(self.device)
        points = points + tee_pos.unsqueeze(1)

        px, py = self.world_points_to_map(points)
        self.height_field_raw[px, py] = np.average(self.height_field_raw[px, py])


    def sample_height_points(self, points, root_states = None, root_points=None, env_ids = None, num_group_people = 512, group_ids = None):
        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        heights1 = heightsamples[px, py]
        heights2 = heightsamples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        return heights * self.vertical_scale


    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.width_per_env_pixels,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0.1, 1)
            slope = difficulty * 0.7
            discrete_obstacles_height = 0.025 + difficulty * 0.15
            stepping_stones_size = 2 - 1.8 * difficulty
            step_height = 0.05 + 0.175 * difficulty
            wave_terrain(terrain, num_waves=2., amplitude=0.5)

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles
