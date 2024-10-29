import os.path as osp

import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
import sys
from scipy.spatial import cKDTree

patchwork_module_path ="/data/huyb/cvpr-2024/NeRF-LOAM/third_party/patchwork-plusplus/python_wrapper"
sys.path.insert(0, patchwork_module_path)
import pypatchworkpp
params = pypatchworkpp.Parameters()
# params.verbose = True

PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)


import pickle
import numpy as np
import os

class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=True, max_depth=-1, min_depth=-1, activate_lidars=5) -> None:
        self.data_path = data_path
        self.num_bin = activate_lidars * len(glob(osp.join(self.data_path, "lidars/lidar_FRONT/*.npz")))
        self.use_gt = use_gt
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        lidar_point_dir = osp.join(self.data_path, "lidars")
        lidar_list = os.listdir(lidar_point_dir)
        lidar_list.sort()
        
        self.all_lidar_files = []
        for lidar in lidar_list:
            this_lidar_dir = osp.join(lidar_point_dir, lidar)
            for frame_idx in range(self.num_bin // activate_lidars):
                self.all_lidar_files.append(osp.join(this_lidar_dir, "{:08d}.npz".format(frame_idx)))
        
        self.gt_pose = self.load_gt_pose() if use_gt else None
        
        self.init_pose = None
        
        

    def get_init_pose(self, frame):
        if self.gt_pose is not None:
            return np.concatenate((self.gt_pose[frame], [0, 0, 0, 1])
                                  ).reshape(4, 4)
        else:
            return np.eye(4)

    def load_gt_pose(self):
        gt_pose = []
        
        for lidar_file_name in self.all_lidar_files:
            lidar_data = np.load(lidar_file_name)
            lidar_position = lidar_data["rays_o"]

            lidar_pose = np.zeros((3,4))
            lidar_pose[:3,:3] = np.eye(3)
            
            lidar_pose[0,3] = lidar_position[0,0]
            lidar_pose[1,3] = lidar_position[0,1]
            lidar_pose[2,3] = lidar_position[0,2]
            
            gt_pose.append(lidar_pose.reshape(1, -1))   
        
        gt_pose = np.concatenate(gt_pose, axis=0)  
        
        # relative pose to the first lidar place
        self.init_pose = gt_pose[0]
        gt_pose = gt_pose - gt_pose[0]
                
        return gt_pose

    def load_points(self, index):
        # 1. load the lidar points in the world coord.
        # 2. load the lidar poses.
        # 3. transform the lidar points to the lidar coord.
        # 4. detect the ground points.
        

        lidar_data = np.load(self.all_lidar_files[index])
        points = lidar_data["rays_d"] * lidar_data['ranges'][...,None]

        points_norm = np.linalg.norm(points[:, :3], axis=-1)
        point_mask = True
        if self.max_depth != -1:
            point_mask = (points_norm < self.max_depth) & point_mask
        if self.min_depth != -1:
            point_mask = (points_norm > self.min_depth) & point_mask

        if isinstance(point_mask, np.ndarray):
            points = points[point_mask]

        # import pdb; pdb.set_trace()
        PatchworkPLUSPLUS.estimateGround(points)
        ground = PatchworkPLUSPLUS.getGround()
        nonground = PatchworkPLUSPLUS.getNonground()
        Patchcenters = PatchworkPLUSPLUS.getCenters()
        normals = PatchworkPLUSPLUS.getNormals()
        T = cKDTree(Patchcenters)
        _, index = T.query(ground)
        if True:
            groundcos = np.abs(np.sum(normals[index] * ground, axis=-1)/np.linalg.norm(ground, axis=-1))
        else:
            groundcos = np.ones(ground.shape[0])
        points = np.concatenate((ground, nonground), axis=0)
        pointcos = np.concatenate((groundcos, np.ones(nonground.shape[0])), axis=0)

        return points, pointcos

    def __len__(self):
        return self.num_bin

    def __getitem__(self, index):
        points, pointcos = self.load_points(index)
        points = torch.from_numpy(points).float()
        pointcos = torch.from_numpy(pointcos).float()
        pose = np.concatenate((self.gt_pose[index], [0, 0, 0, 1])
                              ).reshape(4, 4) if self.use_gt else None
        return index, points, pointcos, pose


if __name__ == "__main__":
    path = "/data/huyb/cvpr-2024/data/ss3dm/DATA/Town01/Town01_150/"
    loader = DataLoader(path)
    for data in loader:
        index, points, _, pose = data
        print("current index ", index)
        print("first 10th points:\n", points[:10])
        if index > 10:
            break
        index += 1
