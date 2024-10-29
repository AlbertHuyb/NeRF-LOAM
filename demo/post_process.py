import os  # noqa
import sys  # noqa
sys.path.insert(0, os.path.abspath('src')) # noqa
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from parser import get_parser
import numpy as np
import torch
from nerfloam import nerfloam
import os
import open3d as o3d
from utils.import_util import get_dataset

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = get_parser().parse_args()
    if hasattr(args, 'seeding'):
        setup_seed(args.seeding)
    else:
        setup_seed(777)

    data_stream = get_dataset(args)
    data_stream.load_gt_pose()
    
    # import pdb; pdb.set_trace()
    
    central_pose = data_stream.init_pose
    central_pose = central_pose.reshape(3,4)
    
    seq_name = data_stream.data_path.split('/')[-2]
    seq_log_dir = os.path.join('/data/huyb/cvpr-2024/NeRF-LOAM/logs/ss3dm', seq_name)
    exist_dir = os.path.join(seq_log_dir, os.listdir(seq_log_dir)[-1])
    
    final_mesh_path = os.path.join(exist_dir, 'mesh', 'final_mesh.ply')
    
    mesh = o3d.io.read_triangle_mesh(final_mesh_path)
    
    # transform the mesh to the central pose
    trans_pose = np.zeros((4,4))
    trans_pose[:3,:4] = central_pose
    
    # import pdb; pdb.set_trace()
    
    # mesh.transform(trans_pose)
    # trans_pose[0,3] = central_pose[1,3]
    # trans_pose[1,3] = central_pose[0,3]
    mesh.translate(trans_pose[:3,3])
    
    o3d.io.write_triangle_mesh(final_mesh_path.replace('final_mesh.ply', 'final_mesh_transformed.ply'), mesh)