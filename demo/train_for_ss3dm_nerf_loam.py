import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--town_name', type=str, default=None)
parser.add_argument('--seq_name', type=str, default=None)
parser.add_argument('--mesh_res', type=float, default=0.1)
args = parser.parse_args() 

data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

if args.town_name is not None:
    town_list = [args.town_name]
else:
    town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']


for town_name in town_list:
    if os.path.isdir(os.path.join(data_root, town_name)):
        town_dir = os.path.join(data_root, town_name)
        
        if args.seq_name is not None:
            seq_list = [args.seq_name]
        else:
            seq_list = os.listdir(town_dir)
        
        # import pdb; pdb.set_trace()
        for seq_name in seq_list:
        # for seq_name in ['Town01_300']:
            if os.path.isdir(os.path.join(town_dir, seq_name)):
                for config_folder, config_name in zip(['urban_nerf'],['withmask_withlidar_withnormal_all_cameras']):
                    train_cmd = 'python demo/run.py configs/ss3dm/{}.yaml'.format(seq_name)
                    
                    log_dir = '/data/huyb/cvpr-2024/NeRF-LOAM/logs/ss3dm/{}'.format(seq_name)
                    if not os.path.exists(log_dir):
                        os.system(train_cmd)
                
