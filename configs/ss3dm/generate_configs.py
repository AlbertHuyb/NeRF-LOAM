import os
import yaml
import shutil

template_yaml = 'Town01_150.yaml'
template_town_name = 'Town01'
template_seq_name = 'Town01_150'

data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

for town_name in os.listdir(data_root):
    if os.path.isdir(os.path.join(data_root, town_name)):
        town_dir = os.path.join(data_root, town_name)
        for seq_name in os.listdir(town_dir):
            if os.path.isdir(os.path.join(town_dir, seq_name)):
                if seq_name == template_seq_name:
                    continue
                
                seq_dir = os.path.join(town_dir, seq_name)
                
                save_template_name = template_yaml.replace(template_seq_name, seq_name)
                
                # copy the template yaml file
                shutil.copy(template_yaml,  save_template_name)
                
                # load the yaml file as txt
                with open(save_template_name, 'r') as f:
                    yaml_txt = f.read()
                
                # modify the file
                yaml_txt = yaml_txt.replace(template_seq_name, seq_name)
                yaml_txt = yaml_txt.replace(template_town_name, town_name)
                
                # save the yaml file
                with open(save_template_name, 'w') as f:
                    f.write(yaml_txt)
                
                