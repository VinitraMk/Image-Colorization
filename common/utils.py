from config.config import root_dir

from torchvision.io import read_image
from PIL import Image
import torch
import os
import yaml

def save_df_to_csv(df, filename, columns = []):
    if len(columns) == 0:
        columns = df.columns.tolist()
    df.to_csv(filename, index = False)
    
def read_imgtensor(imgpath):
    return read_image(imgpath)

def read_imgnp(imgpath):
    img = Image.open(imgpath)
    return img

def img2tensor(imgnparr):
    return torch.from_numpy(imgnparr)
   
def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)

def get_exp_params():
    yaml_fp = os.path.join(root_dir, 'run.yaml')
    exp_params = {}
    with open(yaml_fp, "r") as stream:
        try:
            exp_params = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(err)
    return exp_params
            
        
    
    